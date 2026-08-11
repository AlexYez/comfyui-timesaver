"""TS Super Prompt — voice dictation + Qwen prompt enhancement.

Public node + the four aiohttp routes that drive the in-canvas UI:

- POST /ts_voice_recognition/transcribe : multipart audio → Whisper text.
- POST /ts_voice_recognition/preload    : warm up the Whisper model.
- GET  /ts_voice_recognition/status     : reports model cache / load state.
- POST /ts_super_prompt/enhance         : runs Qwen prompt enhancement.

The image reference is **not** taken from a workflow IMAGE input. Instead
the JS frontend lets the user attach a file (drag-drop / paste / file
picker), uploads it through ComfyUI's standard ``/upload/image`` endpoint,
and stores the resulting ComfyUI-annotated path
(``"subfolder/filename [type]"``) in the hidden ``attached_image``
socketless widget. The ``/ts_super_prompt/enhance`` route then resolves
that annotated path back to a PIL image at request time and passes it to
Qwen — so the Enhance button works without running the entire workflow.

Heavy logic lives in the sibling private modules:

- ``_helpers.py`` — config constants, logger, PromptServer event dispatch,
                    aiohttp route decorators, byte/dir formatting.
- ``_voice.py``   — Whisper download/load + ffmpeg decode + VAD + transcribe.
- ``_qwen.py``    — preset loading + chat-template + Qwen generation.

node_id: TS_SuperPrompt
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import folder_paths
from aiohttp import web
from comfy_api.v0_0_2 import IO

from ._helpers import (
    cancel_operation,
    forget_operation,
    resolve_prompt_model,
    ACTIVE_MODEL,
    AI_ROUTE_BASE,
    ALLOWED_AUDIO_SUFFIXES,
    DEFAULT_MODEL_ID,
    DEFAULT_PRESET,
    DEVICE,
    ENHANCE_MAX_TEXT_LEN,
    LOG_PREFIX,
    LOGGER,
    MODEL_LOCK,
    MODELS_WITHOUT_TRANSLATE,
    SOURCE_LANGUAGE,
    SUPER_PROMPT_ENHANCE_ON_EXECUTE,
    TRANSLATE_TO_ENGLISH,
    VOICE_LOG_PREFIX,
    VOICE_MODEL_CACHE,
    VOICE_ROUTE_BASE,
    VOICE_UPLOAD_MAX_BYTES,
    register_get,
    register_post,
    send_done,
    send_error,
    send_voice_event,
    send_voice_status,
    voice_log_warning,
)
from ._qwen import _generate_with_qwen, default_preset, preset_options
from ._voice import (
    _audio_tmp_dir,
    _configured_initial_prompt,
    _ensure_runtime_dirs,
    _missing_runtime_packages,
    _model_file_path,
    _resolve_voice_model,
    ensure_model,
    is_model_cached,
    load_model,
    transcribe_audio,
)

# One transcription at a time per process. Whisper decoding is GPU work, and
# nothing else bounded it: several browser tabs (or one impatient user) could
# start N transcriptions in parallel, each claiming VRAM next to the resident
# diffusion pipeline. Waiting is the right answer — the request still completes.
_TRANSCRIBE_GATE = asyncio.Semaphore(1)


# ---------------------------------------------------------------------------
# Attached image helpers
# ---------------------------------------------------------------------------

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def _allowed_image_roots() -> list[Path]:
    """The directories an attached image may legitimately live in."""
    roots: list[Path] = []
    for getter_name in ("get_input_directory", "get_output_directory", "get_temp_directory"):
        getter = getattr(folder_paths, getter_name, None)
        if not callable(getter):
            continue
        try:
            roots.append(Path(getter()).resolve(strict=False))
        except (OSError, TypeError, ValueError):
            continue
    return roots


def _is_inside_allowed_root(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return False
    for root in _allowed_image_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _resolve_annotated_image_path(annotated: str) -> str:
    """Resolve a ComfyUI annotated filepath (``"name.png [input]"``) to a real path.

    Returns an empty string if the input is empty, unresolvable, or points
    outside ComfyUI's input/output/temp directories.

    The path always originates from the node's own upload flow (the JS sends
    the file through ``/upload/image`` and stores the annotated result), so it
    is always one of those roots. The raw-string fallback below exists for
    annotations ``folder_paths`` refuses to parse — but it used to accept ANY
    absolute path, and ``/ts_super_prompt/enhance`` takes this value straight
    from the request body: a caller could name any image on the machine and get
    Qwen's description of it back. Every other path-taking route in the pack
    scopes to these roots; this one now does too.
    """
    path = str(annotated or "").strip()
    if not path:
        return ""
    try:
        resolved = folder_paths.get_annotated_filepath(path)
    except Exception:
        resolved = ""
    for candidate in (resolved, path):
        if not candidate:
            continue
        as_path = Path(candidate)
        if as_path.is_file() and _is_inside_allowed_root(as_path):
            return str(candidate)
    LOGGER.warning(
        "%s Attached image is not inside the input/output/temp directories; ignoring it.",
        LOG_PREFIX,
    )
    return ""


# What a wired IMAGE is shrunk to before it reaches Qwen. A 4K frame carries
# nothing extra for a model deciding what a shot is about, and every pixel of it
# is memory and time. Measured by AREA rather than by the longest side: a 16:9
# frame capped by its width keeps far fewer pixels than a square one, and the
# thing that costs is the area.
#
# The model-facing cap further down (_qwen._build_messages, 1024 px on the long
# side) is unchanged — this is about not carrying a 4K tensor to it.
MAX_INPUT_MEGAPIXELS = 1.0

# How many frames of a batch are read. Two is what the prompts speak about
# today — a first and a last — and the input is a batch precisely so that three
# or four can follow without growing a socket each. The ceiling is here because
# the other likely thing to be wired in is a whole video: a 2B model handed
# ninety frames would spend its entire context on pictures and have none left
# for the idea.
MAX_REFERENCE_FRAMES = 4


def _fit_megapixels(image, max_megapixels: float = MAX_INPUT_MEGAPIXELS):
    """Shrink an image to at most `max_megapixels`, never enlarge, never crop.

    A reference image is read for what is IN it, so cropping to a tidy multiple
    would throw away the edges of the very thing being described. A picture
    already under the budget is handed over untouched: upscaling invents detail
    the model would then describe.
    """
    try:
        width, height = image.size
    except AttributeError:
        return image
    budget = max(0.1, float(max_megapixels)) * 1_000_000
    pixels = width * height
    if pixels <= budget or pixels <= 0:
        return image
    scale = (budget / pixels) ** 0.5
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    try:
        from PIL import Image as _Image

        return image.resize(target, _Image.LANCZOS)
    except Exception as exc:  # PIL missing or a resampling filter it dislikes
        LOGGER.warning("%s Could not resize a reference image: %s", LOG_PREFIX, exc)
        return image


def _socket_images(tensor, limit: int = MAX_REFERENCE_FRAMES) -> list:
    """The PIL frames behind a wired IMAGE input, in order, at working size.

    ORDER IS THE MEANING. A batch of one is a plain reference; the first image
    of a longer batch is the first frame of the shot and the last image is the
    last. That is the whole reason this is one input taking a batch rather than
    a socket per frame — three or four frames need no new wiring, only a longer
    batch.
    """
    if tensor is None:
        return []
    try:
        from ._qwen import _get_qwen_engine

        frames = _get_qwen_engine().normalize_to_pil_list(tensor)
    except Exception as exc:
        LOGGER.warning("%s Could not read a connected IMAGE input: %s", LOG_PREFIX, exc)
        return []
    if len(frames) > limit:
        LOGGER.info(
            "%s %d frames connected; using the first %d.", LOG_PREFIX, len(frames), limit,
        )
    return [_fit_megapixels(frame) for frame in frames[:limit]]


def _load_attached_images(*annotated: str) -> list:
    """The attached references, in the order the person attached them.

    One image is a reference. Two are the FIRST and the LAST frame of the shot
    the person is describing — which is why order matters and why a missing
    first slot does not promote the second one into its place: the labels the
    model is given come from these positions (see _qwen._build_messages).

    A path that will not load is skipped with a warning rather than failing the
    request: enhancing with fewer images is still useful, and refusing to
    enhance at all because one thumbnail went missing is not.
    """
    images = []
    for path in annotated:
        text = str(path or "")
        if not text:
            continue
        loaded = _load_attached_image_pil(text)
        if loaded is None:
            LOGGER.warning(
                "%s Attached image %r could not be loaded; enhancing without it.",
                LOG_PREFIX, text,
            )
            continue
        images.append(loaded)
    return images


def _load_attached_image_pil(annotated: str):
    """Resolve + load the attached image as a PIL.Image, or None on failure."""
    resolved = _resolve_annotated_image_path(annotated)
    if not resolved:
        return None
    suffix = Path(resolved).suffix.lower()
    if suffix and suffix not in _IMAGE_SUFFIXES:
        LOGGER.warning("%s Attached image has unsupported suffix %s", LOG_PREFIX, suffix)
        return None
    try:
        from PIL import Image
    except Exception as exc:
        LOGGER.warning("%s PIL not available for attached image: %s", LOG_PREFIX, exc)
        return None
    try:
        with Image.open(resolved) as img:
            return img.convert("RGB").copy()
    except Exception as exc:
        LOGGER.warning(
            "%s Failed to open attached image %s: %s", LOG_PREFIX, resolved, exc
        )
        return None



def _safe_audio_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_AUDIO_SUFFIXES and len(suffix) <= 10:
        return suffix
    return ".webm"


async def _read_audio_upload(request: web.Request) -> tuple[dict[str, Any] | None, dict[str, str]]:
    reader = await request.multipart()
    fields: dict[str, str] = {}
    upload: dict[str, Any] | None = None

    while True:
        part = await reader.next()
        if part is None:
            return upload, fields
        if part.name == "audio":
            chunks = []
            total_bytes = 0
            while True:
                chunk = await part.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > VOICE_UPLOAD_MAX_BYTES:
                    raise web.HTTPRequestEntityTooLarge(
                        max_size=VOICE_UPLOAD_MAX_BYTES,
                        actual_size=total_bytes,
                    )
                chunks.append(chunk)
            upload = {"filename": part.filename, "data": b"".join(chunks)}
            continue
        if part.name:
            # Cap text fields too: part.text() used to read without limit,
            # so the audio size cap could be bypassed via any other field.
            max_field_bytes = 64 * 1024
            chunks = []
            total = 0
            while True:
                chunk = await part.read_chunk(size=8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_field_bytes:
                    raise web.HTTPRequestEntityTooLarge(
                        max_size=max_field_bytes,
                        actual_size=total,
                    )
                chunks.append(chunk)
            fields[part.name] = b"".join(chunks).decode("utf-8", errors="replace")


@register_post(f"{VOICE_ROUTE_BASE}/transcribe")
async def transcribe_endpoint(request: web.Request) -> web.StreamResponse:
    audio_upload, fields = await _read_audio_upload(request)
    if audio_upload is None:
        return web.json_response({"error": "Missing audio field."}, status=400)

    _ensure_runtime_dirs()
    filename = f"{uuid.uuid4().hex}{_safe_audio_suffix(audio_upload.get('filename'))}"
    filepath = _audio_tmp_dir() / filename

    try:
        with filepath.open("wb") as handle:
            handle.write(audio_upload["data"])

        model_name = _resolve_voice_model(fields.get("high_quality"), fields.get("model"))
        translate_to_english = bool(TRANSLATE_TO_ENGLISH)
        if translate_to_english and model_name in MODELS_WITHOUT_TRANSLATE:
            voice_log_warning(f"Model '{model_name}' does not support translation. Using transcription.")
            translate_to_english = False

        target_language = "en" if translate_to_english else "same"
        async with _TRANSCRIBE_GATE:
            result = await asyncio.to_thread(
                transcribe_audio,
                str(filepath),
                model_name,
                DEVICE,
                SOURCE_LANGUAGE,
                target_language,
                _configured_initial_prompt(),
            )
        return web.json_response(result)
    except Exception as exc:
        LOGGER.exception("%s Voice transcription failed", VOICE_LOG_PREFIX)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            pass


@register_post(f"{VOICE_ROUTE_BASE}/preload")
async def preload_endpoint(request: web.Request) -> web.StreamResponse:
    try:
        data = await request.json()
    except Exception:
        data = {}

    name = _resolve_voice_model(data.get("high_quality"), data.get("model", ACTIVE_MODEL))
    force = bool(data.get("force", False))

    try:
        send_voice_status(name, "Preparing voice model", 5.0)
        await asyncio.to_thread(ensure_model, name, force)
        send_voice_status(name, "Loading voice model into memory", 78.0)
        await asyncio.to_thread(load_model, name, DEVICE)
        send_voice_event("done", {"model": name, "text": "Voice model ready", "percent": 100.0})
        return web.json_response({"ok": True})
    except Exception as exc:
        LOGGER.exception("%s Whisper preload failed", VOICE_LOG_PREFIX)
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


@register_get(f"{VOICE_ROUTE_BASE}/status")
async def status_endpoint(request: web.Request) -> web.StreamResponse:
    name = _resolve_voice_model(request.query.get("high_quality"), request.query.get("model", ACTIVE_MODEL))
    return web.json_response(
        {
            name: {
                "downloaded": is_model_cached(name),
                "loaded": any(cache_key[0] == name for cache_key in VOICE_MODEL_CACHE.keys()),
                "path": str(_model_file_path(name)),
                "missing_dependencies": _missing_runtime_packages(),
                "translate_to_english": bool(TRANSLATE_TO_ENGLISH),
            }
        }
    )


# ---------------------------------------------------------------------------
# Prompt-enhancement HTTP route
# ---------------------------------------------------------------------------


@register_post("/ts_super_prompt/cancel")
async def cancel_endpoint(request: web.Request) -> web.StreamResponse:
    """Отменить идущее усиление.

    ⚠️ Отмена честная, но не мгновенная везде одинаково: генерация встаёт на
    ближайшем токене, а начатое скачивание модели доводится до конца — рвать его
    на середине дороже, чем дождаться. Интерфейс об этом и сообщает.
    """
    try:
        data = await request.json()
    except Exception:                       # noqa: BLE001 - пустое тело тоже ответ
        data = {}
    operation_id = str(data.get("operation_id") or "")
    if not cancel_operation(operation_id):
        return web.json_response({"ok": False, "error": "operation_id is required."},
                                 status=400)
    LOGGER.info("%s cancel requested for %s", LOG_PREFIX, operation_id)
    return web.json_response({"ok": True})


@register_post(f"{AI_ROUTE_BASE}/enhance")
async def enhance_endpoint(request: web.Request) -> web.StreamResponse:
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body."}, status=400)
    except Exception:
        data = {}

    text = str(data.get("text") or "")
    if len(text) > ENHANCE_MAX_TEXT_LEN:
        return web.json_response(
            {"error": f"text exceeds {ENHANCE_MAX_TEXT_LEN} characters."},
            status=413,
        )

    # Галочка приезжает вместе с остальными значениями виджетов. Её нет в
    # запросе со страницы, которую не перезагружали, — тогда прежняя модель.
    bigger_model = bool(data.get("bigger_model"))
    preset = str(data.get("system_preset") or DEFAULT_PRESET)
    if preset not in preset_options():
        LOGGER.warning(
            "%s Unknown system_preset %r in /enhance; falling back to %r.",
            LOG_PREFIX,
            preset,
            DEFAULT_PRESET,
        )
        preset = DEFAULT_PRESET

    # Optional reference image attached via the in-node "Attach image" UI.
    # The frontend uploads through /upload/image first and only sends the
    # resulting annotated path here. An empty / unresolvable path simply
    # means "text-only enhance" — not an error.
    # A LIST, because the node's `images` input is a batch and the frames the
    # button resolves from it can be more than two. The two old fields are
    # still read: a page that has not been reloaded since the update keeps
    # sending them, and they are what the attach buttons fill.
    raw_paths = data.get("attached_images")
    if isinstance(raw_paths, list):
        wanted = [str(item or "") for item in raw_paths]
    else:
        wanted = [
            str(data.get("attached_image") or ""),
            str(data.get("attached_image_2") or ""),
        ]
    images = _load_attached_images(*wanted[:MAX_REFERENCE_FRAMES])
    image_pil = images or None

    # Racy fast-fail for obviously-busy Qwen. Internal _generate_with_qwen still
    # acquires MODEL_LOCK with proper blocking, so this only spares the caller
    # the trip into the worker thread when the model is already in use.
    if MODEL_LOCK.locked():
        return web.json_response(
            {"error": "Qwen is busy with another request."},
            status=429,
        )

    operation_id = str(data.get("operation_id") or "")
    # A caller with a "generate again" button sends the seed it just bumped, so
    # a second press samples differently instead of returning the same caption.
    raw_seed = data.get("seed")
    try:
        seed = int(raw_seed) if raw_seed is not None and str(raw_seed).strip() != "" else None
    except (TypeError, ValueError):
        seed = None
    try:
        result = await asyncio.to_thread(
            _generate_with_qwen,
            text,
            preset,
            operation_id,
            image_pil,
            seed,
            True,
            resolve_prompt_model(bigger_model),
        )
        send_done(operation_id, "AI prompt ready")
        return web.json_response(
            {
                "ok": True,
                "text": result,
                "thinking": False,
                "model": DEFAULT_MODEL_ID,
                "used_image": image_pil is not None,
            }
        )
    except Exception as exc:
        LOGGER.exception("%s AI prompt enhancement failed", LOG_PREFIX)
        send_error(operation_id, str(exc))
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------


class TS_SuperPrompt(IO.ComfyNode):
    """Compact prompt node: microphone dictation plus Qwen prompt enhancement."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        options = preset_options()
        return IO.Schema(
            node_id="TS_SuperPrompt",
            display_name="TS Super Prompt",
            category="TS/LLM",
            essentials_category="Text",
            description=(
                "Voice prompt field with optional Qwen3.5 AI prompt enhancement for image, video, and music prompts. "
                "Attach an optional reference image right in the node (drag-drop / paste / file picker) — "
                "it is used by the Enhance button without running the workflow."
            ),
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Prompt field: recognized speech lands here, and the Ai prompt "
                        "button replaces the text with an enhanced prompt."
                    ),
                ),
                IO.Boolean.Input(
                    "high_quality",
                    default=False,
                    tooltip=(
                        "Enable to transcribe speech with Whisper turbo (large-v3 turbo). "
                        "Off: uses the fast base model."
                    ),
                ),
                IO.Combo.Input(
                    "system_preset",
                    options=options,
                    default=default_preset(options),
                    tooltip="System preset from qwen_3_vl_presets.json used to enhance the prompt.",
                ),
                IO.String.Input(
                    "attached_image",
                    default="",
                    tooltip=(
                        "Internal field: annotated path of the attached image "
                        "(filled by the Attach button in the node)."
                    ),
                    socketless=True,
                ),
                # LAST in the list, and it has to stay last. `widgets_values` is
                # positional: a new widget inserted anywhere else would shift
                # every value in every workflow already saved with this node.
                IO.String.Input(
                    "attached_image_2",
                    default="",
                    tooltip=(
                        "Internal field: annotated path of the second attached image. "
                        "With two images the first is the FIRST frame and this one the "
                        "LAST frame of the shot."
                    ),
                    socketless=True,
                ),
                # Тоже последним и по той же причине: `widgets_values`
                # позиционен. Сохранённый workflow приходит вообще без этого
                # поля, получает False и продолжает работать на прежней модели.
                IO.Boolean.Input(
                    "bigger_model",
                    default=False,
                    tooltip=(
                        "Off: the 2B prompt model (fast, already downloaded). "
                        "On: the 4B one — better prompts, about twice the download "
                        "and twice the VRAM. The 4B model is fetched on first use."
                    ),
                ),
                # Wired references, for when the frames come from the graph
                # rather than from the node's own attach buttons. ONE input
                # taking a batch, not a socket per frame: the order inside the
                # batch is what says which frame is which, so three or four
                # frames will need no new wiring. Optional, and last — an input
                # added anywhere earlier would renumber the sockets of every
                # workflow already saved with this node.
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip=(
                        "Optional reference images from the graph. A single image is a "
                        "plain reference; in a batch the FIRST image is the first frame "
                        "of the shot and the LAST is the last frame. Takes precedence "
                        "over images attached in the node. Each frame is shrunk to about "
                        "1 MP on the way in."
                    ),
                ),
            ],
            outputs=[IO.String.Output(display_name="text", tooltip="Prompt text (enhanced when enhancement runs).")],
            search_aliases=[
                "super prompt",
                "ai prompt",
                "prompt enhancer",
                "voice recognition",
                "qwen prompt",
                "speech to prompt",
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        text: str = "",
        high_quality: bool = False,
        system_preset: str = DEFAULT_PRESET,
        attached_image: str = "",
        attached_image_2: str = "",
        bigger_model: bool = False,
        images: Any = None,
        **_: Any,
    ) -> bool | str:
        if not isinstance(text, str):
            return "text must be a string."
        if not isinstance(high_quality, bool):
            return "high_quality must be a boolean."
        if not isinstance(system_preset, str):
            return "system_preset must be a string."
        if not isinstance(attached_image, str):
            return "attached_image must be a string (annotated filepath)."
        if not isinstance(attached_image_2, str):
            return "attached_image_2 must be a string (annotated filepath)."
        if not isinstance(bigger_model, bool):
            return "bigger_model must be a boolean."
        # Unknown ``system_preset`` values are intentionally accepted here:
        # ``_resolve_preset`` in _qwen.py silently falls back to the canonical
        # ``DEFAULT_PRESET`` ("Prompts enhance") so workflows saved before a
        # preset rename keep loading + running without an error.
        return True

    @classmethod
    def execute(
        cls,
        text: str = "",
        high_quality: bool = False,
        system_preset: str = DEFAULT_PRESET,
        attached_image: str = "",
        attached_image_2: str = "",
        bigger_model: bool = False,
        images: Any = None,
        **_: Any,
    ) -> IO.NodeOutput:
        _ = high_quality
        if not SUPER_PROMPT_ENHANCE_ON_EXECUTE:
            return IO.NodeOutput(text or "")

        # A wired batch wins outright, and as a whole: it already states the
        # order of the frames, so mixing it with the node's attachments could
        # only produce a sequence nobody asked for. Connecting images is an
        # explicit act in the graph, while an attachment is a leftover from the
        # last time someone pressed the button in the node.
        frames = _socket_images(images) or _load_attached_images(
            attached_image, attached_image_2)
        enhanced = _generate_with_qwen(
            text=text or "",
            system_preset=system_preset,
            operation_id=None,
            image=frames or None,
            model_id=resolve_prompt_model(bigger_model),
        )
        return IO.NodeOutput(enhanced)


NODE_CLASS_MAPPINGS = {"TS_SuperPrompt": TS_SuperPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_SuperPrompt": "TS Super Prompt"}
