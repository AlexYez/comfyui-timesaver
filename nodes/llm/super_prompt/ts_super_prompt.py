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

from aiohttp import web

import folder_paths
from comfy_api.v0_0_2 import IO

from ._helpers import (
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
    SUPER_PROMPT_MAX_IMAGE_SIZE,
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


# ---------------------------------------------------------------------------
# Attached image helpers
# ---------------------------------------------------------------------------

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def _resolve_annotated_image_path(annotated: str) -> str:
    """Resolve a ComfyUI annotated filepath (``"name.png [input]"``) to a real path.

    Returns an empty string if the input is empty or unresolvable. Falls back
    to checking the raw string as a path if ``folder_paths`` rejects it.
    """
    path = str(annotated or "").strip()
    if not path:
        return ""
    try:
        resolved = folder_paths.get_annotated_filepath(path)
    except Exception:
        resolved = ""
    if resolved and Path(resolved).is_file():
        return resolved
    if Path(path).is_file():
        return path
    return ""


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


def _attached_image_as_tensor(annotated: str):
    """Load the attached image as a ComfyUI IMAGE tensor (``[1, H, W, C]``)."""
    pil = _load_attached_image_pil(annotated)
    if pil is None:
        return None
    try:
        import numpy as np
        import torch
    except Exception as exc:
        LOGGER.warning("%s torch/numpy unavailable for attached image: %s", LOG_PREFIX, exc)
        return None
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# Voice recognition HTTP routes
# ---------------------------------------------------------------------------

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
            fields[part.name] = await part.text()


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
    attached_image = str(data.get("attached_image") or "")
    image_pil = _load_attached_image_pil(attached_image) if attached_image else None
    if attached_image and image_pil is None:
        LOGGER.warning(
            "%s Attached image %r could not be loaded; falling back to text-only enhance.",
            LOG_PREFIX,
            attached_image,
        )

    # Racy fast-fail for obviously-busy Qwen. Internal _generate_with_qwen still
    # acquires MODEL_LOCK with proper blocking, so this only spares the caller
    # the trip into the worker thread when the model is already in use.
    if MODEL_LOCK.locked():
        return web.json_response(
            {"error": "Qwen is busy with another request."},
            status=429,
        )

    operation_id = str(data.get("operation_id") or "")
    try:
        result = await asyncio.to_thread(
            _generate_with_qwen,
            text,
            preset,
            operation_id,
            image_pil,
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
                        "Поле промпта: сюда попадает распознанная речь, "
                        "а кнопка Ai prompt заменяет текст улучшенным промптом."
                    ),
                ),
                IO.Boolean.Input(
                    "high_quality",
                    default=False,
                    tooltip=(
                        "Включите, чтобы распознавать речь моделью Whisper turbo (large-v3 turbo). "
                        "Выключено: используется быстрая base."
                    ),
                ),
                IO.Combo.Input(
                    "system_preset",
                    options=options,
                    default=default_preset(options),
                    tooltip="Выберите системный пресет из qwen_3_vl_presets.json для улучшения промпта.",
                ),
                IO.String.Input(
                    "attached_image",
                    default="",
                    tooltip=(
                        "Внутреннее поле: аннотированный путь приложенного изображения "
                        "(заполняется кнопкой Attach в самой ноде)."
                    ),
                    socketless=True,
                ),
            ],
            outputs=[IO.String.Output(display_name="text")],
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
        **_: Any,
    ) -> IO.NodeOutput:
        _ = high_quality
        if not SUPER_PROMPT_ENHANCE_ON_EXECUTE:
            return IO.NodeOutput(text or "")

        image_pil = _load_attached_image_pil(attached_image) if attached_image else None
        enhanced = _generate_with_qwen(
            text=text or "",
            system_preset=system_preset,
            operation_id=None,
            image=image_pil,
        )
        return IO.NodeOutput(enhanced)


NODE_CLASS_MAPPINGS = {"TS_SuperPrompt": TS_SuperPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_SuperPrompt": "TS Super Prompt"}
