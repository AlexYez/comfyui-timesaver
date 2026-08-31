"""TS Super Prompt RT — the same prompt work on Google's on-device runtime.

Sibling of TS Super Prompt, and deliberately a separate node rather than a mode
inside it: the two run different models under different constraints, and the
memory rules of this one are strict enough that hiding them behind a checkbox
would mislead. Measured on an RTX 3080 Ti Laptop (2026-08-26):

    Gemma 4 E4B via LiteRT   43.4 tok/s decode, ~1.7 GB VRAM
    Qwen3.5-4B via bf16      19.8 tok/s decode,  8.5 GB VRAM

Twice the speed at a fifth of the memory — but ComfyUI's memory manager cannot
see a single byte of it, because LiteRT runs on WebGPU rather than CUDA. That is
why the model is unloaded when the work is done unless the user says otherwise,
and why the ``keep_loaded`` tooltip says plainly what keeping it costs.

Routes, mirroring the Qwen node so the frontend stays familiar:

- POST /ts_super_prompt_rt/enhance    : text (+ pictures) → enhanced prompt.
- POST /ts_super_prompt_rt/transcribe : multipart audio → text.
- GET  /ts_super_prompt_rt/status     : runtime present? model downloaded?
- POST /ts_super_prompt_rt/unload     : give the card back right now.

node_id: TS_SuperPromptRT
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from comfy_api.v0_0_2 import IO

from ..._shared import make_route_registrars
from .._litert_engine import (
    CATALOGUE,
    FAST_MODEL,
    HIGH_QUALITY_MODEL,
    cleanup,
    context_error,
    engine_status,
    estimate_prompt_tokens,
    fits_in_context,
    generate,
    model_is_present,
    model_names,
    runtime_available,
    stage_images,
    unload_engine,
)
from ._helpers import (
    ENHANCE_MAX_TEXT_LEN,
    LOG_PREFIX,
    TRANSCRIBE_MAX_UPLOAD,
    default_preset,
    logger,
    preset_generation_params,
    preset_options,
    resolve_annotated_path,
    resolve_preset,
    send_done,
    send_error,
    send_progress,
)
from ._voice import transcribe as transcribe_audio

_DEFAULT_PRESET = default_preset(preset_options())


# ---------------------------------------------------------------------------
# Shared work: one enhance, whatever asked for it
# ---------------------------------------------------------------------------
def _seed_from_body(raw: object) -> int | None:
    """The request's seed, or ``None`` when it carries none worth using.

    A missing, empty or unparsable field means "let the runtime choose" rather
    than an error: an older frontend that never learned to send one must keep
    working against a freshly updated server.
    """
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None



def _enhance(
    *,
    text: str,
    preset: str,
    model_key: str,
    image_paths: list[Path],
    keep_loaded: bool,
    seed: int | None = None,
    operation_id: str | None = None,
) -> str:
    resolved, system_prompt, gen_params = resolve_preset(preset)
    params = preset_generation_params(resolved, gen_params)

    prompt_tokens = estimate_prompt_tokens(
        f"{system_prompt}\n{text}", images=len(image_paths),
    )
    if not fits_in_context(prompt_tokens, params["max_new_tokens"]):
        raise RuntimeError(context_error(prompt_tokens, params["max_new_tokens"]))

    send_progress(operation_id, "Preparing the prompt", 5.0)
    result = generate(
        model_key=model_key,
        system_prompt=system_prompt,
        user_text=text,
        images=image_paths,
        keep_loaded=keep_loaded,
        seed=seed,
        allow_download=True,
        on_progress=lambda stage, percent: send_progress(operation_id, stage, percent),
        **params,
    )
    send_progress(operation_id, "Generating the prompt", 100.0)
    send_done(operation_id, str(result.get("text", "")))

    benchmark = result.get("benchmark") or {}
    logger.info(
        "%s %s on %s: %s tokens in %.1f s (%.1f tok/s)",
        LOG_PREFIX, resolved, result.get("backend"),
        benchmark.get("decode_tokens"), result.get("seconds", 0.0),
        float(benchmark.get("decode_tps") or 0.0),
    )
    return str(result.get("text", "")).strip()


def model_for(high_quality: bool | None) -> str:
    """The one place that turns the switch into a model name.

    Both jobs — writing the prompt and hearing the recording — go through here,
    so the node can never end up transcribing with one model and enhancing with
    another.
    """
    return HIGH_QUALITY_MODEL if bool(high_quality) else FAST_MODEL


def _images_from_widgets(*annotated: str) -> list[Path]:
    """Attached-image widgets → real paths, skipping the ones that are empty."""
    paths: list[Path] = []
    for value in annotated:
        resolved = resolve_annotated_path(value)
        if resolved is not None:
            paths.append(resolved)
    return paths


class TS_SuperPromptRT(IO.ComfyNode):
    """Prompt writing and dictation on Gemma 4 through LiteRT-LM."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        options = preset_options()
        return IO.Schema(
            node_id="TS_SuperPromptRT",
            display_name="TS Super Prompt RT",
            category="TS/LLM",
            essentials_category="Text",
            description=(
                "Prompt field with Gemma 4 enhancement on Google's on-device LiteRT runtime: "
                "about twice the speed of the transformers path at a fifth of the VRAM, and "
                "the same model also transcribes speech. The model is unloaded when the work "
                "is done, because ComfyUI cannot see memory held outside CUDA."
            ),
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Prompt field: recognised speech lands here, and the Enhance "
                        "button replaces the text with the written prompt."
                    ),
                ),
                # Mirrors TS Super Prompt's own layout, down to the position of
                # this switch: one control decides the model for BOTH jobs,
                # because the same artefact writes the prompt and hears the
                # recording. Two switches would imply two models.
                IO.Boolean.Input(
                    "high_quality",
                    default=False,
                    tooltip=(
                        "Off: Gemma 4 E2B — quicker, 2.4 GB. On: E4B — better prompts and "
                        "a more careful ear, 3.4 GB. The same model does both the prompt "
                        "and the speech, so this one switch decides both."
                    ),
                ),
                IO.Combo.Input(
                    "system_preset",
                    options=options,
                    default=_DEFAULT_PRESET,
                    tooltip="System preset from qwen_3_vl_presets.json — the same set the Qwen node uses.",
                ),
                IO.String.Input(
                    "attached_image",
                    default="",
                    tooltip="Internal field: path of the image attached in the node.",
                    socketless=True,
                ),
                IO.String.Input(
                    "attached_image_2",
                    default="",
                    tooltip="Internal field: path of the second attached image.",
                    socketless=True,
                ),
                IO.Boolean.Input(
                    "keep_loaded",
                    default=False,
                    tooltip=(
                        "Off: the model leaves the card as soon as the prompt is written "
                        "(about a second) and comes back in roughly five. On: it stays "
                        "resident — faster on repeated presses, but ComfyUI CANNOT see this "
                        "memory and will plan its own sampling as if the card were free."
                    ),
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip=(
                        "Optional reference images from the graph. Takes precedence over "
                        "images attached in the node. Up to four frames, each shrunk to "
                        "1024 px on the way in — every pixel costs context, and the window "
                        "is 4096 tokens."
                    ),
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip=(
                        "Optional recording to transcribe. The transcript replaces the text "
                        "field before enhancement, so a spoken idea can go straight into a "
                        "prompt. Longer recordings are transcribed in 30-second segments."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="text",
                    tooltip="Prompt text — enhanced when enhancement runs, otherwise passed through.",
                ),
            ],
            search_aliases=[
                "super prompt rt",
                "litert",
                "gemma",
                "on-device prompt",
                "prompt enhancer",
                "speech to prompt",
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        text: str = "",
        high_quality: bool = False,
        system_preset: str = _DEFAULT_PRESET,
        attached_image: str = "",
        attached_image_2: str = "",
        keep_loaded: bool = False,
        images: Any = None,
        audio: Any = None,
        **_: Any,
    ) -> bool | str:
        if not isinstance(text, str):
            return "text must be a string."
        if not isinstance(high_quality, bool):
            return "high_quality must be a boolean."
        if not isinstance(system_preset, str):
            return "system_preset must be a string."
        if not isinstance(keep_loaded, bool):
            return "keep_loaded must be a boolean."
        # An unknown preset or model name is accepted here and resolved to a
        # working default at run time: a workflow saved before a rename must
        # still open. Same rule as the Qwen node.
        return True

    @classmethod
    def execute(
        cls,
        text: str = "",
        high_quality: bool = False,
        system_preset: str = _DEFAULT_PRESET,
        attached_image: str = "",
        attached_image_2: str = "",
        keep_loaded: bool = False,
        images: Any = None,
        audio: Any = None,
        **_: Any,
    ) -> IO.NodeOutput:
        model_key = model_for(high_quality)
        prompt_text = str(text or "")

        # A recording on the wire is dictation: it becomes the text, and the
        # text is then enhanced exactly as if it had been typed.
        if audio is not None:
            prompt_text = transcribe_audio(
                audio, model_key=model_key, keep_loaded=True,
            ) or prompt_text

        staged: list[Path] = []
        try:
            if images is not None:
                staged = stage_images(images)
                image_paths = list(staged)
            else:
                image_paths = _images_from_widgets(attached_image, attached_image_2)

            if not prompt_text.strip() and not image_paths:
                # Nothing to work from: pass the field through rather than
                # loading three gigabytes to describe silence.
                if not keep_loaded:
                    unload_engine()
                return IO.NodeOutput(prompt_text)

            enhanced = _enhance(
                text=prompt_text or "Write the prompt for the attached reference.",
                preset=system_preset,
                model_key=model_key,
                image_paths=image_paths,
                keep_loaded=keep_loaded,
            )
            return IO.NodeOutput(enhanced or prompt_text)
        finally:
            cleanup(staged)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
try:  # ComfyUI is running
    import server as _comfy_server

    _PROMPT_SERVER = getattr(_comfy_server.PromptServer, "instance", None)
except Exception:  # noqa: BLE001 - imported by the test suite without a server
    _PROMPT_SERVER = None

_register_get, _register_post = make_route_registrars(
    _PROMPT_SERVER, lambda message: logger.warning("%s %s", LOG_PREFIX, message)
)


def _json(payload: dict[str, Any], status: int = 200):
    from aiohttp import web

    return web.json_response(payload, status=status)


@_register_get("/ts_super_prompt_rt/status")
async def _status_route(_request):
    """Per-model readiness, keyed by model name.

    ⚠️ The shape is dictated by the frontend, which looks up
    ``data[activeModelName]`` — the same contract the Qwen node's voice status
    route has. Keeping it identical is what let the interface be a copy.
    """
    available = runtime_available()
    payload: dict[str, Any] = {
        name: {
            "downloaded": model_is_present(name),
            "size_gb": CATALOGUE[name]["size_gb"],
            "repo": CATALOGUE[name]["repo_id"],
            # An absent runtime is reported as a missing dependency rather than
            # an error, so the node greys its buttons out and says why.
            "missing_dependencies": [] if available else ["litert-lm"],
        }
        for name in model_names()
    }
    payload["runtime"] = available
    payload["engine"] = engine_status()
    return _json(payload)


@_register_get("/ts_super_prompt_rt/model_status")
async def _model_status_route(request):
    """What the download prompt needs: is it here, how big, and where it goes."""
    high_quality = str(request.query.get("high_quality", "")).strip() in {"1", "true", "yes"}
    name = model_for(high_quality)
    entry = CATALOGUE[name]
    from .._litert_engine import models_root

    return _json({
        "ok": True,
        "model_id": entry["repo_id"],
        "model": name,
        "present": model_is_present(name),
        "size_gb": entry["size_gb"],
        "local_dir": str(models_root()),
        # No architecture gate here: unlike the transformers path, these
        # artefacts either load in this runtime or do not exist for it.
        "supported": runtime_available(),
    })


@_register_post("/ts_super_prompt_rt/preload")
async def _preload_route(request):
    """Warm the model up so the first press does not wait for a 3 GB read."""
    import asyncio

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}

    name = model_for(body.get("high_quality"))
    try:
        from .._litert_engine import load_engine

        await asyncio.to_thread(load_engine, name, allow_download=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("%s Preload failed: %s", LOG_PREFIX, exc)
        return _json({"ok": False, "error": str(exc)}, status=500)
    return _json({"ok": True, "model": name})


@_register_post("/ts_super_prompt_rt/cancel")
async def _cancel_route(_request):
    """There is no mid-generation cancel in this runtime — say so honestly.

    ⚠️ The frontend offers a Cancel button because the Qwen node can honour it.
    Here the generation runs inside one blocking call, and pretending to stop it
    would leave the UI unlocked while the model kept writing. Returning
    ``cancelled: False`` lets the button report the truth instead.
    """
    return _json({"ok": True, "cancelled": False})


@_register_post("/ts_super_prompt_rt/unload")
async def _unload_route(_request):
    return _json({"freed": unload_engine()})


@_register_post("/ts_super_prompt_rt/enhance")
async def _enhance_route(request):
    import asyncio

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - malformed body
        return _json({"error": "Expected a JSON body."}, status=400)

    text = str(body.get("text") or "")
    if len(text) > ENHANCE_MAX_TEXT_LEN:
        return _json({"error": f"Text longer than {ENHANCE_MAX_TEXT_LEN} characters."}, status=413)

    preset = str(body.get("system_preset") or _DEFAULT_PRESET)
    model_key = model_for(body.get("high_quality"))
    keep_loaded = bool(body.get("keep_loaded") or False)
    # The button sends a fresh seed on every press: without one the runtime
    # answered the same prompt with the same words, and "generate again" was
    # indistinguishable from a no-op. A body without the field still works.
    seed = _seed_from_body(body.get("seed"))
    operation_id = str(body.get("operation_id") or uuid.uuid4().hex)
    image_paths = _images_from_widgets(
        str(body.get("attached_image") or ""), str(body.get("attached_image_2") or ""),
    )

    if not text.strip() and not image_paths:
        return _json({"error": "Nothing to enhance: no text and no image."}, status=400)

    try:
        # The runtime call is blocking and long; keeping it off the event loop
        # is what lets the canvas stay responsive while the model writes.
        enhanced = await asyncio.to_thread(
            _enhance,
            text=text or "Write the prompt for the attached reference.",
            preset=preset,
            model_key=model_key,
            image_paths=image_paths,
            keep_loaded=keep_loaded,
            seed=seed,
            operation_id=operation_id,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the user verbatim
        logger.error("%s Enhance failed: %s", LOG_PREFIX, exc)
        send_error(operation_id, str(exc))
        return _json({"error": str(exc)}, status=500)

    return _json({"ok": True, "text": enhanced, "operation_id": operation_id})


@_register_post("/ts_super_prompt_rt/transcribe")
async def _transcribe_route(request):
    import asyncio
    import tempfile

    reader = await request.multipart()
    audio_bytes = b""
    model_key = FAST_MODEL
    keep_loaded = False

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "audio":
            while True:
                chunk = await part.read_chunk()
                if not chunk:
                    break
                audio_bytes += chunk
                if len(audio_bytes) > TRANSCRIBE_MAX_UPLOAD:
                    return _json({"error": "Recording too large."}, status=413)
        elif part.name == "high_quality":
            model_key = model_for((await part.text()).strip().lower() in {"1", "true", "yes"})
        elif part.name == "keep_loaded":
            keep_loaded = (await part.text()).strip().lower() in {"1", "true", "yes"}

    if not audio_bytes:
        return _json({"error": "No audio in the request."}, status=400)

    suffix = ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(audio_bytes)
        upload = Path(handle.name)

    try:
        audio = await asyncio.to_thread(_decode_upload, upload)
        text = await asyncio.to_thread(
            transcribe_audio,
            audio,
            model_key=model_key,
            keep_loaded=keep_loaded,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("%s Transcribe failed: %s", LOG_PREFIX, exc)
        return _json({"error": str(exc)}, status=500)
    finally:
        cleanup([upload])

    return _json({"ok": True, "text": text})


def _decode_upload(path: Path) -> dict[str, Any]:
    """Browser recording → ComfyUI AUDIO dict, via the pack's own ffmpeg."""
    import subprocess

    import numpy as np
    import torch

    from ..._ffmpeg import require_ffmpeg

    executable = require_ffmpeg()
    process = subprocess.run(
        [executable, "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
         "-f", "f32le", "-"],
        capture_output=True,
        check=False,
    )
    if process.returncode != 0 or not process.stdout:
        tail = process.stderr.decode("utf-8", "ignore")[-400:]
        raise RuntimeError(f"{LOG_PREFIX} Could not decode the recording. {tail}")
    samples = np.frombuffer(process.stdout, dtype=np.float32).copy()
    waveform = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": 16000}


NODE_CLASS_MAPPINGS = {"TS_SuperPromptRT": TS_SuperPromptRT}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_SuperPromptRT": "TS Super Prompt RT"}
