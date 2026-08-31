"""Shared LiteRT-LM engine: Gemma 4 on-device, the runtime ComfyUI cannot see.

This is the LiteRT twin of :mod:`nodes.llm._qwen_engine` — model catalogue,
download, load/unload, generation — but the runtime underneath is a different
animal, and three of its properties shape everything here. All three were
measured on this machine before a line was written (2026-08-26):

* **ComfyUI cannot see this memory.** LiteRT runs on WebGPU/Direct3D, not CUDA,
  so ``torch.cuda.mem_get_info`` reports the same free bytes whether Gemma is
  resident or not — measured: 14.86 GB either way, while ``nvidia-smi`` moved
  between 5176 and 1340 MiB. ComfyUI's memory manager therefore counts our
  gigabytes as free and will happily overcommit them. That is why unloading is
  the DEFAULT here and not an option: ``generate()`` returns with the card
  clean unless the caller explicitly asks otherwise.
* **Unloading is cheap, loading is not.** Measured: ``engine.close()`` 0.9 s,
  warm reload 4.8 s, cold 13.7 s. So "unload after every run" costs about five
  seconds — worth paying, since the alternative is a sampler that OOMs on
  memory it was told it had.
* **The context is 4096 tokens, hard.** Our longest preset plus a picture
  measured 2636 prompt tokens, which fits with ~1400 to spare — but two
  pictures, or a long recording, will not. The budget is checked BEFORE
  generation and refused with a readable message rather than silently truncated.

Media crosses the boundary as FILES (``Content.ImageFile`` / ``AudioFile``),
never as tensors, so IMAGE inputs are written to ComfyUI's temp directory and
removed afterwards. The modality order is fixed — images, then the text, then
audio — because that is the order Gemma 4 was trained on.

The loader skips ``_``-prefixed modules, so this is never registered as a node.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from .._deps import TSDependencyManager

logger = logging.getLogger("comfyui_timesaver.litert_engine")
LOG_PREFIX = "[TS LiteRT]"

# --------------------------------------------------------------------------
# Catalogue
#
# ⚠️ ONLY the full artefacts. The repositories also carry ``-gpu`` and ``-web``
# variants that are 0.6 GB lighter, and the temptation to offer them is real —
# but their headers say what the card only hints at: measured with the vendor's
# own ``litertlm_peek``, ``gemma-4-E4B-it-gpu.litertlm`` holds 3 sections
# (tokenizer + one text decoder) against 12 in the full file, and the missing
# nine are exactly ``tf_lite_vision_encoder``, ``tf_lite_audio_encoder_hw`` and
# their adapters. They are text-only builds; a node that reads pictures cannot
# use them. The NPU builds (Qualcomm/Intel/Tensor) are out for the same class of
# reason — they target hardware a desktop ComfyUI does not have.
# --------------------------------------------------------------------------
MODEL_FOLDER_NAME = "litert"

CATALOGUE: dict[str, dict[str, Any]] = {
    "Gemma 4 E2B (2.4 GB)": {
        "repo_id": "hfmaster/Gemma-4-RT",
        "filename": "gemma-4-E2B-it-abliterated.litertlm",
        "size_gb": 2.41,
    },
    "Gemma 4 E4B (3.4 GB)": {
        "repo_id": "hfmaster/Gemma-4-RT",
        "filename": "gemma-4-E4B-it-abliterated.litertlm",
        "size_gb": 3.41,
    },
}

#: The two ends of the node's single quality switch. Named rather than indexed
#: so a future third artefact cannot silently become "the fast one".
FAST_MODEL = "Gemma 4 E2B (2.4 GB)"
HIGH_QUALITY_MODEL = "Gemma 4 E4B (3.4 GB)"

DEFAULT_MODEL = HIGH_QUALITY_MODEL

#: Context window of these artefacts. Not the architecture's limit — Gemma 4
#: itself does 128K — but what these files were built with, and the runtime
#: enforces it.
CONTEXT_TOKENS = 4096

#: Left for the answer when checking whether a prompt fits.
_MIN_ANSWER_TOKENS = 256

#: Cost of media, used to refuse BEFORE loading a 3 GB model.
#:
#: The audio figure is not an estimate: Google's Gemma documentation states 25
#: tokens per second of sound, and measuring prefill against clip length gave
#: exactly that — 1688 tokens at 60 s, 1938 at 70 s, 2313 at 85 s, i.e. 25 per
#: second plus the instruction. The picture figure is measured only.
_AUDIO_TOKENS_PER_SECOND = 25
_IMAGE_TOKENS = 250

#: ⚠️ The documented ceiling for ONE audio clip: "Audio supports a maximum
#: length of 30 seconds". Longer clips are not refused by this runtime — 85 s
#: still returned a sensible transcript here, and only at 90 s did it stop with
#: "4688 >= 4096" — but past 30 s the model is outside what it was trained for,
#: and nothing guarantees it keeps hearing the whole clip. So the voice helper
#: segments at this boundary rather than leaning on what happens to work.
AUDIO_CLIP_SECONDS = 30.0

#: Rough characters-per-token for English/Russian mixed prompt text. Only used
#: for the pre-flight estimate; the real count comes from the runtime.
_CHARS_PER_TOKEN = 3.6


class _EngineState:
    """Module-level state (§5 — class attributes are locked on V3 nodes)."""

    def __init__(self) -> None:
        self.engine: Any = None
        self.model_key: str | None = None
        self.backend: str | None = None
        self.mtp: bool | None = None
        self.loaded_at: float = 0.0
        self.lock = threading.RLock()


_state = _EngineState()


# --------------------------------------------------------------------------
# Runtime availability
# --------------------------------------------------------------------------
def import_runtime() -> Any:
    """The ``litert_lm`` module, or ``None`` when it is not installed."""
    return TSDependencyManager.import_optional("litert_lm")


def runtime_available() -> bool:
    return import_runtime() is not None


def require_runtime() -> Any:
    """The runtime, or a RuntimeError that says what to do about it.

    ⚠️ There are no Linux wheels for ``litert-lm`` (checked against every
    manylinux tag at 0.16.1), so on Linux this is not a "you forgot to pip
    install" situation and must not pretend to be one.
    """
    runtime = import_runtime()
    if runtime is not None:
        return runtime

    import sys

    if sys.platform.startswith("linux"):
        raise RuntimeError(
            f"{LOG_PREFIX} LiteRT-LM publishes no Linux wheels (checked at 0.16.1), "
            "so this node cannot run on Linux. Use TS Super Prompt, which runs on "
            "transformers and works on every platform."
        )
    raise RuntimeError(
        f"{LOG_PREFIX} The LiteRT-LM runtime is missing. Install it into the Python "
        "that runs ComfyUI:  python -m pip install litert-lm==0.16.1"
    )


# --------------------------------------------------------------------------
# Model files
# --------------------------------------------------------------------------
def models_root() -> Path:
    """``models/LLM/litert`` — created on demand, registered with ComfyUI."""
    import folder_paths

    base = Path(folder_paths.models_dir) / "LLM" / MODEL_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    if hasattr(folder_paths, "add_model_folder_path"):
        try:
            folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(base))
        except Exception as exc:  # noqa: BLE001 - registration is a convenience
            logger.debug("%s Could not register the model folder: %s", LOG_PREFIX, exc)
    return base


def model_names() -> list[str]:
    return list(CATALOGUE)


def model_path(model_key: str) -> Path:
    entry = CATALOGUE.get(model_key)
    if entry is None:
        raise RuntimeError(f"{LOG_PREFIX} Unknown model: {model_key!r}")
    return models_root() / str(entry["filename"])


def model_is_present(model_key: str) -> bool:
    path = model_path(model_key)
    if not path.is_file():
        return False
    # A download interrupted halfway leaves a short file that loads and then
    # dies deep inside the runtime; size is the cheap guard against that.
    expected = float(CATALOGUE[model_key]["size_gb"]) * 1024 ** 3
    return path.stat().st_size > expected * 0.9


def ensure_model(model_key: str, *, allow_download: bool = True) -> Path:
    """Local path to the artefact, downloading it only when allowed.

    ⚠️ ``allow_download=False`` is what the node passes on its first look, so a
    3.4 GB download never starts because someone opened a workflow.
    """
    path = model_path(model_key)
    if model_is_present(model_key):
        return path
    if not allow_download:
        entry = CATALOGUE[model_key]
        raise RuntimeError(
            f"{LOG_PREFIX} {model_key} is not downloaded yet "
            f"({entry['size_gb']:.1f} GB from {entry['repo_id']})."
        )

    from .._hf_download import snapshot_download_resilient

    entry = CATALOGUE[model_key]
    logger.info("%s Downloading %s from %s", LOG_PREFIX, entry["filename"], entry["repo_id"])
    snapshot_download_resilient(
        repo_id=str(entry["repo_id"]),
        local_dir=str(models_root()),
        revision="main",
        allow_patterns=[str(entry["filename"])],
        log=logger,
        log_prefix=LOG_PREFIX,
    )
    if not model_is_present(model_key):
        raise RuntimeError(
            f"{LOG_PREFIX} Download finished but {path.name} is missing or short."
        )
    return path


# --------------------------------------------------------------------------
# Token budget
# --------------------------------------------------------------------------
def estimate_prompt_tokens(
    text: str,
    *,
    images: int = 0,
    audio_seconds: float = 0.0,
) -> int:
    """Rough prompt cost, for refusing BEFORE a 3 GB model is loaded.

    Deliberately an estimate: the exact count needs a live conversation, and
    the whole point is to answer before paying for one. Measured constants,
    not guesses — see the module docstring.
    """
    words = max(0, len(str(text or "")) ) / _CHARS_PER_TOKEN
    return int(words + images * _IMAGE_TOKENS + audio_seconds * _AUDIO_TOKENS_PER_SECOND)


def fits_in_context(prompt_tokens: int, answer_tokens: int) -> bool:
    return prompt_tokens + max(answer_tokens, _MIN_ANSWER_TOKENS) <= CONTEXT_TOKENS


def context_error(prompt_tokens: int, answer_tokens: int) -> str:
    over = prompt_tokens + max(answer_tokens, _MIN_ANSWER_TOKENS) - CONTEXT_TOKENS
    return (
        f"{LOG_PREFIX} This does not fit the model's {CONTEXT_TOKENS}-token window: "
        f"the prompt is about {prompt_tokens} tokens and the answer needs "
        f"{answer_tokens}, which is {over} too many. Shorten the text, attach "
        f"fewer pictures, or cut the recording."
    )


# --------------------------------------------------------------------------
# Engine lifecycle
# --------------------------------------------------------------------------
def load_engine(
    model_key: str,
    *,
    backend: str = "GPU",
    mtp: bool = True,
    allow_download: bool = True,
    on_progress: Any = None,
) -> Any:
    """Load (or reuse) the engine for ``model_key``.

    Falls back to CPU when the GPU backend refuses to come up — the vendor's own
    studio does the same, and a machine without a working WebGPU stack is a real
    case rather than a theoretical one.
    """
    runtime = require_runtime()
    backend = str(backend or "GPU").upper()
    if backend not in {"GPU", "CPU"}:
        raise RuntimeError(f"{LOG_PREFIX} Unknown backend: {backend!r}")

    with _state.lock:
        if (
            _state.engine is not None
            and _state.model_key == model_key
            and _state.backend == backend
            and _state.mtp == bool(mtp)
        ):
            return _state.engine

        unload_engine()
        # ⚠️ Слова в этих сообщениях подобраны под регулярки `BUSY_STEPS` во
        # фронтенде: «downloading» — стадия скачивания, «loading … into memory»
        # — стадия загрузки. Переписывать вольно нельзя, панель перестанет их
        # различать.
        if not model_is_present(model_key) and allow_download and on_progress:
            entry = CATALOGUE[model_key]
            on_progress(f"Downloading {model_key} ({entry['size_gb']:.1f} GB)", 8.0)
        path = ensure_model(model_key, allow_download=allow_download)
        if on_progress:
            on_progress(f"Loading {model_key} into memory", 25.0)

        failures: list[str] = []
        for candidate in ([backend, "CPU"] if backend == "GPU" else [backend]):
            device = runtime.Backend.GPU() if candidate == "GPU" else runtime.Backend.CPU()
            started = time.perf_counter()
            try:
                engine = runtime.Engine(
                    model_path=str(path),
                    backend=device,
                    vision_backend=device,
                    # Audio stays on the CPU even when everything else is on the
                    # card: that is how the vendor ships it, and it keeps the
                    # recording out of VRAM entirely.
                    audio_backend=runtime.Backend.CPU(),
                    max_num_images=4,
                    enable_benchmark=True,
                    enable_speculative_decoding=bool(mtp),
                )
            except Exception as exc:  # noqa: BLE001 - reported after the loop
                failures.append(f"{candidate}: {exc}")
                continue

            _state.engine = engine
            _state.model_key = model_key
            _state.backend = candidate
            _state.mtp = bool(mtp)
            _state.loaded_at = time.time()
            logger.info(
                "%s Loaded %s on %s in %.1f s",
                LOG_PREFIX, model_key, candidate, time.perf_counter() - started,
            )
            if candidate != backend:
                logger.warning("%s GPU backend unavailable, running on CPU.", LOG_PREFIX)
            return engine

        raise RuntimeError(f"{LOG_PREFIX} Could not load {model_key}. " + " | ".join(failures))


def unload_engine() -> bool:
    """Drop the engine and give the card back. True when something was freed."""
    with _state.lock:
        engine, _state.engine = _state.engine, None
        _state.model_key = None
        _state.backend = None
        _state.mtp = None
        if engine is None:
            return False
        try:
            engine.close()
        except Exception as exc:  # noqa: BLE001 - closing must never be fatal
            logger.debug("%s engine.close() failed: %s", LOG_PREFIX, exc)
        logger.info("%s Model unloaded, VRAM released.", LOG_PREFIX)
        return True


def engine_status() -> dict[str, Any]:
    with _state.lock:
        return {
            "loaded": _state.engine is not None,
            "model": _state.model_key,
            "backend": _state.backend,
            "mtp": _state.mtp,
            "loaded_at": _state.loaded_at,
        }


# --------------------------------------------------------------------------
# Media staging
# --------------------------------------------------------------------------
def _temp_dir() -> Path:
    import folder_paths

    base = Path(folder_paths.get_temp_directory()) / "ts_litert"
    base.mkdir(parents=True, exist_ok=True)
    return base


def stage_images(image: Any, *, max_side: int = 1024, limit: int = 4) -> list[Path]:
    """Write an IMAGE batch out as PNG files for the runtime to read.

    LiteRT takes paths, not tensors, so the tensor has to land on disk. Frames
    are shrunk to ``max_side`` first: every pixel costs context, and the window
    is 4096 tokens wide.
    """
    if image is None:
        return []

    import numpy as np
    from PIL import Image

    frames = image if getattr(image, "ndim", 0) == 4 else None
    if frames is None:
        raise RuntimeError(f"{LOG_PREFIX} IMAGE input must be [B, H, W, C].")

    written: list[Path] = []
    directory = _temp_dir()
    for index in range(min(int(frames.shape[0]), limit)):
        array = (frames[index].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(array)
        if max(pil.size) > max_side:
            scale = max_side / max(pil.size)
            pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
                             Image.LANCZOS)
        path = directory / f"frame_{uuid.uuid4().hex}.png"
        pil.save(path)
        written.append(path)
    return written


def stage_audio(audio: Any, *, max_seconds: float = 30.0) -> tuple[Path | None, float]:
    """Write a ComfyUI AUDIO dict out as 16 kHz mono WAV.

    Returns the path and the duration actually written. Longer recordings are
    cut, because audio costs ~28 tokens a second and the caller is expected to
    have segmented anything longer (see ``transcribe_long`` in the voice
    helper).
    """
    if not audio:
        return None, 0.0

    import wave

    import numpy as np
    import torch

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    limit = int(max_seconds * sample_rate)
    if waveform.shape[-1] > limit:
        waveform = waveform[..., :limit]

    # ⚠️ Written with the standard library, NOT ``torchaudio.save``: since
    # torchaudio 2.9 that call routes through TorchCodec, which is not part of
    # a ComfyUI install and raises ImportError on a machine that otherwise has
    # everything. A 16-bit mono WAV needs no third-party writer.
    samples = waveform.to(torch.float32).cpu().numpy().reshape(-1)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    path = _temp_dir() / f"audio_{uuid.uuid4().hex}.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path, waveform.shape[-1] / float(sample_rate)


def cleanup(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def _chunk_text(chunk: Any) -> str:
    """Text out of one stream chunk.

    The runtime yields ``{'role': ..., 'content': [{'type': 'text', ...}]}``,
    and thinking arrives on its own channel — which we drop, because the node
    returns a prompt, not a monologue about writing one.
    """
    if isinstance(chunk, dict):
        parts = chunk.get("content") or []
        out: list[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(str(part.get("text", "")))
        return "".join(out)
    if chunk is None:
        return ""
    return str(chunk)


def generate(
    *,
    model_key: str,
    system_prompt: str,
    user_text: str,
    images: Sequence[Path] = (),
    audio: Path | None = None,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 64,
    max_new_tokens: int = 512,
    seed: int | None = None,
    thinking: bool = False,
    backend: str = "GPU",
    mtp: bool = True,
    keep_loaded: bool = False,
    allow_download: bool = True,
    on_progress: Any = None,
) -> dict[str, Any]:
    """One prompt in, one answer out, card clean on the way back.

    ``seed=None`` lets the runtime choose; pass one to make a repeat press
    sample differently (or to reproduce an answer exactly).

    ``keep_loaded=True`` skips the unload — faster on repeated presses, but the
    memory it holds is memory ComfyUI believes it still has (see the module
    docstring), so the caller must say so deliberately.
    """
    runtime = require_runtime()

    # ⚠️ ВЕСЬ разговор с движком — под одним замком, а не только загрузка.
    #
    # Замерено крашем: пока шла генерация на E4B, фронтенд параллельно позвал
    # `preload` на E2B; тот вызвал `unload_engine()`, закрыв движок прямо из-под
    # работающего `send_message_async`, и процесс упал с
    # `access violation reading 0x0000000000000000`. Питоновского исключения тут
    # не будет — падает нативный код, вместе с ComfyUI.
    #
    # Замок реентерабельный, поэтому `load_engine` внутри берёт его же. Плата —
    # второй запрос ждёт первый; это правильная плата: движок в процессе один.
    #
    # Об ожидании говорим вслух — приём взят у Qwen-ноды, где `MODEL_LOCK`
    # сначала пробуется неблокирующе. Иначе второе нажатие выглядит зависшим:
    # полоса стоит, а почему — неизвестно.
    if not _state.lock.acquire(blocking=False):
        if on_progress is not None:
            on_progress("Waiting for the model to finish the previous prompt", 2.0)
        _state.lock.acquire()
    try:
        return _generate_locked(
            runtime=runtime, model_key=model_key, system_prompt=system_prompt,
            user_text=user_text, images=images, audio=audio, temperature=temperature,
            top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens, seed=seed,
            thinking=thinking,
            backend=backend, mtp=mtp, keep_loaded=keep_loaded,
            allow_download=allow_download, on_progress=on_progress,
        )
    finally:
        _state.lock.release()


def _generate_locked(
    *,
    runtime: Any,
    model_key: str,
    system_prompt: str,
    user_text: str,
    images: Sequence[Path],
    audio: Path | None,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int | None,
    thinking: bool,
    backend: str,
    mtp: bool,
    keep_loaded: bool,
    allow_download: bool,
    on_progress: Any,
) -> dict[str, Any]:
    """Тело :func:`generate`, вызываемое строго под ``_state.lock``."""
    engine = load_engine(model_key, backend=backend, mtp=mtp,
                         allow_download=allow_download, on_progress=on_progress)

    conversation = None
    started = time.perf_counter()
    try:
        conversation = engine.create_conversation(
            messages=[],
            system_message=(str(system_prompt).strip() or None),
            # ``seed=None`` leaves the runtime to pick one, which is what a
            # plain node run wants. A caller with a "generate again" button
            # sends a fresh seed on every press so the same prompt sampled
            # twice does not come back word for word.
            sampler_config=runtime.SamplerConfig(
                temperature=float(temperature), top_k=int(top_k), top_p=float(top_p),
                seed=(None if seed is None else int(seed)),
            ),
            thinking_config=runtime.ThinkingConfig(
                enable_thinking=bool(thinking),
                thinking_token_budget=(512 if thinking else 0),
            ),
            # Keeps the thinking channel out of the KV cache, so a second turn
            # does not read back the model's own scratch work as context.
            filter_channel_content_from_kv_cache=True,
        )

        # Fixed modality order: pictures, then the instruction, then sound.
        #
        # ⚠️ A lone ``Content.Text`` is NOT a valid message — the runtime rejects
        # it with "Unsupported message type" and wants a bare ``str`` instead.
        # Only a multi-part payload goes through ``Contents``.
        if not images and audio is None:
            payload: Any = str(user_text)
        else:
            parts: list[Any] = [runtime.Content.ImageFile(str(Path(p).resolve())) for p in images]
            parts.append(runtime.Content.Text(str(user_text)))
            if audio is not None:
                parts.append(runtime.Content.AudioFile(str(Path(audio).resolve())))
            payload = runtime.Contents(parts)

        pieces: list[str] = []
        # ⚠️ Длину ответа не знает никто, и делать вид, что знаем, — плохо в обе
        # стороны. Первая версия делила номер куска на потолок токенов (700), а
        # кусков приходит около тридцати: полоса доползала до 33% и прыгала на
        # сто. Здесь она приближается к 95% асимптотически — движется всегда,
        # никогда не обещает конца и не откатывается назад. Сотню ставит
        # вызывающий, когда текст уже на руках.
        import math

        _PACE = 22.0        # столько кусков даёт примерно две трети пути
        reported = -1
        for index, chunk in enumerate(conversation.send_message_async(payload)):
            pieces.append(_chunk_text(chunk))
            if on_progress is not None:
                percent = 30.0 + 65.0 * (1.0 - math.exp(-index / _PACE))
                step = int(percent // 5)
                if step != reported:
                    reported = step
                    on_progress("Generating the prompt", round(percent, 1))
        text = "".join(pieces).strip()

        benchmark: dict[str, Any] = {}
        try:
            info = conversation.get_benchmark_info()
            benchmark = {
                "prefill_tokens": getattr(info, "last_prefill_token_count", None),
                "prefill_tps": getattr(info, "last_prefill_tokens_per_second", None),
                "decode_tokens": getattr(info, "last_decode_token_count", None),
                "decode_tps": getattr(info, "last_decode_tokens_per_second", None),
                "ttft": getattr(info, "time_to_first_token_in_second", None),
            }
        except Exception as exc:  # noqa: BLE001 - metrics are optional
            logger.debug("%s No benchmark info: %s", LOG_PREFIX, exc)

        return {
            "text": text,
            "seconds": round(time.perf_counter() - started, 2),
            "benchmark": benchmark,
            "backend": engine_status().get("backend"),
        }
    finally:
        if conversation is not None:
            try:
                conversation.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s conversation.close() failed: %s", LOG_PREFIX, exc)
        if not keep_loaded:
            unload_engine()
