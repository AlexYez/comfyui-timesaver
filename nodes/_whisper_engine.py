"""Unified native OpenAI Whisper engine shared by TS Whisper and TS Super Prompt.

Single source of truth for:
  * the Whisper model registry (native model names + on-disk filenames),
  * the on-disk model directory (``models/whisper``),
  * an in-memory model cache shared across nodes, so loading e.g. ``large-v3``
    in TS Whisper and again for Super Prompt voice reuses the exact same model
    object instead of holding two copies in VRAM/RAM.

Heavy deps (``torch``, ``whisper``, ``numpy``, ``torchaudio``) are imported
lazily so this module is cheap to import at ComfyUI startup even when the user
never touches a Whisper node.

UI-agnostic: progress is reported only through an optional ``progress_cb``
callback, never by importing any node's event system. Private (``_``-prefixed)
so the node loader skips it (see ``__init__.py`` discovery rules).
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import folder_paths

LOGGER = logging.getLogger("comfyui_timesaver.whisper_engine")
LOG_PREFIX = "[TS Whisper Engine]"

ProgressCb = Optional[Callable[[str, dict], None]]

AUDIO_SAMPLE_RATE = 16000

# Native whisper model names (whisper.available_models()). ``turbo`` is the
# alias OpenAI ships for the distilled large-v3-turbo checkpoint.
ALL_MODELS = (
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3", "turbo",
)

# Approximate download sizes (bytes) for the file-size download monitor.
MODEL_SIZES = {
    "tiny": 75_000_000,
    "tiny.en": 75_000_000,
    "base": 145_000_000,
    "base.en": 145_000_000,
    "small": 465_000_000,
    "small.en": 465_000_000,
    "medium": 1_500_000_000,
    "medium.en": 1_500_000_000,
    "large-v1": 3_000_000_000,
    "large-v2": 3_000_000_000,
    "large-v3": 3_000_000_000,
    "turbo": 1_620_000_000,
}

# Cached on-disk filename per model. Native whisper saves ``turbo`` to
# ``large-v3-turbo.pt``; everything else is ``<name>.pt``.
MODEL_FILE_NAMES = {
    "turbo": "large-v3-turbo.pt",
}

# The distilled turbo checkpoint is transcription-only — it has no
# X->English translation head.
MODELS_WITHOUT_TRANSLATE = {"turbo"}

# Shared on-disk location and in-memory cache. Both Whisper nodes import these
# (directly or via super_prompt/_helpers re-export) so models are shared.
WHISPER_DIR = Path(getattr(folder_paths, "models_dir", Path.cwd() / "models")) / "whisper"
DOWNLOAD_LOCK = threading.Lock()
MODEL_CACHE: dict[tuple[str, str, bool], Any] = {}

# torchaudio resampler cache keyed by (orig, new, device, quality).
_RESAMPLER_CACHE: dict[tuple[int, int, str, str], Any] = {}


# --------------------------------------------------------------------------- #
# Runtime / registry helpers
# --------------------------------------------------------------------------- #
def _emit(cb: ProgressCb, stage: str, **info: Any) -> None:
    if cb is None:
        return
    try:
        cb(stage, info)
    except Exception as exc:  # noqa: BLE001 - progress reporting must never fail a run
        LOGGER.debug("%s progress callback failed: %s", LOG_PREFIX, exc)


def missing_runtime_packages() -> list[str]:
    missing = []
    if importlib.util.find_spec("torch") is None:
        missing.append("torch")
    if importlib.util.find_spec("whisper") is None:
        missing.append("openai-whisper")
    if importlib.util.find_spec("numpy") is None:
        missing.append("numpy")
    return missing


def load_runtime():
    """Lazily import and return ``(torch, whisper)``; raise a clear error if missing."""
    missing = missing_runtime_packages()
    if missing:
        raise RuntimeError(
            "Missing dependencies for native Whisper: "
            f"{', '.join(missing)}. Run `pip install -r requirements.txt` and restart ComfyUI."
        )
    import torch
    import whisper
    return torch, whisper


def supports_translate(name: str) -> bool:
    return name not in MODELS_WITHOUT_TRANSLATE


def model_file_path(name: str) -> Path:
    return WHISPER_DIR / MODEL_FILE_NAMES.get(name, f"{name}.pt")


def is_model_cached(name: str) -> bool:
    model_file = model_file_path(name)
    try:
        return model_file.is_file() and model_file.stat().st_size > 10_000_000
    except OSError:
        return False


def _ensure_runtime_dirs() -> None:
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Model lifecycle
# --------------------------------------------------------------------------- #
def ensure_model(name: str, force: bool = False, progress_cb: ProgressCb = None) -> Path:
    """Make sure ``name``'s weights are on disk under ``WHISPER_DIR``.

    Downloads via native whisper (loading on CPU once, with ``download_root``
    pointed at the shared dir) while a background thread polls the growing file
    so ``progress_cb`` can drive a UI. Returns the cached file path.
    """
    if name not in ALL_MODELS:
        raise ValueError(f"Model '{name}' is not supported. Use one of: {', '.join(ALL_MODELS)}")

    _, whisper = load_runtime()
    _ensure_runtime_dirs()

    with DOWNLOAD_LOCK:
        target_file = model_file_path(name)

        if force and target_file.exists():
            try:
                target_file.unlink()
            except OSError as exc:
                LOGGER.warning("%s Could not remove old model file for '%s': %s", LOG_PREFIX, name, exc)

        if not force and is_model_cached(name):
            _emit(progress_cb, "cached", name=name, file=str(target_file))
            return target_file

        _emit(progress_cb, "download_start", name=name)
        LOGGER.info("%s Downloading Whisper model '%s' to %s", LOG_PREFIX, name, WHISPER_DIR)

        stop_monitor = threading.Event()
        estimated_total = MODEL_SIZES.get(name, 500_000_000)

        def monitor() -> None:
            while not stop_monitor.is_set():
                size = 0
                try:
                    if target_file.exists():
                        size = target_file.stat().st_size
                except OSError:
                    pass
                _emit(progress_cb, "download_progress", name=name,
                      file=f"{name}.pt", downloaded=size, total=estimated_total)
                stop_monitor.wait(0.4)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

        try:
            whisper.load_model(name, device="cpu", download_root=str(WHISPER_DIR), in_memory=False)
        except Exception as exc:
            _emit(progress_cb, "download_error", name=name, error=str(exc))
            LOGGER.exception("%s Whisper model download failed for '%s'", LOG_PREFIX, name)
            raise
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=1)

        if not is_model_cached(name):
            _emit(progress_cb, "download_error", name=name, error="Downloaded file not found")
            raise RuntimeError(
                f"Model '{name}' should be available at {target_file}. "
                f"Check write permissions for {WHISPER_DIR}."
            )

        final_size = target_file.stat().st_size
        _emit(progress_cb, "download_done", name=name, file=f"{name}.pt", downloaded=final_size, total=final_size)
        LOGGER.info("%s Whisper model '%s' ready (%.0f MB)", LOG_PREFIX, name, final_size / (1024 * 1024))
        return target_file


def resolve_device(device: str = "auto") -> str:
    torch, _ = load_runtime()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    target = str(device)
    if target == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("%s CUDA requested but unavailable; using CPU.", LOG_PREFIX)
        return "cpu"
    return target


def load_model(
    name: str,
    device: str = "auto",
    fp16_pref: bool = True,
    progress_cb: ProgressCb = None,
):
    """Load ``name`` onto the resolved device and return ``(model, device, use_fp16)``.

    Uses the shared ``MODEL_CACHE`` keyed by ``(name, device, use_fp16)`` so the
    same model object is reused across nodes and calls. Falls back to CPU if a
    CUDA load fails (e.g. OOM).
    """
    torch, whisper = load_runtime()
    target_device = resolve_device(device)
    use_fp16 = target_device == "cuda" and bool(fp16_pref)

    cache_key = (name, target_device, use_fp16)
    if cache_key in MODEL_CACHE:
        _emit(progress_cb, "already_cached", name=name, device=target_device)
        return MODEL_CACHE[cache_key], target_device, use_fp16

    ensure_model(name, progress_cb=progress_cb)
    LOGGER.info("%s Loading Whisper '%s' on %s (%s)", LOG_PREFIX, name, target_device, "fp16" if use_fp16 else "fp32")
    _emit(progress_cb, "load_start", name=name, device=target_device)

    try:
        model = whisper.load_model(name, device=target_device, download_root=str(WHISPER_DIR), in_memory=False)
    except Exception as exc:
        if target_device != "cuda":
            raise
        LOGGER.warning("%s GPU load failed for '%s': %s. Falling back to CPU.", LOG_PREFIX, name, exc)
        _emit(progress_cb, "gpu_fallback", name=name, error=str(exc))
        target_device = "cpu"
        use_fp16 = False
        cache_key = (name, target_device, use_fp16)
        if cache_key in MODEL_CACHE:
            return MODEL_CACHE[cache_key], target_device, use_fp16
        model = whisper.load_model(name, device=target_device, download_root=str(WHISPER_DIR), in_memory=False)

    MODEL_CACHE[cache_key] = model
    _emit(progress_cb, "loaded", name=name, device=target_device)
    return model, target_device, use_fp16


# --------------------------------------------------------------------------- #
# Audio decoding
# --------------------------------------------------------------------------- #
def get_ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def decode_audio_file(filepath: str):
    """Decode any audio/video file to 16 kHz mono float32 numpy via ffmpeg.

    Falls back to ``whisper.load_audio`` if ffmpeg is unavailable. Used by the
    Super Prompt voice path (file uploads).
    """
    import numpy as np

    command = [
        get_ffmpeg_executable(),
        "-i", filepath,
        "-f", "f32le", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
        "-loglevel", "error", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True, timeout=300)
        return np.frombuffer(result.stdout, dtype=np.float32).copy()
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        LOGGER.warning("%s ffmpeg timed out decoding audio; falling back to whisper.load_audio.", LOG_PREFIX)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        LOGGER.warning("%s ffmpeg failed to decode audio: %s", LOG_PREFIX, stderr)

    _, whisper = load_runtime()
    try:
        return whisper.load_audio(filepath)
    except Exception as exc:
        raise RuntimeError(f"Cannot decode audio: {exc}") from exc


def _get_resampler(orig_freq: int, new_freq: int, device, quality: str):
    if quality not in ("fast", "balanced", "high"):
        quality = "high"
    key = (int(orig_freq), int(new_freq), device.type, quality)
    cached = _RESAMPLER_CACHE.get(key)
    if cached is not None:
        return cached

    import torch
    from torchaudio.transforms import Resample

    if quality == "fast":
        lowpass_filter_width, rolloff = 4, 0.95
    elif quality == "high":
        lowpass_filter_width, rolloff = 8, 0.99
    else:
        lowpass_filter_width, rolloff = 6, 0.99

    resampler = Resample(
        orig_freq=int(orig_freq),
        new_freq=int(new_freq),
        resampling_method="sinc_interp_hann",
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
        dtype=torch.float32,
    )
    if device.type != "cpu":
        resampler = resampler.to(device)
    _RESAMPLER_CACHE[key] = resampler
    return resampler


def comfy_audio_to_mono16k(
    audio: dict,
    normalize: bool = True,
    device: str = "auto",
    resample_quality: str = "high",
):
    """Convert a ComfyUI ``AUDIO`` dict to 16 kHz mono float32 numpy for whisper.

    Accepts ``{"waveform": tensor[B,C,T] | [C,T] | [T], "sample_rate": int}``.
    Mono-downmixes (mean), resamples to 16 kHz (torchaudio), optionally peak-
    normalizes, and returns a contiguous float32 ndarray. Ported from the old
    TS Whisper ``_prepare_audio`` so both nodes share identical preprocessing.
    """
    import numpy as np
    import torch

    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Invalid audio input: expected a ComfyUI AUDIO dict with waveform + sample_rate.")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if isinstance(waveform, list):
        waveform = waveform[0]
    if not isinstance(waveform, torch.Tensor):
        raise ValueError("Audio waveform is not a torch.Tensor.")

    wave = waveform.detach()
    if wave.ndim == 3:
        if wave.shape[0] > 1:
            LOGGER.warning("%s Audio batch > 1; using first item.", LOG_PREFIX)
        wave = wave[0]
    if wave.ndim == 2:
        # Normalize to channels-first then downmix to mono.
        channels_first = wave.shape[0] <= 8 and wave.shape[1] > wave.shape[0]
        if not channels_first:
            wave = wave.transpose(0, 1)
        wave = wave.mean(dim=0)
    elif wave.ndim != 1:
        raise ValueError(f"Unsupported waveform dims: {wave.ndim}")

    wave = wave.to(dtype=torch.float32)

    if device == "cuda" and torch.cuda.is_available():
        target_device = torch.device("cuda")
    elif device == "cpu":
        target_device = torch.device("cpu")
    else:
        target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if wave.device != target_device:
        wave = wave.to(target_device)

    if sample_rate != AUDIO_SAMPLE_RATE:
        try:
            wave = _get_resampler(sample_rate, AUDIO_SAMPLE_RATE, wave.device, resample_quality)(wave)
        except Exception as exc:
            LOGGER.warning("%s Resample failed on %s, retrying on CPU: %s", LOG_PREFIX, wave.device, exc)
            wave = wave.cpu()
            wave = _get_resampler(sample_rate, AUDIO_SAMPLE_RATE, wave.device, resample_quality)(wave)

    if normalize:
        peak = wave.abs().max().item() if wave.numel() else 0.0
        if peak > 0:
            wave = wave / peak
        wave = wave.clamp(-1.0, 1.0)

    return wave.cpu().numpy().astype(np.float32, copy=False)
