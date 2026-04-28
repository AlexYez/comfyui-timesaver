from __future__ import annotations

import asyncio
import importlib.util
import logging
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folder_paths
from aiohttp import web

try:
    import server
except Exception:
    server = None


LOGGER = logging.getLogger("comfyui_timesaver.ts_voice_recognition_service")
LOG_PREFIX = "[TS Voice Recognition Service]"

# User-configurable recognition settings. Keep the ComfyUI widget surface compact;
# tune recognition behavior here instead of adding workflow-breaking controls.
ACTIVE_MODEL = "base"
DEVICE = "auto"
GPU_PRECISION = "fp16"
SOURCE_LANGUAGE = "ru"
TRANSLATE_TO_ENGLISH = False
BEAM_SIZE = 5
TEMPERATURE = 0.0

# Whisper context prompt for prompt-dictation. Keep it concise: Whisper uses it as
# vocabulary/style context, not as an instruction-following system prompt.
INITIAL_PROMPT_ENABLED = True
INITIAL_PROMPT = """
Это диктовка промпта для генерации изображений, видео или музыки в ComfyUI.
Сохраняй смешанный русский и английский текст, не переводя названия стилей и технические термины.
Частые слова и фразы: prompt, negative prompt, cinematic, photorealistic, ultra detailed,
high detail, sharp focus, soft focus, depth of field, bokeh, volumetric lighting,
rim light, backlight, golden hour, moody lighting, color grading, composition,
close-up, medium shot, wide shot, establishing shot, macro shot, low angle,
high angle, top view, eye level, portrait, landscape, 35mm, 50mm, 85mm,
wide angle lens, telephoto lens, anamorphic lens, fisheye, dolly zoom,
pan, tilt, tracking shot, handheld camera, slow motion, time lapse,
Unreal Engine, Octane render, Redshift, Blender, 3D render, anime style,
watercolor, oil painting, concept art, storyboard, music prompt, ambient,
synthwave, orchestral, cinematic trailer, vocals, drums, bass, guitar, piano.
""".strip()
INITIAL_PROMPT_EXTRA = ""

# Audio preparation. These steps run before Whisper to reduce silence/noise work
# and keep the model focused on speech rather than microphone artifacts.
AUDIO_SAMPLE_RATE = 16000
AUDIO_TRIM_ENABLED = True
AUDIO_NORMALIZE_ENABLED = True
AUDIO_VAD_ENABLED = True
AUDIO_VAD_FRAME_MS = 30
AUDIO_VAD_HOP_MS = 10
AUDIO_VAD_RMS_THRESHOLD = 0.003
AUDIO_VAD_ADAPTIVE_MULTIPLIER = 2.2
AUDIO_VAD_MIN_SPEECH_SEC = 0.18
AUDIO_VAD_PADDING_SEC = 0.20
AUDIO_NORMALIZE_TARGET_PEAK = 0.92
AUDIO_NORMALIZE_MAX_GAIN_DB = 12.0

ALL_MODELS = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
)

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

MODELS_WITHOUT_TRANSLATE = {"turbo"}
ALLOWED_AUDIO_SUFFIXES = {".aac", ".aiff", ".flac", ".m4a", ".mp3", ".mp4", ".ogg", ".opus", ".wav", ".webm"}
EVENT_PREFIX = "ts_voice_recognition"
ROUTE_BASE = "/ts_voice_recognition"

WHISPER_DIR = Path(getattr(folder_paths, "models_dir", Path.cwd() / "models")) / "whisper"
_DOWNLOAD_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str, bool], Any] = {}


# ---------------------------------------------------------------------------
# ComfyUI server helpers
# ---------------------------------------------------------------------------


def _log_info(message: str) -> None:
    LOGGER.info("%s %s", LOG_PREFIX, message)


def _log_warning(message: str) -> None:
    LOGGER.warning("%s %s", LOG_PREFIX, message)


def _resolve_prompt_server():
    if server is None:
        _log_warning("PromptServer unavailable. HTTP routes disabled.")
        return None
    try:
        return server.PromptServer.instance
    except Exception as exc:
        _log_warning(f"PromptServer init failed. HTTP routes disabled: {exc}")
        return None


_PROMPT_SERVER = _resolve_prompt_server()


def _register_get(path: str):
    def decorator(func):
        if _PROMPT_SERVER is None:
            return func
        try:
            _PROMPT_SERVER.routes.get(path)(func)
        except Exception as exc:
            _log_warning(f"Failed to register GET route '{path}': {exc}")
        return func

    return decorator


def _register_post(path: str):
    def decorator(func):
        if _PROMPT_SERVER is None:
            return func
        try:
            _PROMPT_SERVER.routes.post(path)(func)
        except Exception as exc:
            _log_warning(f"Failed to register POST route '{path}': {exc}")
        return func

    return decorator


def _send_event(event: str, payload: dict[str, Any]) -> None:
    if _PROMPT_SERVER is None:
        return
    try:
        _PROMPT_SERVER.send_sync(f"{EVENT_PREFIX}.{event}", payload)
    except Exception as exc:
        LOGGER.debug("%s WebSocket event send failed: %s", LOG_PREFIX, exc)


def _audio_tmp_dir() -> Path:
    return Path(folder_paths.get_input_directory()) / "ts_voice_recognition_tmp"


def _ensure_runtime_dirs() -> None:
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    _audio_tmp_dir().mkdir(parents=True, exist_ok=True)


def _missing_runtime_packages() -> list[str]:
    missing = []
    if importlib.util.find_spec("torch") is None:
        missing.append("torch")
    if importlib.util.find_spec("whisper") is None:
        missing.append("openai-whisper")
    if importlib.util.find_spec("numpy") is None:
        missing.append("numpy")
    return missing


def _load_whisper_runtime():
    missing = _missing_runtime_packages()
    if missing:
        raise RuntimeError(
            "Missing dependencies for TS Voice Recognition Service: "
            f"{', '.join(missing)}. Install requirements.txt and restart ComfyUI."
        )

    try:
        import torch
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "TS Voice Recognition Service requires openai-whisper and torch. "
            "Install requirements.txt and restart ComfyUI."
        ) from exc

    return torch, whisper


def _configured_initial_prompt() -> str | None:
    if not INITIAL_PROMPT_ENABLED:
        return None
    parts = [str(INITIAL_PROMPT or "").strip(), str(INITIAL_PROMPT_EXTRA or "").strip()]
    prompt = "\n".join(part for part in parts if part)
    return prompt or None


def _get_ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


@dataclass(frozen=True)
class AudioPreprocessResult:
    """Prepared audio plus diagnostics returned to the browser for debugging."""

    audio: Any
    original_duration: float
    processed_duration: float
    speech_detected: bool
    speech_start: float
    speech_end: float
    trimmed: bool
    normalized: bool
    gain: float
    peak_before: float
    peak_after: float
    vad_threshold: float


class ProgressBroadcaster:
    """Throttle Whisper model download progress events for ComfyUI widgets."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.last_sent = 0.0
        self.last_bytes = 0
        self.last_time = time.time()

    def _send(self, event: str, data: dict[str, Any]) -> None:
        _send_event(event, {"model": self.model_name, **data})

    def status(self, text: str) -> None:
        self._send("status", {"text": text})

    def progress(self, filename: str, downloaded: int, total: int, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_sent < 0.2:
            return
        elapsed = now - self.last_time
        speed = (downloaded - self.last_bytes) / elapsed if elapsed > 0 else 0.0
        self.last_sent = now
        self.last_bytes = downloaded
        self.last_time = now
        self._send(
            "progress",
            {
                "file": filename,
                "downloaded": downloaded,
                "total": total,
                "speed": speed,
                "percent": round(100 * downloaded / total, 1) if total > 0 else 0,
            },
        )

    def done(self) -> None:
        self._send("done", {})

    def error(self, text: str) -> None:
        self._send("error", {"text": text})


# ---------------------------------------------------------------------------
# Whisper model lifecycle
# ---------------------------------------------------------------------------


def _model_file_path(name: str) -> Path:
    return WHISPER_DIR / f"{name}.pt"


def is_model_cached(name: str) -> bool:
    model_file = _model_file_path(name)
    try:
        return model_file.is_file() and model_file.stat().st_size > 10_000_000
    except OSError:
        return False


def ensure_model(name: str, force: bool = False) -> Path:
    if name not in ALL_MODELS:
        raise ValueError(f"Model '{name}' is not supported. Use one of: {', '.join(ALL_MODELS)}")

    _, whisper = _load_whisper_runtime()
    _ensure_runtime_dirs()

    with _DOWNLOAD_LOCK:
        progress = ProgressBroadcaster(name)
        target_file = _model_file_path(name)

        if force and target_file.exists():
            try:
                target_file.unlink()
            except OSError as exc:
                _log_warning(f"Could not remove old model file for '{name}': {exc}")

        if not force and is_model_cached(name):
            progress.status(f"{name} ready")
            progress.done()
            return target_file

        progress.status(f"Downloading {name}")
        _log_info(f"Downloading Whisper model '{name}' to {WHISPER_DIR}")

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
                progress.progress(f"{name}.pt", size, estimated_total)
                stop_monitor.wait(0.4)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

        try:
            whisper.load_model(
                name,
                device="cpu",
                download_root=str(WHISPER_DIR),
                in_memory=False,
            )
        except Exception as exc:
            progress.error(f"Download failed: {exc}")
            LOGGER.exception("%s Whisper model download failed for '%s'", LOG_PREFIX, name)
            raise
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=1)

        if not is_model_cached(name):
            progress.error("Downloaded file was not found")
            raise RuntimeError(
                f"Model '{name}' should be available at {_model_file_path(name)}. "
                f"Check write permissions for {WHISPER_DIR}."
            )

        final_size = target_file.stat().st_size
        progress.progress(f"{name}.pt", final_size, final_size, force=True)
        progress.status(f"{name} ready")
        progress.done()
        _log_info(f"Whisper model '{name}' ready ({final_size / (1024 * 1024):.0f} MB)")
        return target_file


def load_model(name: str, device: str = "auto"):
    torch, whisper = _load_whisper_runtime()

    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = str(device)
        if target_device == "cuda" and not torch.cuda.is_available():
            _log_warning("CUDA was requested but is not available. Falling back to CPU.")
            target_device = "cpu"

    use_fp16 = target_device == "cuda" and GPU_PRECISION == "fp16"
    cache_key = (name, target_device, use_fp16)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], target_device, use_fp16

    ensure_model(name)
    _log_info(f"Loading Whisper model '{name}' on {target_device} ({'fp16' if use_fp16 else 'fp32'})")

    try:
        model = whisper.load_model(
            name,
            device=target_device,
            download_root=str(WHISPER_DIR),
            in_memory=False,
        )
    except Exception as exc:
        if target_device != "cuda":
            raise
        _log_warning(f"GPU load failed for '{name}': {exc}. Falling back to CPU.")
        _send_event("status", {"model": name, "text": "GPU load failed; using CPU"})
        target_device = "cpu"
        use_fp16 = False
        cache_key = (name, target_device, use_fp16)
        model = whisper.load_model(
            name,
            device=target_device,
            download_root=str(WHISPER_DIR),
            in_memory=False,
        )

    _MODEL_CACHE[cache_key] = model
    return model, target_device, use_fp16


# ---------------------------------------------------------------------------
# Audio decoding and preprocessing
# ---------------------------------------------------------------------------


def _read_audio(filepath: str):
    import numpy as np

    command = [
        _get_ffmpeg_executable(),
        "-i",
        filepath,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-loglevel",
        "error",
        "-",
    ]

    try:
        result = subprocess.run(command, capture_output=True, check=True)
        return np.frombuffer(result.stdout, dtype=np.float32).copy()
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        _log_warning(f"ffmpeg failed to decode audio: {stderr}")

    _, whisper = _load_whisper_runtime()
    try:
        return whisper.load_audio(filepath)
    except Exception as exc:
        raise RuntimeError(f"Cannot decode audio: {exc}") from exc


def _as_float32_audio(audio: Any):
    import numpy as np

    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return array.copy()
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0, copy=True)


def _frame_rms(audio, sample_rate: int) -> tuple[Any, Any, int, int]:
    import numpy as np

    if audio.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.int64), 0, 0

    frame_size = max(1, int(sample_rate * AUDIO_VAD_FRAME_MS / 1000))
    hop_size = max(1, int(sample_rate * AUDIO_VAD_HOP_MS / 1000))
    starts = np.arange(0, audio.size, hop_size, dtype=np.int64)
    ends = np.minimum(starts + frame_size, audio.size)
    lengths = np.maximum(ends - starts, 1)

    squared = audio.astype(np.float64, copy=False) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared, dtype=np.float64)))
    energy = cumulative[ends] - cumulative[starts]
    rms = np.sqrt(energy / lengths).astype(np.float32)
    return rms, starts, frame_size, hop_size


def _adaptive_vad_threshold(rms) -> float:
    import numpy as np

    if rms.size == 0:
        return float(AUDIO_VAD_RMS_THRESHOLD)
    noise_floor = float(np.percentile(rms, 20))
    max_rms = float(np.max(rms))
    adaptive = noise_floor * float(AUDIO_VAD_ADAPTIVE_MULTIPLIER)
    if max_rms > 0:
        adaptive = min(adaptive, max_rms * 0.35)
    return max(float(AUDIO_VAD_RMS_THRESHOLD), adaptive)


def _detect_speech_bounds(audio, sample_rate: int) -> tuple[int, int, bool, float]:
    import numpy as np

    rms, starts, frame_size, hop_size = _frame_rms(audio, sample_rate)
    threshold = _adaptive_vad_threshold(rms)
    if rms.size == 0:
        return 0, 0, False, threshold

    speech_mask = rms >= threshold
    if not bool(np.any(speech_mask)):
        return 0, 0, False, threshold

    voiced_duration = float(np.count_nonzero(speech_mask) * hop_size) / float(sample_rate)
    if voiced_duration < float(AUDIO_VAD_MIN_SPEECH_SEC):
        return 0, 0, False, threshold

    speech_indices = np.flatnonzero(speech_mask)
    padding = max(0, int(sample_rate * AUDIO_VAD_PADDING_SEC))
    start = max(0, int(starts[int(speech_indices[0])]) - padding)
    end = min(audio.size, int(starts[int(speech_indices[-1])]) + frame_size + padding)
    return start, end, end > start, threshold


def _normalize_audio(audio):
    import numpy as np

    if audio.size == 0:
        return audio, False, 1.0, 0.0, 0.0

    peak_before = float(np.max(np.abs(audio)))
    if not AUDIO_NORMALIZE_ENABLED or peak_before <= 1e-6:
        return audio.astype(np.float32, copy=False), False, 1.0, peak_before, peak_before

    target_peak = min(0.99, max(0.05, float(AUDIO_NORMALIZE_TARGET_PEAK)))
    max_gain = 10.0 ** (float(AUDIO_NORMALIZE_MAX_GAIN_DB) / 20.0)
    requested_gain = target_peak / peak_before
    gain = min(requested_gain, max_gain) if requested_gain >= 1.0 else requested_gain
    normalized = np.clip(audio * gain, -1.0, 1.0).astype(np.float32, copy=False)
    peak_after = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    return normalized, abs(gain - 1.0) > 0.01, float(gain), peak_before, peak_after


def _preprocess_audio(audio: Any, sample_rate: int = AUDIO_SAMPLE_RATE) -> AudioPreprocessResult:
    import numpy as np

    original = _as_float32_audio(audio)
    original_duration = float(original.size) / float(sample_rate)
    peak_before = float(np.max(np.abs(original))) if original.size else 0.0

    if original.size == 0:
        return AudioPreprocessResult(
            audio=original,
            original_duration=0.0,
            processed_duration=0.0,
            speech_detected=False,
            speech_start=0.0,
            speech_end=0.0,
            trimmed=False,
            normalized=False,
            gain=1.0,
            peak_before=0.0,
            peak_after=0.0,
            vad_threshold=float(AUDIO_VAD_RMS_THRESHOLD),
        )

    start, end, speech_detected, vad_threshold = _detect_speech_bounds(original, sample_rate)
    if not speech_detected and AUDIO_VAD_ENABLED:
        empty = np.asarray([], dtype=np.float32)
        return AudioPreprocessResult(
            audio=empty,
            original_duration=original_duration,
            processed_duration=0.0,
            speech_detected=False,
            speech_start=0.0,
            speech_end=0.0,
            trimmed=True,
            normalized=False,
            gain=1.0,
            peak_before=peak_before,
            peak_after=0.0,
            vad_threshold=vad_threshold,
        )

    if speech_detected and AUDIO_TRIM_ENABLED:
        processed = original[start:end].copy()
        speech_start = float(start) / float(sample_rate)
        speech_end = float(end) / float(sample_rate)
        trimmed = start > 0 or end < original.size
    else:
        processed = original.copy()
        speech_start = 0.0
        speech_end = original_duration
        trimmed = False

    processed, normalized, gain, _trimmed_peak_before, peak_after = _normalize_audio(processed)
    return AudioPreprocessResult(
        audio=processed,
        original_duration=original_duration,
        processed_duration=float(processed.size) / float(sample_rate),
        speech_detected=bool(speech_detected),
        speech_start=speech_start,
        speech_end=speech_end,
        trimmed=trimmed,
        normalized=normalized,
        gain=gain,
        peak_before=peak_before,
        peak_after=peak_after,
        vad_threshold=vad_threshold,
    )


def _audio_metadata(audio: AudioPreprocessResult) -> dict[str, Any]:
    return {
        "duration": round(audio.original_duration, 2),
        "processed_duration": round(audio.processed_duration, 2),
        "speech_detected": audio.speech_detected,
        "speech_start": round(audio.speech_start, 2),
        "speech_end": round(audio.speech_end, 2),
        "audio_trimmed": audio.trimmed,
        "audio_normalized": audio.normalized,
        "audio_gain": round(audio.gain, 3),
        "audio_peak_before": round(audio.peak_before, 4),
        "audio_peak_after": round(audio.peak_after, 4),
        "vad_threshold": round(audio.vad_threshold, 5),
    }


# ---------------------------------------------------------------------------
# Recognition endpoints
# ---------------------------------------------------------------------------


def transcribe_audio(
    filepath: str,
    model_name: str,
    device: str,
    source_language: str | None,
    target_language: str,
    initial_prompt: str | None = None,
) -> dict[str, Any]:
    torch, _ = _load_whisper_runtime()
    task = "translate" if target_language == "en" else "transcribe"
    language = None if source_language in (None, "", "auto") else source_language

    _send_event("status", {"model": model_name, "text": "Preparing audio"})
    audio_info = _preprocess_audio(_read_audio(filepath), AUDIO_SAMPLE_RATE)
    metadata = _audio_metadata(audio_info)
    if not audio_info.speech_detected or len(audio_info.audio) == 0:
        _send_event("status", {"model": model_name, "text": "No speech detected"})
        return {
            "text": "",
            "language": language or "?",
            "task": task,
            **metadata,
        }

    _send_event("status", {"model": model_name, "text": "Recognizing speech"})
    model, _, use_fp16 = load_model(model_name, device)

    transcribe_kwargs = {
        "task": task,
        "language": language,
        "fp16": use_fp16,
        "temperature": TEMPERATURE,
        "compression_ratio_threshold": 1.35,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
    }
    if BEAM_SIZE > 1:
        transcribe_kwargs["beam_size"] = BEAM_SIZE
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    with torch.inference_mode():
        result = model.transcribe(audio_info.audio, **transcribe_kwargs)

    text = (result.get("text") or "").strip()
    detected = result.get("language", language or "?")
    return {
        "text": text,
        "language": detected,
        "task": task,
        **metadata,
    }


def _safe_audio_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in ALLOWED_AUDIO_SUFFIXES and len(suffix) <= 10:
        return suffix
    return ".webm"


async def _read_audio_part(request: web.Request):
    reader = await request.multipart()
    while True:
        part = await reader.next()
        if part is None:
            return None
        if part.name == "audio":
            return part


@_register_post(f"{ROUTE_BASE}/transcribe")
async def transcribe_endpoint(request: web.Request) -> web.StreamResponse:
    audio_part = await _read_audio_part(request)
    if audio_part is None:
        return web.json_response({"error": "Missing audio field."}, status=400)

    _ensure_runtime_dirs()
    filename = f"{uuid.uuid4().hex}{_safe_audio_suffix(audio_part.filename)}"
    filepath = _audio_tmp_dir() / filename

    try:
        with filepath.open("wb") as handle:
            while True:
                chunk = await audio_part.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

        model_name = ACTIVE_MODEL
        translate_to_english = bool(TRANSLATE_TO_ENGLISH)
        if translate_to_english and model_name in MODELS_WITHOUT_TRANSLATE:
            _log_warning(f"Model '{model_name}' does not support translation. Using transcription.")
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
        LOGGER.exception("%s Voice transcription failed", LOG_PREFIX)
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            pass


@_register_post(f"{ROUTE_BASE}/preload")
async def preload_endpoint(request: web.Request) -> web.StreamResponse:
    try:
        data = await request.json()
    except Exception:
        data = {}

    name = data.get("model", ACTIVE_MODEL)
    force = bool(data.get("force", False))

    try:
        await asyncio.to_thread(ensure_model, name, force)
        await asyncio.to_thread(load_model, name, DEVICE)
        return web.json_response({"ok": True})
    except Exception as exc:
        LOGGER.exception("%s Whisper preload failed", LOG_PREFIX)
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


@_register_get(f"{ROUTE_BASE}/status")
async def status_endpoint(request: web.Request) -> web.StreamResponse:
    name = ACTIVE_MODEL
    return web.json_response(
        {
            name: {
                "downloaded": is_model_cached(name),
                "loaded": any(cache_key[0] == name for cache_key in _MODEL_CACHE.keys()),
                "path": str(_model_file_path(name)),
                "missing_dependencies": _missing_runtime_packages(),
                "translate_to_english": bool(TRANSLATE_TO_ENGLISH),
            }
        }
    )
