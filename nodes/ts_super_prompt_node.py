from __future__ import annotations

import asyncio
import gc
import importlib.util
import inspect
import json
import logging
import re
import subprocess
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folder_paths
from aiohttp import web
from comfy_api.latest import IO

try:
    import server
except Exception:
    server = None

try:
    from .ts_qwen3_vl_v3_node import TS_Qwen3_VL_V3
except Exception:
    from ts_qwen3_vl_v3_node import TS_Qwen3_VL_V3


LOGGER = logging.getLogger("comfyui_timesaver.ts_super_prompt")
LOG_PREFIX = "[TS Super Prompt]"
AI_ROUTE_BASE = "/ts_super_prompt"
AI_EVENT_PREFIX = "ts_super_prompt"
VOICE_ROUTE_BASE = "/ts_voice_recognition"
VOICE_EVENT_PREFIX = "ts_voice_recognition"
VOICE_LOG_PREFIX = "[TS Super Prompt Voice]"

# User-configurable settings for the compact TS Super Prompt UI.
# Change these values here instead of exposing extra widgets in ComfyUI.
SUPER_PROMPT_MODEL_HUIHUI_2B = "huihui-ai/Huihui-Qwen3.5-2B-abliterated"
SUPER_PROMPT_MODEL_QWEN_2B = "Qwen/Qwen3.5-2B"
SUPER_PROMPT_MODEL_OPTIONS = (
    SUPER_PROMPT_MODEL_HUIHUI_2B,
    SUPER_PROMPT_MODEL_QWEN_2B,
)
# To switch back to stock Qwen, set DEFAULT_MODEL_ID = SUPER_PROMPT_MODEL_QWEN_2B.
DEFAULT_MODEL_ID = SUPER_PROMPT_MODEL_HUIHUI_2B
DEFAULT_PRESET = "Prompts enhance"
CUSTOM_PRESET = "Your instruction"
SUPER_PROMPT_TARGET = "auto"
SUPER_PROMPT_ENHANCE_ON_EXECUTE = False
SUPER_PROMPT_SEED = 42
SUPER_PROMPT_MAX_NEW_TOKENS = 512
SUPER_PROMPT_PRECISION = "auto"
SUPER_PROMPT_ATTENTION_MODE = "auto"
SUPER_PROMPT_OFFLINE_MODE = False
SUPER_PROMPT_UNLOAD_AFTER_GENERATION = False
SUPER_PROMPT_MAX_IMAGE_SIZE = 1024
SUPER_PROMPT_HF_TOKEN = ""
SUPER_PROMPT_HF_ENDPOINT = "huggingface.co, hf-mirror.com"
SUPER_PROMPT_CUSTOM_SYSTEM_PROMPT = ""
SUPER_PROMPT_DOWNLOAD_SIZE_ESTIMATES = {
    SUPER_PROMPT_MODEL_HUIHUI_2B: 4_500_000_000,
    SUPER_PROMPT_MODEL_QWEN_2B: 4_500_000_000,
}

# User-configurable recognition settings. Keep the ComfyUI widget surface compact;
# tune recognition behavior here instead of adding workflow-breaking controls.
VOICE_MODEL_BASE = "base"
VOICE_MODEL_HIGH_QUALITY = "turbo"
ACTIVE_MODEL = VOICE_MODEL_BASE
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
AUDIO_VAD_PADDING_SEC = 0.30
AUDIO_EDGE_FADE_MS = 12
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

MODEL_FILE_NAMES = {
    "turbo": "large-v3-turbo.pt",
}
MODELS_WITHOUT_TRANSLATE = {"turbo"}
ALLOWED_AUDIO_SUFFIXES = {".aac", ".aiff", ".flac", ".m4a", ".mp3", ".mp4", ".ogg", ".opus", ".wav", ".webm"}
WHISPER_DIR = Path(getattr(folder_paths, "models_dir", Path.cwd() / "models")) / "whisper"
_DOWNLOAD_LOCK = threading.Lock()
_VOICE_MODEL_CACHE: dict[tuple[str, str, bool], Any] = {}

PROMPT_TARGETS = ("auto", "image", "video", "music")

_MODEL_LOCK = threading.Lock()
_QWEN_ENGINE: TS_Qwen3_VL_V3 | None = None


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


def _send_ai_event(event: str, payload: dict[str, Any]) -> None:
    if _PROMPT_SERVER is None:
        return
    try:
        _PROMPT_SERVER.send_sync(f"{AI_EVENT_PREFIX}.{event}", payload)
    except Exception as exc:
        LOGGER.debug("%s WebSocket event send failed: %s", LOG_PREFIX, exc)


def _send_voice_event(event: str, payload: dict[str, Any]) -> None:
    if _PROMPT_SERVER is None:
        return
    try:
        _PROMPT_SERVER.send_sync(f"{VOICE_EVENT_PREFIX}.{event}", payload)
    except Exception as exc:
        LOGGER.debug("%s WebSocket event send failed: %s", VOICE_LOG_PREFIX, exc)


def _send_progress(operation_id: str | None, text: str, percent: float | None = None) -> None:
    payload: dict[str, Any] = {"text": text}
    if operation_id:
        payload["operation_id"] = operation_id
    if percent is not None:
        payload["percent"] = max(0.0, min(100.0, float(percent)))
    _send_ai_event("progress", payload)


def _send_done(operation_id: str | None, text: str = "Ready") -> None:
    payload: dict[str, Any] = {"text": text, "percent": 100.0}
    if operation_id:
        payload["operation_id"] = operation_id
    _send_ai_event("done", payload)


def _send_error(operation_id: str | None, text: str) -> None:
    payload: dict[str, Any] = {"text": text}
    if operation_id:
        payload["operation_id"] = operation_id
    _send_ai_event("error", payload)


def _format_bytes(size_bytes: int) -> str:
    value = float(max(0, int(size_bytes)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    try:
        iterator = path.rglob("*")
        for item in iterator:
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


# ---------------------------------------------------------------------------
# Voice recognition backend
# ---------------------------------------------------------------------------

def _voice_log_info(message: str) -> None:
    LOGGER.info("%s %s", VOICE_LOG_PREFIX, message)


def _voice_log_warning(message: str) -> None:
    LOGGER.warning("%s %s", VOICE_LOG_PREFIX, message)


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
            "Missing dependencies for TS Super Prompt voice recognition: "
            f"{', '.join(missing)}. Install requirements.txt and restart ComfyUI."
        )

    try:
        import torch
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "TS Super Prompt voice recognition requires openai-whisper and torch. "
            "Install requirements.txt and restart ComfyUI."
        ) from exc

    return torch, whisper


def _configured_initial_prompt() -> str | None:
    if not INITIAL_PROMPT_ENABLED:
        return None
    parts = [str(INITIAL_PROMPT or "").strip(), str(INITIAL_PROMPT_EXTRA or "").strip()]
    prompt = "\n".join(part for part in parts if part)
    return prompt or None


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _resolve_voice_model(high_quality: Any = False, requested_model: str | None = None) -> str:
    _ = requested_model
    if _parse_bool(high_quality):
        return VOICE_MODEL_HIGH_QUALITY
    return VOICE_MODEL_BASE


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
        _send_voice_event(event, {"model": self.model_name, **data})

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

    def done(self, text: str = "Voice model file ready") -> None:
        self._send("status", {"text": text, "percent": 100.0})

    def error(self, text: str) -> None:
        self._send("error", {"text": text})


class QwenDownloadProgressMonitor:
    """Poll local HuggingFace files while snapshot_download runs."""

    def __init__(
        self,
        operation_id: str | None,
        model_id: str,
        local_dir: Path,
        total_bytes: int,
        start_percent: float = 20.0,
        end_percent: float = 44.0,
        enabled: bool = True,
    ):
        self.operation_id = operation_id
        self.model_id = model_id
        self.local_dir = local_dir
        self.total_bytes = max(1, int(total_bytes))
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.enabled = bool(enabled)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_size = -1

    def start(self) -> None:
        if not self.enabled:
            return
        _send_progress(self.operation_id, f"Connecting to HuggingFace for {self.model_id}", self.start_percent)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, success: bool) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if success:
            size = _directory_size(self.local_dir)
            _send_progress(
                self.operation_id,
                f"Qwen model files ready ({_format_bytes(size)})",
                self.end_percent,
            )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._emit_progress()
            self._stop.wait(0.7)

    def _emit_progress(self) -> None:
        size = _directory_size(self.local_dir)
        if size == self._last_size and size > 0:
            return
        self._last_size = size
        ratio = max(0.0, min(1.0, size / float(self.total_bytes)))
        percent = self.start_percent + (self.end_percent - self.start_percent) * ratio
        if size <= 0:
            text = f"Downloading Qwen model {self.model_id}"
        else:
            text = f"Downloading Qwen model {self.model_id}: {_format_bytes(size)}"
        _send_progress(self.operation_id, text, percent)


# ---------------------------------------------------------------------------
# Whisper model lifecycle
# ---------------------------------------------------------------------------


def _model_file_path(name: str) -> Path:
    return WHISPER_DIR / MODEL_FILE_NAMES.get(name, f"{name}.pt")


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
                _voice_log_warning(f"Could not remove old model file for '{name}': {exc}")

        if not force and is_model_cached(name):
            progress.done(f"{name} file ready")
            return target_file

        progress.status(f"Downloading {name}")
        _voice_log_info(f"Downloading Whisper model '{name}' to {WHISPER_DIR}")

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
            LOGGER.exception("%s Whisper model download failed for '%s'", VOICE_LOG_PREFIX, name)
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
        progress.done(f"{name} file ready")
        _voice_log_info(f"Whisper model '{name}' ready ({final_size / (1024 * 1024):.0f} MB)")
        return target_file


def load_model(name: str, device: str = "auto", progress_start: float = 82.0, progress_end: float = 96.0):
    torch, whisper = _load_whisper_runtime()

    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = str(device)
        if target_device == "cuda" and not torch.cuda.is_available():
            _voice_log_warning("CUDA was requested but is not available. Falling back to CPU.")
            target_device = "cpu"

    use_fp16 = target_device == "cuda" and GPU_PRECISION == "fp16"
    cache_key = (name, target_device, use_fp16)
    if cache_key in _VOICE_MODEL_CACHE:
        _send_voice_status(name, "Voice model already in memory", progress_end)
        return _VOICE_MODEL_CACHE[cache_key], target_device, use_fp16

    ensure_model(name)
    _voice_log_info(f"Loading Whisper model '{name}' on {target_device} ({'fp16' if use_fp16 else 'fp32'})")
    _send_voice_status(name, f"Loading {name} into memory on {target_device}", progress_start)

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
        _voice_log_warning(f"GPU load failed for '{name}': {exc}. Falling back to CPU.")
        _send_voice_status(name, "GPU load failed; using CPU", min(progress_end, progress_start + 2.0))
        target_device = "cpu"
        use_fp16 = False
        cache_key = (name, target_device, use_fp16)
        _send_voice_status(name, f"Loading {name} into memory on CPU", min(progress_end, progress_start + 4.0))
        model = whisper.load_model(
            name,
            device=target_device,
            download_root=str(WHISPER_DIR),
            in_memory=False,
        )

    _VOICE_MODEL_CACHE[cache_key] = model
    _send_voice_status(name, "Voice model loaded into memory", progress_end)
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
        _voice_log_warning(f"ffmpeg failed to decode audio: {stderr}")

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


def _apply_edge_fade(audio, sample_rate: int):
    import numpy as np

    fade_samples = max(0, int(sample_rate * float(AUDIO_EDGE_FADE_MS) / 1000.0))
    if fade_samples <= 1 or audio.size <= fade_samples * 2:
        return audio.astype(np.float32, copy=False)

    processed = audio.astype(np.float32, copy=True)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    processed[:fade_samples] *= fade_in
    processed[-fade_samples:] *= fade_in[::-1]
    return processed


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

    processed = _apply_edge_fade(processed, sample_rate)
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


_DUPLICATE_TRANSCRIPTION_WORDS = {
    "с",
    "со",
    "в",
    "во",
    "к",
    "ко",
    "у",
    "о",
    "об",
    "от",
    "до",
    "из",
    "за",
    "на",
    "по",
    "под",
    "над",
    "при",
    "для",
    "через",
    "без",
    "with",
    "in",
    "on",
    "at",
    "to",
    "from",
    "of",
    "for",
    "by",
}


def _clean_transcription_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""

    for word in sorted(_DUPLICATE_TRANSCRIPTION_WORDS, key=len, reverse=True):
        pattern = re.compile(rf"(?iu)(\b{re.escape(word)}\b)(?:\s+)(\b{re.escape(word)}\b)")
        while True:
            collapsed = pattern.sub(r"\1", cleaned)
            if collapsed == cleaned:
                break
            cleaned = collapsed
    return cleaned


def _send_voice_status(model_name: str, text: str, percent: float | None = None) -> None:
    payload: dict[str, Any] = {"model": model_name, "text": text}
    if percent is not None:
        payload["percent"] = max(0.0, min(100.0, float(percent)))
    _send_voice_event("status", payload)


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

    _send_voice_status(model_name, "Preparing audio", 10.0)
    audio_info = _preprocess_audio(_read_audio(filepath), AUDIO_SAMPLE_RATE)
    metadata = _audio_metadata(audio_info)
    if not audio_info.speech_detected or len(audio_info.audio) == 0:
        _send_voice_status(model_name, "No speech detected", 100.0)
        return {
            "text": "",
            "language": language or "?",
            "task": task,
            **metadata,
        }

    _send_voice_status(model_name, "Loading voice model", 40.0)
    model, _, use_fp16 = load_model(model_name, device, progress_start=42.0, progress_end=64.0)
    _send_voice_status(model_name, "Recognizing speech", 68.0)

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

    _send_voice_status(model_name, "Finalizing speech text", 92.0)
    text = _clean_transcription_text(result.get("text") or "")
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
            while True:
                chunk = await part.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            upload = {"filename": part.filename, "data": b"".join(chunks)}
            continue
        if part.name:
            fields[part.name] = await part.text()


@_register_post(f"{VOICE_ROUTE_BASE}/transcribe")
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
            _voice_log_warning(f"Model '{model_name}' does not support translation. Using transcription.")
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


@_register_post(f"{VOICE_ROUTE_BASE}/preload")
async def preload_endpoint(request: web.Request) -> web.StreamResponse:
    try:
        data = await request.json()
    except Exception:
        data = {}

    name = _resolve_voice_model(data.get("high_quality"), data.get("model", ACTIVE_MODEL))
    force = bool(data.get("force", False))

    try:
        _send_voice_status(name, "Preparing voice model", 5.0)
        await asyncio.to_thread(ensure_model, name, force)
        _send_voice_status(name, "Loading voice model into memory", 78.0)
        await asyncio.to_thread(load_model, name, DEVICE)
        _send_voice_event("done", {"model": name, "text": "Voice model ready", "percent": 100.0})
        return web.json_response({"ok": True})
    except Exception as exc:
        LOGGER.exception("%s Whisper preload failed", VOICE_LOG_PREFIX)
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


@_register_get(f"{VOICE_ROUTE_BASE}/status")
async def status_endpoint(request: web.Request) -> web.StreamResponse:
    name = _resolve_voice_model(request.query.get("high_quality"), request.query.get("model", ACTIVE_MODEL))
    return web.json_response(
        {
            name: {
                "downloaded": is_model_cached(name),
                "loaded": any(cache_key[0] == name for cache_key in _VOICE_MODEL_CACHE.keys()),
                "path": str(_model_file_path(name)),
                "missing_dependencies": _missing_runtime_packages(),
                "translate_to_english": bool(TRANSLATE_TO_ENGLISH),
            }
        }
    )


def _qwen_model_dir(model_id: str) -> Path:
    return Path(getattr(folder_paths, "models_dir", Path.cwd() / "models")) / "LLM" / str(model_id).split("/")[-1]


def _qwen_download_estimate(model_id: str) -> int:
    explicit = SUPER_PROMPT_DOWNLOAD_SIZE_ESTIMATES.get(model_id)
    if explicit:
        return int(explicit)
    try:
        size_b = float(_get_qwen_engine()._model_size_b(model_id))
    except Exception:
        size_b = 2.0
    return int(max(1.0, size_b) * 2_250_000_000)


def _is_qwen_model_available(engine: TS_Qwen3_VL_V3, model_id: str) -> bool:
    checker = getattr(engine, "_check_model_integrity", None)
    if not callable(checker):
        return False
    try:
        return bool(checker(str(_qwen_model_dir(model_id))))
    except Exception:
        return False


def _get_qwen_engine() -> TS_Qwen3_VL_V3:
    global _QWEN_ENGINE
    if _QWEN_ENGINE is None:
        _QWEN_ENGINE = TS_Qwen3_VL_V3()
    return _QWEN_ENGINE


def _load_presets() -> tuple[dict[str, Any], list[str]]:
    presets, keys = TS_Qwen3_VL_V3._load_presets()
    if not isinstance(presets, dict):
        return {}, []
    return presets, [key for key in keys if isinstance(key, str) and key]


def _preset_options() -> list[str]:
    _presets, keys = _load_presets()
    options = list(keys)
    if CUSTOM_PRESET not in options:
        options.append(CUSTOM_PRESET)
    return options or [CUSTOM_PRESET]


def _default_preset(options: list[str]) -> str:
    if DEFAULT_PRESET in options:
        return DEFAULT_PRESET
    return options[0] if options else CUSTOM_PRESET


def _resolve_preset(system_preset: str, custom_system_prompt: str | None) -> tuple[str, dict[str, Any]]:
    presets, _keys = _load_presets()
    preset_name = str(system_preset or "").strip()

    if preset_name == CUSTOM_PRESET:
        prompt = str(custom_system_prompt or "").strip()
        if prompt:
            return prompt, {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05}

    preset_data = presets.get(preset_name)
    if not isinstance(preset_data, dict):
        preset_data = presets.get(DEFAULT_PRESET)
    if not isinstance(preset_data, dict) and presets:
        first_key = next(iter(presets.keys()))
        preset_data = presets.get(first_key)

    if isinstance(preset_data, dict):
        system_prompt = str(preset_data.get("system_prompt") or "").strip()
        gen_params = preset_data.get("gen_params") or {}
        if not isinstance(gen_params, dict):
            gen_params = {}
        return system_prompt, dict(gen_params)

    return (
        "You are a senior prompt engineer. Translate the user's idea to English if needed and "
        "return only one polished generation prompt with no commentary.",
        {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05},
    )


def _target_instruction(prompt_target: str, has_image: bool) -> str:
    target = str(prompt_target or "auto").strip().lower()
    if target not in PROMPT_TARGETS:
        target = "auto"

    if target == "image":
        return (
            "Target output: image generation prompt. Create one vivid English paragraph focused on "
            "subject, composition, materials, environment, light, lens/camera feel, color palette, "
            "and style. Preserve the user's intent and do not add lists or quality-tag spam."
        )
    if target == "video":
        return (
            "Target output: video generation prompt. Create one cinematic English prompt with a clear "
            "camera move, subject action, motion physics, atmosphere, temporal flow, and visual continuity. "
            "If an image is provided, use it as the visual reference."
        )
    if target == "music":
        return (
            "Target output: music generation prompt. Create one English prompt describing genre, mood, "
            "tempo, rhythm, instrumentation, arrangement, dynamics, production style, and emotional arc. "
            "Do not describe non-audio visuals unless they directly inform the music."
        )

    if has_image:
        return (
            "Target output: infer whether image or video generation is more appropriate from the user's "
            "idea and the visual input, then return one polished English generation prompt."
        )
    return (
        "Target output: infer whether the user needs an image, video, or music generation prompt, then "
        "return one polished English prompt for that medium."
    )


def _build_messages(system_prompt: str, text: str, prompt_target: str, image: Any, max_image_size: int) -> list[dict[str, Any]]:
    engine = _get_qwen_engine()
    user_content: list[dict[str, Any]] = []

    user_text = (
        f"{_target_instruction(prompt_target, image is not None)}\n\n"
        "Hard rules:\n"
        "- Translate the source idea to English when needed.\n"
        "- Keep the user's core meaning, named subjects, and constraints.\n"
        "- Return only the final prompt, with no preface, no analysis, and no markdown.\n"
        "- Do not use thinking mode. Do not output chain-of-thought or hidden reasoning.\n\n"
        f"Source idea:\n{text or ''}"
    )

    system_text = (
        f"{system_prompt.strip()}\n\n"
        "Runtime mode: non-thinking. Produce the answer directly and never include a <think> block."
    )

    if image is not None:
        for pil_image in engine._tensor_to_pil_list(image):
            user_content.append(
                {
                    "type": "image",
                    "image": engine._resize_and_crop_image(pil_image, int(max_image_size)),
                }
            )

    user_content.append({"type": "text", "text": user_text})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {"role": "user", "content": user_content},
    ]


def _messages_have_visuals(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"image", "video"}:
                return True
    return False


def _flatten_text_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = [
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            text = "\n\n".join(part for part in text_parts if part)
        else:
            text = str(content or "")
        flattened.append({"role": str(message.get("role") or "user"), "content": text})
    return flattened


def _chat_template_functions(engine: TS_Qwen3_VL_V3, processor) -> list[Any]:
    template_fns: list[Any] = []
    processor_template = getattr(processor, "apply_chat_template", None)
    if processor_template is not None:
        template_fns.append(processor_template)

    get_tokenizer = getattr(engine, "_get_tokenizer_from_processor", None)
    tokenizer = get_tokenizer(processor) if callable(get_tokenizer) else getattr(processor, "tokenizer", processor)
    tokenizer_template = getattr(tokenizer, "apply_chat_template", None)
    if tokenizer_template is not None and tokenizer_template not in template_fns:
        template_fns.append(tokenizer_template)

    if not template_fns:
        raise RuntimeError("Loaded processor/tokenizer does not provide apply_chat_template.")
    return template_fns


def _template_accepts_kwargs(template_fn, kwargs: dict[str, Any]) -> bool:
    try:
        signature = inspect.signature(template_fn)
    except Exception:
        return True
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return True
    return all(key in parameters for key in kwargs.keys())


def _apply_chat_template_no_thinking(engine: TS_Qwen3_VL_V3, processor, messages: list[dict[str, Any]]):
    base_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }

    thinking_variants = (
        {"enable_thinking": False},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {},
    )
    message_variants = [messages]
    if not _messages_have_visuals(messages):
        flattened_messages = _flatten_text_messages(messages)
        if flattened_messages != messages:
            message_variants.append(flattened_messages)

    last_error: Exception | None = None
    for template_fn in _chat_template_functions(engine, processor):
        for candidate_messages in message_variants:
            for thinking_kwargs in thinking_variants:
                kwargs = {**base_kwargs, **thinking_kwargs}
                if not _template_accepts_kwargs(template_fn, kwargs):
                    continue
                try:
                    return template_fn(candidate_messages, **kwargs)
                except TypeError as exc:
                    last_error = exc
                    continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Loaded chat template rejected all supported argument variants.")


def _filter_generation_params(model, params: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.generate)
    except Exception:
        return params
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return params

    allowed = set(signature.parameters.keys())
    return {key: value for key, value in params.items() if key in allowed}


_GENERATION_PARAM_ALIASES = {
    "max_tokens": "max_new_tokens",
    "max_completion_tokens": "max_new_tokens",
}
_UNSUPPORTED_GENERATION_PARAMS = {
    "frequency_penalty",
    "n",
    "presence_penalty",
    "response_format",
    "stop",
    "stream",
}


def _normalize_generation_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    for alias, target in _GENERATION_PARAM_ALIASES.items():
        value = normalized.pop(alias, None)
        if value is not None and target not in normalized:
            normalized[target] = value
    for key in _UNSUPPORTED_GENERATION_PARAMS:
        normalized.pop(key, None)
    return normalized


def _unused_model_kwargs_from_error(exc: ValueError) -> list[str]:
    match = re.search(r"not used by the model:\s*(\[[^\]]+\])", str(exc))
    if not match:
        return []
    return re.findall(r"'([^']+)'", match.group(1))


def _generate_with_filtered_kwargs(model, inputs: dict[str, Any], gen_params: dict[str, Any]):
    current_params = dict(gen_params)
    for _attempt in range(4):
        try:
            return model.generate(**inputs, **current_params)
        except ValueError as exc:
            unused_keys = set(_unused_model_kwargs_from_error(exc))
            if not unused_keys:
                raise
            next_params = {key: value for key, value in current_params.items() if key not in unused_keys}
            if len(next_params) == len(current_params):
                raise
            _log_warning(f"Dropping unsupported Qwen generation params: {', '.join(sorted(unused_keys))}")
            current_params = next_params
    return model.generate(**inputs, **current_params)


def _clean_model_output(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    cleaned = re.sub(r"^\s*```(?:text|markdown|prompt)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
    cleaned = re.sub(
        r"^\s*(?:final\s+prompt|prompt|english\s+prompt|enhanced\s+prompt|result)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _generate_with_qwen(
    text: str,
    system_preset: str,
    operation_id: str | None = None,
    image: Any = None,
) -> str:
    if not str(text or "").strip() and image is None:
        return ""

    lock_acquired = _MODEL_LOCK.acquire(blocking=False)
    if not lock_acquired:
        _send_progress(operation_id, "Waiting for Qwen", 2.0)
        _MODEL_LOCK.acquire()

    try:
        _send_progress(operation_id, "Preparing prompt", 5.0)
        engine = _get_qwen_engine()
        system_prompt, gen_params = _resolve_preset(system_preset, SUPER_PROMPT_CUSTOM_SYSTEM_PROMPT)
        resolved_precision = engine._resolve_precision(SUPER_PROMPT_PRECISION, DEFAULT_MODEL_ID)
        resolved_attention = engine._resolve_attention(SUPER_PROMPT_ATTENTION_MODE, resolved_precision)
        estimated_vram = engine._estimate_vram_usage(DEFAULT_MODEL_ID, resolved_precision)

        _log_info(
            f"model={DEFAULT_MODEL_ID} precision={resolved_precision} "
            f"attention={resolved_attention} thinking=disabled"
        )
        _send_progress(operation_id, "Checking memory", 12.0)
        engine._ensure_memory_available(estimated_vram)

        _send_progress(operation_id, "Checking Qwen model files", 16.0)
        qwen_model_available = _is_qwen_model_available(engine, DEFAULT_MODEL_ID)
        if qwen_model_available:
            _send_progress(operation_id, "Qwen model found locally", 22.0)
        elif SUPER_PROMPT_OFFLINE_MODE:
            _send_progress(operation_id, "Using offline Qwen model files", 22.0)
        else:
            _send_progress(operation_id, "Qwen model download starting", 20.0)

        if not qwen_model_available and not bool(SUPER_PROMPT_OFFLINE_MODE):
            qwen_monitor = QwenDownloadProgressMonitor(
                operation_id=operation_id,
                model_id=DEFAULT_MODEL_ID,
                local_dir=_qwen_model_dir(DEFAULT_MODEL_ID),
                total_bytes=_qwen_download_estimate(DEFAULT_MODEL_ID),
            )
            qwen_monitor.start()
            download_success = False
            try:
                engine._ensure_model_available(
                    DEFAULT_MODEL_ID,
                    bool(SUPER_PROMPT_OFFLINE_MODE),
                    str(SUPER_PROMPT_HF_TOKEN or ""),
                    str(SUPER_PROMPT_HF_ENDPOINT or ""),
                )
                download_success = True
            finally:
                qwen_monitor.stop(download_success)

        _send_progress(operation_id, "Loading Qwen model into memory", 46.0)
        model, processor = engine._load_model(
            DEFAULT_MODEL_ID,
            resolved_precision,
            resolved_attention,
            bool(SUPER_PROMPT_OFFLINE_MODE),
            str(SUPER_PROMPT_HF_TOKEN or ""),
            str(SUPER_PROMPT_HF_ENDPOINT or ""),
        )
        _send_progress(operation_id, "Qwen model loaded", 50.0)

        target_device = engine._get_device()
        moved_to_gpu = False
        if target_device.type == "cuda" and not engine._model_has_cuda_device(model):
            try:
                _send_progress(operation_id, "Moving Qwen to GPU", 54.0)
                engine._ensure_memory_available(estimated_vram, force_unload=True)
                model.to(target_device)
                moved_to_gpu = True
            except RuntimeError as exc:
                if engine._is_oom_error(exc):
                    try:
                        model.to("cpu")
                    except Exception:
                        pass
                    engine._prepare_memory(force=True)
                    raise RuntimeError("Out of memory during Qwen GPU transfer.") from exc
                raise

        try:
            if image is not None and not engine._supports_multimodal_inputs(processor):
                raise RuntimeError(
                    "Loaded processor/tokenizer does not support image input. "
                    "Use a Qwen vision-language model or disconnect image."
                )

            _send_progress(operation_id, "Preparing Qwen input", 62.0)
            messages = _build_messages(
                system_prompt,
                text,
                SUPER_PROMPT_TARGET,
                image,
                int(SUPER_PROMPT_MAX_IMAGE_SIZE),
            )
            inputs = _apply_chat_template_no_thinking(engine, processor, messages)
            input_device = engine._select_input_device(model)
            inputs = engine._move_inputs_to_device(inputs, input_device)
            engine._log_processing_device("super_prompt_inputs", input_device, model, inputs)

            gen_params = dict(gen_params)
            gen_params.setdefault("temperature", 0.7 if image is not None else 1.0)
            gen_params.setdefault("top_p", 0.8 if image is not None else 1.0)
            gen_params.setdefault("top_k", 20)
            gen_params.setdefault("repetition_penalty", 1.0)
            gen_params["max_new_tokens"] = int(SUPER_PROMPT_MAX_NEW_TOKENS)
            gen_params["use_cache"] = True
            gen_params["pad_token_id"] = engine._get_pad_token_id(processor, model)
            gen_params["do_sample"] = float(gen_params.get("temperature", 0.0) or 0.0) > 0.0
            gen_params = _normalize_generation_params(gen_params)

            rng_cuda_devices = engine._cuda_indices_for_rng(model, input_device)
            if engine._supports_generator(model):
                gen_device = engine._select_generator_device(input_device)
                gen = __import__("torch").Generator(device=gen_device)
                gen.manual_seed(int(SUPER_PROMPT_SEED))
                gen_params["generator"] = gen
                rng_context = nullcontext()
            else:
                torch = __import__("torch")
                rng_context = torch.random.fork_rng(devices=rng_cuda_devices) if rng_cuda_devices else torch.random.fork_rng()

            torch = __import__("torch")
            dtype = engine._dtype_from_precision(resolved_precision)
            autocast_device = input_device if hasattr(input_device, "type") else getattr(model, "device", None)
            use_autocast = getattr(autocast_device, "type", None) == "cuda" and dtype in (torch.float16, torch.bfloat16)
            gen_params = _filter_generation_params(model, gen_params)

            with rng_context:
                if "generator" not in gen_params:
                    torch.manual_seed(int(SUPER_PROMPT_SEED))
                    for idx in rng_cuda_devices:
                        with torch.cuda.device(idx):
                            torch.cuda.manual_seed(int(SUPER_PROMPT_SEED))

                _send_progress(operation_id, "Generating AI prompt", 78.0)
                with torch.inference_mode():
                    if use_autocast:
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            generated_ids = _generate_with_filtered_kwargs(model, inputs, gen_params)
                    else:
                        generated_ids = _generate_with_filtered_kwargs(model, inputs, gen_params)

            _send_progress(operation_id, "Decoding prompt", 92.0)
            generated_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = engine._batch_decode(
                processor,
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]
            return _clean_model_output(output_text)
        finally:
            if SUPER_PROMPT_UNLOAD_AFTER_GENERATION:
                _send_progress(operation_id, "Unloading Qwen", 96.0)
                engine._unload_model(DEFAULT_MODEL_ID, resolved_precision, resolved_attention)
            elif moved_to_gpu:
                try:
                    model.to("cpu")
                    engine._prepare_memory(force=True)
                except Exception:
                    pass
            gc.collect()
    finally:
        _MODEL_LOCK.release()


@_register_post(f"{AI_ROUTE_BASE}/enhance")
async def enhance_endpoint(request: web.Request) -> web.StreamResponse:
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body."}, status=400)
    except Exception:
        data = {}

    operation_id = str(data.get("operation_id") or "")
    try:
        result = await asyncio.to_thread(
            _generate_with_qwen,
            str(data.get("text") or ""),
            str(data.get("system_preset") or DEFAULT_PRESET),
            operation_id,
            None,
        )
        _send_done(operation_id, "AI prompt ready")
        return web.json_response({"ok": True, "text": result, "thinking": False, "model": DEFAULT_MODEL_ID})
    except Exception as exc:
        LOGGER.exception("%s AI prompt enhancement failed", LOG_PREFIX)
        _send_error(operation_id, str(exc))
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


class TS_SuperPrompt(IO.ComfyNode):
    """Compact prompt node: microphone dictation plus Qwen prompt enhancement."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        preset_options = _preset_options()
        return IO.Schema(
            node_id="TS_SuperPrompt",
            display_name="TS Super Prompt",
            category="TS/LLM",
            description=(
                "Voice prompt field with optional Qwen3.5 AI prompt enhancement for image, video, and music prompts."
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
                    options=preset_options,
                    default=_default_preset(preset_options),
                    tooltip="Выберите системный пресет из qwen_3_vl_presets.json для улучшения промпта.",
                ),
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "Опциональное изображение-референс для улучшения промпта, "
                        "если SUPER_PROMPT_ENHANCE_ON_EXECUTE включен в коде."
                    ),
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
        **_: Any,
    ) -> bool | str:
        if not isinstance(text, str):
            return "text must be a string."
        if not isinstance(high_quality, bool):
            return "high_quality must be a boolean."
        if system_preset not in _preset_options():
            return "system_preset must be one of the presets from qwen_3_vl_presets.json."
        return True

    @classmethod
    def execute(
        cls,
        text: str = "",
        high_quality: bool = False,
        translate_to_english: bool = False,
        system_preset: str = DEFAULT_PRESET,
        image: Any = None,
        **_: Any,
    ) -> IO.NodeOutput:
        _ = high_quality, translate_to_english
        if not SUPER_PROMPT_ENHANCE_ON_EXECUTE:
            return IO.NodeOutput(text or "")

        enhanced = _generate_with_qwen(
            text=text or "",
            system_preset=system_preset,
            operation_id=None,
            image=image,
        )
        return IO.NodeOutput(enhanced)


NODE_CLASS_MAPPINGS = {"TS_SuperPrompt": TS_SuperPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_SuperPrompt": "TS Super Prompt"}
