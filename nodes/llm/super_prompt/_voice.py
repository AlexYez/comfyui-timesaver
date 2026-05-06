"""Whisper voice-recognition pipeline for TS Super Prompt.

Owns: Whisper model download/load lifecycle, ffmpeg audio decoding, VAD-based
trimming + edge-fade + RMS normalization preprocessing, duplicate-stop-word
cleanup, and the public ``transcribe_audio`` entry point used by the
``/ts_voice_recognition/transcribe`` HTTP route.

Heavy deps (``torch``, ``whisper``, ``numpy``) are imported lazily inside
``_load_whisper_runtime`` so the module is cheap to import even when the user
never calls voice features.

Private — loader skips paths with `_`-prefixed components.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folder_paths

from ._helpers import (
    ACTIVE_MODEL,
    ALL_MODELS,
    AUDIO_EDGE_FADE_MS,
    AUDIO_NORMALIZE_ENABLED,
    AUDIO_NORMALIZE_MAX_GAIN_DB,
    AUDIO_NORMALIZE_TARGET_PEAK,
    AUDIO_SAMPLE_RATE,
    AUDIO_TRIM_ENABLED,
    AUDIO_VAD_ADAPTIVE_MULTIPLIER,
    AUDIO_VAD_ENABLED,
    AUDIO_VAD_FRAME_MS,
    AUDIO_VAD_HOP_MS,
    AUDIO_VAD_MIN_SPEECH_SEC,
    AUDIO_VAD_PADDING_SEC,
    AUDIO_VAD_RMS_THRESHOLD,
    BEAM_SIZE,
    DOWNLOAD_LOCK,
    GPU_PRECISION,
    INITIAL_PROMPT,
    INITIAL_PROMPT_ENABLED,
    INITIAL_PROMPT_EXTRA,
    LOGGER,
    MODEL_FILE_NAMES,
    MODEL_SIZES,
    TEMPERATURE,
    VOICE_LOG_PREFIX,
    VOICE_MODEL_BASE,
    VOICE_MODEL_CACHE,
    VOICE_MODEL_HIGH_QUALITY,
    WHISPER_DIR,
    send_voice_event,
    send_voice_status,
    voice_log_info,
    voice_log_warning,
)


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
        send_voice_event(event, {"model": self.model_name, **data})

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

    with DOWNLOAD_LOCK:
        progress = ProgressBroadcaster(name)
        target_file = _model_file_path(name)

        if force and target_file.exists():
            try:
                target_file.unlink()
            except OSError as exc:
                voice_log_warning(f"Could not remove old model file for '{name}': {exc}")

        if not force and is_model_cached(name):
            progress.done(f"{name} file ready")
            return target_file

        progress.status(f"Downloading {name}")
        voice_log_info(f"Downloading Whisper model '{name}' to {WHISPER_DIR}")

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
        voice_log_info(f"Whisper model '{name}' ready ({final_size / (1024 * 1024):.0f} MB)")
        return target_file


def load_model(name: str, device: str = "auto", progress_start: float = 82.0, progress_end: float = 96.0):
    torch, whisper = _load_whisper_runtime()

    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = str(device)
        if target_device == "cuda" and not torch.cuda.is_available():
            voice_log_warning("CUDA was requested but is not available. Falling back to CPU.")
            target_device = "cpu"

    use_fp16 = target_device == "cuda" and GPU_PRECISION == "fp16"
    cache_key = (name, target_device, use_fp16)
    if cache_key in VOICE_MODEL_CACHE:
        send_voice_status(name, "Voice model already in memory", progress_end)
        return VOICE_MODEL_CACHE[cache_key], target_device, use_fp16

    ensure_model(name)
    voice_log_info(f"Loading Whisper model '{name}' on {target_device} ({'fp16' if use_fp16 else 'fp32'})")
    send_voice_status(name, "Loading model...", progress_start)

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
        voice_log_warning(f"GPU load failed for '{name}': {exc}. Falling back to CPU.")
        send_voice_status(name, "GPU load failed; using CPU", min(progress_end, progress_start + 2.0))
        target_device = "cpu"
        use_fp16 = False
        cache_key = (name, target_device, use_fp16)
        send_voice_status(name, "Loading model...", min(progress_end, progress_start + 4.0))
        model = whisper.load_model(
            name,
            device=target_device,
            download_root=str(WHISPER_DIR),
            in_memory=False,
        )

    VOICE_MODEL_CACHE[cache_key] = model
    send_voice_status(name, "Voice model loaded into memory", progress_end)
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
        # Voice clips are short (≤10 min after VAD); 5 min is a generous cap
        # for slow disks and prevents a hung ffmpeg from blocking the worker.
        result = subprocess.run(command, capture_output=True, check=True, timeout=300)
        return np.frombuffer(result.stdout, dtype=np.float32).copy()
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        voice_log_warning("ffmpeg timed out decoding voice clip; falling back to whisper.load_audio.")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        voice_log_warning(f"ffmpeg failed to decode audio: {stderr}")

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
    "с", "со", "в", "во", "к", "ко", "у", "о", "об", "от", "до", "из", "за",
    "на", "по", "под", "над", "при", "для", "через", "без",
    "with", "in", "on", "at", "to", "from", "of", "for", "by",
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


# ---------------------------------------------------------------------------
# Recognition entry point
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

    send_voice_status(model_name, "Preparing audio", 10.0)
    audio_info = _preprocess_audio(_read_audio(filepath), AUDIO_SAMPLE_RATE)
    metadata = _audio_metadata(audio_info)
    if not audio_info.speech_detected or len(audio_info.audio) == 0:
        send_voice_status(model_name, "No speech detected", 100.0)
        return {
            "text": "",
            "language": language or "?",
            "task": task,
            **metadata,
        }

    send_voice_status(model_name, "Loading voice model", 40.0)
    model, _, use_fp16 = load_model(model_name, device, progress_start=42.0, progress_end=64.0)
    send_voice_status(model_name, "Recognizing speech", 68.0)

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

    send_voice_status(model_name, "Finalizing speech text", 92.0)
    text = _clean_transcription_text(result.get("text") or "")
    detected = result.get("language", language or "?")
    return {
        "text": text,
        "language": detected,
        "task": task,
        **metadata,
    }


__all__ = [
    "_audio_tmp_dir",
    "_ensure_runtime_dirs",
    "_missing_runtime_packages",
    "_configured_initial_prompt",
    "_resolve_voice_model",
    "is_model_cached",
    "ensure_model",
    "load_model",
    "transcribe_audio",
    "_model_file_path",
    "ACTIVE_MODEL",
]
