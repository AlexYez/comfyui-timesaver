from __future__ import annotations

import asyncio
import importlib.util
import logging
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import folder_paths
from aiohttp import web
from comfy_api.latest import IO

try:
    import server
except Exception:
    server = None


LOGGER = logging.getLogger("comfyui_timesaver.ts_voice_recognition")
LOG_PREFIX = "[TS Voice Recognition]"

ACTIVE_MODEL = "base"
DEVICE = "auto"
GPU_PRECISION = "fp16"
SOURCE_LANGUAGE = "ru"
INITIAL_PROMPT = ""
BEAM_SIZE = 5
TEMPERATURE = 0.0

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
            "Missing dependencies for TS Voice Recognition: "
            f"{', '.join(missing)}. Install requirements.txt and restart ComfyUI."
        )

    try:
        import torch
        import whisper
    except ImportError as exc:
        raise RuntimeError(
            "TS Voice Recognition requires openai-whisper and torch. "
            "Install requirements.txt and restart ComfyUI."
        ) from exc

    return torch, whisper


def _get_ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


class ProgressBroadcaster:
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


def transcribe_audio(
    filepath: str,
    model_name: str,
    device: str,
    source_language: str | None,
    target_language: str,
    initial_prompt: str | None = None,
) -> dict[str, Any]:
    torch, _ = _load_whisper_runtime()
    model, _, use_fp16 = load_model(model_name, device)

    audio = _read_audio(filepath)
    duration = len(audio) / 16000.0
    task = "translate" if target_language == "en" else "transcribe"
    language = None if source_language in (None, "", "auto") else source_language

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

    with torch.no_grad():
        result = model.transcribe(audio, **transcribe_kwargs)

    text = (result.get("text") or "").strip()
    detected = result.get("language", language or "?")
    return {
        "text": text,
        "language": detected,
        "duration": round(duration, 2),
        "task": task,
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

        translate = request.query.get("translate", "false").lower() in {"true", "1", "yes"}
        model_name = ACTIVE_MODEL
        if translate and model_name in MODELS_WITHOUT_TRANSLATE:
            _log_warning(f"Model '{model_name}' does not support translation. Using transcription.")
            translate = False

        target_language = "en" if translate else "same"
        result = await asyncio.to_thread(
            transcribe_audio,
            str(filepath),
            model_name,
            DEVICE,
            SOURCE_LANGUAGE,
            target_language,
            INITIAL_PROMPT or None,
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
            }
        }
    )


class TS_VoiceRecognition(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_VoiceRecognition",
            display_name="TS Voice Recognition",
            category="TS/audio",
            description="Record speech from the browser microphone and insert recognized Whisper text into the node.",
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Recognized text buffer. The voice button inserts text at the current cursor position.",
                ),
                IO.Boolean.Input(
                    "translate_to_english",
                    default=False,
                    tooltip="Translate speech to English instead of keeping the source language.",
                ),
            ],
            outputs=[IO.String.Output(display_name="text")],
            search_aliases=["voice recognition", "whisper recorder", "speech to text", "microphone"],
        )

    @classmethod
    def execute(cls, text: str = "", translate_to_english: bool = False) -> IO.NodeOutput:
        _ = translate_to_english
        return IO.NodeOutput(text or "")


NODE_CLASS_MAPPINGS = {"TS_VoiceRecognition": TS_VoiceRecognition}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_VoiceRecognition": "TS Voice Recognition"}
