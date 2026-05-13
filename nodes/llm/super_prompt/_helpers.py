"""Shared infrastructure for the TS Super Prompt subpackage.

Owns the cross-module bits that both the voice (_voice.py) and Qwen
(_qwen.py) pipelines need: package logger, PromptServer event dispatch,
aiohttp route decorators, byte/dir formatting helpers, and the global
locks. Keeping this in one place avoids circular imports between the
voice and Qwen modules.

Private — loader skips paths with `_`-prefixed components.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import folder_paths

try:
    import server
except Exception:
    server = None


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

# Trailing-tail Whisper hallucinations. Whisper trained on YouTube subtitles
# tends to invent typical "outro" phrases when the audio fades to silence/noise
# at the end. The filter only matches the END of the cleaned text (anchored
# `$` after _collapse_repeated_phrases) so legitimate dictation containing
# these words mid-sentence is preserved. Patterns are case-insensitive and
# tolerate trailing punctuation/ellipsis variants. Disable by flipping the
# flag below if you ever need raw Whisper output.
WHISPER_HALLUCINATION_FILTER_ENABLED = True
WHISPER_HALLUCINATION_PATTERNS = (
    r"продолжение\s+следует",
)

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
# Hysteresis: high threshold (above) decides "this IS speech"; low threshold
# (below) extends the boundaries outward through quieter frames so short
# unvoiced consonants and Russian one-letter prepositions ("с", "в", "к") at
# the edges of an utterance are not clipped. Low must stay ≤ high or
# expansion would never trigger.
AUDIO_VAD_LOW_MULTIPLIER = 1.3
AUDIO_VAD_MIN_SPEECH_SEC = 0.18
# Padding bumped from 0.30 → 0.40s after reports of clipped utterance tails.
# This is a safety margin AFTER hysteresis expansion, not the only mechanism.
AUDIO_VAD_PADDING_SEC = 0.40
# Edge fade reduced from 12ms → 6ms so a fast consonant landing at the very
# start/end of the trimmed window keeps most of its energy.
AUDIO_EDGE_FADE_MS = 6
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

PROMPT_TARGETS = ("auto", "image", "video", "music")

# Hard cap on /ts_super_prompt/enhance text length. Anything bigger is almost
# certainly a misuse or DoS attempt — Qwen3.5 has a much smaller context budget
# in practice and 8 KiB is well above any reasonable creative prompt.
ENHANCE_MAX_TEXT_LEN = 8192
# Hard cap on /ts_voice_recognition/transcribe upload size. The compact voice
# recorder in the UI produces ≤2 MB clips for typical 30-second prompts; 50 MB
# is well above any reasonable speech upload and stops a hostile / runaway
# client from filling RAM via repeated multipart streams. ComfyUI ships
# without auth, so this guard is the only thing standing between LAN clients
# and unbounded buffering.
VOICE_UPLOAD_MAX_BYTES = 50 * 1024 * 1024


# Module-level locks shared across voice + qwen pipelines.
DOWNLOAD_LOCK = threading.Lock()
VOICE_MODEL_CACHE: dict[tuple[str, str, bool], Any] = {}
MODEL_LOCK = threading.Lock()


def log_info(message: str) -> None:
    LOGGER.info("%s %s", LOG_PREFIX, message)


def log_warning(message: str) -> None:
    LOGGER.warning("%s %s", LOG_PREFIX, message)


def voice_log_info(message: str) -> None:
    LOGGER.info("%s %s", VOICE_LOG_PREFIX, message)


def voice_log_warning(message: str) -> None:
    LOGGER.warning("%s %s", VOICE_LOG_PREFIX, message)


def _resolve_prompt_server():
    if server is None:
        log_warning("PromptServer unavailable. HTTP routes disabled.")
        return None
    try:
        return server.PromptServer.instance
    except Exception as exc:
        log_warning(f"PromptServer init failed. HTTP routes disabled: {exc}")
        return None


PROMPT_SERVER = _resolve_prompt_server()


def register_post(path: str):
    def decorator(func):
        if PROMPT_SERVER is None:
            return func
        try:
            PROMPT_SERVER.routes.post(path)(func)
        except Exception as exc:
            log_warning(f"Failed to register POST route '{path}': {exc}")
        return func

    return decorator


def register_get(path: str):
    def decorator(func):
        if PROMPT_SERVER is None:
            return func
        try:
            PROMPT_SERVER.routes.get(path)(func)
        except Exception as exc:
            log_warning(f"Failed to register GET route '{path}': {exc}")
        return func

    return decorator


def send_ai_event(event: str, payload: dict[str, Any]) -> None:
    if PROMPT_SERVER is None:
        return
    try:
        PROMPT_SERVER.send_sync(f"{AI_EVENT_PREFIX}.{event}", payload)
    except Exception as exc:
        LOGGER.debug("%s WebSocket event send failed: %s", LOG_PREFIX, exc)


def send_voice_event(event: str, payload: dict[str, Any]) -> None:
    if PROMPT_SERVER is None:
        return
    try:
        PROMPT_SERVER.send_sync(f"{VOICE_EVENT_PREFIX}.{event}", payload)
    except Exception as exc:
        LOGGER.debug("%s WebSocket event send failed: %s", VOICE_LOG_PREFIX, exc)


def send_progress(operation_id: str | None, text: str, percent: float | None = None) -> None:
    payload: dict[str, Any] = {"text": text}
    if operation_id:
        payload["operation_id"] = operation_id
    if percent is not None:
        payload["percent"] = max(0.0, min(100.0, float(percent)))
    send_ai_event("progress", payload)


def send_done(operation_id: str | None, text: str = "Ready") -> None:
    payload: dict[str, Any] = {"text": text, "percent": 100.0}
    if operation_id:
        payload["operation_id"] = operation_id
    send_ai_event("done", payload)


def send_error(operation_id: str | None, text: str) -> None:
    payload: dict[str, Any] = {"text": text}
    if operation_id:
        payload["operation_id"] = operation_id
    send_ai_event("error", payload)


def send_voice_status(model_name: str, text: str, percent: float | None = None) -> None:
    payload: dict[str, Any] = {"model": model_name, "text": text}
    if percent is not None:
        payload["percent"] = max(0.0, min(100.0, float(percent)))
    send_voice_event("status", payload)


def format_bytes(size_bytes: int) -> str:
    value = float(max(0, int(size_bytes)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def directory_size(path: Path) -> int:
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
