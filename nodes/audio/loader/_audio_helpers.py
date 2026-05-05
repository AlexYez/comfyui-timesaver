"""Shared helpers and aiohttp routes for TS_AudioLoader / TS_AudioPreview.

Private module: not registered as a public node by the loader.
HTTP routes register at import time so the frontend (ts-audio-loader.js)
keeps working regardless of which node was instantiated first.
"""

from __future__ import annotations

__all__ = [
    "MediaMetadata",
    "LOG_PREFIX",
    "_log_info",
    "_log_warning",
    "_empty_audio",
    "_normalize_selected_path",
    "_probe_media",
    "_sanitize_crop",
    "_decode_audio_segment",
    "_seconds_to_hms",
    "_get_uploadable_media_options",
    "_coerce_audio_tensor",
    "_hash_audio_tensor",
    "_build_generated_audio_preview_payload",
]

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import time
import urllib.parse
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folder_paths
import numpy as np
import torch
from aiohttp import web
from comfy_api.latest import IO

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

try:
    import server
except Exception:
    server = None

LOGGER = logging.getLogger("comfyui_timesaver.ts_audio_loader")
LOG_PREFIX = "[TS Audio Loader]"
NODE_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = NODE_ROOT / ".cache" / "ts_audio_loader"
RECORDINGS_DIR = Path(folder_paths.get_input_directory()) / "ts_audio_loader_recordings"
GENERATED_AUDIO_DIR = CACHE_DIR / "generated_audio"
PREVIEW_BINS = 2048
PREVIEW_SAMPLE_RATE = 4000
SUPPORTED_AUDIO_EXTENSIONS = {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma", ".webm"}
SUPPORTED_VIDEO_EXTENSIONS = {".avi", ".flv", ".m2ts", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".mts", ".ts", ".webm"}


@dataclass
class MediaMetadata:
    filepath: str
    filename: str
    duration_seconds: float
    sample_rate: int
    channels: int
    target_channels: int
    media_type: str
    audio_codec: str
    has_video: bool


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


def _normalize_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _get_ffmpeg_executable() -> str:
    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            LOGGER.debug("%s imageio_ffmpeg.get_ffmpeg_exe() failed, falling back to PATH: %s", LOG_PREFIX, exc)
    return "ffmpeg"


def _allowed_view_roots() -> tuple[Path, ...]:
    # Resolved on every call so changes to folder_paths (rare) are picked up.
    roots: list[Path] = []
    try:
        roots.append(Path(folder_paths.get_input_directory()).resolve())
    except (OSError, ValueError):
        pass
    try:
        roots.append(Path(folder_paths.get_output_directory()).resolve())
    except (OSError, ValueError, AttributeError):
        pass
    for candidate in (RECORDINGS_DIR, GENERATED_AUDIO_DIR, CACHE_DIR):
        try:
            roots.append(candidate.resolve())
        except OSError:
            continue
    return tuple(roots)


def _is_inside_allowed_root(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return False
    for root in _allowed_view_roots():
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _hash_file_identity(filepath: str) -> str:
    stat = os.stat(filepath)
    payload = f"{_normalize_path(filepath)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(filepath: str) -> Path:
    return CACHE_DIR / f"{_hash_file_identity(filepath)}.json"


def _seconds_to_hms(value: float) -> str:
    total = max(0.0, float(value))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    seconds = total % 60.0
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def _parse_duration_from_ffmpeg(stderr_text: str) -> float:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr_text)
    if not match:
        return 0.0
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return (hours * 3600) + (minutes * 60) + seconds


def _parse_audio_metadata(stderr_text: str, filepath: str) -> MediaMetadata:
    audio_match = re.search(r"Audio:\s*([^,\n]+)(.*)", stderr_text)
    if not audio_match:
        raise RuntimeError("No audio stream found in the selected file.")
    audio_codec = audio_match.group(1).strip()
    audio_rest = audio_match.group(2)
    rate_match = re.search(r"(\d+)\s*Hz", audio_rest)
    sample_rate = int(rate_match.group(1)) if rate_match else 44100
    channels = 2
    if re.search(r"\bmono\b", audio_rest, flags=re.IGNORECASE):
        channels = 1
    elif re.search(r"\bstereo\b", audio_rest, flags=re.IGNORECASE):
        channels = 2
    else:
        channel_match = re.search(r"(\d+(?:\.\d+)?)\s*channels?", audio_rest, flags=re.IGNORECASE)
        if channel_match:
            channels = max(1, int(float(channel_match.group(1))))
        elif re.search(r"\b5\.1\b", audio_rest):
            channels = 6
        elif re.search(r"\b7\.1\b", audio_rest):
            channels = 8
    has_video = re.search(r"Video:\s*", stderr_text) is not None
    ext = Path(filepath).suffix.lower()
    media_type = "video" if has_video or ext in SUPPORTED_VIDEO_EXTENSIONS else "audio"
    return MediaMetadata(
        filepath=_normalize_path(filepath),
        filename=os.path.basename(filepath),
        duration_seconds=_parse_duration_from_ffmpeg(stderr_text),
        sample_rate=max(1, int(sample_rate)),
        channels=max(1, int(channels)),
        target_channels=1 if channels <= 1 else 2,
        media_type=media_type,
        audio_codec=audio_codec,
        has_video=has_video,
    )


def _probe_media(filepath: str) -> MediaMetadata:
    ffmpeg_exe = _get_ffmpeg_executable()
    result = subprocess.run([ffmpeg_exe, "-hide_banner", "-i", filepath], capture_output=True, check=False)
    return _parse_audio_metadata(result.stderr.decode("utf-8", errors="replace"), filepath)


def _normalize_selected_path(source_path: str) -> str:
    if not source_path:
        return ""
    normalized = os.path.normpath(source_path)
    if os.path.isfile(normalized):
        return normalized
    try:
        annotated = folder_paths.get_annotated_filepath(source_path)
    except Exception:
        annotated = ""
    if annotated and os.path.isfile(annotated):
        return annotated
    return normalized


def _get_uploadable_media_options() -> list[str]:
    """Return input-directory files compatible with ComfyUI's native upload combo."""
    input_dir = folder_paths.get_input_directory()
    try:
        files, _ = folder_paths.recursive_search(input_dir)
    except OSError as exc:
        _log_warning(f"Failed to scan input directory for media files: {exc}")
        return []
    return sorted(folder_paths.filter_files_content_types(files, ["audio", "video"]))


def _to_input_annotation(filepath: Path) -> str:
    """Convert an input-directory file path to ComfyUI's annotated widget value."""
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    resolved_path = filepath.resolve()
    try:
        relative_path = resolved_path.relative_to(input_dir)
    except ValueError:
        return _normalize_path(str(resolved_path))
    return f"{_normalize_path(str(relative_path))} [input]"


def _collapse_peaks(peaks: list[float], target_bins: int) -> list[float]:
    if len(peaks) <= target_bins:
        return peaks
    values = np.asarray(peaks, dtype=np.float32)
    sections = np.array_split(values, target_bins)
    return [float(section.max()) if section.size else 0.0 for section in sections]


def _build_waveform_preview(metadata: MediaMetadata, target_bins: int = PREVIEW_BINS) -> list[float]:
    ffmpeg_exe = _get_ffmpeg_executable()
    command = [
        ffmpeg_exe, "-v", "error", "-nostdin", "-i", metadata.filepath, "-vn", "-ac", "1",
        "-ar", str(PREVIEW_SAMPLE_RATE), "-f", "s16le", "-"
    ]
    expected_samples = max(1, int(math.ceil(metadata.duration_seconds * PREVIEW_SAMPLE_RATE)))
    samples_per_bin = max(1, int(math.ceil(expected_samples / max(1, target_bins))))
    peaks: list[float] = []
    remainder = np.empty(0, dtype=np.int16)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    assert process.stdout is not None
    assert process.stderr is not None
    while True:
        chunk = process.stdout.read(65536)
        if not chunk:
            break
        data = np.frombuffer(chunk, dtype=np.int16)
        if remainder.size:
            data = np.concatenate((remainder, data))
        full_count = (data.size // samples_per_bin) * samples_per_bin
        if full_count:
            reshaped = data[:full_count].reshape(-1, samples_per_bin)
            peaks.extend((np.max(np.abs(reshaped), axis=1) / 32768.0).astype(np.float32).tolist())
        remainder = data[full_count:]
    stderr_text = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0 and not peaks and not remainder.size:
        raise RuntimeError(stderr_text.strip() or "ffmpeg failed to generate waveform preview.")
    if remainder.size:
        peaks.append(float(np.max(np.abs(remainder)) / 32768.0))
    if not peaks:
        peaks = [0.0]
    return _collapse_peaks(peaks, target_bins=target_bins)


def _read_cached_preview(filepath: str) -> dict[str, Any]:
    cache_file = _cache_path(filepath)
    if not cache_file.is_file():
        return {}
    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("cache_key") != _hash_file_identity(filepath):
        return {}
    return payload


def _write_cached_preview(filepath: str, payload: dict[str, Any]) -> None:
    cache_file = _cache_path(filepath)
    cache_payload = dict(payload)
    cache_payload["cache_key"] = _hash_file_identity(filepath)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as handle:
            json.dump(cache_payload, handle, ensure_ascii=True)
    except OSError as exc:
        _log_warning(f"Failed to write preview cache: {exc}")


def _get_media_preview(filepath: str) -> dict[str, Any]:
    cached = _read_cached_preview(filepath)
    if cached:
        return cached
    metadata = _probe_media(filepath)
    payload = {
        "filepath": metadata.filepath,
        "filename": metadata.filename,
        "duration_seconds": metadata.duration_seconds,
        "sample_rate": metadata.sample_rate,
        "channels": metadata.channels,
        "target_channels": metadata.target_channels,
        "media_type": metadata.media_type,
        "audio_codec": metadata.audio_codec,
        "has_video": metadata.has_video,
        "peaks": _build_waveform_preview(metadata),
    }
    _write_cached_preview(filepath, payload)
    return payload


def _decode_audio_segment(metadata: MediaMetadata, start_seconds: float, end_seconds: float | None) -> tuple[torch.Tensor, int]:
    ffmpeg_exe = _get_ffmpeg_executable()
    command = [ffmpeg_exe, "-v", "error", "-nostdin"]
    if start_seconds > 0:
        command.extend(["-ss", f"{start_seconds:.6f}"])
    command.extend(["-i", metadata.filepath, "-vn"])
    if end_seconds is not None and end_seconds > start_seconds:
        command.extend(["-t", f"{max(0.0, end_seconds - start_seconds):.6f}"])
    command.extend(["-ac", str(metadata.target_channels), "-ar", str(metadata.sample_rate), "-f", "f32le", "-acodec", "pcm_f32le", "-"])
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg failed to decode audio.")
    raw = np.frombuffer(process.stdout, dtype=np.float32)
    if raw.size == 0:
        return torch.zeros((metadata.target_channels, 1), dtype=torch.float32), metadata.sample_rate
    remainder = raw.size % metadata.target_channels
    if remainder:
        raw = raw[: raw.size - remainder]
    if raw.size == 0:
        return torch.zeros((metadata.target_channels, 1), dtype=torch.float32), metadata.sample_rate
    waveform = torch.from_numpy(raw.reshape(-1, metadata.target_channels)).transpose(0, 1).contiguous()
    return waveform.clamp(-1.0, 1.0), metadata.sample_rate


def _empty_audio() -> dict[str, Any]:
    return {"waveform": torch.zeros((1, 1, 1024), dtype=torch.float32), "sample_rate": 44100}


def _sanitize_crop(duration_seconds: float, crop_start_seconds: float, crop_end_seconds: float) -> tuple[float, float]:
    duration = max(0.0, float(duration_seconds))
    start = max(0.0, float(crop_start_seconds))
    end = float(crop_end_seconds)
    if duration <= 0.0:
        return 0.0, 0.0
    start = min(start, duration)
    end = duration if end <= 0.0 or end <= start else min(end, duration)
    if end <= start:
        start = 0.0
        end = duration
    return start, end


def _coerce_audio_tensor(audio: dict[str, Any]) -> tuple[torch.Tensor, int]:
    """Normalize ComfyUI audio input to a CPU float tensor shaped [channels, samples]."""
    waveform = audio.get("waveform")
    sample_rate = max(1, int(audio.get("sample_rate", 44100)))
    if waveform is None:
        return torch.zeros((1, 1), dtype=torch.float32), sample_rate
    tensor = torch.as_tensor(waveform).detach().cpu().float()
    if tensor.ndim == 3:
        tensor = tensor[0]
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 2:
        raise ValueError(f"Unsupported audio waveform shape: {tuple(tensor.shape)}")
    if tensor.shape[0] == 0 or tensor.shape[-1] == 0:
        return torch.zeros((1, 1), dtype=torch.float32), sample_rate
    return tensor.contiguous(), sample_rate


def _audio_duration_seconds(audio: dict[str, Any]) -> float:
    waveform, sample_rate = _coerce_audio_tensor(audio)
    return float(waveform.shape[-1]) / float(sample_rate)


def _crop_audio_input(audio: dict[str, Any], crop_start_seconds: float, crop_end_seconds: float) -> tuple[dict[str, Any], int]:
    """Crop a ComfyUI audio payload without mutating the input."""
    waveform, sample_rate = _coerce_audio_tensor(audio)
    duration_seconds = float(waveform.shape[-1]) / float(sample_rate)
    start_seconds, end_seconds = _sanitize_crop(duration_seconds, crop_start_seconds, crop_end_seconds)
    start_index = min(waveform.shape[-1], max(0, int(math.floor(start_seconds * sample_rate))))
    end_index = min(waveform.shape[-1], max(start_index + 1, int(math.ceil(end_seconds * sample_rate))))
    cropped = waveform[:, start_index:end_index].contiguous()
    duration_int = int(math.ceil(cropped.shape[-1] / float(sample_rate))) if cropped.shape[-1] > 0 else 0
    return {"waveform": cropped.unsqueeze(0), "sample_rate": sample_rate}, duration_int


def _build_peaks_from_audio_tensor(
    waveform: torch.Tensor,
    sample_rate: int,
    target_bins: int = PREVIEW_BINS,
) -> list[float]:
    """Create waveform peaks from a [channels, samples] tensor using loader-like preview sampling."""
    if waveform.ndim != 2:
        raise ValueError(f"Expected [channels, samples] waveform, got {tuple(waveform.shape)}")
    if waveform.shape[-1] == 0:
        return [0.0]
    mono = waveform.mean(dim=0, keepdim=True)
    preview_sample_count = max(1, int(math.ceil((waveform.shape[-1] / float(max(1, sample_rate))) * PREVIEW_SAMPLE_RATE)))
    if mono.shape[-1] != preview_sample_count:
        mono = torch.nn.functional.interpolate(
            mono.unsqueeze(0),
            size=preview_sample_count,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
    mono = mono.abs().squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    if mono.size == 0:
        return [0.0]
    if mono.size <= target_bins:
        return [float(value) for value in mono.tolist()]
    sections = np.array_split(mono, target_bins)
    return [float(section.max()) if section.size else 0.0 for section in sections]


def _hash_audio_tensor(waveform: torch.Tensor, sample_rate: int, prefix: str) -> str:
    """Build a stable cache key for generated preview audio."""
    hasher = hashlib.sha256()
    hasher.update(prefix.encode("utf-8"))
    hasher.update(str(sample_rate).encode("utf-8"))
    hasher.update(str(tuple(waveform.shape)).encode("utf-8"))
    hasher.update(waveform.contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _write_preview_audio_file(waveform: torch.Tensor, sample_rate: int, prefix: str) -> Path:
    """Persist a preview WAV file that the browser player can stream."""
    preview_key = _hash_audio_tensor(waveform, sample_rate, prefix)
    output_path = GENERATED_AUDIO_DIR / f"{preview_key}.wav"
    if output_path.is_file():
        return output_path
    GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    channel_count = min(2, max(1, int(waveform.shape[0])))
    pcm = waveform[:channel_count].transpose(0, 1).numpy()
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm16 = np.round(pcm * 32767.0).astype(np.int16, copy=False)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channel_count)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return output_path


def _build_generated_audio_preview_payload(audio: dict[str, Any], prefix: str, filename: str) -> dict[str, Any]:
    """Create browser preview metadata for an in-memory ComfyUI audio object."""
    waveform, sample_rate = _coerce_audio_tensor(audio)
    preview_path = _write_preview_audio_file(waveform, sample_rate, prefix)
    return {
        "preview_path": _normalize_path(str(preview_path)),
        "filename": filename,
        "duration_seconds": float(waveform.shape[-1]) / float(sample_rate),
        "sample_rate": sample_rate,
        "channels": int(waveform.shape[0]),
        "target_channels": int(min(2, max(1, waveform.shape[0]))),
        "media_type": "audio",
        "audio_codec": "pcm_s16le",
        "has_video": False,
        "peaks": _build_peaks_from_audio_tensor(waveform, sample_rate),
    }



@_register_get("/ts_audio_loader/metadata")
async def ts_audio_loader_metadata(request: web.Request) -> web.StreamResponse:
    filepath = _normalize_selected_path(urllib.parse.unquote(request.query.get("filepath", "")))
    if not filepath or not os.path.isfile(filepath):
        return web.json_response({"error": "File not found."}, status=404)
    if not _is_inside_allowed_root(Path(filepath)):
        _log_warning(f"Rejected metadata request outside allowed roots: {filepath}")
        return web.json_response({"error": "File not found."}, status=404)
    try:
        return web.json_response(await asyncio.to_thread(_get_media_preview, filepath))
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


@_register_get("/ts_audio_loader/view")
async def ts_audio_loader_view(request: web.Request) -> web.StreamResponse:
    filepath = _normalize_selected_path(urllib.parse.unquote(request.query.get("filepath", "")))
    if not filepath or not os.path.isfile(filepath):
        return web.Response(status=404)
    if not _is_inside_allowed_root(Path(filepath)):
        _log_warning(f"Rejected view request outside allowed roots: {filepath}")
        return web.Response(status=404)
    try:
        return web.FileResponse(filepath)
    except Exception:
        return web.Response(status=500)


@_register_post("/ts_audio_loader/upload_recording")
async def ts_audio_loader_upload_recording(request: web.Request) -> web.StreamResponse:
    try:
        reader = await request.multipart()
        audio_part = await reader.next()
        if audio_part is None or audio_part.name != "audio":
            return web.json_response({"error": "Missing audio field."}, status=400)
        original_name = audio_part.filename or "recording.webm"
        suffix = Path(original_name).suffix.lower() or ".webm"
        if len(suffix) > 10:
            suffix = ".webm"
        output_name = f"ts_audio_recording_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{int(time.time_ns() % 1000000)}{suffix}"
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RECORDINGS_DIR / output_name
        with output_path.open("wb") as handle:
            while True:
                chunk = await audio_part.read_chunk(size=1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        return web.json_response({"path": _to_input_annotation(output_path), "filename": output_name})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)

