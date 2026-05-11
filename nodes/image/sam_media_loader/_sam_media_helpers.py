"""Shared helpers and aiohttp routes for TS_SAM_MediaLoader.

Private module: not registered as a public node by the loader.
HTTP routes register at import time so the frontend
(``js/image/sam_media_loader/ts-sam-media-loader.js``) keeps working
regardless of when the node was instantiated.
"""

from __future__ import annotations

__all__ = [
    "LOG_PREFIX",
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "_log_info",
    "_log_warning",
    "_log_error",
    "_resolve_media_path",
    "_classify_media",
    "_load_image_tensor",
    "_load_video_tensor",
    "_video_meta",
    "_first_frame_base64",
    "_extract_video_audio",
    "_empty_audio",
    "_normalize_path",
    "_working_file_signature",
]

import asyncio
import base64
import hashlib
import io as _io
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import folder_paths
import numpy as np
import torch
from aiohttp import web

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow ships with ComfyUI
    Image = None  # noqa: N816

try:
    import server
except Exception:
    server = None

# SAM3 preview imports are deferred — comfy.sd / comfy.model_management can
# only be imported once ComfyUI core has finished initialising. ``importlib``
# avoids a circular import at module load time and lets the helpers fail
# gracefully if ComfyUI itself is unavailable (e.g. running unit tests).
try:
    import comfy.model_management as _comfy_model_management
except Exception:
    _comfy_model_management = None

try:
    import comfy.sd as _comfy_sd
except Exception:
    _comfy_sd = None

try:
    import comfy.utils as _comfy_utils
except Exception:
    _comfy_utils = None


LOGGER = logging.getLogger("comfyui_timesaver.ts_sam_media_loader")
LOG_PREFIX = "[TS SAM Media Loader]"

ROUTE_BASE = "/ts_sam_media_loader"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg"}
PREVIEW_LONG_EDGE = 1024
MAX_UPLOAD_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB hard cap


def _log_info(message: str) -> None:
    LOGGER.info("%s %s", LOG_PREFIX, message)


def _log_warning(message: str) -> None:
    LOGGER.warning("%s %s", LOG_PREFIX, message)


def _log_error(message: str) -> None:
    LOGGER.error("%s %s", LOG_PREFIX, message)


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


def _normalize_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")


def _input_root() -> Path:
    return Path(folder_paths.get_input_directory())


def _upload_root() -> Path:
    """Uploads land directly in ComfyUI's input/ folder so they are picked up
    by every node that browses the input directory (no nested subfolder)."""
    root = _input_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _allowed_view_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    try:
        roots.append(_input_root().resolve())
    except (OSError, ValueError):
        pass
    try:
        roots.append(Path(folder_paths.get_output_directory()).resolve())
    except (OSError, ValueError, AttributeError):
        pass
    try:
        roots.append(Path(folder_paths.get_temp_directory()).resolve())
    except (OSError, ValueError, AttributeError):
        pass
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


def _classify_media(filepath: str) -> str:
    suffix = Path(filepath).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    return ""


def _resolve_media_path(source_path: str) -> str:
    """Resolve an annotated or absolute path to an absolute filepath."""
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


def _to_input_annotation(filepath: Path) -> str:
    """Convert an input-directory file path to ComfyUI's annotated widget value."""
    input_dir = _input_root().resolve()
    resolved = filepath.resolve()
    try:
        relative = resolved.relative_to(input_dir)
    except ValueError:
        return _normalize_path(str(resolved))
    return f"{_normalize_path(str(relative))} [input]"


def _safe_filename(name: str) -> str:
    """Strip dangerous bits from a user-supplied filename."""
    base = Path(str(name or "")).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    base = base.strip("._-") or "media"
    if len(base) > 96:
        stem, _, suffix = base.rpartition(".")
        base = f"{stem[:80]}.{suffix}" if suffix else stem[:96]
    return base


def _target_upload_path(filename: str) -> Path:
    """Resolve the destination path for an upload. Always the plain
    ``input/<safe_filename>`` — collisions are handled by the upload route
    (it reuses the existing file instead of saving a duplicate)."""
    safe = _safe_filename(filename)
    return _upload_root() / safe


def _working_file_signature(filepath: str) -> str:
    if not filepath or not os.path.isfile(filepath):
        return ""
    stat = os.stat(filepath)
    payload = f"{_normalize_path(filepath)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Tensor loaders
# ---------------------------------------------------------------------------


def _load_image_tensor(filepath: str) -> torch.Tensor:
    """Load an image as ComfyUI IMAGE tensor ``[1, H, W, 3]`` float32 in [0, 1]."""
    if Image is None:
        raise RuntimeError(f"{LOG_PREFIX} Pillow is required to load images.")
    with Image.open(filepath) as handle:
        rgb = handle.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    if array.ndim != 3:
        raise RuntimeError(f"{LOG_PREFIX} Unsupported image shape: {array.shape}")
    return torch.from_numpy(array).unsqueeze(0).contiguous()


def _cv2():
    """Lazy import: cv2 is optional and only needed for video frames."""
    import cv2

    return cv2


def _open_video_capture(filepath: str):
    cv2 = _cv2()
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError(f"{LOG_PREFIX} Could not open video: {filepath}")
    return cap


def _video_meta(filepath: str) -> dict:
    """Return ``{width, height, frame_count, fps}`` for the supplied video."""
    cv2 = _cv2()
    cap = _open_video_capture(filepath)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return {
        "width": int(max(0, width)),
        "height": int(max(0, height)),
        "frame_count": int(max(0, frame_count)),
        "fps": float(fps if fps == fps and fps > 0 else 0.0),
    }


def _first_video_frame_rgb(filepath: str) -> "Image.Image | None":
    """Return the first decodable frame of ``filepath`` as a PIL RGB image."""
    if Image is None:
        return None
    cv2 = _cv2()
    cap = _open_video_capture(filepath)
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return Image.fromarray(rgb)


def _load_video_tensor(
    filepath: str,
    max_frames: int = 0,
    frame_stride: int = 1,
    progress_callback=None,
) -> torch.Tensor:
    """Decode a video into ComfyUI IMAGE tensor ``[N, H, W, 3]`` float32 in [0, 1].

    Args:
        filepath: Absolute path to the video file.
        max_frames: Hard cap on the number of returned frames (0 == unlimited).
        frame_stride: Take every N-th frame (1 == every frame).
        progress_callback: Optional ``callable(kept_count, total_estimated)``
            invoked after every kept frame. Used by the node's execute() to
            drive a ComfyUI ProgressBar; safe to omit for direct calls.
    """
    cv2 = _cv2()
    stride = max(1, int(frame_stride or 1))
    limit = max(0, int(max_frames or 0))
    cap = _open_video_capture(filepath)

    # Pre-compute the number of frames we will actually keep so the caller's
    # progress bar can show "kept/N" rather than indeterminate progress.
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    except Exception:
        total_frames = 0
    if total_frames > 0:
        estimated_kept = (total_frames + stride - 1) // stride
    else:
        estimated_kept = 0
    if limit > 0:
        estimated_kept = (
            min(estimated_kept, limit) if estimated_kept > 0 else limit
        )

    frames: list[np.ndarray] = []
    try:
        index = 0
        kept = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if (index % stride) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
                kept += 1
                if progress_callback is not None:
                    try:
                        progress_callback(kept, max(estimated_kept, kept))
                    except Exception:
                        # Progress is best-effort; never block decoding.
                        pass
                if limit > 0 and kept >= limit:
                    break
            index += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"{LOG_PREFIX} Failed to decode any frame from video: {filepath}")
    # Final tick so the UI bar reaches 100% even when the container's
    # frame-count metadata was missing or wrong (e.g. some webm muxes).
    if progress_callback is not None:
        try:
            progress_callback(len(frames), len(frames))
        except Exception:
            pass
    array = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(array).contiguous()


def _empty_audio() -> dict:
    """Return a ComfyUI-shaped empty AUDIO payload."""
    return {
        "waveform": torch.zeros((1, 1, 1024), dtype=torch.float32),
        "sample_rate": 44100,
    }


def _extract_video_audio(filepath: str) -> dict:
    """Decode the full audio track of a media file as a ComfyUI AUDIO dict.

    Returns ``_empty_audio()`` for media without an audio stream (or when
    ffmpeg-based decoding is unavailable). Reuses the ffmpeg pipeline from
    ``nodes/audio/loader/_audio_helpers.py`` so codec handling stays in one
    place; the import is deferred so this module loads cleanly even when the
    audio package's optional deps are missing.
    """
    if not filepath or not os.path.isfile(filepath):
        return _empty_audio()
    try:
        from ...audio.loader._audio_helpers import (
            _decode_audio_segment,
            _probe_media,
        )
    except Exception as exc:
        _log_warning(f"Audio helpers unavailable, returning silent audio: {exc}")
        return _empty_audio()
    try:
        metadata = _probe_media(filepath)
    except Exception as exc:
        # Most common case: ``No audio stream found`` for muted clips. Demote
        # to debug — it's expected for plenty of inputs.
        LOGGER.debug("%s No audio track in '%s': %s", LOG_PREFIX, filepath, exc)
        return _empty_audio()
    try:
        waveform, sample_rate = _decode_audio_segment(metadata, 0.0, None)
    except Exception as exc:
        _log_warning(f"Audio decode failed for '{filepath}': {exc}")
        return _empty_audio()
    return {
        "waveform": waveform.unsqueeze(0).contiguous(),
        "sample_rate": int(sample_rate),
    }


# ---------------------------------------------------------------------------
# Preview helpers
# ---------------------------------------------------------------------------


def _downscale_for_preview(image: "Image.Image") -> "Image.Image":
    if Image is None:
        return image
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= PREVIEW_LONG_EDGE:
        return image
    scale = PREVIEW_LONG_EDGE / float(long_edge)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.LANCZOS)


def _image_to_base64_jpeg(image: "Image.Image", quality: int = 88) -> str:
    if Image is None:
        return ""
    preview = _downscale_for_preview(image)
    buffer = _io.BytesIO()
    preview.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _first_frame_base64(filepath: str, media_type: str) -> dict:
    """Return ``{width, height, frame_count, fps, first_frame_b64}`` for preview."""
    if not filepath or not os.path.isfile(filepath):
        return {"width": 0, "height": 0, "frame_count": 0, "fps": 0.0, "first_frame_b64": ""}
    if media_type == "image":
        if Image is None:
            raise RuntimeError(f"{LOG_PREFIX} Pillow is required to read images.")
        with Image.open(filepath) as handle:
            rgb = handle.convert("RGB")
            width, height = rgb.size
            b64 = _image_to_base64_jpeg(rgb)
        return {
            "width": int(width),
            "height": int(height),
            "frame_count": 1,
            "fps": 0.0,
            "first_frame_b64": b64,
        }
    if media_type == "video":
        meta = _video_meta(filepath)
        frame = _first_video_frame_rgb(filepath)
        b64 = _image_to_base64_jpeg(frame) if frame is not None else ""
        # If cv2 reports 0×0 (some containers), use the decoded frame size.
        if frame is not None and (meta["width"] == 0 or meta["height"] == 0):
            meta["width"], meta["height"] = frame.size
        meta["first_frame_b64"] = b64
        return meta
    return {"width": 0, "height": 0, "frame_count": 0, "fps": 0.0, "first_frame_b64": ""}


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@_register_post(f"{ROUTE_BASE}/upload")
async def ts_sam_media_loader_upload(request: web.Request) -> web.StreamResponse:
    """Receive a multipart upload, store under ``input/ts_sam_media``,
    return annotated path plus first-frame preview payload."""
    try:
        reader = await request.multipart()
    except Exception as exc:
        return web.json_response({"error": f"Invalid multipart payload: {exc}"}, status=400)

    saved_path: Path | None = None
    filename: str = ""
    total_bytes = 0
    reused_existing = False
    try:
        async for part in reader:
            if part is None:
                break
            if part.name != "file":
                await part.release()
                continue
            filename = _safe_filename(part.filename or f"upload_{uuid.uuid4().hex[:8]}")
            media_type = _classify_media(filename)
            if not media_type:
                return web.json_response(
                    {"error": f"Unsupported media type: {filename}"}, status=400
                )
            target = _target_upload_path(filename)
            if target.is_file():
                # Same filename already in input/ — reuse it instead of
                # creating a duplicate. The user explicitly asked for this
                # idempotent behaviour (drag-and-dropping the same file twice
                # should not pollute the input folder with copies).
                await part.release()
                saved_path = target
                total_bytes = target.stat().st_size
                reused_existing = True
                break
            tmp = target.with_suffix(target.suffix + ".part")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as handle:
                while True:
                    chunk = await part.read_chunk(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        handle.close()
                        tmp.unlink(missing_ok=True)
                        return web.json_response(
                            {"error": "Upload exceeds the 1 GB limit."}, status=413
                        )
                    handle.write(chunk)
            os.replace(tmp, target)
            saved_path = target
            break
    except Exception as exc:
        if saved_path is not None and saved_path.exists() and not reused_existing:
            try:
                saved_path.unlink()
            except OSError:
                pass
        _log_warning(f"Upload failed: {exc}")
        return web.json_response({"error": f"Upload failed: {exc}"}, status=500)

    if saved_path is None:
        return web.json_response({"error": "Missing 'file' part in upload."}, status=400)

    media_type = _classify_media(saved_path.name)
    try:
        preview = _first_frame_base64(str(saved_path), media_type)
    except Exception as exc:
        _log_warning(f"Preview extraction failed for '{saved_path}': {exc}")
        preview = {"width": 0, "height": 0, "frame_count": 0, "fps": 0.0, "first_frame_b64": ""}

    annotated = _to_input_annotation(saved_path)
    response = {
        "ok": True,
        "source_path": annotated,
        "filename": saved_path.name,
        "media_type": media_type,
        "size_bytes": int(total_bytes),
        "reused_existing": reused_existing,
        **preview,
    }
    _log_info(
        f"{'Reused existing' if reused_existing else 'Uploaded'} '{saved_path.name}' "
        f"({media_type}, {preview.get('width')}x{preview.get('height')}, "
        f"frames={preview.get('frame_count')}, bytes={total_bytes})."
    )
    return web.json_response(response)


@_register_post(f"{ROUTE_BASE}/probe")
async def ts_sam_media_loader_probe(request: web.Request) -> web.StreamResponse:
    """Return preview payload for a previously-uploaded source path.

    Used when a workflow is restored from JSON: the JS widget calls this with
    the persisted ``source_path`` so the first frame appears without forcing a
    fresh upload.
    """
    try:
        payload = await request.json()
    except Exception as exc:
        return web.json_response({"error": f"Invalid JSON body: {exc}"}, status=400)
    source_path = str(payload.get("source_path") or "").strip()
    if not source_path:
        return web.json_response({"error": "Missing source_path."}, status=400)
    resolved = _resolve_media_path(source_path)
    if not resolved or not os.path.isfile(resolved):
        return web.json_response({"error": "Source file not found."}, status=404)
    path_obj = Path(resolved)
    if not _is_inside_allowed_root(path_obj):
        return web.json_response({"error": "Source path is outside allowed roots."}, status=403)
    media_type = _classify_media(resolved)
    if not media_type:
        return web.json_response({"error": "Unsupported media type."}, status=400)
    try:
        preview = _first_frame_base64(resolved, media_type)
    except Exception as exc:
        _log_warning(f"Probe failed for '{resolved}': {exc}")
        return web.json_response({"error": str(exc)}, status=500)
    annotated = _to_input_annotation(path_obj) if path_obj.is_absolute() else source_path
    return web.json_response(
        {
            "ok": True,
            "source_path": annotated,
            "filename": path_obj.name,
            "media_type": media_type,
            **preview,
        }
    )


@_register_get(f"{ROUTE_BASE}/preview")
async def ts_sam_media_loader_preview(request: web.Request) -> web.StreamResponse:
    """Return the first frame of ``source_path`` as JPEG bytes (no JSON wrap)."""
    source_path = request.query.get("source_path", "").strip()
    if not source_path:
        return web.Response(status=400, text="Missing source_path.")
    resolved = _resolve_media_path(source_path)
    if not resolved or not os.path.isfile(resolved):
        return web.Response(status=404, text="Source not found.")
    if not _is_inside_allowed_root(Path(resolved)):
        return web.Response(status=403, text="Path is outside allowed roots.")
    media_type = _classify_media(resolved)
    if not media_type:
        return web.Response(status=400, text="Unsupported media type.")
    try:
        if media_type == "image":
            if Image is None:
                return web.Response(status=500, text="Pillow unavailable.")
            with Image.open(resolved) as handle:
                rgb = _downscale_for_preview(handle.convert("RGB"))
                buffer = _io.BytesIO()
                rgb.save(buffer, format="JPEG", quality=88, optimize=True)
        else:
            frame = _first_video_frame_rgb(resolved)
            if frame is None:
                return web.Response(status=500, text="Could not decode first frame.")
            preview = _downscale_for_preview(frame)
            buffer = _io.BytesIO()
            preview.save(buffer, format="JPEG", quality=88, optimize=True)
    except Exception as exc:
        _log_warning(f"Preview render failed for '{resolved}': {exc}")
        return web.Response(status=500, text=str(exc))
    return web.Response(
        body=buffer.getvalue(),
        content_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=60"},
    )


# ---------------------------------------------------------------------------
# SAM3 preview model (singleton) + /preview_mask route
# ---------------------------------------------------------------------------


def _resolve_checkpoint_path(name: str) -> str:
    """Resolve a SAM3 checkpoint filename to an absolute path.

    SAM3 ships as a regular ComfyUI checkpoint (``supported_models.SAM3``), so
    we try the standard ``checkpoints`` / ``diffusion_models`` folders that
    every loader uses.
    """
    if not name:
        return ""
    for folder in ("checkpoints", "diffusion_models", "unet"):
        try:
            path = folder_paths.get_full_path(folder, name)
        except Exception:
            path = None
        if path and os.path.isfile(path):
            return path
    # As a last resort accept an absolute or input-relative path.
    if os.path.isfile(name):
        return name
    return ""


class _patched_prompt_server:
    """Temporarily set ``server.PromptServer.instance.last_prompt_id`` / ``last_node_id``.

    ``comfy.utils.ProgressBar.update`` calls into a hook that reads these
    attributes; when we drive SAM3 from a custom aiohttp route they are unset
    and the call raises ``AttributeError``. We give them a recognisable
    placeholder for the duration of the inference call.
    """

    PROMPT_ID = "ts_sam_media_loader_preview"
    NODE_ID = "ts_sam_media_loader_preview"

    def __init__(self) -> None:
        self._instance = None
        self._prev_prompt = None
        self._prev_node = None
        self._prev_client = None
        self._had_prompt = False
        self._had_node = False
        self._had_client = False

    def __enter__(self) -> "_patched_prompt_server":
        if server is None:
            return self
        try:
            self._instance = server.PromptServer.instance
        except Exception:
            self._instance = None
            return self
        inst = self._instance
        self._had_prompt = hasattr(inst, "last_prompt_id")
        self._had_node = hasattr(inst, "last_node_id")
        self._had_client = hasattr(inst, "client_id")
        if self._had_prompt:
            self._prev_prompt = getattr(inst, "last_prompt_id")
        if self._had_node:
            self._prev_node = getattr(inst, "last_node_id")
        if self._had_client:
            self._prev_client = getattr(inst, "client_id")
        try:
            inst.last_prompt_id = self.PROMPT_ID
            inst.last_node_id = self.NODE_ID
            # ``client_id=None`` makes ProgressBar's hook skip ``send_sync``,
            # so the preview run never pushes a fake progress message into
            # the user's websocket stream.
            inst.client_id = None
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        inst = self._instance
        if inst is None:
            return
        try:
            if self._had_prompt:
                inst.last_prompt_id = self._prev_prompt
            else:
                try:
                    delattr(inst, "last_prompt_id")
                except AttributeError:
                    pass
            if self._had_node:
                inst.last_node_id = self._prev_node
            else:
                try:
                    delattr(inst, "last_node_id")
                except AttributeError:
                    pass
            if self._had_client:
                inst.client_id = self._prev_client
        except Exception:
            pass


class _Sam3PreviewModel:
    """Thread-safe singleton wrapping a SAM3 checkpoint for preview inference.

    Keyed by checkpoint filename; reloading swaps the cached model. Status is
    polled by the frontend so the user gets a "loading" indicator during the
    first call (which can take ~10 seconds for a multi-GB SAM3 checkpoint).
    """

    _instance: "_Sam3PreviewModel | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Any = None
        self._loaded_name: str = ""
        self._load_lock = threading.Lock()
        self._status: dict[str, Any] = {
            "loaded": False,
            "loading": False,
            "error": "",
            "message": "Connect a SAM3 model loader to enable preview.",
            "checkpoint": "",
            "updated_at": time.time(),
        }
        self._status_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "_Sam3PreviewModel":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_status(self) -> dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    def _set_status(self, **fields: Any) -> None:
        with self._status_lock:
            self._status.update(fields)
            self._status["updated_at"] = time.time()

    def ensure_loaded(self, checkpoint_name: str) -> Any:
        if _comfy_sd is None or _comfy_model_management is None:
            raise RuntimeError(
                f"{LOG_PREFIX} ComfyUI core is unavailable; preview disabled."
            )
        if not checkpoint_name:
            raise RuntimeError(f"{LOG_PREFIX} No SAM3 checkpoint specified.")
        with self._load_lock:
            if self._model is not None and self._loaded_name == checkpoint_name:
                return self._model
            # Release the old one before loading a new SAM3 checkpoint —
            # SAM3 weights are several GB and we don't want two in VRAM.
            self._unload_locked()
            ckpt_path = _resolve_checkpoint_path(checkpoint_name)
            if not ckpt_path:
                raise RuntimeError(
                    f"{LOG_PREFIX} Checkpoint not found in checkpoints/diffusion_models: "
                    f"{checkpoint_name}"
                )
            self._set_status(
                loading=True,
                loaded=False,
                error="",
                message=f"Loading {checkpoint_name}...",
                checkpoint=checkpoint_name,
            )
            try:
                # SAM3 has both UNet (detector) and CLIP weights inside one
                # checkpoint, so ``load_checkpoint_guess_config`` is the right
                # entry point. We only need the model patcher (index 0).
                result = _comfy_sd.load_checkpoint_guess_config(
                    ckpt_path,
                    output_vae=False,
                    output_clip=False,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                )
                model_patcher = result[0] if isinstance(result, tuple) else result
                if model_patcher is None:
                    raise RuntimeError(
                        f"{LOG_PREFIX} load_checkpoint_guess_config returned no model."
                    )
            except Exception as exc:
                self._set_status(
                    loading=False,
                    loaded=False,
                    error=str(exc),
                    message=f"Failed to load {checkpoint_name}: {exc}",
                )
                raise
            self._model = model_patcher
            self._loaded_name = checkpoint_name
            self._set_status(
                loading=False,
                loaded=True,
                error="",
                message=f"Loaded {checkpoint_name}.",
                checkpoint=checkpoint_name,
            )
            _log_info(f"SAM3 preview model loaded: {checkpoint_name}")
            return self._model

    def _unload_locked(self) -> None:
        """Caller must hold ``self._load_lock``."""
        if self._model is None:
            return
        try:
            if _comfy_model_management is not None and hasattr(
                _comfy_model_management, "unload_model_clones"
            ):
                _comfy_model_management.unload_model_clones(self._model)
        except Exception as exc:
            _log_warning(f"Failed to unload previous SAM3 preview model: {exc}")
        self._model = None
        self._loaded_name = ""

    def segment_first_frame(
        self,
        image_pil: "Image.Image",
        positive_pts: list[dict[str, float]],
        negative_pts: list[dict[str, float]],
        refine_iterations: int = 2,
    ) -> "np.ndarray":
        """Run the native ``SAM3_Detect`` node on a single PIL frame.

        Delegating to the upstream node keeps the preview in lock-step with
        what the workflow will compute: SAM3.1 multiplex's multi-candidate
        ``forward_segment`` output is collapsed correctly (per-object union),
        and any future fix in the core node propagates automatically.

        Returns a uint8 [H, W] binary mask (0 or 255) sized to the input frame.
        """
        if self._model is None:
            raise RuntimeError(f"{LOG_PREFIX} SAM3 preview model not loaded.")

        # Empty prompt set short-circuits to an empty mask; the core node
        # would still tolerate it but we save a forward pass.
        if not positive_pts and not negative_pts:
            return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)

        try:
            from comfy_extras.nodes_sam3 import SAM3_Detect
        except Exception as exc:
            raise RuntimeError(
                f"{LOG_PREFIX} comfy_extras.nodes_sam3 unavailable: {exc}"
            ) from exc

        rgb = image_pil.convert("RGB")
        image_np = np.asarray(rgb, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).contiguous()  # [1, H, W, 3]

        # SAM3_Detect parses positive_coords / negative_coords with json.loads
        # and expects {"x": int, "y": int} in pixel coordinates of `image`.
        def _coords_json(points: list[dict[str, float]]) -> str:
            payload: list[dict[str, int]] = []
            for point in points:
                try:
                    x = int(round(float(point["x"])))
                    y = int(round(float(point["y"])))
                except (TypeError, ValueError, KeyError):
                    continue
                payload.append({"x": x, "y": y})
            return json.dumps(payload, separators=(",", ":"))

        positive_json = _coords_json(positive_pts)
        negative_json = _coords_json(negative_pts)

        # SAM3_Detect uses comfy.utils.ProgressBar, whose hook reads
        # ``server.PromptServer.instance.last_prompt_id`` / ``last_node_id``.
        # These attributes are only populated while the main executor is
        # processing a queued prompt — since the preview route runs outside
        # that loop, we provide a no-op placeholder so the hook doesn't raise
        # ``AttributeError``. The placeholder values are overwritten by the
        # next real prompt the user queues, so this is non-destructive.
        with _patched_prompt_server():
            with torch.no_grad():
                output = SAM3_Detect.execute(
                    model=self._model,
                    image=image_tensor,
                    conditioning=None,
                    bboxes=None,
                    positive_coords=positive_json,
                    negative_coords=negative_json,
                    threshold=0.5,
                    refine_iterations=int(refine_iterations or 2),
                    individual_masks=False,
                )

        if output is None or output.result is None or len(output.result) == 0:
            return np.zeros((rgb.height, rgb.width), dtype=np.uint8)

        mask_tensor = output.result[0]
        # SAM3_Detect emits MASK with shape [B, H, W] in [0, 1].
        if not torch.is_tensor(mask_tensor):
            return np.zeros((rgb.height, rgb.width), dtype=np.uint8)
        if mask_tensor.ndim == 3:
            mask_2d = mask_tensor[0]
        elif mask_tensor.ndim == 2:
            mask_2d = mask_tensor
        else:
            return np.zeros((rgb.height, rgb.width), dtype=np.uint8)
        binary = (mask_2d > 0.5).to(torch.uint8) * 255
        return binary.detach().cpu().numpy()


# Per-checkpoint asyncio.Lock so concurrent /preview_mask calls don't trample
# each other while sharing the same SAM3 model on GPU.
_PREVIEW_LOCKS: dict[str, asyncio.Lock] = {}


def _get_preview_lock(checkpoint: str) -> asyncio.Lock:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", checkpoint or "default")
    lock = _PREVIEW_LOCKS.get(safe)
    if lock is None:
        lock = asyncio.Lock()
        _PREVIEW_LOCKS[safe] = lock
    return lock


def _mask_to_base64_png(mask: "np.ndarray") -> str:
    """Encode a binary mask as a blue-tinted RGBA PNG.

    The frontend expects an RGBA bitmap that it can blit straight on top of
    the canvas with ``globalAlpha`` — without any compositing tricks. Storing
    the SAM3 mask as the alpha channel of an RGBA image is the simplest way:
    the browser preserves the alpha exactly (a grayscale PNG would be lifted
    to alpha=255 everywhere, which is why an earlier version produced a
    full-canvas overlay).
    """
    if Image is None or mask is None or mask.size == 0:
        return ""
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.ndim != 2:
        return ""
    h, w = arr.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    # Tint matches the frontend ts-sml overlay colour. The exact RGB value
    # only matters when the mask alpha > 0; transparent pixels are skipped
    # by the browser regardless of RGB.
    rgba[..., 0] = 74
    rgba[..., 1] = 144
    rgba[..., 2] = 226
    rgba[..., 3] = arr
    img = Image.fromarray(rgba, mode="RGBA")
    buffer = _io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_preview_points(raw: Any) -> list[dict[str, float]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        out.append({"x": x, "y": y})
    return out


@_register_get(f"{ROUTE_BASE}/sam3_status")
async def ts_sam_media_loader_sam3_status(_: web.Request) -> web.StreamResponse:
    return web.json_response(_Sam3PreviewModel.instance().get_status())


@_register_post(f"{ROUTE_BASE}/preview_mask")
async def ts_sam_media_loader_preview_mask(request: web.Request) -> web.StreamResponse:
    """Run SAM3 segmentation on the first frame and return a mask PNG (base64).

    Body:
        {
            "source_path": "<annotated path>",
            "checkpoint_name": "<sam3.safetensors>",
            "positive": [{"x": float, "y": float}, ...],
            "negative": [{"x": float, "y": float}, ...],
            "refine_iterations": 2          # optional
        }
    """
    try:
        payload = await request.json()
    except Exception as exc:
        return web.json_response({"error": f"Invalid JSON body: {exc}"}, status=400)

    source_path = str(payload.get("source_path") or "").strip()
    checkpoint_name = str(payload.get("checkpoint_name") or "").strip()
    refine_iterations = int(payload.get("refine_iterations") or 2)
    positive = _parse_preview_points(payload.get("positive"))
    negative = _parse_preview_points(payload.get("negative"))

    if not source_path:
        return web.json_response({"error": "Missing source_path."}, status=400)
    if not checkpoint_name:
        return web.json_response(
            {
                "error": "No SAM3 checkpoint connected. Connect a SAM3 model loader to the 'model' input.",
                "needs_model": True,
            },
            status=400,
        )

    resolved = _resolve_media_path(source_path)
    if not resolved or not os.path.isfile(resolved):
        return web.json_response({"error": "Source file not found."}, status=404)
    if not _is_inside_allowed_root(Path(resolved)):
        return web.json_response({"error": "Source path is outside allowed roots."}, status=403)

    media_type = _classify_media(resolved)
    if not media_type:
        return web.json_response({"error": "Unsupported media type."}, status=400)
    if Image is None:
        return web.json_response({"error": "Pillow unavailable on the server."}, status=500)

    # Decode first frame as PIL (full resolution — SAM3 normalises internally).
    try:
        if media_type == "image":
            with Image.open(resolved) as handle:
                first_frame = handle.convert("RGB").copy()
        else:
            frame = _first_video_frame_rgb(resolved)
            if frame is None:
                return web.json_response({"error": "Could not decode first frame."}, status=500)
            first_frame = frame
    except Exception as exc:
        return web.json_response({"error": f"Frame decode failed: {exc}"}, status=500)

    preview = _Sam3PreviewModel.instance()
    lock = _get_preview_lock(checkpoint_name)
    async with lock:
        try:
            preview.ensure_loaded(checkpoint_name)
        except Exception as exc:
            _log_warning(f"SAM3 model load failed: {exc}")
            return web.json_response({"error": str(exc)}, status=500)
        try:
            mask = await asyncio.to_thread(
                preview.segment_first_frame,
                first_frame,
                positive,
                negative,
                refine_iterations,
            )
        except Exception as exc:
            _log_warning(f"SAM3 preview inference failed: {exc}")
            return web.json_response({"error": str(exc)}, status=500)

    mask_b64 = _mask_to_base64_png(mask)
    pos_count = int((mask > 0).sum()) if mask is not None and mask.size > 0 else 0
    return web.json_response(
        {
            "ok": True,
            "checkpoint": checkpoint_name,
            "media_type": media_type,
            "width": int(first_frame.width),
            "height": int(first_frame.height),
            "mask_b64": mask_b64,
            "positive_pixels": pos_count,
        }
    )
