"""Shared helpers and aiohttp routes for TS_LamaCleanup.

Private module: not registered as a public node by the loader.
HTTP routes register at import time so the frontend (ts-lama-cleanup.js)
keeps working regardless of when the node was instantiated.
"""

from __future__ import annotations

__all__ = [
    "LOG_PREFIX",
    "_log_info",
    "_log_warning",
    "_log_error",
    "_resolve_image_path",
    "_load_image_tensor",
    "_get_uploadable_image_options",
    "_session_working_path",
    "_working_file_signature",
    "MODEL_DOWNLOAD_URL",
    "MODEL_FILENAME",
    "MODEL_LEGACY_FILENAME",
]

import asyncio
import base64
import hashlib
import io
import logging
import os
import shutil
import threading
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import folder_paths
import numpy as np
import torch
from aiohttp import web

try:
    from PIL import Image
except ImportError:
    Image = None  # noqa: N816

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None

try:
    import server
except Exception:
    server = None

try:
    from safetensors.torch import load_file as _safetensors_load_file
except ImportError:
    _safetensors_load_file = None

from ._lama_arch import build_lama_inpainter


LOGGER = logging.getLogger("comfyui_timesaver.ts_lama_cleanup")
LOG_PREFIX = "[TS Lama Cleanup]"

MODEL_FILENAME = "big-lama.safetensors"
MODEL_LEGACY_FILENAME = "big-lama.pt"
MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/hfmaster/models-moved/resolve/main/lama/big-lama.safetensors"
)
MODEL_FOLDER_NAME = "lama"

NODE_ROOT = Path(__file__).resolve().parents[2]
WORKING_SUBDIR = "ts_lama_cleanup"
OUTPUT_SUBDIR = "lama-cleanup"
OUTPUT_NAME_TAG = "lama-cleanup"
SUPPORTED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
}


def _register_model_folder() -> None:
    """Register `models/lama` so ComfyUI's folder_paths is aware of it."""
    try:
        base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
        base.mkdir(parents=True, exist_ok=True)
        if hasattr(folder_paths, "add_model_folder_path"):
            folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(base))
    except Exception as exc:
        _LOGGER_INIT_WARNING = f"Failed to register '{MODEL_FOLDER_NAME}' model folder: {exc}"
        LOGGER.warning("%s %s", LOG_PREFIX, _LOGGER_INIT_WARNING)


_register_model_folder()


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


def _temp_root() -> Path:
    try:
        base = folder_paths.get_temp_directory()
    except Exception:
        base = str(NODE_ROOT / ".cache" / "temp_fallback")
    root = Path(base) / WORKING_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _output_root() -> Path:
    base = folder_paths.get_output_directory()
    return Path(base)


def _allowed_view_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    try:
        roots.append(Path(folder_paths.get_input_directory()).resolve())
    except (OSError, ValueError):
        pass
    try:
        roots.append(Path(folder_paths.get_output_directory()).resolve())
    except (OSError, ValueError, AttributeError):
        pass
    try:
        roots.append(_temp_root().resolve())
    except OSError:
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


def _resolve_image_path(source_path: str) -> str:
    """Resolve an annotated or absolute path to an absolute image filepath."""
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


def _get_uploadable_image_options() -> list[str]:
    """Return input-directory image files compatible with ComfyUI's native upload combo."""
    input_dir = folder_paths.get_input_directory()
    try:
        files, _ = folder_paths.recursive_search(input_dir)
    except OSError as exc:
        _log_warning(f"Failed to scan input directory for image files: {exc}")
        return []
    return sorted(folder_paths.filter_files_content_types(files, ["image"]))


def _safe_session_id(session_id: str) -> str:
    """Sanitize a client-supplied session id to a filesystem-safe slug."""
    cleaned = "".join(ch for ch in str(session_id or "") if ch.isalnum() or ch in "-_")
    if not cleaned:
        cleaned = uuid.uuid4().hex
    return cleaned[:64]


def _session_working_path(session_id: str) -> Path:
    """Canonical (latest) working path for a session — used by execute()."""
    return _temp_root() / f"{_safe_session_id(session_id)}.png"


def _versioned_working_path(session_id: str, suffix: str) -> Path:
    """Generate a unique working file path per inpaint/seed call so each step
    in the user's edit history has its own file (needed for Undo/Redo)."""
    safe = _safe_session_id(session_id)
    nanos = time.time_ns()
    tag = "".join(ch for ch in str(suffix or "") if ch.isalnum() or ch in "-_")[:24] or "step"
    return _temp_root() / f"{safe}_{tag}_{nanos:020d}.png"


# Per-session asyncio.Lock used to serialise /inpaint requests for the same
# editing session. Without it, double-clicks on Save or rapid mouseup events
# producing back-to-back inpaint calls can race and produce confused state.
_session_locks: dict[str, asyncio.Lock] = {}


def _get_session_lock(session_id: str) -> asyncio.Lock:
    safe = _safe_session_id(session_id)
    lock = _session_locks.get(safe)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[safe] = lock
    return lock


def _cleanup_session_files(session_id: str, keep: set[str] | None = None) -> int:
    """Delete every ``{session_id}_*.png`` file in temp_root, except those in
    ``keep``. Used to garbage-collect old working files when the user starts a
    fresh session, hits Reset, or trims their undo history."""
    safe = _safe_session_id(session_id)
    if not safe:
        return 0
    keep_normalized: set[str] = set()
    for raw in keep or set():
        if not raw:
            continue
        try:
            keep_normalized.add(_normalize_path(str(Path(raw).resolve(strict=False))))
        except OSError as exc:
            LOGGER.debug("%s Could not resolve keep-path '%s': %s", LOG_PREFIX, raw, exc)
            continue
    removed = 0
    try:
        root = _temp_root()
    except Exception as exc:
        _log_warning(f"Could not resolve temp root for cleanup: {exc}")
        return 0
    for path in root.glob(f"{safe}_*.png"):
        if not path.is_file():
            continue
        try:
            resolved = _normalize_path(str(path.resolve(strict=False)))
        except Exception:
            resolved = _normalize_path(str(path))
        if resolved in keep_normalized:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            _log_warning(f"Failed to remove session file '{path}': {exc}")
    return removed


def _cleanup_session_paths(session_id: str, paths: list[str]) -> int:
    """Delete the supplied list of working paths if they belong to this
    session (filename starts with ``{session_id}_``) and live inside the
    allowed temp root. Used by the JS history-limit eviction path."""
    safe = _safe_session_id(session_id)
    if not safe or not paths:
        return 0
    removed = 0
    for raw in paths:
        if not raw:
            continue
        resolved = _resolve_image_path(str(raw))
        if not resolved:
            continue
        try:
            path_obj = Path(resolved)
        except (TypeError, ValueError) as exc:
            LOGGER.debug("%s Could not build Path('%s'): %s", LOG_PREFIX, resolved, exc)
            continue
        if not path_obj.is_file():
            continue
        if not _is_inside_allowed_root(path_obj):
            continue
        if not path_obj.name.startswith(f"{safe}_"):
            continue
        try:
            path_obj.unlink()
            removed += 1
        except OSError as exc:
            _log_warning(f"Failed to remove '{path_obj}': {exc}")
    return removed


def _working_file_signature(filepath: str) -> str:
    if not filepath or not os.path.isfile(filepath):
        return ""
    stat = os.stat(filepath)
    payload = f"{_normalize_path(filepath)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_image_tensor(filepath: str) -> torch.Tensor:
    """Load an image file as a ComfyUI IMAGE tensor [1, H, W, 3] float32 in [0, 1]."""
    if Image is None:
        raise RuntimeError("Pillow is required to load images.")
    if not filepath or not os.path.isfile(filepath):
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    with Image.open(filepath) as handle:
        image = handle.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    tensor = torch.from_numpy(array).unsqueeze(0).contiguous()
    return tensor


def _save_image_atomic(image: "Image.Image", destination: Path) -> None:
    """Write a PIL image to disk atomically (write to .tmp then rename)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        image.save(tmp_path, format="PNG", optimize=False)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _to_input_annotation(filepath: Path) -> str:
    """Convert an input-directory file path to ComfyUI's annotated widget value."""
    input_dir = Path(folder_paths.get_input_directory()).resolve()
    resolved_path = filepath.resolve()
    try:
        relative_path = resolved_path.relative_to(input_dir)
    except ValueError:
        return _normalize_path(str(resolved_path))
    return f"{_normalize_path(str(relative_path))} [input]"


# -----------------------------------------------------------------------------
# LaMa model loader (singleton with thread-safe lazy init)
# -----------------------------------------------------------------------------


class _LamaModel:
    """Thread-safe singleton wrapper around the LaMa JIT-scripted model."""

    _instance: "_LamaModel | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Any = None
        self._device: torch.device | None = None
        self._load_lock = threading.Lock()
        self._status: dict[str, Any] = {
            "loaded": False,
            "loading": False,
            "error": "",
            "message": "Model not loaded yet.",
            "updated_at": time.time(),
        }
        self._status_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "_LamaModel":
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

    def _resolve_device(self) -> torch.device:
        if self._device is not None:
            return self._device
        if model_management is None:
            raise RuntimeError(
                f"{LOG_PREFIX} comfy.model_management is unavailable; "
                "LaMa inpaint requires a running ComfyUI runtime."
            )
        device = model_management.get_torch_device()
        self._device = device
        return device

    def _model_cache_dir(self) -> Path:
        """Resolve the canonical models/lama directory used for downloads."""
        try:
            base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
        except Exception:
            base = NODE_ROOT / ".cache" / MODEL_FOLDER_NAME
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _find_existing_model(self) -> Path | None:
        """Look for big-lama.safetensors (preferred) or big-lama.pt in any models/lama folder."""
        candidates: list[Path] = []
        cache_dir = self._model_cache_dir()
        for name in (MODEL_FILENAME, MODEL_LEGACY_FILENAME):
            candidates.append(cache_dir / name)
        try:
            registered = folder_paths.get_folder_paths(MODEL_FOLDER_NAME)
        except Exception:
            registered = []
        for folder in registered or []:
            for name in (MODEL_FILENAME, MODEL_LEGACY_FILENAME):
                candidates.append(Path(folder) / name)
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size > 1024:
                return candidate
        return None

    def _download_model(self) -> Path:
        existing = self._find_existing_model()
        if existing is not None:
            self._set_status(
                loading=True,
                message=f"Found cached model at {existing}. Loading...",
                error="",
            )
            return existing
        cache_dir = self._model_cache_dir()
        local_path = cache_dir / MODEL_FILENAME
        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        self._set_status(
            loading=True,
            message=f"Downloading LaMa model from {MODEL_DOWNLOAD_URL} to {cache_dir}...",
            error="",
        )
        request = Request(MODEL_DOWNLOAD_URL, headers={"User-Agent": "comfyui-timesaver"})
        try:
            with urlopen(request, timeout=300) as response, tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp_path.replace(local_path)
        except URLError as exc:
            raise RuntimeError(
                f"Failed to download LaMa model from {MODEL_DOWNLOAD_URL}: {exc}"
            ) from exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return local_path

    def _load_state_dict(self, model_path: Path) -> dict[str, torch.Tensor]:
        suffix = model_path.suffix.lower()
        if suffix == ".safetensors":
            if _safetensors_load_file is None:
                raise RuntimeError(
                    f"{LOG_PREFIX} safetensors is required to load {model_path.name}. "
                    "Install it via `pip install safetensors`."
                )
            return _safetensors_load_file(str(model_path), device="cpu")
        if suffix == ".pt":
            ts_module = torch.jit.load(str(model_path), map_location="cpu")
            return {key: tensor.detach().clone() for key, tensor in ts_module.state_dict().items()}
        raise RuntimeError(f"{LOG_PREFIX} unsupported LaMa weights format: {model_path}")

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                model_path = self._download_model()
                self._set_status(
                    loading=True,
                    message="Loading LaMa model into memory...",
                    error="",
                )
                state_dict = self._load_state_dict(model_path)
                model = build_lama_inpainter(state_dict)
                device = self._resolve_device()
                model.to(device)
                self._model = model
                self._set_status(
                    loaded=True,
                    loading=False,
                    message=f"LaMa model loaded on {device}.",
                    error="",
                )
                _log_info(f"LaMa model loaded on {device} from {model_path}")
            except Exception as exc:
                self._set_status(
                    loaded=False,
                    loading=False,
                    error=str(exc),
                    message=f"Failed to load LaMa model: {exc}",
                )
                _log_error(f"Failed to load LaMa model: {exc}")
                raise

    def inpaint(self, image_rgb: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
        """Run LaMa inference on a single RGB image with a binary mask.

        Args:
            image_rgb: HxWx3 uint8 array.
            mask_gray: HxW uint8 array, 255 where to inpaint, 0 to keep.
        Returns:
            HxWx3 uint8 array with inpainted regions.
        """
        if Image is None:
            raise RuntimeError("Pillow is required for inpainting.")
        self.ensure_loaded()
        device = self._resolve_device()
        height, width = image_rgb.shape[:2]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        if pad_h or pad_w:
            image_rgb = np.pad(
                image_rgb,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="reflect",
            )
            mask_gray = np.pad(
                mask_gray,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )
        image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
        mask_tensor = torch.from_numpy(mask_gray.astype(np.float32) / 255.0)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).contiguous()
        image_tensor = image_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        with torch.no_grad():
            output = self._model(image_tensor, mask_tensor)
        output = output.detach()
        if output.ndim == 4:
            output = output[0]
        output = output.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        output = (output * 255.0).round().astype(np.uint8)
        if pad_h or pad_w:
            output = output[:height, :width, :]
        return output


# -----------------------------------------------------------------------------
# Inpainting orchestration: bbox crop + resize + composite
# -----------------------------------------------------------------------------


def _decode_mask_data_url(data_url: str) -> np.ndarray:
    """Decode a base64 data-URL PNG into a HxW uint8 mask (alpha or luminance)."""
    if Image is None:
        raise RuntimeError("Pillow is required to decode mask images.")
    if not data_url:
        raise ValueError("Empty mask payload.")
    if data_url.startswith("data:"):
        _, _, encoded = data_url.partition(",")
    else:
        encoded = data_url
    raw = base64.b64decode(encoded)
    with Image.open(io.BytesIO(raw)) as handle:
        if handle.mode in ("RGBA", "LA"):
            mask_image = handle.split()[-1]
        else:
            mask_image = handle.convert("L")
        return np.asarray(mask_image, dtype=np.uint8)


def _compute_bbox(mask: np.ndarray, threshold: int = 8) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) bbox of non-zero mask pixels, or None if empty."""
    coords = np.argwhere(mask > threshold)
    if coords.size == 0:
        return None
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return int(x0), int(y0), int(x1), int(y1)


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(width, x1 + padding)
    y1 = min(height, y1 + padding)
    return x0, y0, x1, y1


def _resize_with_pil(array: np.ndarray, size: tuple[int, int], resample: int) -> np.ndarray:
    """Resize using PIL preserving channel layout."""
    if Image is None:
        raise RuntimeError("Pillow is required for resize operations.")
    if array.ndim == 2:
        pil_image = Image.fromarray(array, mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:
        pil_image = Image.fromarray(array, mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape for resize: {array.shape}")
    resized = pil_image.resize(size, resample=resample)
    return np.asarray(resized)


def _process_inpaint(
    image_rgb: np.ndarray,
    mask_full: np.ndarray,
    max_resolution: int,
    mask_padding: int,
    feather: int,
) -> np.ndarray:
    """High-level inpaint flow: bbox crop, optional resize, LaMa, composite back.

    Args:
        image_rgb: HxWx3 uint8.
        mask_full: HxW uint8 (255 = inpaint).
        max_resolution: max longest side fed to LaMa.
        mask_padding: pixels of context around mask bbox.
        feather: pixels for soft mask blending on composite.
    Returns:
        HxWx3 uint8 with inpainted region composited.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for inpainting.")
    if image_rgb.dtype != np.uint8:
        raise ValueError("Image must be uint8 RGB.")
    if mask_full.dtype != np.uint8:
        mask_full = mask_full.astype(np.uint8)
    if mask_full.shape[:2] != image_rgb.shape[:2]:
        mask_full = _resize_with_pil(mask_full, (image_rgb.shape[1], image_rgb.shape[0]), Image.NEAREST)
    bbox = _compute_bbox(mask_full)
    if bbox is None:
        return image_rgb
    height, width = image_rgb.shape[:2]
    padding = max(0, int(mask_padding))
    x0, y0, x1, y1 = _expand_bbox(bbox, width, height, padding)
    crop_image = image_rgb[y0:y1, x0:x1, :].copy()
    crop_mask = mask_full[y0:y1, x0:x1].copy()
    crop_h, crop_w = crop_image.shape[:2]
    longest = max(crop_h, crop_w)
    target_max = max(64, int(max_resolution))
    if longest > target_max:
        scale = target_max / float(longest)
        new_w = max(8, int(round(crop_w * scale)))
        new_h = max(8, int(round(crop_h * scale)))
        small_image = _resize_with_pil(crop_image, (new_w, new_h), Image.LANCZOS)
        small_mask = _resize_with_pil(crop_mask, (new_w, new_h), Image.BILINEAR)
    else:
        small_image = crop_image
        small_mask = crop_mask
    small_mask_bin = (small_mask > 8).astype(np.uint8) * 255
    inpainted_small = _LamaModel.instance().inpaint(small_image, small_mask_bin)
    if inpainted_small.shape[:2] != (crop_h, crop_w):
        inpainted_crop = _resize_with_pil(inpainted_small, (crop_w, crop_h), Image.LANCZOS)
    else:
        inpainted_crop = inpainted_small
    soft_mask = crop_mask.astype(np.float32) / 255.0
    if feather > 0:
        try:
            import cv2

            kernel_size = max(3, int(feather) * 2 + 1)
            soft_mask = cv2.GaussianBlur(soft_mask, (kernel_size, kernel_size), 0)
        except (ImportError, Exception) as exc:
            LOGGER.debug("%s Feather GaussianBlur skipped: %s", LOG_PREFIX, exc)
    soft_mask = np.clip(soft_mask, 0.0, 1.0)[..., None]
    blended = inpainted_crop.astype(np.float32) * soft_mask + crop_image.astype(np.float32) * (1.0 - soft_mask)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    result = image_rgb.copy()
    result[y0:y1, x0:x1, :] = blended
    return result


def _read_image_rgb(filepath: str) -> np.ndarray:
    """Load an image file as HxWx3 uint8 RGB array."""
    if Image is None:
        raise RuntimeError("Pillow is required to read images.")
    if not filepath or not os.path.isfile(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")
    with Image.open(filepath) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


# -----------------------------------------------------------------------------
# aiohttp routes
# -----------------------------------------------------------------------------


@_register_get("/ts_lama_cleanup/view")
async def ts_lama_cleanup_view(request: web.Request) -> web.StreamResponse:
    filepath = _resolve_image_path(urllib.parse.unquote(request.query.get("filepath", "")))
    if not filepath or not os.path.isfile(filepath):
        return web.Response(status=404)
    if not _is_inside_allowed_root(Path(filepath)):
        _log_warning(f"Rejected view request outside allowed roots: {filepath}")
        return web.Response(status=404)
    try:
        return web.FileResponse(
            filepath,
            headers={
                "Cache-Control": "no-store, max-age=0",
            },
        )
    except Exception:
        return web.Response(status=500)


@_register_get("/ts_lama_cleanup/model_status")
async def ts_lama_cleanup_model_status(_: web.Request) -> web.StreamResponse:
    status = _LamaModel.instance().get_status()
    return web.json_response(status)


@_register_post("/ts_lama_cleanup/inpaint")
async def ts_lama_cleanup_inpaint(request: web.Request) -> web.StreamResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        return web.json_response({"error": f"Invalid JSON body: {exc}"}, status=400)

    session_id = _safe_session_id(payload.get("session_id", ""))
    source_path_raw = str(payload.get("source_path", "") or "")
    working_path_raw = str(payload.get("working_path", "") or "")
    mask_data_url = str(payload.get("mask", "") or "")
    max_resolution = int(payload.get("max_resolution", 512) or 512)
    mask_padding = int(payload.get("mask_padding", 64) or 0)
    feather = int(payload.get("feather", 4) or 0)

    if not mask_data_url:
        return web.json_response({"error": "Missing mask payload."}, status=400)

    current_path = ""
    if working_path_raw:
        candidate = _resolve_image_path(working_path_raw)
        if candidate and os.path.isfile(candidate) and _is_inside_allowed_root(Path(candidate)):
            current_path = candidate
    if not current_path:
        candidate = _resolve_image_path(source_path_raw)
        if candidate and os.path.isfile(candidate):
            if not _is_inside_allowed_root(Path(candidate)):
                return web.json_response({"error": "Source path is outside allowed roots."}, status=403)
            current_path = candidate
    if not current_path:
        return web.json_response({"error": "No source image available."}, status=400)

    # Per-session lock: if the user fires off a second inpaint while the first
    # is still running, the new request waits instead of racing on the same
    # working file (or returning a result based on a stale snapshot).
    lock = _get_session_lock(session_id)
    try:
        async with lock:
            result = await asyncio.to_thread(
                _run_inpaint_job,
                session_id,
                current_path,
                mask_data_url,
                max_resolution,
                mask_padding,
                feather,
            )
        return web.json_response(result)
    except Exception as exc:
        _log_error(f"Inpaint job failed: {exc}")
        return web.json_response({"error": str(exc)}, status=500)


def _run_inpaint_job(
    session_id: str,
    current_path: str,
    mask_data_url: str,
    max_resolution: int,
    mask_padding: int,
    feather: int,
) -> dict[str, Any]:
    image_rgb = _read_image_rgb(current_path)
    mask = _decode_mask_data_url(mask_data_url)
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = _resize_with_pil(mask, (image_rgb.shape[1], image_rgb.shape[0]), Image.NEAREST)
    output_array = _process_inpaint(
        image_rgb=image_rgb,
        mask_full=mask,
        max_resolution=max_resolution,
        mask_padding=mask_padding,
        feather=feather,
    )
    output_image = Image.fromarray(output_array, mode="RGB")
    working_path = _versioned_working_path(session_id, "edit")
    _save_image_atomic(output_image, working_path)
    return {
        "working_path": _normalize_path(str(working_path)),
        "width": int(output_array.shape[1]),
        "height": int(output_array.shape[0]),
        "signature": _working_file_signature(str(working_path)),
    }


@_register_post("/ts_lama_cleanup/save")
async def ts_lama_cleanup_save(request: web.Request) -> web.StreamResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        return web.json_response({"error": f"Invalid JSON body: {exc}"}, status=400)

    working_path_raw = str(payload.get("working_path", "") or "")
    suggested_name = str(payload.get("filename", "") or "")
    if not working_path_raw:
        return web.json_response({"error": "Missing working_path."}, status=400)

    working_path = _resolve_image_path(working_path_raw)
    if not working_path or not os.path.isfile(working_path):
        return web.json_response({"error": "Working file not found."}, status=404)
    if not _is_inside_allowed_root(Path(working_path)):
        return web.json_response({"error": "Working path is outside allowed roots."}, status=403)

    try:
        return await asyncio.to_thread(_save_to_output, working_path, suggested_name)
    except Exception as exc:
        _log_error(f"Save job failed: {exc}")
        return web.json_response({"error": str(exc)}, status=500)


def _save_to_output(working_path: str, suggested_name: str) -> web.StreamResponse:
    # Save into ``output/<OUTPUT_SUBDIR>/<base>_<OUTPUT_NAME_TAG>_<timestamp>.png``
    # so cleaned images are grouped in one folder and easy to spot in the
    # ComfyUI output viewer.
    output_root = _output_root() / OUTPUT_SUBDIR
    output_root.mkdir(parents=True, exist_ok=True)
    fallback_stem = OUTPUT_NAME_TAG.replace("-", "_") or "lama_cleanup"
    base_stem = Path(suggested_name).stem if suggested_name else fallback_stem
    base_stem = "".join(ch for ch in base_stem if ch.isalnum() or ch in "-_") or fallback_stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    counter = 0
    while True:
        suffix = f"_{counter:03d}" if counter else ""
        candidate = output_root / f"{base_stem}_{OUTPUT_NAME_TAG}_{timestamp}{suffix}.png"
        if not candidate.exists():
            break
        counter += 1
        if counter > 9999:
            return web.json_response({"error": "Failed to allocate output filename."}, status=500)
    shutil.copy2(working_path, candidate)
    _log_info(f"Saved cleaned image to: {candidate}")
    return web.json_response(
        {
            "saved_path": _normalize_path(str(candidate)),
            "filename": candidate.name,
            "subfolder": OUTPUT_SUBDIR,
            "type": "output",
        }
    )


@_register_post("/ts_lama_cleanup/reset")
async def ts_lama_cleanup_reset(request: web.Request) -> web.StreamResponse:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    session_id = _safe_session_id(payload.get("session_id", ""))
    # Reset wipes EVERY working file for this session so the user starts
    # totally clean. The frontend immediately re-seeds from the source image.
    removed = await asyncio.to_thread(_cleanup_session_files, session_id, set())
    if removed:
        _log_info(f"Reset session '{session_id}': removed {removed} working file(s)")
    return web.json_response({"ok": True, "removed": int(removed)})


@_register_post("/ts_lama_cleanup/seed")
async def ts_lama_cleanup_seed(request: web.Request) -> web.StreamResponse:
    """Copy the source image to the working slot so that the working file always exists."""
    try:
        payload = await request.json()
    except Exception as exc:
        return web.json_response({"error": f"Invalid JSON body: {exc}"}, status=400)
    session_id = _safe_session_id(payload.get("session_id", ""))
    source_path_raw = str(payload.get("source_path", "") or "")
    if not source_path_raw:
        return web.json_response({"error": "Missing source_path."}, status=400)
    source_path = _resolve_image_path(source_path_raw)
    if not source_path or not os.path.isfile(source_path):
        return web.json_response({"error": "Source image not found."}, status=404)
    if not _is_inside_allowed_root(Path(source_path)):
        return web.json_response({"error": "Source path is outside allowed roots."}, status=403)
    working_path = _versioned_working_path(session_id, "seed")
    try:
        await asyncio.to_thread(_seed_working_copy, source_path, working_path)
    except Exception as exc:
        _log_error(f"Failed to seed working file: {exc}")
        return web.json_response({"error": str(exc)}, status=500)
    # Drop every previous working file for this session — the new seed is the
    # only point in history. Skip the seed itself via the ``keep`` set.
    removed = await asyncio.to_thread(
        _cleanup_session_files, session_id, {str(working_path)}
    )
    if removed:
        _log_info(f"Seed for session '{session_id}': cleaned {removed} stale file(s)")
    return web.json_response(
        {
            "working_path": _normalize_path(str(working_path)),
            "signature": _working_file_signature(str(working_path)),
        }
    )


@_register_post("/ts_lama_cleanup/cleanup_paths")
async def ts_lama_cleanup_cleanup_paths(request: web.Request) -> web.StreamResponse:
    """Delete a list of working files belonging to the caller's session.
    Used by the JS history-cap eviction to free disk as edits accumulate."""
    try:
        payload = await request.json()
    except Exception as exc:
        return web.json_response({"error": f"Invalid JSON body: {exc}"}, status=400)
    session_id = _safe_session_id(payload.get("session_id", ""))
    paths = payload.get("paths", [])
    if not isinstance(paths, list):
        return web.json_response({"error": "paths must be a list of strings."}, status=400)
    removed = await asyncio.to_thread(_cleanup_session_paths, session_id, [str(p) for p in paths if p])
    return web.json_response({"removed": int(removed)})


def _seed_working_copy(source_path: str, working_path: Path) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required to seed working file.")
    working_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as handle:
        rgb = handle.convert("RGB")
        _save_image_atomic(rgb, working_path)
