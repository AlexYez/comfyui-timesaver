import gc
import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import folder_paths
from comfy.utils import ProgressBar, load_torch_file

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_video_depth")
LOG_PREFIX = "[TS Video Depth]"


# --- VideoDepthAnything model import (multi-path for legacy installs) ---
try:
    from ..video_depth_anything.video_depth import VideoDepthAnything, _preprocess_frames_gpu
except ImportError:
    try:
        from video_depth_anything.video_depth import VideoDepthAnything, _preprocess_frames_gpu
    except ImportError as e:
        logger.error("%s CRITICAL IMPORT ERROR: Could not import VideoDepthAnything model: %s", LOG_PREFIX, e)
        VideoDepthAnything = None
        _preprocess_frames_gpu = None


# Matplotlib colormaps are pulled lazily — `matplotlib` adds ~150 ms to
# ComfyUI startup if imported eagerly. We materialise a 256-entry RGB LUT the
# first time each colormap is requested and reuse it forever after.
def _resolve_cmap(name: str):
    """Resolve a colormap by name across matplotlib versions.

    Newer (>=3.7) matplotlib exposes a top-level `matplotlib.colormaps`
    registry; older releases require `matplotlib.cm.get_cmap` (deprecated
    but still functional).
    """
    import matplotlib
    if hasattr(matplotlib, "colormaps"):
        try:
            return matplotlib.colormaps[name]
        except KeyError as exc:
            raise ValueError(f"Unknown colormap {name!r}") from exc
    import matplotlib.cm as cm
    return cm.get_cmap(name)


# ---------------------------------------------------------------------------
# Module-level mutable state
# ---------------------------------------------------------------------------
# ComfyUI V3 locks the registered node class, so kwargs like `cls.X = ...` raise
# AttributeError on first execute. Keep the model patcher, LUT cache and the
# blue-noise tile on a private module-level object instead.
class _VideoDepthState:
    model_patcher = None
    loaded_filename: str | None = None
    colormap_luts: dict[str, torch.Tensor] = {}
    blue_noise_tile: torch.Tensor | None = None
    cudnn_benchmark_set: bool = False


_state = _VideoDepthState()


# ---------------------------------------------------------------------------
# GPU helpers: dithering, colormap, normalization, edge-aware upscale
# ---------------------------------------------------------------------------

_BAYER_8 = torch.tensor([
    [0, 32,  8, 40,  2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44,  4, 36, 14, 46,  6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43,  1, 33,  9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47,  7, 39, 13, 45,  5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=torch.float32)


def _get_bayer_tile(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Centered Bayer 8×8 matrix in [-0.5, 0.5), cached per device/dtype."""
    tile = _BAYER_8.to(device=device, dtype=dtype) / 64.0 - 0.5
    return tile


def _apply_dither(depth: torch.Tensor, strength: float, pattern: str) -> torch.Tensor:
    """Add sub-LSB noise to a [0,1] depth map to reduce 8-bit banding.

    Args:
        depth: (N, H, W) float in [0, 1] on any device.
        strength: amplitude in [0, 0.02] roughly matching the legacy widget.
        pattern: "white" (TPDF triangular noise, audio antibanding standard)
            or "bayer" (8×8 ordered, deterministic, banding-free).

    Notes on "white" mode: we use Triangular-Probability-Density-Function noise
    (sum of two uniform samples) rather than plain uniform. TPDF eliminates the
    "dead-zone" near quantizer boundaries that uniform PDF leaves visible, and
    is the gold standard for sub-LSB dithering. Same range as uniform but
    distributed differently.
    """
    if strength <= 0.0:
        return depth
    if pattern == "bayer":
        tile = _get_bayer_tile(depth.device, depth.dtype)
        n, h, w = depth.shape
        rep_h = math.ceil(h / 8)
        rep_w = math.ceil(w / 8)
        full = tile.repeat(rep_h, rep_w)[:h, :w]
        # Bayer tile sits in [-0.5, +0.484]; scale by 2×strength so the
        # effective amplitude matches `±strength` (one full quant level when
        # strength=1/256), giving meaningful break-up of the residual banding.
        return (depth + full.unsqueeze(0) * (strength * 2.0)).clamp_(0.0, 1.0)
    # "white" — TPDF (sum of two uniforms) is symmetric around zero with
    # peak-to-peak ~= strength, and unlike uniform it kills banding fully.
    n1 = torch.rand_like(depth)
    n2 = torch.rand_like(depth)
    noise = (n1 + n2 - 1.0) * strength
    return (depth + noise).clamp_(0.0, 1.0)


def _get_colormap_lut(name: str, device: torch.device) -> torch.Tensor | None:
    """Return a (256, 3) float32 RGB LUT for the given colormap name, or None
    for the "gray" pass-through case. Cached per name on the device that was
    requested first; subsequent lookups simply move the tensor."""
    if name == "gray":
        return None
    cached = _state.colormap_luts.get(name)
    if cached is not None:
        return cached.to(device, non_blocking=True)
    try:
        cmap = _resolve_cmap(name)
    except (KeyError, ValueError):
        logger.warning("%s Colormap '%s' not found. Falling back to 'gray'.", LOG_PREFIX, name)
        return None
    samples = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    rgba = cmap(samples)
    lut = torch.from_numpy(rgba[:, :3].astype(np.float32))
    _state.colormap_luts[name] = lut
    return lut.to(device, non_blocking=True)


def _apply_colormap(depth: torch.Tensor, lut: torch.Tensor | None) -> torch.Tensor:
    """Apply colormap LUT to a (N, H, W) depth map.

    Returns (N, H, W, 3) float32 in [0, 1]. For "gray" (lut=None) the depth is
    broadcast across the channel axis.

    Uses **bilinear interpolation** between the two adjacent LUT entries
    instead of `.round()`-based nearest-neighbour. The legacy nearest-neighbour
    path quantised the colormap to exactly 256 distinct colours per axis,
    which produced very visible banding on smooth depth gradients. Bilinear
    interpolation lifts that to effectively continuous output while still
    using a tiny 256-entry table.
    """
    if lut is None:
        return depth.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

    idx_float = depth.clamp(0.0, 1.0) * 255.0
    lo_idx = idx_float.floor().long().clamp(0, 254)
    hi_idx = lo_idx + 1
    frac = (idx_float - lo_idx.float()).unsqueeze(-1).clamp(0.0, 1.0)
    lo_color = lut[lo_idx]
    hi_color = lut[hi_idx]
    return lo_color + frac * (hi_color - lo_color)


def _compute_global_normalization(
    depth: torch.Tensor, mode: str, chunk_size: int = 64
) -> tuple[float, float]:
    """Find (lo, hi) for global normalization. Streams chunks through GPU so
    that a 4K × 30 s clip never has to be fully resident at once.

    Returns (lo, hi); caller does ``(depth - lo) / (hi - lo + eps)``.
    """
    n = depth.shape[0]
    if mode == "percentile":
        # Robust [1%, 99%] computed over a uniform spatial subsample of every
        # frame. Avoids materialising the full sort on 4K data.
        target_samples_per_frame = 4096
        h, w = depth.shape[1], depth.shape[2]
        stride_h = max(1, h // int(math.sqrt(target_samples_per_frame)))
        stride_w = max(1, w // int(math.sqrt(target_samples_per_frame)))
        samples_list = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sub = depth[start:end, ::stride_h, ::stride_w].reshape(-1).float()
            samples_list.append(sub.cpu())
        samples = torch.cat(samples_list, dim=0)
        lo = float(torch.quantile(samples, 0.01).item())
        hi = float(torch.quantile(samples, 0.99).item())
        if hi <= lo:
            hi = lo + 1e-3
        return lo, hi
    # "minmax" — legacy semantics, but streamed to keep peak memory bounded.
    lo = math.inf
    hi = -math.inf
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sub = depth[start:end]
        lo = min(lo, float(sub.min().item()))
        hi = max(hi, float(sub.max().item()))
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _resize_depth_chunk(
    depth_chunk: torch.Tensor,
    target_h: int,
    target_w: int,
    method: str,
) -> torch.Tensor:
    """(N, H, W) → (N, target_h, target_w) using F.interpolate.

    "Lanczos4" maps to bicubic+antialias (PyTorch lacks a native Lanczos kernel;
    bicubic-with-antialias is the closest analogue and what kornia uses too).
    """
    inp = depth_chunk.unsqueeze(1)
    mode_map = {
        "Linear": ("bilinear", False),
        "Cubic": ("bicubic", True),
        "Lanczos4": ("bicubic", True),
    }
    mode, antialias = mode_map.get(method, ("bicubic", True))
    out = F.interpolate(inp, size=(target_h, target_w), mode=mode, align_corners=False, antialias=antialias)
    return out.squeeze(1)


def _median_blur_chunk(depth_chunk: torch.Tensor) -> torch.Tensor:
    """3-channel median blur (k=3) implemented via sort over an unfolded window.

    Kernel size 3 keeps the GPU memory cost manageable: unfold to 9 samples per
    pixel, sort, take the median. For (N, 1, H, W) at 1280×720 this is ~50 MB
    per frame at fp32 — well within budget when called chunk-by-chunk.
    """
    pad = 1
    x = depth_chunk.unsqueeze(1)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    patches = F.unfold(x, kernel_size=3)  # (N, 9, H*W)
    sorted_patches, _ = torch.sort(patches, dim=1)
    med = sorted_patches[:, 4, :]
    return med.view(depth_chunk.shape[0], depth_chunk.shape[1], depth_chunk.shape[2])


def _bilateral_blur_chunk(
    depth_chunk: torch.Tensor,
    rgb_chunk: torch.Tensor,
    sigma_color: float = 0.1,
    sigma_space: float = 1.5,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Joint bilateral filter over a 5×5 window guided by ``rgb_chunk`` luma.

    Edge-preserving smoothing. ``depth_chunk`` is (N, H, W); ``rgb_chunk`` is
    (N, 3, H, W) at the same resolution. When ``rgb_chunk`` is None we degrade
    to a Gaussian blur (still preserves edges better than median for noise).
    """
    pad = kernel_size // 2
    n, h, w = depth_chunk.shape
    d = depth_chunk.unsqueeze(1)
    d_pad = F.pad(d, (pad, pad, pad, pad), mode="reflect")
    d_patches = F.unfold(d_pad, kernel_size=kernel_size)  # (N, K*K, H*W)

    # Spatial Gaussian weights (kernel x kernel).
    coords = torch.arange(kernel_size, device=depth_chunk.device, dtype=depth_chunk.dtype) - pad
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    sp = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma_space * sigma_space))
    sp = sp.view(1, kernel_size * kernel_size, 1)

    if rgb_chunk is not None:
        luma = (0.299 * rgb_chunk[:, 0] + 0.587 * rgb_chunk[:, 1] + 0.114 * rgb_chunk[:, 2]).unsqueeze(1)
        luma_pad = F.pad(luma, (pad, pad, pad, pad), mode="reflect")
        luma_patches = F.unfold(luma_pad, kernel_size=kernel_size)  # (N, K*K, H*W)
        center = luma_patches[:, kernel_size * kernel_size // 2:kernel_size * kernel_size // 2 + 1]
        diff = luma_patches - center
        rng = torch.exp(-(diff * diff) / (2.0 * sigma_color * sigma_color))
    else:
        rng = torch.ones_like(d_patches)

    weights = sp * rng
    out = (weights * d_patches).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
    return out.view(n, h, w)


def _guided_upsample_chunk(
    depth_low: torch.Tensor,
    guide_high: torch.Tensor,
    target_h: int,
    target_w: int,
    radius: int = 4,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Fast guided-filter upsampling with a luma guide.

    He et al., 2010. Upsamples ``depth_low`` (N, H_low, W_low) to (target_h,
    target_w) using ``guide_high`` (N, 3, target_h, target_w) RGB as an
    edge-aware guide. Output preserves the silhouette of the RGB but takes its
    values from the depth map.

    The implementation is fully separable (box filters via avg_pool2d) so a
    1280×720 → 4K upsample stays under ~150 MB peak, regardless of N.
    """
    n = depth_low.shape[0]
    # Bicubic-upsample depth to target resolution first.
    depth_up = F.interpolate(depth_low.unsqueeze(1), size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
    # Luma guide.
    luma = (0.299 * guide_high[:, 0] + 0.587 * guide_high[:, 1] + 0.114 * guide_high[:, 2]).unsqueeze(1)

    box_k = 2 * radius + 1
    def _box(x):
        return F.avg_pool2d(x, kernel_size=box_k, stride=1, padding=radius, count_include_pad=False)

    mean_g = _box(luma)
    mean_p = _box(depth_up)
    corr_gg = _box(luma * luma)
    corr_gp = _box(luma * depth_up)
    var_g = corr_gg - mean_g * mean_g
    cov_gp = corr_gp - mean_g * mean_p
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    mean_a = _box(a)
    mean_b = _box(b)
    out = mean_a * luma + mean_b
    return out.squeeze(1).clamp_(0.0, 1.0)


# ---------------------------------------------------------------------------
# Model loading via ComfyUI ModelPatcher (cooperative VRAM management)
# ---------------------------------------------------------------------------

def _load_model_to_offload_cpu(model_filename: str, on_download_start=None):
    """Construct VideoDepthAnything, load weights, return the eval()-d model
    on CPU. Downloads from HuggingFace if the weights are missing.

    ``on_download_start`` is invoked right before triggering the HF download
    so the caller (the node) can surface UI feedback (log line + progress
    bar tick). It is NOT called when the file is already on disk.
    """
    download_path = os.path.join(folder_paths.models_dir, "videodepthanything")
    model_path = os.path.join(download_path, model_filename)
    if not os.path.exists(model_path):
        if on_download_start is not None:
            on_download_start(model_filename)
        logger.info("%s Downloading weights for %s from HuggingFace…", LOG_PREFIX, model_filename)
        os.makedirs(download_path, exist_ok=True)
        from huggingface_hub import snapshot_download
        repo_map = {"vits": "depth-anything/Video-Depth-Anything-Small", "vitl": "depth-anything/Video-Depth-Anything-Large"}
        model_key = next((key for key in repo_map if key in model_filename.lower()), None)
        if not model_key:
            raise ValueError(f"Cannot determine repository for model: {model_filename}.")
        snapshot_download(
            repo_id=repo_map[model_key],
            allow_patterns=[f"*{model_filename}*"],
            local_dir=download_path,
            local_dir_use_symlinks=False,
        )
        logger.info("%s Download complete: %s", LOG_PREFIX, model_filename)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    encoder_key = next((key for key in model_configs if key in model_filename.lower()), None)
    if not encoder_key:
        raise ValueError(f"Cannot determine model config for: {model_filename}")
    model = VideoDepthAnything(**model_configs[encoder_key])
    state_dict = load_torch_file(model_path, device="cpu")
    model.load_state_dict(state_dict)
    return model.eval()


def _ensure_patcher(model_filename: str, load_device: torch.device, on_download_start=None):
    """Return a ModelPatcher for the requested model, building one on demand
    and tearing down the previous one if the user switched files.

    ``on_download_start`` is forwarded to ``_load_model_to_offload_cpu`` and
    fires only when the weights have to be fetched from HuggingFace.
    """
    import comfy.model_patcher as model_patcher

    if _state.model_patcher is not None and _state.loaded_filename == model_filename:
        return _state.model_patcher, False

    if _state.model_patcher is not None:
        # User switched models — release the old patcher so its weights can
        # be freed by ComfyUI's normal eviction.
        try:
            _state.model_patcher.model.to("cpu")
        except Exception as exc:
            logger.warning("%s Could not offload previous patcher: %s", LOG_PREFIX, exc)
        _state.model_patcher = None

    offload_device = mm.unet_offload_device() if hasattr(mm, "unet_offload_device") else torch.device("cpu")
    model = _load_model_to_offload_cpu(model_filename, on_download_start=on_download_start)
    patcher = model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    _state.model_patcher = patcher
    _state.loaded_filename = model_filename
    return patcher, True


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

# How many frames to push through GPU postprocess at once. Tuned so a single
# chunk at 4K stays under ~1.5 GB peak (4K × 16 frames × 3 channels × 4 B).
_POSTPROCESS_CHUNK = 16


class _StagePBar:
    """ProgressBar adapter that maps inner [0..inner_total] progress onto a
    sub-range of an outer ProgressBar.

    The node displays a single 0..100 bar where each stage takes a slice:
    download → preprocess → inference → postprocess. Stage code keeps its
    natural counter (e.g. `infer_video_depth_torch` calls
    ``pbar.update(frame_step)`` per window); the adapter rescales every
    update into the outer bar's coordinate system. This way the user sees
    smooth, weighted progress across all phases instead of the bar jumping
    abruptly between stages.
    """

    def __init__(self, outer, base: int, span: int, inner_total: int):
        self.outer = outer
        self.base = int(base)
        self.span = max(0, int(span))
        self.inner_total = max(1, int(inner_total))
        self.inner = 0

    def _emit(self):
        ratio = min(self.inner / self.inner_total, 1.0)
        self.outer.update_absolute(self.base + int(round(self.span * ratio)))

    def update(self, n: int = 1):
        self.inner = min(self.inner + int(n), self.inner_total)
        self._emit()

    def update_absolute(self, value, total=None, **_kwargs):
        if total is not None:
            self.inner_total = max(1, int(total))
        self.inner = min(int(value), self.inner_total)
        self._emit()

    def finish(self):
        self.inner = self.inner_total
        self._emit()


# Total units of the outer progress bar (always 100 for predictable UI).
_PBAR_TOTAL = 100
# Weight of each stage within the 100-unit budget. Tuned to typical wall-clock
# on a 16 GB CUDA card: inference dominates, postprocess is non-trivial when
# edge_aware_upscale=True, preprocess is a thin slice. Download is given a
# non-zero slot so the bar starts moving immediately when weights are missing.
_PBAR_WEIGHT_DOWNLOAD = 5
_PBAR_WEIGHT_PREPROCESS = 5
_PBAR_WEIGHT_INFERENCE = 75
_PBAR_WEIGHT_POSTPROCESS = 15


class TS_VideoDepth(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        upscale_methods_list = ["Lanczos4", "Cubic", "Linear"]
        return IO.Schema(
            node_id="TS_VideoDepthNode",
            display_name="TS Video Depth",
            category="TS/Video",
            inputs=[
                IO.Image.Input(
                    "images",
                    tooltip=(
                        "Video frames as an IMAGE batch (N, H, W, 3) in 0..1. "
                        "Any aspect ratio; 16:9 is the model's sweet spot."
                    ),
                ),
                IO.Combo.Input(
                    "model_filename",
                    options=['video_depth_anything_vits.pth', 'video_depth_anything_vitl.pth'],
                    default='video_depth_anything_vitl.pth',
                    tooltip=(
                        "Video Depth Anything checkpoint. Downloaded on first use to "
                        "ComfyUI/models/videodepthanything.\n"
                        "• vits (~0.5 GB) — fast, less stable on fine detail.\n"
                        "• vitl (~1.5 GB, default) — best quality, ~2× slower. "
                        "Recommended for production."
                    ),
                ),
                IO.Int.Input(
                    "input_size",
                    default=518,
                    min=64,
                    max=4096,
                    step=2,
                    tooltip=(
                        "Internal resolution the transformer sees (DINOv2 patch size 14, "
                        "snapped automatically). Higher = more depth detail, more VRAM / time.\n"
                        "For 16:9 source (after max_res cap):\n"
                        "• 518 (default, native) — model trained at this size; ~480 K depth pixels.\n"
                        "• 644 — ~740 K depth pixels (+54% detail, +54% VRAM/time). Safe on ≥24 GB.\n"
                        "• 700 — ~872 K depth pixels (+82% detail, +82% VRAM). OOM risk on 16 GB.\n"
                        "• ≥770 — out-of-distribution for DINOv2 (quality can REGRESS).\n"
                        "If OOM, the node auto-retries on 518 → 392 → 280 → 168."
                    ),
                ),
                IO.Int.Input(
                    "max_res",
                    default=1280,
                    min=-1,
                    max=8192,
                    step=1,
                    tooltip=(
                        "Cap on the longer side of input frames before model preprocessing.\n"
                        "• Does NOT change depth detail — the model always resamples to input_size.\n"
                        "• -1 — no cap. Keeps full-resolution RGB as the guide for "
                        "edge_aware_upscale, giving the sharpest silhouettes on the output.\n"
                        "• 1280 (default) — downscales 4K to HD first; saves preprocess RAM and "
                        "speeds up resize, at the cost of a slightly softer edge-aware upscale.\n"
                        "Recommendation: -1 for 4K when edge_aware_upscale=True, 1280 otherwise."
                    ),
                ),
                IO.Combo.Input(
                    "precision",
                    options=['fp16', 'fp32'],
                    default='fp16',
                    tooltip=(
                        "Inference dtype.\n"
                        "• fp16 (default) — 2× faster, ~50% less VRAM. Required for vitl @ 4K "
                        "on a 16 GB card.\n"
                        "• fp32 — marginally cleaner gradients on smooth surfaces; doubles "
                        "VRAM, almost always triggers OOM on 4K. Use only on small inputs or "
                        "≥24 GB cards."
                    ),
                ),
                IO.Combo.Input(
                    "colormap",
                    options=['gray', 'inferno', 'viridis', 'plasma', 'magma', 'cividis'],
                    default='gray',
                    tooltip=(
                        "Output color mapping.\n"
                        "• gray (default) — raw normalized depth in all 3 channels. Use this "
                        "if the depth map feeds downstream nodes (ControlNet, 3D, etc).\n"
                        "• inferno / viridis / plasma / magma / cividis — perceptually uniform "
                        "matplotlib colormaps for visualization only. Bilinear LUT interpolation "
                        "removes 8-bit banding."
                    ),
                ),
                IO.Float.Input(
                    "dithering_strength",
                    default=0.005,
                    min=0.0,
                    max=0.016,
                    step=0.0001,
                    round=0.0001,
                    tooltip=(
                        "Sub-LSB noise added to the normalized depth before colormap, to break "
                        "up 8-bit banding when the result is saved as PNG/JPEG.\n"
                        "• 0 — no dither.\n"
                        "• 0.005 (default) — light, OK with bayer pattern + bilinear LUT.\n"
                        "• 0.016 (max) — aggressive, guaranteed banding-free on gray output.\n"
                        "If you still see bands, raise toward 0.016 and prefer dither_pattern=bayer."
                    ),
                ),
                IO.Boolean.Input(
                    "apply_median_blur",
                    default=True,
                    tooltip=(
                        "Legacy denoise toggle (kept for workflow compatibility). Used only "
                        "when denoise_method=auto: True → 3×3 median, False → none.\n"
                        "When denoise_method is set explicitly (bilateral / median / none), "
                        "this toggle is ignored."
                    ),
                ),
                IO.Combo.Input(
                    "upscale_algorithm",
                    options=upscale_methods_list,
                    default="Lanczos4",
                    tooltip=(
                        "Resampling kernel for upscaling the depth map back to the original "
                        "frame size. Used only when edge_aware_upscale=False.\n"
                        "• Lanczos4 (default) — sharpest details, slightly slower.\n"
                        "• Cubic — balanced.\n"
                        "• Linear — softest, fastest.\n"
                        "All implemented in PyTorch (Lanczos4 ≡ bicubic+antialias)."
                    ),
                ),
                # Optional quality controls. Defaults are tuned for visual
                # quality; the legacy widgets above (precision/apply_median_blur/
                # dithering_strength/upscale_algorithm) keep their original
                # defaults so existing workflows are bit-for-bit unaffected.
                IO.Combo.Input(
                    "normalization_mode",
                    options=["minmax", "percentile"],
                    default="percentile",
                    optional=True,
                    tooltip=(
                        "How to map raw depth onto [0..1].\n"
                        "• minmax — uses global min/max across the whole video. Simple, but "
                        "one outlier frame (object very close or far) can squash the contrast "
                        "of every other frame.\n"
                        "• percentile (default, quality) — robust 1%..99% range. Better contrast "
                        "and temporal stability on long clips. Slightly more memory (samples a "
                        "subset of pixels for quantile)."
                    ),
                ),
                IO.Combo.Input(
                    "denoise_method",
                    options=["auto", "none", "median", "bilateral"],
                    default="bilateral",
                    optional=True,
                    tooltip=(
                        "Spatial denoise applied at low-res depth (before upscale).\n"
                        "• auto — follow legacy apply_median_blur toggle.\n"
                        "• none — no denoise, maximum detail; may show grain on fine textures.\n"
                        "• median — 3×3 median, removes impulse noise, slightly blurs thin geometry.\n"
                        "• bilateral (default, quality) — edge-preserving 5×5; smooths surface "
                        "noise while keeping object silhouettes sharp."
                    ),
                ),
                IO.Combo.Input(
                    "dither_pattern",
                    options=["white", "bayer"],
                    default="bayer",
                    optional=True,
                    tooltip=(
                        "Dither distribution used by dithering_strength.\n"
                        "• white — TPDF (triangular) random noise, full antibanding standard, "
                        "but adds visible grain on flat surfaces.\n"
                        "• bayer (default, quality) — deterministic 8×8 ordered pattern. "
                        "Banding-free, no temporal flicker, no grain. Best paired with bilinear "
                        "colormap LUT (already enabled)."
                    ),
                ),
                IO.Boolean.Input(
                    "edge_aware_upscale",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Final upscale strategy.\n"
                        "• False — plain resampling via upscale_algorithm (Lanczos4 etc). Fastest.\n"
                        "• True (default, quality) — Fast Guided Filter using the input RGB as "
                        "edge guide. Silhouettes snap to real object boundaries; thin geometry "
                        "is preserved. Costs ~5-10% extra postprocess time. Combine with "
                        "max_res=-1 to keep the guide at full 4K for the sharpest result."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="image",
                    tooltip=(
                        "Depth map as IMAGE (N, H, W, 3) float in 0..1 at the original RGB "
                        "resolution. Same as a regular IMAGE — feed directly into ControlNet, "
                        "save nodes, etc."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        model_filename,
        input_size,
        max_res,
        precision,
        colormap,
        dithering_strength,
        apply_median_blur,
        upscale_algorithm,
        normalization_mode="percentile",
        denoise_method="bilateral",
        dither_pattern="bayer",
        edge_aware_upscale=True,
    ) -> IO.NodeOutput:
        if VideoDepthAnything is None:
            raise ImportError("[TS_VideoDepth] Node cannot run: VideoDepthAnything model class failed to import.")

        if not (isinstance(images, torch.Tensor) and images.ndim == 4):
            raise ValueError("[TS_VideoDepth] 'images' must be a 4D float tensor (N, H, W, 3).")

        n_frames, original_h, original_w = images.shape[0], images.shape[1], images.shape[2]
        load_device = mm.get_torch_device()

        # cudnn autotune pays off immediately for the fixed transformer shapes
        # we hit, and is a no-op on non-CUDA devices.
        if load_device.type == "cuda" and not _state.cudnn_benchmark_set:
            torch.backends.cudnn.benchmark = True
            _state.cudnn_benchmark_set = True

        # --- single outer progress bar covering all stages ---
        master_pbar = ProgressBar(_PBAR_TOTAL)
        outer_cursor = 0

        def _emit_outer(value: int):
            # Local helper used for the download stage and final pin to 100.
            master_pbar.update_absolute(min(value, _PBAR_TOTAL))

        # --- model load via ComfyUI's cooperative VRAM manager ---
        def _on_download_start(name: str):
            logger.info("%s Weights missing, downloading %s …", LOG_PREFIX, name)
            _emit_outer(_PBAR_WEIGHT_DOWNLOAD // 2)  # halfway through the download slot

        patcher, _was_just_built = _ensure_patcher(
            model_filename, load_device, on_download_start=_on_download_start,
        )
        outer_cursor += _PBAR_WEIGHT_DOWNLOAD
        _emit_outer(outer_cursor)

        # Estimate a realistic activation budget so ComfyUI evicts other models
        # if needed: backbone features + DPT path tensors + final upsample
        # dominate, all scale linearly with (T × H_in × W_in). The constant
        # was calibrated against vitl @ INFER_LEN=32 frames in fp16 on 4K-source
        # inputs (real measurement: ~6.5 GB peak at 518×924).
        if load_device.type == "cuda":
            est_bytes = max(
                int(2 * 1024 * 1024 * 1024),  # floor: 2 GiB
                int(images.shape[1] * images.shape[2] * 32 * 4),  # ~T×H×W×4
            )
            mm.free_memory(est_bytes, load_device)
        else:
            mm.free_memory(0, load_device)
        mm.load_model_gpu(patcher)
        model = patcher.model

        # --- resolve effective processing resolution ---
        # Apply legacy max_res cap (full-resolution preprocessing crashed VRAM
        # on 4K inputs in the old implementation; we keep the same widget).
        proc_h, proc_w = original_h, original_w
        if max_res > 0 and max(original_h, original_w) > max_res:
            scale = max_res / max(original_h, original_w)
            proc_h = int(original_h * scale)
            proc_w = int(original_w * scale)
            proc_h += proc_h % 2
            proc_w += proc_w % 2

        # 16:9 aspect-ratio guard from the original implementation.
        effective_input_size = input_size
        long_side = max(proc_h, proc_w)
        short_side = min(proc_h, proc_w)
        ratio = long_side / max(short_side, 1)
        if ratio > 1.78:
            effective_input_size = int(effective_input_size * 1.777 / ratio)
            effective_input_size = round(effective_input_size / 14) * 14
        # multiple-of-14 constraint required by DINOv2 patch_size=14.
        if effective_input_size % 14 != 0:
            adjusted = (effective_input_size // 14) * 14
            if adjusted == 0 and effective_input_size > 0:
                adjusted = 14
            if adjusted != effective_input_size:
                logger.info(
                    "%s Adjusted input_size %s → %s (multiple of 14).",
                    LOG_PREFIX, effective_input_size, adjusted,
                )
            effective_input_size = adjusted
        effective_input_size = max(14, effective_input_size)

        # Compute resize target (lower_bound, keep aspect ratio, multiple of 14).
        scale_h = effective_input_size / proc_h
        scale_w = effective_input_size / proc_w
        scale = max(scale_h, scale_w)
        target_h = max(14, round((proc_h * scale) / 14) * 14)
        target_w = max(14, round((proc_w * scale) / 14) * 14)
        if target_h < effective_input_size:
            target_h = ((effective_input_size + 13) // 14) * 14
        if target_w < effective_input_size:
            target_w = ((effective_input_size + 13) // 14) * 14

        # --- GPU preprocess + inference with OOM-retry ---
        # On 4K inputs the very first inference call sometimes still OOMs even
        # after the sub-batching above (depends on what else is resident in
        # VRAM). We retry with a progressively smaller model input until it
        # fits, logging each downgrade. Each retry rebuilds the GPU preprocess
        # tensor at the new target size.
        model_dtype = torch.float32 if precision == "fp32" else torch.float16
        if _preprocess_frames_gpu is None:
            raise RuntimeError("[TS_VideoDepth] internal helper _preprocess_frames_gpu missing.")
        interrupt_cb = getattr(mm, "throw_exception_if_processing_interrupted", None)

        # Stage progress bars: each stage occupies its weighted slot of the
        # outer 0..100 bar. infer_video_depth_torch internally calls
        # pbar.update(frame_step) once per sliding window — our adapter
        # rescales those updates to the right slice.
        preprocess_base = outer_cursor
        preprocess_pbar = _StagePBar(master_pbar, preprocess_base, _PBAR_WEIGHT_PREPROCESS, n_frames)
        inference_base = preprocess_base + _PBAR_WEIGHT_PREPROCESS
        # n_frames is a good proxy for inference progress — every window
        # advances by frame_step (=22) up to ~n_frames-ish total ticks.
        inference_pbar = _StagePBar(master_pbar, inference_base, _PBAR_WEIGHT_INFERENCE, n_frames)
        postprocess_base = inference_base + _PBAR_WEIGHT_INFERENCE

        # Step-wise retry sequence: start at the user's requested size, then
        # fall back to a tested-safe ladder. Each step is a multiple of 14 to
        # satisfy the DINOv2 patch grid. Using fixed steps instead of ×0.7
        # ratios avoids overshooting (e.g. 644 → 450 still spikes; better to
        # land on a known-good 392).
        attempt_sizes: list[int] = [effective_input_size]
        for step in (518, 392, 280, 168):
            if step < attempt_sizes[-1] and step not in attempt_sizes:
                attempt_sizes.append(step)

        depth_raw = None
        last_error: Exception | None = None
        attempt_input_size = effective_input_size
        attempt_target_h, attempt_target_w = target_h, target_w

        def _is_cuda_memory_error(err: BaseException) -> bool:
            """Recognise CUDA-allocator failures beyond plain OutOfMemoryError.
            On Windows + CUDAMallocAsync, a follow-up retry sometimes surfaces
            as a bare RuntimeError with `free_upper_bound + pytorch_used_bytes`
            or `INTERNAL ASSERT FAILED`. We treat those as OOM-class so the
            retry loop catches them too."""
            if isinstance(err, torch.cuda.OutOfMemoryError):
                return True
            if isinstance(err, RuntimeError):
                msg = str(err).lower()
                return (
                    "out of memory" in msg
                    or "free_upper_bound" in msg
                    or "cudamallocasync" in msg
                    or "cudamalloc" in msg
                    or ("alloc" in msg and "cuda" in msg)
                )
            return False

        def _hard_reclaim_vram():
            """Belt-and-suspenders reclamation between retry attempts.
            CUDAMallocAsync's allocator state can stay 'dirty' after an OOM,
            which makes the next allocation fail even when the new request is
            smaller. Forcing GC + sync + soft_empty_cache resets it cleanly."""
            gc.collect()
            if load_device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception as exc:
                    logger.debug("%s VRAM reclaim ignored exception: %s", LOG_PREFIX, exc)
                if hasattr(mm, "soft_empty_cache"):
                    try:
                        mm.soft_empty_cache()
                    except Exception:
                        pass

        for attempt_idx, attempt_input_size in enumerate(attempt_sizes):
            # Recompute resize target for this attempt's input_size.
            scale = max(attempt_input_size / proc_h, attempt_input_size / proc_w)
            attempt_target_h = max(14, round((proc_h * scale) / 14) * 14)
            attempt_target_w = max(14, round((proc_w * scale) / 14) * 14)
            if attempt_target_h < attempt_input_size:
                attempt_target_h = ((attempt_input_size + 13) // 14) * 14
            if attempt_target_w < attempt_input_size:
                attempt_target_w = ((attempt_input_size + 13) // 14) * 14

            try:
                logger.info(
                    "%s [stage 1/3] Preprocess: %s frames %sx%s → %sx%s on %s (%s)",
                    LOG_PREFIX, n_frames, original_h, original_w,
                    attempt_target_h, attempt_target_w, load_device, precision,
                )
                preprocess_pbar.inner = 0
                preprocess_pbar.update_absolute(0, total=n_frames)
                frames_gpu = _preprocess_frames_gpu(
                    images, attempt_target_h, attempt_target_w,
                    load_device, model_dtype, chunk_size=8,
                    on_chunk_done=preprocess_pbar.update,
                )
                preprocess_pbar.finish()

                logger.info(
                    "%s [stage 2/3] Inference: input_size=%s precision=%s frames=%s (attempt %s)",
                    LOG_PREFIX, attempt_input_size, precision, n_frames, attempt_idx + 1,
                )
                inference_pbar.inner = 0
                inference_pbar.update_absolute(0, total=n_frames)
                depth_raw = model.infer_video_depth_torch(
                    frames_gpu,
                    input_size=attempt_input_size,
                    device=load_device,
                    fp32=(precision == "fp32"),
                    pbar=inference_pbar,
                    interrupt_cb=interrupt_cb,
                )
                inference_pbar.finish()
                del frames_gpu
                if load_device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            except Exception as exc:  # noqa: BLE001 — broad on purpose
                if not _is_cuda_memory_error(exc):
                    raise
                last_error = exc
                try:
                    del frames_gpu
                except UnboundLocalError:
                    pass
                _hard_reclaim_vram()
                # Move the patcher's model off-GPU temporarily so the
                # allocator can defragment, then load it back. Otherwise
                # CUDAMallocAsync sometimes refuses smaller allocations
                # even with plenty of headroom reported.
                try:
                    patcher.model.to("cpu")
                except Exception:
                    pass
                _hard_reclaim_vram()
                if attempt_idx + 1 < len(attempt_sizes):
                    logger.warning(
                        "%s CUDA OOM at input_size=%s (%s). Retrying with input_size=%s.",
                        LOG_PREFIX, attempt_input_size, type(exc).__name__,
                        attempt_sizes[attempt_idx + 1],
                    )
                    # Reload model onto GPU before the next attempt.
                    mm.load_model_gpu(patcher)
                    model = patcher.model

        if depth_raw is None:
            raise RuntimeError(
                f"[TS_VideoDepth] CUDA OOM at every input_size in {attempt_sizes}. "
                f"Free more VRAM or lower 'max_res'. Last error: {last_error}"
            )

        # depth_raw: (N, target_h, target_w) float32 on CPU
        # --- GPU postprocess (chunked) ---
        logger.info("%s [stage 3/3] Postprocess: %s frames → %sx%s", LOG_PREFIX, n_frames, original_h, original_w)
        # Postprocess does denoise + final upscale; we give each pass half the
        # postprocess slot. `inner_total` is set lazily inside _postprocess_depth
        # based on the actual number of chunks done.
        postprocess_pbar = _StagePBar(master_pbar, postprocess_base, _PBAR_WEIGHT_POSTPROCESS, n_frames * 2)
        output = _postprocess_depth(
            depth_raw=depth_raw,
            rgb_for_guide=images if edge_aware_upscale else None,
            original_h=original_h,
            original_w=original_w,
            normalization_mode=normalization_mode,
            denoise_method=denoise_method,
            apply_median_blur=apply_median_blur,
            dithering_strength=dithering_strength,
            dither_pattern=dither_pattern,
            edge_aware_upscale=edge_aware_upscale,
            upscale_algorithm=upscale_algorithm,
            colormap=colormap,
            device=load_device,
            pbar=postprocess_pbar,
        )
        postprocess_pbar.finish()
        _emit_outer(_PBAR_TOTAL)

        logger.info("%s Done. Output: %s", LOG_PREFIX, tuple(output.shape))
        return IO.NodeOutput(output)


def _postprocess_depth(
    depth_raw: torch.Tensor,
    rgb_for_guide: torch.Tensor | None,
    original_h: int,
    original_w: int,
    normalization_mode: str,
    denoise_method: str,
    apply_median_blur: bool,
    dithering_strength: float,
    dither_pattern: str,
    edge_aware_upscale: bool,
    upscale_algorithm: str,
    colormap: str,
    device: torch.device,
    pbar=None,
) -> torch.Tensor:
    """GPU-resident, chunked postprocess.

    depth_raw arrives as (N, H_low, W_low) on CPU. We push chunks to the GPU,
    denoise → normalize → upscale → dither → colormap → bring back to CPU as
    float32 NHWC. Peak VRAM is bounded by ``_POSTPROCESS_CHUNK``.

    ``pbar`` (optional ``_StagePBar``) receives `.update(n_frames)` per chunk
    so the node's main bar reflects postprocess progress.
    """
    n = depth_raw.shape[0]

    # The pbar gets ticks for: (denoise pass if active) + (final pass).
    # We pre-size its inner_total so 'finish' aligns even when denoise is off.
    method = denoise_method
    if method == "auto":
        method = "median" if apply_median_blur else "none"
    has_denoise = method != "none"
    if pbar is not None:
        pbar.inner_total = n * (2 if has_denoise else 1)
        pbar.inner = 0
        pbar.update_absolute(0)

    # ----- denoise pass (operates at low-res; keep depth on CPU as fp32 list) -----
    if has_denoise:
        logger.info("%s Postprocess denoise: %s", LOG_PREFIX, method)
        denoised_chunks = []
        for start in range(0, n, _POSTPROCESS_CHUNK):
            end = min(start + _POSTPROCESS_CHUNK, n)
            chunk = depth_raw[start:end].to(device, non_blocking=True).float()
            if method == "median":
                chunk = _median_blur_chunk(chunk)
            elif method == "bilateral":
                chunk = _bilateral_blur_chunk(chunk, rgb_chunk=None)
            denoised_chunks.append(chunk.cpu())
            del chunk
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if pbar is not None:
                pbar.update(end - start)
        depth_raw = torch.cat(denoised_chunks, dim=0)
        del denoised_chunks

    # ----- global normalization (streamed) -----
    lo, hi = _compute_global_normalization(depth_raw, normalization_mode)
    logger.info(
        "%s Normalize: mode=%s lo=%.4f hi=%.4f", LOG_PREFIX, normalization_mode, lo, hi,
    )

    lut = _get_colormap_lut(colormap, device)

    # ----- upscale + dither + colormap (chunked, GPU) -----
    out = torch.empty((n, original_h, original_w, 3), dtype=torch.float32)
    for start in range(0, n, _POSTPROCESS_CHUNK):
        end = min(start + _POSTPROCESS_CHUNK, n)
        chunk = depth_raw[start:end].to(device, non_blocking=True).float()
        chunk = (chunk - lo) / (hi - lo)
        chunk.clamp_(0.0, 1.0)

        if edge_aware_upscale and rgb_for_guide is not None:
            guide = rgb_for_guide[start:end].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous().float()
            chunk = _guided_upsample_chunk(chunk, guide, original_h, original_w)
            del guide
        else:
            chunk = _resize_depth_chunk(chunk, original_h, original_w, upscale_algorithm)
            chunk.clamp_(0.0, 1.0)

        if dithering_strength > 0.0:
            chunk = _apply_dither(chunk, dithering_strength, dither_pattern)

        colored = _apply_colormap(chunk, lut)
        out[start:end].copy_(colored.cpu(), non_blocking=True)
        del chunk, colored
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if pbar is not None:
            pbar.update(end - start)

    return out


NODE_CLASS_MAPPINGS = {"TS_VideoDepthNode": TS_VideoDepth}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_VideoDepthNode": "TS Video Depth"}
