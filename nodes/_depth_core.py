"""Общий движок карт глубины для TS Video Depth и TS Image Depth.

Здесь всё, что у двух нод одинаково: каталог весов, загрузка и кэш патчеров,
нормализация, шумоподавление, дизеринг, апскейл и раскраска. Ноды остаются
тонкими — каждая описывает свою схему и зовёт отсюда.

⚠️ Модуль приватный (`_`-префикс), загрузчик его не сканирует. Держать его на
уровне пакета `nodes/`, а не внутри `nodes/video/`, обязательно: им пользуются
ноды из РАЗНЫХ категорий, а категория чужой приватный модуль импортировать не
должна.
"""

import gc
import logging
import math
import os

import comfy.model_management as mm
import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar, load_torch_file

from ._hf_download import pinned_revision, snapshot_download_resilient

logger = logging.getLogger("comfyui_timesaver.depth_core")
LOG_PREFIX = "[TS Depth]"


# --- VideoDepthAnything model import (multi-path for legacy installs) ---
try:
    from .video_depth_anything.video_depth import VideoDepthAnything, _preprocess_frames_gpu
except ImportError:
    try:
        from video_depth_anything.video_depth import VideoDepthAnything, _preprocess_frames_gpu
    except ImportError as e:
        logger.error("%s CRITICAL IMPORT ERROR: Could not import VideoDepthAnything model: %s", LOG_PREFIX, e)
        VideoDepthAnything = None
        _preprocess_frames_gpu = None

# Depth Anything V2 — отдельная модель для ОДИНОЧНОГО кадра. Не «режим»
# видео-модели: та обучена на 32-кадровом окне, и одиночный снимок ей
# приходится подавать дублированным, отчего диапазон глубины сплющивается.
try:
    from .video_depth_anything.image_depth import DepthAnythingV2
except ImportError as e:  # pragma: no cover - только при битой установке
    logger.error("%s Could not import DepthAnythingV2: %s", LOG_PREFIX, e)
    DepthAnythingV2 = None


# ---------------------------------------------------------------------------
# Каталог моделей
# ---------------------------------------------------------------------------
# ⚠️ Имена файлов — ЧАСТЬ КОНТРАКТА: они лежат в сохранённых графах значениями
# виджета. Старые `.pth` остаются здесь навсегда; новые записи ДОПИСЫВАЮТСЯ.
#
# ⚠️ Веса, которые пак публикует сам, лежат в fp16 safetensors: вдвое меньше
# качать и на порядок быстрее читать (замерено: 0.66 с у .pth против 0.01 с),
# при отклонении от чистого fp32 в 0.02% шкалы. Первоисточники — CC-BY-NC-4.0,
# и наша перезаливка ту же лицензию сохраняет.
TS_DEPTH_REPO = "hfmaster/depth-fp16"

VIDEO_MODELS = {
    "video_depth_anything_vitl_fp16.safetensors": {
        "encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024],
        "repo": TS_DEPTH_REPO,
    },
    "video_depth_anything_vits_fp16.safetensors": {
        "encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384],
        "repo": TS_DEPTH_REPO,
    },
    # ⚠️ Ниже — старые `.pth`. Их НЕТ в списке виджета (см. LEGACY_MODELS), но
    # из каталога они не убраны: значение из сохранённого графа обязано хотя бы
    # разрешиться в веса, иначе граф не откроется, а молча сломается на счёте.
    "video_depth_anything_vits.pth": {
        "encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384],
        "repo": "depth-anything/Video-Depth-Anything-Small",
    },
    "video_depth_anything_vitl.pth": {
        "encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024],
        "repo": "depth-anything/Video-Depth-Anything-Large",
    },
}

IMAGE_MODELS = {
    "depth_anything_v2_vitl_fp16.safetensors": {
        "encoder": "vitl", "repo": TS_DEPTH_REPO,
    },
    "depth_anything_v2_vitl.pth": {
        "encoder": "vitl", "repo": "depth-anything/Depth-Anything-V2-Large",
    },
}

# ⚠️ Что показывать в выпадающем списке. Владелец пака попросил оставить в
# выборе только safetensors: они вдвое легче и читаются на порядок быстрее.
# Старые `.pth` остаются рабочими значениями (см. каталоги выше) — просто их
# больше не предлагают.
LEGACY_MODELS = frozenset(name for name in (*VIDEO_MODELS, *IMAGE_MODELS)
                          if not name.endswith(".safetensors"))


def offered(catalogue: dict) -> list[str]:
    """Имена для виджета: только safetensors, в порядке каталога."""
    return [name for name in catalogue if name not in LEGACY_MODELS]


# Куда какая семья кладёт веса. Папка `depthanything` совпадает с той, что
# использует ComfyUI-DepthAnythingV2, — у кого файл уже есть, второй раз не
# скачается.
MODEL_FOLDERS = {"video": "videodepthanything", "image": "depthanything"}

# ⚠️ Потолок окна — свойство ВЕСОВ, а не вкуса: временной модуль несёт
# абсолютное позиционное кодирование ровно на 32 позиции (
# в motion_module). На 48 кадрах прогон падает несовпадением размеров.
_MAX_WINDOW_LENGTH = 32


def _model_family(model_filename: str) -> str:
    """К какой семье относится файл. Ошибка тут молча вернула бы чушь."""
    if model_filename in IMAGE_MODELS:
        return "image"
    if model_filename in VIDEO_MODELS:
        return "video"
    # Исторические имена без точного совпадения: решаем по подстроке.
    if "video_depth" in model_filename.lower():
        return "video"
    if "depth_anything_v2" in model_filename.lower():
        return "image"
    raise ValueError(f"[TS Universal Depth] Unknown model file: {model_filename!r}")


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
    # По одному патчеру на файл весов: у ноды две семьи моделей, и держать обе
    # заряженными дешевле, чем перезагружать при каждом переключении режима.
    patchers: dict[str, object] = {}
    loaded_filename: str | None = None
    colormap_luts: dict[str, torch.Tensor] = {}


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
        # torch.quantile rejects inputs over 2^24 elements; at ~4k samples per
        # frame that limit is hit around the 4000-frame mark (2.5 min of
        # video). Uniform stride keeps the subsample unbiased.
        max_quantile_samples = 8_000_000
        if samples.numel() > max_quantile_samples:
            stride = (samples.numel() + max_quantile_samples - 1) // max_quantile_samples
            samples = samples[::stride]
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

def _infer_frames_independently(
    model, frames: torch.Tensor, *, device: torch.device, fp32: bool,
    pbar=None, interrupt_cb=None, batch_size: int = 4,
) -> torch.Tensor:
    """Прогнать каждый кадр отдельно моделью одиночного изображения.

    ⚠️ Именно ОТДЕЛЬНО, а не «окном длины 1»: у Depth Anything V2 нет
    временного модуля вовсе, и кадры между собой не связаны. Это ровно то, что
    нужно и одиночному снимку, и пачке НЕсвязанных картинок.

    Args:
        model: :class:`DepthAnythingV2` на устройстве.
        frames: ``(N, 3, H, W)``, уже нормированные, на устройстве.
        device: где считать.
        fp32: выключить autocast.
        pbar: индикатор, получает ``.update(число кадров)``.
        interrupt_cb: проверка отмены, зовётся на каждой пачке.
        batch_size: сколько кадров за раз. Больше — лучше занята видеокарта,
            но и пик памяти выше; на одиночном снимке значения не имеет.

    Returns:
        ``(N, H, W)`` float32 на CPU.
    """
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    n = int(frames.shape[0])
    out: list[torch.Tensor] = []
    for start in range(0, n, batch_size):
        if interrupt_cb is not None:
            interrupt_cb()
        end = min(start + batch_size, n)
        batch = frames[start:end]
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=(not fp32)):
                depth = model(batch)
        out.append(depth.float().cpu())
        del batch, depth
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if pbar is not None:
            pbar.update(end - start)
    return torch.cat(out, dim=0)


def _suppress_flicker(
    depth: torch.Tensor, strength: float, radius: int,
    device: torch.device, chunk_size: int = 32,
) -> torch.Tensor:
    """Убрать temporal-мерцание в карте глубины.

    Берётся МЕДИАНА по окну из ``2*radius+1`` кадров, а не среднее: медиана
    выбрасывает одиночные выбросы (тот самый «поп» на одном кадре), но не
    размазывает настоящее движение — среднее размазало бы. Результат
    подмешивается к исходному с весом ``strength``, поэтому 0 — это в точности
    прежнее поведение, а 1 — чистая медиана.

    Фильтр работает на НИЗКОМ разрешении, до апскейла: там он и дешевле, и
    честнее — мерцание рождается в модели, а не в апскейле.

    Args:
        depth: ``(N, H, W)`` на CPU.
        strength: 0..1, вес отфильтрованного.
        radius: половина окна в кадрах.
        device: где считать.
        chunk_size: сколько кадров держать на устройстве разом.

    Returns:
        ``(N, H, W)`` на CPU.
    """
    n = int(depth.shape[0])
    if strength <= 0.0 or radius < 1 or n < 3:
        return depth

    window = 2 * radius + 1
    out = torch.empty_like(depth)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Берём с запасом по краям, чтобы у каждого кадра было полное окно.
        lo = max(0, start - radius)
        hi = min(n, end + radius)
        block = depth[lo:hi].to(device, non_blocking=True).float()
        # Края дорожки достраиваем повтором: у первого кадра нет прошлого.
        pad_lo, pad_hi = start - lo, hi - end
        need_lo, need_hi = radius - pad_lo, radius - pad_hi
        if need_lo > 0:
            block = torch.cat([block[:1].expand(need_lo, -1, -1), block], dim=0)
        if need_hi > 0:
            block = torch.cat([block, block[-1:].expand(need_hi, -1, -1)], dim=0)
        # (кадры окна, кадры чанка, H, W) -> медиана по первой оси.
        stacked = torch.stack([block[i:i + (end - start)] for i in range(window)], dim=0)
        filtered = stacked.median(dim=0).values
        del stacked
        original = block[radius:radius + (end - start)]
        out[start:end] = torch.lerp(original, filtered, strength).cpu()
        del block, filtered, original
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def _load_model_to_offload_cpu(model_filename: str, on_download_start=None):
    """Собрать модель, залить веса, вернуть её в eval() на CPU.

    Работает с обеими семьями (видео и одиночный кадр) и обеими тарами
    (``.pth`` и ``.safetensors``). Отсутствующие веса скачиваются.
    """
    family = _model_family(model_filename)
    catalogue = IMAGE_MODELS if family == "image" else VIDEO_MODELS
    download_path = os.path.join(folder_paths.models_dir, MODEL_FOLDERS[family])
    model_path = os.path.join(download_path, model_filename)

    if not os.path.exists(model_path):
        if on_download_start is not None:
            on_download_start(model_filename)
        entry = catalogue.get(model_filename)
        if entry is None:
            raise ValueError(
                f"[TS Universal Depth] '{model_filename}' is not in the catalogue and is "
                f"not on disk; put it in models/{MODEL_FOLDERS[family]}/ or pick another."
            )
        logger.info("%s Downloading %s from %s …", LOG_PREFIX, model_filename, entry["repo"])
        os.makedirs(download_path, exist_ok=True)
        # Shared transport: mirror cycling (HF_ENDPOINT) + hub kwarg compatibility.
        snapshot_download_resilient(
            repo_id=entry["repo"],
            local_dir=download_path,
            revision=pinned_revision(entry["repo"]),
            allow_patterns=[f"*{model_filename}*"],
            log=logger,
            log_prefix=LOG_PREFIX,
        )
        logger.info("%s Download complete: %s", LOG_PREFIX, model_filename)

    entry = catalogue.get(model_filename)
    if entry is None:
        # Файл лежит на диске, но в каталоге его нет — выводим конфиг из имени.
        encoder = next((k for k in ("vits", "vitb", "vitl", "vitg") if k in model_filename.lower()), None)
        if encoder is None:
            raise ValueError(f"[TS Universal Depth] Cannot tell the encoder from {model_filename!r}.")
        entry = {"encoder": encoder}

    # ⚠️ device здесь ОБЯЗАН быть torch.device, а не строкой: для .pth строка
    # проходит, а на safetensors comfy падает с
    # "'str' object has no attribute 'type'".
    state_dict = load_torch_file(model_path, device=torch.device("cpu"))
    state_dict = state_dict.get("state_dict", state_dict) if isinstance(state_dict, dict) else state_dict

    if family == "image":
        if DepthAnythingV2 is None:
            raise RuntimeError("[TS Universal Depth] DepthAnythingV2 failed to import.")
        model = DepthAnythingV2(encoder=entry["encoder"])
    else:
        model = VideoDepthAnything(
            encoder=entry["encoder"],
            features=entry.get("features", 256),
            out_channels=entry.get("out_channels", [256, 512, 1024, 1024]),
        )
    # strict=True намеренно: разошедшиеся ключи означают, что конфиг не тот, а
    # такая модель не падает — она возвращает правдоподобный мусор.
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def _ensure_patcher(model_filename: str, load_device: torch.device, on_download_start=None):
    """Return a ModelPatcher for the requested model, building one on demand
    and tearing down the previous one if the user switched files.

    ``on_download_start`` is forwarded to ``_load_model_to_offload_cpu`` and
    fires only when the weights have to be fetched from HuggingFace.
    """
    import comfy.model_patcher as model_patcher

    cached = _state.patchers.get(model_filename)
    if cached is not None:
        return cached, False

    # ⚠️ Патчеры держатся ПО ОДНОМУ НА ФАЙЛ и НЕ выбрасываются при смене
    # модели. Раньше предыдущий сбрасывался, и переключение режима
    # «видео ↔ одиночный» стоило ~4 секунды на перезагрузку весов при том, что
    # сама работа занимает 0.2 (замерено). Выгрузкой из видеопамяти пусть
    # занимается менеджер моделей ComfyUI — он для этого и есть, и он видит
    # общую картину, а мы нет.
    #
    # Больше двух семей у ноды не бывает, так что словарь не растёт: разные
    # чекпойнты ОДНОЙ семьи вытесняют друг друга.
    family = _model_family(model_filename)
    for other in [name for name in _state.patchers if _model_family(name) == family]:
        logger.info("%s Releasing %s (same family, switching to %s).",
                    LOG_PREFIX, other, model_filename)
        _state.patchers.pop(other, None)

    offload_device = mm.unet_offload_device() if hasattr(mm, "unet_offload_device") else torch.device("cpu")
    model = _load_model_to_offload_cpu(model_filename, on_download_start=on_download_start)
    patcher = model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
    _state.patchers[model_filename] = patcher
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
                # Guide the range term with the source luma, resampled to the
                # depth's inference resolution. Without a guide the filter's range
                # weights collapse to 1 and it degrades to a plain Gaussian, which
                # is not the edge-preserving behaviour the option advertises.
                guide_chunk = None
                if rgb_for_guide is not None:
                    guide_chunk = (
                        rgb_for_guide[start:end]
                        .to(device, non_blocking=True)
                        .permute(0, 3, 1, 2)
                        .contiguous()
                        .float()
                    )
                    if guide_chunk.shape[-2:] != chunk.shape[-2:]:
                        guide_chunk = F.interpolate(
                            guide_chunk,
                            size=(int(chunk.shape[1]), int(chunk.shape[2])),
                            mode="bilinear",
                            align_corners=False,
                            antialias=True,
                        )
                chunk = _bilateral_blur_chunk(chunk, rgb_chunk=guide_chunk)
                del guide_chunk
            denoised_chunks.append(chunk.cpu())
            del chunk
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if pbar is not None:
                pbar.update(end - start)
        depth_raw = torch.cat(denoised_chunks, dim=0)
        del denoised_chunks

    # ----- normalization -----
    # ⚠️ "per_frame" — не «ещё один режим», а РАЗНАЯ семантика. Общая пара
    # (lo, hi) на всю пачку нужна видео: иначе яркость карты дышала бы от кадра
    # к кадру. Для пачки НЕ СВЯЗАННЫХ картинок она же вредна — один снимок с
    # очень близким объектом сплющит контраст всех остальных. Эталонная
    # реализация Depth Anything V2 нормирует каждую картинку отдельно.
    per_frame = normalization_mode == "per_frame"
    if per_frame:
        lo = hi = 0.0
        logger.info("%s Normalize: per frame (each picture on its own min/max)", LOG_PREFIX)
    else:
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
        if per_frame:
            flat = chunk.reshape(chunk.shape[0], -1)
            frame_lo = flat.min(dim=1).values.view(-1, 1, 1)
            frame_hi = flat.max(dim=1).values.view(-1, 1, 1)
            chunk = (chunk - frame_lo) / (frame_hi - frame_lo).clamp_min(1e-6)
        else:
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


def run_depth(
    *,
    images,
    model_filename: str,
    single_image: bool,
    precision: str,
    colormap: str,
    max_res: int,
    input_size: int = 518,
    normalization_mode: str = "minmax",
    denoise_method: str = "none",
    apply_median_blur: bool = True,
    dithering_strength: float = 0.0,
    dither_pattern: str = "bayer",
    edge_aware_upscale: bool = True,
    upscale_algorithm: str = "Lanczos4",
    window_length: int = 32,
    window_overlap: int = 10,
    flicker_suppression: float = 0.0,
    flicker_radius: int = 1,
    log_prefix: str = LOG_PREFIX,
) -> torch.Tensor:
    """Посчитать карту глубины и вернуть IMAGE-тензор (N, H, W, 3).

    Один конвейер на обе ноды. Что меняет `single_image`:

    * считает Depth Anything V2, кадр за кадром, без временного окна;
    * `max_res` — разрешение ОБРАБОТКИ, а не только предварительный ужим:
      кадр режется до кратности 14 и уходит в модель как есть;
    * `input_size`, окно и подавление мерцания не участвуют;
    * нормализация всегда minmax.

    ⚠️ Это не «два режима одной ноды», а общий низ у двух разных. Ноды
    сверху дают только свою схему — иначе реализации разъедутся.
    """
    single_image_mode = bool(single_image)
    if single_image_mode:
        if DepthAnythingV2 is None:
            raise ImportError(
                f"{log_prefix} cannot run: DepthAnythingV2 failed to import."
            )
    elif VideoDepthAnything is None:
        raise ImportError(f"{log_prefix} cannot run: VideoDepthAnything failed to import.")

    # ⚠️ Окно длиннее 32 кадров ПОСТРОИТЬ НЕЛЬЗЯ: у временного модуля
    # абсолютное позиционное кодирование ровно на 32 позиции, и на 48
    # прогон падает с несовпадением размеров тензора. Виджет ограничен
    # сверху, но граф в API-формате может прислать что угодно.
    if window_length > _MAX_WINDOW_LENGTH:
        logger.warning(
            "%s window_length=%s exceeds the %s the temporal module was built for; "
            "clamping.", log_prefix, window_length, _MAX_WINDOW_LENGTH,
        )
        window_length = _MAX_WINDOW_LENGTH

    # ⚠️ Перекрытие обязано быть строго меньше окна: равное или большее
    # даёт нулевой или отрицательный шаг, то есть бесконечный цикл.
    if window_overlap >= window_length:
        window_overlap = max(2, window_length // 3)
        logger.warning(
            "%s window_overlap must stay below window_length; using %s.",
            log_prefix, window_overlap,
        )

    if not (isinstance(images, torch.Tensor) and images.ndim == 4):
        raise ValueError(f"{log_prefix} 'images' must be a 4D float tensor (N, H, W, 3).")

    n_frames, original_h, original_w = images.shape[0], images.shape[1], images.shape[2]
    load_device = mm.get_torch_device()

    # --- single outer progress bar covering all stages ---
    master_pbar = ProgressBar(_PBAR_TOTAL)
    outer_cursor = 0

    def _emit_outer(value: int):
        # Local helper used for the download stage and final pin to 100.
        master_pbar.update_absolute(min(value, _PBAR_TOTAL))

    # --- model load via ComfyUI's cooperative VRAM manager ---
    def _on_download_start(name: str):
        logger.info("%s Weights missing, downloading %s …", log_prefix, name)
        _emit_outer(_PBAR_WEIGHT_DOWNLOAD // 2)  # halfway through the download slot

    active_model_file = model_filename
    patcher, _was_just_built = _ensure_patcher(
        active_model_file, load_device, on_download_start=_on_download_start,
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
    cap = max_res
    if cap > 0 and max(original_h, original_w) > cap:
        scale = cap / max(original_h, original_w)
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
                "%s Adjusted input_size %s -> %s (multiple of 14).",
                log_prefix, effective_input_size, adjusted,
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
        raise RuntimeError(f"{log_prefix} internal helper _preprocess_frames_gpu missing.")
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

    # ⚠️ Одиночный режим считает В СВОЁМ разрешении, а не в 518.
    #
    # `input_size` — размер, на котором обучалась ВИДЕО-модель, и «ужать до
    # 518 по короткой стороне» для неё правильно. Для стоп-кадра это и была
    # главная причина мыла: замерено, портрет 1600 px уходил в модель как
    # 784x518 и возвращался заметно мягче эталонной реализации (расхождение
    # 4.10% против 2.36% при полном размере). У Depth Anything V2 нет
    # временного модуля, сетка патчей может быть любой — режем до кратности
    # 14 и всё. Лестница отката тут своя: каждый шаг вдвое мельче.
    single_targets: list[tuple[int, int]] = []
    if single_image_mode:
        base_h = max(14, (proc_h // 14) * 14)
        base_w = max(14, (proc_w // 14) * 14)
        single_targets = [(base_h, base_w)]
        while min(single_targets[-1]) > 280:
            last_h, last_w = single_targets[-1]
            single_targets.append((max(14, (last_h // 28) * 14), max(14, (last_w // 28) * 14)))
        attempt_sizes = [max(h, w) for h, w in single_targets]
        logger.info(
            "%s Single image: %sx%s -> processing at %sx%s (cap %s)",
            log_prefix, original_h, original_w, base_h, base_w,
            "native" if max_res <= 0 else max_res,
        )

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
                logger.debug("%s VRAM reclaim ignored exception: %s", log_prefix, exc)
            if hasattr(mm, "soft_empty_cache"):
                try:
                    mm.soft_empty_cache()
                except Exception:
                    pass

    # cudnn.benchmark autotunes conv algorithms for the transformer's fixed
    # shapes (a real speed-up across the many sliding windows in one run).
    # It is a GLOBAL flag, so we set it only for the inference block and
    # restore the previous value in `finally` — leaving it on would leak
    # into every other node's convs (varying-shape models re-autotune and
    # lose determinism). No-op on non-CUDA devices.
    cudnn_benchmark_prev = torch.backends.cudnn.benchmark
    if load_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    try:
        for attempt_idx, attempt_input_size in enumerate(attempt_sizes):
            if single_image_mode:
                # Свой размер, посчитанный выше: пропорции кадра как есть.
                attempt_target_h, attempt_target_w = single_targets[attempt_idx]
            else:
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
                    "%s [stage 1/3] Preprocess: %s frames %sx%s -> %sx%s on %s (%s)",
                    log_prefix, n_frames, original_h, original_w,
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
                    "%s [stage 2/3] Inference (%s): input_size=%s precision=%s frames=%s (attempt %s)",
                    log_prefix, "single image" if single_image_mode else "video", attempt_input_size, precision, n_frames, attempt_idx + 1,
                )
                inference_pbar.inner = 0
                inference_pbar.update_absolute(0, total=n_frames)
                if single_image_mode:
                    depth_raw = _infer_frames_independently(
                        model, frames_gpu,
                        device=load_device,
                        fp32=(precision == "fp32"),
                        pbar=inference_pbar,
                        interrupt_cb=interrupt_cb,
                    )
                else:
                    depth_raw = model.infer_video_depth_torch(
                        frames_gpu,
                        input_size=attempt_input_size,
                        device=load_device,
                        fp32=(precision == "fp32"),
                        pbar=inference_pbar,
                        interrupt_cb=interrupt_cb,
                        infer_len=window_length,
                        overlap=window_overlap,
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
                    # Deliberate: the name may never have been bound if the
                    # failure happened before the upload, and the whole point
                    # is to drop the VRAM reference when it WAS.
                    del frames_gpu  # noqa: F821
                except UnboundLocalError:
                    pass
                _hard_reclaim_vram()
                # Release the model THROUGH model_management so its
                # bookkeeping stays consistent. The old code called
                # patcher.model.to("cpu") behind the ModelPatcher's back:
                # load_model_gpu() then saw an "already loaded" patcher,
                # no-opped, and the retry crashed with a device mismatch —
                # and the cached MODEL stayed desynced for later prompts.
                # A clean unload also lets CUDAMallocAsync defragment.
                try:
                    if hasattr(mm, "unload_model_and_clones"):
                        mm.unload_model_and_clones(patcher)
                    elif hasattr(mm, "unload_model_clones"):
                        mm.unload_model_clones(patcher)
                except Exception as unload_exc:
                    logger.debug("%s unload between OOM retries failed: %s", log_prefix, unload_exc)
                _hard_reclaim_vram()
                if attempt_idx + 1 < len(attempt_sizes):
                    logger.warning(
                        "%s CUDA OOM at input_size=%s (%s). Retrying with input_size=%s.",
                        log_prefix, attempt_input_size, type(exc).__name__,
                        attempt_sizes[attempt_idx + 1],
                    )
                    # Reload model onto GPU before the next attempt.
                    mm.load_model_gpu(patcher)
                    model = patcher.model
    finally:
        torch.backends.cudnn.benchmark = cudnn_benchmark_prev

    if depth_raw is None:
        raise RuntimeError(
            f"{log_prefix} CUDA OOM at every input_size in {attempt_sizes}. "
            f"Free more VRAM or lower 'max_res'. Last error: {last_error}"
        )

    # depth_raw: (N, target_h, target_w) float32 on CPU
    # ⚠️ Мерцание давим ДО апскейла: оно рождается в модели, а не в
    # интерполяции, и на низком разрешении фильтр и дешевле, и точнее.
    # В одиночном режиме кадры между собой не связаны — усреднять нечего.
    if flicker_suppression > 0.0 and not single_image_mode and n_frames >= 3:
        logger.info(
            "%s Flicker suppression: strength=%.2f radius=%s frames",
            log_prefix, flicker_suppression, flicker_radius,
        )
        depth_raw = _suppress_flicker(
            depth_raw, flicker_suppression, flicker_radius, load_device,
        )

    # ⚠️ `percentile` — инструмент ВРЕМЕННОЙ устойчивости: он режет 1%/99%,
    # чтобы один кадр с близким объектом не сплющил контраст всего клипа. У
    # одиночного снимка соседних кадров нет, и эта обрезка просто выжигает
    # самое близкое и самое далёкое в чистый белый и чёрный. Замерено на
    # живом сервере против эталонной реализации: расхождение 2.19% против
    # 0.38% — весь остаток разницы давала именно она. Поэтому кадр всегда
    # нормируется minmax, а виджет подписан как видеорежимный.
    effective_normalization = "minmax" if single_image_mode else normalization_mode
    if single_image_mode and normalization_mode != "minmax":
        logger.info(
            "%s Single image: normalization forced to minmax (percentile is a video tool).",
            log_prefix,
        )

    # --- GPU postprocess (chunked) ---
    logger.info("%s [stage 3/3] Postprocess: %s frames -> %sx%s", log_prefix, n_frames, original_h, original_w)
    # Postprocess does denoise + final upscale; we give each pass half the
    # postprocess slot. `inner_total` is set lazily inside _postprocess_depth
    # based on the actual number of chunks done.
    postprocess_pbar = _StagePBar(master_pbar, postprocess_base, _PBAR_WEIGHT_POSTPROCESS, n_frames * 2)
    output = _postprocess_depth(
        depth_raw=depth_raw,
        # Needed by BOTH the guided upscale and the bilateral denoise, so it is
        # passed unconditionally; each consumer gates on its own option.
        rgb_for_guide=images,
        original_h=original_h,
        original_w=original_w,
        normalization_mode=effective_normalization,
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

    logger.info("%s Done. Output: %s", log_prefix, tuple(output.shape))
    return output
