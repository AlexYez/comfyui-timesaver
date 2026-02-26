import os
import math
from concurrent.futures import ThreadPoolExecutor

import torch
import comfy.model_management as model_management
from comfy.utils import ProgressBar

_LOG_PREFIX = "[TS Color Match]"
_SUPPORTED_DEVICES = ["auto", "gpu", "cpu"]
_SINKHORN_MAX_POINTS = 8192
_SINKHORN_BLUR = 0.01
_SINKHORN_MAX_POINTS_TENSORIZED_CPU = 1024
_SINKHORN_MAX_POINTS_TENSORIZED_GPU = 2048
_SINKHORN_ITERS = 50
_TEMPORAL_EMA = 0.85
_FIXED_SAMPLE_SEED = 1234567
_DEFAULT_COMPUTE_MAX_SIDE = 1024
_DEFAULT_MKL_SAMPLE_POINTS = 300000
_DEFAULT_SINKHORN_MAX_POINTS = 2048
_SUPPORTED_MASK_MODES = ["none", "rectangle", "ellipse"]

_SINKHORN_AVAILABLE = True

_SUPPORTED_MODES = ["mkl"]
if _SINKHORN_AVAILABLE:
    _SUPPORTED_MODES.append("sinkhorn")


def _log_info(message, enabled):
    if not enabled:
        return
    print(f"{_LOG_PREFIX} {message}")


def _validate_image_tensor(name, image):
    if not torch.is_tensor(image):
        raise ValueError(f"{_LOG_PREFIX} {name} must be a torch.Tensor")
    if image.ndim != 4:
        raise ValueError(f"{_LOG_PREFIX} {name} must have shape [B, H, W, C]")
    if image.shape[-1] != 3:
        raise ValueError(f"{_LOG_PREFIX} {name} must have 3 channels (RGB)")


def _broadcast_batches(fix_image, reference_image):
    fix_b = fix_image.shape[0]
    ref_b = reference_image.shape[0]
    if fix_b == ref_b:
        return fix_image, reference_image, fix_b
    if fix_b == 1 and ref_b > 1:
        return fix_image.expand(ref_b, -1, -1, -1), reference_image, ref_b
    if ref_b == 1 and fix_b > 1:
        return fix_image, reference_image.expand(fix_b, -1, -1, -1), fix_b
    raise ValueError(f"{_LOG_PREFIX} Batch mismatch: fix={fix_b}, reference={ref_b}")


def _symmetrize(mat):
    return (mat + mat.transpose(-1, -2)) * 0.5


def _sqrtm(mat, eps=1e-8):
    e, v = torch.linalg.eigh(mat)
    e = torch.clamp(e, min=eps)
    return (v * torch.sqrt(e)) @ v.T


def _invsqrtm(mat, eps=1e-8):
    e, v = torch.linalg.eigh(mat)
    e = torch.clamp(e, min=eps)
    return (v * (1.0 / torch.sqrt(e))) @ v.T


def _covariance_from_centered(x):
    n = x.shape[0]
    if n < 2:
        return torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    return (x.T @ x) / (n - 1)


def _mkl_compute_transform(src_img, ref_img, sample_points=0, seed=-1, ref_stats=None, mask_mode="none", mask_size=0):
    channels = src_img.shape[-1]
    src = src_img.reshape(-1, channels)
    ref = ref_img.reshape(-1, channels) if ref_img is not None else None

    if src.shape[0] < 2 or (ref is not None and ref.shape[0] < 2):
        return torch.eye(channels, device=src_img.device, dtype=src_img.dtype), torch.zeros(channels, device=src_img.device, dtype=src_img.dtype)

    gen_src = _make_generator(seed, src_img.device)
    gen_ref = _make_generator(seed + 1, src_img.device) if seed is not None and seed >= 0 else None

    if ref_stats is not None:
        mu_r, Cr = ref_stats
    else:
        mu_r, Cr = _mkl_compute_stats(ref_img, sample_points, gen_ref, mask_mode, mask_size)

    mu_s, Cs = _mkl_compute_stats(src_img, sample_points, gen_src, mask_mode, mask_size)

    Cs_sqrt = _sqrtm(Cs)
    Cs_inv = _invsqrtm(Cs)

    middle = _sqrtm(Cs_sqrt @ Cr @ Cs_sqrt)
    A = Cs_inv @ middle @ Cs_inv
    b = mu_r - (A @ mu_s.T).T

    if torch.isnan(A).any() or torch.isinf(A).any() or torch.isnan(b).any() or torch.isinf(b).any():
        return torch.eye(channels, device=src_img.device, dtype=src_img.dtype), torch.zeros(channels, device=src_img.device, dtype=src_img.dtype)

    return A, b.squeeze(0)


def _sample_pixels(flat, max_points, generator=None):
    max_points = max(2, int(max_points))
    n = flat.shape[0]
    if n <= max_points:
        return flat
    idx = torch.randint(0, n, (max_points,), device=flat.device, generator=generator)
    return flat[idx]


def _make_generator(seed, device):
    if seed is None or int(seed) < 0:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def _log_cuda_memory(stage, device, enabled):
    if not enabled:
        return
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    _log_info(f"{stage} | cuda_allocated={allocated:.1f}MiB reserved={reserved:.1f}MiB", enabled)


def _move_to_device(tensor, device):
    if tensor.device == device:
        return tensor
    return tensor.to(device=device)


def _resize_max_side_single(image, max_side):
    max_side = int(max_side)
    if max_side <= 0:
        return image
    h, w = image.shape[0], image.shape[1]
    if max(h, w) <= max_side:
        return image
    if h >= w:
        new_h = max_side
        new_w = max(1, int(round(w * (max_side / h))))
    else:
        new_w = max_side
        new_h = max(1, int(round(h * (max_side / w))))
    img = image.permute(2, 0, 1).unsqueeze(0)
    resized = torch.nn.functional.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0)


def _build_match_mask(img, mask_mode, mask_size):
    if mask_mode == "none" or mask_size <= 0:
        return None
    h, w = img.shape[0], img.shape[1]
    size = int(mask_size)
    if size <= 0:
        return None
    size = min(size, min(h, w) // 2)
    if size <= 0:
        return None

    device = img.device
    if mask_mode == "rectangle":
        mask = torch.zeros((h, w), device=device, dtype=torch.bool)
        mask[:size, :] = True
        mask[-size:, :] = True
        mask[:, :size] = True
        mask[:, -size:] = True
        return mask

    if mask_mode == "ellipse":
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        rx = max((w - 1) * 0.5, 1.0)
        ry = max((h - 1) * 0.5, 1.0)
        outer = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        rx_in = max(rx - size, 1.0)
        ry_in = max(ry - size, 1.0)
        inner = ((xx - cx) / rx_in) ** 2 + ((yy - cy) / ry_in) ** 2 <= 1.0
        return outer & (~inner)

    return None


def _flatten_with_mask(img, mask_mode, mask_size):
    channels = img.shape[-1]
    flat = img.reshape(-1, channels)
    mask = _build_match_mask(img, mask_mode, mask_size)
    if mask is None:
        return flat
    masked = flat[mask.reshape(-1)]
    if masked.shape[0] < 2:
        return flat
    return masked


def _mkl_compute_stats(img, sample_points, generator, mask_mode, mask_size):
    channels = img.shape[-1]
    flat = _flatten_with_mask(img, mask_mode, mask_size)
    if sample_points and sample_points > 0:
        flat = _sample_pixels(flat, sample_points, generator)
    if flat.shape[0] < 2:
        mu = flat.mean(0, keepdim=True) if flat.shape[0] > 0 else torch.zeros((1, channels), device=img.device, dtype=img.dtype)
        cov = torch.eye(channels, device=img.device, dtype=img.dtype)
        return mu, cov
    mu = flat.mean(0, keepdim=True)
    centered = flat - mu
    cov = _symmetrize(_covariance_from_centered(centered))
    return mu, cov


def _cap_sinkhorn_points(max_points, backend, device):
    if backend != "tensorized":
        return max_points
    cap = _SINKHORN_MAX_POINTS_TENSORIZED_GPU if device.type == "cuda" else _SINKHORN_MAX_POINTS_TENSORIZED_CPU
    return min(max_points, cap)


def _fit_affine(src_colors, dst_colors):
    channels = src_colors.shape[1]
    ones = torch.ones((src_colors.shape[0], 1), device=src_colors.device, dtype=src_colors.dtype)
    X = torch.cat([src_colors, ones], dim=1)
    try:
        W = torch.linalg.lstsq(X, dst_colors).solution
    except Exception:
        XtX = X.T @ X
        XtY = X.T @ dst_colors
        W = torch.linalg.pinv(XtX) @ XtY
    A = W[:channels, :]
    b = W[channels, :]
    return A, b


def _apply_affine(image, A, b, clamp_min=None, clamp_max=None):
    channels = image.shape[-1]
    flat = image.reshape(-1, channels)
    out = flat @ A + b
    out = out.reshape_as(image)
    if clamp_min is not None or clamp_max is not None:
        min_val = clamp_min if clamp_min is not None else out.min()
        max_val = clamp_max if clamp_max is not None else out.max()
        out = out.clamp(min_val, max_val)
    return out


def _temporal_smooth_transforms(a_list, b_list, alpha):
    if len(a_list) <= 1:
        return a_list, b_list
    smoothed_a = [a_list[0]]
    smoothed_b = [b_list[0]]
    for i in range(1, len(a_list)):
        prev_a = smoothed_a[-1]
        prev_b = smoothed_b[-1]
        cur_a = a_list[i]
        cur_b = b_list[i]
        smoothed_a.append(prev_a * alpha + cur_a * (1.0 - alpha))
        smoothed_b.append(prev_b * alpha + cur_b * (1.0 - alpha))
    return smoothed_a, smoothed_b


def _sinkhorn_plan(src_points, ref_points, blur, iters):
    n = src_points.shape[0]
    m = ref_points.shape[0]
    if n == 0 or m == 0:
        return None

    dtype = src_points.dtype
    device = src_points.device

    a = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    b = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
    log_a = torch.log(a)
    log_b = torch.log(b)

    eps = max(float(blur) ** 2, 1e-4)
    cost = torch.cdist(src_points, ref_points, p=2) ** 2 / 2.0
    log_k = -cost / eps

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(max(1, int(iters))):
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_k.T + log_u.unsqueeze(0), dim=1)

    log_p = log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)
    return torch.exp(log_p)


def _sinkhorn_barycentric(src_sample, ref_sample, blur, iters):
    n = src_sample.shape[0]
    m = ref_sample.shape[0]
    if n == 0 or m == 0:
        return src_sample

    p = _sinkhorn_plan(src_sample, ref_sample, blur, iters)
    if p is None:
        return src_sample
    return (p @ ref_sample) / (p.sum(dim=1, keepdim=True) + 1e-8)


def _sinkhorn_prepare_reference(ref_img, max_points, seed, mask_mode, mask_size):
    backend = "tensorized"
    max_points = _cap_sinkhorn_points(max_points, backend, ref_img.device)
    channels = ref_img.shape[-1]
    ref = _flatten_with_mask(ref_img, mask_mode, mask_size)
    gen_ref = _make_generator(seed + 1, ref_img.device) if seed is not None and seed >= 0 else None
    return _sample_pixels(ref, max_points, gen_ref)


def _sinkhorn_compute_transform(src_img, ref_img, max_points, blur, seed=-1, ref_sample=None, mask_mode="none", mask_size=0):
    backend = "tensorized"
    max_points = _cap_sinkhorn_points(max_points, backend, src_img.device)

    channels = src_img.shape[-1]
    src = _flatten_with_mask(src_img, mask_mode, mask_size)
    if ref_sample is None:
        ref = _flatten_with_mask(ref_img, mask_mode, mask_size)

    gen_src = _make_generator(seed, src_img.device)
    src_sample = _sample_pixels(src, max_points, gen_src).detach()
    if ref_sample is None:
        gen_ref = _make_generator(seed + 1, src_img.device) if seed is not None and seed >= 0 else None
        ref_sample = _sample_pixels(ref, max_points, gen_ref).detach()
    else:
        ref_sample = ref_sample.detach()

    transported = _sinkhorn_barycentric(src_sample, ref_sample, blur, _SINKHORN_ITERS)
    if transported.shape != src_sample.shape:
        return torch.eye(channels, device=src_img.device, dtype=src_img.dtype), torch.zeros(channels, device=src_img.device, dtype=src_img.dtype)

    return _fit_affine(src_sample, transported)


class TS_Color_Match:
    CATEGORY = "TS/Color"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE", {"description": "Референс для подбора цвета. Можно батч; при batch=1 и включенном reuse_reference статистики переиспользуются.", "tooltip": "Эталон, к которому приводится цвет. При batch=1 можно переиспользовать статистики для всего видео."}),
                "target": ("IMAGE", {"description": "Изображение(я), к которому применяется перенос цвета.", "tooltip": "Кадры/изображения, которые будут перекрашены по референсу."}),
                "mode": (_SUPPORTED_MODES, {"default": "mkl", "description": "Алгоритм матчинга: mkl (средние/ковариация) или sinkhorn (OT по сэмплам).", "tooltip": "mkl быстрее и стабильнее; sinkhorn точнее, но тяжелее по памяти и времени."}),
                "device": (_SUPPORTED_DEVICES, {"default": "auto", "description": "Устройство вычислений: auto использует GPU при наличии, иначе CPU.", "tooltip": "auto = GPU если есть, иначе CPU. В режиме GPU большие тензоры остаются на CPU для защиты от OOM."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1, "description": "Сила применения цветокоррекции (0 = без изменений, 1 = полная коррекция).", "tooltip": "Позволяет мягко смешивать исходник и скорректированный цвет."}),
                "enable": ("BOOLEAN", {"default": True, "description": "Включить обработку (false = вернуть target без изменений).", "tooltip": "Если выключено, нода возвращает входной target без коррекции."}),
                "match_mask": (_SUPPORTED_MASK_MODES, {"default": "none", "description": "Область, по которой считается матчинг (none/rectangle/ellipse).", "tooltip": "rectangle/ellipse берут только края изображения шириной mask_size, для стабилизации."}),
                "mask_size": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1, "description": "Толщина рамки в пикселях для match_mask.", "tooltip": "Используется только при match_mask = rectangle или ellipse."}),
                "compute_max_side": ("INT", {"default": _DEFAULT_COMPUTE_MAX_SIDE, "min": 0, "max": 4096, "step": 1, "description": "Максимальная сторона для расчёта трансформации. 0 = без даунскейла. Применяется и к fix, и к reference.", "tooltip": "Считать A,b на уменьшенной копии (например 512–1024). Экономит память при 4K, качество почти не страдает."}),
                "mkl_sample_points": ("INT", {"default": _DEFAULT_MKL_SAMPLE_POINTS, "min": 0, "max": 2000000, "step": 1, "description": "Сэмплирование пикселей для MKL. 0 = все пиксели. Меньше = быстрее и меньше память.", "tooltip": "Рекомендуется 200k–500k. При 0 берутся все пиксели, что может вызвать OOM на 4K."}),
                "sinkhorn_max_points": ("INT", {"default": _DEFAULT_SINKHORN_MAX_POINTS, "min": 0, "max": 65536, "step": 1, "description": "Жёсткий лимит точек для Sinkhorn. 0 = дефолтный лимит.", "tooltip": "Для 4K обычно 1024–2048. Слишком большое значение может дать OOM."}),
                "reuse_reference": ("BOOLEAN", {"default": True, "description": "Если reference batch=1, считать его статистики/сэмплы один раз и использовать для всех кадров.", "tooltip": "Ускоряет и стабилизирует видео, когда референс один на весь батч."}),
                "chunk_size": ("INT", {"default": 4, "min": 0, "max": 256, "step": 1, "description": "Обработка батча по чанкам. 0 = весь батч. Меньше - ниже пик памяти, но медленнее.", "tooltip": "Рекомендуется 4–8 для длинных 4K видео. 0 обрабатывает весь батч разом."}),
                "logging": ("BOOLEAN", {"default": False, "description": "Включить подробное логирование стадий и памяти (для отладки OOM).", "tooltip": "При включении пишет в консоль стадии обработки, номер чанка и состояние памяти GPU."}),
            },
        }

    def _resolve_device(self, device_choice, log_enabled):
        if device_choice == "auto":
            return model_management.get_torch_device()
        if device_choice == "gpu":
            device = model_management.get_torch_device()
            if device.type == "cpu":
                _log_info("GPU selected but no GPU available. Falling back to CPU.", log_enabled)
            return device
        return torch.device("cpu")

    def _compute_transform(
        self,
        fix_img,
        ref_img,
        mode,
        mkl_sample_points,
        sample_seed,
        sinkhorn_max_points,
        log_enabled,
        match_mask,
        mask_size,
        ref_stats=None,
        ref_sample=None,
    ):
        if mode == "mkl":
            return _mkl_compute_transform(
                fix_img,
                ref_img,
                sample_points=mkl_sample_points,
                seed=sample_seed,
                ref_stats=ref_stats,
                mask_mode=match_mask,
                mask_size=mask_size,
            )
        if mode == "sinkhorn":
            if not _SINKHORN_AVAILABLE:
                _log_info("Sinkhorn requested but not available. Falling back to MKL.", log_enabled)
                return _mkl_compute_transform(
                    fix_img,
                    ref_img,
                    sample_points=mkl_sample_points,
                    seed=sample_seed,
                    ref_stats=ref_stats,
                    mask_mode=match_mask,
                    mask_size=mask_size,
                )
            return _sinkhorn_compute_transform(
                fix_img,
                ref_img,
                sinkhorn_max_points,
                _SINKHORN_BLUR,
                seed=sample_seed,
                ref_sample=ref_sample,
                mask_mode=match_mask,
                mask_size=mask_size,
            )
        raise ValueError(f"{_LOG_PREFIX} Unknown mode: {mode}")

    def _process_sequence(
        self,
        fix_image,
        reference_image,
        mode,
        strength,
        match_mask,
        mask_size,
        compute_max_side,
        mkl_sample_points,
        sample_seed,
        sinkhorn_max_points,
        reuse_reference,
        chunk_size,
        compute_device,
        output_device,
        use_threads,
        log_enabled,
    ):
        batch = fix_image.shape[0]
        output = torch.empty((batch, fix_image.shape[1], fix_image.shape[2], fix_image.shape[3]), device=output_device, dtype=fix_image.dtype)

        step = chunk_size if chunk_size and chunk_size > 0 else batch
        num_chunks = int(math.ceil(batch / step)) if step > 0 else 1

        progress_bar = ProgressBar(batch)
        processed_frames = 0

        _log_info(f"stage=sequence_start mode={mode} batch={batch} chunk_size={step} chunks={num_chunks}", log_enabled)
        _log_cuda_memory("stage=sequence_start", compute_device, log_enabled)

        ref_stats = None
        ref_sample = None
        ref_compute = None

        if reuse_reference:
            _log_info("stage=prepare_reference reuse_reference=1", log_enabled)
            ref_compute_cpu = _resize_max_side_single(reference_image[0], compute_max_side)
            ref_compute = _move_to_device(ref_compute_cpu, compute_device)
            if mode == "mkl":
                ref_stats = _mkl_compute_stats(
                    ref_compute,
                    mkl_sample_points,
                    _make_generator(sample_seed + 1, ref_compute.device) if sample_seed is not None and sample_seed >= 0 else None,
                    match_mask,
                    mask_size,
                )
            elif mode == "sinkhorn":
                ref_sample = _sinkhorn_prepare_reference(ref_compute, sinkhorn_max_points, sample_seed, match_mask, mask_size)
            _log_cuda_memory("stage=prepare_reference", compute_device, log_enabled)

        prev_a = None
        prev_b = None

        for chunk_index, start in enumerate(range(0, batch, step), start=1):
            end = min(batch, start + step)
            indices = list(range(start, end))
            _log_info(f"stage=chunk_start chunk={chunk_index}/{num_chunks} frames={start}:{end}", log_enabled)
            _log_cuda_memory(f"stage=chunk_start chunk={chunk_index}", compute_device, log_enabled)

            def _compute_for_index(i):
                fix_img = fix_image[i]
                ref_img = reference_image[0] if reuse_reference else reference_image[i]

                fix_compute_cpu = _resize_max_side_single(fix_img, compute_max_side)
                fix_compute = _move_to_device(fix_compute_cpu, compute_device)
                if reuse_reference:
                    ref_compute_local = ref_compute
                else:
                    ref_compute_cpu = _resize_max_side_single(ref_img, compute_max_side)
                    ref_compute_local = _move_to_device(ref_compute_cpu, compute_device)

                return self._compute_transform(
                    fix_compute,
                    ref_compute_local,
                    mode,
                    mkl_sample_points,
                    sample_seed,
                    sinkhorn_max_points,
                    log_enabled,
                    match_mask,
                    mask_size,
                    ref_stats=ref_stats,
                    ref_sample=ref_sample,
                )

            _log_info(f"stage=compute_transforms chunk={chunk_index}/{num_chunks}", log_enabled)
            if use_threads and len(indices) > 1:
                max_workers = min(len(indices), os.cpu_count() or 4)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    transforms = list(executor.map(_compute_for_index, indices))
            else:
                transforms = [ _compute_for_index(i) for i in indices ]
            _log_cuda_memory(f"stage=compute_transforms chunk={chunk_index}", compute_device, log_enabled)

            _log_info(f"stage=apply_transforms chunk={chunk_index}/{num_chunks}", log_enabled)
            for idx, (A, b) in zip(indices, transforms):
                if prev_a is not None:
                    A = prev_a * _TEMPORAL_EMA + A * (1.0 - _TEMPORAL_EMA)
                    b = prev_b * _TEMPORAL_EMA + b * (1.0 - _TEMPORAL_EMA)
                prev_a, prev_b = A, b
                if output_device == fix_image.device and A.device == fix_image.device and b.device == fix_image.device:
                    corrected = _apply_affine(fix_image[idx], A, b, clamp_min=0.0, clamp_max=1.0)
                    if strength >= 1.0:
                        output[idx] = corrected
                    else:
                        output[idx] = fix_image[idx] + (corrected - fix_image[idx]) * strength
                else:
                    A_out = _move_to_device(A, output_device)
                    b_out = _move_to_device(b, output_device)
                    fix_out = _move_to_device(fix_image[idx], output_device)
                    corrected = _apply_affine(fix_out, A_out, b_out, clamp_min=0.0, clamp_max=1.0)
                    if strength >= 1.0:
                        output[idx] = corrected
                    else:
                        output[idx] = fix_out + (corrected - fix_out) * strength
                progress_bar.update(1)
                processed_frames += 1
            _log_cuda_memory(f"stage=apply_transforms chunk={chunk_index}", compute_device, log_enabled)
            if log_enabled:
                percent = (processed_frames / batch) * 100.0 if batch > 0 else 100.0
                _log_info(f"stage=progress frames={processed_frames}/{batch} ({percent:.1f}%)", log_enabled)

        return output

    def process(
        self,
        reference,
        target,
        mode,
        device,
        strength,
        enable,
        match_mask,
        mask_size,
        compute_max_side,
        mkl_sample_points,
        sinkhorn_max_points,
        reuse_reference,
        chunk_size,
        logging,
    ):
        _validate_image_tensor("reference", reference)
        _validate_image_tensor("target", target)

        log_enabled = bool(logging)

        if not enable:
            return (target.clamp(0.0, 1.0),)

        if mode == "sinkhorn" and not _SINKHORN_AVAILABLE:
            _log_info("Sinkhorn selected but not available. Falling back to MKL.", log_enabled)
            mode = "mkl"

        target_device = self._resolve_device(device, log_enabled)

        original_ref_batch = reference.shape[0]
        original_target_batch = target.shape[0]

        if target_device.type == "cpu":
            reference = reference.to(device=target_device, dtype=torch.float32)
            target = target.to(device=target_device, dtype=torch.float32)
            output_device = target_device
            _log_info("stage=device_policy tensors_on=cpu output_on=cpu", log_enabled)
        else:
            reference = reference.to(device="cpu", dtype=torch.float32)
            target = target.to(device="cpu", dtype=torch.float32)
            output_device = torch.device("cpu")
            _log_info("stage=device_policy tensors_on=cpu compute_on=gpu output_on=cpu", log_enabled)
        _log_cuda_memory("stage=after_device_policy", target_device, log_enabled)

        target, reference, _ = _broadcast_batches(target, reference)

        if sinkhorn_max_points is None or sinkhorn_max_points <= 0:
            sinkhorn_max_points = _SINKHORN_MAX_POINTS

        effective_seed = _FIXED_SAMPLE_SEED

        is_single_pair = (original_ref_batch == 1 and original_target_batch == 1)
        if is_single_pair and mode == "mkl":
            if compute_max_side == _DEFAULT_COMPUTE_MAX_SIDE and mkl_sample_points == _DEFAULT_MKL_SAMPLE_POINTS:
                compute_max_side = 0
                mkl_sample_points = 0
                _log_info("auto_fullres_single=1 (compute_max_side=0, mkl_sample_points=0)", log_enabled)

        _log_info(
            "stage=process_start "
            f"mode={mode} device_choice={device} compute_device={target_device.type} "
            f"strength={strength} compute_max_side={compute_max_side} mkl_sample_points={mkl_sample_points} "
            f"match_mask={match_mask} mask_size={mask_size} sinkhorn_max_points={sinkhorn_max_points} seed={effective_seed} chunk_size={chunk_size} "
            f"reuse_reference={reuse_reference}",
            log_enabled,
        )

        reuse_reference_effective = bool(reuse_reference and original_ref_batch == 1)
        if reuse_reference and not reuse_reference_effective:
            _log_info("reuse_reference requested but reference batch > 1. Ignoring.", log_enabled)

        use_threads = target_device.type == "cpu"

        output = self._process_sequence(
            target,
            reference,
            mode,
            float(strength),
            match_mask,
            int(mask_size),
            compute_max_side,
            mkl_sample_points,
            effective_seed,
            sinkhorn_max_points,
            reuse_reference_effective,
            chunk_size,
            target_device,
            output_device,
            use_threads,
            log_enabled,
        )

        return (output.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "TS_Color_Match": TS_Color_Match,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Color_Match": "TS Color Match",
}
