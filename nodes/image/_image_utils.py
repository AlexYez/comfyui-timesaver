"""Shared image/device/progress helpers for the TS image nodes.

These live here rather than inside a node file because more than one node needs
them: ``ts_bgrm_birefnet`` and ``ts_matting_vitmatte`` share the exact same
tensor<->PIL conversion, device resolution and progress contract, and the two are
documented as drop-in replacements for each other.

The loader skips any path whose component starts with ``_``, so this module is
never registered as a node (CLAUDE.md §7).

``ts_bgrm_birefnet`` re-exports every name below, so pre-existing imports from
that module keep working unchanged.
"""

from __future__ import annotations

import logging

import comfy.model_management as model_management
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)
_LOG_PREFIX = "[TS Image Utils]"


def _get_target_device():
    """Resolve the inference device strictly via ComfyUI's model_management.

    Previously this function silently overrode CPU back to cuda whenever CUDA
    was physically present, which broke `--cpu`, lowvram fallback, and
    multi-GPU index selection. Trusting `model_management.get_torch_device()`
    matches the documented ComfyUI contract — if the user asked for CPU, they
    get CPU; if ComfyUI chose `cuda:N`, that index is preserved.
    """
    try:
        return model_management.get_torch_device()
    except Exception as exc:
        logger.warning("%s Could not resolve ComfyUI device, using CPU: %s", _LOG_PREFIX, exc)
        return torch.device("cpu")


def _format_device_label(target_device):
    device_type = getattr(target_device, "type", str(target_device))
    if device_type == "cuda":
        index = getattr(target_device, "index", None)
        if index is None:
            index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        try:
            name = torch.cuda.get_device_name(index)
        except Exception:
            name = "unknown GPU"
        return f"cuda ({name})"
    return "cpu"


def _safe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _update_progress(progress_bar, value, total=100):
    if progress_bar is not None:
        progress_bar.update_absolute(int(value), total=total)


def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        a = 255
    elif len(hex_color) == 8:
        r, g, b, a = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), int(hex_color[6:8], 16)
    else:
        raise ValueError("Invalid color format")
    return (r, g, b, a)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def _resolve_dtype(target_device, precision: str) -> "torch.dtype":
    """Map the user-facing precision combo to a torch dtype.

    'auto' picks bf16 when the GPU supports it (Ampere+ with Tensor Core
    bf16 — RTX 30/40, A100, H100, etc.), otherwise fp16. CPU always runs
    fp32 — half precision on CPU has no Tensor Core path and is markedly
    slower than fp32 in practice.

    Shared by ``ts_bgrm_birefnet`` and ``ts_matting_vitmatte``. NOTE: BiRefNet
    downgrades bf16 → fp16 via its own ``_resolve_birefnet_dtype`` because
    torchvision's ``deform_conv2d`` has no BF16 CUDA kernel — that wrapper calls
    this function first, so the shared default stays correct for both nodes.
    """
    device_type = getattr(target_device, "type", str(target_device))
    if device_type != "cuda":
        return torch.float32
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    # auto
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _temporal_smooth_alphas(alphas: list, mode: str, ema_alpha: float) -> list:
    """Smooth alpha across the time axis to suppress per-frame edge wobble.

    Per-frame inference (BiRefNet / ViTMatte) has no temporal model, so
    identical objects in adjacent frames produce slightly different alphas —
    the edge "boils". Applied after inference on the already-computed alpha
    sequence:

    - ``median3`` / ``median5`` — N-frame temporal median. Best for random
      flicker; mild lag at clip boundaries (handled by ``mode="nearest"``
      reflection in scipy, or an edge-clamped pure-numpy fallback).
    - ``ema_causal`` — exponential moving average. Causal, no lag, but a sudden
      alpha change blends with the past and can read as motion blur on fast
      objects.
    - ``off`` — passthrough.

    Shared by ``ts_bgrm_birefnet`` and ``ts_matting_vitmatte``.
    """
    n = len(alphas)
    if mode == "off" or n <= 1:
        return alphas
    stack = np.stack(alphas, axis=0)  # [N, H, W]
    if mode in ("median3", "median5"):
        size = 3 if mode == "median3" else 5
        try:
            import scipy.ndimage as _ndi

            stack = _ndi.median_filter(stack, size=(size, 1, 1), mode="nearest")
        except ImportError:
            # Pure-numpy fallback if scipy isn't available.
            radius = size // 2
            padded = np.empty((n + 2 * radius,) + stack.shape[1:], dtype=stack.dtype)
            padded[:radius] = stack[0]
            padded[radius : radius + n] = stack
            padded[radius + n :] = stack[-1]
            out = np.empty_like(stack)
            for i in range(n):
                out[i] = np.median(padded[i : i + size], axis=0)
            stack = out
    elif mode == "ema_causal":
        a = float(np.clip(ema_alpha, 0.0, 0.99))
        if a > 0.0:
            for i in range(1, n):
                stack[i] = a * stack[i - 1] + (1.0 - a) * stack[i]
    return [stack[i] for i in range(n)]
