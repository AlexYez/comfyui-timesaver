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
