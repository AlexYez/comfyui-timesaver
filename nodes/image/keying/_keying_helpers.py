"""Shared math helpers for chroma keying / despill nodes.

Private module: not registered as a public node by the loader (the leading
underscore is honored by `_discover_module_entries` in __init__.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


CHANNEL_TO_INDEX: dict[str, int] = {
    "red": 0,
    "green": 1,
    "blue": 2,
}

INDEX_TO_CHANNEL: tuple[str, str, str] = ("red", "green", "blue")


def gaussian_blur_4d(tensor_bchw: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on a [B, C, H, W] tensor with replicate padding."""
    if sigma <= 0.0:
        return tensor_bchw

    sigma = float(sigma)
    radius = max(1, int(round(sigma * 2.5)))
    x = torch.arange(-radius, radius + 1, device=tensor_bchw.device, dtype=tensor_bchw.dtype)
    kernel_1d = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / torch.clamp(kernel_1d.sum(), min=1e-12)

    channels = tensor_bchw.shape[1]
    kernel_x = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)

    out = F.pad(tensor_bchw, (radius, radius, 0, 0), mode="replicate")
    out = F.conv2d(out, kernel_x, groups=channels)
    out = F.pad(out, (0, 0, radius, radius), mode="replicate")
    out = F.conv2d(out, kernel_y, groups=channels)
    return out
