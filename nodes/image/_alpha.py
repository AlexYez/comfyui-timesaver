"""Work on straight-alpha ComfyUI images without sampling invisible colour."""

import torch


def premultiply(image: torch.Tensor, channel_dim: int = -1) -> torch.Tensor:
    if image.size(channel_dim) != 4:
        return image
    rgb, alpha = image.split((3, 1), dim=channel_dim)
    return torch.cat((rgb * alpha, alpha), dim=channel_dim)


def unpremultiply(image: torch.Tensor, channel_dim: int = -1) -> torch.Tensor:
    if image.size(channel_dim) != 4:
        return image
    rgb, alpha = image.split((3, 1), dim=channel_dim)
    visible = alpha > 1e-6
    rgb = rgb / torch.where(visible, alpha, 1.0)
    rgb = torch.where(visible, rgb, 0.0).clamp(0.0, 1.0)
    return torch.cat((rgb, alpha.clamp(0.0, 1.0)), dim=channel_dim)


def to_channels(image: torch.Tensor, channels: int) -> torch.Tensor:
    """Promote gray to colour and RGB to opaque RGBA when joining batches."""
    if image.shape[-1] == 1 and channels >= 3:
        image = image.expand(*image.shape[:-1], 3)
    if image.shape[-1] == 3 and channels == 4:
        image = torch.cat((image, torch.ones_like(image[..., :1])), dim=-1)
    return image


def mix_rgba(base: torch.Tensor, patch: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Feather between two RGBA layers, preserving exact pixels at either end."""
    mixed = unpremultiply(premultiply(patch) * weight + premultiply(base) * (1.0 - weight))
    return torch.where(weight >= 1, patch, torch.where(weight <= 0, base, mixed))


def resize_with_alpha(image: torch.Tensor, width: int, height: int, method: str,
                      crop: str = "disabled") -> torch.Tensor:
    """Resize a BHWC image. Pillow Lanczos already handles straight RGBA."""
    from comfy.utils import common_upscale

    pixels = image.movedim(-1, 1)
    if image.shape[-1] != 4 or method in ("lanczos", "nearest", "nearest-exact"):
        return common_upscale(pixels, width, height, method, crop).movedim(1, -1)
    # Spherical interpolation couples channels, which is wrong for opacity.
    method = "bilinear" if method == "bislerp" else method
    output = common_upscale(premultiply(pixels, 1), width, height, method, crop)
    return unpremultiply(output, 1).movedim(1, -1)


def checker_preview(frame):
    """Show transparency in a uint8 preview aimed at an RGB-only player."""
    if frame.shape[-1] != 4:
        return frame
    import numpy as np
    from PIL import Image

    height, width = frame.shape[:2]
    block = max(2, min(12, min(height, width) // 8))
    tile = (np.arange(height)[:, None] // block + np.arange(width)[None, :] // block) % 2
    matte = Image.fromarray(np.where(tile, 128, 192).astype(np.uint8)).convert("RGBA")
    return np.ascontiguousarray(np.asarray(Image.alpha_composite(matte, Image.fromarray(frame)).convert("RGB")))
