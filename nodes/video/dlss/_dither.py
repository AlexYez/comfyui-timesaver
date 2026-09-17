"""16-bit to 8-bit quantisation for the DLSS worker.

The worker exchanges RGBA8 frames, while a ComfyUI IMAGE is float32. Rounding
straight to 8 bits turns every smooth gradient into visible steps BEFORE the
network sees it — and the network then sharpens the steps. A tiled blue-noise
ordered dither spends the same 8 bits as noise-shaped detail instead, which the
network can keep.

Vendored from the reference application (``src/video/dither.py``); the noise tile
is deterministic, so two runs of the same graph give the same bytes.
"""

from __future__ import annotations

import numpy as np

_NOISE_SIZE = 64
_NOISE: np.ndarray | None = None


def blue_noise() -> np.ndarray:
    """Deterministic 64×64 tile with a high-pass (blue) spectrum in [0, 1)."""
    global _NOISE  # noqa: PLW0603 - a lazily built constant, not mutable state
    if _NOISE is None:
        rng = np.random.default_rng(20260903)
        white = rng.random((_NOISE_SIZE, _NOISE_SIZE)).astype(np.float32)
        # Remove the low-frequency part with a small box blur, then rank the
        # residual so the distribution is exactly uniform.
        kernel = np.ones((5, 5), dtype=np.float32) / 25.0
        padded = np.pad(white, 2, mode="wrap")
        low = np.zeros_like(white)
        for dy in range(5):
            for dx in range(5):
                low += kernel[dy, dx] * padded[dy:dy + _NOISE_SIZE, dx:dx + _NOISE_SIZE]
        residual = white - low
        order = np.argsort(residual, axis=None)
        ranked = np.empty(residual.size, dtype=np.float32)
        ranked[order] = (np.arange(residual.size, dtype=np.float32) + 0.5) / residual.size
        _NOISE = ranked.reshape(_NOISE_SIZE, _NOISE_SIZE)
    return _NOISE


def _tiled_noise(height: int, width: int) -> np.ndarray:
    """The blue-noise tile, repeated to cover a frame of this size."""
    noise = blue_noise()
    return np.tile(
        noise, (height // _NOISE_SIZE + 1, width // _NOISE_SIZE + 1)
    )[:height, :width]


def quantise_float_to_rgba8(
    rgb: np.ndarray, alpha: np.ndarray | None = None, *, dither: bool
) -> np.ndarray:
    """float32 RGB in 0..1 (and optional alpha) straight to RGBA8, in one pass.

    ⚠️ Мерено: раньше кадр шёл float -> uint16 -> float -> uint8, и промежуточные
    16 бит стоили 51 мс на кадр 1920×1080 — при том что сеть всё равно видит
    восемь. Здесь тот же синий шум, но за один проход.
    """
    if rgb.dtype != np.float32:
        rgb = np.asarray(rgb, dtype=np.float32)
    height, width = rgb.shape[:2]
    scaled = np.empty((height, width, 3), dtype=np.float32)
    np.clip(rgb[..., :3], 0.0, 1.0, out=scaled)
    scaled *= 255.0
    if dither:
        scaled += _tiled_noise(height, width)[..., None]
        np.floor(scaled, out=scaled)
    else:
        np.rint(scaled, out=scaled)
    np.clip(scaled, 0.0, 255.0, out=scaled)

    out = np.empty((height, width, 4), dtype=np.uint8)
    np.copyto(out[..., :3], scaled, casting="unsafe")
    if alpha is None:
        out[..., 3] = 255
    else:
        # Alpha carries no picture information; plain rounding is enough.
        lifted = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0) * 255.0
        np.rint(lifted, out=lifted)
        np.copyto(out[..., 3], lifted, casting="unsafe")
    return out


def dither_rgba16_to_rgba8(rgba16: np.ndarray) -> np.ndarray:
    """Quantise a 16-bit RGBA frame to 8 bits with ordered blue-noise dithering."""
    if rgba16.dtype != np.uint16:
        raise ValueError("dither_rgba16_to_rgba8 expects a uint16 array")
    height, width = rgba16.shape[:2]
    tiled = _tiled_noise(height, width)
    scaled = rgba16.astype(np.float32) * (255.0 / 65535.0)
    out = np.empty_like(rgba16, dtype=np.uint8)
    for channel in range(3):
        out[..., channel] = np.clip(
            np.floor(scaled[..., channel] + tiled), 0, 255
        ).astype(np.uint8)
    # Alpha carries no picture information; plain rounding is enough.
    out[..., 3] = np.clip(np.rint(scaled[..., 3]), 0, 255).astype(np.uint8)
    return out


def quantise_rgba16_to_rgba8(rgba16: np.ndarray, *, dither: bool) -> np.ndarray:
    """Dithered or plainly rounded, depending on what the node was told."""
    if dither:
        return dither_rgba16_to_rgba8(rgba16)
    scaled = rgba16.astype(np.float32) * (255.0 / 65535.0)
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
