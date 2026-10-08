"""TS VAE Fix — remove the 2px pixel lattice some VAEs leave on decoded images.

node_id: TS_VAEFix

Several VAE decoders leave a faint lattice with a period of exactly two pixels:
checkerboard and/or stripes at 1-5/255. It comes from the stride of the
decoder's LAST upsample stage, not from the model family, so the same artifact
shows up on Qwen Image, Qwen Image 2.1 / Edit, Krea 2 and the frames of
Wan 2.1. Nothing here is tuned to one model: the node measures the lattice on
every input and does nothing when there is none.

Three steps:

1. **Extract** the Nyquist component with a separable notch. The 9-tap
   binomial kernel with alternating signs has the response sin^8(w/2): exactly
   1 at the 2px period, exactly 0 at DC with an 8th-order flat zero. The
   correction ``Bx + By - Bxy`` makes the overall filter
   (1 - sin^8(wx/2)) * (1 - sin^8(wy/2)) — a hard zero for stripes in either
   direction and for the checkerboard, while gradients pass without banding.
2. **Clamp** the correction to ``limit`` before subtracting it. The lattice is
   small; a real edge produces a large correction, and the clamp lets it
   through almost untouched.
3. **Detect** whether a lattice exists at all. Its phase is locked to the
   decoder's output stride, so it is the same over the whole frame: averaging
   the four (y%2, x%2) sublattices keeps it, while real detail averages out.
   A frame below ``_NEGLIGIBLE`` is passed through bit for bit — the notch is
   cheap but not free (it shaves ~1/255 of genuine fine texture).

A batch is treated as ONE clip: one decision and one auto limit for all
frames. Deciding per frame would make a video flicker — frames near the
threshold would alternate between filtered and untouched.

Approach after the GLSL notch shared by u/Haiku-575 on r/StableDiffusion and
the ComfyUI-DeGrid node by lunaaispace-eng (Apache-2.0); reimplemented here
from the maths, no code copied.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from comfy_api.v0_0_2 import IO, UI

logger = logging.getLogger("comfyui_timesaver.ts_vae_fix")
LOG_PREFIX = "[TS VAE Fix]"

# Alternating binomial(8): response sin^8(w/2), unity at the 2px period.
_TAPS = (1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0)
_PAD = len(_TAPS) // 2

# Raw VAE lattices measure 1.6-5/255; resized or upscaled images 0.1-0.2/255,
# pure noise 0.05/255. The threshold sits in the middle of a ~10x gap.
_NEGLIGIBLE = 0.5 / 255.0

# The detector caps the correction before averaging. A lattice of even 5/255
# stays below it, while a long straight edge — which is phase-coherent too —
# can no longer pose as one. Without the cap a vertical edge on a 128px-wide
# image read 0.52/255, above the threshold; capped, 0.39 (and 0.05 at 1024px).
_DETECT_CAP = 0.02

# Auto limit: 3x the 75th percentile of |correction|, kept within these bounds.
_AUTO_MULT = 3.0
_AUTO_FLOOR = 0.004
_AUTO_CEIL = 0.05

# torch.quantile refuses inputs above 2^24 elements; a strided sample of this
# size is plenty for a percentile.
_MAX_SAMPLES = 1_000_000

_VIEWS = ("4x zoom", "8x zoom", "full frame")
_ZOOM = {"4x zoom": 4, "8x zoom": 8}


def extract_grid(x: torch.Tensor) -> torch.Tensor:
    """Return the 2px-lattice component of ``x`` ([N, C, H, W] float).

    Subtracting the result from ``x`` is the full, unclamped notch filter.
    Too-small frames return zeros: the reflect padding needs more than
    ``_PAD`` pixels on each side.
    """
    _, c, h, w = x.shape
    if h <= 2 * _PAD or w <= 2 * _PAD:
        return torch.zeros_like(x)
    taps = torch.tensor(_TAPS, dtype=x.dtype, device=x.device) / 256.0
    kx = taps.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = taps.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    bx = F.conv2d(F.pad(x, (_PAD, _PAD, 0, 0), mode="reflect"), kx, groups=c)
    by = F.conv2d(F.pad(x, (0, 0, _PAD, _PAD), mode="reflect"), ky, groups=c)
    # The 2D term is separable: the vertical pass over bx replaces a 9x9 conv.
    bxy = F.conv2d(F.pad(bx, (0, 0, _PAD, _PAD), mode="reflect"), ky, groups=c)
    return bx + by - bxy


def lattice_components(corr: torch.Tensor) -> dict[str, float]:
    """Phase-locked lattice amplitude of one frame's correction ([1, C, H, W]).

    Returns peak-to-peak amplitude plus its checkerboard / vertical-stripe /
    horizontal-stripe parts, in 0..1 units, from the strongest channel.
    """
    h = corr.shape[2] // 2 * 2
    w = corr.shape[3] // 2 * 2
    if h < 2 or w < 2:
        return {"amp": 0.0, "checker": 0.0, "vstripe": 0.0, "hstripe": 0.0}
    c = corr[0, :, :h, :w].clamp(-_DETECT_CAP, _DETECT_CAP)
    m00 = c[:, 0::2, 0::2].mean(dim=(1, 2))
    m01 = c[:, 0::2, 1::2].mean(dim=(1, 2))
    m10 = c[:, 1::2, 0::2].mean(dim=(1, 2))
    m11 = c[:, 1::2, 1::2].mean(dim=(1, 2))
    stacked = torch.stack([m00, m01, m10, m11], dim=-1)
    amp = (stacked.amax(-1) - stacked.amin(-1)).amax()
    checker = (((m00 + m11) - (m01 + m10)) / 2).abs().amax()
    vstripe = (((m00 + m10) - (m01 + m11)) / 2).abs().amax()
    hstripe = (((m00 + m01) - (m10 + m11)) / 2).abs().amax()
    return {
        "amp": float(amp),
        "checker": float(checker),
        "vstripe": float(vstripe),
        "hstripe": float(hstripe),
    }


def auto_limit(corr: torch.Tensor) -> float:
    """Clamp limit for one frame from a robust estimate of the lattice level.

    Smooth regions dominate a photograph, so the 75th percentile of |corr|
    tracks the artifact amplitude; edges are the outliers above it.
    """
    flat = corr.abs().reshape(-1)
    if flat.numel() == 0:
        return _AUTO_FLOOR
    if flat.numel() > _MAX_SAMPLES:
        flat = flat[:: flat.numel() // _MAX_SAMPLES + 1]
    q = float(torch.quantile(flat.float(), 0.75))
    return min(max(q * _AUTO_MULT, _AUTO_FLOOR), _AUTO_CEIL)


def zoom_center(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Nearest-neighbour magnified centre crop of an [B, H, W, C] batch.

    A 2px lattice is invisible in a node thumbnail of a full frame; magnified
    it reads as the pattern it is. Output is roughly the input size.
    """
    _, h, w, _ = x.shape
    ch, cw = max(2, h // factor), max(2, w // factor)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = x[:, y0:y0 + ch, x0:x0 + cw, :]
    return crop.repeat_interleave(factor, dim=1).repeat_interleave(factor, dim=2)


def _frame_chw(image: torch.Tensor, index: int, channels: int) -> torch.Tensor:
    return image[index:index + 1, :, :, :channels].permute(0, 3, 1, 2).float()


def remove_grid(
    image: torch.Tensor,
    mode: str = "auto",
    limit: float = 0.02,
    skip_when_clean: bool = True,
    preview_gain: float = 10.0,
    preview_view: str = "4x zoom",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Remove the 2px lattice from an IMAGE batch ([B, H, W, C], 0..1).

    Works frame by frame, so memory stays at a few copies of ONE frame rather
    than of the whole batch. Colour channels only: a 4th (alpha) channel is
    copied through untouched. The input tensor is never modified.

    Returns ``(cleaned, removed_grid, stats)``. When the batch carries no
    lattice and ``skip_when_clean`` is on, ``cleaned`` IS the input tensor.
    """
    batch, height, width, total_c = image.shape
    color_c = min(total_c, 3)

    # Pass 1: measure. Nothing is kept but a few numbers per frame.
    measured = []
    for i in range(batch):
        corr = extract_grid(_frame_chw(image, i, color_c))
        parts = lattice_components(corr)
        parts["limit"] = auto_limit(corr)
        measured.append(parts)
        del corr

    amp = max((m["amp"] for m in measured), default=0.0)
    detected = amp >= _NEGLIGIBLE
    if mode == "manual":
        lim = float(limit)
    else:
        lim = float(torch.tensor([m["limit"] for m in measured]).median()) if measured else _AUTO_FLOOR
    strongest = max(measured, key=lambda m: m["amp"]) if measured else {}
    stats = {
        "frames": batch,
        "amp_255": amp * 255.0,
        "checker_255": strongest.get("checker", 0.0) * 255.0,
        "vstripe_255": strongest.get("vstripe", 0.0) * 255.0,
        "hstripe_255": strongest.get("hstripe", 0.0) * 255.0,
        "detected": detected,
        "limit": lim,
        "mode": mode,
        "clipped_pct": 0.0,
        "skipped": False,
    }

    if not detected and skip_when_clean:
        stats["skipped"] = True
        gray = torch.full_like(image, 0.5)
        return image, _frame_preview(gray, preview_view), stats

    # Pass 2: apply. The correction is recomputed rather than kept from pass 1,
    # which would hold a full-batch float32 copy for the whole run.
    cleaned = image.clone()
    removed = torch.full_like(image, 0.5)
    clipped_total = 0.0
    for i in range(batch):
        frame = _frame_chw(image, i, color_c)
        corr = extract_grid(frame)
        clipped_total += float((corr.abs() > lim).float().mean())
        corr = corr.clamp(-lim, lim)
        out = (frame - corr).clamp(0.0, 1.0)
        vis = (corr * float(preview_gain) + 0.5).clamp(0.0, 1.0)
        cleaned[i, :, :, :color_c] = out[0].permute(1, 2, 0).to(image.dtype)
        removed[i, :, :, :color_c] = vis[0].permute(1, 2, 0).to(image.dtype)
        del frame, corr, out, vis
    stats["clipped_pct"] = clipped_total / max(batch, 1) * 100.0
    return cleaned, _frame_preview(removed, preview_view), stats


def _frame_preview(removed: torch.Tensor, view: str) -> torch.Tensor:
    factor = _ZOOM.get(view)
    if factor is None:
        return removed
    return zoom_center(removed, factor)


def status_line(stats: dict) -> str:
    """One readable line for the node: what was measured and what was done."""
    amp = stats["amp_255"]
    if not stats["detected"]:
        action = "passed through untouched" if stats["skipped"] else "filtered anyway"
        line = f"grid {amp:.2f}/255 - none detected, {action}"
    else:
        # The dominant orientation hints at which decoder stage left the lattice.
        kind = max(
            (("checker", stats["checker_255"]),
             ("V-stripe", stats["vstripe_255"]),
             ("H-stripe", stats["hstripe_255"])),
            key=lambda p: p[1],
        )[0]
        line = f"grid {amp:.2f}/255 ({kind}) - removed, limit {stats['limit']:.3f} {stats['mode']}"
    if not stats["skipped"]:
        line += f" | edges protected: {stats['clipped_pct']:.1f}%"
    if stats["frames"] > 1:
        line += f" | {stats['frames']} frames, one decision for the batch"
    return line


class TS_VAEFix(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_VAEFix",
            display_name="TS VAE Fix",
            category="TS/Image/Retouch",
            description=(
                "Remove the faint 2-pixel grid that some VAEs leave on decoded images: "
                "Qwen Image, Qwen Image 2.1 / Edit, Krea 2, Wan 2.1 frames. Put it right "
                "after VAE Decode, before any resize, sharpening or upscaler. The node "
                "measures the grid on every run and leaves images without one untouched, "
                "so it is safe to keep in any workflow."
            ),
            search_aliases=[
                "vae fix", "degrid", "pixel grid", "grid artifact", "checkerboard", "vae grid",
                "qwen vae", "krea", "wan vae", "notch",
            ],
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Straight from VAE Decode. A batch of video frames is treated as one clip.",
                ),
                IO.Combo.Input(
                    "mode",
                    options=["auto", "manual"],
                    default="auto",
                    tooltip=(
                        "auto: the removal limit is measured from the image itself. "
                        "manual: use the 'limit' value below."
                    ),
                ),
                IO.Float.Input(
                    "limit",
                    default=0.02,
                    min=0.0,
                    max=0.1,
                    step=0.001,
                    tooltip=(
                        "Manual mode only. Largest correction per pixel, on the 0-1 scale. "
                        "Too low: the grid survives in contrasty areas. Too high: the finest "
                        "texture (pores, fabric) softens slightly."
                    ),
                ),
                IO.Boolean.Input(
                    "skip_when_clean",
                    default=True,
                    tooltip=(
                        "Pass the image through bit for bit when no grid is found, e.g. after "
                        "an upscaler or a resize. Off: filter regardless."
                    ),
                ),
                IO.Float.Input(
                    "preview_gain",
                    default=10.0,
                    min=1.0,
                    max=50.0,
                    step=1.0,
                    tooltip="Brightness of the removed_grid preview only. Never changes the image.",
                ),
                IO.Combo.Input(
                    "preview_view",
                    options=list(_VIEWS),
                    default="4x zoom",
                    tooltip=(
                        "Framing of the removed_grid preview. A 2px grid turns into gray noise "
                        "in a full-frame thumbnail; a magnified centre crop shows the pattern."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="image",
                    tooltip="The cleaned image, same size and batch as the input.",
                ),
                IO.Image.Output(
                    display_name="removed_grid",
                    tooltip=(
                        "What was subtracted, amplified and centred on gray. Healthy: a uniform "
                        "fine pattern. Recognisable faces or edges here mean the limit is too high."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mode: str = "auto",
        limit: float = 0.02,
        skip_when_clean: bool = True,
        preview_gain: float = 10.0,
        preview_view: str = "4x zoom",
    ) -> IO.NodeOutput:
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            shape: Optional[tuple] = tuple(image.shape) if isinstance(image, torch.Tensor) else None
            raise ValueError(f"{LOG_PREFIX} Expected IMAGE [B, H, W, C], got {shape or type(image).__name__}.")
        if preview_view not in _VIEWS:
            preview_view = "4x zoom"
        with torch.no_grad():
            cleaned, removed, stats = remove_grid(
                image,
                mode=mode,
                limit=limit,
                skip_when_clean=skip_when_clean,
                preview_gain=preview_gain,
                preview_view=preview_view,
            )
        line = status_line(stats)
        logger.info("%s %s", LOG_PREFIX, line)
        return IO.NodeOutput(cleaned, removed, ui=UI.PreviewText(line))


NODE_CLASS_MAPPINGS = {"TS_VAEFix": TS_VAEFix}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_VAEFix": "TS VAE Fix"}
