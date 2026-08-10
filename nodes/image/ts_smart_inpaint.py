"""TS Smart Inpaint — single-file headless port of the "Xtra-Fine" "Smart
Inpaint" / "Refine" path.

Given the FULL source image + a painted MASK (white = inpaint) + the model / vae /
positive / negative, this node reproduces the algorithm BYTE-FOR-BYTE — the
`_refine_with_fine_upscaling` helper + its sampling / VAE / mask helpers are
extracted verbatim BELOW (deliberately kept in this ONE file — it's small enough
that a single file reads clearer than a split):

  bbox(mask) + context_pct band -> crop pixels -> resize to the `megapixels`
  budget (small crops upscale, capped at `max_linear`; oversized crops downscale,
  so the VAE/sampler workload is bounded at any mask size) -> VAE-encode ->
  [Smart Inpaint: reference_latents = the crop + zero the masked latent] ->
  noise-injection inpaint (denoise) -> VAE-decode -> resize back to native bbox
  -> [color_correct: neutralise Flux's per-channel colour shift from the
  preserved ring] -> feather-composite back into the source frame (the composite
  IS the output; no full-frame re-encode — see A0).

`replace` (checkbox, label_on="Replace" / label_off="Refine"):
  - Replace (ON)  = "Smart Inpaint": reference_latents = the crop + zero the
    masked latent, regenerates the painted region as a Kontext edit. The
    `denoise` widget is IGNORED and locked to 1.0.
  - Refine (OFF)  = standard Xtra-Fine refine (ADetailer-style) of the painted
    region at the `denoise` value, no reference.

`reference` (optional IMAGE, Replace only): when connected, it is VAE-encoded and
CHAINED as a SECOND `reference_latents` entry (after the crop) — "fill the hole
with THIS picture". Left unconnected (or fed a non-image), the second reference
chain is simply not engaged and Replace behaves exactly as above.

Colour correction (always ON — module constant `_COLOR_CORRECT`; `max_linear` is
likewise the constant `_MAX_LINEAR` — both hoisted out of the UI): the
Flux VAE round-trip introduces a slight systematic colour shift (mild reddening)
on the decoded patch. This estimates that shift from the PRESERVED ring around
the mask — pixels inside the crop but outside the painted region, where the
decoded patch and the original are the SAME content, so their per-channel
difference is purely Flux's shift (independent of whatever was generated inside
the mask). The inverse per-channel gain+offset is applied to the patch before
compositing, so the new content matches the surrounding colours WITHOUT being
tinted toward them (a red object on a blue surround stays red). To anchor the
estimate the crop is grown by a small analysis margin when `color_correct` is on.
OFF skips the correction.

`megapixels` is always a true processing cap: an oversized masked crop (e.g. a
big selection in a 6-8K image) is downscaled to the budget before encode/sample
and scaled back up for the composite — so the node never hangs/OOMs on a large
mask. Small crops upscale toward the budget (capped by `max_linear`). Raise
`megapixels` for more detail.

The crop + composite happen INSIDE the node, so the app just uploads the full
source + mask (+ optional reference).

Algorithm credit: the "Xtra-Fine" inpaint path of ComfyUI-Angelo
(shootthesound/ComfyUI-Angelo, angelo_nodes.py) — the helpers below are taken
from its `_refine_with_fine_upscaling` so the behaviour matches it, with ONE
local extension (the optional `extra_reference_latents` chaining used in Replace
mode). Upstream is MIT-licensed; its notice is reproduced below as required:

    MIT License — Copyright (c) 2026 Peter Neill

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

node_id: TSSmartInpaint
"""
from __future__ import annotations

import logging
import math

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import node_helpers
import torch
from comfy_api.v0_0_2 import IO

# The geometry of this node — crop by mask, normalise to a megapixel budget,
# feather back in linear light — is not specific to the sampler below it, so it
# lives in _inpaint_crop.py and is shared with the studio's LanPaint backends.
# These names are the module's, re-bound to the private spellings this file has
# always used; there is exactly ONE implementation of each.
from ._inpaint_crop import (  # noqa: E402  (kept next to the constants it replaces)
    CC_ANALYSIS_MARGIN_PX as _CC_ANALYSIS_MARGIN_PX,
    CONTEXT_CEIL_PX as _CONTEXT_CEIL_PX,
    FEATHER_CEIL_PX as _FEATHER_CEIL_PX,
    FEATHER_FLOOR_PX as _FEATHER_FLOOR_PX,
    MAX_LINEAR as _MAX_LINEAR,
    RESIZE_METHODS as _FINE_UPSCALE_RESIZE_METHODS,
    color_correct_patch as _color_correct_patch,
    downscale_to_megapixels as _downscale_pixels_to_megapixels,
    fine_upscale_factor as _fine_upscale_factor_px,
    gaussian_blur_2d as _gaussian_blur_2d,
    linear_to_srgb as _linear_to_srgb,
    mask_bbox as _mask_bbox_latent,
    pct_to_px as _pct_to_px,
    resize_spatial as _resize_latent,
    srgb_to_linear as _srgb_to_linear,
)

_COLOR_CORRECT = True   # always neutralise Flux's colour shift (see color_correct_patch)


def _do_sample(
        *,
        model,
        noise: torch.Tensor,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive,
        negative,
        source_latent: torch.Tensor,
        denoise: float,
        callback,
        disable_pbar: bool,
        seed: int,
        noise_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single sample dispatch for every Angelo sample call — the standard
    ``comfy.sample.sample(...)`` path that has existed since v1.0."""
    return comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, source_latent,
        denoise=denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )


def _fine_upscale_factor(
    bbox_w_latent: int,
    bbox_h_latent: int,
    scale_x: float,
    scale_y: float,
    target_mp: float,
    max_linear: float,
) -> float:
    """Latent-space wrapper over the shared pixel-space factor: this node's
    geometry lives in latents, so the bbox is converted to image pixels first."""
    if scale_x <= 0 or scale_y <= 0:
        return 1.0
    return _fine_upscale_factor_px(
        bbox_w_latent / scale_x, bbox_h_latent / scale_y, target_mp, max_linear,
    )


def _vae_decode(vae, latent: torch.Tensor) -> torch.Tensor:
    """Decode a latent to pixels. Single decode chokepoint — see the
    VAE-boundary note above. Always returns a 4D image batch
    (B, H, W, C) float in [0, 1].

    Temporal/video VAEs (Qwen Image Edit, Wan) keep a frame axis: their
    latents are 5D ([B, C, T, H, W]) and `vae.decode` accordingly returns
    a 5D frame stack ([B, T, H, W, C] — ComfyUI moves channels last). The
    rest of the node, and ComfyUI's PreviewImage/PIL path, only understand
    4D image batches, so fold the frame axis into the batch dim. For image
    editing T is 1, so this is just dropping the singleton frame axis; if a
    future model ever produces T>1 the frames surface as extra batch items
    rather than crashing. The latent is passed through to `vae.decode`
    untouched — the video VAE wants its native 5D input — we only normalise
    the *pixels* it returns."""
    image = vae.decode(latent)
    if image.ndim == 5:
        b, t, h, w, c = image.shape
        image = image.reshape(b * t, h, w, c)
    return image


def _vae_encode(vae, pixels: torch.Tensor) -> torch.Tensor:
    """Encode pixels to latent samples. Single encode chokepoint —
    counterpart to _vae_decode. See the VAE-boundary note above.

    Deliberately returns the VAE's native latent shape WITHOUT collapsing
    it: a temporal/video VAE (Qwen, Wan) returns a 5D latent
    ([B, C, T, H, W]) and the sampler + model require that 5D shape to flow
    through unchanged (comfy.sample.sample is ndim-agnostic and prepare_noise
    matches the latent's shape exactly). Squeezing the frame axis here would
    break Qwen sampling — do not add a squeeze."""
    return vae.encode(pixels)


def _refine_with_fine_upscaling(
    *,
    model,
    vae,
    current: torch.Tensor | None,        # full-res latent; None under A1 (lazily encoded ONLY by the no-upscale Refine path). Geometry now comes from `mask` + `current_pixels`.
    current_pixels: torch.Tensor,        # [B, H_pix, W_pix, C] full-res pixels — the canvas this node composites into and returns
    mask: torch.Tensor,                  # [1, H_lat, W_lat] feathered mask, latent res
    scale_x: float,
    scale_y: float,
    target_mp: float,
    max_linear: float,
    resize_method: str,
    context_pad_pixel: int,
    inpainting_mode: str,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive,
    negative,
    denoise: float,
    callback,
    disable_pbar: bool,
    # LOCAL extension (not from upstream): extra reference_latents to CHAIN after
    # the crop's own reference in Smart Inpaint mode — e.g. a user "fill with
    # THIS picture" image. None / empty = behaves exactly like upstream.
    extra_reference_latents=None,
    # LOCAL option (not from upstream): when True, neutralise Flux's per-channel
    # colour shift on the decoded patch using the preserved ring (see
    # _color_correct_patch) right before the composite.
    color_correct: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pixel-space crop + upscale + VAE encode + refine + VAE decode +
    downscale + composite. The latent-space crop+upscale
    approach smears bilinearly-interpolated latents into a low-freq
    starting state that the model can't recover detail from. Going
    through pixel space (where there's an image-upscale toolkit that's
    been tuned for natural images) and re-encoding gives the model a
    "natural" latent at the higher resolution to denoise from.

    Returns (new_latent, new_pixels). `new_pixels` is the feathered composite —
    the node's actual output, always non-None on the processing path (A1: the
    caller only ever consumes `new_pixels`). `new_latent` is always None here.
    On an empty / degenerate mask returns (current, current_pixels) unchanged.
    """
    bbox = _mask_bbox_latent(mask)
    if bbox is None:
        return current, current_pixels
    y0_tight, y1_tight, x0_tight, x1_tight = bbox

    # Apply context padding: grow the bbox outward by context_pad_pixel
    # in every direction (clamped to the latent boundaries). This is
    # the area the model SEES during refine. The painted-shape mask
    # stays unchanged — areas inside the padded bbox but outside the
    # painted shape have mask=0 in the cropped tensor, so the noise-
    # injection inpaint preserves them as context (the model uses them
    # to inform what to draw inside the mask, but doesn't overwrite
    # them). All downstream code uses the PADDED bbox.
    # A1: latent dims come from the feathered `mask` (already at latent res),
    # so the core no longer needs a full-image `current` just to read its shape.
    H_lat = mask.shape[-2]
    W_lat = mask.shape[-1]
    pad_lat_y = max(0, round(context_pad_pixel * scale_y))
    pad_lat_x = max(0, round(context_pad_pixel * scale_x))
    y0 = max(0, y0_tight - pad_lat_y)
    y1 = min(H_lat, y1_tight + pad_lat_y)
    x0 = max(0, x0_tight - pad_lat_x)
    x1 = min(W_lat, x1_tight + pad_lat_x)

    bbox_h_lat = y1 - y0
    bbox_w_lat = x1 - x0
    if bbox_h_lat <= 0 or bbox_w_lat <= 0:
        return current, current_pixels

    # `megapixels` is a true cap (see _fine_upscale_factor): small crops upscale
    # toward it, oversized crops scale DOWN to it. Every mode goes through the
    # crop+composite path below — there is no whole-frame latent shortcut (it
    # would sample the entire 6-8K latent and hang; the bounded crop path is
    # always used instead).
    scale = _fine_upscale_factor(
        bbox_w_lat, bbox_h_lat, scale_x, scale_y, target_mp, max_linear,
    )

    # ----- VAE decode the full cached latent → cached pixels -----
    # Optimization: Reuse cached pixels if available to prevent VAE degradation 
    # (loss of high-frequency details) across multiple consecutive edits.
    if current_pixels is not None:
        cached_pixels = current_pixels
    elif current is not None:
        cached_pixels = _vae_decode(vae, current)  # (B, H_pix, W_pix, C) float [0,1]
    else:
        raise ValueError(
            "_refine_with_fine_upscaling needs current_pixels or current (both None)"
        )
        
    H_pix = cached_pixels.shape[1]
    W_pix = cached_pixels.shape[2]
    # Pixel-per-latent ratio per axis (16 for FLUX 2, 8 for SDXL/SD1.5).
    # round(), not floor-divide: a non-integer true ratio (exotic VAEs)
    # floor-divided gives e.g. 15 for 15.8, drifting the pixel-space bbox
    # ~1px against the latent bbox and leaving a seam in the composite.
    # (#28, from @KursatAs.)
    px_per_lat_y = max(1, round(H_pix / H_lat))
    px_per_lat_x = max(1, round(W_pix / W_lat))

    # Pixel-space bbox derived from the latent-space bbox.
    y0_p = y0 * px_per_lat_y
    y1_p = y1 * px_per_lat_y
    x0_p = x0 * px_per_lat_x
    x1_p = x1 * px_per_lat_x
    bbox_h_p = y1_p - y0_p
    bbox_w_p = x1_p - x0_p

    # Upscaled target dims in pixel space. Snap to multiples of the
    # VAE downscale (16 for FLUX 2) so the subsequent VAE encode
    # produces a clean integer-dim latent.
    vae_snap = max(px_per_lat_y, px_per_lat_x)
    target_h_p = max(vae_snap, math.ceil(bbox_h_p * scale / vae_snap) * vae_snap)
    target_w_p = max(vae_snap, math.ceil(bbox_w_p * scale / vae_snap) * vae_snap)

    logger.debug(
        "%s fine-upscale: bbox_lat=(h=%s, w=%s) bbox_px=(h=%s, w=%s) scale=%.2f "
        "target_px=(h=%s, w=%s) resize=%s max_linear=%s vae_ratio=(x=%s, y=%s)",
        LOG_PREFIX, bbox_h_lat, bbox_w_lat, bbox_h_p, bbox_w_p, scale,
        target_h_p, target_w_p, resize_method, max_linear, px_per_lat_x, px_per_lat_y,
    )

    # ----- Crop pixel image + upscale in pixel space -----
    pixel_crop = cached_pixels[:, y0_p:y1_p, x0_p:x1_p, :]  # (B, h, w, C)
    # common_upscale expects (B, C, H, W) — permute, upscale, permute back.
    pixel_crop_chw = pixel_crop.movedim(-1, 1)
    pixel_crop_up_chw = comfy.utils.common_upscale(
        pixel_crop_chw, target_w_p, target_h_p, resize_method, "disabled",
    )
    pixel_crop_up = pixel_crop_up_chw.movedim(1, -1)  # back to (B, H, W, C)

    # ----- VAE encode the upscaled pixel crop → latent at high res -----
    latent_up = _vae_encode(vae, pixel_crop_up)
    target_h_lat = latent_up.shape[-2]
    target_w_lat = latent_up.shape[-1]

    # ----- Build mask at the upscaled latent resolution -----
    # Mask resizing always uses bilinear regardless of the user's choice.
    # The user's resize_method is for the IMAGE content upscale (where
    # lanczos / bicubic / etc. have real quality differences). The mask
    # is a 1-channel feathered alpha where we just want smooth values;
    # lanczos's grayscale-branch returns a transposed 3D tensor (PIL
    # quirk) and bislerp's spherical-vector math is semantically wrong
    # on a single channel.
    mask_crop = mask[..., y0:y1, x0:x1].contiguous()
    mask_crop_up = _resize_latent(mask_crop, target_h_lat, target_w_lat, "bilinear").clamp(0.0, 1.0)

    # ===== Smart Inpaint pre-processing on the upscaled patch =====
    # Klein 9B's edit branch only activates when reference_latents is
    # present on the conditioning. We then zero the masked area so the
    # sampler regenerates that region from full noise at sigma_max
    # (the denoise=1.0 lock makes this clean: every pixel in the
    # painted rect is brand-new content, with the surrounding context
    # band restored each step by the noise_mask compositing). The
    # reference uses the PRE-ZERO upscaled patch so Klein still sees
    # what was there before we blanked it.
    # POSITIVE ONLY — putting reference_latents on negative would tell
    # CFG>1 samplers to steer AWAY from the reference scene. Non-edit
    # models ignore the field, so this is harmless on any checkpoint.
    #
    # append=False (REPLACE, not append): the reference must be ONLY this
    # upscaled crop. When the Area Prompt is empty, refine_positive falls back
    # to the node's `positive` input, which in a Klein edit workflow already
    # carries reference_latents = the WHOLE source image (from an upstream
    # ReferenceLatent node). append=True stacked the crop onto that whole-image
    # reference, and the whole-image one dominated — so the patch reproduced
    # the entire original scene instead of editing the selected region.
    # Replacing guarantees Klein sees the crop and nothing else.
    # SAMPLING mask (both the masked-zero and the noise_mask) is HARD (binary):
    # the painted region is fully regenerated and the feather band is NOT
    # half-zeroed / half-denoised. A SOFT sampling mask scaled the feather-band
    # latent toward the mean (≈ mid-gray on Flux) and only partially regenerated
    # it, which decoded to a dark, muddy line along the mask edge. The visible
    # feather is produced ONLY by the soft PIXEL composite downstream, over this
    # clean content — the recipe the 5D temporal path (Qwen/Wan) always used, now
    # applied to 4D (FLUX/Klein) too. The composite still uses the full-res
    # feathered `mask`, so the soft edge survives without the sampling muddiness.
    sample_mask = (mask_crop_up >= 0.5).to(mask_crop_up.dtype)
    if inpainting_mode == "Smart Inpaint":
        reference_latent = latent_up.clone()  # PRE-zero patch — Klein's reference
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": [reference_latent]}, append=False,
        )
        # LOCAL extension: CHAIN extra reference latents (e.g. a user "fill with
        # THIS picture" image) AFTER the crop's own reference. append=True so the
        # crop stays first (scene/structure) and the user reference rides along
        # (content). Empty/None → no-op → identical to upstream.
        if extra_reference_latents:
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": list(extra_reference_latents)}, append=True,
            )
        # Zero ONLY the painted region (binary sample_mask) so Klein regenerates it
        # from full noise; the context band is preserved intact (no gray pull).
        latent_up = (1.0 - sample_mask.unsqueeze(0)) * latent_up

    # ----- Refine via noise-injection inpaint on the upscaled latent -----
    noise = comfy.sample.prepare_noise(latent_up, seed, None)
    refined_latent_up = _do_sample(
        model=model, noise=noise,
        steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
        positive=positive, negative=negative,
        source_latent=latent_up,
        denoise=denoise,
        noise_mask=sample_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )

    # ----- VAE decode refined latent → high-res pixel patch -----
    refined_pixel_up = _vae_decode(vae, refined_latent_up)  # (B, target_h_p, target_w_p, C)

    # ----- Downscale refined patch back to original bbox pixel size -----
    refined_pixel_up_chw = refined_pixel_up.movedim(-1, 1)
    refined_pixel_chw = comfy.utils.common_upscale(
        refined_pixel_up_chw, bbox_w_p, bbox_h_p, resize_method, "disabled",
    )
    refined_pixel = refined_pixel_chw.movedim(1, -1)  # (B, bbox_h_p, bbox_w_p, C)

    # ----- Composite refined patch into the cached pixel image -----
    # Build a pixel-space alpha by resizing the latent feathered mask to
    # full pixel resolution, cropping to the bbox. Always bilinear for
    # the same reasons as the mask upscale above — lanczos's grayscale
    # path is broken, bislerp doesn't apply to 1-channel.
    mask_4d = mask.unsqueeze(0)  # [1, 1, H_lat, W_lat]
    pixel_mask = comfy.utils.common_upscale(
        mask_4d, W_pix, H_pix, "bilinear", "disabled",
    ).clamp(0.0, 1.0)  # [1, 1, H_pix, W_pix]
    pixel_alpha_crop = pixel_mask[0, 0, y0_p:y1_p, x0_p:x1_p]  # [bbox_h_p, bbox_w_p]
    pixel_alpha_crop = pixel_alpha_crop.unsqueeze(0).unsqueeze(-1)  # [1, h, w, 1]

    new_pixels = cached_pixels.clone()
    pixel_orig_crop = cached_pixels[:, y0_p:y1_p, x0_p:x1_p, :]
    # Neutralise Flux's per-channel colour shift (the mild reddening) on the
    # decoded patch BEFORE compositing. The estimate comes from the preserved
    # ring (alpha≈0), where refined and original are the SAME content — so it
    # measures only Flux's shift and never drifts with the generated content's
    # colours. No-op when color_correct is off or the ring is too small.
    if color_correct:
        refined_pixel = _color_correct_patch(refined_pixel, pixel_orig_crop, pixel_alpha_crop)
    # Composite in LINEAR light: alpha-blending in gamma (sRGB) space darkens the
    # cross-fade and leaves a dark line along the feather. Linearise both sides,
    # blend, re-encode. At alpha 0/1 this round-trips to the input, so only the
    # soft feather band is affected.
    composited = _linear_to_srgb(
        _srgb_to_linear(refined_pixel) * pixel_alpha_crop
        + _srgb_to_linear(pixel_orig_crop) * (1.0 - pixel_alpha_crop)
    )
    new_pixels[:, y0_p:y1_p, x0_p:x1_p, :] = composited

    # The feathered pixel composite above IS this node's output. Upstream Angelo
    # additionally VAE-encoded `new_pixels` and latent-blended it with `current`
    # here, to (a) carry a bit-exact latent "canvas" across successive
    # interactive clicks and (b) feed its LATENT output. This headless port has
    # NEITHER — it is single-shot with only an IMAGE output — so that full-frame
    # re-encode + blend produced a latent the caller always discarded (A0).
    # Dropping it removes one whole VAE pass over the FULL image (the dominant
    # cost at 6-8K) and changes the output by exactly zero pixels. Return None
    # for the latent: `execute` uses `new_pixels`, which is always non-None on
    # this (the only) processing path.
    return None, new_pixels


logger = logging.getLogger("comfyui_timesaver.ts_smart_inpaint")
LOG_PREFIX = "[TS Smart Inpaint]"


class TSSmartInpaint(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TSSmartInpaint",
            display_name="TS Smart Inpaint",
            category="TS/Image/Retouch",
            description="Regenerate or refine only the masked region: it crops with context, samples at full detail and composites the result back.",
            inputs=[
                IO.Model.Input("model", tooltip="Diffusion model used to regenerate the masked region."),
                IO.Vae.Input("vae", tooltip="VAE used to encode the crop to latent and decode the result."),
                IO.Conditioning.Input("positive", tooltip="Positive conditioning describing what to generate inside the mask."),
                IO.Conditioning.Input("negative", tooltip="Negative conditioning describing what to avoid."),
                IO.Image.Input("image", tooltip="Full source image to inpaint."),
                IO.Mask.Input("mask", tooltip="Mask marking the region to inpaint (white = regenerate)."),
                IO.Boolean.Input(
                    "replace",
                    default=True,
                    label_on="Replace",
                    label_off="Refine",
                    tooltip="Replace = Smart Inpaint: regenerates the masked region "
                    "from scratch (reference_latents = the crop; an optional "
                    "`reference` image is chained as a 2nd reference — 'fill with "
                    "THIS'). Denoise is IGNORED and locked to 1.0. Refine = partial "
                    "denoise of the existing content at the Denoise value (no "
                    "reference).",
                ),
                IO.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="Refine mode only: how much of the existing content is redrawn (1.0 = fully). Ignored in Replace mode (locked to 1.0)."),
                IO.Float.Input("megapixels", default=1.5, min=0.1, max=8.0, step=0.1, tooltip="Processing budget for the masked crop. Small crops upscale toward it; oversized crops downscale to it to bound VAE/sampler cost. Raise for more detail."),
                IO.Float.Input(
                    "context_pct", default=8.0, min=0.0, max=50.0, step=0.5,
                    tooltip="Context band around the mask the model sees during "
                    "refine, as a PERCENT of the mask's own size (not fixed pixels) "
                    "— so it scales with the selection. Also hosts the colour-"
                    "correction ring. ~8% is a sensible default; raise for more "
                    "surrounding context.",
                ),
                IO.Float.Input(
                    "feather_pct", default=3.0, min=0.0, max=25.0, step=0.5,
                    tooltip="Feather (edge blend) width as a PERCENT of the mask's "
                    "own size rather than fixed pixels — a small mask gets a "
                    "proportionally small feather (no over-soft ghosting on thin "
                    "strokes), a big mask a wider blend. Clamped to a small px "
                    "floor. ~3% is a sensible default; 0 = hard edge.",
                ),
                IO.Combo.Input("resize_method", options=_FINE_UPSCALE_RESIZE_METHODS, default="lanczos", tooltip="Interpolation used to up/downscale the image crop. lanczos and bicubic keep the most detail."),
                # See ts_film_grain.py: an undeclared control_after_generate makes
                # the frontend add a widget the node definition does not know
                # about, and the last widget's value is lost on reload.
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True, tooltip="Noise seed for the sampler. Change for a different variation of the inpainted region."),
                IO.Int.Input("steps", default=4, min=1, max=100, tooltip="Number of sampling steps. More steps trade speed for quality."),
                IO.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1, tooltip="Classifier-free guidance scale. Higher values follow the prompt more strongly."),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler", tooltip="Sampling algorithm used to denoise the region."),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple", tooltip="Noise schedule that controls how sigma decreases across steps."),
                # Optional "fill with THIS picture" image (Replace mode only) —
                # chained as a 2nd Kontext reference. Absent → plain Smart Inpaint.
                IO.Image.Input(
                    "reference",
                    optional=True,
                    tooltip="Optional reference image (Replace only). VAE-encoded and "
                    "chained as a 2nd reference_latents after the crop, so the masked "
                    "region is filled toward THIS picture's content. Leave unconnected "
                    "for plain Smart Inpaint.",
                ),
            ],
            outputs=[IO.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        model,
        vae,
        positive,
        negative,
        image,
        mask,
        replace,
        denoise,
        megapixels,
        context_pct,
        feather_pct,
        resize_method,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        reference=None,
    ) -> IO.NodeOutput:
        # Replace = "Smart Inpaint" mode; unchecked = "Refine".
        inpainting_mode = "Smart Inpaint" if replace else "Refine"
        # Replace IGNORES the Denoise widget — Smart Inpaint locks denoise=1.0
        # (it regenerates the region from scratch). Refine uses the slider.
        effective_denoise = 1.0 if replace else float(denoise)
        # A1 — the core needs full-res pixels (the canvas) plus latent GEOMETRY,
        # NOT a full-image latent. We no longer VAE-encode the whole frame up
        # front: that latent's only consumers were a final full-frame re-encode +
        # blend the caller discarded (removed in A0) and the no-upscale Refine
        # path (which now encodes lazily, only when actually taken). So Smart
        # Inpaint / Refine-with-upscale push ONLY the crop through the VAE — the
        # whole point for 6-8K inputs.
        current_pixels = image  # (B, H_pix, W_pix, C) float [0,1] — the canvas

        H_img = int(image.shape[1])
        W_img = int(image.shape[2])
        # Latent spatial dims WITHOUT a full encode: comfy's VAE center-crops the
        # pixels to a multiple of spacial_compression_encode() before encoding,
        # so each latent side is exactly img_side // ratio for crop_input VAEs
        # (SD/SDXL/FLUX/Qwen/Wan — i.e. everything this node targets). For the
        # rare crop_input=False VAE, fall back to a real encode so dims stay exact.
        current = None
        try:
            sp = int(vae.spacial_compression_encode())
        except Exception:
            sp = 0
        if sp > 0 and getattr(vae, "crop_input", True):
            H_lat = max(1, H_img // sp)
            W_lat = max(1, W_img // sp)
        else:
            current = _vae_encode(vae, image)
            H_lat = int(current.shape[-2])
            W_lat = int(current.shape[-1])
        scale_x = (W_lat / W_img) if W_img else 1.0
        scale_y = (H_lat / H_img) if H_img else 1.0
        scale_geom = math.sqrt(max(1e-9, scale_x * scale_y))

        # Painted MASK (image res, white = inpaint) -> latent res -> gaussian
        # feather (sigma = feather_px * geometric-mean scale; feather_px below).
        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0)  # [1, H, W]
        elif m.dim() == 4:
            m = m[:1, 0]  # [1, H, W]
        else:
            m = m[:1]  # [1, H, W] from [B, H, W]
        mask_lat = _resize_latent(m, H_lat, W_lat, "bilinear").clamp(0.0, 1.0)

        # feather_pct / context_pct are PERCENTAGES of the mask's own size, not
        # absolute pixels — so the same setting blends/pads proportionally on any
        # mask (a fixed px feather is ~20% of a small mask but ~1% of a big one,
        # which over-softened small/thin selections). Base = the short side of the
        # tight painted bbox in IMAGE px, taken from the UN-feathered mask.
        tight = _mask_bbox_latent(mask_lat)
        if tight is not None:
            by0, by1, bx0, bx1 = tight
            mask_min_side_img = min(
                ((by1 - by0) / scale_y) if scale_y else 0.0,
                ((bx1 - bx0) / scale_x) if scale_x else 0.0,
            )
        else:
            mask_min_side_img = 0.0
        feather_px = _pct_to_px(float(feather_pct), mask_min_side_img, _FEATHER_FLOOR_PX, _FEATHER_CEIL_PX)
        context_px = _pct_to_px(float(context_pct), mask_min_side_img, 0.0, _CONTEXT_CEIL_PX)

        sigma_latent = (feather_px * scale_geom) if feather_px > 0 else 0.0
        if sigma_latent > 0:
            mask_lat = _gaussian_blur_2d(mask_lat, max(0.5, sigma_latent)).clamp(0.0, 1.0)

        # Optional reference image (Replace only) → a 2nd chained reference_latent
        # ("fill with THIS"). Skip when not connected (None) or not an image
        # tensor — the 2nd reference chain is simply not engaged. The reference
        # can be huge, so DOWNSCALE it to the same `megapixels` budget before
        # encoding (never upscales) — bounds its latent + token cost by the
        # resolution slider.
        extra_reference_latents = None
        if replace and reference is not None and hasattr(reference, "ndim"):
            ref_h, ref_w = int(reference.shape[1]), int(reference.shape[2])
            ref_pixels = _downscale_pixels_to_megapixels(
                reference, float(megapixels), resize_method
            )
            extra_reference_latents = [_vae_encode(vae, ref_pixels)]
            logger.info(
                "%s + reference image chained as 2nd reference_latent "
                "(%dx%d -> %dx%d, <=%.2f MP)",
                LOG_PREFIX, ref_w, ref_h,
                int(ref_pixels.shape[2]), int(ref_pixels.shape[1]), float(megapixels),
            )

        callback = latent_preview.prepare_callback(model, int(steps))
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Colour correction needs a preserved ring around the mask to measure the
        # Flux shift; grow the context band to at least the analysis margin so the
        # ring exists even at context_pct=0 (the extra context also aids generation).
        effective_context_pad = int(round(
            max(context_px, float(_CC_ANALYSIS_MARGIN_PX)) if _COLOR_CORRECT else context_px
        ))

        logger.info(
            "%s mode=%s color_correct=%s denoise=%.2f mp=%.2f max_linear=%.1f "
            "context=%.1f%%->%dpx feather=%.1f%%->%dpx img=%dx%d lat=%dx%d",
            LOG_PREFIX, inpainting_mode, bool(_COLOR_CORRECT), effective_denoise,
            float(megapixels), float(_MAX_LINEAR),
            float(context_pct), effective_context_pad, float(feather_pct), int(round(feather_px)),
            W_img, H_img, W_lat, H_lat,
        )

        new_latent, new_pixels = _refine_with_fine_upscaling(
            model=model,
            vae=vae,
            current=current,
            current_pixels=current_pixels,
            mask=mask_lat,
            scale_x=scale_x,
            scale_y=scale_y,
            target_mp=float(megapixels),
            max_linear=_MAX_LINEAR,
            resize_method=resize_method,
            context_pad_pixel=effective_context_pad,
            inpainting_mode=inpainting_mode,
            seed=int(seed),
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            denoise=effective_denoise,
            callback=callback,
            disable_pbar=disable_pbar,
            extra_reference_latents=extra_reference_latents,
            color_correct=_COLOR_CORRECT,
        )

        out = new_pixels if new_pixels is not None else _vae_decode(vae, new_latent)
        return IO.NodeOutput(out)


NODE_CLASS_MAPPINGS = {"TSSmartInpaint": TSSmartInpaint}
NODE_DISPLAY_NAME_MAPPINGS = {"TSSmartInpaint": "TS Smart Inpaint"}
