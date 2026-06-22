"""TS Smart Inpaint — single-file headless port of the "Xtra-Fine" "Smart
Inpaint" / "Refine" path.

Given the FULL source image + a painted MASK (white = inpaint) + the model / vae /
positive / negative, this node reproduces the algorithm BYTE-FOR-BYTE — the
`_refine_with_fine_upscaling` helper + its sampling / VAE / mask helpers are
extracted verbatim BELOW (deliberately kept in this ONE file — it's small enough
that a single file reads clearer than a split):

  bbox(mask) + context_pad band -> crop pixels -> upscale to `megapixels`
  (capped at `max_linear`) -> VAE-encode -> [Smart Inpaint: reference_latents =
  the crop + zero the masked latent] -> noise-injection inpaint (denoise) ->
  VAE-decode -> downscale -> feather-composite back -> VAE-encode + latent-blend
  (mask alpha) so unaltered regions stay bit-exact.

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

The crop + composite happen INSIDE the node, so the app just uploads the full
source + mask (+ optional reference).

Algorithm credit: the "Xtra-Fine" inpaint path of ComfyUI-Angelo
(shootthesound/ComfyUI-Angelo, angelo_nodes.py) — the helpers below are taken
from its `_refine_with_fine_upscaling` so the behaviour matches it, with ONE
local extension (the optional `extra_reference_latents` chaining used in Replace
mode). The custom-sampler helpers (`_guider_*`, `_do_sample`'s guider path) are
from @KursatAs's customSampler branch. Upstream is MIT-licensed; its notice is
reproduced below as required:

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

import copy
import logging
import math

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import node_helpers

from comfy_api.v0_0_2 import IO


_FINE_UPSCALE_RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp", "lanczos"]


def _guider_sample(
        temp_g,
        noise: torch.Tensor,
        latent: torch.Tensor,
        sampler,
        sigmas: torch.Tensor,
        *,
        denoise_mask: torch.Tensor | None = None,
        callback=None,
        disable_pbar: bool = False,
        seed: int | None = None,
) -> torch.Tensor:
    """Device-safe wrapper around guider.sample(). (From @KursatAs.)

    ComfyUI's built-in CFGGuider.sample() moves noise, latent, and
    denoise_mask to the model's load device before sampling. Some
    third-party extensions (e.g. ComfyUI-NAG-Extended) override
    inner_sample() without repeating that movement, so CPU tensors
    survive into the k-sampler's inpaint path where they collide with
    GPU tensors and raise a device-mismatch RuntimeError. Moving
    everything to load_device here is a safe no-op when the built-in
    already does it, and fixes the crash for extensions that don't."""
    device = temp_g.model_patcher.load_device
    noise = noise.to(device)
    latent = latent.to(device)
    if denoise_mask is not None:
        denoise_mask = denoise_mask.to(device)
    samples = temp_g.sample(
        noise, latent, sampler, sigmas,
        denoise_mask=denoise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    # Mirror comfy.sample.sample()'s exit move so the rest of Angelo's
    # pipeline (VAE decode, pixel composite, latent blend in Fine Upscale)
    # sees the same intermediate device + dtype it does on the default
    # path. Without this, the guider path returns the sampled latent on
    # the model's load_device (cuda) while cached_pixels / mask are on
    # intermediate_device (typically CPU), causing a device-mismatch at
    # the composite step in _refine_with_fine_upscaling.
    return samples.to(
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )


def _guider_with_conds(guider, positive, negative):
    """Copy a wired GUIDER and apply Angelo's per-call positive/negative
    conds to it. Handles both CFGGuider (takes both conds) and
    BasicGuider (positive only). (From @KursatAs.) Lets the user wire
    a generic guider once and Angelo keeps using its dynamic conds
    (Refine vs Area Prompt vs Smart Inpaint reference_latents etc.)
    for each sample call."""
    g = copy.copy(guider)
    try:
        g.set_conds(positive, negative)  # comfy.samplers.CFGGuider
    except TypeError:
        g.set_conds(positive)            # BasicGuider / BaseGuider
    return g


def _truncate_sigmas_for_denoise(sigmas: torch.Tensor, denoise: float) -> torch.Tensor:
    """Tail-slice the wired SIGMAS tensor by Angelo's per-call denoise,
    matching ComfyUI's SplitSigmasDenoise convention. (From @KursatAs.)
    A wired sigmas tensor comes pre-baked from a scheduler node at
    denoise=1.0; Angelo applies its own refine denoise here so the
    refine slider keeps meaning what it did before."""
    if denoise >= 1.0:
        return sigmas
    if denoise <= 0.0:
        return sigmas[-1:].new_zeros(2)
    n_total = len(sigmas) - 1
    n_refine = max(1, round(n_total * denoise))
    return sigmas[-(n_refine + 1):]


def _do_sample(
        *,
        guider,
        sampler,
        sigmas,
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
    """Single sample dispatch for every Angelo sample call. If a custom
    guider + sampler + sigmas trio is wired through Overrides, take the
    custom path (via _guider_sample); otherwise fall through to the
    standard comfy.sample.sample(...) path that's existed since v1.0.

    All-or-nothing on the trio: partial wiring (e.g. sampler without
    sigmas) silently falls through to the default path. Users see the
    custom-sampler kwargs only by wiring the full bundle from a proper
    GUIDER + SAMPLER + SIGMAS chain in their workflow."""
    if guider is not None and sampler is not None and sigmas is not None:
        g = _guider_with_conds(guider, positive, negative)
        s = _truncate_sigmas_for_denoise(sigmas, denoise)
        return _guider_sample(
            g, noise, source_latent, sampler, s,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
    return comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, source_latent,
        denoise=denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )


def _mask_bbox_latent(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Tight latent-space bbox of non-zero mask values. Returns
    (y_min, y_max, x_min, x_max) or None if the mask is empty.

    Threshold of 0.01 includes the feathered edge — bbox covers the
    full soft-edge region not just the binary interior.
    """
    m = mask[0] if mask.dim() == 3 else mask
    nz = m > 0.01
    if not nz.any():
        return None
    rows = nz.any(dim=1)
    cols = nz.any(dim=0)
    ridx = rows.nonzero(as_tuple=False).squeeze(-1)
    cidx = cols.nonzero(as_tuple=False).squeeze(-1)
    return (
        int(ridx[0].item()),
        int(ridx[-1].item()) + 1,
        int(cidx[0].item()),
        int(cidx[-1].item()) + 1,
    )


def _fine_upscale_factor(
    bbox_w_latent: int,
    bbox_h_latent: int,
    scale_x: float,
    scale_y: float,
    target_mp: float,
    max_linear: float,
) -> float:
    """Linear scale factor to apply to the cropped latent so that the
    crop is processed at ≥ target_mp (in image-pixel-equivalent terms),
    clamped to max_linear. Returns 1.0 when the crop already meets
    target — no upscale needed."""
    if scale_x <= 0 or scale_y <= 0:
        return 1.0
    bbox_w_pix = bbox_w_latent / scale_x
    bbox_h_pix = bbox_h_latent / scale_y
    current_mp = bbox_w_pix * bbox_h_pix / 1_000_000.0
    if current_mp <= 0 or current_mp >= target_mp:
        return 1.0
    needed = math.sqrt(target_mp / current_mp)
    return min(needed, max_linear)


def _resize_latent(t: torch.Tensor, target_h: int, target_w: int, method: str) -> torch.Tensor:
    """Resize the spatial dims of a latent or mask tensor using one of
    ComfyUI's standard latent-resize methods. Accepts [C,H,W], [B,C,H,W],
    or [1,H,W] (mask). Returns the same rank as input.

    `method` is one of _FINE_UPSCALE_RESIZE_METHODS. Routes through
    comfy.utils.common_upscale so bislerp + lanczos custom paths work."""
    method = method if method in _FINE_UPSCALE_RESIZE_METHODS else "nearest-exact"
    if t.dim() == 3:
        t4 = t.unsqueeze(0)
        out = comfy.utils.common_upscale(t4, target_w, target_h, method, "disabled")
        return out.squeeze(0)
    if t.dim() == 4:
        return comfy.utils.common_upscale(t, target_w, target_h, method, "disabled")
    raise ValueError(f"_resize_latent: unexpected ndim {t.dim()}")


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


def _downscale_pixels_to_megapixels(
    pixels: torch.Tensor, target_mp: float, method: str
) -> torch.Tensor:
    """Downscale a pixel image batch (B, H, W, C) so its area is <= `target_mp`
    megapixels. NEVER upscales — a reference smaller than the budget passes
    through untouched (upscaling a reference adds no information). Each side is
    snapped to a multiple of 8 for VAE-friendliness.

    Used to bound an optional, possibly-huge reference image by the same
    resolution budget the user picked for the inpaint, so its VAE latent (and
    thus the token cost of the chained reference) stays sane. `method` is one of
    _FINE_UPSCALE_RESIZE_METHODS; comfy.utils.common_upscale wants channels-first
    so we permute around it."""
    if pixels is None or pixels.dim() != 4:
        return pixels
    h = int(pixels.shape[1])
    w = int(pixels.shape[2])
    if h <= 0 or w <= 0:
        return pixels
    budget = max(1, int(float(target_mp) * 1_000_000))
    if h * w <= budget:
        return pixels  # already within budget — never upscale
    s = math.sqrt(budget / float(h * w))
    new_w = max(8, (int(round(w * s)) // 8) * 8)
    new_h = max(8, (int(round(h * s)) // 8) * 8)
    method = method if method in _FINE_UPSCALE_RESIZE_METHODS else "lanczos"
    chw = pixels.permute(0, 3, 1, 2)
    chw = comfy.utils.common_upscale(chw, new_w, new_h, method, "disabled")
    return chw.permute(0, 2, 3, 1).contiguous()


def _gaussian_blur_2d(mask: torch.Tensor, sigma_latent: float) -> torch.Tensor:
    """Separable gaussian blur on a [B, H, W] or [H, W] mask tensor.

    sigma_latent is in latent-space units.
    """
    if sigma_latent <= 0:
        return mask
    # Kernel covers ~±3σ. Force odd size.
    ksize = int(2 * math.ceil(3 * sigma_latent) + 1)
    half = ksize // 2
    x = torch.arange(ksize, device=mask.device, dtype=torch.float32) - half
    k1d = torch.exp(-0.5 * (x / sigma_latent) ** 2)
    k1d = k1d / k1d.sum()

    # Reshape mask to [N, 1, H, W]
    orig_ndim = mask.dim()
    if orig_ndim == 2:
        m = mask.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        m = mask.unsqueeze(1)
    elif orig_ndim == 4:
        m = mask
    else:
        raise ValueError(f"gaussian_blur_2d: unexpected mask ndim {orig_ndim}")

    kh = k1d.view(1, 1, 1, ksize)
    kv = k1d.view(1, 1, ksize, 1)

    m = torch.nn.functional.pad(m, (half, half, 0, 0), mode="replicate")
    m = torch.nn.functional.conv2d(m, kh)
    m = torch.nn.functional.pad(m, (0, 0, half, half), mode="replicate")
    m = torch.nn.functional.conv2d(m, kv)

    if orig_ndim == 2:
        return m.squeeze(0).squeeze(0)
    if orig_ndim == 3:
        return m.squeeze(1)
    return m


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
    # #8 custom-sampler trio — None = use the default comfy.sample.sample
    # path; all three set = dispatch via _do_sample to the guider path.
    ov_guider=None,
    ov_sampler=None,
    ov_sigmas=None,
    # LOCAL extension (not from upstream): extra reference_latents to CHAIN after
    # the crop's own reference in Smart Inpaint mode — e.g. a user "fill with
    # THIS picture" image. None / empty = behaves exactly like upstream.
    extra_reference_latents=None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pixel-space crop + upscale + VAE encode + refine + VAE decode +
    downscale + composite. The latent-space crop+upscale
    approach smears bilinearly-interpolated latents into a low-freq
    starting state that the model can't recover detail from. Going
    through pixel space (where there's an image-upscale toolkit that's
    been tuned for natural images) and re-encoding gives the model a
    "natural" latent at the higher resolution to denoise from.

    Returns (new_latent, new_pixels). `new_pixels` is the feathered composite —
    the node's actual output. `new_latent` is None on the crop path (A1: the
    caller only ever consumes `new_pixels` there); it is a real latent only on
    the no-upscale Refine shortcut, which `execute` then decodes. On an empty /
    degenerate mask returns (current, current_pixels) unchanged.
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

    scale = _fine_upscale_factor(bbox_w_lat, bbox_h_lat, scale_x, scale_y, target_mp, max_linear)
    if scale <= 1.0 and inpainting_mode != "Smart Inpaint":
        # Refine with no upscale needed — fall back to the standard latent-space
        # noise-injection inpaint. Avoids unnecessary VAE round-trips when the
        # painted region already meets the MP target.
        #
        # Smart Inpaint must NOT take this shortcut. It needs the crop +
        # reference_latents + masked-zero treatment below regardless of rect
        # size; skipping it made a LARGE rectangle (already at/above the MP
        # target, so scale<=1.0 — roughly >1024px on FLUX 2) degrade to a
        # whole-latent edit with NO crop reference, so the model worked on the
        # whole image instead of the selected rect.
        print(f"[Angelo fine-upscale] scale=1.0 — using latent-space path (no VAE round-trip)")
        # A1: this no-upscale Refine path is the ONLY one that consumes the
        # full-image latent, so encode it lazily here (it is None on A1's
        # geometry-only fast path). Smart Inpaint never reaches this branch.
        if current is None:
            current = _vae_encode(vae, current_pixels)
        noise = comfy.sample.prepare_noise(current, seed, None)
        new_latent = _do_sample(
            guider=ov_guider, sampler=ov_sampler, sigmas=ov_sigmas,
            model=model, noise=noise,
            steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
            positive=positive, negative=negative,
            source_latent=current,
            denoise=denoise,
            noise_mask=mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        # Return None for pixels because the latent was modified directly;
        # this forces a fresh VAE decode for the preview in the main run() method.
        return new_latent, None

    # Smart Inpaint with a large rectangle still crops + references the selected
    # region; it just doesn't upscale (and must never downscale) — clamp the
    # factor to identity so the crop is taken at native resolution.
    if inpainting_mode == "Smart Inpaint":
        scale = max(1.0, scale)

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

    print(f"[Angelo fine-upscale] bbox_lat=(h={bbox_h_lat}, w={bbox_w_lat}) "
          f"bbox_px=(h={bbox_h_p}, w={bbox_w_p}) scale={scale:.2f} "
          f"target_px=(h={target_h_p}, w={target_w_p}) "
          f"resize={resize_method} max_linear={max_linear} "
          f"vae_ratio=(x={px_per_lat_x}, y={px_per_lat_y})")

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
    # Mask used for the SAMPLING step (both the masked-zero and the noise_mask).
    # Defaults to the feathered mask; hardened to binary for Smart Inpaint on
    # 5D temporal latents — see the note in the Smart Inpaint block below.
    sample_mask = mask_crop_up
    if inpainting_mode == "Smart Inpaint":
        reference_latent = latent_up.clone()
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
        # Feather goes in the COMPOSITE, not the denoise mask — for Qwen/Wan.
        #
        # FLUX/Klein (4D, ~zero-mean latents): a soft mask works directly as
        # both the masked-zero and the noise_mask; their inpaint blend handles a
        # soft boundary cleanly, so keep the feathered mask.
        #
        # Qwen Image Edit / Wan (5D temporal latents): sample with a HARD
        # (binary) mask. These models have no clean soft-mask-during-denoise
        # behaviour — the community-standard Qwen-edit inpaint recipe is "blank
        # the region, regenerate, then composite back with a feathered blend",
        # NOT feathering the denoise mask. A soft noise_mask at denoise=1.0 on a
        # non-zero-mean latent space distorts exactly the feather band (the
        # symptom: artifacts only where the feathering happens). The smooth
        # visible edge is still produced downstream by the feathered PIXEL
        # composite + final latent blend (both use the full-res feathered
        # `mask`), so the feather is preserved without the sampling artifacts.
        if latent_up.ndim == 5:
            sample_mask = (mask_crop_up >= 0.5).to(mask_crop_up.dtype)
        latent_up = (1.0 - sample_mask.unsqueeze(0)) * latent_up

    # ----- Refine via noise-injection inpaint on the upscaled latent -----
    noise = comfy.sample.prepare_noise(latent_up, seed, None)
    refined_latent_up = _do_sample(
        guider=ov_guider, sampler=ov_sampler, sigmas=ov_sigmas,
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
    composited = refined_pixel * pixel_alpha_crop + pixel_orig_crop * (1.0 - pixel_alpha_crop)
    new_pixels[:, y0_p:y1_p, x0_p:x1_p, :] = composited

    # The feathered pixel composite above IS this node's output. Upstream Angelo
    # additionally VAE-encoded `new_pixels` and latent-blended it with `current`
    # here, to (a) carry a bit-exact latent "canvas" across successive
    # interactive clicks and (b) feed its LATENT output. This headless port has
    # NEITHER — it is single-shot with only an IMAGE output — so that full-frame
    # re-encode + blend produced a latent the caller always discarded (A0).
    # Dropping it removes one whole VAE pass over the FULL image (the dominant
    # cost at 6-8K) and changes the output by exactly zero pixels. Return None
    # for the latent: `execute` uses `new_pixels` whenever it is not None (always
    # on this crop path); only the no-upscale Refine shortcut returns a latent.
    return None, new_pixels


logger = logging.getLogger("comfyui_timesaver.ts_smart_inpaint")
LOG_PREFIX = "[TS Smart Inpaint]"


class TSSmartInpaint(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TSSmartInpaint",
            display_name="TS Smart Inpaint",
            category="TS/Image",
            inputs=[
                IO.Model.Input("model"),
                IO.Vae.Input("vae"),
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Image.Input("image"),
                IO.Mask.Input("mask"),
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
                IO.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("megapixels", default=1.5, min=0.1, max=8.0, step=0.1),
                IO.Float.Input("max_linear", default=3.0, min=1.0, max=8.0, step=0.1),
                IO.Int.Input("context_pad", default=0, min=0, max=512, step=8),
                IO.Int.Input("feather", default=15, min=0, max=200, step=1),
                IO.Combo.Input("resize_method", options=_FINE_UPSCALE_RESIZE_METHODS, default="lanczos"),
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF),
                IO.Int.Input("steps", default=4, min=1, max=100),
                IO.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
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
        max_linear,
        context_pad,
        feather,
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
        # feather (sigma = feather * geometric-mean scale).
        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0)  # [1, H, W]
        elif m.dim() == 4:
            m = m[:1, 0]  # [1, H, W]
        else:
            m = m[:1]  # [1, H, W] from [B, H, W]
        mask_lat = _resize_latent(m, H_lat, W_lat, "bilinear").clamp(0.0, 1.0)
        sigma_latent = (float(feather) * scale_geom) if feather > 0 else 0.0
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

        logger.info(
            "%s mode=%s denoise=%.2f mp=%.2f max_linear=%.1f ctx_pad=%d feather=%d img=%dx%d lat=%dx%d",
            LOG_PREFIX, inpainting_mode, effective_denoise, float(megapixels), float(max_linear),
            int(context_pad), int(feather), W_img, H_img, W_lat, H_lat,
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
            max_linear=float(max_linear),
            resize_method=resize_method,
            context_pad_pixel=int(context_pad),
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
        )

        out = new_pixels if new_pixels is not None else _vae_decode(vae, new_latent)
        return IO.NodeOutput(out)


NODE_CLASS_MAPPINGS = {"TSSmartInpaint": TSSmartInpaint}
NODE_DISPLAY_NAME_MAPPINGS = {"TSSmartInpaint": "TS Smart Inpaint"}
