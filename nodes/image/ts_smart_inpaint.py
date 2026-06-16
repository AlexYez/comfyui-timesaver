"""TS Smart Inpaint — headless port of ComfyUI-Angelo's "Smart Inpaint" /
"Refine" Xtra-Fine path.

Given the FULL source image + a painted MASK (white = inpaint) + the model / vae /
positive / negative, this node reproduces Angelo's algorithm BYTE-FOR-BYTE
(the actual `_refine_with_fine_upscaling` is extracted verbatim into
`_angelo_xtrafine.py`):

  bbox(mask) + context_pad band -> crop pixels -> upscale to `megapixels`
  (capped at `max_linear`) -> VAE-encode -> [Smart Inpaint: reference_latents =
  the crop + zero the masked latent] -> noise-injection inpaint (denoise) ->
  VAE-decode -> downscale -> feather-composite back -> VAE-encode + latent-blend
  (mask alpha) so unaltered regions stay bit-exact.

`replace` (checkbox, label_on="Replace" / label_off="Refine"):
  - Replace (ON)  = Angelo "Smart Inpaint": reference_latents = the crop + zero
    the masked latent, regenerates the painted region as a Kontext edit. The
    `denoise` widget is IGNORED and locked to 1.0.
  - Refine (OFF)  = standard Xtra-Fine refine (ADetailer-style) of the painted
    region at the `denoise` value, no reference.

The crop + composite happen INSIDE the node, so the app just uploads the full
source + mask. Credit: Angelo (github.com/shootthesound/ComfyUI-Angelo).

node_id: TSSmartInpaint
"""

import logging
import math

import comfy.samplers
import comfy.utils
import latent_preview

from comfy_api.v0_0_2 import IO

from ._angelo_xtrafine import (
    _FINE_UPSCALE_RESIZE_METHODS,
    _gaussian_blur_2d,
    _refine_with_fine_upscaling,
    _resize_latent,
    _vae_decode,
    _vae_encode,
)

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
                    tooltip="Replace = Angelo Smart Inpaint: regenerates the masked "
                    "region from scratch (reference_latents = the crop), Denoise "
                    "is IGNORED and locked to 1.0. Refine = partial denoise of the "
                    "existing content at the Denoise value (no reference).",
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
    ) -> IO.NodeOutput:
        # Replace = Angelo's "Smart Inpaint" mode; unchecked = "Refine".
        inpainting_mode = "Smart Inpaint" if replace else "Refine"
        # Replace IGNORES the Denoise widget — Smart Inpaint locks denoise=1.0
        # (it regenerates the region from scratch). Refine uses the slider.
        effective_denoise = 1.0 if replace else float(denoise)
        # Angelo operates on a cached full-res latent + pixels. We have the
        # pixels (the IMAGE input); encode them once to get the latent.
        current = _vae_encode(vae, image)  # native VAE latent shape
        current_pixels = image  # (B, H_pix, W_pix, C) float [0,1]

        H_img = int(image.shape[1])
        W_img = int(image.shape[2])
        H_lat = int(current.shape[-2])
        W_lat = int(current.shape[-1])
        scale_x = (W_lat / W_img) if W_img else 1.0
        scale_y = (H_lat / H_img) if H_img else 1.0
        scale_geom = math.sqrt(max(1e-9, scale_x * scale_y))

        # Painted MASK (image res, white = inpaint) -> latent res -> Angelo's
        # gaussian feather (sigma = feather * geometric-mean scale).
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
        )

        out = new_pixels if new_pixels is not None else _vae_decode(vae, new_latent)
        return IO.NodeOutput(out)


NODE_CLASS_MAPPINGS = {"TSSmartInpaint": TSSmartInpaint}
NODE_DISPLAY_NAME_MAPPINGS = {"TSSmartInpaint": "TS Smart Inpaint"}
