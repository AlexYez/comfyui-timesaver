"""TS Universal Inpaint Sampler — training-free inpainting for any model.

A custom SAMPLER running resample ("think") iterations during denoising —
the RePaint principle (Lugmayr et al., CVPR 2022) expressed in ComfyUI's
x0-prediction terms: at each noise level the masked region is repeatedly
re-noised from the model's own prediction and re-predicted against the
anchored context, which is what removes the seams a plain masked Euler
leaves. Clean-room from the public papers; no third-party sampler code.

How it works, in ComfyUI terms: the latent carries a noise_mask, so the
engine's KSamplerX0Inpaint wrapper re-anchors the KNOWN region to the
forward-noised source on every model call. Between ordinary Euler steps this
sampler runs K inner iterations that pull the MASKED region toward the
model's manifold at the current noise level — the masked content gets
"thought about" against the anchored context K times per step instead of
once, which is what removes the seams a plain masked Euler leaves. Works
with eps- and flow-parameterised models alike, because everything is
expressed through the model's denoised (x0) output.

Slower than a plain sampler by roughly (1 + think_steps) model calls per
step — that is the price of inpainting without a dedicated inpaint model.
"""
from __future__ import annotations

import logging

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_universal_inpaint")
LOG_PREFIX = "[TS Universal Inpaint]"


def _build_sampler_function(think_steps: int, resample_strength: float, model_sampling):
    import torch

    def scales(sigma_tensor, x):
        """a(sigma), b(sigma) of the model's linear noising x = a*x0 + b*eps.

        Probed through the model's own noise_scaling so the resample is
        parameterisation-agnostic: EPS gives a=1, b=sigma; flow (CONST) gives
        a=1-sigma, b=sigma. Probes are 1-element tensors — negligible cost.
        """
        one = x.new_ones([1, 1, 1, 1])
        zero = x.new_zeros([1, 1, 1, 1])
        a = model_sampling.noise_scaling(sigma_tensor, zero, one)
        b = model_sampling.noise_scaling(sigma_tensor, one, zero)
        return a, b

    def sample_fn(model, x, sigmas, extra_args=None, callback=None, disable=None, **_kwargs):
        extra_args = extra_args or {}
        mask = extra_args.get("denoise_mask")
        s_in = x.new_ones([x.shape[0]])
        strength = min(max(float(resample_strength), 0.0), 1.0)
        keep = (1.0 - strength ** 2) ** 0.5

        total = len(sigmas) - 1
        for i in range(total):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_item = max(float(sigma), 1e-8)

            if mask is not None and think_steps > 0 and float(sigma_next) < sigma_item:
                # Resample iterations at a FIXED noise level: take the model's
                # current prediction for the masked region and re-noise it back
                # to sigma with (partly) fresh noise. The level is preserved by
                # construction, and every iteration lets the masked content be
                # re-predicted against the context that the engine's inpaint
                # wrapper keeps anchored — self-consistency instead of a seam.
                a, b = scales(sigma, x)
                b_safe = torch.clamp(b.abs(), min=1e-8) * torch.sign(
                    torch.where(b == 0, torch.ones_like(b), b))
                for _ in range(think_steps):
                    denoised = model(x, sigma * s_in, **extra_args)
                    eps_old = (x - a * denoised) / b_safe
                    fresh = torch.randn_like(x)
                    mixed = keep * eps_old + strength * fresh
                    x_re = a * denoised + b * mixed
                    x = x * (1.0 - mask) + x_re * mask

            denoised = model(x, sigma * s_in, **extra_args)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma,
                          "sigma_hat": sigma, "denoised": denoised})
            if float(sigma_next) == 0.0:
                x = denoised
            else:
                d = (x - denoised) / sigma_item
                x = x + d * (sigma_next - sigma)
        return x

    return sample_fn


class TS_UniversalInpaintSampler(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_UniversalInpaintSampler",
            display_name="TS Universal Inpaint Sampler",
            category="TS/Image/Retouch",
            description=(
                "Training-free inpainting sampler for ANY diffusion model — no "
                "Fill checkpoint or ControlNet needed. Plug into "
                "SamplerCustomAdvanced and feed a latent with a noise mask "
                "(SetLatentNoiseMask): between denoise steps it runs extra "
                "'think' iterations that settle the masked region against the "
                "surrounding context. Roughly (1 + think_steps)x slower than a "
                "plain sampler."
            ),
            inputs=[
                IO.Model.Input(
                    "model",
                    tooltip="The SAME model the guider uses — needed to read its "
                            "noising schedule so resampling matches the "
                            "parameterisation (eps or flow).",
                ),
                IO.Int.Input(
                    "think_steps", default=4, min=0, max=20,
                    tooltip="Resample iterations per denoise step. 0 = plain masked "
                            "Euler; 3-6 removes seams; more = slower, cleaner.",
                ),
                IO.Float.Input(
                    "resample_strength", default=1.0, min=0.2, max=1.0, step=0.05,
                    tooltip="Share of fresh noise per iteration. 1.0 = full resample "
                            "(recommended); lower keeps more of the current draft.",
                ),
            ],
            outputs=[IO.Custom("SAMPLER").Output(display_name="sampler")],
            search_aliases=["universal inpaint", "training-free inpaint",
                            "langevin sampler", "inpaint any model"],
        )

    @classmethod
    def execute(cls, model, think_steps: int, resample_strength: float) -> IO.NodeOutput:
        import comfy.samplers

        model_sampling = model.get_model_object("model_sampling")
        function = _build_sampler_function(
            int(think_steps), float(resample_strength), model_sampling)
        return IO.NodeOutput(comfy.samplers.KSAMPLER(function))


NODE_CLASS_MAPPINGS = {"TS_UniversalInpaintSampler": TS_UniversalInpaintSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_UniversalInpaintSampler": "TS Universal Inpaint Sampler"}
