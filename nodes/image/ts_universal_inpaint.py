"""TS Universal Inpaint Sampler — training-free inpainting for any model.

A custom SAMPLER implementing Langevin-style "think" iterations during
denoising, in the spirit of the published method "Training-Free Diffusion
Inpainting with Asymptotically Exact and Fast Conditional Sampling"
(TMLR 2025). Clean-room: written from the paper's public description in
ComfyUI's x0-prediction terms; no third-party sampler code was used.

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
import math

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_universal_inpaint")
LOG_PREFIX = "[TS Universal Inpaint]"


def _build_sampler_function(think_steps: int, lambda_strength: float,
                            step_size: float, friction: float, think_noise: float):
    import torch

    def sample_fn(model, x, sigmas, extra_args=None, callback=None, disable=None, **_kwargs):
        extra_args = extra_args or {}
        mask = extra_args.get("denoise_mask")
        s_in = x.new_ones([x.shape[0]])
        velocity = torch.zeros_like(x)
        # Momentum retention per inner iteration: friction integrates the
        # underdamped dynamics; high friction -> almost plain gradient steps.
        beta = math.exp(-max(friction, 0.0) * max(step_size, 1e-4))

        total = len(sigmas) - 1
        for i in range(total):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_item = max(float(sigma), 1e-8)

            if mask is not None and think_steps > 0 and float(sigma_next) < sigma_item:
                # Inner Langevin iterations at a FIXED noise level. The known
                # region is re-anchored inside every model call, so each
                # iteration lets the masked region re-negotiate with fresh
                # context. lambda scales the pull toward the model manifold.
                gap = (sigma_item - float(sigma_next)) / sigma_item
                eta = step_size * gap * lambda_strength
                for _ in range(think_steps):
                    denoised = model(x, sigma * s_in, **extra_args)
                    grad = (x - denoised) / sigma_item
                    velocity = beta * velocity + (1.0 - beta) * grad
                    step = eta * sigma_item * velocity
                    if think_noise > 0.0:
                        step = step - math.sqrt(2.0 * eta) * sigma_item * think_noise \
                            * torch.randn_like(x)
                    x = x - step * mask

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
                IO.Int.Input(
                    "think_steps", default=4, min=0, max=20,
                    tooltip="Inner iterations per denoise step. 0 = plain masked "
                            "Euler; 3-6 removes seams; more = slower, cleaner.",
                ),
                IO.Float.Input(
                    "lambda_strength", default=4.0, min=0.1, max=20.0, step=0.1,
                    tooltip="How strongly a think iteration pulls the masked region "
                            "toward the model's prediction (4-10 is sensible).",
                ),
                IO.Float.Input(
                    "step_size", default=0.15, min=0.01, max=1.0, step=0.01,
                    tooltip="Inner iteration step size. Too large diverges, too "
                            "small does nothing; 0.1-0.3 is sensible.",
                ),
                IO.Float.Input(
                    "friction", default=12.0, min=0.0, max=30.0, step=0.5,
                    tooltip="Damping of the inner dynamics. Higher = more stable, "
                            "lower = faster but can oscillate (10-20 is sensible).",
                ),
                IO.Float.Input(
                    "think_noise", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Stochasticity of think iterations. 0 = deterministic "
                            "(recommended); small values add exploration.",
                ),
            ],
            outputs=[IO.Custom("SAMPLER").Output(display_name="sampler")],
            search_aliases=["universal inpaint", "training-free inpaint",
                            "langevin sampler", "inpaint any model"],
        )

    @classmethod
    def execute(cls, think_steps: int, lambda_strength: float, step_size: float,
                friction: float, think_noise: float) -> IO.NodeOutput:
        import comfy.samplers

        function = _build_sampler_function(
            int(think_steps), float(lambda_strength), float(step_size),
            float(friction), float(think_noise),
        )
        return IO.NodeOutput(comfy.samplers.KSAMPLER(function))


NODE_CLASS_MAPPINGS = {"TS_UniversalInpaintSampler": TS_UniversalInpaintSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_UniversalInpaintSampler": "TS Universal Inpaint Sampler"}
