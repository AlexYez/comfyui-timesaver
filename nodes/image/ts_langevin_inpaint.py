"""TS Langevin Inpaint — inertial Langevin inpainting, implemented in-pack.

node_id: TS_LangevinInpaint

Ordinary masked sampling re-predicts the hidden region against a context it
never revisits, which is why hard cases come back seam-y or semantically
detached. This sampler instead spends a few *inner* steps at every noise level
running an inertial Langevin update whose drift differs by region:

  • outside the mask the score is the plain denoiser direction, so the known
    content stays put;
  • inside it, a bidirectional term pulls the state towards agreement with the
    known pixels with strength `guidance` (λ), while `beta` scales that
    branch's time step.

The update is the exact transition kernel of the resulting linear SDE — see
_langevin_inpaint.py, where the closed forms are derived — not an Euler step,
so large λ stays stable. If the inertial update ever goes non-finite the step
falls back to the overdamped limit rather than failing the run.

The scheme follows the method published as LanPaint (Langevin inpainting);
the mathematics is standard Ornstein–Uhlenbeck theory and every line here is
this pack's own.
"""
from __future__ import annotations

import logging

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import torch
from comfy_api.v0_0_2 import IO

from ._langevin_inpaint import evolve, evolve_overdamped

logger = logging.getLogger("comfyui_timesaver.ts_langevin_inpaint")
LOG_PREFIX = "[TS Langevin Inpaint]"


def _times_from_sigma(sigma: torch.Tensor, is_flow: bool):
    """Express the sampler's sigma in the variance-preserving quantities the
    dynamics needs: (variance-exploding sigma, alpha-bar, flow time).

    Flow models parameterise x_t = (1-t)·x0 + t·eps, ordinary ones
    x_t = x0 + sigma·eps; both map onto the same alpha-bar axis.
    """
    if is_flow:
        flow_t = sigma
        one_minus = 1.0 - flow_t
        abt = one_minus**2 / (one_minus**2 + flow_t**2)
        ve_sigma = flow_t / torch.clamp(one_minus, min=1e-6)
    else:
        ve_sigma = sigma
        abt = 1.0 / (1.0 + ve_sigma**2)
        flow_t = (1.0 - abt) ** 0.5 / ((1.0 - abt) ** 0.5 + abt**0.5)
    return ve_sigma, abt, flow_t


class _LangevinInner:
    """One noise level's worth of Langevin refinement."""

    def __init__(self, model, latent_image, noise, mask_known, options):
        self.model = model
        self.latent_image = latent_image
        self.noise = noise
        self.mask_known = mask_known          # 1 where the pixels are known
        self.steps = int(options["steps"])
        self.lam = float(options["guidance"])
        self.step_size = float(options["step_size"])
        self.beta = float(options["beta"])
        self.friction = float(options["friction"])

    def _denoise(self, x, sigma, extra_args):
        s_in = x.new_ones([x.shape[0]])
        out = self.model(x, sigma * s_in, **extra_args)
        if isinstance(out, (tuple, list)):
            return out[0], (out[1] if len(out) > 1 else out[0])
        return out, out

    def refine(self, x, sigma, is_flow, extra_args, model_sampling):
        """Return (updated x_t in the sampler's own space, denoised prediction)."""
        ve_sigma, abt, flow_t = _times_from_sigma(sigma, is_flow)
        mask = self.mask_known

        # Re-anchor the known region to the source at this noise level.
        sigma_b = sigma.reshape([1] * x.ndim) if sigma.ndim == 0 else sigma
        anchored = model_sampling.noise_scaling(sigma_b, self.noise, self.latent_image)
        x = x * (1.0 - mask) + anchored * mask

        if self.steps <= 0:
            denoised, _ = self._denoise(x, sigma, extra_args)
            return x, denoised * (1.0 - mask) + self.latent_image * mask

        # Move to the variance-preserving frame the dynamics is written in.
        if is_flow:
            scale = abt**0.5 + (1.0 - abt) ** 0.5
            x_t = x * scale
        else:
            scale = (1.0 + ve_sigma**2) ** 0.5
            x_t = x / scale

        one_minus_abt = torch.clamp(1.0 - abt, min=1e-6)
        # Time steps: the masked branch runs on its own clock (beta).
        base = self.step_size * one_minus_abt
        dt_x = base                      # sigma_x == 1
        dt_y = base * self.beta
        dt = dt_x * (1.0 - mask) + dt_y * mask

        a_x = 1.0 / one_minus_abt
        a_y = (1.0 + self.lam) / one_minus_abt
        a = a_x * (1.0 - mask) + a_y * mask

        # Damping. The published scaling fixes the DIMENSIONLESS product Γ·dt
        # at friction²·step_size/0.2 — the same on both branches — so Γ itself
        # scales as 1/dt and grows as the noise level falls.
        gamma_dt = self.friction**2 * self.step_size / 0.2
        gamma = gamma_dt / torch.clamp(dt, min=1e-8)
        d_amp = torch.full_like(x_t, 2.0**0.5)

        velocity = None
        drive = None
        for _ in range(self.steps):
            if drive is None:
                drive, _ = self._drive(x_t, sigma, abt, ve_sigma, flow_t, a, is_flow, extra_args)
                x_t, velocity = self._advance(x_t, velocity, gamma, a, drive, d_amp, dt)
            else:
                x_t, velocity = self._advance(x_t, velocity, gamma, a, drive, d_amp, dt * 0.5)
                drive_new, _ = self._drive(x_t, sigma, abt, ve_sigma, flow_t, a, is_flow, extra_args)
                if velocity is not None:
                    velocity = velocity + torch.sqrt(gamma) * (drive_new - drive) * dt
                x_t, velocity = self._advance(x_t, velocity, gamma, a, drive_new, d_amp, dt * 0.5)
                drive = drive_new

        x = x_t / scale if is_flow else x_t * scale
        denoised, _ = self._denoise(x, sigma, extra_args)
        return x, denoised * (1.0 - mask) + self.latent_image * mask

    def _drive(self, x_t, sigma, abt, ve_sigma, flow_t, a, is_flow, extra_args):
        """The constant force C of the SDE, from this state's score."""
        if is_flow:
            x_model = x_t / (abt**0.5 + (1.0 - abt) ** 0.5)
            level = flow_t
        else:
            x_model = x_t * (1.0 + ve_sigma**2) ** 0.5
            level = ve_sigma
        x0_pred, x0_big = self._denoise(x_model, level, extra_args)

        score_free = -(x_t - x0_pred)
        score_known = (-(1.0 + self.lam) * (x_t - self.latent_image)
                       + self.lam * (x_t - x0_big))
        score = score_free * (1.0 - self.mask_known) + score_known * self.mask_known

        x0 = x_t + score
        one_minus_abt = torch.clamp(1.0 - abt, min=1e-6)
        drive = (abt**0.5 * x0 - x_t) / one_minus_abt + a * x_t
        return drive, x0

    @staticmethod
    def _advance(x_t, velocity, gamma, a, drive, d_amp, dt):
        dtype = x_t.dtype
        y, v = evolve(x_t.float(), None if velocity is None else velocity.float(),
                      gamma.float(), a.float(), drive.float(), d_amp.float(), dt.float())
        if not torch.isfinite(y).all() or (v is not None and not torch.isfinite(v).all()):
            logger.warning("%s inertial step went non-finite — using the overdamped limit",
                           LOG_PREFIX)
            y = evolve_overdamped(x_t.float(), gamma.float(), a.float(),
                                  drive.float(), d_amp.float(), dt.float())
            v = None
        return y.to(dtype), (None if v is None else v.to(dtype))


class TS_LangevinInpaint(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LangevinInpaint",
            display_name="TS Langevin Inpaint",
            category="TS/Image/Retouch",
            description=(
                "Inpaint sampler that spends inner Langevin steps at every noise "
                "level so the repainted region agrees with the pixels around it. "
                "Works with any model family; feed it a latent carrying a noise mask."
            ),
            inputs=[
                IO.Model.Input("model"),
                IO.Conditioning.Input("positive"),
                IO.Conditioning.Input("negative"),
                IO.Latent.Input("latent_image", tooltip="Latent with a noise mask (white = repaint)."),
                IO.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True),
                IO.Int.Input("steps", default=20, min=1, max=200),
                IO.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                IO.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                IO.Int.Input(
                    "think_steps", default=5, min=0, max=20,
                    tooltip="Inner Langevin steps per noise level. 0 disables the "
                            "refinement and leaves an ordinary masked sample.",
                ),
                IO.Float.Input(
                    "guidance", default=16.0, min=0.1, max=50.0, step=0.1, advanced=True,
                    tooltip="How hard the masked region is pulled towards agreement "
                            "with the known pixels. Higher binds tighter, too high "
                            "can oscillate.",
                ),
                IO.Float.Input(
                    "step_size", default=0.2, min=0.001, max=1.0, step=0.01, advanced=True,
                    tooltip="Langevin step size. Larger converges faster, less stably.",
                ),
                IO.Float.Input(
                    "beta", default=1.0, min=0.01, max=5.0, step=0.1, advanced=True,
                    tooltip="Time-step ratio of the masked branch. Lower it to "
                            "compensate a high guidance value.",
                ),
                IO.Float.Input(
                    "friction", default=15.0, min=0.0, max=50.0, step=0.1, advanced=True,
                    tooltip="Damping of the inertial dynamics. Lower converges "
                            "faster and is less stable.",
                ),
            ],
            outputs=[IO.Latent.Output(display_name="LATENT")],
            search_aliases=["langevin inpaint", "lanpaint", "inpaint sampler"],
        )

    @classmethod
    def execute(cls, model, positive, negative, latent_image, seed: int, steps: int,
                cfg: float, sampler_name: str, scheduler: str, denoise: float,
                think_steps: int, guidance: float, step_size: float, beta: float,
                friction: float) -> IO.NodeOutput:
        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        noise_mask = latent_image.get("noise_mask")
        if noise_mask is None:
            raise RuntimeError(
                f"{LOG_PREFIX} the latent carries no mask — add Set Latent Noise Mask "
                "(or a VAE Encode for Inpainting) before this node."
            )

        noise = comfy.sample.prepare_noise(latent, seed, latent_image.get("batch_index"))
        model_sampling = model.get_model_object("model_sampling")
        sampler_object = comfy.samplers.sampler_object(sampler_name)
        sigmas = comfy.samplers.calculate_sigmas(
            model_sampling, scheduler, steps).to(latent.device)
        if denoise < 1.0:
            keep = max(1, int(round(steps * denoise)))
            sigmas = sigmas[-(keep + 1):]

        is_flow = _looks_like_flow(model)
        mask_known = _match_mask(noise_mask, latent).to(latent.device)
        # 1 marks what must be preserved; the input mask marks what to repaint.
        mask_known = 1.0 - (mask_known > 0.5).to(latent.dtype)

        inner = _LangevinInner(
            model=None, latent_image=latent.to(latent.device), noise=noise.to(latent.device),
            mask_known=mask_known,
            options={"steps": think_steps, "guidance": guidance, "step_size": step_size,
                     "beta": beta, "friction": friction},
        )

        pbar = comfy.utils.ProgressBar(len(sigmas) - 1)

        def sample_fn(inner_model, x, sigmas_in, extra_args=None, callback=None,
                      disable=None, **_kwargs):
            inner.model = inner_model
            # Take the anchor latent and its noise from the engine's own inpaint
            # wrapper. Ours would be the RAW latent, while the engine samples in
            # process_latent_in space — anchoring with the raw one printed a
            # bright block into the mask (measured, even with the dynamics off).
            engine_latent = getattr(inner_model, "latent_image", None)
            engine_noise = getattr(inner_model, "noise", None)
            if engine_latent is not None:
                inner.latent_image = engine_latent.to(x.device, x.dtype)
            else:
                inner.latent_image = inner.latent_image.to(x.device, x.dtype)
            if engine_noise is not None:
                inner.noise = engine_noise.to(x.device, x.dtype)
            else:
                inner.noise = inner.noise.to(x.device, x.dtype)
            inner.mask_known = inner.mask_known.to(x.device, x.dtype)
            extra_args = dict(extra_args or {})
            # The engine's inpaint wrapper takes denoise_mask positionally; we
            # do the masking ourselves, so hand it an explicit None.
            extra_args["denoise_mask"] = None
            for index in range(len(sigmas_in) - 1):
                sigma = sigmas_in[index]
                sigma_next = sigmas_in[index + 1]
                x, denoised = inner.refine(x, sigma, is_flow, extra_args, model_sampling)
                if callback is not None:
                    callback({"x": x, "i": index, "sigma": sigma,
                              "sigma_hat": sigma, "denoised": denoised})
                pbar.update(1)
                if float(sigma_next) == 0.0:
                    x = denoised
                else:
                    step = _sampler_step(x, denoised, sigma, sigma_next, is_flow)
                    x = step
            return x

        sampler = comfy.samplers.KSAMPLER(sample_fn)
        logger.info("%s %d steps x %d inner (guidance %.1f, step %.2f, beta %.2f, "
                    "friction %.1f) sampler=%s flow=%s sigma[0]=%.3f",
                    LOG_PREFIX, len(sigmas) - 1, think_steps, guidance, step_size,
                    beta, friction, sampler_name, is_flow, float(sigmas[0]))

        callback = latent_preview.prepare_callback(model, len(sigmas) - 1)
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative, latent,
            noise_mask=None, callback=callback, disable_pbar=True, seed=seed,
        )
        del sampler_object
        out = dict(latent_image)
        out["samples"] = samples
        return IO.NodeOutput(out)


def _sampler_step(x, denoised, sigma, sigma_next, is_flow):
    """One deterministic Euler step towards the next noise level."""
    if is_flow:
        # x_t = (1-t) x0 + t eps  ->  eps = (x - (1-t) x0) / t
        t = torch.clamp(sigma, min=1e-6)
        eps = (x - (1.0 - t) * denoised) / t
        return (1.0 - sigma_next) * denoised + sigma_next * eps
    d = (x - denoised) / torch.clamp(sigma, min=1e-6)
    return x + d * (sigma_next - sigma)


def _looks_like_flow(model) -> bool:
    """Is this a flow-matching family?

    Decided by the sampling object rather than the model type: CONST is exactly
    the class whose noise_scaling reads (1-sigma)*x0 + sigma*eps, which is the
    parameterisation the dynamics has to switch frames for. A model type string
    can lag behind a ModelSampling patch; this cannot.
    """
    try:
        import comfy.model_sampling

        sampling = model.get_model_object("model_sampling")
        if isinstance(sampling, comfy.model_sampling.CONST):
            return True
        return isinstance(sampling, comfy.model_sampling.ModelSamplingDiscreteFlow)
    except Exception:                       # noqa: BLE001 - detection must not fail a run
        return False


def _match_mask(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """Bring a mask to the latent's shape, whatever rank it arrives in."""
    m = mask
    while m.ndim < latent.ndim:
        m = m.unsqueeze(1) if m.ndim == 3 else m.unsqueeze(0)
    if m.shape[-2:] != latent.shape[-2:]:
        m = comfy.utils.common_upscale(
            m.reshape(-1, 1, m.shape[-2], m.shape[-1]),
            latent.shape[-1], latent.shape[-2], "bilinear", "center")
        m = m.reshape(latent.shape[0], 1, latent.shape[-2], latent.shape[-1])
    return m.expand_as(latent) if m.shape != latent.shape else m


NODE_CLASS_MAPPINGS = {"TS_LangevinInpaint": TS_LangevinInpaint}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LangevinInpaint": "TS Langevin Inpaint"}
