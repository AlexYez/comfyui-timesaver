"""TS Samplers Injector — registers extra samplers into ComfyUI core.

Like ``ts_schedulers_injector``, this module has no public node
(``NODE_CLASS_MAPPINGS`` is empty). It exists only for the import-time side
effect of registering a new sampler function so the name shows up in every
native ``KSampler`` / ``SamplerCustom`` dropdown.

Registered sampler:
- ``res_2s`` — a 2nd-order single-step exponential Runge-Kutta ("RES") solver.

Attribution / licensing
-----------------------
The ``res_2s`` *name* and method were popularised by the RES4LYF pack
(https://github.com/ClownsharkBatwing/RES4LYF, AGPL-3.0). RES4LYF's code is
NOT used or copied here. The implementation below is written independently
from the public mathematics of exponential Runge-Kutta integrators (the
"Refined Exponential Solver" / DPM-Solver family): the order-2 exponential RK
with the Butcher tableau

    c  = [0, c2]
    a21 = c2 * phi_1(c2 * z)
    b1  = phi_1(z) - phi_2(z) / c2
    b2  = phi_2(z) / c2

evaluated in the half-log-SNR (lambda = -log(sigma)) domain, where
phi_1(z) = (e^z - 1)/z and phi_2(z) = (e^z - 1 - z)/z^2 and z = -h is the
step in lambda. This reduces to exponential Euler (DDIM) when the two model
evaluations agree, and is exact to 2nd order otherwise.
"""

from __future__ import annotations

import logging

import torch
from tqdm.auto import trange

LOG_PREFIX = "[TS SamplersInjector]"
_LOGGER = logging.getLogger("comfyui_timesaver.ts_samplers_injector")

# Loader expects these symbols on every nodes/**/ts_*.py file. The injector is
# intentionally invisible in the node menu — only the registration below runs.
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

_RES_2S_C2 = 0.5  # intermediate stage position (free parameter of res_2s)


def sample_res_2s(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    """2nd-order exponential Runge-Kutta ("RES" res_2s), deterministic ODE.

    ``model(x, sigma)`` returns the denoised x0 prediction. Each step solves
    the probability-flow ODE in the lambda = -log(sigma) domain with the
    linear part integrated exactly (exponential integrator).
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    c2 = _RES_2S_C2

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

        if sigma_next == 0:
            # Final step: exponential Euler to sigma=0 collapses to the data
            # prediction. (The 2-stage correction is singular as h -> inf.)
            x = denoised
            continue

        lam = -torch.log(sigma)
        lam_next = -torch.log(sigma_next)
        h = lam_next - lam  # > 0, step in lambda

        # Stage 2: exponential-Euler probe to the intermediate level c2*h.
        a = torch.exp(-c2 * h)            # e^{-c2 h}
        sigma_s = sigma * a               # = exp(-(lam + c2 h))
        x_2 = a * x + (1.0 - a) * denoised
        denoised_2 = model(x_2, sigma_s * s_in, **extra_args)

        # Final update: exponential Euler + the phi_2 second-order correction.
        e = torch.exp(-h)                 # e^{-h}
        phi2h = (e - 1.0 + h) / h         # = h * phi_2(-h)
        x = e * x + (1.0 - e) * denoised + (phi2h / c2) * (denoised_2 - denoised)

    return x


def _register_res_2s() -> None:
    try:
        import comfy.samplers as comfy_samplers
        from comfy.k_diffusion import sampling as k_diffusion_sampling
    except ImportError as exc:
        _LOGGER.warning(
            "%s comfy sampler API unavailable (%s); skipping res_2s registration.",
            LOG_PREFIX,
            exc,
        )
        return
    except Exception as exc:  # pragma: no cover - never crash pack load
        _LOGGER.warning("%s Unexpected error resolving comfy sampler API: %s; skipping.", LOG_PREFIX, exc)
        return

    name = "res_2s"

    # ksampler() resolves a sampler via getattr(k_diffusion_sampling,
    # "sample_<name>"), so attach the function there. Don't clobber an existing
    # one (e.g. if RES4LYF itself is installed).
    if hasattr(k_diffusion_sampling, f"sample_{name}"):
        _LOGGER.info("%s 'sample_%s' already present; leaving it untouched.", LOG_PREFIX, name)
    else:
        setattr(k_diffusion_sampling, f"sample_{name}", sample_res_2s)

    # The dropdowns read SAMPLER_NAMES (KSampler.SAMPLERS is the same list
    # object). KSAMPLER_NAMES is a separate list, so append to both.
    added = False
    for list_name in ("KSAMPLER_NAMES", "SAMPLER_NAMES"):
        names = getattr(comfy_samplers, list_name, None)
        if isinstance(names, list) and name not in names:
            names.append(name)
            added = True

    if added:
        _LOGGER.info("%s Registered sampler '%s' (exponential RK2 / RES).", LOG_PREFIX, name)
    else:
        _LOGGER.info("%s Sampler '%s' already registered; nothing to do.", LOG_PREFIX, name)


_register_res_2s()
