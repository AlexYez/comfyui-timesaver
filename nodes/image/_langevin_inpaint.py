"""Langevin inpainting core — the maths, written from the equations.

The scheme this implements is the one LanPaint describes: at every noise level
of an ordinary sampler, run a few steps of *inertial* Langevin dynamics that
pull the masked region towards a state consistent with the known pixels, then
hand the result back to the sampler. Two things make it stronger than plain
resampling (denoise → re-noise): the masked branch is driven by a different
score, and the update is the EXACT solution of the linear SDE rather than an
Euler step.

Nothing here is transcribed from another implementation — the closed forms
below are derived from the process itself:

    dy = √Γ · v dt
    dv = √Γ (C − A y) dt − Γ v dt + √Γ D dW

This is an Ornstein–Uhlenbeck process in (y, v). Writing it as
d z = (M z + b) dt + Σ dW with

    M = [[0, √Γ], [−√Γ A, −Γ]],   b = [0, √Γ C],   Σ = diag(0, √Γ D)

three facts give the exact update:

  1. the drift has a fixed point z* = (C/A, 0), and z(t) − z* evolves by the
     matrix exponential of M;
  2. e^{Mt} for a 2×2 matrix is  e^{−Γt/2}·[cosh(ωt)·I + sinh(ωt)/ω ·(M + Γ/2·I)]
     with ω² = Γ²/4 − ΓA — real for an overdamped system, imaginary for an
     underdamped one, and the same formula covers both once cosh/sinh are
     evaluated on the imaginary axis (they become cos/sin);
  3. the stationary covariance of this process is diag(D²/2A, D²/2), so the
     covariance after time t is Σ∞ − e^{Mt} Σ∞ e^{Mᵀt} — no integral needed.

Everything else here is numerical care: series expansions where the closed
forms cancel to 0/0, and a Cholesky written out by hand for the 2×2 case.
"""
from __future__ import annotations

import math

import torch

# Below these thresholds the closed forms lose precision to cancellation and
# the series expansions take over.
_SMALL_ARG = 1e-3
_TINY = 1e-12


def _damped_cosh_sinh(gamma: torch.Tensor, omega_sq: torch.Tensor, t: torch.Tensor):
    """Return e^{−Γt/2}·cosh(ωt) and e^{−Γt/2}·sinh(ωt)/ω.

    The two factors have to be combined BEFORE evaluating: the sampler runs at
    Γt of a few hundred, where cosh(ωt) overflows on its own even though the
    product is O(1) — ω approaches Γ/2, so the surviving exponent is
    (Γt/2)(√Δ − 1), a small number. Writing the product as a sum of two
    exponentials keeps every exponent in range (measured: evaluating the
    factors separately cost 0.5 of absolute error here).

    For ω² < 0 the motion oscillates; cosh/sinh become cos/sin, which are
    bounded, so the plain product is safe there.
    """
    half_gamma_t = gamma * t / 2.0
    positive = omega_sq > 0
    omega = torch.sqrt(torch.clamp(omega_sq.abs(), min=_TINY))
    wt = omega * t

    # Damped branch: e^{-Γt/2}·cosh(ωt) = (e^{ωt-Γt/2} + e^{-ωt-Γt/2}) / 2
    hi = torch.exp(torch.clamp(wt - half_gamma_t, max=60.0))
    lo = torch.exp(torch.clamp(-wt - half_gamma_t, max=60.0))
    cosh_damped = (hi + lo) / 2.0
    sinh_over_w_damped = (hi - lo) / (2.0 * torch.clamp(omega, min=_TINY))

    # Oscillatory branch.
    decay = torch.exp(-half_gamma_t)
    cosh_osc = decay * torch.cos(wt)
    sinh_over_w_osc = decay * t * torch.sinc(wt / math.pi)

    cosh = torch.where(positive, cosh_damped, cosh_osc)
    sinh_over_w = torch.where(positive, sinh_over_w_damped, sinh_over_w_osc)

    # Series when ωt is negligible: cosh -> 1, sinh/ω -> t, both times decay.
    arg = omega_sq * t * t
    small = arg.abs() < _SMALL_ARG
    cosh = torch.where(small, decay * (1.0 + arg / 2.0 + arg * arg / 24.0), cosh)
    sinh_over_w = torch.where(
        small, decay * t * (1.0 + arg / 6.0 + arg * arg / 120.0), sinh_over_w)
    return cosh, sinh_over_w


def _propagator(gamma: torch.Tensor, a: torch.Tensor, t: torch.Tensor):
    """e^{Mt} for M = [[0, √Γ], [−√Γ A, −Γ]], returned as four blocks."""
    omega_sq = gamma * gamma / 4.0 - gamma * a
    cosh, sinh_over_w = _damped_cosh_sinh(gamma, omega_sq, t)
    root_gamma = torch.sqrt(torch.clamp(gamma, min=_TINY))

    # e^{Mt} = e^{−Γt/2}·[cosh·I + sinh/ω·(M + Γ/2·I)], with the decay already
    # folded into both terms above.
    e11 = cosh + sinh_over_w * gamma / 2.0
    e12 = sinh_over_w * root_gamma
    e21 = -sinh_over_w * root_gamma * a
    e22 = cosh - sinh_over_w * gamma / 2.0
    return e11, e12, e21, e22


def evolve(y0: torch.Tensor, v0: torch.Tensor, gamma: torch.Tensor, a: torch.Tensor,
           c: torch.Tensor, d: torch.Tensor, t: torch.Tensor,
           generator: torch.Generator | None = None):
    """Advance (y, v) by time t under the SDE above, exactly.

    @param y0 Position (the latent being refined).
    @param v0 Velocity, or None to draw it from equilibrium.
    @param gamma Damping Γ, @param a Restoring strength A.
    @param c Constant drive C, @param d Noise amplitude D.
    @returns (y, v) after time t, sampled from the exact transition kernel.
    """
    a_safe = torch.clamp(a, min=_TINY)
    if v0 is None:
        # Equilibrium velocity: variance D²/2.
        v0 = torch.randn(y0.shape, device=y0.device, dtype=y0.dtype,
                         generator=generator) * (d / math.sqrt(2.0))

    e11, e12, e21, e22 = _propagator(gamma, a_safe, t)

    # Mean: the fixed point plus the propagated offset.
    y_star = c / a_safe
    dy = y0 - y_star
    mean_y = y_star + e11 * dy + e12 * v0
    mean_v = e21 * dy + e22 * v0

    # Covariance: Σ∞ − E Σ∞ Eᵀ with Σ∞ = diag(D²/2A, D²/2).
    var_y_inf = d * d / (2.0 * a_safe)
    var_v_inf = d * d / 2.0
    cov_yy = var_y_inf - (e11 * e11 * var_y_inf + e12 * e12 * var_v_inf)
    cov_vv = var_v_inf - (e21 * e21 * var_y_inf + e22 * e22 * var_v_inf)
    cov_yv = -(e11 * e21 * var_y_inf + e12 * e22 * var_v_inf)

    cov_yy = torch.clamp(cov_yy, min=0.0)
    cov_vv = torch.clamp(cov_vv, min=0.0)

    # 2×2 Cholesky by hand: [[s, 0], [r, q]].
    s = torch.sqrt(torch.clamp(cov_yy, min=_TINY))
    r = cov_yv / s
    q = torch.sqrt(torch.clamp(cov_vv - r * r, min=0.0))

    n1 = torch.randn(y0.shape, device=y0.device, dtype=y0.dtype, generator=generator)
    n2 = torch.randn(y0.shape, device=y0.device, dtype=y0.dtype, generator=generator)
    y = mean_y + s * n1
    v = mean_v + r * n1 + q * n2
    return y, v


def evolve_overdamped(y: torch.Tensor, gamma: torch.Tensor, a: torch.Tensor,
                      c: torch.Tensor, d: torch.Tensor, t: torch.Tensor,
                      generator: torch.Generator | None = None):
    """The Γ → ∞ limit, used when the inertial update goes non-finite.

    Here dy = (C − A y) dt + D dW, whose exact kernel is a plain OU step.
    """
    a_safe = torch.clamp(a, min=_TINY)
    decay = torch.exp(-a_safe * t)
    mean = c / a_safe + (y - c / a_safe) * decay
    var = (d * d / (2.0 * a_safe)) * (1.0 - decay * decay)
    noise = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
    return mean + torch.sqrt(torch.clamp(var, min=0.0)) * noise
