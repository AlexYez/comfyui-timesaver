"""TS Schedulers Injector — registers extra sigma schedulers into ComfyUI core.

This module has no public ComfyUI node (``NODE_CLASS_MAPPINGS`` is empty on
purpose) — it is loaded by the pack's auto-discovery loader purely for the
import-time side effect of inserting new entries into
``comfy.samplers.SCHEDULER_HANDLERS`` / ``SCHEDULER_NAMES``. After this module
imports successfully, the new scheduler names show up in every native
``KSampler`` / ``KSamplerAdvanced`` / ``BasicScheduler`` dropdown without any
monkey-patching of the core nodes themselves.
"""

from __future__ import annotations

import logging
import math
from functools import partial

LOG_PREFIX = "[TS SchedulersInjector]"
_LOGGER = logging.getLogger("comfyui_timesaver.ts_schedulers_injector")

# Pack-level loader expects this symbol on every nodes/**/ts_*.py file.
# The injector is intentionally invisible in the node menu — we only need the
# import-time registration below.
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def _register_beta57() -> None:
    # The "beta57" preset (alpha=0.5, beta=0.7 over comfy.samplers.beta_scheduler)
    # was popularised by the RES4LYF pack (https://github.com/ClownsharkBatwing/RES4LYF).
    # No code from that pack is used here — only the well-known parameter pair.
    try:
        from comfy.samplers import (
            SCHEDULER_HANDLERS,
            SCHEDULER_NAMES,
            SchedulerHandler,
            beta_scheduler,
        )
    except ImportError as exc:
        _LOGGER.warning(
            "%s comfy.samplers public scheduler API unavailable (%s); skipping beta57 registration.",
            LOG_PREFIX,
            exc,
        )
        return
    except Exception as exc:  # pragma: no cover - defensive: never crash pack load
        _LOGGER.warning(
            "%s Unexpected error while resolving comfy.samplers API: %s; skipping beta57 registration.",
            LOG_PREFIX,
            exc,
        )
        return

    name = "beta57"
    if name in SCHEDULER_HANDLERS:
        _LOGGER.info(
            "%s Scheduler '%s' is already registered (another pack added it); leaving the existing handler untouched.",
            LOG_PREFIX,
            name,
        )
        return

    handler_callable = partial(beta_scheduler, alpha=0.5, beta=0.7)
    SCHEDULER_HANDLERS[name] = SchedulerHandler(handler=handler_callable, use_ms=True)
    if name not in SCHEDULER_NAMES:
        SCHEDULER_NAMES.append(name)
    _LOGGER.info("%s Registered scheduler '%s' (beta_scheduler alpha=0.5, beta=0.7).", LOG_PREFIX, name)


# ---------------------------------------------------------------------------
# bong_tangent — a two-stage arctangent sigma curve.
#
# Name/curve popularised by RES4LYF (https://github.com/ClownsharkBatwing/RES4LYF,
# AGPL-3.0). No RES4LYF code is used; the curve is reimplemented from its public
# math (a normalized arctangent easing, ((2/pi)*atan(-slope*(x-pivot))+1)/2,
# split into two stages start->middle->end). RES4LYF emits a normalized [0, 1]
# schedule (built for flow-matching models where sigma_max ~= 1); here it is
# scaled onto the model's actual sigma range so it also works on eps models.
# ---------------------------------------------------------------------------

def _bong_tangent_curve(steps: int, slope: float, pivot: float, start: float, end: float) -> list[float]:
    """One descending arctangent easing segment, normalized from `start` to `end`."""
    def f(x: float) -> float:
        return ((2.0 / math.pi) * math.atan(-slope * (x - pivot)) + 1.0) / 2.0

    if steps <= 1:
        return [start]
    smax = f(0)
    smin = f(steps - 1)
    srange = smax - smin
    if srange == 0:
        return [start for _ in range(steps)]
    scale = start - end
    return [(f(x) - smin) / srange * scale + end for x in range(steps)]


def _bong_tangent_normalized(steps: int) -> list[float]:
    """Two-stage arctangent curve normalized to [0, 1], descending, length
    steps+1 (first=1.0, last=0.0). Defaults match RES4LYF's bong_tangent."""
    n = int(steps) + 2
    start, middle, end = 1.0, 0.5, 0.0
    pivot_1 = pivot_2 = 0.6
    slope_1 = slope_2 = 0.2
    midpoint = int((n * pivot_1 + n * pivot_2) / 2)
    p1 = int(n * pivot_1)
    p2 = int(n * pivot_2)
    s1 = slope_1 / (n / 40.0)
    s2 = slope_2 / (n / 40.0)
    stage2_len = n - midpoint
    stage1_len = n - stage2_len  # == midpoint
    seg1 = _bong_tangent_curve(stage1_len, s1, p1, start, middle)
    seg2 = _bong_tangent_curve(stage2_len, s2, p2 - stage1_len, middle, end)
    return seg1[:-1] + seg2  # length n-1 == steps+1


def _bong_tangent_scheduler(model_sampling, steps):
    """ComfyUI scheduler handler (use_ms=True): bong_tangent curve scaled to
    the model's real sigma range so it works on both flow and eps models."""
    import torch

    sigma_max = float(getattr(model_sampling, "sigma_max", 1.0))
    norm = _bong_tangent_normalized(steps)
    return torch.tensor([v * sigma_max for v in norm], dtype=torch.float32)


def _register_bong_tangent() -> None:
    try:
        from comfy.samplers import SCHEDULER_HANDLERS, SCHEDULER_NAMES, SchedulerHandler
    except ImportError as exc:
        _LOGGER.warning(
            "%s comfy.samplers public scheduler API unavailable (%s); skipping bong_tangent registration.",
            LOG_PREFIX,
            exc,
        )
        return
    except Exception as exc:  # pragma: no cover - defensive: never crash pack load
        _LOGGER.warning("%s Unexpected error resolving comfy.samplers API: %s; skipping bong_tangent.", LOG_PREFIX, exc)
        return

    name = "bong_tangent"
    if name in SCHEDULER_HANDLERS:
        _LOGGER.info("%s Scheduler '%s' already registered; leaving the existing handler untouched.", LOG_PREFIX, name)
        return

    SCHEDULER_HANDLERS[name] = SchedulerHandler(handler=_bong_tangent_scheduler, use_ms=True)
    if name not in SCHEDULER_NAMES:
        SCHEDULER_NAMES.append(name)
    _LOGGER.info("%s Registered scheduler '%s' (two-stage arctangent curve).", LOG_PREFIX, name)


_register_beta57()
_register_bong_tangent()
