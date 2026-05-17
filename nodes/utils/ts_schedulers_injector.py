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


_register_beta57()
