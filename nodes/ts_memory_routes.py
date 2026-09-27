"""HTTP surface of the «Free memory» button (js/utils/ts-free-memory.js).

A side-effect module, not a node: it registers one route and exposes no class
(same pattern as ``ts_pass_routes.py``). Nothing runs on import beyond the
registration.

What the button frees, in two halves:

* **ComfyUI's own models and cached results** — by the same flags the built-in
  «Unload Models and Execution Cache» command raises (``unload_models`` +
  ``free_memory`` on the prompt queue). ComfyUI acts on them in its prompt
  worker, between runs, so a running prompt is never pulled from under.
  Clearing the result cache is what gives the RAM back: an unloaded model
  moves to system memory and stays there while a cached loader output still
  references it.
* **The pack's own models**, which ComfyUI does not know about — Gemma on
  WebGPU, Qwen, Whisper, BiRefNet and the rest. Each owner registered how to
  release its cache (``nodes/_shared.py: register_memory_release``); this route
  only calls them. They are released only while nothing is running, for the
  same reason ComfyUI waits: some caches are read by a node mid-run.
"""
from __future__ import annotations

import asyncio
import gc
import inspect
import logging
import time

from ._shared import (
    MemoryReleaseBusy,
    make_route_registrars,
    memory_releasers,
    resolve_prompt_server,
)

logger = logging.getLogger("comfyui_timesaver.free_memory")
LOG_PREFIX = "[TS FreeMemory]"

_PROMPT_SERVER = resolve_prompt_server(lambda message: logger.warning("%s %s", LOG_PREFIX, message))
_register_get, _register_post = make_route_registrars(
    _PROMPT_SERVER, lambda message: logger.warning("%s %s", LOG_PREFIX, message))

# How long to watch the numbers settle after asking ComfyUI to unload. The
# worker acts at once when idle, but moving a 20 GB model off the card and
# collecting it takes seconds, and reporting before that would say "0 freed".
SETTLE_TIMEOUT = 15.0
SETTLE_STEP = 0.3
SETTLE_EPSILON = 64 * 1024 * 1024
# While a prompt runs, the pack's caches are released once the queue is empty.
DEFER_POLL = 1.0
DEFER_LIMIT = 6 * 3600.0


class _RouteState:
    """The one deferred release in flight, if any — a second press waits on it."""

    deferred: asyncio.Task | None = None


_state = _RouteState()


def _queue():
    return getattr(_PROMPT_SERVER, "prompt_queue", None)


def queue_busy(queue) -> bool:
    """Is a prompt running or waiting? Unknown counts as idle."""
    if queue is None:
        return False
    try:
        return int(queue.get_tasks_remaining()) > 0
    except Exception as exc:                # noqa: BLE001 - someone else's queue
        logger.debug("%s cannot read the queue: %s", LOG_PREFIX, exc)
        return False


def snapshot() -> dict:
    """Free video memory (the driver's view) and available RAM, in bytes.

    ``None`` where it cannot be measured — no CUDA device, no psutil. The
    driver's number is the one another application will see, which is the
    whole point of the button; torch's own reserve is not.
    """
    vram = None
    try:
        import comfy.model_management as mm
        import torch

        device = mm.get_torch_device()
        if getattr(device, "type", "") == "cuda" and torch.cuda.is_available():
            vram = int(torch.cuda.mem_get_info(device)[0])
    except Exception as exc:                # noqa: BLE001 - measuring is best-effort
        logger.debug("%s cannot read video memory: %s", LOG_PREFIX, exc)
    ram = None
    try:
        import psutil

        ram = int(psutil.virtual_memory().available)
    except Exception as exc:                # noqa: BLE001
        logger.debug("%s cannot read RAM: %s", LOG_PREFIX, exc)
    return {"vram_free": vram, "ram_available": ram}


def _collect() -> None:
    """One garbage collection and one cache flush after all releases."""
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception as exc:                # noqa: BLE001 - outside ComfyUI
        logger.debug("%s soft_empty_cache failed: %s", LOG_PREFIX, exc)


async def release_pack_memory() -> dict:
    """Call every registered release; report who let go, who was busy.

    Each release runs in a worker thread — several of them wait on a lock
    that a generation holds — except the ones declared ``async`` (their lock
    lives on the event loop). One failing release never stops the others.
    """
    released: list[str] = []
    busy: list[str] = []
    failed: list[str] = []
    for name, release in memory_releasers().items():
        try:
            if inspect.iscoroutinefunction(release):
                freed = await release()
            else:
                freed = await asyncio.to_thread(release)
        except MemoryReleaseBusy:
            busy.append(name)
            continue
        except Exception as exc:            # noqa: BLE001 - one cache must not stop the rest
            logger.warning("%s releasing '%s' failed: %s", LOG_PREFIX, name, exc)
            failed.append(name)
            continue
        if freed:
            released.append(name)
    if released:
        await asyncio.to_thread(_collect)
    if released or busy or failed:
        logger.info("%s pack models released: %s; busy: %s; failed: %s", LOG_PREFIX,
                    ", ".join(released) or "none", ", ".join(busy) or "none",
                    ", ".join(failed) or "none")
    return {"released": released, "busy": busy, "failed": failed}


def ask_comfyui_to_unload(queue) -> bool:
    """Raise ComfyUI's own flags: unload every model, reset the result cache."""
    if queue is None:
        return False
    try:
        queue.set_flag("unload_models", True)
        queue.set_flag("free_memory", True)
        return True
    except Exception as exc:                # noqa: BLE001
        logger.warning("%s ComfyUI refused the unload flags: %s", LOG_PREFIX, exc)
        return False


def _flags_pending(queue) -> bool:
    try:
        return bool(queue.get_flags(reset=False))
    except Exception:                       # noqa: BLE001 - older queue API
        return False


async def settle(queue, before: dict) -> dict:
    """Wait for ComfyUI to act on the flags, then for the numbers to stop moving."""
    deadline = time.monotonic() + SETTLE_TIMEOUT
    while queue is not None and _flags_pending(queue) and time.monotonic() < deadline:
        await asyncio.sleep(SETTLE_STEP)
    previous = before
    current = await asyncio.to_thread(snapshot)
    while time.monotonic() < deadline:
        await asyncio.sleep(SETTLE_STEP)
        previous, current = current, await asyncio.to_thread(snapshot)
        grew = [
            (current.get(key) or 0) - (previous.get(key) or 0)
            for key in ("vram_free", "ram_available")
        ]
        if all(delta < SETTLE_EPSILON for delta in grew):
            break
    return current


def freed(before: dict, after: dict, key: str):
    """Bytes freed by one measure, or None when it could not be measured."""
    if before.get(key) is None or after.get(key) is None:
        return None
    return max(0, int(after[key]) - int(before[key]))


async def _release_when_idle(queue) -> None:
    started = time.monotonic()
    try:
        while time.monotonic() - started < DEFER_LIMIT:
            await asyncio.sleep(DEFER_POLL)
            if not queue_busy(queue):
                await release_pack_memory()
                return
        logger.info("%s the queue never went idle; pack models were left loaded.", LOG_PREFIX)
    finally:
        _state.deferred = None


@_register_post("/ts_memory/free")
async def free_memory_route(_request):
    """Unload every model and clear cached results; say how much came back."""
    from aiohttp import web

    queue = _queue()
    before = await asyncio.to_thread(snapshot)
    asked = ask_comfyui_to_unload(queue)

    if queue_busy(queue):
        # ComfyUI frees its part after the running prompt; the pack's part
        # waits until the queue is empty. Say so instead of reporting zero.
        if _state.deferred is None:
            _state.deferred = asyncio.get_running_loop().create_task(_release_when_idle(queue))
        return web.json_response({"deferred": True, "comfyui": asked})

    report = await release_pack_memory()
    after = await settle(queue if asked else None, before)
    return web.json_response({
        "deferred": False,
        "comfyui": asked,
        **report,
        "vram_freed": freed(before, after, "vram_free"),
        "ram_freed": freed(before, after, "ram_available"),
        "vram_free": after.get("vram_free"),
        "ram_available": after.get("ram_available"),
    })


NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
