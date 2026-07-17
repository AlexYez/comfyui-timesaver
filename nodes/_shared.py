"""Pack-level shared helpers used by multiple subpackages.

Private module: not registered as a public node by the loader (the
underscore prefix is honored by `_discover_module_entries` in __init__.py).
"""

import logging
from typing import Callable

_logger = logging.getLogger("comfyui_timesaver.ts_shared")


class TS_Logger:
    """Thin facade over stdlib logging for slider/switch/math/animation_preview nodes."""

    @staticmethod
    def log(node_name: str, message: str) -> None:
        _logger.info("[TS %s] %s", node_name, message)

    @staticmethod
    def warn(node_name: str, message: str) -> None:
        _logger.warning("[TS %s] %s", node_name, message)

    @staticmethod
    def error(node_name: str, message: str) -> None:
        _logger.error("[TS %s] %s", node_name, message)


# ---------------------------------------------------------------------------
# aiohttp route registration (shared by the route-owning helper modules)
# ---------------------------------------------------------------------------
# TS_LamaCleanup, TS_SAM_MediaLoader, TS_AudioLoader, TS_SuperPrompt and the
# text nodes each register aiohttp routes on the ComfyUI PromptServer with the
# exact same boilerplate. These two helpers own that boilerplate in one place;
# each caller keeps its own module-level registrar names and warning logger.


def resolve_prompt_server(warn: Callable[[str], None]):
    """Return ``server.PromptServer.instance`` or None (HTTP routes disabled).

    ``warn`` is the caller's own warning logger, so each module's log prefix is
    preserved. ``server`` is imported lazily so this module stays cheap to
    import and free of a hard dependency on the ComfyUI runtime being ready.
    """
    try:
        import server
    except Exception:
        server = None
    if server is None:
        warn("PromptServer unavailable. HTTP routes disabled.")
        return None
    try:
        return server.PromptServer.instance
    except Exception as exc:
        warn(f"PromptServer init failed. HTTP routes disabled: {exc}")
        return None


def make_route_registrars(prompt_server, warn: Callable[[str], None]):
    """Return ``(register_get, register_post)`` decorators bound to
    ``prompt_server``.

    Each decorator registers ``func`` on the given HTTP method and returns it
    unchanged, and is a no-op passthrough when ``prompt_server`` is None (server
    not ready) — matching the standalone helpers these replace exactly.
    """

    def _make(method_name: str):
        def register(path: str):
            def decorator(func):
                if prompt_server is None:
                    return func
                try:
                    getattr(prompt_server.routes, method_name)(path)(func)
                except Exception as exc:
                    warn(f"Failed to register {method_name.upper()} route '{path}': {exc}")
                return func

            return decorator

        return register

    return _make("get"), _make("post")
