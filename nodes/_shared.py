"""Pack-level shared helpers used by multiple subpackages.

Private module: not registered as a public node by the loader (the
underscore prefix is honored by `_discover_module_entries` in __init__.py).
"""

import logging

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
