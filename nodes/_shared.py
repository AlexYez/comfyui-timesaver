"""Pack-level shared helpers used by multiple subpackages.

Private module: not registered as a public node by the loader (the
underscore prefix is honored by `_discover_module_entries` in __init__.py).
"""


class TS_Logger:
    """Minimal logging utility used by interface/animation/utility nodes."""

    @staticmethod
    def log(node_name, message, color="cyan"):
        print(f"[TS {node_name}] {message}")

    @staticmethod
    def error(node_name, message):
        TS_Logger.log(node_name, message, "red")
