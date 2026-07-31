import functools
import importlib
import inspect
import logging
import re
from typing import Any


class TSDependencyManager:
    """Centralized dependency / runtime guard for TS nodes.

    Since release 8.9 the pack is V3-only; the legacy V1 fallback that
    invented typed default outputs from `RETURN_TYPES` was removed in
    favour of a single `_wrap_execute_classmethod` path that re-raises
    a normalised `RuntimeError`.
    """

    _IMPORT_CACHE: dict[str, Any] = {}

    @classmethod
    def import_optional(cls, module_name: str) -> Any | None:
        if module_name in cls._IMPORT_CACHE:
            return cls._IMPORT_CACHE[module_name]
        try:
            module = importlib.import_module(module_name)
            cls._IMPORT_CACHE[module_name] = module
            return module
        except Exception:
            # Deliberately caches the failure for the process lifetime:
            # installing a package into a running ComfyUI requires a restart
            # anyway, and retry-per-call would re-pay the import machinery on
            # every execute() of every node that probes an absent dependency.
            cls._IMPORT_CACHE[module_name] = None
            return None

    @staticmethod
    def extract_missing_dependency(exc: Exception) -> str | None:
        if isinstance(exc, ModuleNotFoundError):
            missing = getattr(exc, "name", None)
            if missing:
                return str(missing)
        msg = str(exc)
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", msg)
        return match.group(1) if match else None

    # How to actually install what is missing. An import name is not always the
    # package name, and several of the pack's optional dependencies come as a
    # set — telling someone `pip install demucs` alone leaves them stuck on the
    # next missing import. Extras from pyproject.toml are preferred where they
    # exist; anything unlisted falls back to its own import name.
    _INSTALL_HINTS = {
        "demucs": "pip install comfyui-timesaver[audio-stems]",
        "geomloss": "pip install comfyui-timesaver[audio-stems]",
        "pykeops": "pip install comfyui-timesaver[audio-stems]",
        "bitsandbytes": "pip install comfyui-timesaver[llm-quant]",
        "silero": "pip install comfyui-timesaver[audio-silero]",
        "silero_stress": "pip install comfyui-timesaver[audio-silero]",
        "nvvfx": "pip install nvidia-vfx  (NVIDIA RTX GPUs only)",
        "cv2": "pip install opencv-python",
        "PIL": "pip install Pillow",
        "spandrel": "pip install spandrel",
        "py360convert": "pip install py360convert",
        "importlib_resources": "pip install importlib_resources",
    }

    @classmethod
    def install_hint(cls, module_name: str) -> str:
        root = str(module_name or "").split(".", 1)[0]
        return cls._INSTALL_HINTS.get(root, f"pip install {root}" if root else "")

    @classmethod
    def build_error_message(cls, node_name: str, method_name: str, exc: Exception) -> str:
        missing = cls.extract_missing_dependency(exc)
        if missing:
            hint = cls.install_hint(missing)
            hint_text = f" Install it with: {hint}." if hint else ""
            return (
                f"[TS RuntimeGuard] Node '{node_name}' failed in '{method_name}'. "
                f"Missing dependency: '{missing}'.{hint_text} Original error: {exc}"
            )
        return (
            f"[TS RuntimeGuard] Node '{node_name}' failed in '{method_name}'. "
            f"Original error: {exc}"
        )

    @classmethod
    def _wrap_execute_classmethod(cls, node_cls: type, logger: logging.Logger, node_name: str) -> None:
        """V3 runtime guard: catch every exception out of `execute()` and
        re-raise it as a `RuntimeError` whose message has the TS prefix and
        an actionable hint when a dependency is missing. ComfyUI's own
        prompt-execution code surfaces the message directly to the workflow
        UI, so the user sees the cleaned-up text instead of a raw stack
        trace from inside an optional dependency.
        """
        method_name = "execute"
        static_attr = inspect.getattr_static(node_cls, method_name, None)
        if not isinstance(static_attr, classmethod):
            return

        original = static_attr.__func__
        if getattr(original, "_ts_runtime_guard_wrapped", False):
            return

        if inspect.iscoroutinefunction(original):
            # An async node must stay async through the guard. ComfyUI decides
            # whether to await a node by calling inspect.iscoroutinefunction on
            # the bound function (execution.py), and that check does NOT follow
            # functools.wraps' __wrapped__ link — a sync wrapper would hand the
            # engine an un-awaited coroutine object as the node's output.
            @functools.wraps(original)
            async def wrapped(inner_cls, *args, **kwargs):
                try:
                    return await original(inner_cls, *args, **kwargs)
                except Exception as exc:
                    message = cls.build_error_message(node_name, method_name, exc)
                    logger.exception(message)
                    raise RuntimeError(message) from exc
        else:
            @functools.wraps(original)
            def wrapped(inner_cls, *args, **kwargs):
                try:
                    return original(inner_cls, *args, **kwargs)
                except Exception as exc:
                    message = cls.build_error_message(node_name, method_name, exc)
                    logger.exception(message)
                    raise RuntimeError(message) from exc

        wrapped._ts_runtime_guard_wrapped = True
        setattr(node_cls, method_name, classmethod(wrapped))

    @classmethod
    def wrap_node_runtime(cls, node_name: str, node_cls: type, logger: logging.Logger) -> None:
        cls._wrap_execute_classmethod(node_cls, logger, node_name)
