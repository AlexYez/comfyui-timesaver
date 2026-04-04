import importlib
import inspect
import logging
import re
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class TSDependencyManager:
    """
    Centralized dependency/runtime guard for TS nodes.
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

    @classmethod
    def build_error_message(cls, node_name: str, method_name: str, exc: Exception) -> str:
        missing = cls.extract_missing_dependency(exc)
        if missing:
            return (
                f"[TS RuntimeGuard] Node '{node_name}' failed in '{method_name}'. "
                f"Missing dependency: '{missing}'. Original error: {exc}"
            )
        return (
            f"[TS RuntimeGuard] Node '{node_name}' failed in '{method_name}'. "
            f"Original error: {exc}"
        )

    @classmethod
    def fallback_value_for_type(cls, output_type: Any, error_message: str) -> Any:
        if not isinstance(output_type, str):
            return None

        output_type = output_type.upper()
        if output_type == "STRING":
            return error_message
        if output_type == "INT":
            return 0
        if output_type == "FLOAT":
            return 0.0
        if output_type == "BOOLEAN":
            return False
        if output_type == "IMAGE":
            if torch is not None:
                return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return None
        if output_type == "MASK":
            if torch is not None:
                return torch.zeros((1, 64, 64), dtype=torch.float32)
            return None
        if output_type == "LATENT":
            if torch is not None:
                return {"samples": torch.zeros((1, 4, 8, 8), dtype=torch.float32)}
            return {"samples": None}
        if output_type == "AUDIO":
            if torch is not None:
                return {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}
            return {"waveform": None, "sample_rate": 44100}
        return None

    @classmethod
    def build_v1_fallback_output(cls, node_cls: type, error_message: str) -> tuple[Any, ...]:
        return_types = getattr(node_cls, "RETURN_TYPES", ())
        if not isinstance(return_types, (tuple, list)):
            return ()
        if len(return_types) == 0:
            return ()
        return tuple(cls.fallback_value_for_type(output_type, error_message) for output_type in return_types)

    @classmethod
    def _wrap_plain_method(cls, node_cls: type, method_name: str, logger: logging.Logger, node_name: str) -> None:
        original = getattr(node_cls, method_name, None)
        if original is None or not callable(original):
            return
        if getattr(original, "_ts_runtime_guard_wrapped", False):
            return

        def wrapped(self, *args, **kwargs):
            try:
                return original(self, *args, **kwargs)
            except Exception as exc:
                message = cls.build_error_message(node_name, method_name, exc)
                logger.exception(message)
                return cls.build_v1_fallback_output(node_cls, message)

        wrapped._ts_runtime_guard_wrapped = True
        setattr(node_cls, method_name, wrapped)

    @classmethod
    def _wrap_class_method(cls, node_cls: type, method_name: str, logger: logging.Logger, node_name: str) -> None:
        static_attr = inspect.getattr_static(node_cls, method_name, None)
        if not isinstance(static_attr, classmethod):
            return

        original = static_attr.__func__
        if getattr(original, "_ts_runtime_guard_wrapped", False):
            return

        def wrapped(inner_cls, *args, **kwargs):
            try:
                return original(inner_cls, *args, **kwargs)
            except Exception as exc:
                message = cls.build_error_message(node_name, method_name, exc)
                logger.exception(message)
                return cls.build_v1_fallback_output(node_cls, message)

        wrapped._ts_runtime_guard_wrapped = True
        setattr(node_cls, method_name, classmethod(wrapped))

    @classmethod
    def _wrap_execute_classmethod(cls, node_cls: type, logger: logging.Logger, node_name: str) -> None:
        """
        Best-effort guard for V3-style execute classmethod.
        For V3 nodes we re-raise as RuntimeError with normalized message.
        """
        method_name = "execute"
        static_attr = inspect.getattr_static(node_cls, method_name, None)
        if not isinstance(static_attr, classmethod):
            return

        original = static_attr.__func__
        if getattr(original, "_ts_runtime_guard_wrapped", False):
            return

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
        function_name = getattr(node_cls, "FUNCTION", None)
        if isinstance(function_name, str) and function_name:
            static_attr = inspect.getattr_static(node_cls, function_name, None)
            if isinstance(static_attr, classmethod):
                cls._wrap_class_method(node_cls, function_name, logger, node_name)
            else:
                cls._wrap_plain_method(node_cls, function_name, logger, node_name)
            return

        # V3 fallback: no FUNCTION attribute but classmethod execute exists.
        cls._wrap_execute_classmethod(node_cls, logger, node_name)
