"""Shared fakes for ``nodes.llm._qwen_engine`` used by the test suite.

The real engine pulls in ``torch`` / ``comfy.model_management`` /
``folder_paths`` / ``transformers`` / ``PIL`` — everything we don't want to
require for cheap contract tests. The fakes here implement only the surface
that ``nodes.llm.super_prompt._qwen`` and ``nodes.llm.ts_qwen3_vl`` reach
into when running the tests.

The ``_DummyEngine`` class plus the four module-level chat-template helpers
mirror the real engine API one-for-one. Keep them in lockstep with
``nodes/llm/_qwen_engine.py`` — if you add a method that production code
calls during contract tests, add the stub here too.

Files in ``tests/`` whose names start with ``_`` are skipped by pytest's
default collection, so this module won't be picked up as a test file.
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Any


# ---------------------------------------------------------------------------
# DummyEngine — minimal stand-in for QwenEngine
# ---------------------------------------------------------------------------


class _DummyEngine:
    """Implements only the surface the test suite actually pokes at."""

    @classmethod
    def is_bitsandbytes_available(cls) -> bool:
        return False

    @classmethod
    def model_size_b(cls, _model_id: str) -> float:
        return 2.0

    @staticmethod
    def get_tokenizer_from_processor(processor):
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        return processor

    @staticmethod
    def tensor_to_pil_list(_tensor):  # pragma: no cover — defensive default
        return []

    @staticmethod
    def resize_and_crop_image(image, _max_size):  # pragma: no cover
        return image

    @staticmethod
    def normalize_to_pil_list(image):  # pragma: no cover
        if image is None:
            return []
        if isinstance(image, (list, tuple)):
            return list(image)
        return [image]

    @staticmethod
    def strip_thinking_block(text: str) -> str:
        import re

        cleaned = re.sub(
            r"<think>.*?</think>", "", str(text or ""), flags=re.IGNORECASE | re.DOTALL
        )
        return cleaned.strip()

    def apply_chat_template_no_thinking(self, processor, messages):
        return apply_chat_template_no_thinking(self, processor, messages)


# ---------------------------------------------------------------------------
# Module-level chat-template helpers (mirror engine module)
# ---------------------------------------------------------------------------


def _chat_template_functions(engine, processor) -> list[Any]:
    template_fns: list[Any] = []
    processor_template = getattr(processor, "apply_chat_template", None)
    if processor_template is not None:
        template_fns.append(processor_template)

    tokenizer = engine.get_tokenizer_from_processor(processor)
    tokenizer_template = getattr(tokenizer, "apply_chat_template", None)
    if tokenizer_template is not None and tokenizer_template not in template_fns:
        template_fns.append(tokenizer_template)

    if not template_fns:
        raise RuntimeError("Loaded processor/tokenizer does not provide apply_chat_template.")
    return template_fns


def _template_accepts_kwargs(template_fn, kwargs: dict[str, Any]) -> bool:
    try:
        signature = inspect.signature(template_fn)
    except Exception:
        return True
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return True
    return all(key in parameters for key in kwargs.keys())


def _messages_have_visuals(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"image", "video"}:
                return True
    return False


def _flatten_text_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    flattened: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = [
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            text = "\n\n".join(part for part in text_parts if part)
        else:
            text = str(content or "")
        flattened.append({"role": str(message.get("role") or "user"), "content": text})
    return flattened


def apply_chat_template_no_thinking(engine, processor, messages):
    base_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    thinking_variants = (
        {"enable_thinking": False},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {},
    )
    message_variants: list[list[dict[str, Any]]] = [messages]
    if not _messages_have_visuals(messages):
        flattened_messages = _flatten_text_messages(messages)
        if flattened_messages != messages:
            message_variants.append(flattened_messages)

    last_error: Exception | None = None
    for template_fn in _chat_template_functions(engine, processor):
        for candidate_messages in message_variants:
            for thinking_kwargs in thinking_variants:
                kwargs = {**base_kwargs, **thinking_kwargs}
                if not _template_accepts_kwargs(template_fn, kwargs):
                    continue
                try:
                    return template_fn(candidate_messages, **kwargs)
                except TypeError as exc:
                    last_error = exc
                    continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Loaded chat template rejected all supported argument variants.")


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------


def install_engine_module_stub(monkeypatch) -> types.ModuleType:
    """Register a ``nodes.llm._qwen_engine`` stub module in ``sys.modules``.

    Returns the module so callers can poke additional attributes onto it
    (e.g. overriding ``get_qwen_engine`` to return a custom instance).
    """

    engine_module = types.ModuleType("nodes.llm._qwen_engine")
    engine_module.QwenEngine = _DummyEngine
    engine_module.get_qwen_engine = lambda: _DummyEngine()
    engine_module.apply_chat_template_no_thinking = apply_chat_template_no_thinking
    engine_module._chat_template_functions = _chat_template_functions
    engine_module._template_accepts_kwargs = _template_accepts_kwargs
    engine_module._messages_have_visuals = _messages_have_visuals
    engine_module._flatten_text_messages = _flatten_text_messages
    monkeypatch.setitem(sys.modules, "nodes.llm._qwen_engine", engine_module)
    return engine_module
