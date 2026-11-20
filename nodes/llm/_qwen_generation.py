"""Normalize saved preset parameters for Hugging Face Qwen generation."""

from typing import Any

_ALIASES = {
    "max_tokens": "max_new_tokens",
    "max_completion_tokens": "max_new_tokens",
}
_UNSUPPORTED = {
    "frequency_penalty", "n", "presence_penalty", "response_format", "stop", "stream",
}


def normalize_generation_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    for alias, target in _ALIASES.items():
        value = normalized.pop(alias, None)
        if value is not None and target not in normalized:
            normalized[target] = value
    for key in _UNSUPPORTED:
        normalized.pop(key, None)
    return normalized
