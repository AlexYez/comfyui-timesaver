"""Shared bits of TS Super Prompt RT: presets, events, attached-image paths.

Everything here is deliberately borrowed from the Qwen-based Super Prompt so
the two nodes behave the same where a user would expect them to — same preset
file, same attach-and-enhance flow, same progress events. What differs sits in
``_litert_engine`` and is about the runtime, not about the node.

The loader skips ``_``-prefixed modules, so this is never registered as a node.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("comfyui_timesaver.super_prompt_rt")
LOG_PREFIX = "[TS Super Prompt RT]"

#: Same preset file as the Qwen node — ONE source of truth for prompt rules.
#: They were tuned on Qwen, and Gemma is a different animal, so per-preset
#: generation parameters are re-derived in ``preset_generation_params`` rather
#: than taken as given.
PRESETS_FILENAME = "qwen_3_vl_presets.json"

DEFAULT_PRESET = "Prompts enhance"
CUSTOM_PRESET = "Your instruction"

#: Hard cap on the enhance route's text field — same reasoning as the Qwen
#: node: anything larger is misuse, and the context is 4096 tokens anyway.
ENHANCE_MAX_TEXT_LEN = 8192

#: Hard cap on a transcribe upload (bytes). Voice notes are small; a 200 MB
#: "recording" is someone testing the door handle.
TRANSCRIBE_MAX_UPLOAD = 64 * 1024 * 1024


def presets_path() -> Path:
    """``nodes/qwen_3_vl_presets.json`` — two levels up from this file."""
    return Path(__file__).resolve().parents[2] / PRESETS_FILENAME


def load_presets() -> tuple[dict[str, Any], list[str]]:
    path = presets_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a broken file must not kill the node
        logger.error("%s Could not read %s: %s", LOG_PREFIX, path.name, exc)
        return {}, []
    if not isinstance(data, dict):
        logger.error("%s %s is not a JSON object.", LOG_PREFIX, path.name)
        return {}, []
    return data, [key for key in data if isinstance(key, str) and key]


def preset_options() -> list[str]:
    """Preset names for the combo, minus the one that needs a widget we lack.

    ``CUSTOM_PRESET`` is omitted for the same reason as in the Qwen node: there
    is no field in which to type a custom system prompt, so offering it would
    silently fall back to the default and puzzle whoever picked it.
    """
    _presets, keys = load_presets()
    options = [key for key in keys if key != CUSTOM_PRESET]
    return options or [DEFAULT_PRESET]


def default_preset(options: list[str]) -> str:
    return DEFAULT_PRESET if DEFAULT_PRESET in options else (options[0] if options else DEFAULT_PRESET)


def resolve_preset(name: str) -> tuple[str, str, dict[str, Any]]:
    """``(resolved_name, system_prompt, gen_params)``.

    An unknown name falls back to the default preset without raising — a
    workflow saved before a preset was renamed must keep running. The rename
    table lives in the Qwen node's helpers and is reused here so both nodes
    resolve an old name identically.
    """
    from ..super_prompt._helpers import resolve_preset_alias

    presets, _keys = load_presets()
    resolved = resolve_preset_alias(name)
    if resolved not in presets:
        if resolved:
            logger.warning("%s Unknown preset %r, using %r.", LOG_PREFIX, resolved, DEFAULT_PRESET)
        resolved = DEFAULT_PRESET
    entry = presets.get(resolved) or {}
    return resolved, str(entry.get("system_prompt", "")), dict(entry.get("gen_params") or {})


def preset_generation_params(name: str, gen_params: dict[str, Any]) -> dict[str, Any]:
    """Sampling settings for Gemma, derived from the preset's Qwen settings.

    ⚠️ The numbers in the preset file were measured against Qwen and are kept
    as the starting point — temperature and top_p carry over, because they mean
    the same thing to any sampler. The token ceiling does NOT carry over
    unchanged: it is clamped to what is left of a 4096-token window, and that
    clamp is the whole reason this function exists rather than a dict copy.
    """
    from .._litert_engine import CONTEXT_TOKENS

    temperature = float(gen_params.get("temperature", 0.6) or 0.6)
    top_p = float(gen_params.get("top_p", 0.9) or 0.9)
    top_k = int(gen_params.get("top_k", 64) or 64)
    max_new = int(gen_params.get("max_new_tokens", 512) or 512)
    # Never promise more answer than the window can hold, whatever the file says.
    max_new = max(64, min(max_new, CONTEXT_TOKENS // 2))
    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new,
    }


# ---------------------------------------------------------------------------
# Attached images
# ---------------------------------------------------------------------------
def resolve_annotated_path(annotated: str) -> Path | None:
    """Turn ComfyUI's ``"subfolder/name.png [input]"`` into a real path.

    Reuses the Qwen node's resolver so both nodes accept exactly the same
    strings, including the traversal guard that comes with it.
    """
    text = str(annotated or "").strip()
    if not text:
        return None

    import folder_paths

    name, kind = text, "input"
    if text.endswith("]") and " [" in text:
        name, _, tail = text.rpartition(" [")
        kind = tail[:-1].strip() or "input"

    roots = {
        "input": folder_paths.get_input_directory(),
        "output": folder_paths.get_output_directory(),
        "temp": folder_paths.get_temp_directory(),
    }
    root = Path(roots.get(kind, roots["input"])).resolve()
    candidate = (root / name).resolve()
    # ⚠️ The name comes from a widget, i.e. from a workflow file: it must not be
    # able to point outside the directory it claims to live in.
    if root not in candidate.parents and candidate != root:
        logger.warning("%s Rejected a path outside %s: %r", LOG_PREFIX, kind, text)
        return None
    return candidate if candidate.is_file() else None


# ---------------------------------------------------------------------------
# Progress events (same channel names as the Qwen node's, with an rt suffix)
# ---------------------------------------------------------------------------
def _prompt_server() -> Any:
    try:
        import server

        return getattr(server.PromptServer, "instance", None)
    except Exception:  # noqa: BLE001 - no server in tests
        return None


def send_event(event: str, payload: dict[str, Any]) -> None:
    instance = _prompt_server()
    if instance is None:
        return
    try:
        instance.send_sync(event, payload)
    except Exception as exc:  # noqa: BLE001 - a dropped event is not fatal
        logger.debug("%s Could not send %s: %s", LOG_PREFIX, event, exc)


#: ⚠️ Имя события — КОНТРАКТ с фронтендом: он подписывается на
#: `${AI_EVENT_PREFIX}.progress`, то есть ровно на эту строку с суффиксом.
#: Первая версия слала `ts-super-prompt-rt-progress` через дефисы, никто её не
#: слушал, и панель стадий прыгала с «Prepare» сразу на готовый промпт.
EVENT_PREFIX = "ts_super_prompt_rt"


def send_progress(operation_id: str | None, text: str, percent: float | None = None) -> None:
    """Стадия работы — в панель.

    ⚠️ Текст не произвольный: фронтенд определяет стадию РЕГУЛЯРКОЙ по нему
    (prepare / download / load / generate). Формулировки здесь подобраны под
    эти правила, поэтому переписывать их «покрасивее» нельзя, не поправив
    `BUSY_STEPS` в js.
    """
    if not operation_id:
        return
    payload: dict[str, Any] = {"operation_id": operation_id, "text": text}
    if percent is not None:
        payload["percent"] = float(percent)
    send_event(f"{EVENT_PREFIX}.progress", payload)


def send_done(operation_id: str | None, text: str = "") -> None:
    if not operation_id:
        return
    send_event(f"{EVENT_PREFIX}.done", {"operation_id": operation_id, "text": text})


def send_error(operation_id: str | None, message: str) -> None:
    if not operation_id:
        return
    send_event(f"{EVENT_PREFIX}.error", {"operation_id": operation_id, "error": message})
