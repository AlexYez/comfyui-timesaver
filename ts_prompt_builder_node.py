import hashlib
import json
import logging
import os
import random
import tempfile

import folder_paths
from aiohttp import web
from server import PromptServer


ts_logger = logging.getLogger("TimesaverVFX_Pack")
TS_PROMPTS_DIRNAME = "prompts"
TS_PROMPT_BUILDER_CONFIG_FILENAME = "ts-prompt-builder-config.json"


def ts_find_pack_root():
    try:
        for base in folder_paths.get_folder_paths("custom_nodes"):
            candidate = os.path.join(base, "comfyui-timesaver")
            if os.path.isdir(candidate):
                return candidate
    except Exception as exc:
        ts_logger.warning("[TS Prompt Builder] Failed to resolve pack root: %s", exc)
    return os.path.dirname(__file__)


def ts_prompts_dir():
    return os.path.join(ts_find_pack_root(), TS_PROMPTS_DIRNAME)


def ts_prompt_builder_config_path():
    return os.path.join(ts_prompts_dir(), TS_PROMPT_BUILDER_CONFIG_FILENAME)


def ts_safe_join(base_dir, rel_path):
    if not rel_path:
        return None
    rel_path = rel_path.replace("\\", "/")
    if rel_path.startswith("/") or ":" in rel_path:
        return None
    normalized = os.path.normpath(rel_path)
    abs_base = os.path.abspath(base_dir)
    abs_target = os.path.abspath(os.path.join(base_dir, normalized))
    if not abs_target.startswith(abs_base + os.sep):
        return None
    return abs_target


def ts_list_prompt_files():
    prompts_dir = ts_prompts_dir()
    if not os.path.isdir(prompts_dir):
        return []
    files = []
    for name in os.listdir(prompts_dir):
        if not name.lower().endswith(".txt"):
            continue
        full_path = os.path.join(prompts_dir, name)
        if os.path.isfile(full_path):
            files.append(name)
    files.sort(key=str.lower)
    return files


def ts_read_prompt_lines(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            lines = []
            for raw in handle:
                cleaned = raw.strip().lstrip("\ufeff")
                if cleaned:
                    lines.append(cleaned)
        return lines
    except Exception as exc:
        ts_logger.warning("[TS Prompt Builder] Failed to read prompts: %s", exc)
        return []


def ts_parse_config_data(data):
    if isinstance(data, dict):
        if isinstance(data.get("blocks"), list):
            data = data["blocks"]
        elif isinstance(data.get("order"), list):
            data = data["order"]
        else:
            return []
    if not isinstance(data, list):
        return []
    result = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("file") or entry.get("name") or "").strip()
        if not name:
            continue
        enabled = entry.get("enabled", True)
        result.append({"file": name, "enabled": bool(enabled)})
    return result


def ts_parse_config(config_json):
    if not config_json:
        return []
    try:
        data = json.loads(config_json)
    except Exception:
        return []
    return ts_parse_config_data(data)


def ts_block_seed(seed_value, file_name):
    payload = f"{seed_value}:{file_name}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def ts_merge_config_with_files(config_items, files):
    files_set = set(files)
    merged = []
    seen = set()
    for entry in config_items:
        file_name = entry.get("file")
        if not file_name or file_name not in files_set or file_name in seen:
            continue
        merged.append({"file": file_name, "enabled": bool(entry.get("enabled", True))})
        seen.add(file_name)
    for file_name in files:
        if file_name in seen:
            continue
        merged.append({"file": file_name, "enabled": True})
    return merged


def ts_load_prompt_builder_config():
    path = ts_prompt_builder_config_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return ts_parse_config_data(data)
    except Exception as exc:
        ts_logger.warning("[TS Prompt Builder] Failed to read config: %s", exc)
        return []


def ts_write_prompt_builder_config(config_items):
    prompts_dir = ts_prompts_dir()
    os.makedirs(prompts_dir, exist_ok=True)
    payload = {"blocks": config_items}
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=prompts_dir,
            delete=False,
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = handle.name
        os.replace(temp_path, ts_prompt_builder_config_path())
        return True
    except Exception as exc:
        ts_logger.warning("[TS Prompt Builder] Failed to write config: %s", exc)
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False


def ts_get_config_state(files, override_items=None):
    if override_items is not None:
        merged = ts_merge_config_with_files(override_items, files)
        ts_write_prompt_builder_config(merged)
        return merged

    stored_items = ts_load_prompt_builder_config()
    if stored_items is None:
        merged = ts_merge_config_with_files([], files)
        ts_write_prompt_builder_config(merged)
        return merged

    merged = ts_merge_config_with_files(stored_items, files)
    if merged != stored_items:
        ts_write_prompt_builder_config(merged)
    return merged


@PromptServer.instance.routes.get("/ts_prompt_builder/files")
async def ts_prompt_builder_files(request):
    return web.json_response({"files": ts_list_prompt_files()})


@PromptServer.instance.routes.get("/ts_prompt_builder/state")
async def ts_prompt_builder_state(request):
    files = ts_list_prompt_files()
    blocks = ts_get_config_state(files)
    return web.json_response({"files": files, "blocks": blocks})


@PromptServer.instance.routes.post("/ts_prompt_builder/config")
async def ts_prompt_builder_config(request):
    try:
        payload = await request.json()
    except Exception:
        return web.Response(status=400)
    files = ts_list_prompt_files()
    items = ts_parse_config_data(payload)
    blocks = ts_get_config_state(files, override_items=items)
    return web.json_response({"files": files, "blocks": blocks})


class TS_PromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "0 = auto-random each run, >0 = deterministic per seed",
                    },
                ),
                "config_json": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build_prompt"
    CATEGORY = "TS/Prompt"

    @classmethod
    def IS_CHANGED(cls, seed, config_json):
        try:
            seed_value = int(seed)
        except Exception:
            seed_value = 0
        if seed_value == 0:
            return (config_json, random.SystemRandom().random())
        return (config_json, seed_value)

    def build_prompt(self, seed, config_json):
        try:
            seed_value = int(seed)
        except Exception:
            seed_value = 0

        files = ts_list_prompt_files()
        if not files:
            ts_logger.warning("[TS Prompt Builder] No prompt files found.")
            return ("",)

        config_items = ts_parse_config(config_json)
        if config_items:
            ordered = ts_get_config_state(files, override_items=config_items)
        else:
            ordered = ts_get_config_state(files)

        parts = []
        prompts_dir = ts_prompts_dir()
        for item in ordered:
            if not item.get("enabled", True):
                continue
            file_name = item.get("file")
            file_path = ts_safe_join(prompts_dir, file_name)
            if not file_path or not os.path.isfile(file_path):
                ts_logger.warning("[TS Prompt Builder] Prompt file missing: %s", file_name)
                continue
            lines = ts_read_prompt_lines(file_path)
            if not lines:
                continue
            if seed_value == 0:
                parts.append(random.SystemRandom().choice(lines))
            else:
                block_seed = ts_block_seed(seed_value, file_name)
                parts.append(random.Random(block_seed).choice(lines))

        return (", ".join(parts),)


NODE_CLASS_MAPPINGS = {
    "TS_PromptBuilder": TS_PromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_PromptBuilder": "TS Prompt Builder",
}
