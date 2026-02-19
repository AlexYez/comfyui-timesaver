import json
import logging
import os

import folder_paths
from aiohttp import web
from server import PromptServer


logger = logging.getLogger("TimesaverVFX_Pack")
_NO_STYLE_OPTION = "None"


def _find_pack_root():
    try:
        for base in folder_paths.get_folder_paths("custom_nodes"):
            candidate = os.path.join(base, "comfyui-timesaver")
            if os.path.isdir(candidate):
                return candidate
    except Exception as exc:
        logger.warning("[TS Style Prompt Selector] Failed to resolve pack root: %s", exc)
    return os.path.dirname(__file__)


def _styles_dir():
    return os.path.join(_find_pack_root(), "styles")


def _styles_json_path():
    return os.path.join(_styles_dir(), "styles.json")


def _safe_join(base_dir, rel_path):
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


def _load_styles():
    styles_path = _styles_json_path()
    if not os.path.exists(styles_path):
        logger.error("[TS Style Prompt Selector] Styles file not found: %s", styles_path)
        return []
    try:
        with open(styles_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        styles = data.get("styles", [])
        if not isinstance(styles, list):
            logger.error("[TS Style Prompt Selector] Invalid styles format in %s", styles_path)
            return []
        return styles
    except Exception as exc:
        logger.error("[TS Style Prompt Selector] Failed to load styles: %s", exc)
        return []


@PromptServer.instance.routes.get("/ts_styles")
async def ts_styles_list(request):
    return web.json_response({"styles": _load_styles()})


@PromptServer.instance.routes.get("/ts_styles/preview")
async def ts_styles_preview(request):
    rel_path = request.query.get("path", "")
    file_path = _safe_join(_styles_dir(), rel_path)
    if not file_path or not os.path.isfile(file_path):
        return web.Response(status=404)
    return web.FileResponse(file_path)


class TS_StylePromptSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_prompt"
    CATEGORY = "TS/Prompt"

    def get_prompt(self, style_id):
        styles = _load_styles()
        selected_id = (style_id or "").strip()
        if not selected_id or selected_id == _NO_STYLE_OPTION:
            return (" ",)
        prompt = ""
        if styles and selected_id:
            for style in styles:
                if style.get("id") == selected_id or style.get("name") == selected_id:
                    prompt = style.get("prompt", "") or ""
                    break
        if not prompt:
            logger.warning("[TS Style Prompt Selector] Style not found: %s", selected_id)
        return (prompt or "",)


NODE_CLASS_MAPPINGS = {
    "TS_StylePromptSelector": TS_StylePromptSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_StylePromptSelector": "TS Style Prompt Selector",
}
