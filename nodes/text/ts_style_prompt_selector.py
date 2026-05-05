import json
import logging
from pathlib import Path

import folder_paths
from aiohttp import web

try:
    from server import PromptServer
except Exception:
    PromptServer = None


logger = logging.getLogger("TimesaverVFX_Pack")
_NO_STYLE_OPTION = "None"
_LOG_PREFIX = "[TS Style Prompt Selector]"


if PromptServer is None:
    logger.warning("%s PromptServer unavailable. API routes will be disabled.", _LOG_PREFIX)


def _register_get(path):
    def decorator(func):
        if PromptServer is None:
            return func
        try:
            PromptServer.instance.routes.get(path)(func)
        except Exception as exc:
            logger.warning("%s Failed to register route '%s': %s", _LOG_PREFIX, path, exc)
        return func

    return decorator


def _find_pack_root():
    try:
        for base in folder_paths.get_folder_paths("custom_nodes"):
            candidate = Path(base) / "comfyui-timesaver"
            if candidate.is_dir():
                return candidate
    except Exception as exc:
        logger.warning("%s Failed to resolve pack root: %s", _LOG_PREFIX, exc)
    return Path(__file__).resolve().parent


def _styles_dir():
    pack_root = _find_pack_root()
    preferred = Path(pack_root) / "nodes" / "styles"
    if preferred.is_dir():
        return preferred
    return Path(pack_root) / "styles"


def _styles_json_path():
    return _styles_dir() / "styles.json"


def _safe_join(base_dir, rel_path):
    if not rel_path:
        return None

    rel_path = str(rel_path).replace("\\", "/")
    if rel_path.startswith("/") or ":" in rel_path:
        return None

    base_path = Path(base_dir).resolve()
    target_path = (base_path / Path(rel_path)).resolve()
    try:
        target_path.relative_to(base_path)
    except ValueError:
        return None
    return str(target_path)


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _load_styles():
    styles_path = _styles_json_path()
    if not styles_path.exists():
        logger.error("%s Styles file not found: %s", _LOG_PREFIX, styles_path)
        return []
    try:
        with styles_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        styles = data.get("styles", [])
        if not isinstance(styles, list):
            logger.error("%s Invalid styles format in %s", _LOG_PREFIX, styles_path)
            return []

        normalized = []
        for style in styles:
            if not isinstance(style, dict):
                continue

            item = {
                "id": _as_text(style.get("id")).strip(),
                "name": _as_text(style.get("name")).strip(),
                "prompt": _as_text(style.get("prompt")),
                "description": _as_text(style.get("description")),
                "preview": _as_text(style.get("preview")).strip(),
            }
            if not item["id"] and not item["name"]:
                continue
            normalized.append(item)
        return normalized
    except Exception as exc:
        logger.error("%s Failed to load styles: %s", _LOG_PREFIX, exc)
        return []


@_register_get("/ts_styles")
async def ts_styles_list(request):
    return web.json_response({"styles": _load_styles()})


@_register_get("/ts_styles/preview")
async def ts_styles_preview(request):
    rel_path = request.query.get("path", "")
    file_path = _safe_join(_styles_dir(), rel_path)
    if not file_path or not Path(file_path).is_file():
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
    CATEGORY = "TS/Text"

    @classmethod
    def VALIDATE_INPUTS(cls, style_id):
        if style_id is None:
            return True
        if isinstance(style_id, str):
            return True
        return "style_id must be STRING."

    @classmethod
    def IS_CHANGED(cls, style_id):
        return _as_text(style_id).strip()

    def get_prompt(self, style_id):
        styles = _load_styles()
        selected_id = _as_text(style_id).strip()
        if not selected_id or selected_id == _NO_STYLE_OPTION:
            return (" ",)

        prompt = ""
        if styles and selected_id:
            for style in styles:
                style_id_value = _as_text(style.get("id")).strip()
                style_name_value = _as_text(style.get("name")).strip()
                if selected_id in {style_id_value, style_name_value}:
                    prompt = style.get("prompt", "") or ""
                    break

        if not prompt:
            logger.warning("%s Style not found: %s", _LOG_PREFIX, selected_id)
        return (prompt or "",)


NODE_CLASS_MAPPINGS = {
    "TS_StylePromptSelector": TS_StylePromptSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_StylePromptSelector": "TS Style Prompt Selector",
}
