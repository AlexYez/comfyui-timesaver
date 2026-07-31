import json
import logging
from pathlib import Path

import folder_paths
from aiohttp import web
from comfy_api.v0_0_2 import IO

from .._shared import make_route_registrars, resolve_prompt_server

logger = logging.getLogger("TimesaverVFX_Pack")
_NO_STYLE_OPTION = "None"
_LOG_PREFIX = "[TS Style Prompt Selector]"


def _warn_route(message: str) -> None:
    logger.warning("%s %s", _LOG_PREFIX, message)


_PROMPT_SERVER = resolve_prompt_server(_warn_route)
# Only GET routes are used here; the POST registrar is intentionally unused.
_register_get, _ = make_route_registrars(_PROMPT_SERVER, _warn_route)


def _find_pack_root():
    # This file lives at ``<pack>/nodes/text/ts_style_prompt_selector.py``, so
    # the pack root is parents[2] — robust regardless of what the clone folder
    # is named (a hardcoded "comfyui-timesaver" scan silently missed a renamed
    # clone, and the old ``.parent`` fallback pointed at nodes/text/, where the
    # styles folder does not exist).
    pack_root = Path(__file__).resolve().parents[2]
    if pack_root.is_dir():
        return pack_root
    # Legacy fallback: scan ComfyUI's registered custom_nodes for the pack.
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
                # Russian name and the category pair drive the frontend's
                # locale-aware labels and the category filter.
                "name_ru": _as_text(style.get("name_ru")).strip(),
                "category": _as_text(style.get("category")).strip(),
                "category_ru": _as_text(style.get("category_ru")).strip(),
                "prompt": _as_text(style.get("prompt")),
                # Descriptions follow the same convention as the name/category
                # pair: the bare key is English, the `_ru` suffix is Russian.
                "description": _as_text(style.get("description")),
                "description_ru": _as_text(style.get("description_ru")),
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


class TS_StylePromptSelector(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_StylePromptSelector",
            display_name="TS Style Prompt Selector",
            category="TS/Text",
            essentials_category="Text",
            description="Pick an art style from a browsable library of 157 previews and output its prompt modifier.",
            inputs=[
                IO.String.Input(
                    "style_id",
                    default="",
                    tooltip="Id or name of the style to load. The prompt text itself comes from styles.json.",
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="prompt",
                    tooltip="Prompt text of the selected style, or a blank space when no style is chosen.",
                )
            ],
        )

    @classmethod
    def validate_inputs(cls, style_id):
        if style_id is None:
            return True
        if isinstance(style_id, str):
            return True
        return "style_id must be STRING."

    @classmethod
    def fingerprint_inputs(cls, style_id):
        # The prompt text lives in styles.json, not in the inputs, so the id alone
        # adds nothing ComfyUI does not already hash — editing a style's prompt
        # while keeping its id would keep serving the cached text. Track the file's
        # mtime too, mirroring TS_Qwen3_VL_V3's preset handling.
        styles_path = _styles_json_path()
        try:
            mtime = styles_path.stat().st_mtime if styles_path.exists() else None
        except OSError:
            mtime = None
        return (_as_text(style_id).strip(), mtime)

    @classmethod
    def execute(cls, style_id) -> IO.NodeOutput:
        styles = _load_styles()
        selected_id = _as_text(style_id).strip()
        if not selected_id or selected_id == _NO_STYLE_OPTION:
            return IO.NodeOutput(" ")

        prompt = ""
        found = False
        if styles and selected_id:
            for style in styles:
                style_id_value = _as_text(style.get("id")).strip()
                style_name_value = _as_text(style.get("name")).strip()
                if selected_id in {style_id_value, style_name_value}:
                    prompt = style.get("prompt", "") or ""
                    found = True
                    break

        if not found:
            logger.warning("%s Style not found: %s", _LOG_PREFIX, selected_id)
        elif not prompt:
            # The style exists but carries no prompt text — a library problem,
            # not a missing selection. Saying "not found" sent people looking
            # for a broken workflow instead of a broken entry.
            logger.warning("%s Style '%s' has an empty prompt.", _LOG_PREFIX, selected_id)
        return IO.NodeOutput(prompt or "")


NODE_CLASS_MAPPINGS = {
    "TS_StylePromptSelector": TS_StylePromptSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_StylePromptSelector": "TS Style Prompt Selector",
}
