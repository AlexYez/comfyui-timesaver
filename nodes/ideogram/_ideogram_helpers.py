"""Core caption-building helpers for the TS Ideogram Designer node.

This module owns the *correctness-critical* logic and is intentionally
import-light so it can be unit-tested standalone (``python _ideogram_helpers.py``).

------------------------------------------------------------------------------
design_json contract (written by the JS editor, read here)
------------------------------------------------------------------------------
{
  "version": 1,
  "aspect_ratio": "16x9",                 # separate output, NOT a caption field
  "high_level_description": "string",
  "background": "string",
  "style": {                              # resolved style_description fields
    "preset_id": "...",
    "aesthetics": "...", "lighting": "...",
    "medium": "graphic_design|photograph|illustration|3d_render|painting|digital_painting",
    "photo": "",                          # used iff medium == photograph
    "art_style": "...",                   # used otherwise
    "color_palette": ["#RRGGBB", ...]     # image-level, <=16
  },
  "blocks": [
    {
      "id": "...", "type": "text" | "obj",
      "rect": {"x":0.04,"y":0.12,"w":0.5,"h":0.3},   # fractions 0..1, top-left origin
      # text-only:
      "text": "ЛИТЕРАЛ\nстрока2",
      "font_preset_id": "grotesque_black",
      "weight": "Thin|Regular|Bold|Heavy",
      "case": "As-typed|UPPERCASE|Title",
      "prominence": "Caption|Body|Headline|Hero",
      "legibility": {"outline":true,"high_contrast":true,"solid_block":false},
      "visual_only": false,
      "color": "#FFE000",
      "desc_override": "",
      # obj-only:
      "desc": "...",
      # both:
      "color_palette": ["#RRGGBB", ...]    # element-level, <=5
    }
  ]
}

Verified Ideogram 4 invariants honored here (adversarially confirmed against
``ideogram-oss/ideogram4/docs/prompting.md`` and the developer.ideogram.ai
OpenAPI schema):
 - bbox = [y_min, x_min, y_max, x_max], integers 0-1000, origin top-left (y first!).
 - element key order: type, bbox, text, desc, color_palette.
 - color_palette: uppercase #RRGGBB only; image-level <=16, element-level <=5.
 - high_level_description always emitted; photo XOR art_style by medium.
 - serialize compact with ensure_ascii=False.
"""

import json
import logging
import os
import re

ts_logger = logging.getLogger("comfyui_timesaver.ts_ideogram")
LOG_PREFIX = "[TS Ideogram Designer]"

FONTS_FILENAME = "ideogram_fonts.json"
STYLES_FILENAME = "ideogram_styles.json"
LAST_DESIGN_FILENAME = "ideogram_last_design.json"

DEFAULT_ASPECT_RATIO = "16x9"
# v4 aspect-ratio enum (API uses the 'x' separator).
ASPECT_RATIOS = [
    "1x4", "1x3", "1x2", "9x16", "10x16", "2x3", "3x4", "4x5",
    "1x1", "5x4", "4x3", "3x2", "16x10", "16x9", "2x1", "3x1", "4x1",
]
PHOTO_MEDIUM = "photograph"
IMAGE_PALETTE_CAP = 16
ELEMENT_PALETTE_CAP = 5

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")

# node_id -> input-dir filename of the last graph IMAGE saved for preview.
_GRAPH_REF_BY_NODE: dict[str, str] = {}
_ROUTES_REGISTERED = False


# --------------------------------------------------------------------------- #
# Paths / preset loading
# --------------------------------------------------------------------------- #
def _module_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _preset_path(filename: str) -> str:
    return os.path.join(_module_dir(), filename)


def _load_json_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001 - best-effort, never crash the node
        ts_logger.warning("%s Failed to read %s: %s", LOG_PREFIX, path, exc)
        return None


def load_fonts() -> list[dict]:
    data = _load_json_file(_preset_path(FONTS_FILENAME))
    if isinstance(data, dict) and isinstance(data.get("fonts"), list):
        return [f for f in data["fonts"] if isinstance(f, dict)]
    return []


def load_styles() -> list[dict]:
    data = _load_json_file(_preset_path(STYLES_FILENAME))
    if isinstance(data, dict) and isinstance(data.get("styles"), list):
        return [s for s in data["styles"] if isinstance(s, dict)]
    return []


def _fonts_by_id() -> dict[str, dict]:
    return {str(f.get("id")): f for f in load_fonts() if f.get("id")}


# --------------------------------------------------------------------------- #
# Small pure utilities
# --------------------------------------------------------------------------- #
def has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text or ""))


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def norm_hex(value) -> str | None:
    """Uppercase + validate a single #RRGGBB color. None if invalid."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not _HEX_RE.match(value):
        return None
    return value.upper()


def clean_palette(values, cap: int) -> list[str]:
    """Filter to valid uppercase #RRGGBB, dedupe (order-preserving), cap length."""
    out: list[str] = []
    if not isinstance(values, (list, tuple)):
        return out
    for raw in values:
        hexv = norm_hex(raw)
        if hexv and hexv not in out:
            out.append(hexv)
        if len(out) >= cap:
            break
    return out


def frac_to_bbox(x: float, y: float, w: float, h: float) -> list[int]:
    """Editor rect (fractions 0..1, top-left origin) -> Ideogram bbox.

    Returns [y_min, x_min, y_max, x_max] as integers in 0-1000 (y FIRST).
    This is the single place the (x,y) -> (y,x) swap happens.
    """
    x_min = _clamp(round(x * 1000), 0, 1000)
    y_min = _clamp(round(y * 1000), 0, 1000)
    x_max = _clamp(round((x + w) * 1000), 0, 1000)
    y_max = _clamp(round((y + h) * 1000), 0, 1000)
    if x_max <= x_min:
        x_max = min(1000, x_min + 1)
        if x_max <= x_min:  # x_min was already at the 1000 ceiling
            x_min = max(0, x_max - 1)
    if y_max <= y_min:
        y_max = min(1000, y_min + 1)
        if y_max <= y_min:
            y_min = max(0, y_max - 1)
    return [y_min, x_min, y_max, x_max]


def _bbox_from_block(block: dict) -> list[int] | None:
    rect = block.get("rect")
    if not isinstance(rect, dict):
        return None
    try:
        x = float(rect.get("x"))
        y = float(rect.get("y"))
        w = float(rect.get("w"))
        h = float(rect.get("h"))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return frac_to_bbox(x, y, w, h)


def _apply_case(text: str, case: str) -> str:
    if case == "UPPERCASE":
        return text.upper()
    if case == "Title":
        # Per-line title casing keeps explicit \n line breaks intact.
        return "\n".join(line.title() for line in text.split("\n"))
    return text


# --------------------------------------------------------------------------- #
# desc composition for text blocks
# --------------------------------------------------------------------------- #
_WEIGHT_PHRASE = {"Thin": "thin light weight", "Bold": "bold weight", "Heavy": "heavy black weight"}
_CASE_PHRASE = {"UPPERCASE": "all uppercase", "Title": "title case"}
_PROMINENCE_PHRASE = {
    "Caption": "small caption text",
    "Body": "medium body text",
    "Headline": "large prominent headline",
    "Hero": "huge dominant hero headline",
}


def compose_text_desc(block: dict, fonts_by_id: dict[str, dict] | None = None) -> str:
    """Build a text element's ``desc`` from ordered, non-empty slots.

    Mirror of the JS ``composeTextDesc`` in _ideogram_shared.js — keep in sync.
    Order: preset snippet (load-bearing, first) -> weight -> case -> prominence
    -> color -> legibility -> Cyrillic hint -> user override (verbatim, last).
    """
    fonts_by_id = fonts_by_id if fonts_by_id is not None else _fonts_by_id()
    slots: list[str] = []

    preset = fonts_by_id.get(str(block.get("font_preset_id") or ""))
    snippet = (preset or {}).get("desc_snippet") if preset else None
    slots.append(str(snippet).strip() if snippet else "bold clean sans-serif")

    weight = _WEIGHT_PHRASE.get(str(block.get("weight") or ""))
    if weight:
        slots.append(weight)

    case_phrase = _CASE_PHRASE.get(str(block.get("case") or ""))
    if case_phrase:
        slots.append(case_phrase)

    prominence = _PROMINENCE_PHRASE.get(str(block.get("prominence") or ""))
    if prominence:
        slots.append(prominence)

    color = norm_hex(block.get("color"))
    if color:
        slots.append(f"{color} colored letters")

    legibility = block.get("legibility")
    if isinstance(legibility, dict):
        if legibility.get("outline"):
            slots.append("with a thin dark outline")
        if legibility.get("high_contrast"):
            slots.append("high contrast")
        if legibility.get("solid_block"):
            slots.append("on a solid color block behind the text")
    slots.append("crisp clean edges, readable at small thumbnail size")

    if has_cyrillic(str(block.get("text") or "")):
        slots.append("Cyrillic script, Russian text")

    override = str(block.get("desc_override") or "").strip()
    if override:
        slots.append(override)

    return ", ".join(s for s in slots if s)


_VISUAL_ONLY_DESC = (
    "clean empty solid color banner reserved for text, flat fill, no lettering, "
    "leave clear for manual text overlay in post-production"
)


# --------------------------------------------------------------------------- #
# Element + caption construction
# --------------------------------------------------------------------------- #
def _build_element(block: dict, fonts_by_id: dict[str, dict]) -> dict | None:
    if not isinstance(block, dict):
        return None
    btype = str(block.get("type") or "obj")
    bbox = _bbox_from_block(block)
    palette = clean_palette(block.get("color_palette"), ELEMENT_PALETTE_CAP)

    is_text = btype == "text"
    visual_only = bool(block.get("visual_only"))

    if is_text and not visual_only:
        text = str(block.get("text") or "")
        text = _apply_case(text, str(block.get("case") or ""))
        if not text.strip():
            return None
        # A text block's primary color reinforces the element palette.
        color = norm_hex(block.get("color"))
        if color and color not in palette and len(palette) < ELEMENT_PALETTE_CAP:
            palette = [color] + palette
            palette = palette[:ELEMENT_PALETTE_CAP]
        element: dict = {"type": "text"}
        if bbox is not None:
            element["bbox"] = bbox
        element["text"] = text
        element["desc"] = compose_text_desc(block, fonts_by_id)
        if palette:
            element["color_palette"] = palette
        return element

    # obj element (also the emit path for visual_only text blocks)
    if is_text and visual_only:
        desc = _VISUAL_ONLY_DESC
    else:
        desc = str(block.get("desc") or "").strip()
    if not desc:
        return None
    element = {"type": "obj"}
    if bbox is not None:
        element["bbox"] = bbox
    element["desc"] = desc
    if palette:
        element["color_palette"] = palette
    return element


def _build_style_description(style: dict) -> dict | None:
    if not isinstance(style, dict):
        return None
    out: dict = {}
    aesthetics = str(style.get("aesthetics") or "").strip()
    lighting = str(style.get("lighting") or "").strip()
    medium = str(style.get("medium") or "").strip()
    photo = str(style.get("photo") or "").strip()
    art_style = str(style.get("art_style") or "").strip()
    palette = clean_palette(style.get("color_palette"), IMAGE_PALETTE_CAP)

    if aesthetics:
        out["aesthetics"] = aesthetics
    if lighting:
        out["lighting"] = lighting
    # photo XOR art_style, decided by medium.
    if medium == PHOTO_MEDIUM:
        if photo:
            out["photo"] = photo
    else:
        if art_style:
            out["art_style"] = art_style
    if medium:
        out["medium"] = medium
    if palette:
        out["color_palette"] = palette  # last key
    return out or None


def build_caption(design_json: str) -> tuple[str, str]:
    """Parse the editor state and return (compact_json_caption, aspect_ratio)."""
    aspect = DEFAULT_ASPECT_RATIO
    try:
        design = json.loads(design_json) if design_json else {}
    except Exception as exc:  # noqa: BLE001
        ts_logger.warning("%s Invalid design_json: %s", LOG_PREFIX, exc)
        return "", aspect
    if not isinstance(design, dict):
        return "", aspect

    raw_aspect = str(design.get("aspect_ratio") or "").strip()
    if raw_aspect in ASPECT_RATIOS:
        aspect = raw_aspect

    fonts_by_id = _fonts_by_id()

    elements: list[dict] = []
    for block in design.get("blocks", []) or []:
        element = _build_element(block, fonts_by_id)
        if element is not None:
            elements.append(element)

    background = str(design.get("background") or "").strip()

    # Nothing meaningful designed yet -> emit empty string (downstream no-op).
    if not elements and not background and not str(design.get("high_level_description") or "").strip():
        return "", aspect

    caption: dict = {"high_level_description": str(design.get("high_level_description") or "").strip()}
    style_description = _build_style_description(design.get("style"))
    if style_description:
        caption["style_description"] = style_description
    caption["compositional_deconstruction"] = {
        "background": background,
        "elements": elements,
    }

    serialized = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
    return serialized, aspect


# --------------------------------------------------------------------------- #
# Graph IMAGE -> input dir (best-effort preview underlay)
# --------------------------------------------------------------------------- #
def _sanitize_node_id(node_id) -> str:
    return re.sub(r"[^0-9A-Za-z_-]", "_", str(node_id))[:64] or "node"


def save_graph_reference(image_tensor, node_id) -> str | None:
    """Save the first frame of a graph IMAGE tensor into the input dir so the
    editor can show it as a reference underlay. Best-effort; returns the
    filename (in the input dir) or None.
    """
    if image_tensor is None:
        return None
    try:
        import numpy as np  # noqa: PLC0415 - lazy, optional at import time
        from PIL import Image  # noqa: PLC0415
        import folder_paths  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        ts_logger.debug("%s Graph-ref save unavailable: %s", LOG_PREFIX, exc)
        return None
    try:
        tensor = image_tensor
        if hasattr(tensor, "ndim") and tensor.ndim == 4:
            tensor = tensor[0]
        array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        array = np.clip(array * 255.0, 0, 255).astype("uint8")
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        image = Image.fromarray(array)
        filename = f"ts_ideogram_ref_{_sanitize_node_id(node_id)}.png"
        out_path = os.path.join(folder_paths.get_input_directory(), filename)
        image.save(out_path)
        _GRAPH_REF_BY_NODE[str(node_id)] = filename
        return filename
    except Exception as exc:  # noqa: BLE001
        ts_logger.warning("%s Failed to save graph reference: %s", LOG_PREFIX, exc)
        return None


def get_graph_reference(node_id) -> str | None:
    filename = _GRAPH_REF_BY_NODE.get(str(node_id))
    if not filename:
        return None
    try:
        import folder_paths  # noqa: PLC0415
        if os.path.isfile(os.path.join(folder_paths.get_input_directory(), filename)):
            return filename
    except Exception:  # noqa: BLE001
        return filename
    return None


# --------------------------------------------------------------------------- #
# Config persistence (optional convenience)
# --------------------------------------------------------------------------- #
def save_last_design(design_json: str) -> bool:
    try:
        with open(_preset_path(LAST_DESIGN_FILENAME), "w", encoding="utf-8") as handle:
            handle.write(design_json or "")
        return True
    except Exception as exc:  # noqa: BLE001
        ts_logger.warning("%s Failed to persist last design: %s", LOG_PREFIX, exc)
        return False


def load_last_design() -> str:
    path = _preset_path(LAST_DESIGN_FILENAME)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #
def register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    try:
        from server import PromptServer  # noqa: PLC0415
        from aiohttp import web  # noqa: PLC0415
        instance = PromptServer.instance
        if instance is None:
            raise RuntimeError("PromptServer.instance is None")
        routes = instance.routes
    except Exception as exc:  # noqa: BLE001
        ts_logger.warning("%s API routes disabled: %s", LOG_PREFIX, exc)
        return

    _ROUTES_REGISTERED = True

    @routes.get("/ts_ideogram/presets")
    async def _presets(_request):
        return web.json_response({
            "styles": load_styles(),
            "fonts": load_fonts(),
            "aspect_ratios": ASPECT_RATIOS,
            "default_aspect_ratio": DEFAULT_ASPECT_RATIO,
        })

    @routes.get("/ts_ideogram/graph_ref")
    async def _graph_ref(request):
        node_id = request.query.get("node_id", "")
        filename = get_graph_reference(node_id)
        return web.json_response({"filename": filename} if filename else {})

    @routes.post("/ts_ideogram/config")
    async def _config(request):
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400)
        design_json = payload.get("design_json", "") if isinstance(payload, dict) else ""
        ok = save_last_design(design_json if isinstance(design_json, str) else "")
        return web.json_response({"ok": bool(ok)})

    @routes.post("/ts_ideogram/preview")
    async def _preview(request):
        """Server-authoritative caption preview for the editor's JSON panel."""
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400)
        design_json = payload.get("design_json", "") if isinstance(payload, dict) else ""
        caption, aspect = build_caption(design_json if isinstance(design_json, str) else "")
        return web.json_response({"json_prompt": caption, "aspect_ratio": aspect})


# --------------------------------------------------------------------------- #
# Standalone self-test:  python _ideogram_helpers.py
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    failures = 0

    def check(name, cond):
        nonlocal failures
        status = "ok" if cond else "FAIL"
        if not cond:
            failures += 1
        print(f"[{status}] {name}")

    # bbox y-first canonical fixture
    bbox = frac_to_bbox(0.642, 0.055, 0.295, 0.945)
    check(f"frac_to_bbox -> {bbox} == [55,642,1000,937]", bbox == [55, 642, 1000, 937])

    # clamp + min<max degenerate rect
    tiny = frac_to_bbox(0.999, 0.999, 0.0005, 0.0005)
    check(f"degenerate bbox keeps min<max ({tiny})", tiny[0] < tiny[2] and tiny[1] < tiny[3])

    # hex
    check("norm_hex lowercase->upper", norm_hex("#ffe000") == "#FFE000")
    check("norm_hex shorthand invalid", norm_hex("#fff") is None)
    check("clean_palette caps at 5", len(clean_palette([f"#{i:02X}0000" for i in range(10)], 5)) == 5)

    # cyrillic detection
    check("has_cyrillic russian", has_cyrillic("СРОЧНО") is True)
    check("has_cyrillic latin", has_cyrillic("URGENT") is False)

    # full caption build
    design = {
        "aspect_ratio": "16x9",
        "high_level_description": "test thumbnail",
        "background": "dark navy gradient",
        "style": {
            "medium": "graphic_design",
            "aesthetics": "bold punchy",
            "lighting": "bright key",
            "art_style": "flat vector",
            "photo": "should be dropped",
            "color_palette": ["#0b1020", "#ffd500", "bad"],
        },
        "blocks": [
            {
                "type": "text", "rect": {"x": 0.04, "y": 0.12, "w": 0.5, "h": 0.3},
                "text": "срочно", "font_preset_id": "grotesque_black",
                "weight": "Heavy", "case": "UPPERCASE", "prominence": "Hero",
                "legibility": {"outline": True, "high_contrast": True},
                "color": "#ffe000", "color_palette": ["#ffe000"],
            },
            {
                "type": "obj", "rect": {"x": 0.6, "y": 0.1, "w": 0.35, "h": 0.85},
                "desc": "shocked creator face on the right",
            },
            {
                "type": "text", "visual_only": True,
                "rect": {"x": 0.1, "y": 0.7, "w": 0.5, "h": 0.15},
                "text": "Длинный текст",
            },
        ],
    }
    caption_str, aspect = build_caption(json.dumps(design))
    caption = json.loads(caption_str)
    check("aspect parsed", aspect == "16x9")
    check("high_level_description present", "high_level_description" in caption)
    check(
        "top-level key order",
        list(caption.keys()) == ["high_level_description", "style_description", "compositional_deconstruction"],
    )
    sd = caption["style_description"]
    check("photo dropped for non-photograph medium", "photo" not in sd and "art_style" in sd)
    check("image palette uppercased + filtered", sd["color_palette"] == ["#0B1020", "#FFD500"])
    check("color_palette is last key in style_description", list(sd.keys())[-1] == "color_palette")
    els = caption["compositional_deconstruction"]["elements"]
    check("three elements built", len(els) == 3)
    t0 = els[0]
    check("text literal uppercased", t0["text"] == "СРОЧНО")
    check("text element key order", list(t0.keys()) == ["type", "bbox", "text", "desc", "color_palette"])
    check("bbox y-first in element", t0["bbox"] == frac_to_bbox(0.04, 0.12, 0.5, 0.3))
    check("cyrillic hint in desc", "Cyrillic script" in t0["desc"])
    check("desc snippet is first slot", t0["desc"].startswith("bold grotesque sans-serif"))
    check("visual_only text became obj placeholder", els[2]["type"] == "obj" and "text" not in els[2])
    check("ensure_ascii=False keeps Cyrillic literal", "СРОЧНО" in caption_str)

    print("-" * 40)
    print("ALL PASSED" if failures == 0 else f"{failures} FAILURE(S)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    _self_test()
