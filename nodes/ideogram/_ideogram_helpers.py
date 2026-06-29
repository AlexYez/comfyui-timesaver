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
      "weight": "Thin|Regular|Bold",
      "case": "As-typed|UPPERCASE|Title",            # size is derived from rect.h, not a field
      "legibility": {"outline":true,"solid_block":false},
      "visual_only": false,
      "color": "#FFE000",                            # text color
      "outline_color": "#000000",                    # used when legibility.outline
      "plate_color": "#1A1A1A",                       # used when legibility.solid_block
      "desc_override": "",
      # obj-only:
      "desc": "...",
      "color_palette": ["#RRGGBB", ...]              # obj element-level palette, <=5
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
LAYOUTS_FILENAME = "ideogram_layouts.json"
LAST_DESIGN_FILENAME = "ideogram_last_design.json"
USER_PRESETS_FILENAME = "ideogram_user_presets.json"
# Imported presets are copied here as individual JSON files (one per preset),
# under per-kind subfolders: user_presets/layouts/*.json, user_presets/styles/*.json.
USER_PRESETS_DIRNAME = "user_presets"

DEFAULT_ASPECT_RATIO = "16x9"
# v4 aspect-ratio enum (API uses the 'x' separator).
ASPECT_RATIOS = [
    "1x4", "1x3", "1x2", "9x16", "10x16", "2x3", "3x4", "4x5",
    "1x1", "5x4", "4x3", "3x2", "16x10", "16x9", "2x1", "3x1", "4x1",
]
PHOTO_MEDIUM = "photograph"
IMAGE_PALETTE_CAP = 16
ELEMENT_PALETTE_CAP = 5
DEFAULT_MEGAPIXELS = 1.0
MIN_MEGAPIXELS = 0.5
MAX_MEGAPIXELS = 2.0
DIM_MULTIPLE = 32

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


def load_layouts() -> list[dict]:
    data = _load_json_file(_preset_path(LAYOUTS_FILENAME))
    if isinstance(data, dict) and isinstance(data.get("layouts"), list):
        return [layout for layout in data["layouts"] if isinstance(layout, dict)]
    return []


def _preset_dir(kind_dir: str) -> str:
    return os.path.join(_module_dir(), USER_PRESETS_DIRNAME, kind_dir)


def _load_preset_dir(kind_dir: str) -> list[dict]:
    """Load every *.json preset from user_presets/<kind_dir>/ (one preset per file)."""
    out: list[dict] = []
    directory = _preset_dir(kind_dir)
    if not os.path.isdir(directory):
        return out
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return out
    for name in names:
        if not name.lower().endswith(".json"):
            continue
        data = _load_json_file(os.path.join(directory, name))
        if isinstance(data, dict) and data.get("id"):
            data.setdefault("custom", True)
            out.append(data)
    return out


def load_user_presets() -> dict:
    """User-saved custom presets: the legacy single JSON file PLUS per-file
    presets imported into user_presets/<kind>/. Shape: {"layouts": [...], "styles": [...]}."""
    out = {"layouts": [], "styles": []}
    path = _preset_path(USER_PRESETS_FILENAME)
    data = _load_json_file(path) if os.path.isfile(path) else None
    if isinstance(data, dict):
        for kind in ("layouts", "styles"):
            items = data.get(kind)
            if isinstance(items, list):
                out[kind] = [it for it in items if isinstance(it, dict) and it.get("id")]
    # Merge imported folder presets (dedupe by id; an imported file wins on conflict).
    for kind in ("layouts", "styles"):
        for it in _load_preset_dir(kind):
            pid = it.get("id")
            out[kind] = [x for x in out[kind] if x.get("id") != pid]
            out[kind].append(it)
    return out


def save_user_preset(kind: str, preset: dict) -> bool:
    """Append (replacing any same-id entry) a custom layout/style into the user file."""
    if kind not in ("layouts", "styles") or not isinstance(preset, dict):
        return False
    pid = str(preset.get("id") or "").strip()
    if not pid:
        return False
    store = load_user_presets()
    store[kind] = [it for it in store.get(kind, []) if it.get("id") != pid]
    store[kind].append(preset)
    try:
        # Atomic write (tmp + os.replace): a crash mid-write used to truncate
        # the store and lose every saved preset.
        target = _preset_path(USER_PRESETS_FILENAME)
        tmp_path = f"{target}.tmp-{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(store, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, target)
        return True
    except Exception as exc:  # noqa: BLE001
        ts_logger.warning("%s Failed to save user preset: %s", LOG_PREFIX, exc)
        return False


def _slug(value) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", str(value or "")).strip("_")[:40]


def save_imported_preset(kind: str, preset: dict) -> dict | None:
    """Validate + copy an imported layout/style into user_presets/<kind>/<id>.json.

    Returns the normalized preset (id ensured, custom=True) or None on failure.
    """
    if kind not in ("layouts", "styles") or not isinstance(preset, dict):
        return None
    import uuid  # noqa: PLC0415 - lazy

    item = dict(preset)
    pid = str(item.get("id") or "").strip()
    if not pid:
        base = _slug(item.get("name_en") or item.get("name_ru") or kind[:-1]) or "preset"
        pid = f"user_{base}_{uuid.uuid4().hex[:8]}"
    item["id"] = pid
    item["custom"] = True
    # Light shape guards so a malformed file cannot break the pickers later.
    if kind == "styles":
        item["color_palette"] = clean_palette(item.get("color_palette"), IMAGE_PALETTE_CAP)
    elif not isinstance(item.get("blocks"), list):
        item["blocks"] = []

    directory = _preset_dir(kind)
    try:
        os.makedirs(directory, exist_ok=True)
        filename = f"{_slug(pid) or 'preset'}.json"
        with open(os.path.join(directory, filename), "w", encoding="utf-8") as handle:
            json.dump(item, handle, ensure_ascii=False, indent=2)
        return item
    except Exception as exc:  # noqa: BLE001
        ts_logger.warning("%s Failed to save imported preset: %s", LOG_PREFIX, exc)
        return None


# --------------------------------------------------------------------------- #
# Full-design presets (top level): the complete editor state (design_json),
# stored one-per-file under user_presets/designs/<id>.json so a whole design —
# layout + style + objects + literal text + per-block prompt mods — can be
# saved, loaded, exported and imported as a single reusable preset.
# --------------------------------------------------------------------------- #
DESIGNS_DIRNAME = "designs"


def load_design_presets() -> list[dict]:
    """Every saved full-design preset: [{id, name, design, ...}, ...]."""
    return [
        item for item in _load_preset_dir(DESIGNS_DIRNAME)
        if isinstance(item.get("design"), dict) and item.get("id")
    ]


def design_presets_index() -> list[dict]:
    """Lightweight [{id, name}] list for the /presets payload (omits full designs)."""
    return [{"id": str(d["id"]), "name": str(d.get("name") or d["id"])} for d in load_design_presets()]


def save_design_preset(name, design, design_id=None) -> dict | None:
    """Write the full design under user_presets/designs/<id>.json. Returns {id, name}."""
    if not isinstance(design, dict):
        return None
    import uuid  # noqa: PLC0415 - lazy

    name = str(name or "").strip() or "design"
    did = str(design_id or "").strip() or f"design_{uuid.uuid4().hex[:10]}"
    item = {"id": did, "name": name, "version": 1, "custom": True, "design": design}
    directory = _preset_dir(DESIGNS_DIRNAME)
    try:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{_slug(did) or 'design'}.json"), "w", encoding="utf-8") as handle:
            json.dump(item, handle, ensure_ascii=False, indent=2)
        return {"id": did, "name": name}
    except Exception as exc:  # noqa: BLE001
        ts_logger.warning("%s Failed to save design preset: %s", LOG_PREFIX, exc)
        return None


def get_design_preset(design_id) -> dict | None:
    """Resolve a saved design preset by id -> {id, name, design}."""
    target = str(design_id or "")
    for d in load_design_presets():
        if str(d.get("id")) == target:
            return {"id": str(d["id"]), "name": str(d.get("name") or d["id"]), "design": d["design"]}
    return None


def import_design_preset(payload: dict) -> dict | None:
    """Accept an exported design file ({name, design} or a raw design/work object)
    and save it under a FRESH id, so imports never clobber an existing preset."""
    if not isinstance(payload, dict):
        return None
    design = payload.get("design")
    name = payload.get("name")
    if not isinstance(design, dict):
        # The file may BE the raw design (an editor work object).
        if isinstance(payload.get("blocks"), list) or isinstance(payload.get("style"), dict):
            design = payload
        else:
            return None
    return save_design_preset(name or "imported design", design)


def delete_design_preset(design_id) -> bool:
    directory = _preset_dir(DESIGNS_DIRNAME)
    if not os.path.isdir(directory):
        return False
    target = str(design_id or "")
    try:
        for fname in os.listdir(directory):
            if not fname.lower().endswith(".json"):
                continue
            data = _load_json_file(os.path.join(directory, fname))
            if isinstance(data, dict) and str(data.get("id")) == target:
                os.remove(os.path.join(directory, fname))
                return True
    except OSError as exc:
        ts_logger.warning("%s Failed to delete design preset: %s", LOG_PREFIX, exc)
    return False


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


def _title_case_word(word: str) -> str:
    """Capitalize only the first letter; leave the rest of the word alone.

    str.title() corrupted acronyms ("AI" -> "Ai") and apostrophe
    contractions ("don't" -> "Don'T") — it treats every non-letter as a
    word boundary.
    """
    for index, char in enumerate(word):
        if char.isalpha():
            return word[:index] + char.upper() + word[index + 1:]
    return word


def _apply_case(text: str, case: str) -> str:
    if case == "UPPERCASE":
        return text.upper()
    if case == "Title":
        # Per-line title casing keeps explicit \n line breaks intact.
        return "\n".join(
            " ".join(_title_case_word(word) for word in line.split(" "))
            for line in text.split("\n")
        )
    return text


# --------------------------------------------------------------------------- #
# desc composition for text blocks
# --------------------------------------------------------------------------- #
_WEIGHT_PHRASE = {"Thin": "thin light weight", "Bold": "bold weight", "Heavy": "heavy black weight"}
_CASE_PHRASE = {"UPPERCASE": "all uppercase", "Title": "title case"}
def _size_phrase_from_rect(rect) -> str:
    """Size derived from the DRAWN block height (mirrors JS sizePhraseFromRect)."""
    try:
        h = float(rect.get("h")) if isinstance(rect, dict) else 0.0
    except (TypeError, ValueError):
        h = 0.0
    if h >= 0.25:
        return "huge dominant hero headline"
    if h >= 0.12:
        return "large prominent headline"
    if h >= 0.06:
        return "medium body text"
    return "small caption text"


def compose_text_desc(block: dict, fonts_by_id: dict[str, dict] | None = None) -> str:
    """Build a text element's ``desc`` from ordered, non-empty slots.

    Mirror of the JS ``composeTextDesc`` in _ideogram_shared.js — keep in sync.
    Order: preset snippet (load-bearing, first) -> weight -> case -> size (from the
    drawn block) -> color -> legibility (outline/plate colors) -> Cyrillic hint ->
    user override (verbatim, last).
    """
    fonts_by_id = fonts_by_id if fonts_by_id is not None else _fonts_by_id()
    slots: list[str] = []

    # Slot 1 (lettering style): a per-block ``font_desc`` override wins over the
    # chosen font's snippet (mirror of JS) — lets the inspector show/edit this prose
    # and flip the font selector to "Custom" without fighting the snippet.
    font_desc = str(block.get("font_desc") or "").strip()
    preset = fonts_by_id.get(str(block.get("font_preset_id") or ""))
    snippet = (preset or {}).get("desc_snippet") if preset else None
    slots.append(font_desc or (str(snippet).strip() if snippet else "bold clean sans-serif"))

    weight = _WEIGHT_PHRASE.get(str(block.get("weight") or ""))
    if weight:
        slots.append(weight)

    case_phrase = _CASE_PHRASE.get(str(block.get("case") or ""))
    if case_phrase:
        slots.append(case_phrase)

    slots.append(_size_phrase_from_rect(block.get("rect")))

    color = norm_hex(block.get("color"))
    if color:
        slots.append(f"{color} colored letters")

    legibility = block.get("legibility")
    if isinstance(legibility, dict):
        if legibility.get("outline"):
            oc = norm_hex(block.get("outline_color"))
            slots.append(f"with a {oc} outline" if oc else "with a thin dark outline")
        if legibility.get("solid_block"):
            pc = norm_hex(block.get("plate_color"))
            slots.append(
                f"on a solid {pc} color block behind the text" if pc
                else "on a solid color block behind the text"
            )
    # Rendering style (crisp / soft / blurry / glowing) is intentionally NOT
    # hardcoded — it comes from the font descriptor + the user's desc_override,
    # so a "soft blurry letters" override is never fought by a forced "crisp" hint.

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
        # Text element colors = text + outline + plate colors (the per-block
        # palette UI was replaced by these explicit color pickers).
        leg = block.get("legibility") if isinstance(block.get("legibility"), dict) else {}
        colors = [norm_hex(block.get("color"))]
        if leg.get("outline"):
            colors.append(norm_hex(block.get("outline_color")))
        if leg.get("solid_block"):
            colors.append(norm_hex(block.get("plate_color")))
        text_palette = clean_palette([c for c in colors if c], ELEMENT_PALETTE_CAP)
        element: dict = {"type": "text"}
        if bbox is not None:
            element["bbox"] = bbox
        element["text"] = text
        element["desc"] = compose_text_desc(block, fonts_by_id)
        if text_palette:
            element["color_palette"] = text_palette
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

    # Fold the user's lighting colors into the lighting prose (Ideogram has no
    # separate lighting-color field).
    light_pal = clean_palette(style.get("lighting_palette"), ELEMENT_PALETTE_CAP)
    if light_pal:
        hint = ", ".join(light_pal) + " colored light"
        lighting = f"{lighting}, {hint}" if lighting else hint

    if aesthetics:
        out["aesthetics"] = aesthetics
    if lighting:
        out["lighting"] = lighting
    # photo XOR art_style, decided by medium. Per the Ideogram 4 spec the key
    # order differs by type: photo -> (photo, medium); non-photo -> (medium,
    # art_style). See docs/prompting.md "Key order is strict".
    if medium == PHOTO_MEDIUM:
        if photo:
            out["photo"] = photo
        if medium:
            out["medium"] = medium
    else:
        if medium:
            out["medium"] = medium
        if art_style:
            out["art_style"] = art_style
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
    # Fold the user's background colors into the background prose.
    bg_pal = clean_palette(design.get("background_palette"), ELEMENT_PALETTE_CAP)
    if bg_pal:
        hint = "dominant colors " + ", ".join(bg_pal)
        background = f"{background}, {hint}" if background else hint

    hld = str(design.get("high_level_description") or "").strip()
    style_description = _build_style_description(design.get("style"))

    # Nothing meaningful designed yet -> emit empty string (downstream no-op).
    # Includes style so a style-only design isn't silently dropped.
    if not elements and not background and not hld and not style_description:
        return "", aspect

    # Editor-state fields intentionally NOT in the caption: version/language/
    # layout_id (editor metadata), megapixels (consumed by dims_from_design, not
    # here), ref (preview underlay), style.preset_id & style.font_preset_id and
    # block.role (JS canvas-preview / new-block defaults only). Per-block
    # font_preset_id IS consumed, via compose_text_desc.
    caption: dict = {}
    if hld:  # optional per the spec — omit rather than emit an empty string
        caption["high_level_description"] = hld
    if style_description:
        caption["style_description"] = style_description
    # Omit empty background / elements (and the whole block when both are empty)
    # rather than leaking "background":"" / "elements":[] into the prompt.
    comp: dict = {}
    if background:
        comp["background"] = background
    if elements:
        comp["elements"] = elements
    if comp:
        caption["compositional_deconstruction"] = comp

    serialized = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
    return serialized, aspect


def aspect_ratio_value(aspect: str) -> float:
    """Parse a 'WxH' aspect token into the float ratio W/H."""
    try:
        w_str, h_str = str(aspect or DEFAULT_ASPECT_RATIO).split("x")
        w, h = float(w_str), float(h_str)
        if w > 0 and h > 0:
            return w / h
    except (ValueError, TypeError):
        pass
    return 16.0 / 9.0


def dims_from_aspect_mp(aspect: str, megapixels) -> tuple[int, int]:
    """Compute (width, height) for ``aspect`` at the target megapixels, each
    rounded to a multiple of DIM_MULTIPLE (32) and >= DIM_MULTIPLE."""
    import math

    ratio = aspect_ratio_value(aspect)
    try:
        mp = float(megapixels)
    except (ValueError, TypeError):
        mp = DEFAULT_MEGAPIXELS
    mp = max(MIN_MEGAPIXELS, min(MAX_MEGAPIXELS, mp))
    total = mp * 1_000_000.0
    height = math.sqrt(total / ratio)
    width = height * ratio

    def _round32(value: float) -> int:
        return max(DIM_MULTIPLE, int(round(value / DIM_MULTIPLE)) * DIM_MULTIPLE)

    return _round32(width), _round32(height)


def dims_from_design(design_json: str) -> tuple[int, int]:
    """Resolve (width, height) from the design's aspect_ratio + megapixels."""
    try:
        design = json.loads(design_json) if design_json else {}
    except Exception:  # noqa: BLE001
        design = {}
    if not isinstance(design, dict):
        design = {}
    aspect = str(design.get("aspect_ratio") or DEFAULT_ASPECT_RATIO)
    if aspect not in ASPECT_RATIOS:
        aspect = DEFAULT_ASPECT_RATIO
    return dims_from_aspect_mp(aspect, design.get("megapixels", DEFAULT_MEGAPIXELS))


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
        user = load_user_presets()
        for item in user.get("layouts", []):
            item.setdefault("custom", True)
        for item in user.get("styles", []):
            item.setdefault("custom", True)
        return web.json_response({
            "layouts": load_layouts() + user.get("layouts", []),
            "styles": load_styles() + user.get("styles", []),
            "fonts": load_fonts(),
            "designs": design_presets_index(),
            "aspect_ratios": ASPECT_RATIOS,
            "default_aspect_ratio": DEFAULT_ASPECT_RATIO,
        })

    # ── Full-design presets (top level) ──────────────────────────────────── #
    @routes.get("/ts_ideogram/design")
    async def _get_design(request):
        item = get_design_preset(request.query.get("id", ""))
        return web.json_response(item if item else {})

    @routes.post("/ts_ideogram/save_design")
    async def _save_design(request):
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400)
        if not isinstance(payload, dict):
            return web.json_response({"ok": False})
        saved = save_design_preset(payload.get("name"), payload.get("design"), payload.get("id"))
        return web.json_response({"ok": bool(saved), **(saved or {})})

    @routes.post("/ts_ideogram/import_design")
    async def _import_design(request):
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400)
        saved = import_design_preset(payload if isinstance(payload, dict) else {})
        return web.json_response({"ok": bool(saved), **(saved or {})})

    @routes.post("/ts_ideogram/delete_design")
    async def _delete_design(request):
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400)
        ok = delete_design_preset(payload.get("id")) if isinstance(payload, dict) else False
        return web.json_response({"ok": bool(ok)})

    @routes.post("/ts_ideogram/save_preset")
    async def _save_preset(request):
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400)
        if not isinstance(payload, dict):
            return web.json_response({"ok": False})
        kind = payload.get("kind")
        store_key = "layouts" if kind == "layout" else "styles" if kind == "style" else ""
        ok = save_user_preset(store_key, payload.get("preset") or {})
        return web.json_response({"ok": bool(ok)})

    @routes.post("/ts_ideogram/import_preset")
    async def _import_preset(request):
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return web.Response(status=400)
        if not isinstance(payload, dict):
            return web.json_response({"ok": False})
        kind = payload.get("kind")
        store_key = "layouts" if kind == "layout" else "styles" if kind == "style" else ""
        saved = save_imported_preset(store_key, payload.get("preset") or {}) if store_key else None
        if saved:
            return web.json_response({"ok": True, "preset": saved})
        return web.json_response({"ok": False})

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
        dj = design_json if isinstance(design_json, str) else ""
        caption, aspect = build_caption(dj)
        width, height = dims_from_design(dj)
        return web.json_response({"json_prompt": caption, "aspect_ratio": aspect, "width": width, "height": height})


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
                "weight": "Bold", "case": "UPPERCASE",
                "legibility": {"outline": True, "solid_block": True},
                "color": "#ffe000", "outline_color": "#000000", "plate_color": "#101010",
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
    # Ideogram 4 spec: non-photo key order is aesthetics, lighting, MEDIUM, art_style, palette.
    check("non-photo style key order (medium before art_style)",
          list(sd.keys()) == ["aesthetics", "lighting", "medium", "art_style", "color_palette"])
    # ...and the photo path swaps to aesthetics, lighting, PHOTO, medium, palette.
    photo_sd = json.loads(build_caption(json.dumps({
        "background": "studio", "blocks": [],
        "style": {"medium": "photograph", "aesthetics": "a", "lighting": "l",
                  "photo": "85mm f/1.4", "color_palette": ["#000000"]},
    }))[0])["style_description"]
    check("photo style key order (photo before medium)",
          list(photo_sd.keys()) == ["aesthetics", "lighting", "photo", "medium", "color_palette"])
    els = caption["compositional_deconstruction"]["elements"]
    check("three elements built", len(els) == 3)
    t0 = els[0]
    check("text literal uppercased", t0["text"] == "СРОЧНО")
    check("text element key order", list(t0.keys()) == ["type", "bbox", "text", "desc", "color_palette"])
    check("bbox y-first in element", t0["bbox"] == frac_to_bbox(0.04, 0.12, 0.5, 0.3))
    check("cyrillic hint in desc", "Cyrillic script" in t0["desc"])
    check("desc snippet is first slot", t0["desc"].startswith("bold grotesque sans-serif"))
    check("size derived from drawn rect (h=0.3 -> hero)", "huge dominant hero headline" in t0["desc"])
    check("outline color folded into desc", "with a #000000 outline" in t0["desc"])
    check("plate color folded into desc", "on a solid #101010 color block behind the text" in t0["desc"])
    check("text palette = text + outline + plate", t0["color_palette"] == ["#FFE000", "#000000", "#101010"])
    check("visual_only text became obj placeholder", els[2]["type"] == "obj" and "text" not in els[2])
    check("ensure_ascii=False keeps Cyrillic literal", "СРОЧНО" in caption_str)

    print("-" * 40)
    print("ALL PASSED" if failures == 0 else f"{failures} FAILURE(S)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    _self_test()
