// Shared constants + pure utilities for the TS Ideogram Designer front-end.
// The in-node preview (_ideogram_node.js) and the modal editor
// (_ideogram_editor.js) both import from here. Keep the data-shaping helpers
// (fracToBbox, composeTextDesc) in sync with the Python side
// (nodes/ideogram/_ideogram_helpers.py). The authoritative caption is built
// by the backend (/ts_ideogram/preview); the JS mirrors only drive previews.

import { api } from "/scripts/api.js";

export const NODE_NAME = "TS_IdeogramDesigner";
export const DESIGN_INPUT = "design_json";
export const ROUTE_BASE = "/ts_ideogram";

export const ASPECT_RATIOS = [
    "1x4", "1x3", "1x2", "9x16", "10x16", "2x3", "3x4", "4x5",
    "1x1", "5x4", "4x3", "3x2", "16x10", "16x9", "2x1", "3x1", "4x1",
];
export const DEFAULT_ASPECT_RATIO = "16x9";
export const DEFAULT_LANG = "ru";
export const LANGS = ["ru", "en"];
export const WEIGHTS = ["Thin", "Regular", "Bold"];
export const CASES = ["As-typed", "UPPERCASE", "Title"];
export const PHOTO_MEDIUM = "photograph";
export const IMAGE_PALETTE_CAP = 16;
export const ELEMENT_PALETTE_CAP = 5;
export const DEFAULT_MEGAPIXELS = 1.0;
export const MIN_MEGAPIXELS = 0.5;
export const MAX_MEGAPIXELS = 2.0;
export const DIM_MULTIPLE = 32;

// ── DOM/widget helpers (mirrors of the sam_media_loader patterns) ─────────── //
export function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || null;
}

export function hideWidget(node, name) {
    const widget = getWidget(node, name);
    if (widget) {
        widget.hidden = true;
        widget.type = "hidden";
        widget.serialize = true;
        widget.options = { ...(widget.options || {}), hidden: true, serialize: true };
        widget.computeSize = () => [0, -4];
    }
    const input = node?.inputs?.find((item) => item?.name === name);
    if (input) input.hidden = true;
}

export function getWidgetValue(node, name, fallback) {
    return getWidget(node, name)?.value ?? fallback;
}

export function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (widget) {
        widget.value = value;
        if (typeof widget.callback === "function") widget.callback(value);
    }
}

export function stopPropagation(element, events) {
    events.forEach((name) => element.addEventListener(name, (event) => event.stopPropagation()));
}

export function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

// ── Data shaping (mirror of _ideogram_helpers.py) ─────────────────────────── //
const HEX_RE = /^#[0-9A-Fa-f]{6}$/;
const CYRILLIC_RE = /[Ѐ-ӿ]/;

export function hasCyrillic(text) {
    return CYRILLIC_RE.test(text || "");
}

export function normHex(value) {
    if (typeof value !== "string") return null;
    const v = value.trim();
    return HEX_RE.test(v) ? v.toUpperCase() : null;
}

export function cleanPalette(values, cap) {
    const out = [];
    if (!Array.isArray(values)) return out;
    for (const raw of values) {
        const hex = normHex(raw);
        if (hex && !out.includes(hex)) out.push(hex);
        if (out.length >= cap) break;
    }
    return out;
}

function clampInt(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

// Editor rect (fractions 0..1, top-left origin) -> [y_min,x_min,y_max,x_max] 0-1000.
export function fracToBbox(x, y, w, h) {
    let x0 = clampInt(Math.round(x * 1000), 0, 1000);
    let y0 = clampInt(Math.round(y * 1000), 0, 1000);
    let x1 = clampInt(Math.round((x + w) * 1000), 0, 1000);
    let y1 = clampInt(Math.round((y + h) * 1000), 0, 1000);
    if (x1 <= x0) {
        x1 = Math.min(1000, x0 + 1);
        if (x1 <= x0) x0 = Math.max(0, x1 - 1); // x0 was already at the 1000 ceiling
    }
    if (y1 <= y0) {
        y1 = Math.min(1000, y0 + 1);
        if (y1 <= y0) y0 = Math.max(0, y1 - 1);
    }
    return [y0, x0, y1, x1];
}

export function applyCase(text, mode) {
    if (mode === "UPPERCASE") return (text || "").toUpperCase();
    if (mode === "Title") {
        // Mirror Python _apply_case/_title_case_word: per line, per space-split
        // word, uppercase only the FIRST Unicode letter and leave the rest
        // verbatim (do NOT lowercase the tail — keeps 'AI'→'AI', 'iPhone'→'IPhone',
        // and works for Cyrillic). The old /\w\S*/ + toLowerCase diverged from the
        // authoritative backend, so the WYSIWYG canvas preview now matches it.
        const titleWord = (word) => {
            const m = word.match(/\p{L}/u);
            if (!m) return word;
            const i = m.index;
            return word.slice(0, i) + m[0].toUpperCase() + word.slice(i + m[0].length);
        };
        return (text || "")
            .split("\n")
            .map((line) => line.split(" ").map(titleWord).join(" "))
            .join("\n");
    }
    return text || "";
}

const WEIGHT_PHRASE = { Thin: "thin light weight", Bold: "bold weight", Heavy: "heavy black weight" };
const CASE_PHRASE = { UPPERCASE: "all uppercase", Title: "title case" };
// Size is derived from the DRAWN block height (not a manual picker), reinforcing
// the bbox the model already gets — so the rendered size IS the requested size.
export function sizePhraseFromRect(rect) {
    const h = (rect && Number(rect.h)) || 0;
    if (h >= 0.25) return "huge dominant hero headline";
    if (h >= 0.12) return "large prominent headline";
    if (h >= 0.06) return "medium body text";
    return "small caption text";
}

// Mirror of compose_text_desc(): ordered, non-empty slots joined with ", ".
export function composeTextDesc(block, fontsById) {
    const slots = [];
    const preset = fontsById?.[block.font_preset_id || ""];
    slots.push((preset?.desc_snippet || "").trim() || "bold clean sans-serif");

    const weight = WEIGHT_PHRASE[block.weight];
    if (weight) slots.push(weight);
    const casePhrase = CASE_PHRASE[block.case];
    if (casePhrase) slots.push(casePhrase);
    slots.push(sizePhraseFromRect(block.rect));

    const color = normHex(block.color);
    if (color) slots.push(`${color} colored letters`);

    const leg = block.legibility || {};
    if (leg.outline) {
        const oc = normHex(block.outline_color);
        slots.push(oc ? `with a ${oc} outline` : "with a thin dark outline");
    }
    if (leg.solid_block) {
        const pc = normHex(block.plate_color);
        slots.push(pc ? `on a solid ${pc} color block behind the text` : "on a solid color block behind the text");
    }
    // Rendering style (crisp / soft / blurry / glowing) is intentionally NOT
    // hardcoded — it comes from the font descriptor + the user's desc_override,
    // so a "soft blurry letters" override is never fought by a forced "crisp" hint.

    if (hasCyrillic(block.text || "")) slots.push("Cyrillic script, Russian text");

    const override = (block.desc_override || "").trim();
    if (override) slots.push(override);

    return slots.filter(Boolean).join(", ");
}

// ── Design state ──────────────────────────────────────────────────────────── //
export function makeDefaultDesign() {
    return {
        version: 1,
        language: DEFAULT_LANG,
        layout_id: "",
        aspect_ratio: DEFAULT_ASPECT_RATIO,
        megapixels: DEFAULT_MEGAPIXELS,
        high_level_description: "",
        background: "",
        background_palette: [],
        style: {
            preset_id: "",
            aesthetics: "",
            lighting: "",
            lighting_palette: [],
            medium: "graphic_design",
            photo: "",
            art_style: "",
            color_palette: [],
            font_preset_id: "",
        },
        blocks: [],
    };
}

export function parseDesign(raw) {
    if (!raw || typeof raw !== "string") return makeDefaultDesign();
    try {
        const data = JSON.parse(raw);
        if (!data || typeof data !== "object") return makeDefaultDesign();
        return { ...makeDefaultDesign(), ...data, style: { ...makeDefaultDesign().style, ...(data.style || {}) } };
    } catch {
        return makeDefaultDesign();
    }
}

let _uidCounter = 0;
export function makeBlockId() {
    _uidCounter += 1;
    return `b${Date.now().toString(36)}_${_uidCounter}`;
}

export function aspectToRatio(aspect) {
    // Mirror Python aspect_ratio_value: whole-token 16/9 fallback on a malformed
    // token (e.g. "0x5"/"8x0"), not per-component defaults.
    const parts = String(aspect || DEFAULT_ASPECT_RATIO).split("x");
    const w = Number(parts[0]);
    const h = Number(parts[1]);
    if (w > 0 && h > 0) return w / h;
    return 16 / 9;
}

// Compute an artboard pixel box for a given aspect that fits inside (maxW,maxH).
export function aspectFitBox(aspect, maxW, maxH) {
    const ratio = aspectToRatio(aspect);
    let w = maxW;
    let h = w / ratio;
    if (h > maxH) {
        h = maxH;
        w = h * ratio;
    }
    return { w: Math.max(1, Math.round(w)), h: Math.max(1, Math.round(h)) };
}

// Resolve output pixel dimensions from aspect + megapixels, each a multiple of
// DIM_MULTIPLE (32). Mirror of dims_from_aspect_mp in _ideogram_helpers.py.
export function dimsFromAspectMp(aspect, megapixels) {
    const ratio = aspectToRatio(aspect);
    let mp = Number(megapixels);
    if (!(mp > 0)) mp = DEFAULT_MEGAPIXELS;
    mp = Math.max(MIN_MEGAPIXELS, Math.min(MAX_MEGAPIXELS, mp));
    const total = mp * 1e6;
    const h = Math.sqrt(total / ratio);
    const w = h * ratio;
    const r32 = (v) => Math.max(DIM_MULTIPLE, Math.round(v / DIM_MULTIPLE) * DIM_MULTIPLE);
    return { w: r32(w), h: r32(h) };
}

// ── Backend calls ─────────────────────────────────────────────────────────── //
let _presetsCache = null;
export function invalidatePresetsCache() { _presetsCache = null; }
export async function loadPresets() {
    if (_presetsCache) return _presetsCache;
    try {
        const response = await fetch(api.apiURL(`${ROUTE_BASE}/presets`));
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        _presetsCache = await response.json();
    } catch (error) {
        console.error("[TS Ideogram] Failed to load presets:", error);
        _presetsCache = { layouts: [], styles: [], fonts: [], aspect_ratios: ASPECT_RATIOS, default_aspect_ratio: DEFAULT_ASPECT_RATIO };
    }
    return _presetsCache;
}

export function fontsById(presets) {
    const map = {};
    (presets?.fonts || []).forEach((font) => {
        if (font?.id) map[font.id] = font;
    });
    return map;
}

export async function fetchCaptionPreview(designJson) {
    try {
        const response = await fetch(api.apiURL(`${ROUTE_BASE}/preview`), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ design_json: designJson }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.warn("[TS Ideogram] Caption preview failed:", error);
        return null;
    }
}

// ── Full-design presets (top level) ───────────────────────────────────────── //
// A design preset = the complete editor state (work / design_json), saved by
// name in the node's user_presets/designs/ folder. One import/export covers the
// whole design (layout + style + objects + literal text + per-block prompt mods).
export function designsList(presets) { return presets?.designs || []; }

export async function saveDesignPreset(name, design) {
    try {
        const r = await fetch(api.apiURL(`${ROUTE_BASE}/save_design`), {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, design }),
        });
        return await r.json();
    } catch (e) { console.warn("[TS Ideogram] save_design failed:", e); return null; }
}

export async function fetchDesignPreset(id) {
    try {
        const r = await fetch(api.apiURL(`${ROUTE_BASE}/design?id=${encodeURIComponent(id)}`));
        if (!r.ok) return null;
        const d = await r.json();
        return d && d.design ? d : null;
    } catch (e) { console.warn("[TS Ideogram] fetch design failed:", e); return null; }
}

export async function importDesignPreset(payload) {
    try {
        const r = await fetch(api.apiURL(`${ROUTE_BASE}/import_design`), {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        return await r.json();
    } catch (e) { console.warn("[TS Ideogram] import_design failed:", e); return null; }
}

export async function deleteDesignPreset(id) {
    try {
        const r = await fetch(api.apiURL(`${ROUTE_BASE}/delete_design`), {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id }),
        });
        return await r.json();
    } catch (e) { console.warn("[TS Ideogram] delete_design failed:", e); return null; }
}

export async function persistDesign(designJson) {
    try {
        await fetch(api.apiURL(`${ROUTE_BASE}/config`), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ design_json: designJson }),
        });
    } catch {
        /* best-effort */
    }
}

// Build a /view URL for an input-dir image (reference underlay).
export function inputViewUrl(filename, subfolder = "", type = "input") {
    const params = new URLSearchParams({ filename: filename || "", type, subfolder: subfolder || "" });
    return api.apiURL(`/view?${params.toString()}`);
}

// ── Curated style preset libraries ──────────────────────────────────────────── //
// Each preset carries a localized label (ru/en) and `v` — the English prompt
// fragment dropped verbatim into the matching style_description field. The
// caption stays plain prose, so the Python builder is untouched: these dropdowns
// only fill style.aesthetics / style.lighting / style.art_style|photo / background.

// Art style ALSO sets the whole-image medium (the photo-XOR-art_style switch):
// `medium === "photograph"` → `v` goes into style.photo, otherwise into art_style.
export const ARTSTYLE_PRESETS = [
    { id: "flat_vector", ru: "Плоский вектор", en: "Flat vector", medium: "illustration", v: "flat vector illustration, clean geometric shapes, bold flat colors, crisp edges" },
    { id: "poster_graphic", ru: "Плакатная графика", en: "Poster graphics", medium: "graphic_design", v: "bold graphic poster design, high contrast, screen-print look, strong shapes" },
    { id: "watercolor", ru: "Акварель", en: "Watercolor", medium: "painting", v: "loose watercolor painting, soft pigment bleeds, textured paper, hand-painted" },
    { id: "oil_paint", ru: "Масляная живопись", en: "Oil painting", medium: "painting", v: "rich oil painting, visible brush strokes, painterly impasto, classical" },
    { id: "render_3d", ru: "3D-рендер", en: "3D render", medium: "3d_render", v: "polished 3D render, soft global illumination, subsurface scattering, studio look" },
    { id: "lowpoly_iso", ru: "Low-poly изометрия", en: "Low-poly iso", medium: "3d_render", v: "low-poly isometric 3D, faceted shapes, soft studio render, clean palette" },
    { id: "clay_3d", ru: "Пластилин / clay", en: "Clay 3D", medium: "3d_render", v: "cute clay plasticine 3D render, soft matte material, rounded forms" },
    { id: "anime", ru: "Аниме", en: "Anime", medium: "illustration", v: "anime style, clean cel shading, crisp linework, expressive" },
    { id: "comic_pop", ru: "Комикс / поп-арт", en: "Comic pop-art", medium: "illustration", v: "comic book pop-art, halftone dots, bold ink outlines, punchy colors" },
    { id: "engraving", ru: "Гравюра / линогравюра", en: "Engraving", medium: "illustration", v: "vintage engraving and linocut, fine hatching, monochrome ink, woodcut texture" },
    { id: "risograph", ru: "Ризограф", en: "Risograph", medium: "graphic_design", v: "risograph print, grainy overprint, limited spot colors, slight misregistration" },
    { id: "photoreal", ru: "Фотореализм", en: "Photoreal", medium: "photograph", v: "DSLR photograph, 50mm lens, natural depth of field, true-to-life detail" },
    { id: "cinematic_photo", ru: "Кинокадр", en: "Cinematic photo", medium: "photograph", v: "cinematic film still, anamorphic, shallow depth of field, color graded" },
];

export const MOOD_PRESETS = [
    { id: "bold", ru: "Дерзко и сочно", en: "Bold & punchy", v: "bold and punchy, vibrant, high energy" },
    { id: "calm", ru: "Спокойно и минимально", en: "Calm & minimal", v: "calm, minimal, lots of negative space" },
    { id: "luxe", ru: "Премиально / люкс", en: "Luxurious", v: "luxurious, premium, elegant and refined" },
    { id: "retro", ru: "Ретро / винтаж", en: "Retro vintage", v: "retro vintage, nostalgic, aged texture" },
    { id: "cozy", ru: "Тёплый и уютный", en: "Warm & cozy", v: "warm and cozy, inviting, soft and friendly" },
    { id: "tech", ru: "Технологично / футуризм", en: "High-tech", v: "high-tech, futuristic, sleek and clean" },
    { id: "playful", ru: "Игриво и весело", en: "Playful & fun", v: "playful, fun, whimsical and cheerful" },
    { id: "cinematic", ru: "Драматично / кино", en: "Cinematic", v: "dramatic, cinematic, moody atmosphere" },
    { id: "organic", ru: "Натурально / эко", en: "Natural & organic", v: "natural, organic, earthy and handmade" },
    { id: "edgy", ru: "Гранж / андеграунд", en: "Grungy & edgy", v: "grungy, edgy, raw underground vibe" },
    { id: "dreamy", ru: "Мечтательно / пастель", en: "Dreamy pastel", v: "dreamy, soft pastel, ethereal and airy" },
    { id: "corporate", ru: "Деловой / чистый", en: "Clean corporate", v: "clean corporate, professional, trustworthy" },
];

export const LIGHTING_PRESETS = [
    { id: "daylight", ru: "Яркий дневной", en: "Bright daylight", v: "bright natural daylight" },
    { id: "studio", ru: "Мягкий студийный", en: "Soft studio", v: "soft even studio lighting" },
    { id: "golden", ru: "Золотой час", en: "Golden hour", v: "warm golden hour backlight" },
    { id: "dramatic", ru: "Драматичные тени", en: "Dramatic shadows", v: "dramatic chiaroscuro shadows, strong contrast" },
    { id: "neon", ru: "Неоновое свечение", en: "Neon glow", v: "vivid neon glow, colorful rim light" },
    { id: "backlit", ru: "Контровой свет", en: "Backlit rim", v: "strong backlight, glowing rim, silhouette" },
    { id: "overcast", ru: "Рассеянный пасмурный", en: "Soft overcast", v: "soft diffuse overcast light" },
    { id: "spotlight", ru: "Ночь / софит", en: "Night spotlight", v: "dark scene, focused spotlight" },
    { id: "highkey", ru: "Высокий ключ", en: "High-key", v: "high-key, bright and airy, minimal shadows" },
    { id: "lowkey", ru: "Низкий ключ", en: "Low-key", v: "low-key, deep shadows, moody" },
    { id: "godrays", ru: "Объёмные лучи света", en: "Sun beams", v: "volumetric god rays, atmospheric haze" },
    { id: "product", ru: "Предметный софтбокс", en: "Product softbox", v: "clean product softbox lighting, soft reflections" },
];

export const BACKGROUND_PRESETS = [
    { id: "solid", ru: "Сплошной цвет", en: "Solid color", v: "clean solid color background", colors: ["#0F172A", "#F1F5F9", "#2563EB", "#E2E8F0"] },
    { id: "gradient", ru: "Плавный градиент", en: "Smooth gradient", v: "smooth multi-tone gradient background", colors: ["#6D28D9", "#2563EB", "#06B6D4", "#F472B6"] },
    { id: "studio_dark", ru: "Тёмная студия", en: "Dark studio", v: "seamless dark studio backdrop with a soft vignette", colors: ["#0A0A0F", "#1A1A22", "#2A2A35", "#454552"] },
    { id: "studio_light", ru: "Светлая студия", en: "Light studio", v: "clean light seamless studio backdrop, soft falloff", colors: ["#FFFFFF", "#F2F4F7", "#E2E8F0", "#CBD5E1"] },
    { id: "bokeh", ru: "Размытие / боке", en: "Blurred bokeh", v: "blurred background with soft glowing bokeh orbs", colors: ["#1A1330", "#3B2A6B", "#FFB37D", "#FFE08A"] },
    { id: "neon_city", ru: "Неоновый город", en: "Neon city night", v: "blurred neon city street at night, glowing signs and wet reflections", colors: ["#0B0B16", "#1A0E2E", "#FF2E97", "#22D3FF", "#B026FF"] },
    { id: "dark_moody", ru: "Тёмный драматичный", en: "Dark moody", v: "dark moody atmospheric backdrop with drifting haze", colors: ["#08090C", "#14161C", "#22262F", "#3A3F4B"] },
    { id: "brick_wall", ru: "Кирпичная стена", en: "Brick wall", v: "dark textured brick wall at night, weathered mortar", colors: ["#1A1012", "#2E1A1A", "#5C2E2A", "#8A4B3C"] },
    { id: "sunset_sky", ru: "Закатное небо", en: "Sunset sky", v: "warm sunset sky with soft gradient clouds", colors: ["#2B1055", "#7B2D8E", "#FF6B6B", "#FF9E5E", "#FFD36E"] },
    { id: "cosmic_space", ru: "Космос / туманность", en: "Cosmic space", v: "deep space with stars and a colorful glowing nebula", colors: ["#05050F", "#140A2E", "#3B1E6B", "#7A2E8C", "#22D3FF"] },
    { id: "luxury_dark", ru: "Тёмный люкс", en: "Luxe dark", v: "dark elegant backdrop with subtle gold accents", colors: ["#0A0A0A", "#1A1A1A", "#C9A227", "#E8D8A0"] },
    { id: "nature_green", ru: "Природа / зелень", en: "Nature green", v: "lush green natural outdoor scenery with soft light", colors: ["#1B3A2B", "#2E6B4F", "#6BBF59", "#A7D88B", "#EAF3D6"] },
    { id: "abstract_fluid", ru: "Абстракция / флюид", en: "Abstract fluid", v: "abstract flowing liquid shapes, vivid and layered", colors: ["#3A0CA3", "#7209B7", "#F72585", "#4CC9F0", "#4361EE"] },
    { id: "paper_craft", ru: "Бумага / крафт", en: "Paper & craft", v: "textured kraft paper background, organic grain", colors: ["#F3E9D2", "#E4D2A8", "#C9A878", "#8A6E45"] },
    { id: "grunge", ru: "Гранж-текстура", en: "Grunge texture", v: "grungy distressed textured wall, raw and edgy", colors: ["#1C1B19", "#3A362E", "#6B6051", "#9A8C73"] },
    { id: "geometric", ru: "Геометрия / паттерн", en: "Geometric pattern", v: "bold geometric pattern background, repeating motif", colors: ["#0F172A", "#F1F5F9", "#FF2D55", "#FFD500", "#1E90FF"] },
    { id: "glow_mesh", ru: "Свечение / меш", en: "Glow mesh", v: "soft glowing mesh gradient, dreamy and ethereal", colors: ["#1A1330", "#5B3A8C", "#C77DFF", "#7DD3FC", "#FBC2EB"] },
    { id: "minimal", ru: "Минимальный / пусто", en: "Minimal / none", v: "minimal plain background, lots of whitespace", colors: ["#FFFFFF", "#FAFAFA", "#EEEEEE", "#111111"] },
];

// Object presets — popular, vivid subjects for an `obj` block's `desc`. Mix of
// emotional characters and hero objects. Character descs fix the gender/identity
// (the obj desc dominates its bbox) while staying emotion-flexible so the
// high_level_description can still drive the specific mood. Referenced by id from
// LAYOUT_BRIEFS[*].subjectPreset to keep ideas and the rendered subject in sync.
export const OBJECT_PRESETS = [
    { id: "bold_girl", ru: "Дерзкая девушка", en: "Bold girl", v: "a strikingly beautiful young woman looking at the viewer with a bold confident vibe, expressive, dramatic rim light, cinematic, ultra-detailed" },
    { id: "glamour_woman", ru: "Гламурная девушка", en: "Glamorous woman", v: "a glamorous stylish beautiful woman, elegant and confident, flawless skin, soft cinematic glamour lighting, ultra-detailed" },
    { id: "brutal_man", ru: "Брутальный мужчина", en: "Brutal man", v: "a brutally handsome rugged man with a strong jaw and confident presence looking at the viewer, dramatic side light, cinematic, ultra-detailed" },
    { id: "business_pro", ru: "Деловой профи", en: "Business pro", v: "a confident successful person in a sharp tailored suit with a self-assured look, clean premium corporate lighting" },
    { id: "shocked_face", ru: "Шок / эмоция", en: "Shocked face", v: "a person with a hugely shocked wide-eyed open-mouth reaction looking straight at the viewer, very expressive, bright punchy rim light" },
    { id: "happy_smile", ru: "Счастливая улыбка", en: "Happy smile", v: "an attractive person with a warm genuine bright smile, friendly inviting mood, soft flattering glow" },
    { id: "athlete", ru: "Спортсмен", en: "Athlete in motion", v: "a dynamic athletic person mid-action full of energy and motion, dramatic sport lighting, sweat highlights, powerful" },
    { id: "gamer", ru: "Геймер", en: "Gamer", v: "an intense focused young gamer wearing headphones lit by colorful RGB neon, immersive and cinematic" },
    { id: "couple", ru: "Влюблённая пара", en: "Couple in love", v: "a beautiful couple in a tender emotional embrace, romantic warm golden-hour light, cinematic and heartfelt" },
    { id: "mascot", ru: "Милый маскот", en: "Cute mascot", v: "a cute friendly cartoon mascot character with big expressive eyes, bold clean shapes, playful and memorable" },
    { id: "hero_product", ru: "Продукт-герой", en: "Hero product", v: "a hero product centered with premium studio lighting, soft reflections, sharp focus, immaculate e-commerce quality" },
    { id: "tasty_food", ru: "Аппетитная еда", en: "Tasty food", v: "a delicious appetizing dish with fresh ingredients, mouth-watering detail, vibrant professional food photography" },
    { id: "gadget", ru: "Гаджет / девайс", en: "Gadget", v: "a sleek modern gadget device floating with a soft reflection, clean futuristic tech presentation" },
    { id: "sneaker", ru: "Кроссовок", en: "Sneaker", v: "a stylish sneaker shot at a dynamic angle with crisp detail and vivid product lighting" },
    { id: "supercar", ru: "Суперкар", en: "Supercar", v: "a sleek glossy supercar at a dramatic three-quarter angle, cinematic reflections, golden-hour glow" },
    { id: "pet", ru: "Милый питомец", en: "Cute pet", v: "an adorable expressive pet animal looking at the viewer, soft warm light, charming and heart-melting" },
    { id: "fruit_splash", ru: "Фруктовый всплеск", en: "Fruit Splash", v: "A vibrant midair burst of strawberries, citrus wedges and glistening berries exploding through a crown of splashing juice, every glossy droplet frozen razor-sharp in crisp high-key commercial light." },
    { id: "gourmet_dish", ru: "Изысканное блюдо", en: "Gourmet Dish", v: "An exquisitely plated fine-dining course with delicate sauce swirls, jewel-like micro-herb garnish and faint wisps of rising steam, shot close in soft moody restaurant light against deep shadow." },
    { id: "majestic_lion", ru: "Величественный лев", en: "Majestic Lion", v: "A majestic male lion with a thick wind-tousled mane and an intense regal amber gaze, bathed in dramatic golden side light that rims every individual hair in razor-sharp detail." },
    { id: "exotic_plant", ru: "Тропическое растение", en: "Exotic Plant", v: "Lush exotic monstera and tropical foliage with glossy deep-green fenestrated leaves layered through the frame, traced by dappled shafts of soft diffused jungle light." },
    { id: "crystal_gem", ru: "Сияющий самоцвет", en: "Crystal Gem", v: "A glowing faceted crystal gemstone refracting prismatic rainbows through its sharp cleaved edges, suspended in a luminous magical aura that fades into soft velvety darkness." },
    { id: "vintage_car", ru: "Винтажный автомобиль", en: "Vintage Car", v: "A glossy classic vintage car posed at a dramatic three-quarter angle, its gleaming chrome trim and sculpted curved bodywork catching warm liquid golden-hour reflections." },
    { id: "astronaut", ru: "Космонавт", en: "Astronaut", v: "An astronaut drifting weightless in the silent void, a swirling nebula and scattered stars mirrored in luminous detail across the curved reflective visor, cinematic and awe-inspiring." },
    { id: "friendly_robot", ru: "Дружелюбный робот", en: "Friendly Robot", v: "A charming friendly robot character with big expressive glowing eyes and a smooth glossy injection-molded 3D body, lit by soft wraparound studio light against a clean seamless backdrop." },
    { id: "dragon", ru: "Дракон", en: "Dragon", v: "A powerful majestic dragon with intricately overlapping scales and vast spread wings, rearing through drifting embers and curling mist in an epic, breathtaking fantasy atmosphere." },
    { id: "mountain_vista", ru: "Горная панорама", en: "Mountain Vista", v: "A breathtaking mountain range at sunrise, jagged snow-dusted peaks glowing rose-pink while rivers of drifting mist pool through the valleys in a vast cinematic vista." },
    { id: "ocean_wave", ru: "Океанская волна", en: "Ocean Wave", v: "A powerful cresting ocean wave curling into a translucent turquoise barrel, backlit sunlit spray flung in golden arcs from its feathering crest, dynamic and majestic." },
    { id: "koi_fish", ru: "Карп кои", en: "Koi Fish", v: "Elegant orange-and-white koi gliding through dark glassy water, gentle silver ripples spreading around floating lily pads in a serene, graceful overhead scene." },
    { id: "butterfly", ru: "Бабочка", en: "Butterfly", v: "A vivid butterfly with iridescent blue-and-amber wings perched on a dew-kissed blossom, captured in delicate macro detail with luminous creamy bokeh dissolving behind it." },
    { id: "cosmic_nebula", ru: "Космическая туманность", en: "Cosmic Nebula", v: "A vast cosmic nebula of swirling magenta and teal stardust threaded with glowing newborn stars, drifting in luminous silence through the boundless depths of deep space." },
];

// "Main idea" (high_level_description) presets, 10 per layout, adapted to that
// layout. `v` is the English one-sentence HLD fed to the model; ru/en are the
// dropdown labels. Many feature vivid, emotional characters (a beautiful woman,
// a brutally handsome man) so preset designs come out cinematic and expressive.
// Keyed by the builtin layout id (see ideogram_layouts.json).
export const LAYOUT_BRIEFS = {
    youtube_thumbnail: [
        { ru: "Шок на лице", en: "Shocked reaction", v: "A high-energy YouTube thumbnail: a brutally handsome man with a shocked wide-eyed reaction on the right, dramatic rim light, a huge punchy headline on the left.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "dramatic", background_id: "studio_dark",
          objects: [{ role: "subject", desc: "A brutally handsome rugged man on the right, eyes wide and jaw dropped in genuine shock, hands flying up, hit by a sharp cyan-orange rim light that carves his silhouette." }],
          texts: [{ role: "headline", en: "I CAN'T BELIEVE THIS", ru: "Я В ШОКЕ" }, { role: "accent", en: "WATCH NOW", ru: "СМОТРИ" }] },
        { ru: "Дерзкая девушка", en: "Bold confident girl", v: "A vibrant thumbnail with a strikingly beautiful young woman smirking with confidence, glowing neon rim light, a bold contrasty headline beside her.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "neon", background_id: "gradient",
          objects: [{ role: "subject", desc: "A strikingly beautiful young woman with a sly confident smirk, arms crossed, edged by hot-pink and cyan neon rim light that traces her hair and shoulders." }],
          texts: [{ role: "headline", en: "SHE DID IT AGAIN", ru: "ОНА СНОВА В ИГРЕ" }, { role: "accent", en: "NEW", ru: "НОВОЕ" }] },
        { ru: "Деньги и успех", en: "Money & success", v: "A flashy success thumbnail: a confident man in a sharp suit with cash flying behind him, gold accents, an explosive headline.",
          style_preset_id: "bold_commercial", aesthetics_id: "luxe", lighting_id: "dramatic", background_id: "luxury_dark",
          objects: [{ role: "subject", desc: "A confident man in a sharp tailored navy suit, arms spread wide and a winning grin, cash raining down behind him as gold light reflects off his watch." }],
          texts: [{ role: "headline", en: "HOW I MADE $1,000,000", ru: "КАК Я ЗАРАБОТАЛ МИЛЛИОН" }, { role: "accent", en: "FREE", ru: "БЕСПЛАТНО" }] },
        { ru: "До и после", en: "Before & after", v: "A split before/after thumbnail with a transformed, glowing person, a bold arrow and a punchy comparison headline.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "golden", background_id: "gradient",
          objects: [{ role: "subject", desc: "A person shown in a striking transformation, the left side tired and slouched in muted tones, the right side radiant, fit and confident bathed in warm glowing light, with a bold yellow arrow sweeping from old to new." }],
          texts: [{ role: "headline", en: "30 DAYS LATER", ru: "СПУСТЯ 30 ДНЕЙ" }, { role: "accent", en: "VS", ru: "ПРОТИВ" }] },
        { ru: "Гнев / спор", en: "Rage / drama", v: "A dramatic confrontation thumbnail: a furious brutal man pointing straight at the viewer, intense red glow, an aggressive headline.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "spotlight", background_id: "dark_moody",
          objects: [{ role: "subject", desc: "A furious brutal man jabbing his finger straight at the viewer, veins tensed and teeth bared, half his face drowned in deep red shadow and aggressive rim light." }],
          texts: [{ role: "headline", en: "WE NEED TO TALK", ru: "НАМ НАДО ПОГОВОРИТЬ" }, { role: "accent", en: "DRAMA", ru: "СКАНДАЛ" }] },
        { ru: "Восторг / вау", en: "Amazed wow", v: "An exciting reveal thumbnail: a beautiful woman with a delighted, amazed expression, sparkles and glow, a bright energetic headline.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "highkey", background_id: "bokeh",
          objects: [{ role: "subject", desc: "A beautiful woman with a delighted open-mouthed gasp and hands on her cheeks, eyes sparkling with amazement, surrounded by glittering sparkles and a warm glow." }],
          texts: [{ role: "headline", en: "THIS IS INSANE!", ru: "ЭТО НЕРЕАЛЬНО!" }, { role: "accent", en: "WOW", ru: "ВАУ" }] },
        { ru: "Геймер в наушниках", en: "Gamer", v: "A gaming thumbnail: an intense focused gamer in headphones lit by colorful RGB neon, explosive action behind, a bold headline.",
          style_preset_id: "futuristic_tech", aesthetics_id: "tech", lighting_id: "neon", background_id: "neon_city",
          objects: [{ role: "subject", desc: "An intense focused young gamer wearing glowing RGB headphones, eyes locked forward in concentration, face lit by vivid magenta and cyan neon reflections." }],
          texts: [{ role: "headline", en: "WORLD RECORD RUN", ru: "МИРОВОЙ РЕКОРД" }, { role: "accent", en: "LIVE", ru: "В ЭФИРЕ" }] },
        { ru: "Загадка / интрига", en: "Mystery hook", v: "A mysterious thumbnail: a hooded figure half in shadow under a single dramatic light, intriguing atmosphere and a teasing headline.",
          style_preset_id: "cinematic", aesthetics_id: "edgy", lighting_id: "lowkey", background_id: "grunge",
          objects: [{ role: "subject", desc: "A hooded figure with the face half swallowed by shadow, only one eye and the jawline catching the harsh single beam of light, an aura of secrecy around them." }],
          texts: [{ role: "headline", en: "NOBODY TALKS ABOUT THIS", ru: "ОБ ЭТОМ МОЛЧАТ" }, { role: "accent", en: "SECRET", ru: "СЕКРЕТ" }] },
        { ru: "Роскошный образ жизни", en: "Luxury lifestyle", v: "A luxury lifestyle thumbnail: a stylish woman beside a supercar at golden hour, aspirational glamour and a bold headline.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "golden", background_id: "sunset_sky",
          objects: [{ role: "subject", desc: "A stylish elegant woman in designer attire leaning against a gleaming low supercar, golden-hour sunlight catching her hair and the car's polished bodywork." }],
          texts: [{ role: "headline", en: "THE BILLIONAIRE LIFE", ru: "ЖИЗНЬ МИЛЛИАРДЕРА" }, { role: "accent", en: "LUXURY", ru: "ЛЮКС" }] },
        { ru: "Эксперт-объяснение", en: "Expert explainer", v: "An educational thumbnail: a charismatic presenter gesturing at glowing infographic elements, a clean confident look and a clear headline.",
          style_preset_id: "modern_clean", aesthetics_id: "corporate", lighting_id: "studio", background_id: "solid",
          objects: [{ role: "subject", desc: "A charismatic presenter in smart-casual attire gesturing confidently toward bright floating infographic graphics, a warm trustworthy smile and clean studio lighting." }],
          texts: [{ role: "headline", en: "EXPLAINED IN 5 MINUTES", ru: "ОБЪЯСНЯЮ ЗА 5 МИНУТ" }, { role: "accent", en: "GUIDE", ru: "ГАЙД" }] },
    ],
    ad_poster: [
        { ru: "Большая распродажа", en: "Big sale", v: "A punchy sale poster with an explosive discount headline, the hero product centered in a spotlight and a vivid color burst.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "spotlight", background_id: "geometric",
          objects: [{ role: "product", desc: "the hero product blasting forward in a tight center spotlight, edges crackling with energy and a glowing color halo behind it" }],
          texts: [{ role: "headline", en: "BIG SALE", ru: "БОЛЬШАЯ РАСПРОДАЖА" }, { role: "cta", en: "Shop Now", ru: "Купить сейчас" }] },
        { ru: "Модель с продуктом", en: "Model with product", v: "A glossy advertising poster: a beautiful confident model holding the product in glamorous studio light, a bold headline and price.",
          style_preset_id: "editorial", aesthetics_id: "luxe", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "product", desc: "a beautiful confident model with flawless makeup holding the product up to the camera, glossy studio key light wrapping her face and the product surface" }],
          texts: [{ role: "headline", en: "Your New Look", ru: "Твой новый образ" }, { role: "cta", en: "Only $49", ru: "Всего 49 $" }] },
        { ru: "Брутальный герой бренда", en: "Brand hero (man)", v: "A bold poster with a brutal rugged man as the brand hero, dramatic side light, a strong product and a powerful tagline.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "dramatic", background_id: "dark_moody",
          objects: [{ role: "product", desc: "a rugged brutal man with a stubbled jaw and intense gaze, gripping the brand product, his muscular silhouette sculpted by hard cinematic side light" }],
          texts: [{ role: "headline", en: "Built Tough", ru: "Сделано на века" }, { role: "cta", en: "Join the Crew", ru: "Вступай в команду" }] },
        { ru: "Премиальный запуск", en: "Premium launch", v: "An elegant product-launch poster: the new product floating in soft premium light, minimal luxury type and a refined palette.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "lowkey", background_id: "luxury_dark",
          objects: [{ role: "product", desc: "the new product floating weightlessly at center, kissed by soft diffused premium light with delicate gold reflections along its polished edges" }],
          texts: [{ role: "headline", en: "Introducing", ru: "Премьера" }, { role: "cta", en: "Discover More", ru: "Узнать больше" }] },
        { ru: "Фестиваль / событие", en: "Event / festival", v: "A vibrant event poster bursting with energy and dynamic shapes, a big date headline and an exciting atmosphere.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "neon", background_id: "neon_city",
          objects: [{ role: "product", desc: "a glowing festival stage cluster with spotlights, balloons and a giant illuminated ticket bursting with celebratory energy" }],
          texts: [{ role: "headline", en: "Summer Fest", ru: "Летний фестиваль" }, { role: "cta", en: "Get Tickets", ru: "Купить билеты" }] },
        { ru: "Еда крупным планом", en: "Food hero", v: "An appetizing food poster: a delicious dish in mouth-watering detail with fresh ingredients and a bold tasty headline.",
          style_preset_id: "gourmet_food", aesthetics_id: "cozy", lighting_id: "daylight", background_id: "paper_craft",
          objects: [{ role: "product", desc: "a mouth-watering signature dish in glistening macro detail, steam curling up, fresh vibrant ingredients and a juicy splash frozen mid-air" }],
          texts: [{ role: "headline", en: "So Delicious", ru: "Невероятно вкусно" }, { role: "cta", en: "Order Today", ru: "Заказать сегодня" }] },
        { ru: "Фитнес / энергия", en: "Fitness energy", v: "A high-energy fitness poster: an athletic woman mid-motion with dynamic light and sweat, a motivational bold headline.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "neon", background_id: "dark_moody",
          objects: [{ role: "product", desc: "an athletic woman caught mid-motion in a powerful lunge, toned muscles glistening with sweat, dynamic rim light tracing her explosive movement" }],
          texts: [{ role: "headline", en: "Push Your Limit", ru: "Преодолей себя" }, { role: "cta", en: "Start Training", ru: "Начать тренировки" }] },
        { ru: "Скидка −50%", en: "50% off", v: "A loud discount poster with a giant -50% burst, the product in a beam of light and urgent contrasty colors.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "dramatic", background_id: "geometric",
          objects: [{ role: "product", desc: "the product standing in a hot diagonal beam of light, bold and crisp, with a giant exploding -50% sticker bursting beside it" }],
          texts: [{ role: "headline", en: "-50% OFF", ru: "СКИДКА −50%" }, { role: "cta", en: "Buy Now", ru: "Купить сейчас" }] },
        { ru: "Услуга / доверие", en: "Trusted service", v: "A clean trustworthy service poster: a friendly professional, a clear benefit headline and a calm confident palette.",
          style_preset_id: "modern_clean", aesthetics_id: "corporate", lighting_id: "studio", background_id: "minimal",
          objects: [{ role: "product", desc: "a friendly approachable professional in neat attire, smiling warmly with arms relaxed, bathed in soft trustworthy daylight" }],
          texts: [{ role: "headline", en: "We've Got You", ru: "Мы всё решим" }, { role: "cta", en: "Book a Call", ru: "Записаться" }] },
        { ru: "Ретро-постер", en: "Retro poster", v: "A stylish retro advertising poster with vintage textures, bold mid-century type and warm nostalgic colors.",
          style_preset_id: "retro_vintage", aesthetics_id: "retro", lighting_id: "golden", background_id: "paper_craft",
          objects: [{ role: "product", desc: "the hero product styled in nostalgic mid-century fashion, rendered with vintage print grain and a warm retro color wash" }],
          texts: [{ role: "headline", en: "The Classic Choice", ru: "Классика на все времена" }, { role: "cta", en: "Try It Today", ru: "Попробуй сегодня" }] },
    ],
    social_post: [
        { ru: "Топ лайфхак", en: "Top lifehack", v: "A clean square social post with a short bold lifehack headline, a simple friendly illustration and a small swipe footer.",
          style_preset_id: "modern_clean", aesthetics_id: "calm", lighting_id: "highkey", background_id: "glow_mesh",
          objects: [{ role: "illustration", desc: "a simple friendly flat-style illustration of a glowing lightbulb sprouting a tiny green leaf, drawn with clean rounded lines and a single accent color" }],
          texts: [{ role: "headline", en: "Top lifehack", ru: "Топ-лайфхак" }, { role: "footer", en: "Swipe", ru: "Листай дальше" }] },
        { ru: "Цитата дня", en: "Quote of the day", v: "An inspiring quote post: elegant typography over a soft gradient, a small accent mark and a calm aesthetic.",
          style_preset_id: "editorial", aesthetics_id: "dreamy", lighting_id: "studio", background_id: "gradient",
          objects: [{ role: "illustration", desc: "a small elegant gold quotation mark set as a refined accent in the upper corner, thin and minimal" }],
          texts: [{ role: "headline", en: "Believe in yourself", ru: "Верь в себя" }, { role: "footer", en: "Quote of the day", ru: "Цитата дня" }] },
        { ru: "Эмоция девушки", en: "Girl's mood", v: "A lifestyle social post: a beautiful woman laughing candidly in soft natural light, a warm authentic mood and a short caption.",
          style_preset_id: "cinematic", aesthetics_id: "cozy", lighting_id: "golden", background_id: "bokeh",
          objects: [{ role: "illustration", desc: "a beautiful young woman laughing candidly, head tilted back, soft natural light catching her hair, relaxed and authentic" }],
          texts: [{ role: "headline", en: "Good vibes only", ru: "Только хорошее настроение" }, { role: "footer", en: "Enjoy the moment", ru: "Лови момент" }] },
        { ru: "Брутальный портрет", en: "Bold male portrait", v: "A striking square portrait of a brutal stylish man with an intense gaze, moody studio light and a short punchy caption.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "dramatic", background_id: "studio_dark",
          objects: [{ role: "illustration", desc: "a brutal stylish man with a sharp jawline and stubble, intense piercing gaze straight at camera, lit by hard moody studio light" }],
          texts: [{ role: "headline", en: "Stay sharp", ru: "Держи стиль" }, { role: "footer", en: "No compromise", ru: "Без компромиссов" }] },
        { ru: "Анонс / новинка", en: "Announcement", v: "A bold announcement post with a big NEW headline, a simple product hint and energetic accent shapes.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "studio", background_id: "geometric",
          objects: [{ role: "illustration", desc: "a sleek mystery product silhouette wrapped in a glowing spotlight, hinted but not fully revealed, with a burst of dynamic accent shapes around it" }],
          texts: [{ role: "headline", en: "NEW", ru: "НОВИНКА" }, { role: "footer", en: "Coming soon", ru: "Уже скоро" }] },
        { ru: "Большая цифра", en: "Big number fact", v: "A bold data post built around one huge number, a short caption and clean geometric accents.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "studio", background_id: "solid",
          objects: [{ role: "illustration", desc: "a single oversized bold numeral as the hero element, paired with clean minimal geometric shapes and a thin accent line" }],
          texts: [{ role: "headline", en: "90%", ru: "90%" }, { role: "footer", en: "of people don't know this", ru: "людей этого не знают" }] },
        { ru: "До / после", en: "Before / after", v: "A clean before/after social post showing a transformation, a divider line and a short result caption.",
          style_preset_id: "modern_clean", aesthetics_id: "corporate", lighting_id: "overcast", background_id: "studio_light",
          objects: [{ role: "illustration", desc: "a side-by-side transformation split down the middle, dull faded version on the left and bright vibrant improved version on the right, divided by a clean accent line" }],
          texts: [{ role: "headline", en: "Before / After", ru: "До и после" }, { role: "footer", en: "The result speaks", ru: "Результат налицо" }] },
        { ru: "Минимал-арт", en: "Minimal art", v: "A minimal aesthetic post: a single elegant object on lots of negative space, refined type and a calm palette.",
          style_preset_id: "modern_clean", aesthetics_id: "calm", lighting_id: "studio", background_id: "minimal",
          objects: [{ role: "illustration", desc: "a single elegant minimalist object — a smooth ceramic vase with one slender stem — centered amid generous empty space, casting a delicate shadow" }],
          texts: [{ role: "headline", en: "Less is more", ru: "Меньше — значит больше" }, { role: "footer", en: "Minimal", ru: "Минимализм" }] },
        { ru: "Праздник", en: "Celebration", v: "A festive greeting post with a warm celebratory mood, soft confetti accents and a heartfelt short message.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "golden", background_id: "bokeh",
          objects: [{ role: "illustration", desc: "a cheerful illustrated party scene with floating balloons, a small frosted cake topped with sparklers and drifting confetti ribbons" }],
          texts: [{ role: "headline", en: "Congratulations!", ru: "Поздравляем!" }, { role: "footer", en: "Wishing you the best", ru: "Всего самого лучшего" }] },
        { ru: "Юмор / мем", en: "Fun / meme", v: "A playful humorous post with a punchy funny line, a bold expressive character and bright cheeky colors.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "daylight", background_id: "solid",
          objects: [{ role: "illustration", desc: "a bold expressive cartoon character with a wildly exaggerated surprised face, big googly eyes and a goofy open-mouth reaction" }],
          texts: [{ role: "headline", en: "Me on Monday", ru: "Я в понедельник" }, { role: "footer", en: "Relatable, right?", ru: "Знакомо, да?" }] },
    ],
    web_banner: [
        { ru: "SaaS-герой", en: "SaaS hero", v: "A clean landing hero: a bold value headline and CTA on the left, a sleek product UI mockup on a device on the right, airy whitespace.",
          style_preset_id: "modern_clean", aesthetics_id: "tech", lighting_id: "highkey", background_id: "gradient",
          objects: [{ role: "product", desc: "A sleek silver laptop angled three-quarters on the right, screen glowing with a crisp SaaS dashboard UI of charts, metrics and a tidy sidebar" }],
          texts: [{ role: "headline", en: "Run your business on autopilot", ru: "Бизнес на автопилоте" }, { role: "subtitle", en: "All your workflows in one clean dashboard", ru: "Все процессы в одной панели" }, { role: "cta", en: "Start free", ru: "Начать бесплатно" }] },
        { ru: "Команда / люди", en: "Team / people", v: "A warm landing hero with a friendly team smiling in soft daylight, a clear headline and a CTA button.",
          style_preset_id: "editorial", aesthetics_id: "cozy", lighting_id: "daylight", background_id: "nature_green",
          objects: [{ role: "product", desc: "A friendly diverse team of four professionals standing together and smiling, relaxed and confident in a bright workspace" }],
          texts: [{ role: "headline", en: "Great work starts with great people", ru: "Сильная команда — сильный результат" }, { role: "subtitle", en: "Join a team that has your back", ru: "Команда, на которую можно положиться" }, { role: "cta", en: "Join us", ru: "Присоединиться" }] },
        { ru: "Девушка-амбассадор", en: "Brand ambassador", v: "A bright hero banner: a beautiful confident woman as the brand ambassador on the right, a bold benefit headline and CTA on the left.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "product", desc: "A beautiful confident young woman on the right, mid-laugh with a radiant smile, stylish casual outfit, arms relaxed and posture open" }],
          texts: [{ role: "headline", en: "Look good. Feel unstoppable.", ru: "Выгляди ярко, чувствуй силу" }, { role: "subtitle", en: "The everyday boost you deserve", ru: "Твой заряд на каждый день" }, { role: "cta", en: "Shop now", ru: "Купить сейчас" }] },
        { ru: "Распродажа / промо", en: "Sale promo", v: "A high-contrast promo banner with a big discount headline, the product and an urgent CTA in an energetic palette.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "dramatic", background_id: "geometric",
          objects: [{ role: "product", desc: "A glossy pair of premium sneakers floating mid-air with a bright price-burst sticker, catching punchy studio highlights" }],
          texts: [{ role: "headline", en: "Up to 50% OFF", ru: "Скидки до 50%" }, { role: "subtitle", en: "Limited time only — don't miss out", ru: "Только сейчас — успей купить" }, { role: "cta", en: "Grab the deal", ru: "Забрать скидку" }] },
        { ru: "Мобильное приложение", en: "App showcase", v: "A modern app-launch hero: a phone mockup with a slick UI, short benefit copy and a download CTA over a clean gradient.",
          style_preset_id: "modern_clean", aesthetics_id: "tech", lighting_id: "studio", background_id: "glow_mesh",
          objects: [{ role: "product", desc: "A modern smartphone tilted slightly, screen showing a sleek minimal app interface with rounded cards and a vibrant accent color" }],
          texts: [{ role: "headline", en: "Your day, simplified", ru: "Проще каждый день" }, { role: "subtitle", en: "Everything you need in one app", ru: "Всё нужное в одном приложении" }, { role: "cta", en: "Download now", ru: "Скачать" }] },
        { ru: "Премиум-бренд", en: "Premium brand", v: "An elegant minimal hero: a premium product in soft light, a refined serif headline and generous negative space.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "product", background_id: "studio_light",
          objects: [{ role: "product", desc: "A luxury crystal perfume bottle with a gold cap, standing on a subtle pedestal, catching delicate soft highlights" }],
          texts: [{ role: "headline", en: "Crafted for the few", ru: "Создано для избранных" }, { role: "subtitle", en: "Timeless elegance, refined to perfection", ru: "Вечная элегантность в каждой детали" }, { role: "cta", en: "Discover", ru: "Узнать больше" }] },
        { ru: "Курс / вебинар", en: "Course / webinar", v: "An education hero: a confident expert on the right, a clear course headline and a sign-up CTA on the left.",
          style_preset_id: "editorial", aesthetics_id: "corporate", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "product", desc: "A confident expert presenter on the right, smiling and gesturing, smart-casual outfit, holding a tablet" }],
          texts: [{ role: "headline", en: "Master new skills in 30 days", ru: "Новый навык за 30 дней" }, { role: "subtitle", en: "Live webinar with a top industry expert", ru: "Живой вебинар с экспертом" }, { role: "cta", en: "Sign up free", ru: "Записаться" }] },
        { ru: "Тёмная тех-тема", en: "Dark tech", v: "A sleek dark-mode tech hero with glowing accents, a bold headline, a futuristic product render and a CTA.",
          style_preset_id: "futuristic_tech", aesthetics_id: "tech", lighting_id: "neon", background_id: "dark_moody",
          objects: [{ role: "product", desc: "A futuristic glass-and-metal device floating in the dark, edges rimmed with glowing cyan and violet light, sleek and minimal" }],
          texts: [{ role: "headline", en: "The future is now", ru: "Будущее уже здесь" }, { role: "subtitle", en: "Next-gen power, beautifully engineered", ru: "Мощь нового поколения" }, { role: "cta", en: "Explore tech", ru: "Смотреть" }] },
        { ru: "Эко / природа", en: "Eco / nature", v: "A fresh natural hero with organic textures and greenery, a calm headline and a soft CTA in an earthy palette.",
          style_preset_id: "botanical_watercolor", aesthetics_id: "organic", lighting_id: "overcast", background_id: "paper_craft",
          objects: [{ role: "product", desc: "A natural amber glass bottle of eco skincare nestled among fresh green leaves and dewdrops" }],
          texts: [{ role: "headline", en: "Pure by nature", ru: "Чистота природы" }, { role: "subtitle", en: "Kind to your skin, kind to the planet", ru: "Бережно к коже и природе" }, { role: "cta", en: "Go natural", ru: "Выбрать эко" }] },
        { ru: "Чёрная пятница", en: "Black Friday", v: "A bold Black Friday hero: a dramatic dark background, a huge sale headline, a glowing product and an urgent CTA.",
          style_preset_id: "bold_commercial", aesthetics_id: "cinematic", lighting_id: "spotlight", background_id: "luxury_dark",
          objects: [{ role: "product", desc: "Premium wireless headphones glowing under a sharp spotlight, gold accents gleaming against the dark, floating above a reflective surface" }],
          texts: [{ role: "headline", en: "BLACK FRIDAY", ru: "ЧЁРНАЯ ПЯТНИЦА" }, { role: "subtitle", en: "Biggest discounts of the year", ru: "Самые большие скидки года" }, { role: "cta", en: "Shop the sale", ru: "За покупками" }] },
    ],
    book_cover: [
        { ru: "Деловой бестселлер", en: "Business bestseller", v: "An elegant business book cover: a bold title on top, a striking conceptual illustration and the author name at the bottom.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "spotlight", background_id: "luxury_dark",
          objects: [{ role: "illustration", desc: "A striking conceptual illustration of a golden ascending arrow forming a staircase of stacked coins, rendered as a sleek metallic emblem at the cover's heart." }],
          texts: [{ role: "title", en: "THE GROWTH CODE", ru: "КОД РОСТА" }, { role: "author", en: "Daniel Hart", ru: "Даниэль Харт" }] },
        { ru: "Героиня романа", en: "Novel heroine", v: "A dramatic novel cover featuring a beautiful woman with an emotional gaze, atmospheric lighting and elegant title typography.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "lowkey", background_id: "dark_moody",
          objects: [{ role: "illustration", desc: "A beautiful woman in a flowing dark gown, half-lit by dramatic side light, gazing past the viewer with quiet emotional intensity." }],
          texts: [{ role: "title", en: "THE SILENT HOUR", ru: "ТИХИЙ ЧАС" }, { role: "author", en: "Elena Vassar", ru: "Елена Вассар" }] },
        { ru: "Брутальный триллер", en: "Thriller hero", v: "A tense thriller cover: a brutal man's silhouette in moody shadow with dramatic rim light and a bold ominous title.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "spotlight", background_id: "neon_city",
          objects: [{ role: "illustration", desc: "The brutal silhouette of a broad-shouldered man in a long coat, carved out by a sharp blue rim light, fists clenched in the dark." }],
          texts: [{ role: "title", en: "NO WAY BACK", ru: "ПУТИ НАЗАД НЕТ" }, { role: "author", en: "Mark Devlin", ru: "Марк Девлин" }] },
        { ru: "Фэнтези-мир", en: "Fantasy world", v: "An epic fantasy cover with a breathtaking magical landscape, a lone heroic figure and ornate title lettering.",
          style_preset_id: "surreal_concept", aesthetics_id: "dreamy", lighting_id: "godrays", background_id: "glow_mesh",
          objects: [{ role: "illustration", desc: "A lone cloaked hero standing on a cliff edge, sword glinting, silhouetted against a towering crystal spire wreathed in arcane light." }],
          texts: [{ role: "title", en: "THE LAST EMBER", ru: "ПОСЛЕДНИЙ УГОЛЁК" }, { role: "author", en: "Aria Thornwood", ru: "Ариа Торнвуд" }] },
        { ru: "Саморазвитие", en: "Self-help", v: "A bright uplifting self-help cover: a bold motivational title, a clean symbolic illustration and a warm optimistic palette.",
          style_preset_id: "modern_clean", aesthetics_id: "calm", lighting_id: "highkey", background_id: "gradient",
          objects: [{ role: "illustration", desc: "A clean symbolic illustration of a sprouting seedling growing through an open mind silhouette, drawn in confident minimal line art." }],
          texts: [{ role: "title", en: "BECOME UNSTOPPABLE", ru: "СТАНЬ НЕОСТАНОВИМЫМ" }, { role: "author", en: "Sofia Lane", ru: "София Лейн" }] },
        { ru: "Тёмный детектив", en: "Noir mystery", v: "A noir detective cover with a rain-soaked moody scene, a mysterious silhouette and classic dramatic typography.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "neon", background_id: "neon_city",
          objects: [{ role: "illustration", desc: "A mysterious figure in a fedora and trench coat seen from behind under a streetlamp, long shadow stretching across wet cobblestones." }],
          texts: [{ role: "title", en: "THE LAST WITNESS", ru: "ПОСЛЕДНИЙ СВИДЕТЕЛЬ" }, { role: "author", en: "Victor Reyes", ru: "Виктор Рейес" }] },
        { ru: "Любовный роман", en: "Romance", v: "A tender romance cover: a beautiful couple in a soft emotional embrace at golden hour and an elegant flowing title.",
          style_preset_id: "cinematic", aesthetics_id: "cozy", lighting_id: "golden", background_id: "sunset_sky",
          objects: [{ role: "illustration", desc: "A beautiful couple in a soft, tender embrace, foreheads touching, their hair catching the warm glow of the setting sun." }],
          texts: [{ role: "title", en: "Until We Meet Again", ru: "Пока мы не встретимся снова" }, { role: "author", en: "Clara Bennett", ru: "Клара Беннетт" }] },
        { ru: "Научпоп", en: "Science / non-fiction", v: "A clean science non-fiction cover with an elegant conceptual illustration, a confident modern title and a refined palette.",
          style_preset_id: "modern_clean", aesthetics_id: "corporate", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "illustration", desc: "An elegant conceptual illustration of a glowing neural network blooming into a constellation, rendered in clean teal and graphite linework." }],
          texts: [{ role: "title", en: "THE THINKING BRAIN", ru: "МЫСЛЯЩИЙ МОЗГ" }, { role: "author", en: "Dr. Owen Pratt", ru: "Д-р Оуэн Пратт" }] },
        { ru: "Детская книга", en: "Children's book", v: "A charming children's book cover with a cute friendly character, a playful colorful illustration and a rounded title.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "daylight", background_id: "nature_green",
          objects: [{ role: "illustration", desc: "A cute round little fox with oversized sparkling eyes and a big happy grin, waving cheerfully from a patch of daisies." }],
          texts: [{ role: "title", en: "Pip the Brave Little Fox", ru: "Храбрый лисёнок Пип" }, { role: "author", en: "Mia Foster", ru: "Миа Фостер" }] },
        { ru: "Минимал-обложка", en: "Minimal cover", v: "A striking minimal book cover: one bold symbolic shape on a refined color field and elegant restrained typography.",
          style_preset_id: "editorial", aesthetics_id: "calm", lighting_id: "studio", background_id: "solid",
          objects: [{ role: "illustration", desc: "One bold symbolic shape: a perfect deep-black circle with a single clean fracture line cutting through its center." }],
          texts: [{ role: "title", en: "FRACTURE", ru: "РАЗЛОМ" }, { role: "author", en: "J. R. Holt", ru: "Дж. Р. Холт" }] },
    ],
    logo: [
        { ru: "Чистый текстовый логотип", en: "Clean wordmark", v: "A clean centered wordmark logo with a small minimal icon above and a tagline below, balanced and legible.",
          style_preset_id: "modern_clean", aesthetics_id: "calm", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "icon", desc: "a small minimal geometric icon mark — a single thin-line abstract circle with a precise notch, perfectly centered above the wordmark" }],
          texts: [{ role: "wordmark", en: "AETHER", ru: "ЭФИР" }, { role: "tagline", en: "Simply made", ru: "Сделано просто" }] },
        { ru: "Геометрический знак", en: "Geometric mark", v: "A modern logo with a bold geometric icon mark, a confident sans-serif wordmark and generous whitespace.",
          style_preset_id: "modern_clean", aesthetics_id: "corporate", lighting_id: "studio", background_id: "minimal",
          objects: [{ role: "icon", desc: "a bold geometric icon built from interlocking triangles and a solid square, rendered in deep navy with one sharp accent corner" }],
          texts: [{ role: "wordmark", en: "VERTEX", ru: "ВЕРТЕКС" }, { role: "tagline", en: "Build forward", ru: "Строим вперёд" }] },
        { ru: "Винтажный значок", en: "Vintage badge", v: "A vintage emblem logo: a circular badge with classic lettering, a small crest icon and a refined retro feel.",
          style_preset_id: "retro_vintage", aesthetics_id: "retro", lighting_id: "overcast", background_id: "paper_craft",
          objects: [{ role: "icon", desc: "a small heraldic crest icon with crossed laurel branches and a banner, engraved in classic two-tone retro linework inside a circular badge frame" }],
          texts: [{ role: "wordmark", en: "OAK & IRON CO.", ru: "ДУБ И ЖЕЛЕЗО" }, { role: "tagline", en: "Est. 1924", ru: "Основано в 1924" }] },
        { ru: "Премиум-монограмма", en: "Luxury monogram", v: "An elegant luxury monogram logo with a refined serif initial, a thin frame and a premium minimal palette.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "lowkey", background_id: "luxury_dark",
          objects: [{ role: "icon", desc: "an elegant interlocking serif monogram of the letters L and M in brushed gold, enclosed in a thin minimalist gold frame" }],
          texts: [{ role: "wordmark", en: "MAISON LUMIÈRE", ru: "МЕЗОН ЛЮМЬЕР" }, { role: "tagline", en: "Quietly exceptional", ru: "Тихая роскошь" }] },
        { ru: "Игривый бренд", en: "Playful brand", v: "A friendly playful logo with a rounded wordmark, a cute simple mascot icon and cheerful colors.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "highkey", background_id: "gradient",
          objects: [{ role: "icon", desc: "a cute simple smiling cloud mascot with round rosy cheeks and tiny waving arms, drawn in friendly rounded shapes" }],
          texts: [{ role: "wordmark", en: "Puffy", ru: "Пушинка" }, { role: "tagline", en: "Made to smile", ru: "Создано радовать" }] },
        { ru: "Тех-стартап", en: "Tech startup", v: "A sleek tech-startup logo with a clean geometric mark, a modern wordmark and a confident gradient accent.",
          style_preset_id: "futuristic_tech", aesthetics_id: "tech", lighting_id: "neon", background_id: "studio_dark",
          objects: [{ role: "icon", desc: "a sleek geometric icon of an abstract upward arrow fused with a node circuit, filled with a vivid blue-to-violet gradient and a clean glow" }],
          texts: [{ role: "wordmark", en: "NOVALABS", ru: "НОВАЛАБ" }, { role: "tagline", en: "Ship the future", ru: "Запускаем будущее" }] },
        { ru: "Кофейня / крафт", en: "Coffee / craft", v: "A cozy craft logo with a hand-drawn icon, warm rustic lettering and an artisanal vibe.",
          style_preset_id: "handcrafted_artistic", aesthetics_id: "cozy", lighting_id: "daylight", background_id: "paper_craft",
          objects: [{ role: "icon", desc: "a hand-drawn steaming coffee cup with rustic ink linework and a single coffee bean, sketched in a warm cozy illustrative style" }],
          texts: [{ role: "wordmark", en: "Daily Grind", ru: "Зёрна и Хлопоты" }, { role: "tagline", en: "Roasted with love", ru: "Обжарено с любовью" }] },
        { ru: "Спортивный жирный", en: "Sport bold", v: "A bold athletic logo with a strong angular wordmark, a dynamic icon and high-energy contrast.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "dramatic", background_id: "dark_moody",
          objects: [{ role: "icon", desc: "a dynamic angular icon of a forward-charging stylized bolt fused with a sprinting figure, in stark black and vivid red with sharp italic motion lines" }],
          texts: [{ role: "wordmark", en: "APEX FORCE", ru: "АПЕКС ФОРС" }, { role: "tagline", en: "Push the limit", ru: "Бей рекорды" }] },
        { ru: "Бьюти / эстетика", en: "Beauty / elegant", v: "A delicate beauty logo with a thin elegant wordmark, a subtle floral mark and a soft refined palette.",
          style_preset_id: "editorial", aesthetics_id: "dreamy", lighting_id: "highkey", background_id: "studio_light",
          objects: [{ role: "icon", desc: "a subtle minimalist floral mark — a single thin-line blooming magnolia in soft rose-gold, drawn with delicate elegant strokes" }],
          texts: [{ role: "wordmark", en: "Florence", ru: "Флоранс" }, { role: "tagline", en: "Effortless glow", ru: "Естественное сияние" }] },
        { ru: "Минимал-иконка", en: "Minimal icon", v: "A minimal logo: one clever simple icon mark with a quiet wordmark and lots of negative space.",
          style_preset_id: "modern_clean", aesthetics_id: "calm", lighting_id: "highkey", background_id: "minimal",
          objects: [{ role: "icon", desc: "one clever minimal icon — a single continuous line forming a folded paper bird, using clean negative space, in solid charcoal" }],
          texts: [{ role: "wordmark", en: "fold", ru: "фолд" }, { role: "tagline", en: "Less, better", ru: "Меньше, но лучше" }] },
    ],
    music_cover: [
        { ru: "Атмосферный абстракт", en: "Atmospheric abstract", v: "A square album cover with striking abstract atmospheric art matching the genre mood, with the title and artist at the bottom.",
          style_preset_id: "editorial", aesthetics_id: "dreamy", lighting_id: "lowkey", background_id: "abstract_fluid",
          objects: [{ role: "art", desc: "A swirling abstract mass of liquid color, fluid teal and amber smoke twisting through space with glowing emissive threads of light cutting across the composition" }],
          texts: [{ role: "title", en: "DRIFT", ru: "ДРЕЙФ" }, { role: "artist", en: "Aurora Vale", ru: "Аврора Вейл" }] },
        { ru: "Портрет артистки", en: "Female artist portrait", v: "A moody album cover: an emotional close-up of a beautiful singer in dramatic light, with title and artist name below.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "dramatic", background_id: "studio_dark",
          objects: [{ role: "art", desc: "An intimate close-up of a beautiful young singer with eyes half-closed in feeling, dramatic chiaroscuro light sculpting her cheekbones and catching a single tear-bright highlight on her lip" }],
          texts: [{ role: "title", en: "Quiet Hours", ru: "Тихие часы" }, { role: "artist", en: "Mara Lune", ru: "Мара Люн" }] },
        { ru: "Брутальный рэп", en: "Hip-hop / bold man", v: "A bold hip-hop cover: a brutal confident man with an intense presence, gritty urban texture and heavy title type.",
          style_preset_id: "bold_commercial", aesthetics_id: "edgy", lighting_id: "lowkey", background_id: "grunge",
          objects: [{ role: "art", desc: "A brutal, muscular man in a black hoodie and heavy gold chains, jaw set and eyes locked on the viewer with unshakable confidence, scuffed urban grit clinging to his silhouette" }],
          texts: [{ role: "title", en: "NO MERCY", ru: "БЕЗ ПОЩАДЫ" }, { role: "artist", en: "BLOK", ru: "БЛОК" }] },
        { ru: "Неон-синтвейв", en: "Synthwave neon", v: "A retro synthwave cover with neon grids, a glowing sunset and chrome lettering in a nostalgic 80s mood.",
          style_preset_id: "synthwave_retro", aesthetics_id: "retro", lighting_id: "neon", background_id: "sunset_sky",
          objects: [{ role: "art", desc: "A chrome 1980s sports car silhouette gliding down the neon grid road, its mirror-bright body reflecting the pink sunset and the laser-cyan horizon lines" }],
          texts: [{ role: "title", en: "MIDNIGHT DRIVE", ru: "Полночный заезд" }, { role: "artist", en: "Neon Pulse", ru: "Неон Пульс" }] },
        { ru: "Лоу-фай уют", en: "Lo-fi cozy", v: "A cozy lo-fi cover: a calm illustrated scene with warm lamplight and rain and a soft nostalgic title.",
          style_preset_id: "handcrafted_artistic", aesthetics_id: "cozy", lighting_id: "golden", background_id: "dark_moody",
          objects: [{ role: "art", desc: "An illustrated girl in headphones resting her chin on her arms by the rainy window, a steaming mug and a sleeping cat beside her, bathed in soft nostalgic lamplight" }],
          texts: [{ role: "title", en: "Rainy Nights", ru: "Дождливые ночи" }, { role: "artist", en: "Lo Tide", ru: "Ло Тайд" }] },
        { ru: "Тёмный метал", en: "Dark metal", v: "A dark metal cover with dramatic ominous art, harsh textures and bold aggressive lettering.",
          style_preset_id: "surreal_concept", aesthetics_id: "cinematic", lighting_id: "lowkey", background_id: "dark_moody",
          objects: [{ role: "art", desc: "A monstrous horned skull wreathed in black smoke and cracked iron, eye sockets smoldering with ember-red light, rendered in harsh menacing texture" }],
          texts: [{ role: "title", en: "ASHEN THRONE", ru: "ПЕПЕЛЬНЫЙ ТРОН" }, { role: "artist", en: "Voidborne", ru: "Войдборн" }] },
        { ru: "Инди-акварель", en: "Indie watercolor", v: "A dreamy indie cover with a delicate watercolor illustration, soft pastels and a hand-lettered title.",
          style_preset_id: "botanical_watercolor", aesthetics_id: "dreamy", lighting_id: "highkey", background_id: "paper_craft",
          objects: [{ role: "art", desc: "A delicate watercolor illustration of a girl with flowers blooming from her hair, translucent pastel washes and fine ink line accents trailing into wildflowers and eucalyptus" }],
          texts: [{ role: "title", en: "Bloom Slowly", ru: "Цвети неспешно" }, { role: "artist", en: "Hazel & June", ru: "Хейзел и Джун" }] },
        { ru: "Танцевальный поп", en: "Dance pop", v: "A vibrant pop cover bursting with color and energy, a glamorous figure and a bold playful title.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "neon", background_id: "geometric",
          objects: [{ role: "art", desc: "A glamorous dancer mid-pose in a sequined outfit, arms thrown up in joy, glitter and confetti swirling around her against the riot of vibrant color" }],
          texts: [{ role: "title", en: "ALL NIGHT", ru: "ВСЮ НОЧЬ" }, { role: "artist", en: "Cici Star", ru: "Сиси Стар" }] },
        { ru: "Джаз-нуар", en: "Jazz noir", v: "A classy jazz cover with a smoky noir scene, a warm spotlight and elegant vintage typography.",
          style_preset_id: "retro_vintage", aesthetics_id: "luxe", lighting_id: "spotlight", background_id: "dark_moody",
          objects: [{ role: "art", desc: "A lone saxophonist silhouetted in the warm spotlight, brass instrument gleaming gold, head bowed into the solo as smoke drifts around the elegant noir scene" }],
          texts: [{ role: "title", en: "After Midnight", ru: "После полуночи" }, { role: "artist", en: "The Velvet Trio", ru: "Вельвет Трио" }] },
        { ru: "Эмбиент-минимал", en: "Ambient minimal", v: "A minimal ambient cover: a single serene gradient field, a tiny title and a calm meditative mood.",
          style_preset_id: "modern_clean", aesthetics_id: "calm", lighting_id: "highkey", background_id: "gradient",
          objects: [{ role: "art", desc: "One small luminous circle hovering low in the gradient field, its faint glow the only focal point in the vast serene emptiness" }],
          texts: [{ role: "title", en: "stillness", ru: "затишье" }, { role: "artist", en: "Halo", ru: "Хало" }] },
    ],
    product_card: [
        { ru: "Чистая студия", en: "Clean studio", v: "An e-commerce product card: the product centered on a seamless studio background, sharp focus, with name and a bold price.",
          style_preset_id: "photoreal_product", aesthetics_id: "calm", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "product", desc: "a single hero product standing perfectly centered, crisply lit with soft shadows and a gentle reflection on the glossy floor" }],
          texts: [{ role: "name", en: "PureLine Bottle", ru: "Бутылка PureLine" }, { role: "price", en: "$29", ru: "2 990 ₽" }] },
        { ru: "Косметика + модель", en: "Cosmetics + model", v: "A beauty product card: the cosmetic product with a beautiful model's glowing skin behind it in soft light, with name and price.",
          style_preset_id: "editorial", aesthetics_id: "luxe", lighting_id: "studio", background_id: "bokeh",
          objects: [{ role: "product", desc: "an elegant glass serum bottle with a gold dropper cap, catching soft highlights against the radiant skin behind it" }],
          texts: [{ role: "name", en: "Radiance Serum", ru: "Сыворотка «Сияние»" }, { role: "price", en: "$48", ru: "4 800 ₽" }] },
        { ru: "Гаджет / техно", en: "Gadget / tech", v: "A sleek tech product card: a gadget floating with a soft reflection on a dark gradient, with a clean name and price.",
          style_preset_id: "futuristic_tech", aesthetics_id: "tech", lighting_id: "neon", background_id: "studio_dark",
          objects: [{ role: "product", desc: "a sleek wireless earbuds case floating mid-air, brushed metal finish glinting with a cyan rim light and a soft reflection beneath" }],
          texts: [{ role: "name", en: "AeroBuds Pro", ru: "Наушники AeroBuds Pro" }, { role: "price", en: "$129", ru: "12 900 ₽" }] },
        { ru: "Еда / вкусно", en: "Tasty food", v: "A delicious food product card: the dish with fresh ingredients in appetizing light, with a bold name and price.",
          style_preset_id: "gourmet_food", aesthetics_id: "cozy", lighting_id: "golden", background_id: "nature_green",
          objects: [{ role: "product", desc: "a juicy gourmet burger with melting cheese and crisp greens, steam rising, fresh ingredients glistening around it" }],
          texts: [{ role: "name", en: "Classic Smash Burger", ru: "Бургер «Классика»" }, { role: "price", en: "$9.90", ru: "490 ₽" }] },
        { ru: "Мода / одежда", en: "Fashion item", v: "A stylish fashion product card: a clothing item on a clean backdrop with editorial flair, name and price.",
          style_preset_id: "editorial", aesthetics_id: "luxe", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "product", desc: "a tailored wool coat draped elegantly on an invisible form, fabric texture rich, posed with editorial confidence" }],
          texts: [{ role: "name", en: "Tailored Wool Coat", ru: "Шерстяное пальто" }, { role: "price", en: "$189", ru: "18 900 ₽" }] },
        { ru: "Эко / натурально", en: "Eco / natural", v: "A natural product card: the product among organic textures and greenery in warm earthy light, with name and price.",
          style_preset_id: "botanical_watercolor", aesthetics_id: "organic", lighting_id: "daylight", background_id: "paper_craft",
          objects: [{ role: "product", desc: "a natural skincare jar in frosted amber glass resting on smooth river stones, surrounded by fresh green leaves" }],
          texts: [{ role: "name", en: "Pure Botanic Balm", ru: "Натуральный бальзам" }, { role: "price", en: "$24", ru: "2 400 ₽" }] },
        { ru: "Премиум-люкс", en: "Premium luxury", v: "A luxury product card: the premium item in refined dramatic light on a dark field, with an elegant name and price.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "spotlight", background_id: "luxury_dark",
          objects: [{ role: "product", desc: "a faceted crystal perfume flacon with a polished gold cap, catching a sharp dramatic highlight against the dark luxurious backdrop" }],
          texts: [{ role: "name", en: "Noir Élégance", ru: "Noir Élégance" }, { role: "price", en: "$240", ru: "24 000 ₽" }] },
        { ru: "Скидка / акция", en: "On sale", v: "A promo product card: the product with a bold discount badge, an energetic accent and a striking sale price.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "studio", background_id: "geometric",
          objects: [{ role: "product", desc: "a popular sneaker shown at a punchy dynamic angle with a bold red discount starburst badge bursting beside it" }],
          texts: [{ role: "name", en: "Mega Sale!", ru: "Мега Распродажа!" }, { role: "price", en: "$49 -50%", ru: "4 900 ₽ −50%" }] },
        { ru: "Напиток / свежесть", en: "Drink / fresh", v: "A refreshing drink product card: the beverage with splashes and condensation in vivid fresh colors, with name and price.",
          style_preset_id: "gourmet_food", aesthetics_id: "bold", lighting_id: "studio", background_id: "gradient",
          objects: [{ role: "product", desc: "a chilled glass bottle of citrus soda covered in beaded condensation, a dynamic splash of juice frozen mid-air around it" }],
          texts: [{ role: "name", en: "Citrus Splash", ru: "Цитрусовый фреш" }, { role: "price", en: "$3.50", ru: "180 ₽" }] },
        { ru: "Хендмейд / крафт", en: "Handmade / craft", v: "A cozy handmade product card: the artisanal item on rustic textures in warm authentic light, with name and price.",
          style_preset_id: "handcrafted_artistic", aesthetics_id: "cozy", lighting_id: "golden", background_id: "paper_craft",
          objects: [{ role: "product", desc: "a hand-poured soy candle in a textured ceramic vessel with a tied twine label, wax surface softly catching the warm light" }],
          texts: [{ role: "name", en: "Handmade Candle", ru: "Свеча ручной работы" }, { role: "price", en: "$18", ru: "1 800 ₽" }] },
    ],
    packaging_label: [
        { ru: "Премиум-этикетка", en: "Premium label", v: "A premium packaging label: the brand on top, a refined bottle or box mockup, the product name and a small detail line.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "spotlight", background_id: "studio_dark",
          objects: [{ role: "product", desc: "An elegant frosted-glass bottle with a deep matte-black label, embossed gold foil border and a slim brushed-metal cap, standing in pristine focus" }],
          texts: [{ role: "brand", en: "AURELIA", ru: "АВРЕЛИЯ" }, { role: "product_name", en: "Signature Reserve", ru: "Особый резерв" }, { role: "detail", en: "Limited Edition · 500 ml", ru: "Лимитированная серия · 500 мл" }] },
        { ru: "Крафт / эко", en: "Kraft / eco", v: "An eco packaging label with natural kraft textures, hand-drawn botanical accents and warm organic typography.",
          style_preset_id: "handcrafted_artistic", aesthetics_id: "organic", lighting_id: "daylight", background_id: "paper_craft",
          objects: [{ role: "product", desc: "A rustic brown kraft pouch with a kraft-paper label, hand-inked botanical line drawings of herbs and a natural twine tie" }],
          texts: [{ role: "brand", en: "Greenroot", ru: "Зелёный корень" }, { role: "product_name", en: "Wild Herb Blend", ru: "Дикие травы" }, { role: "detail", en: "100% Natural · Handmade", ru: "100% натурально · ручная работа" }] },
        { ru: "Косметика-люкс", en: "Luxury cosmetics", v: "A luxury cosmetics label: an elegant jar or bottle in soft light, a refined serif brand and a delicate palette.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "product", desc: "A pearlescent ivory cosmetic jar with a polished rose-gold lid and a glossy cream label in soft serif type, catching gentle highlights" }],
          texts: [{ role: "brand", en: "Lumière", ru: "Люмьер" }, { role: "product_name", en: "Radiance Cream", ru: "Крем сияния" }, { role: "detail", en: "Day & Night · 50 ml", ru: "День и ночь · 50 мл" }] },
        { ru: "Крафтовое пиво", en: "Craft beer", v: "A bold craft-beer label with a striking illustrated emblem, vintage lettering and a punchy color scheme.",
          style_preset_id: "retro_vintage", aesthetics_id: "retro", lighting_id: "lowkey", background_id: "brick_wall",
          objects: [{ role: "product", desc: "An amber glass beer bottle with a bold illustrated label featuring a roaring stag emblem, ornate vintage lettering and crackled retro texture, condensation beading on the glass" }],
          texts: [{ role: "brand", en: "Iron Stag", ru: "Железный олень" }, { role: "product_name", en: "Hazy IPA", ru: "Мутный IPA" }, { role: "detail", en: "Brewed Small · 6.5% ABV", ru: "Малая партия · 6,5%" }] },
        { ru: "Кофе / зерно", en: "Coffee pack", v: "A warm coffee packaging label with a rich illustrated mark, cozy earthy tones and confident brand type.",
          style_preset_id: "handcrafted_artistic", aesthetics_id: "cozy", lighting_id: "golden", background_id: "dark_moody",
          objects: [{ role: "product", desc: "A matte dark-brown coffee bag with a kraft label, a richly illustrated mountain-and-bean emblem, a sealed valve and confident hand-lettered type" }],
          texts: [{ role: "brand", en: "Ember Roasters", ru: "Угольная обжарка" }, { role: "product_name", en: "Dark Roast", ru: "Тёмная обжарка" }, { role: "detail", en: "Whole Bean · 250 g", ru: "Зерно · 250 г" }] },
        { ru: "Снек / яркий", en: "Snack / bold", v: "A fun snack package label bursting with appetizing color, bold playful type and an energetic mascot.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "daylight", background_id: "geometric",
          objects: [{ role: "product", desc: "A glossy bright-orange snack bag bursting with color, featuring a cheeky grinning chili mascot, bold rounded type and a tempting pile of crisps spilling from the open top" }],
          texts: [{ role: "brand", en: "Crunch Buddies", ru: "Хрустики" }, { role: "product_name", en: "Spicy Crisps", ru: "Острые чипсы" }, { role: "detail", en: "Extra Crunchy · 120 g", ru: "Супер-хруст · 120 г" }] },
        { ru: "Парфюм / минимал", en: "Perfume minimal", v: "A minimal perfume label: an elegant flacon on a soft field, thin refined typography and a luxurious calm.",
          style_preset_id: "modern_clean", aesthetics_id: "calm", lighting_id: "highkey", background_id: "minimal",
          objects: [{ role: "product", desc: "A slim rectangular clear-glass perfume flacon holding pale-amber liquid, a thin silver collar and a minimal frosted label, standing solitary in soft light" }],
          texts: [{ role: "brand", en: "NOIR", ru: "НУАР" }, { role: "product_name", en: "Eau de Parfum", ru: "Парфюмерная вода" }, { role: "detail", en: "No. 7 · 75 ml", ru: "№ 7 · 75 мл" }] },
        { ru: "Фарма / чистый", en: "Pharma / clean", v: "A clean clinical product label: a trustworthy bottle, precise legible typography and a fresh medical palette.",
          style_preset_id: "modern_clean", aesthetics_id: "corporate", lighting_id: "product", background_id: "studio_light",
          objects: [{ role: "product", desc: "A clean white pharmaceutical bottle with a glossy white-and-blue label, a precise dosage panel and a tamper-evident cap, lit evenly with sterile clarity" }],
          texts: [{ role: "brand", en: "MediCare", ru: "МедиКэр" }, { role: "product_name", en: "Vitamin D3", ru: "Витамин D3" }, { role: "detail", en: "60 Tablets · 2000 IU", ru: "60 таблеток · 2000 МЕ" }] },
        { ru: "Винтаж / аптека", en: "Vintage apothecary", v: "A vintage apothecary label with ornate frames, classic serif lettering and an aged refined texture.",
          style_preset_id: "retro_vintage", aesthetics_id: "retro", lighting_id: "lowkey", background_id: "luxury_dark",
          objects: [{ role: "product", desc: "An amber apothecary bottle with a cork stopper and an aged cream label framed by ornate Victorian borders, classic serif lettering and a faded engraved emblem" }],
          texts: [{ role: "brand", en: "Apothecary No. 1", ru: "Аптека №1" }, { role: "product_name", en: "Herbal Tincture", ru: "Травяная настойка" }, { role: "detail", en: "Est. 1898 · 100 ml", ru: "Основано в 1898 · 100 мл" }] },
        { ru: "Спорт-питание", en: "Sports nutrition", v: "A bold sports-nutrition label: a powerful container with dynamic accents, strong type and high-energy contrast.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "dramatic", background_id: "studio_dark",
          objects: [{ role: "product", desc: "A muscular matte-black protein tub with aggressive electric-lime accents, sharp angular graphics, bold stenciled type and a chrome screw lid, lit with high contrast" }],
          texts: [{ role: "brand", en: "TITAN FUEL", ru: "ТИТАН ФЬЮЛ" }, { role: "product_name", en: "Whey Protein", ru: "Сывороточный протеин" }, { role: "detail", en: "30g Protein · 2 kg", ru: "30 г белка · 2 кг" }] },
    ],
    merch_print: [
        { ru: "Дерзкий слоган", en: "Bold slogan", v: "A bold merch print with a big punchy slogan, a striking central graphic and a small accent line in high contrast.",
          style_preset_id: "streetwear_vector", aesthetics_id: "bold", lighting_id: "dramatic", background_id: "solid",
          objects: [{ role: "illustration", desc: "A bold vector lightning bolt smashing through a cracked circular emblem, rendered in crisp monochrome with one electric-yellow accent and thick confident outlines." }],
          texts: [{ role: "slogan", en: "STAY BOLD", ru: "БУДЬ ДЕРЗКИМ" }, { role: "accent", en: "no limits", ru: "без границ" }] },
        { ru: "Брутальный маскот", en: "Bold mascot", v: "A high-contrast t-shirt print with a fierce brutal mascot character, heavy lettering and an edgy streetwear vibe.",
          style_preset_id: "streetwear_vector", aesthetics_id: "edgy", lighting_id: "lowkey", background_id: "grunge",
          objects: [{ role: "illustration", desc: "A snarling bulldog mascot with bared teeth and a spiked collar, drawn as sharp vector shapes in stark black-and-white with a single blood-red accent." }],
          texts: [{ role: "slogan", en: "UNLEASHED", ru: "СПУЩЕН С ЦЕПИ" }, { role: "accent", en: "est. street", ru: "рождён на улице" }] },
        { ru: "Девушка / поп-арт", en: "Pop-art girl", v: "A pop-art merch print with a striking stylish woman illustration, bold outlines, halftone dots and punchy colors.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "studio", background_id: "geometric",
          objects: [{ role: "illustration", desc: "A confident stylish woman in cat-eye sunglasses winking, rendered in bold pop-art outlines, Ben-Day halftone dots and punchy pink, cyan and yellow color blocks." }],
          texts: [{ role: "slogan", en: "WOW!", ru: "ВАУ!" }, { role: "accent", en: "be iconic", ru: "будь иконой" }] },
        { ru: "Скейт / стрит", en: "Skate / street", v: "A gritty street-style print with a rebellious graphic, distressed textures and a bold slogan.",
          style_preset_id: "streetwear_vector", aesthetics_id: "edgy", lighting_id: "dramatic", background_id: "grunge",
          objects: [{ role: "illustration", desc: "A worn skateboard deck snapped in half mid-air with sparks flying, drawn in raw monochrome vector with a single neon-green accent and distressed grunge edges." }],
          texts: [{ role: "slogan", en: "SKATE OR DIE", ru: "КАТАЙ ИЛИ ПАДАЙ" }, { role: "accent", en: "no brakes", ru: "без тормозов" }] },
        { ru: "Природа / горы", en: "Outdoor / mountains", v: "An outdoor adventure print with a scenic mountain illustration, vintage badge lettering and an earthy palette.",
          style_preset_id: "retro_vintage", aesthetics_id: "retro", lighting_id: "golden", background_id: "paper_craft",
          objects: [{ role: "illustration", desc: "A circular vintage adventure badge framing snow-capped pine-covered peaks under a setting sun, rendered in muted forest-green, burnt-orange and cream with retro line work." }],
          texts: [{ role: "slogan", en: "INTO THE WILD", ru: "НАВСТРЕЧУ ДИКОЙ ПРИРОДЕ" }, { role: "accent", en: "explore more", ru: "исследуй больше" }] },
        { ru: "Аниме-вайб", en: "Anime vibe", v: "An anime-style merch print with an expressive character, clean cel-shaded art and a bold catchphrase.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "backlit", background_id: "gradient",
          objects: [{ role: "illustration", desc: "An expressive spiky-haired anime hero mid-shout with a determined grin and a glowing energy aura, rendered in clean cel-shaded art with crisp outlines and vibrant colors." }],
          texts: [{ role: "slogan", en: "NEVER GIVE UP", ru: "НИКОГДА НЕ СДАВАЙСЯ" }, { role: "accent", en: "full power", ru: "на полную" }] },
        { ru: "Готика / тёмный", en: "Dark gothic", v: "A dark gothic print with an intricate ominous illustration, sharp blackletter type and a moody palette.",
          style_preset_id: "handcrafted_artistic", aesthetics_id: "cinematic", lighting_id: "lowkey", background_id: "dark_moody",
          objects: [{ role: "illustration", desc: "An ornate horned skull crowned with thorny roses and intertwined serpents, drawn in intricate fine black line work with cold silver highlights and deep crimson shadows." }],
          texts: [{ role: "slogan", en: "MEMENTO MORI", ru: "ПОМНИ О СМЕРТИ" }, { role: "accent", en: "forever dark", ru: "вечная тьма" }] },
        { ru: "Юмор / мем", en: "Funny meme", v: "A funny merch print with a quirky humorous character, a witty slogan and bright cheeky colors.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "highkey", background_id: "solid",
          objects: [{ role: "illustration", desc: "A goofy cartoon cat slumped over a coffee mug with bloodshot tired eyes and a deadpan expression, drawn in bright friendly flat illustration with thick comedic outlines." }],
          texts: [{ role: "slogan", en: "NOT TODAY", ru: "НЕ СЕГОДНЯ" }, { role: "accent", en: "need coffee", ru: "нужен кофе" }] },
        { ru: "Ретро 80-е", en: "Retro 80s", v: "A retro 80s print with a neon sunset, bold chrome lettering and a nostalgic synthwave mood.",
          style_preset_id: "synthwave_retro", aesthetics_id: "retro", lighting_id: "neon", background_id: "sunset_sky",
          objects: [{ role: "illustration", desc: "A retro-futuristic palm-lined horizon with a giant banded neon sun and a chrome sports car gliding along the glowing grid into the sunset haze." }],
          texts: [{ role: "slogan", en: "RETRO WAVE", ru: "РЕТРО ВОЛНА" }, { role: "accent", en: "since '85", ru: "с 85-го" }] },
        { ru: "Минимал-лайн", en: "Minimal line art", v: "A minimal line-art merch print: a single elegant continuous-line illustration and a small refined slogan.",
          style_preset_id: "modern_clean", aesthetics_id: "calm", lighting_id: "highkey", background_id: "minimal",
          objects: [{ role: "illustration", desc: "A serene face in profile drawn as one elegant unbroken continuous line, minimal and refined, in a single thin charcoal stroke." }],
          texts: [{ role: "slogan", en: "less is more", ru: "меньше значит больше" }, { role: "accent", en: "one line", ru: "одна линия" }] },
    ],
    material_typography: [
        { ru: "Ледяные буквы", en: "Frozen Ice", v: "A single giant short word fills the entire frame, its letters carved from translucent blue glacial ice with frost-feathered surfaces, sharp crystalline edges and tiny trapped air bubbles, faint cold mist curling at the bases, set against a clean dark teal void lit with a soft cold rim light.",
          style_preset_id: "studio_3d_render", aesthetics_id: "tech", lighting_id: "backlit", background_id: "dark_moody",
          objects: [],
          texts: [{ role: "hero", en: "FROST", ru: "ХОЛОД" }, { role: "caption", en: "Carved from glacial ice", ru: "Высечено изо льда" }] },
        { ru: "Сочные фрукты", en: "Ripe Fruit", v: "One bold short headline towers across the frame with each letter sculpted from fresh ripe fruit — glossy juicy strawberries, citrus segments and dewy berries pressed into the strokes — droplets of juice beading on the surfaces against a clean creamy backdrop lit like a bright food commercial.",
          style_preset_id: "gourmet_food", aesthetics_id: "playful", lighting_id: "product", background_id: "studio_light",
          objects: [],
          texts: [{ role: "hero", en: "JUICY", ru: "СОЧНО" }, { role: "caption", en: "Made of fresh ripe fruit", ru: "Из спелых фруктов" }] },
        { ru: "Расплавленное золото", en: "Molten Gold", v: "A massive short word dominates the frame rendered in flowing molten gold, mirror-bright liquid metal dripping and pooling along the letterforms with rich warm highlights and deep amber shadows, floating in a minimal jet-black studio space.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "lowkey", background_id: "luxury_dark",
          objects: [],
          texts: [{ role: "hero", en: "GOLD", ru: "ЗОЛОТО" }, { role: "caption", en: "Pure molten luxury", ru: "Чистая роскошь" }] },
        { ru: "Неоновое стекло", en: "Neon Glass", v: "One short punchy word fills the frame as glowing neon glass tubes bent into the letters, electric magenta and cyan light blooming with soft halos and gentle reflections on a dark wet floor against an otherwise empty midnight backdrop.",
          style_preset_id: "neon_signage", aesthetics_id: "tech", lighting_id: "neon", background_id: "glow_mesh",
          objects: [],
          texts: [{ role: "hero", en: "GLOW", ru: "СВЕЧЕНИЕ" }, { role: "caption", en: "Hand-bent neon glass", ru: "Гнутое неоновое стекло" }] },
        { ru: "Прозрачное стекло", en: "Clear Glass", v: "A giant short headline spans the frame built from thick transparent glass, the letters refracting and bending the light behind them with caustic glints and crisp bevelled edges, resting on a pale seamless gradient with delicate soft shadows.",
          style_preset_id: "modern_clean", aesthetics_id: "calm", lighting_id: "studio", background_id: "gradient",
          objects: [],
          texts: [{ role: "hero", en: "CLEAR", ru: "ЯСНО" }, { role: "caption", en: "Pure transparent glass", ru: "Прозрачное стекло" }] },
        { ru: "Цветущие буквы", en: "Blooming Flowers", v: "One large short word fills the frame formed entirely from blooming flowers and lush green foliage, dense rose and peony petals packed into each letter with tiny leaves spilling at the edges, photographed against a soft pastel sky.",
          style_preset_id: "botanical_watercolor", aesthetics_id: "dreamy", lighting_id: "highkey", background_id: "glow_mesh",
          objects: [],
          texts: [{ role: "hero", en: "BLOOM", ru: "ЦВЕТЕНИЕ" }, { role: "caption", en: "Petals in full bloom", ru: "Лепестки в цвету" }] },
        { ru: "Пылающий огонь", en: "Blazing Fire", v: "A bold short word commands the frame sculpted from blazing fire and glowing embers, flames licking up the letterforms with bright orange cores fading to smoky edges and sparks drifting into a deep black night.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "lowkey", background_id: "dark_moody",
          objects: [],
          texts: [{ role: "hero", en: "BLAZE", ru: "ПЛАМЯ" }, { role: "caption", en: "Forged in living fire", ru: "Рождено в огне" }] },
        { ru: "Хром и жидкий металл", en: "Liquid Chrome", v: "One giant short headline fills the frame as flawless mirror chrome, the liquid-metal letters reflecting a soft studio gradient with smooth rounded curves and razor-sharp specular highlights, isolated on a minimal pearl-grey backdrop.",
          style_preset_id: "futuristic_tech", aesthetics_id: "tech", lighting_id: "studio", background_id: "gradient",
          objects: [],
          texts: [{ role: "hero", en: "CHROME", ru: "ХРОМ" }, { role: "caption", en: "Liquid mirror metal", ru: "Жидкий металл" }] },
        { ru: "Брызги воды", en: "Water Splash", v: "A massive short word spans the frame shaped from a splash of crystal-clear water, the letters frozen mid-motion with translucent flowing strokes, scattering droplets and glassy highlights against a fresh aqua-blue void.",
          style_preset_id: "photoreal_product", aesthetics_id: "bold", lighting_id: "studio", background_id: "solid",
          objects: [],
          texts: [{ role: "hero", en: "SPLASH", ru: "БРЫЗГИ" }, { role: "caption", en: "Frozen mid-splash", ru: "Застывшая вода" }] },
        { ru: "Мягкий пушистый мех", en: "Fluffy Fur", v: "One short cuddly headline fills the frame rendered in soft fluffy fur, each letter a plush mound of fine warm-toned hairs catching gentle light with a cozy depth-of-field blur, set on a clean muted beige background.",
          style_preset_id: "playful_fun", aesthetics_id: "cozy", lighting_id: "studio", background_id: "bokeh",
          objects: [],
          texts: [{ role: "hero", en: "FLUFFY", ru: "ПУШИСТО" }, { role: "caption", en: "Soft and cuddly", ru: "Мягко и уютно" }] },
    ],
    neon_sign: [
        { ru: "Открыто — розовый неон", en: "OPEN — pink neon", v: "A glowing pink neon-tube sign spelling the word OPEN in flowing cursive script, buzzing softly against a dark weathered red-brick wall at night, its rosy light bleeding warm halos across the mortar and casting a faint reflection in a wet sidewalk below.",
          style_preset_id: "neon_signage", aesthetics_id: "cozy", lighting_id: "neon", background_id: "brick_wall",
          objects: [{ role: "icon", desc: "a small hand-bent pink neon underline curl beneath the script, its glass tube glowing hot magenta-pink with a faint warm halo" }],
          texts: [{ role: "hero", en: "OPEN", ru: "ОТКРЫТО" }, { role: "subtitle", en: "Come on in", ru: "Заходите" }] },
        { ru: "Бар — синяя вывеска", en: "BAR — electric blue", v: "An electric-blue neon sign reading BAR in bold blocky tubes, humming above a shadowed brick alley wall, cold cobalt light spilling across the rough masonry while a thin coil of cigarette smoke drifts through the glow.",
          style_preset_id: "neon_signage", aesthetics_id: "edgy", lighting_id: "neon", background_id: "brick_wall",
          objects: [{ role: "icon", desc: "a glowing electric-blue neon cocktail-glass icon beside the letters, its hand-bent glass tube buzzing cold cobalt against the dark brick" }],
          texts: [{ role: "hero", en: "BAR", ru: "БАР" }, { role: "subtitle", en: "Open till late", ru: "Открыто допоздна" }] },
        { ru: "Открыто 24 часа", en: "OPEN 24 HOURS", v: "A vintage diner neon sign reading OPEN 24 HOURS, the words stacked in warm amber and ruby tubes with a small glowing arrow beneath, mounted on a grimy dark-brick facade slick with light drizzle that scatters the colored light into soft streaks.",
          style_preset_id: "neon_signage", aesthetics_id: "retro", lighting_id: "neon", background_id: "brick_wall",
          objects: [{ role: "icon", desc: "a small glowing ruby-red neon arrow pointing down beneath the words, its bent glass tube buzzing warmly against the wet brick" }],
          texts: [{ role: "hero", en: "OPEN 24 HOURS", ru: "ОТКРЫТО 24 ЧАСА" }, { role: "subtitle", en: "Always serving", ru: "Всегда работаем" }] },
        { ru: "Розовое сердце", en: "Glowing heart", v: "A single oversized neon heart outlined in hot magenta tubes pulsing on a moody charcoal-brick wall, its pink radiance pooling into a soft circular bloom on the bricks while one tube flickers as if on the edge of burning out.",
          style_preset_id: "neon_signage", aesthetics_id: "bold", lighting_id: "neon", background_id: "brick_wall",
          objects: [{ role: "icon", desc: "an oversized hand-bent neon heart outlined in hot magenta glass tubes, pulsing and glowing with one tube flickering on the edge of burning out" }],
          texts: [{ role: "hero", en: "LOVE", ru: "ЛЮБОВЬ" }, { role: "subtitle", en: "Always", ru: "Навсегда" }] },
        { ru: "Коктейли — мятный неон", en: "COCKTAILS — mint", v: "A retro neon sign spelling COCKTAILS in elegant minty-green script tubes beside a tilted glowing martini-glass icon, glowing on a dim textured brick wall, its cool emerald light catching every crack and chip in the old painted bricks.",
          style_preset_id: "neon_signage", aesthetics_id: "retro", lighting_id: "neon", background_id: "brick_wall",
          objects: [{ role: "icon", desc: "a tilted glowing martini-glass icon in minty-green neon glass tubes, its olive depicted as a tiny ruby-red bulb against the dark brick" }],
          texts: [{ role: "hero", en: "COCKTAILS", ru: "КОКТЕЙЛИ" }, { role: "subtitle", en: "Happy hour 6-9", ru: "Счастливый час 18:00–21:00" }] },
        { ru: "Закрыто — красный", en: "CLOSED — red", v: "A tired red neon sign reading CLOSED in slumping handwritten tubes, half its glow dimmed and one letter dark, fixed to a cold rain-darkened brick wall at night with a lonely crimson reflection trembling in a puddle below.",
          style_preset_id: "neon_signage", aesthetics_id: "cinematic", lighting_id: "lowkey", background_id: "brick_wall",
          objects: [{ role: "icon", desc: "a dim red neon dash beneath the word, one segment of its glass tube gone dark and unlit against the rain-soaked brick" }],
          texts: [{ role: "hero", en: "CLOSED", ru: "ЗАКРЫТО" }, { role: "subtitle", en: "See you tomorrow", ru: "До завтра" }] },
        { ru: "Стрелка налево", en: "Neon arrow", v: "A bright yellow-orange neon arrow pointing left, its chasing tube segments suggesting motion, mounted on a dark soot-stained brick wall in a narrow night alley where the warm light rakes sharply across the rough brick texture.",
          style_preset_id: "neon_signage", aesthetics_id: "bold", lighting_id: "neon", background_id: "brick_wall",
          objects: [{ role: "icon", desc: "a bright yellow-orange neon arrow pointing left, built from chasing hand-bent glass tube segments that suggest sequential motion" }],
          texts: [{ role: "hero", en: "THIS WAY", ru: "СЮДА" }, { role: "subtitle", en: "Entrance around the corner", ru: "Вход за углом" }] },
        { ru: "Мечтай — фиолетовый", en: "DREAM — purple", v: "A dreamy violet-and-pink neon sign spelling DREAM in soft rounded cursive tubes, glowing gently against a deep indigo-shadowed brick wall, the dual-tone light blending into a hazy lavender halo that softens the gritty masonry behind it.",
          style_preset_id: "neon_signage", aesthetics_id: "dreamy", lighting_id: "neon", background_id: "brick_wall",
          objects: [{ role: "icon", desc: "a small glowing neon crescent-moon icon in soft violet glass tubes beside the word, blending into the lavender halo against the dark brick" }],
          texts: [{ role: "hero", en: "DREAM", ru: "МЕЧТАЙ" }, { role: "subtitle", en: "Make it real", ru: "Воплоти в жизнь" }] },
        { ru: "Кофе — оранжевый", en: "COFFEE — amber", v: "A cozy amber neon sign reading COFFEE in warm hand-script tubes above a tiny glowing steaming-cup icon, mounted on a dark espresso-brown brick wall at night, its honeyed light wrapping the bricks in an inviting golden warmth.",
          style_preset_id: "neon_signage", aesthetics_id: "cozy", lighting_id: "neon", background_id: "brick_wall",
          objects: [{ role: "icon", desc: "a tiny glowing steaming-cup icon in warm amber neon glass tubes, with two delicate curling steam lines rising above the brimming cup" }],
          texts: [{ role: "hero", en: "COFFEE", ru: "КОФЕ" }, { role: "subtitle", en: "Freshly brewed", ru: "Свежесваренный" }] },
        { ru: "Сломанный отель", en: "Flickering HOTEL", v: "A weathered turquoise neon sign reading HOTEL in tall narrow tubes, one letter buzzing and stuttering with a dying flicker, bolted to a grimy noir brick wall at night where the unsteady teal glow flares and fades across the damp stone.",
          style_preset_id: "neon_signage", aesthetics_id: "cinematic", lighting_id: "lowkey", background_id: "brick_wall",
          objects: [{ role: "icon", desc: "a tall narrow turquoise neon vacancy-bar beneath the word, its glass tube buzzing and stuttering with a dying half-lit flicker" }],
          texts: [{ role: "hero", en: "HOTEL", ru: "ОТЕЛЬ" }, { role: "subtitle", en: "Vacancy", ru: "Есть места" }] },
    ],
    food_typography_ad: [
        { ru: "Арбузный заголовок", en: "Watermelon Headline", v: "A bold summer headline spelling 'JUICY' built from glistening watermelon flesh studded with glossy black seeds and dripping pink juice, beside a tall sweating glass of watermelon cooler and a crisp price tag, on a sun-warmed coral backdrop scattered with chunks of green rind.",
          style_preset_id: "gourmet_food", aesthetics_id: "bold", lighting_id: "daylight", background_id: "studio_light",
          objects: [{ role: "product", desc: "A tall frosted glass of watermelon cooler sweating with cold condensation, packed with crushed ice and a glossy wedge of seed-studded watermelon perched on the rim, droplets running down its sides." }],
          texts: [{ role: "headline", en: "JUICY", ru: "СОЧНО" }, { role: "price", en: "ONLY $4.99", ru: "ВСЕГО 299 ₽" }] },
        { ru: "Цитрусовый взрыв", en: "Citrus Splash", v: "A zesty headline reading 'FRESH' formed from vivid orange and lemon segments bursting with a fine citrus mist, surrounded by a chilled bottle of cold-pressed juice and a clean price callout, against a bright tangerine background flecked with droplets and glossy green leaves.",
          style_preset_id: "gourmet_food", aesthetics_id: "bold", lighting_id: "daylight", background_id: "studio_light",
          objects: [{ role: "product", desc: "A chilled clear bottle of cold-pressed citrus juice glowing vivid orange, beaded with condensation, flanked by halved oranges and lemons bursting with a fine spritz of zest." }],
          texts: [{ role: "headline", en: "FRESH", ru: "СВЕЖО" }, { role: "price", en: "JUST $3.49", ru: "ОТ 199 ₽" }] },
        { ru: "Ягодный десерт", en: "Berry Indulgence", v: "A tempting headline spelling 'SWEET' sculpted from plump strawberries, blueberries and raspberries glazed with a sheen of cream, next to a layered berry parfait in a tall glass and a clean buy-now button, on a soft blush-pink studio surface dusted with fine sugar.",
          style_preset_id: "gourmet_food", aesthetics_id: "dreamy", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "product", desc: "A tall layered berry parfait in a clear glass, alternating bands of whipped cream, glossy strawberries, blueberries and raspberries, topped with a swirl of cream and a single mint sprig." }],
          texts: [{ role: "headline", en: "SWEET", ru: "СЛАДКО" }, { role: "price", en: "BUY NOW $5.90", ru: "КУПИТЬ 349 ₽" }] },
        { ru: "Тропический микс", en: "Tropical Mix", v: "An exotic headline reading 'PARADISE' built from sliced mango, kiwi, dragonfruit and pineapple wedges glistening under bright sun, paired with a frosty smoothie bowl and a bold price banner, on a turquoise backdrop framed by monstera leaves and clinging water beads.",
          style_preset_id: "gourmet_food", aesthetics_id: "playful", lighting_id: "daylight", background_id: "nature_green",
          objects: [{ role: "product", desc: "A frosty tropical smoothie bowl brimming with sliced mango, kiwi, dragonfruit and pineapple wedges, drizzled with coconut and dusted with seeds, glistening under bright sun." }],
          texts: [{ role: "headline", en: "PARADISE", ru: "РАЙ" }, { role: "price", en: "$6.50", ru: "399 ₽" }] },
        { ru: "Хрустящее зелёное яблоко", en: "Crisp Green Apple", v: "A snappy headline spelling 'CRISP' carved from crunchy green apple slices with bright dewy skin and a tart spritz of mist, beside a whole mirror-shiny apple and a chalkboard-style price, on a cool mint backdrop with crystalline droplets catching the light.",
          style_preset_id: "gourmet_food", aesthetics_id: "calm", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "product", desc: "A whole mirror-shiny green apple with dewy taut skin, flanked by crunchy fresh-cut apple slices spritzed with a tart cooling mist, droplets gleaming on every surface." }],
          texts: [{ role: "headline", en: "CRISP", ru: "ХРУСТ" }, { role: "price", en: "$2.99 / KG", ru: "149 ₽ / КГ" }] },
        { ru: "Виноградная роскошь", en: "Grape Luxe", v: "An elegant headline reading 'PURE' formed from dewy clusters of deep-purple and emerald grapes wearing a frosted bloom, alongside a premium bottle of grape nectar and a refined price plate, on a moody plum backdrop bathed in soft vineyard light.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "golden", background_id: "luxury_dark",
          objects: [{ role: "product", desc: "A premium tall glass bottle of grape nectar with an elegant minimalist label, deep-purple liquid glowing within, set beside dewy frosted clusters of purple and emerald grapes." }],
          texts: [{ role: "headline", en: "PURE", ru: "ЧИСТЫЙ ВКУС" }, { role: "price", en: "$12.00", ru: "790 ₽" }] },
        { ru: "Гранатовая энергия", en: "Pomegranate Power", v: "A vibrant headline spelling 'BOOST' built from glossy ruby pomegranate arils glistening like cut jewels with a few crimson splashes, next to a cracked-open pomegranate half and an energetic price burst, on a deep garnet backdrop streaked with running juice.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "dramatic", background_id: "dark_moody",
          objects: [{ role: "product", desc: "A cracked-open pomegranate half overflowing with glossy ruby arils glistening like cut jewels, surrounded by a few dynamic crimson juice splashes frozen mid-air." }],
          texts: [{ role: "headline", en: "BOOST", ru: "ЗАРЯД" }, { role: "price", en: "SALE $4.20", ru: "АКЦИЯ 259 ₽" }] },
        { ru: "Утренний завтрак", en: "Breakfast Fresh", v: "A cheerful headline reading 'MORNING' assembled from banana coins, peach slices and bright berries swirled into creamy yogurt, beside a wholesome granola bowl and a friendly price sticker, on a warm cream tabletop bathed in soft sunrise light.",
          style_preset_id: "gourmet_food", aesthetics_id: "cozy", lighting_id: "golden", background_id: "studio_light",
          objects: [{ role: "product", desc: "A wholesome granola breakfast bowl with creamy yogurt swirled with banana coins, peach slices and bright berries, scattered with crunchy clusters and a drizzle of honey." }],
          texts: [{ role: "headline", en: "MORNING", ru: "УТРО" }, { role: "price", en: "$3.90", ru: "229 ₽" }] },
        { ru: "Манговое лето", en: "Mango Season", v: "A juicy headline spelling 'RIPE' sculpted from golden mango cubes oozing nectar with velvety skin highlights, paired with a creamy mango lassi and a summery price tag, on a saffron-yellow backdrop scattered with mint leaves and sticky droplets.",
          style_preset_id: "gourmet_food", aesthetics_id: "bold", lighting_id: "daylight", background_id: "solid",
          objects: [{ role: "product", desc: "A tall creamy mango lassi in a frosted glass, thick golden swirl topped with diced ripe mango cubes oozing nectar and a sprig of mint, condensation beading on the glass." }],
          texts: [{ role: "headline", en: "RIPE", ru: "СПЕЛО" }, { role: "price", en: "$4.50", ru: "269 ₽" }] },
        { ru: "Вишнёвый соблазн", en: "Cherry Temptation", v: "A playful headline reading 'YUM' formed from glossy red cherries on slender stems with mirror-bright skins and tiny clinging drips, beside a swirl of cherry sorbet in a cone and a bold sale price, on a candy-red backdrop sparkling with sugar specks.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "studio", background_id: "solid",
          objects: [{ role: "product", desc: "A swirl of glossy cherry sorbet piled high in a crisp waffle cone, topped with a pair of mirror-bright red cherries on slender stems with tiny drips clinging to their skins." }],
          texts: [{ role: "headline", en: "YUM", ru: "ВКУСНЯТИНА" }, { role: "price", en: "SALE $2.50", ru: "АКЦИЯ 159 ₽" }] },
    ],
    logo_emblem: [
        { ru: "Кофейня — горный обжарщик", en: "Mountain Coffee Roastery", v: "A vintage circular coffee-roastery badge on a warm cream background, curved top text \"MOUNTAIN ROAST\" and bottom text \"EST. 1974\", a central engraved emblem of a steaming coffee cup framed by twin mountain peaks, double pinstripe rings and tiny star separators, rendered in deep espresso brown and burnt amber #3B2417 and #C8843A with subtle aged-ink texture.",
          style_preset_id: "retro_vintage", aesthetics_id: "retro", lighting_id: "studio", background_id: "paper_craft",
          objects: [{ role: "icon", desc: "A finely engraved emblem of a steaming coffee cup cradled between twin mountain peaks, rendered in deep espresso brown and burnt amber with double pinstripe rings and tiny star separators." }],
          texts: [{ role: "title", en: "MOUNTAIN ROAST", ru: "ГОРНАЯ ОБЖАРКА" }, { role: "tagline", en: "EST. 1974", ru: "С 1974 ГОДА" }] },
        { ru: "Крафтовая пивоварня", en: "Craft Brewery Seal", v: "A bold craft-brewery emblem on a deep navy background, arched top text \"IRON HOPS\" and lower banner \"BREWING CO.\", a centred icon of crossed wheat sheaves over a foaming beer barrel, surrounded by a beaded ring and hop-leaf flourishes, painted in antique gold and oxblood red #D4A017 and #7A1F1F with a hand-stamped letterpress feel.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "icon", desc: "A centred icon of crossed wheat sheaves arching over a foaming beer barrel, ringed by a beaded border and hop-leaf flourishes in antique gold and oxblood red." }],
          texts: [{ role: "title", en: "IRON HOPS", ru: "ЖЕЛЕЗНЫЙ ХМЕЛЬ" }, { role: "tagline", en: "BREWING CO.", ru: "ПИВОВАРНЯ" }] },
        { ru: "Барбершоп для джентльменов", en: "Barbershop Emblem", v: "A classic barbershop badge on a charcoal background, curved top text \"SHARP & CO.\" and bottom text \"GROOMING\", a central icon of crossed straight razor and comb beneath a striped barber pole, encircled by a fine rope border, rendered in ivory and brushed steel blue #EDE6D6 and #5C7A99 with a crisp engraved line style.",
          style_preset_id: "editorial", aesthetics_id: "corporate", lighting_id: "highkey", background_id: "solid",
          objects: [{ role: "icon", desc: "A central icon of a crossed straight razor and comb beneath a striped barber pole, encircled by a fine rope border in ivory and brushed steel blue with crisp engraved lines." }],
          texts: [{ role: "title", en: "SHARP & CO.", ru: "БРИТВА И КО." }, { role: "tagline", en: "GROOMING", ru: "УХОД ЗА БОРОДОЙ" }] },
        { ru: "Серфинг и океан", en: "Surf Coast Badge", v: "A sun-faded surf-club badge on a sandy teal background, arched top text \"WILD COAST\" and bottom text \"SURF CLUB\", a central emblem of a breaking wave cradling a vintage longboard with a rising sun behind, framed by a dotted ring and small palm motifs, in washed turquoise and coral #2E8B8B and #E8704A with a soft retro screen-print grain.",
          style_preset_id: "retro_vintage", aesthetics_id: "retro", lighting_id: "daylight", background_id: "paper_craft",
          objects: [{ role: "icon", desc: "A central emblem of a breaking wave cradling a vintage longboard with a rising sun behind, framed by a dotted ring and small palm motifs in washed turquoise and coral." }],
          texts: [{ role: "title", en: "WILD COAST", ru: "ДИКИЙ БЕРЕГ" }, { role: "tagline", en: "SURF CLUB", ru: "СЁРФ-КЛУБ" }] },
        { ru: "Горный поход — заповедник", en: "Wilderness Outdoors Crest", v: "A rugged outdoor-adventure crest on a forest-green background, curved top text \"GREAT NORTH\" and lower text \"TRAIL CO.\", a central icon of a pine tree before a snow-capped mountain under a compass star, bordered by a notched ring and tiny arrowheads, rendered in cream and burnt orange #F2EAD3 and #C25B2C with a worn enamel-pin look.",
          style_preset_id: "handcrafted_artistic", aesthetics_id: "organic", lighting_id: "studio", background_id: "minimal",
          objects: [{ role: "icon", desc: "A central icon of a pine tree before a snow-capped mountain under a compass star, bordered by a notched ring and tiny arrowheads in cream and burnt orange with a worn enamel-pin finish." }],
          texts: [{ role: "title", en: "GREAT NORTH", ru: "ВЕЛИКИЙ СЕВЕР" }, { role: "tagline", en: "TRAIL CO.", ru: "ТРОПЫ И МАРШРУТЫ" }] },
        { ru: "Ремесленная пекарня", en: "Artisan Bakery Mark", v: "A warm artisan-bakery badge on a soft wheat-cream background, arched top text \"GOLDEN CRUST\" and bottom text \"BAKED DAILY\", a central emblem of a crusty round loaf with a wheat stalk and rolling pin crossed beneath it, ringed by a delicate scalloped border, in toasted brown and honey gold #6B4226 and #E0A53B with a gentle vintage paper texture.",
          style_preset_id: "handcrafted_artistic", aesthetics_id: "cozy", lighting_id: "product", background_id: "paper_craft",
          objects: [{ role: "icon", desc: "A central emblem of a crusty round loaf with a wheat stalk and rolling pin crossed beneath it, ringed by a delicate scalloped border in toasted brown and honey gold." }],
          texts: [{ role: "title", en: "GOLDEN CRUST", ru: "ЗОЛОТАЯ КОРОЧКА" }, { role: "tagline", en: "BAKED DAILY", ru: "СВЕЖАЯ ВЫПЕЧКА" }] },
        { ru: "Мотоклуб и гараж", en: "Motorcycle Garage Patch", v: "A tough motorcycle-garage emblem on a matte black background, curved top text \"ROUTE 66\" and bottom banner \"MOTOR WORKS\", a central icon of a winged engine piston with crossed wrenches and a single flame, encircled by a heavy chain-link ring, rendered in chrome silver and racing red #C0C0C0 and #B22222 with a distressed sticker finish.",
          style_preset_id: "streetwear_vector", aesthetics_id: "bold", lighting_id: "studio", background_id: "geometric",
          objects: [{ role: "icon", desc: "A central icon of a winged engine piston with crossed wrenches and a single flame, encircled by a heavy chain-link ring in chrome silver and racing red with a distressed sticker finish." }],
          texts: [{ role: "title", en: "ROUTE 66", ru: "ТРАССА 66" }, { role: "tagline", en: "MOTOR WORKS", ru: "МОТОМАСТЕРСКАЯ" }] },
        { ru: "Винодельня и виноград", en: "Vineyard Estate Seal", v: "An elegant vineyard-estate seal on a deep burgundy background, arched top text \"VALLE D'ORO\" and bottom text \"WINE ESTATE\", a central emblem of a grape cluster draped over a curling vine with a sunlit hilltop villa, framed by a fine laurel wreath, in aged gold and dusty plum #C9A24B and #4E2A3E with a refined embossed-foil texture.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "product", background_id: "gradient",
          objects: [{ role: "icon", desc: "A central emblem of a grape cluster draped over a curling vine with a sunlit hilltop villa behind, framed by a fine laurel wreath in aged gold and dusty plum with embossed-foil detail." }],
          texts: [{ role: "title", en: "VALLE D'ORO", ru: "ВАЛЛЕ Д'ОРО" }, { role: "tagline", en: "WINE ESTATE", ru: "ВИННОЕ ПОМЕСТЬЕ" }] },
        { ru: "Морской — компас и якорь", en: "Nautical Compass Badge", v: "A maritime navigation badge on a stormy slate-blue background, curved top text \"NORTH STAR\" and bottom text \"SAIL & SEA\", a central icon of a brass compass rose over a crossed anchor and oar, ringed by a twisted rope border with tiny ship wheels, rendered in pale sea-foam and weathered brass #DCE6E4 and #B08D4C with a nautical-chart engraving style.",
          style_preset_id: "retro_vintage", aesthetics_id: "retro", lighting_id: "overcast", background_id: "studio_light",
          objects: [{ role: "icon", desc: "A central icon of a brass compass rose over a crossed anchor and oar, ringed by a twisted rope border with tiny ship wheels in pale sea-foam and weathered brass engraving." }],
          texts: [{ role: "title", en: "NORTH STAR", ru: "ПОЛЯРНАЯ ЗВЕЗДА" }, { role: "tagline", en: "SAIL & SEA", ru: "ПАРУС И МОРЕ" }] },
        { ru: "Острый соус — огненный перец", en: "Hot Sauce Firebrand", v: "A fiery hot-sauce brand badge on a deep crimson background, arched top text \"DRAGON HEAT\" and bottom text \"SMALL BATCH\", a central emblem of a flaming chili pepper wreathed in licking flames above a stylised skull, encircled by a jagged sunburst ring, in molten orange and charred black #F25C1E and #1A0E0A with a bold woodcut-poster grain.",
          style_preset_id: "bold_commercial", aesthetics_id: "calm", lighting_id: "highkey", background_id: "solid",
          objects: [{ role: "icon", desc: "A central emblem of a flaming chili pepper wreathed in licking flames above a stylised skull, encircled by a jagged sunburst ring in molten orange and charred black with a bold woodcut grain." }],
          texts: [{ role: "title", en: "DRAGON HEAT", ru: "ДРАКОНИЙ ЖАР" }, { role: "tagline", en: "SMALL BATCH", ru: "МАЛАЯ ПАРТИЯ" }] },
    ],
    logo_mascot: [
        { ru: "Лисёнок-обжарщик кофе", en: "Coffee-roaster fox", v: "A friendly round-cheeked fox mascot clutching a steaming espresso cup, centered above a clean bold wordmark whose letters are formed from rich glossy roasted coffee beans with deep oily highlights, plus a small tagline beneath, on a warm cream lockup with a palette of #2E1A0F, #C9702E and #F2E2C4.",
          style_preset_id: "playful_fun", aesthetics_id: "cozy", lighting_id: "golden", background_id: "paper_craft",
          objects: [{ role: "mascot", desc: "A friendly round-cheeked cartoon fox with rust-orange fur and a creamy belly, big bright eyes, clutching a tiny steaming espresso cup in both paws and grinning warmly." }],
          texts: [{ role: "wordmark", en: "FOX ROAST", ru: "ЛИСИЙ ОБЖАР" }, { role: "tagline", en: "Roasted with heart", ru: "Обжарено с душой" }] },
        { ru: "Робот-доставщик пиццы", en: "Pizza delivery robot", v: "A cheerful boxy delivery robot mascot balancing a fresh pizza slice, set above a confident chunky wordmark whose letters are built from melted stretchy mozzarella with golden toasted crust edges and oozing strings, plus a tiny tagline, on a punchy palette of #E2412B, #F4B12A and #FFF6E8.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "mascot", desc: "A cheerful boxy delivery robot with rounded metal limbs, glowing friendly eye-screen and a red cap, balancing a fresh gooey pizza slice on one outstretched hand." }],
          texts: [{ role: "wordmark", en: "PIZZA BOT", ru: "ПИЦЦА БОТ" }, { role: "tagline", en: "Hot in 20 minutes", ru: "Горячая за 20 минут" }] },
        { ru: "Сова-наставник для онлайн-школы", en: "Wise owl tutor", v: "A scholarly wide-eyed owl mascot perched with tiny round glasses, set above a crisp bold wordmark whose letters are carved from warm polished wood grain with soft golden highlights, plus a small tagline, on a trustworthy palette of #1B3A4B, #3F8E7A and #F1E7D0.",
          style_preset_id: "playful_fun", aesthetics_id: "corporate", lighting_id: "lowkey", background_id: "dark_moody",
          objects: [{ role: "mascot", desc: "A scholarly wide-eyed owl with fluffy teal-and-cream plumage, tiny round golden glasses and a small graduation tassel, perched proudly with an attentive friendly expression." }],
          texts: [{ role: "wordmark", en: "OWL ACADEMY", ru: "СОВА АКАДЕМИЯ" }, { role: "tagline", en: "Learn smarter", ru: "Учись с умом" }] },
        { ru: "Кит-космонавт для стартапа", en: "Astronaut whale startup", v: "A dreamy floating whale mascot in a glass space helmet with tiny stars drifting around it, hovering above a sleek bold wordmark whose letters are sculpted from glowing neon glass tubes ringed by a soft halo, plus a small tagline, on a cosmic palette of #0B1026, #6C5CE7 and #2BD4D9.",
          style_preset_id: "surreal_concept", aesthetics_id: "tech", lighting_id: "neon", background_id: "cosmic_space",
          objects: [{ role: "mascot", desc: "A dreamy plump whale mascot in a clear glass space helmet, soft violet body with cyan glowing markings, fins gently spread as it floats weightless among tiny twinkling stars." }],
          texts: [{ role: "wordmark", en: "WHALE ORBIT", ru: "КИТ ОРБИТА" }, { role: "tagline", en: "Launch your idea", ru: "Запусти свою идею" }] },
        { ru: "Медведь-пекарь", en: "Baker bear", v: "A plump apron-wearing bear mascot proudly holding a fresh loaf, standing above a friendly rounded wordmark whose letters are shaped from golden braided bread dough dusted with fine flour, plus a small tagline, on a cozy bakery palette of #5A3420, #D99A4E and #FBF1DC.",
          style_preset_id: "playful_fun", aesthetics_id: "cozy", lighting_id: "golden", background_id: "studio_light",
          objects: [{ role: "mascot", desc: "A plump brown bear mascot in a flour-dusted apron and small baker's cap, warm friendly smile, proudly cradling a fresh golden loaf of bread in both paws." }],
          texts: [{ role: "wordmark", en: "BEAR BAKERY", ru: "МИШКА ПЕКАРНЯ" }, { role: "tagline", en: "Baked fresh daily", ru: "Свежая выпечка каждый день" }] },
        { ru: "Ленивец для йога-студии", en: "Zen sloth yoga", v: "A serene smiling sloth mascot hanging in a calm meditation pose, placed above a soft bold wordmark whose letters bloom from delicate flowers and lush green foliage with dewy petals, plus a small tagline, on a tranquil palette of #2F4F3E, #88B49A and #F4F0E4.",
          style_preset_id: "playful_fun", aesthetics_id: "calm", lighting_id: "overcast", background_id: "nature_green",
          objects: [{ role: "mascot", desc: "A serene smiling sloth mascot with soft sage-green fur, half-closed peaceful eyes, hanging gently from a leafy branch in a calm cross-legged meditation pose." }],
          texts: [{ role: "wordmark", en: "ZEN SLOTH", ru: "ДЗЕН ЛЕНИВЕЦ" }, { role: "tagline", en: "Slow down, breathe", ru: "Замедлись и дыши" }] },
        { ru: "Дракончик-геймер", en: "Gamer dragon", v: "A spunky baby dragon mascot gripping a glowing game controller with sparks in its eyes, set above an edgy bold wordmark whose letters blaze with bright fire and drifting orange embers, plus a small tagline, on a high-energy palette of #14091F, #FF4D2E and #FFC93C.",
          style_preset_id: "playful_fun", aesthetics_id: "edgy", lighting_id: "dramatic", background_id: "dark_moody",
          objects: [{ role: "mascot", desc: "A spunky baby dragon mascot with deep purple scales and tiny horns, fiery-glow eyes, gripping a glowing game controller in clawed hands with an excited fierce grin." }],
          texts: [{ role: "wordmark", en: "DRAGON PLAY", ru: "ДРАКОН ИГРА" }, { role: "tagline", en: "Level up", ru: "Прокачайся" }] },
        { ru: "Пингвин-сёрфер", en: "Surfer penguin", v: "A cool sunglasses-wearing penguin mascot riding a tiny surfboard, positioned above a breezy bold wordmark whose letters are formed from a splash of clear curling water crowned with foamy white crests, plus a small tagline, on a fresh coastal palette of #073B4C, #06B6D4 and #FDFCDC.",
          style_preset_id: "playful_fun", aesthetics_id: "playful", lighting_id: "daylight", background_id: "gradient",
          objects: [{ role: "mascot", desc: "A cool penguin mascot in dark sunglasses, balancing confidently on a tiny tropical surfboard with flippers out, riding a small curling turquoise wave with a relaxed grin." }],
          texts: [{ role: "wordmark", en: "SURF PENGUIN", ru: "СЁРФ ПИНГВИН" }, { role: "tagline", en: "Catch the wave", ru: "Поймай волну" }] },
        { ru: "Лев-чемпион для фитнес-бренда", en: "Champion lion fitness", v: "A powerful flexing lion mascot with a proud golden mane, towering above a heavy bold wordmark whose letters are forged from polished mirror chrome with razor-sharp metallic highlights, plus a small tagline, on a strong gym palette of #14171C, #E63946 and #C0C5CE.",
          style_preset_id: "bold_commercial", aesthetics_id: "bold", lighting_id: "spotlight", background_id: "studio_dark",
          objects: [{ role: "mascot", desc: "A powerful muscular lion mascot with a proud golden mane, flexing both arms in a champion pose, fierce confident eyes and a strong determined expression." }],
          texts: [{ role: "wordmark", en: "LION FIT", ru: "ЛЕВ ФИТ" }, { role: "tagline", en: "Train like a champion", ru: "Тренируйся как чемпион" }] },
        { ru: "Котик-садовод", en: "Gardener cat", v: "A gentle whiskered cat mascot in a straw hat cradling a little potted sprout, set above a wholesome bold wordmark whose letters are made of fresh green leaves and trailing vines beaded with dewdrops, plus a small tagline, on an earthy palette of #2D4A2B, #7FB069 and #FAF3DD.",
          style_preset_id: "playful_fun", aesthetics_id: "organic", lighting_id: "golden", background_id: "nature_green",
          objects: [{ role: "mascot", desc: "A gentle whiskered cat mascot in a woven straw hat and tiny gardening apron, soft fur and kind eyes, cradling a little terracotta pot with a fresh green sprout." }],
          texts: [{ role: "wordmark", en: "GARDEN CAT", ru: "КОТ САДОВОД" }, { role: "tagline", en: "Grow something good", ru: "Выращивай хорошее" }] },
    ],
    art_hero: [
        { ru: "Воин с трещинами расплавленного золота", en: "Cracked-Gold Warrior Portrait", v: "A solemn close-up of an ancient warrior whose weathered skin is fractured by veins of molten gold, ember-bright eyes burning beneath a battered bronze helmet as dramatic chiaroscuro carves every scar against a deep void-black background.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "lowkey", background_id: "studio_dark",
          objects: [{ role: "hero_subject", desc: "A solemn ancient warrior in tight close-up, weathered bronze-toned skin split by glowing veins of molten liquid gold, ember-bright eyes burning beneath a dented patinated bronze helmet, every scar and rivet carved out by hard dramatic chiaroscuro." }],
          texts: [] },
        { ru: "Кит из звёздной пыли", en: "Cosmic Whale of Stardust", v: "A colossal humpback whale drifting through deep space, its translucent body woven from swirling nebulae, glittering stardust and scattered constellations, fins trailing luminous galactic mist across an endless indigo cosmos.",
          style_preset_id: "surreal_concept", aesthetics_id: "dreamy", lighting_id: "godrays", background_id: "cosmic_space",
          objects: [{ role: "hero_subject", desc: "A colossal humpback whale gliding majestically through deep space, its translucent body woven from swirling violet-and-teal nebulae and glittering stardust, constellations mapped across its flanks and fins trailing luminous galactic mist." }],
          texts: [] },
        { ru: "Балерина из жидкого стекла", en: "Liquid-Glass Ballerina", v: "A lone ballerina frozen mid-pirouette, her flowing gown sculpted entirely from clear shattering glass and arcing ribbons of splashing water, soft studio light refracting through every translucent fold against a misty pale-grey backdrop.",
          style_preset_id: "studio_3d_render", aesthetics_id: "calm", lighting_id: "studio", background_id: "studio_light",
          objects: [{ role: "hero_subject", desc: "A lone ballerina frozen mid-pirouette, her flowing gown and tutu sculpted entirely from clear shattering glass and arcing ribbons of splashing water, soft studio light refracting rainbows through every translucent fold and frozen droplet." }],
          texts: [] },
        { ru: "Лис в пылающей осенней листве", en: "Autumn-Ember Fox", v: "A majestic red fox standing alert in a glowing autumn forest, its fur rendered in fiery amber and crimson, golden afternoon light streaming through falling maple leaves and drifting motes of dust around its sharp, watchful gaze.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "golden", background_id: "nature_green",
          objects: [{ role: "hero_subject", desc: "A majestic red fox standing alert with one paw raised, its dense fur rendered in fiery amber and crimson tipped with backlit gold, ears pricked and sharp watchful eyes catching the warm light." }],
          texts: [] },
        { ru: "Парящий самурайский шлем", en: "Floating Samurai Helmet", v: "An ornate antique samurai helmet floating dead-centre against pure darkness, its lacquered black iron and gold inlay catching a single dramatic rim light while intricate dragon engravings and a deep-red horsehair crest glow with museum-grade detail.",
          style_preset_id: "premium_luxury", aesthetics_id: "luxe", lighting_id: "backlit", background_id: "studio_dark",
          objects: [{ role: "hero_subject", desc: "An ornate antique samurai kabuto helmet floating dead-centre, lacquered black iron with delicate gold inlay and intricate dragon engravings, a deep-red horsehair crest and curved menpo mask catching the dramatic rim light in museum-grade detail." }],
          texts: [] },
        { ru: "Богиня из цветущих лиан", en: "Goddess of Blooming Vines", v: "A serene forest goddess emerging from the gloom, her face and shoulders formed from blooming flowers, soft moss and curling green vines, dewdrops glistening on the petals as dappled sunlight filters through a deep verdant jungle canopy.",
          style_preset_id: "surreal_concept", aesthetics_id: "organic", lighting_id: "godrays", background_id: "nature_green",
          objects: [{ role: "hero_subject", desc: "A serene forest goddess emerging from the gloom, her face and shoulders formed entirely from blooming flowers, soft moss and curling green vines, dewdrops glistening on the petals and her eyes closed in tranquil calm." }],
          texts: [] },
        { ru: "Хрустальный колибри", en: "Crystal Hummingbird", v: "A single hummingbird hovering in mid-air, its body and outstretched wings carved from faceted prismatic crystal that scatters tiny rainbows, suspended before a soft teal-and-rose gradient flecked with delicate bokeh sparkles.",
          style_preset_id: "studio_3d_render", aesthetics_id: "dreamy", lighting_id: "studio", background_id: "gradient",
          objects: [{ role: "hero_subject", desc: "A single hummingbird hovering in mid-air, its body and outstretched blurred wings carved from faceted prismatic crystal that splits light into tiny scattered rainbows, glassy edges catching crisp highlights." }],
          texts: [] },
        { ru: "Космонавт в океанской бездне", en: "Astronaut in the Deep", v: "A lone astronaut suspended in the silent dark of a bioluminescent ocean, glowing jellyfish and drifting plankton lighting the scratched helmet visor as cold blue light gleams off the worn white suit in haunting cinematic detail.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "lowkey", background_id: "dark_moody",
          objects: [{ role: "hero_subject", desc: "A lone astronaut suspended weightless in the deep, a scratched curved helmet visor reflecting the glow of nearby jellyfish, cold blue bioluminescent light gleaming off the worn white pressure suit and trailing bubbles." }],
          texts: [] },
        { ru: "Лев из расплавленной лавы", en: "Molten-Lava Lion", v: "A powerful lion's head emerging from darkness, its mane formed of cracking molten lava and rising embers, glowing orange fissures threading through obsidian-black skin as sparks drift upward into the smoke-filled void.",
          style_preset_id: "surreal_concept", aesthetics_id: "bold", lighting_id: "dramatic", background_id: "dark_moody",
          objects: [{ role: "hero_subject", desc: "A powerful lion's head emerging from darkness, its full mane formed of cracking molten lava and rising embers, glowing orange fissures threading through obsidian-black skin, eyes burning like coals as sparks drift upward." }],
          texts: [] },
        { ru: "Журавль-оригами в тумане", en: "Origami Crane in Mist", v: "A single elegant paper origami crane perched on a moss-covered stone, its crisp folded planes catching gentle morning light amid soft rolling mist and the faint blur of a distant pale-pink cherry-blossom branch.",
          style_preset_id: "editorial", aesthetics_id: "calm", lighting_id: "overcast", background_id: "minimal",
          objects: [{ role: "hero_subject", desc: "A single elegant white paper origami crane perched on a moss-covered grey stone, its crisp folded planes and sharp creases catching gentle morning light, casting soft delicate shadows." }],
          texts: [] },
    ],
    art_composition: [
        { ru: "Маяк и одинокая чайка", en: "Lighthouse and lone gull", v: "A weathered white lighthouse stands on the lower-left third atop dark wet basalt rocks beneath a vast bruised twilight sky, while a single gull glides off toward the upper-right, the wide negative space filled with rolling sea mist and a thin cold horizon line.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "golden", background_id: "sunset_sky",
          objects: [{ role: "foreground_subject", desc: "A weathered white lighthouse with a tarnished iron lantern room, peeling paint and a faint warm beam, perched atop dark wet basalt rocks slick with spray." }, { role: "background_counterpoint", desc: "A single grey-and-white gull gliding small and distant toward the upper sky, wings outstretched against the cold luminous haze." }, { role: "midground_accent", desc: "A low bank of rolling sea mist curling across jagged half-submerged rocks, catching the last cold light of dusk." }],
          texts: [] },
        { ru: "Чаепитие у дождливого окна", en: "Tea by the rainy window", v: "A steaming ceramic cup rests on a worn wooden sill in the lower-right third, its curl of vapour catching warm lamplight, while the upper-left two-thirds dissolve into a rain-streaked pane and the blurred amber glow of a city dusk beyond.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "godrays", background_id: "dark_moody",
          objects: [{ role: "foreground_subject", desc: "A steaming hand-thrown ceramic cup of dark tea on a worn wooden sill, a delicate curl of vapour rising and catching warm lamplight." }, { role: "background_counterpoint", desc: "The blurred amber glow of distant city windows and streetlamps seen through the rain-smeared glass beyond." }, { role: "midground_accent", desc: "Glistening rain droplets and thin downward rivulets running across the cold window pane, refracting the warm interior light." }],
          texts: [] },
        { ru: "Путник на горном гребне", en: "Wanderer on the ridge", v: "A tiny silhouetted hiker pauses on the lower-left third of a knife-edge ridge, dwarfed by a towering snow-dusted peak rising into the upper-right, layered blue valley haze receding behind in cool cinematic depth.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "backlit", background_id: "sunset_sky",
          objects: [{ role: "foreground_subject", desc: "A tiny silhouetted lone hiker with a backpack and trekking pole, pausing on a knife-edge rocky ridge, dwarfed by the scale around them." }, { role: "background_counterpoint", desc: "A towering snow-dusted granite peak rising sharp and immense into the cold upper sky, its summit catching pale light." }, { role: "midground_accent", desc: "A wind-scoured rocky ridge crest threaded with thin snow streaks, dropping away into shadowed slopes." }],
          texts: [] },
        { ru: "Лодка на рассветном озере", en: "Boat on the dawn lake", v: "A small wooden rowboat drifts in the lower-right third of a glass-still alpine lake, its faint wake catching peach dawn light, while pine-dark mountains and a low ribbon of fog occupy the upper-left, mirrored softly in the mirror-calm water.",
          style_preset_id: "cinematic", aesthetics_id: "calm", lighting_id: "golden", background_id: "sunset_sky",
          objects: [{ role: "foreground_subject", desc: "A small weathered wooden rowboat drifting idle on still water, its faint expanding wake catching warm peach dawn light." }, { role: "background_counterpoint", desc: "Dark forested mountains rising on the far shore, their silhouettes mirrored softly in the calm lake surface." }, { role: "midground_accent", desc: "A low ribbon of pale fog hovering above the water's edge, glowing faintly in the first light of morning." }],
          texts: [] },
        { ru: "Красный зонт в снегопаде", en: "Red umbrella in snowfall", v: "A single figure under a vivid red umbrella walks the lower-left third of an empty snow-blanketed boulevard, bare black trees and a faint grey lamppost anchoring the upper-right, fat snowflakes drifting through the muted hushed air.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "dramatic", background_id: "dark_moody",
          objects: [{ role: "foreground_subject", desc: "A lone bundled figure walking beneath a vivid crimson-red umbrella, the single bright accent in a near-monochrome snowy scene." }, { role: "background_counterpoint", desc: "Bare black winter trees standing stark against the pale snow, branches dusted white and reaching into the muffled sky." }, { role: "midground_accent", desc: "A faint grey wrought-iron lamppost half-veiled in falling snow, its glass globe glowing dim and cold." }],
          texts: [] },
        { ru: "Кит под лучом света", en: "Whale beneath the light shaft", v: "A colossal humpback whale glides through the lower-left third of deep teal ocean, a single diver suspended small in the upper-right, golden shafts of sunlight piercing down through the silty water thick with drifting plankton.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "godrays", background_id: "gradient",
          objects: [{ role: "foreground_subject", desc: "A colossal humpback whale gliding gracefully through the water, its barnacled pectoral fins extended and pale grooved underside catching the light." }, { role: "background_counterpoint", desc: "A single scuba diver suspended small and weightless high in the water column, dwarfed by the immense scale of the whale." }, { role: "midground_accent", desc: "Drifting clouds of glittering plankton and tiny bubbles caught in the descending golden sunbeams." }],
          texts: [] },
        { ru: "Скамья под цветущей сакурой", en: "Bench under the cherry tree", v: "An empty weathered park bench sits in the lower-right third of a quiet garden, a gnarled cherry tree heavy with pink blossom leaning in from the upper-left, scattered petals frozen mid-drift across the soft overcast light.",
          style_preset_id: "fine_art_painting", aesthetics_id: "dreamy", lighting_id: "golden", background_id: "bokeh",
          objects: [{ role: "foreground_subject", desc: "An empty weathered wooden park bench with peeling green paint, sitting quietly amid scattered fallen blossom petals." }, { role: "background_counterpoint", desc: "A gnarled old cherry tree heavy with billowing clouds of pink blossom, its dark twisting branches leaning gracefully inward." }, { role: "midground_accent", desc: "Loose cherry petals frozen mid-drift through the still air, falling in soft pink flurries across the garden light." }],
          texts: [] },
        { ru: "Космонавт на алой пустоши", en: "Astronaut on red plains", v: "A lone astronaut stands small on the lower-left third of a rust-red rocky plain, casting a long shadow, while an immense pale crescent planet hangs in the upper-right of a deep starless violet sky, dust hazing the distant horizon.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "daylight", background_id: "sunset_sky",
          objects: [{ role: "foreground_subject", desc: "A lone astronaut in a scuffed white spacesuit standing small on the barren plain, casting a long stark shadow across the red dust." }, { role: "background_counterpoint", desc: "An immense pale crescent planet hanging low in the upper sky, vast and luminous against the deep violet void." }, { role: "midground_accent", desc: "Scattered weathered crimson boulders and wind-carved ridges trailing fine drifting dust across the distant horizon." }],
          texts: [] },
        { ru: "Лиса в зимнем лесу", en: "Fox in the winter wood", v: "A russet fox stands alert in the lower-right third of a snow-laden birch forest, its breath misting in the cold, slanting pale morning sun streaming through the bare trunks in the upper-left and dappling the untouched snow.",
          style_preset_id: "cinematic", aesthetics_id: "organic", lighting_id: "golden", background_id: "minimal",
          objects: [{ role: "foreground_subject", desc: "A vivid russet fox standing alert with ears pricked and bushy tail low, its warm breath misting visibly in the freezing air." }, { role: "background_counterpoint", desc: "Slender pale birch trunks receding into the snowy depths of the forest, lit by slanting golden morning sun." }, { role: "midground_accent", desc: "Untouched powder snow draped over low branches and forest floor, sparkling where the morning light catches it." }],
          texts: [] },
        { ru: "Виолончелистка в пустом зале", en: "Cellist in the empty hall", v: "A solitary cellist sits bathed in a single warm spotlight in the lower-left third of a vast dim concert hall, rows of empty crimson velvet seats receding into shadow in the upper-right, dust motes suspended in the lone beam of light.",
          style_preset_id: "cinematic", aesthetics_id: "cinematic", lighting_id: "spotlight", background_id: "dark_moody",
          objects: [{ role: "foreground_subject", desc: "A solitary cellist seated mid-performance, bow drawn across the strings, bathed in a single warm pool of spotlight." }, { role: "background_counterpoint", desc: "Rows upon rows of empty crimson velvet seats receding into deep shadow, vast and silent in the dim hall." }, { role: "midground_accent", desc: "Fine dust motes drifting and glinting within the lone beam of warm light, tracing its shape through the dark air." }],
          texts: [] },
    ],
    surreal_scene: [
        { ru: "Кит плывёт в облачном небе", en: "Whale in cloud sky", v: "A colossal humpback whale drifts weightlessly through a pastel dawn sky, trailing a long ribbon of golden migrating birds from its flukes while a tiny rowboat with a single lantern floats in its shadow far below, cinematic soft volumetric light, dreamlike fine-art surrealism.",
          style_preset_id: "surreal_concept", aesthetics_id: "dreamy", lighting_id: "godrays", background_id: "sunset_sky",
          objects: [{ role: "primary", desc: "a colossal humpback whale gliding weightlessly through the air, its barnacled skin catching warm dawn light, vast pectoral fins spread like wings" }, { role: "secondary", desc: "a long ribbon of golden migrating birds streaming from the whale's flukes, scattering into shimmering specks across the sky" }, { role: "accent", desc: "a tiny wooden rowboat carrying a single glowing lantern, drifting in the whale's vast shadow far below" }],
          texts: [] },
        { ru: "Дверь посреди пустыни", en: "Doorway in the desert", v: "A solitary antique wooden door stands open in the middle of an endless rippling sand desert, spilling a torrent of clear blue ocean water and darting silver fish through its frame onto the dry dunes, long evening shadows, surreal poetic contrast of two worlds, hyperreal painterly detail.",
          style_preset_id: "surreal_concept", aesthetics_id: "cinematic", lighting_id: "golden", background_id: "sunset_sky",
          objects: [{ role: "primary", desc: "a solitary weathered antique wooden door standing open and unsupported in the middle of the dunes, its faded paint peeling" }, { role: "secondary", desc: "a torrent of clear turquoise ocean water gushing through the open doorframe and spilling across the dry sand" }, { role: "accent", desc: "darting silver fish leaping mid-flow through the cascade of water, glinting in the warm light" }],
          texts: [] },
        { ru: "Чаепитие на спине черепахи", en: "Tea party on a turtle", v: "An ancient mossy giant tortoise carries an entire miniature porcelain tea set with steaming cups and a tilted brass chandelier on its domed shell, wading slowly across a mirror-still lake at twilight as fireflies drift around it, whimsical surreal storybook atmosphere with rich warm light.",
          style_preset_id: "surreal_concept", aesthetics_id: "cozy", lighting_id: "lowkey", background_id: "dark_moody",
          objects: [{ role: "primary", desc: "an ancient giant tortoise with a moss-covered domed shell, wading slowly through the shallow lake, deeply wrinkled and serene" }, { role: "secondary", desc: "a miniature porcelain tea set with steaming cups and a tilted brass chandelier balanced on the tortoise's shell" }, { role: "accent", desc: "a scattering of glowing fireflies drifting and swirling in the warm air around the tortoise" }],
          texts: [] },
        { ru: "Лестница в перевёрнутый океан", en: "Stairs to an upside-down ocean", v: "A spiral marble staircase rises out of a misty meadow and dissolves into an inverted ocean hanging overhead, where jellyfish float like glowing lanterns and a lone deer climbs toward the water-sky, surreal gravity-defying composition, ethereal blue and rose light, fine-art dreamscape.",
          style_preset_id: "surreal_concept", aesthetics_id: "dreamy", lighting_id: "highkey", background_id: "glow_mesh",
          objects: [{ role: "primary", desc: "a spiral white marble staircase rising from the meadow and dissolving upward into the hanging inverted ocean" }, { role: "secondary", desc: "translucent glowing jellyfish floating like lanterns within the upside-down sea, trailing luminous tendrils" }, { role: "accent", desc: "a lone slender deer climbing the staircase, gazing up toward the water-sky above" }],
          texts: [] },
        { ru: "Город в стеклянном пузыре", en: "City in a glass bubble", v: "A delicate floating soap bubble cradles a tiny glowing miniature city of crooked towers inside it, balanced on the fingertip of a giant stone hand emerging from a sea of clouds, refractions and rainbow sheen sliding across the glass surface, surreal intimate scale play, luminous cinematic mood.",
          style_preset_id: "surreal_concept", aesthetics_id: "cinematic", lighting_id: "backlit", background_id: "bokeh",
          objects: [{ role: "primary", desc: "a delicate translucent soap bubble with rainbow sheen sliding across its curved surface, glowing softly from within" }, { role: "secondary", desc: "a tiny luminous miniature city of crooked golden towers cradled inside the bubble, windows twinkling" }, { role: "accent", desc: "a giant weathered stone hand emerging from the clouds, balancing the bubble on a single outstretched fingertip" }],
          texts: [] },
        { ru: "Дерево с планетами вместо плодов", en: "Tree bearing planets", v: "A gnarled ancient tree grows on a small floating island and bears ripe glowing planets instead of fruit, one cracked open to reveal a swirling galaxy inside, while a child reaches up from a ladder built of stacked old books, surreal cosmic still life, deep indigo night and warm amber glow.",
          style_preset_id: "surreal_concept", aesthetics_id: "cinematic", lighting_id: "spotlight", background_id: "cosmic_space",
          objects: [{ role: "primary", desc: "a gnarled ancient tree with twisting bark growing on a small floating island, its branches heavy with ripe glowing planets instead of fruit" }, { role: "secondary", desc: "one cracked-open planet hanging from a branch, revealing a swirling miniature galaxy of light inside" }, { role: "accent", desc: "a small child reaching upward from a precarious ladder built of stacked weathered old books" }],
          texts: [] },
        { ru: "Кит-аэростат над затопленным городом", en: "Airship-whale over a flooded city", v: "A translucent glass whale rigged like a floating airship drifts low over a half-submerged city of drowned rooftops, suspending a single illuminated greenhouse garden from its belly, soft reflections shimmering on the still flood water, melancholic surreal beauty, muted teal and gold palette.",
          style_preset_id: "surreal_concept", aesthetics_id: "cinematic", lighting_id: "overcast", background_id: "dark_moody",
          objects: [{ role: "primary", desc: "a translucent glass whale rigged like a floating airship with brass fittings and rope rigging, drifting low over the water" }, { role: "secondary", desc: "a single illuminated greenhouse garden of lush green plants suspended in a glass dome from the whale's belly" }, { role: "accent", desc: "shimmering soft reflections of the whale and rooftops mirrored on the still flood water below" }],
          texts: [] },
        { ru: "Человек-облако на скамейке", en: "Cloud-headed figure on a bench", v: "A lone figure in a vintage suit sits on a wrought-iron park bench with a swirling thunderstorm cloud where the head should be, gentle rain falling only over the bench while the rest of the autumn park stays sunlit, surreal melancholy portrait, soft cinematic chiaroscuro light.",
          style_preset_id: "surreal_concept", aesthetics_id: "cinematic", lighting_id: "dramatic", background_id: "nature_green",
          objects: [{ role: "primary", desc: "a lone figure in a vintage tailored suit seated calmly on a bench, with a swirling dark thunderstorm cloud where the head should be" }, { role: "secondary", desc: "a wrought-iron park bench with ornate scrollwork, slick and glistening from the falling rain" }, { role: "accent", desc: "a narrow column of gentle rain falling only over the bench, droplets catching the chiaroscuro light" }],
          texts: [] },
        { ru: "Кит из оригами и лунный свет", en: "Origami whale and moonlight", v: "An enormous paper origami whale folded from old maps floats above a calm midnight sea, its translucent body lit from within by a captured full moon while paper birds peel away from its fins and scatter into the stars, surreal monochrome blue dream, delicate luminous craft aesthetic.",
          style_preset_id: "surreal_concept", aesthetics_id: "dreamy", lighting_id: "backlit", background_id: "cosmic_space",
          objects: [{ role: "primary", desc: "an enormous origami whale folded from yellowed old maps, its translucent paper body glowing softly from within with captured moonlight" }, { role: "secondary", desc: "a captured full moon nestled inside the whale's paper body, radiating a warm internal glow through the creased map paper" }, { role: "accent", desc: "delicate folded paper birds peeling away from the whale's fins and scattering upward into the starry sky" }],
          texts: [] },
        { ru: "Музыкант с виолончелью из воды", en: "Cellist of liquid water", v: "A seated cellist plays an instrument made entirely of flowing transparent water that arcs and splashes mid-note into leaping silver fish, sheet music dissolving into a flock of moths drifting upward in a candlelit empty hall, surreal poetic synesthesia, rich warm chiaroscuro and glistening detail.",
          style_preset_id: "surreal_concept", aesthetics_id: "luxe", lighting_id: "dramatic", background_id: "studio_dark",
          objects: [{ role: "primary", desc: "a seated cellist mid-performance, bow drawn across an instrument made entirely of flowing transparent water that arcs and splashes in the air" }, { role: "secondary", desc: "leaping silver fish bursting from the splashing water of the liquid cello, glinting in the candlelight" }, { role: "accent", desc: "sheet music on a stand dissolving into a flock of pale moths drifting upward toward the dark rafters" }],
          texts: [] },
    ],
};

// ── Localization ───────────────────────────────────────────────────────────── //
const I18N = {
    en: {
        clear: "Clear all", mp_label: "MP",
        add_text: "+ Text", add_obj: "+ Object",
        reference: "🖼 Reference", clear_ref: "✕ ref", cancel: "Cancel", save: "Save",
        card_template: "Layout", layout_preset: "Layout template",
        layout_none: "— pick a template —",
        card_style: "Style",
        visual_style: "Visual style",
        tip_visual_style: "Pick a complete visual style — art style, lighting, mood and palette at once. It instantly recolours the canvas; you can still fine-tune everything below.",
        visual_style_hint: "A complete look in one click (art style + palette) — recolours the canvas; refine the details below.",
        hld: "Main idea", aesthetics: "Mood & vibe",
        lighting: "Lighting", art_style: "Art style",
        image_palette: "Image colors (up to {n})", background: "Background", add_color: "Add color",
        lighting_colors: "Lighting colors (up to {n})", background_colors: "Background colors (up to {n})",
        hld_hint: "One sentence describing the whole image — the model leans on this most. E.g. a bold summer-sale poster for a sneaker brand.",
        aesthetics_hint: "The overall feel in a few words. Example: bold and punchy, calm and minimal, retro, luxurious. Safe to leave blank.",
        lighting_hint: "How the scene is lit. Example: bright daylight, soft studio light, moody shadows, neon glow. Safe to leave blank.",
        art_style_hint: "Shown for every image type except Photo — the drawing/rendering style. Example: flat vector, watercolor, low-poly 3D, bold poster graphics.",
        image_palette_hint: "Optional palette the whole image sticks to — add up to {n} colors. Leave empty to let Ideogram choose.",
        lighting_colors_hint: "Up to {n} colors that tint the light (folded into the lighting description). Empty = neutral light.",
        background_colors_hint: "Up to {n} colors the background sticks to (folded into the background description). Empty = Ideogram chooses.",
        background_hint: "What sits behind everything — the backdrop behind your text and objects. Example: smooth orange-to-pink gradient, blurred city street, dark marble. Leave empty for a plain background.",
        block_text_title: "Text block", block_obj_title: "Object (obj)",
        text_literal: "Text (rendered literally)",
        font_preset: "Text style",
        weight: "Weight", case: "Case", text_color: "Text color", outline_color: "Outline color", plate_color: "Plate color",
        legibility: "Legibility", leg_outline: "Outline", leg_block: "Plate",
        visual_only: "Manual text (visual-only — empty placeholder for a hand overlay)",
        override: "Extra description (override, appended last)",
        block_palette: "Block palette (up to {n})", desc_preview: "Final description (desc) for the model:",
        obj_desc: "Object description (desc)",
        select_block: "Select a block on the canvas, or add a new one (+ Text / + Object).",
        visual_only_preview: "(visual-only) this area becomes an empty placeholder — add the text by hand in Figma/Photoshop.",
        export_btn: "⬇ Export", import_btn: "⬆ Import",
        import_done: "Imported: {n}", import_empty: "No valid presets in the file",
        custom_tag: "custom",
        empty_hint: "Click \"✎ Edit design\"", edit_btn: "✎ Edit design",
        badge_text: "Text", badge_obj: "Object", bbox_label: "bbox [y,x,y,x]",
        tip_add_text: "Drops a new text block on the canvas — type the words you want printed on the image.",
        tip_add_obj: "Adds an object block — describe a thing to place in the scene, like a product or icon.",
        tip_duplicate: "Makes an exact copy of the selected block, so you can reuse it without starting over (Ctrl+D).",
        tip_delete: "Removes the block you've selected from the canvas. Other blocks stay untouched.",
        tip_clear: "Wipes every block and starts you with a clean, empty canvas.",
        tip_language: "Switches the interface and new blocks between Russian and English — pick what you're comfy with.",
        tip_aspect: "Sets the shape of your image — wide, tall, or square — to match where it'll be used.",
        tip_megapixels: "Slide to set how detailed the final image is — higher means sharper but a bit slower.",
        tip_reference: "Upload a picture to show faintly behind the canvas, so you can trace your layout over it.",
        tip_clear_ref: "Removes the faint reference picture from behind the canvas.",
        tip_cancel: "Closes the editor and throws away any changes you made here.",
        tip_save: "Saves your design back to the node and closes the editor — your work is kept.",
        tip_layout_preset: "Pick a ready-made layout to drop in text and object blocks with a matching aspect ratio and background.",
        tip_add_color: "Add a color: opens the color picker so you can pin one more shade to this palette.",
        tip_palette_swatch: "Click a color to remove it from the palette.",
        tip_obj_desc: "Describe what this graphic shows — e.g. a smiling sneaker, a coffee cup. Ideogram draws it here.",
        tip_text_literal: "Type the exact words you want to appear — these letters are drawn on the image as-is.",
        tip_font_preset: "The text style — the lettering's typographic character. Ideogram can't use a typeface by name, so this sets a type DESCRIPTION (the real lever). Set weight, case and color below.",
        tip_weight: "How thick the letters look — from delicate Thin up to chunky Heavy.",
        tip_case: "How the letters are cased: as you typed, ALL CAPS, or Title Style.",
        tip_text_color: "Pick the color of these letters — tap the swatch to choose any shade.",
        tip_outline_color: "The color of the outline around the letters.",
        tip_plate_color: "The color of the solid plate behind the text.",
        tip_leg_outline: "Adds an outline around the letters so they stand out on busy backgrounds. Pick its color below.",
        tip_leg_block: "Puts the text on a solid color plate behind it, like a sticker. Pick its color below.",
        tip_visual_only: "Leaves this spot blank so you can drop the text in by hand later in Figma or Photoshop.",
        tip_override: "Add any extra wording for this block — it's tacked on at the very end of the description.",
        tip_block_palette: "Set the colors just for this block — add a few swatches with the ＋ button.",
        tip_edit_design: "Opens the full editor so you can add text, objects, and tweak the whole design.",
        tip_dims_pill: "Shows the final image size in pixels (width × height). Just a hint — nothing to click.",
        tip_node_canvas: "A live preview of your design. Double-click anywhere here to open the editor.",
        tip_block_rect: "A block on the canvas — click to select, drag to move, double-click to edit its text, Alt+drag to clone it.",
        tip_resize_handle: "Drag a corner or edge to resize the selected block.",
        opt_none: "(none)", opt_custom: "Custom…",
        layers_title: "Layers", layers_empty: "No layers yet — add text or an object below.",
        layer_untitled: "(untitled)", general_settings: "General settings",
        json_prompt: "JSON prompt", copy: "Copy", copied: "Copied!",
        tip_copy_json: "Copy the ready JSON prompt to the clipboard.",
        object_preset: "Object preset",
        tip_object_preset: "Pick a ready subject for this object — a character or a hero object. It replaces the description.",
        card_design: "Design preset", design_saved: "Saved designs", design_load: "— load a design —",
        design_apply: "↻ Load", design_save: "💾 Save", design_name_prompt: "Design name:",
        tip_design_card: "Save, load, export and import the WHOLE design — layout, style, objects, text and prompt tweaks — as one reusable preset stored in the node folder.",
        tip_design_load: "Pick one of your saved designs, then press Load to drop it onto the canvas.",
        tip_design_apply: "Load the selected saved design — replaces everything currently on the canvas.",
        tip_design_save: "Save the current design as a named preset in the node folder.",
        tip_design_export: "Download the current design as a JSON file to back it up or share it.",
        tip_design_import: "Load design JSON file(s) from your computer into your saved designs.",
        tip_design_delete: "Delete the selected saved design from the node folder.",
        design_delete_confirm: "Delete this saved design?",
        clear_confirm: "Clear the whole design? Blocks, style, background and colors will all be reset.",
        tip_layers: "Your blocks as a stack — the top of the list is the front of the image.",
        tip_layer_row: "Click to select; drag up/down to change overlap. Higher = closer to the front.",
        tip_layer_up: "Move one step toward the front.",
        tip_layer_down: "Move one step toward the back.",
    },
    ru: {
        clear: "Очистить всё", mp_label: "МП",
        add_text: "+ Текст", add_obj: "+ Объект",
        reference: "🖼 Референс", clear_ref: "✕ реф", cancel: "Отмена", save: "Сохранить",
        card_template: "Макет", layout_preset: "Шаблон-лейаут",
        layout_none: "— выберите шаблон —",
        card_style: "Стиль",
        visual_style: "Визуальный стиль",
        tip_visual_style: "Выберите целиком визуальный стиль — художественный стиль, свет, настроение и палитра сразу. Канвас мгновенно перекрашивается; ниже всё можно донастроить.",
        visual_style_hint: "Готовый визуальный стиль одним кликом (худ. стиль + палитра) — перекрашивает канвас; детали правьте ниже.",
        hld: "Главная идея", aesthetics: "Настроение и вайб",
        lighting: "Освещение", art_style: "Художественный стиль",
        image_palette: "Цвета изображения (до {n})", background: "Фон", add_color: "Добавить цвет",
        lighting_colors: "Цвета освещения (до {n})", background_colors: "Цвета фона (до {n})",
        hld_hint: "Одно предложение про всю картинку — модель опирается на него сильнее всего. Например: яркий постер летней распродажи кроссовок.",
        aesthetics_hint: "Общее ощущение в паре слов. Например: дерзко и сочно, спокойно и минимально, ретро, премиально. Можно оставить пустым.",
        lighting_hint: "Как освещена сцена. Например: яркий дневной свет, мягкий студийный, драматичные тени, неоновое свечение. Можно оставить пустым.",
        art_style_hint: "Показывается для всех типов, кроме «Фото» — стиль отрисовки. Например: плоский вектор, акварель, low-poly 3D, плакатная графика.",
        image_palette_hint: "Необязательная палитра, которой держится вся картинка — до {n} цветов. Оставьте пустым — Ideogram подберёт сам.",
        lighting_colors_hint: "До {n} цветов, которыми подкрашен свет (вшиваются в описание освещения). Пусто — нейтральный свет.",
        background_colors_hint: "До {n} цветов, которых держится фон (вшиваются в описание фона). Пусто — Ideogram подберёт сам.",
        background_hint: "Что находится позади всего — задний фон под текстом и объектами. Например: плавный градиент из оранжевого в розовый, размытая улица, тёмный мрамор. Пусто — простой фон.",
        block_text_title: "Текстовый блок", block_obj_title: "Объект (obj)",
        text_literal: "Текст (рендерится буква-в-букву)",
        font_preset: "Стиль текста",
        weight: "Вес", case: "Регистр", text_color: "Цвет текста", outline_color: "Цвет обводки", plate_color: "Цвет плашки",
        legibility: "Читаемость", leg_outline: "Обводка", leg_block: "Плашка",
        visual_only: "Текст вручную (пустая плашка под ручной оверлей)",
        override: "Доп. описание (добавляется в конец)",
        block_palette: "Палитра блока (до {n})", desc_preview: "Итоговое описание для модели:",
        obj_desc: "Описание объекта (desc)",
        select_block: "Выберите блок на холсте или добавьте новый (+ Текст / + Объект).",
        visual_only_preview: "(visual-only) область станет пустой плашкой без текста — добавьте надпись вручную в Figma/Photoshop.",
        export_btn: "⬇ Экспорт", import_btn: "⬆ Импорт",
        import_done: "Импортировано: {n}", import_empty: "В файле нет валидных пресетов",
        custom_tag: "свой",
        empty_hint: "Нажмите «✎ Редактировать»", edit_btn: "✎ Редактировать",
        badge_text: "Текст", badge_obj: "Объект", bbox_label: "рамка [y,x,y,x]",
        tip_add_text: "Добавляет на холст новый текстовый блок — впишите слова, которые должны быть на картинке.",
        tip_add_obj: "Добавляет блок-объект — опишите предмет для сцены, например товар или иконку.",
        tip_duplicate: "Делает точную копию выбранного блока, чтобы не создавать его заново (Ctrl+D).",
        tip_delete: "Удаляет выбранный блок с холста. Остальные блоки остаются на месте.",
        tip_clear: "Стирает все блоки и даёт чистый пустой холст для нового старта.",
        tip_language: "Переключает интерфейс и новые блоки между русским и английским — выберите удобный.",
        tip_aspect: "Задаёт форму картинки — широкую, высокую или квадрат — под место, где она будет.",
        tip_megapixels: "Ползунком задаёте детализацию картинки — больше значит чётче, но чуть дольше.",
        tip_reference: "Загрузите картинку — она проступит фоном, чтобы выкладывать макет поверх неё.",
        tip_clear_ref: "Убирает картинку-подсказку из-под холста.",
        tip_cancel: "Закрывает редактор и отменяет все изменения, что вы тут сделали.",
        tip_save: "Сохраняет дизайн в ноду и закрывает редактор — ваша работа остаётся.",
        tip_layout_preset: "Выберите готовый лейаут — он расставит блоки текста и объектов с подходящим форматом и фоном.",
        tip_add_color: "Добавить цвет: откроет палитру выбора, чтобы закрепить ещё один оттенок в наборе.",
        tip_palette_swatch: "Нажмите на цвет, чтобы убрать его из палитры.",
        tip_obj_desc: "Опишите, что здесь нарисовать — например, улыбающийся кроссовок или чашка кофе. Ideogram это и нарисует.",
        tip_text_literal: "Впишите точные слова, которые должны появиться — эти буквы рисуются на картинке буква-в-букву.",
        tip_font_preset: "Стиль текста — типографический характер букв. Ideogram не умеет шрифт по имени, поэтому это задаёт ОПИСАНИЕ типа (реальный рычаг). Вес, регистр и цвет — ниже.",
        tip_weight: "Насколько толстые буквы — от тонких до жирных и совсем мощных.",
        tip_case: "Как оформить буквы: как набрали, ВСЕ ЗАГЛАВНЫЕ или С Заглавных.",
        tip_text_color: "Выберите цвет этих букв — нажмите на квадратик и подберите любой оттенок.",
        tip_outline_color: "Цвет обводки вокруг букв.",
        tip_plate_color: "Цвет сплошной плашки под текстом.",
        tip_leg_outline: "Добавит обводку вокруг букв, чтобы они читались на пёстром фоне. Цвет — ниже.",
        tip_leg_block: "Положит текст на сплошную цветную плашку, как на наклейке. Цвет — ниже.",
        tip_visual_only: "Оставит здесь пустое место, чтобы вы потом сами вписали текст в Figma или Photoshop.",
        tip_override: "Допишите что угодно про этот блок — это добавится в самый конец его описания.",
        tip_block_palette: "Задайте цвета только для этого блока — добавьте пару образцов кнопкой ＋.",
        tip_edit_design: "Открывает полный редактор: добавляйте текст, объекты и настраивайте весь дизайн.",
        tip_dims_pill: "Показывает итоговый размер картинки в пикселях (ширина × высота). Просто подсказка.",
        tip_node_canvas: "Живой предпросмотр дизайна. Дважды кликните здесь, чтобы открыть редактор.",
        tip_block_rect: "Блок на холсте — кликните, чтобы выбрать, тяните для перемещения, двойной клик — редактировать текст, Alt+перетаскивание — скопировать.",
        tip_resize_handle: "Тяните за угол или край, чтобы изменить размер выбранного блока.",
        opt_none: "(не задано)", opt_custom: "Своё…",
        layers_title: "Слои", layers_empty: "Пока нет слоёв — добавьте текст или объект ниже.",
        layer_untitled: "(без названия)", general_settings: "Общие настройки",
        json_prompt: "JSON-промпт", copy: "Копировать", copied: "Скопировано!",
        tip_copy_json: "Скопировать готовый JSON-промпт в буфер обмена.",
        object_preset: "Пресет объекта",
        tip_object_preset: "Выберите готовый объект — персонажа или герой-предмет. Заменяет описание.",
        card_design: "Пресет дизайна", design_saved: "Сохранённые дизайны", design_load: "— загрузить дизайн —",
        design_apply: "↻ Загрузить", design_save: "💾 Сохранить", design_name_prompt: "Имя дизайна:",
        tip_design_card: "Сохраняйте, загружайте, экспортируйте и импортируйте ВЕСЬ дизайн — макет, стиль, объекты, тексты и правки промтов — одним пресетом в папке ноды.",
        tip_design_load: "Выберите сохранённый дизайн и нажмите «Загрузить», чтобы выложить его на холст.",
        tip_design_apply: "Загрузить выбранный сохранённый дизайн — заменит всё, что сейчас на холсте.",
        tip_design_save: "Сохранить текущий дизайн именованным пресетом в папке ноды.",
        tip_design_export: "Скачать текущий дизайн в JSON-файл — для бэкапа или чтобы поделиться.",
        tip_design_import: "Загрузить JSON-файлы дизайнов с компьютера в ваши сохранённые дизайны.",
        tip_design_delete: "Удалить выбранный сохранённый дизайн из папки ноды.",
        design_delete_confirm: "Удалить этот сохранённый дизайн?",
        clear_confirm: "Очистить весь дизайн? Блоки, стиль, фон и цвета будут сброшены.",
        tip_layers: "Ваши блоки стопкой — верх списка это передний план картинки.",
        tip_layer_row: "Клик — выбрать; тяните вверх/вниз, чтобы менять перекрытие. Выше = ближе к переднему плану.",
        tip_layer_up: "На шаг к переднему плану.",
        tip_layer_down: "На шаг к заднему плану.",
    },
};

const SEG_LABELS = {
    weight: { en: { Thin: "Thin", Regular: "Regular", Bold: "Bold" },
              ru: { Thin: "Тонкий", Regular: "Обычный", Bold: "Жирный" } },
    case: { en: { "As-typed": "As-typed", UPPERCASE: "UPPERCASE", Title: "Title" },
            ru: { "As-typed": "Как есть", UPPERCASE: "ВЕРХНИЙ", Title: "Заголовок" } },
};

export function t(key, lang = DEFAULT_LANG, vars) {
    const table = I18N[lang] || I18N.ru;
    let s = table[key] != null ? table[key] : (I18N.ru[key] != null ? I18N.ru[key] : key);
    if (vars) for (const k of Object.keys(vars)) s = s.split(`{${k}}`).join(vars[k]);
    return s;
}

export function segLabel(group, value, lang = DEFAULT_LANG) {
    return SEG_LABELS[group]?.[lang]?.[value] ?? SEG_LABELS[group]?.en?.[value] ?? value;
}

export function localizedName(item, lang = DEFAULT_LANG) {
    return (item && (item[`name_${lang}`] || item.name_en || item.name_ru || item.id)) || "";
}

export function localizedDesc(item, lang = DEFAULT_LANG) {
    return (item && (item[`desc_${lang}`] || item.desc_en || item.desc_ru || "")) || "";
}

// ── Layout templates ──────────────────────────────────────────────────────── //
export function layoutsList(presets) { return presets?.layouts || []; }

// Instantiate a layout template's placeholder blocks for the given language.
export function instantiateLayout(layout, lang = DEFAULT_LANG) {
    const blocks = (layout?.blocks || []).map((b) => {
        const rect = { ...(b.rect || { x: 0.1, y: 0.1, w: 0.4, h: 0.2 }) };
        if (b.type === "obj") {
            return {
                id: makeBlockId(), type: "obj", rect,
                desc: b.desc_en || "", role: b.role || "",
                color_palette: Array.isArray(b.color_palette) ? b.color_palette : [],
            };
        }
        const text = lang === "en" ? (b.text_en ?? b.text_ru ?? "") : (b.text_ru ?? b.text_en ?? "");
        // A layout may carry its own legibility (e.g. material lettering wants the
        // outline OFF) and a per-block desc_override (the material/effect hint);
        // fall back to the historical default when absent.
        const leg = (b.legibility && typeof b.legibility === "object")
            ? { outline: !!b.legibility.outline, solid_block: !!b.legibility.solid_block }
            : { outline: true, solid_block: false };
        return {
            id: makeBlockId(), type: "text", rect, text,
            font_preset_id: b.font_preset_id || "grotesque_black",
            weight: b.weight || "Bold", case: b.case || "As-typed",
            color: normHex(b.color) || "#FFFFFF",
            outline_color: normHex(b.outline_color) || "#000000",
            plate_color: normHex(b.plate_color) || "#1A1A1A",
            legibility: leg,
            visual_only: false, desc_override: String(b.desc_override || ""), role: b.role || "",
        };
    });
    return {
        blocks,
        aspect_ratio: layout?.aspect_ratio || DEFAULT_ASPECT_RATIO,
        background: layout?.background_en || "",
        high_level_description: layout?.high_level_description_en || "",
    };
}


// CSS font-family approximation per font preset id — for the canvas style preview only.
const FONT_FAMILY = {
    grotesque_black: "'Arial Black','Helvetica Neue',Arial,sans-serif",
    geometric_sans: "'Century Gothic',Futura,'Trebuchet MS',sans-serif",
    condensed_bold: "'Arial Narrow','Roboto Condensed',sans-serif",
    display_poster_sans: "Impact,Haettenschweiler,'Arial Black',sans-serif",
    slab_serif_heavy: "Rockwell,'Roboto Slab',Georgia,serif",
    rounded_friendly: "'Varela Round','Comic Sans MS','Trebuchet MS',sans-serif",
    stencil_block: "Stencil,'Arial Black',sans-serif",
    humanist_sans: "'Segoe UI','Open Sans',Verdana,sans-serif",
    retro_70s_round: "'Cooper Black','Comic Sans MS',serif",
    comic_cartoon: "'Comic Sans MS','Chalkboard SE',cursive",
    mono_techno: "'Consolas','Courier New',monospace",
    vintage_serif: "Georgia,'Times New Roman',serif",
    graffiti_urban: "Impact,sans-serif",
    didone_luxury: "Didot,'Bodoni MT','Playfair Display',Georgia,serif",
    formal_script: "'Segoe Script','Brush Script MT',cursive",
    brush_marker: "'Segoe Script','Bradley Hand',cursive",
    // Material / effect lettering treatments — preview families approximate the
    // letterform shape; the actual material is rendered by Ideogram from desc_snippet.
    ice_crystal: "Impact,'Arial Black',sans-serif",
    neon_tube: "'Segoe Script','Brush Script MT',cursive",
    fresh_fruit: "'Cooper Black','Arial Rounded MT Bold',sans-serif",
    soft_fur: "'Varela Round','Comic Sans MS',sans-serif",
    water_splash: "'Brush Script MT','Segoe Script',cursive",
    carved_stone: "Rockwell,'Roboto Slab',Georgia,serif",
    clear_glass: "'Century Gothic',Futura,'Trebuchet MS',sans-serif",
    liquid_chrome: "Impact,'Arial Black',sans-serif",
    molten_gold: "Didot,'Bodoni MT','Playfair Display',Georgia,serif",
    flames_fire: "Impact,'Arial Black',sans-serif",
    balloon_3d: "'Cooper Black','Arial Rounded MT Bold',sans-serif",
    lush_floral: "'Segoe Script','Brush Script MT',cursive",
    carved_wood: "Rockwell,'Roboto Slab',Georgia,serif",
    dripping_paint: "Impact,sans-serif",
};
export function fontFamilyForPreset(id) { return FONT_FAMILY[id] || "'Segoe UI',sans-serif"; }

// ── Palette → gradient (shared by the editor artboard/blocks + node canvas) ── //
export function hexToRgb(hex) {
    const h = normHex(hex);
    if (!h) return null;
    return { r: parseInt(h.slice(1, 3), 16), g: parseInt(h.slice(3, 5), 16), b: parseInt(h.slice(5, 7), 16) };
}

export function hexToRgba(hex, alpha = 1) {
    const c = hexToRgb(hex);
    return c ? `rgba(${c.r},${c.g},${c.b},${alpha})` : `rgba(0,0,0,${alpha})`;
}

// Deterministic blob anchor points (percent) for the layered "mesh" gradient.
export const MESH_POSITIONS = [
    [18, 20], [82, 24], [26, 78], [78, 80], [50, 12], [12, 54], [88, 62], [44, 92],
];

// CSS background for a palette. alpha<1 → translucent (block tint).
//   0 colors → "" (caller falls back to a default).
//   1 color  → soft single-color tint.
//   ≥2 + mesh=false → clean diagonal multi-stop gradient (block fills).
//   ≥2 + mesh=true  → layered radial "mesh" gradient (image artboard).
export function paletteGradientCss(colors, { alpha = 1, angle = 135, mesh = true } = {}) {
    const pal = cleanPalette(colors || [], IMAGE_PALETTE_CAP);
    if (!pal.length) return "";
    if (pal.length === 1) {
        return `linear-gradient(${angle}deg, ${hexToRgba(pal[0], alpha)}, ${hexToRgba(pal[0], alpha * 0.4)})`;
    }
    const stops = pal
        .map((hex, i) => `${hexToRgba(hex, alpha)} ${Math.round((i / (pal.length - 1)) * 100)}%`)
        .join(", ");
    const linear = `linear-gradient(${angle}deg, ${stops})`;
    if (!mesh) return linear;
    const layers = pal.map((hex, i) => {
        const [px, py] = MESH_POSITIONS[i % MESH_POSITIONS.length];
        return `radial-gradient(circle at ${px}% ${py}%, ${hexToRgba(hex, alpha)} 0%, ${hexToRgba(hex, 0)} 55%)`;
    });
    layers.push(linear);
    return layers.join(", ");
}
