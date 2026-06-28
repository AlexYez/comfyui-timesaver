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
export const WEIGHTS = ["Thin", "Regular", "Bold", "Heavy"];
export const CASES = ["As-typed", "UPPERCASE", "Title"];
export const PROMINENCE = ["Caption", "Body", "Headline", "Hero"];
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
const PROMINENCE_PHRASE = {
    Caption: "small caption text",
    Body: "medium body text",
    Headline: "large prominent headline",
    Hero: "huge dominant hero headline",
};

// Mirror of compose_text_desc(): ordered, non-empty slots joined with ", ".
export function composeTextDesc(block, fontsById) {
    const slots = [];
    const preset = fontsById?.[block.font_preset_id || ""];
    slots.push((preset?.desc_snippet || "").trim() || "bold clean sans-serif");

    const weight = WEIGHT_PHRASE[block.weight];
    if (weight) slots.push(weight);
    const casePhrase = CASE_PHRASE[block.case];
    if (casePhrase) slots.push(casePhrase);
    const prominence = PROMINENCE_PHRASE[block.prominence];
    if (prominence) slots.push(prominence);

    const color = normHex(block.color);
    if (color) slots.push(`${color} colored letters`);

    const leg = block.legibility || {};
    if (leg.outline) slots.push("with a thin dark outline");
    if (leg.high_contrast) slots.push("high contrast");
    if (leg.solid_block) slots.push("on a solid color block behind the text");
    slots.push("crisp clean edges, readable at small thumbnail size");

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
        style: {
            preset_id: "",
            aesthetics: "",
            lighting: "",
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
    { id: "solid", ru: "Сплошной цвет", en: "Solid color", v: "clean solid color background" },
    { id: "gradient", ru: "Плавный градиент", en: "Smooth gradient", v: "smooth two-tone gradient background" },
    { id: "studio", ru: "Студийный фон", en: "Studio backdrop", v: "seamless studio backdrop with a soft vignette" },
    { id: "bokeh", ru: "Размытие / боке", en: "Blurred bokeh", v: "blurred background with soft glowing bokeh" },
    { id: "marble", ru: "Тёмный мрамор", en: "Dark marble", v: "dark marble surface with subtle veins" },
    { id: "abstract", ru: "Абстрактные формы", en: "Abstract shapes", v: "abstract flowing shapes, layered composition" },
    { id: "nature", ru: "Природа / небо", en: "Nature & sky", v: "natural outdoor scenery with a soft sky" },
    { id: "pattern", ru: "Геометрия / паттерн", en: "Geometric pattern", v: "geometric pattern background, repeating motif" },
    { id: "grunge", ru: "Гранж-текстура", en: "Grunge texture", v: "grungy distressed textured background" },
    { id: "paper", ru: "Бумага / крафт", en: "Paper & craft", v: "textured paper and kraft background" },
    { id: "glow", ru: "Свечение / меш", en: "Glow mesh", v: "soft glowing mesh gradient, dreamy" },
    { id: "minimal", ru: "Минимальный / пусто", en: "Minimal / none", v: "minimal plain background" },
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
];

// Text presets — one-click "tasty" lettering looks for a text block. Each sets
// font_preset_id + weight + case + prominence + color + legibility together.
export const TEXT_PRESETS = [
    { id: "bold_headline", ru: "Жирный заголовок", en: "Bold headline", font_preset_id: "grotesque_black", weight: "Heavy", case: "UPPERCASE", prominence: "Hero", color: "#FFFFFF", legibility: { outline: true, high_contrast: true, solid_block: false } },
    { id: "clean_minimal", ru: "Чистый минимал", en: "Clean minimal", font_preset_id: "geometric_sans", weight: "Regular", case: "As-typed", prominence: "Headline", color: "#0F172A", legibility: { outline: false, high_contrast: false, solid_block: false } },
    { id: "yellow_accent", ru: "Жёлтый акцент", en: "Yellow accent", font_preset_id: "condensed_bold", weight: "Bold", case: "UPPERCASE", prominence: "Headline", color: "#FFD400", legibility: { outline: true, high_contrast: true, solid_block: false } },
    { id: "elegant_serif", ru: "Элегантный сериф", en: "Elegant serif", font_preset_id: "didone_luxury", weight: "Regular", case: "Title", prominence: "Hero", color: "#1A1A1A", legibility: { outline: false, high_contrast: false, solid_block: false } },
    { id: "poster_impact", ru: "Постер-удар", en: "Poster impact", font_preset_id: "display_poster_sans", weight: "Heavy", case: "UPPERCASE", prominence: "Hero", color: "#FFFFFF", legibility: { outline: false, high_contrast: true, solid_block: false } },
    { id: "sticker_block", ru: "Плашка-стикер", en: "Sticker block", font_preset_id: "grotesque_black", weight: "Bold", case: "UPPERCASE", prominence: "Headline", color: "#111111", legibility: { outline: false, high_contrast: true, solid_block: true } },
    { id: "handwritten", ru: "Рукописный", en: "Handwritten", font_preset_id: "brush_marker", weight: "Bold", case: "As-typed", prominence: "Headline", color: "#FFFFFF", legibility: { outline: true, high_contrast: false, solid_block: false } },
    { id: "retro_display", ru: "Ретро-дисплей", en: "Retro display", font_preset_id: "retro_70s_round", weight: "Bold", case: "UPPERCASE", prominence: "Hero", color: "#FF5A36", legibility: { outline: false, high_contrast: false, solid_block: false } },
    { id: "techno_mono", ru: "Техно-моно", en: "Techno mono", font_preset_id: "mono_techno", weight: "Bold", case: "UPPERCASE", prominence: "Body", color: "#00E5A0", legibility: { outline: false, high_contrast: true, solid_block: false } },
    { id: "graffiti", ru: "Граффити", en: "Graffiti", font_preset_id: "graffiti_urban", weight: "Heavy", case: "UPPERCASE", prominence: "Hero", color: "#FFFFFF", legibility: { outline: true, high_contrast: true, solid_block: false } },
    { id: "stencil", ru: "Стенсил", en: "Stencil", font_preset_id: "stencil_block", weight: "Heavy", case: "UPPERCASE", prominence: "Hero", color: "#111111", legibility: { outline: false, high_contrast: true, solid_block: false } },
];

// "Main idea" (high_level_description) presets, 10 per layout, adapted to that
// layout. `v` is the English one-sentence HLD fed to the model; ru/en are the
// dropdown labels. Many feature vivid, emotional characters (a beautiful woman,
// a brutally handsome man) so preset designs come out cinematic and expressive.
// Keyed by the builtin layout id (see ideogram_layouts.json).
export const LAYOUT_BRIEFS = {
    youtube_thumbnail: [
        { ru: "Шок на лице", en: "Shocked reaction", v: "A high-energy YouTube thumbnail: a brutally handsome man with a shocked wide-eyed reaction on the right, dramatic rim light, a huge punchy headline on the left." },
        { ru: "Дерзкая девушка", en: "Bold confident girl", v: "A vibrant thumbnail with a strikingly beautiful young woman smirking with confidence, glowing neon rim light, a bold contrasty headline beside her." },
        { ru: "Деньги и успех", en: "Money & success", v: "A flashy success thumbnail: a confident man in a sharp suit with cash flying behind him, gold accents, an explosive headline." },
        { ru: "До и после", en: "Before & after", v: "A split before/after thumbnail with a transformed, glowing person, a bold arrow and a punchy comparison headline." },
        { ru: "Гнев / спор", en: "Rage / drama", v: "A dramatic confrontation thumbnail: a furious brutal man pointing straight at the viewer, intense red glow, an aggressive headline." },
        { ru: "Восторг / вау", en: "Amazed wow", v: "An exciting reveal thumbnail: a beautiful woman with a delighted, amazed expression, sparkles and glow, a bright energetic headline." },
        { ru: "Геймер в наушниках", en: "Gamer", v: "A gaming thumbnail: an intense focused gamer in headphones lit by colorful RGB neon, explosive action behind, a bold headline." },
        { ru: "Загадка / интрига", en: "Mystery hook", v: "A mysterious thumbnail: a hooded figure half in shadow under a single dramatic light, intriguing atmosphere and a teasing headline." },
        { ru: "Роскошный образ жизни", en: "Luxury lifestyle", v: "A luxury lifestyle thumbnail: a stylish woman beside a supercar at golden hour, aspirational glamour and a bold headline." },
        { ru: "Эксперт-объяснение", en: "Expert explainer", v: "An educational thumbnail: a charismatic presenter gesturing at glowing infographic elements, a clean confident look and a clear headline." },
    ],
    ad_poster: [
        { ru: "Большая распродажа", en: "Big sale", v: "A punchy sale poster with an explosive discount headline, the hero product centered in a spotlight and a vivid color burst." },
        { ru: "Модель с продуктом", en: "Model with product", v: "A glossy advertising poster: a beautiful confident model holding the product in glamorous studio light, a bold headline and price." },
        { ru: "Брутальный герой бренда", en: "Brand hero (man)", v: "A bold poster with a brutal rugged man as the brand hero, dramatic side light, a strong product and a powerful tagline." },
        { ru: "Премиальный запуск", en: "Premium launch", v: "An elegant product-launch poster: the new product floating in soft premium light, minimal luxury type and a refined palette." },
        { ru: "Фестиваль / событие", en: "Event / festival", v: "A vibrant event poster bursting with energy and dynamic shapes, a big date headline and an exciting atmosphere." },
        { ru: "Еда крупным планом", en: "Food hero", v: "An appetizing food poster: a delicious dish in mouth-watering detail with fresh ingredients and a bold tasty headline." },
        { ru: "Фитнес / энергия", en: "Fitness energy", v: "A high-energy fitness poster: an athletic woman mid-motion with dynamic light and sweat, a motivational bold headline." },
        { ru: "Скидка −50%", en: "50% off", v: "A loud discount poster with a giant -50% burst, the product in a beam of light and urgent contrasty colors." },
        { ru: "Услуга / доверие", en: "Trusted service", v: "A clean trustworthy service poster: a friendly professional, a clear benefit headline and a calm confident palette." },
        { ru: "Ретро-постер", en: "Retro poster", v: "A stylish retro advertising poster with vintage textures, bold mid-century type and warm nostalgic colors." },
    ],
    social_post: [
        { ru: "Топ лайфхак", en: "Top lifehack", v: "A clean square social post with a short bold lifehack headline, a simple friendly illustration and a small swipe footer." },
        { ru: "Цитата дня", en: "Quote of the day", v: "An inspiring quote post: elegant typography over a soft gradient, a small accent mark and a calm aesthetic." },
        { ru: "Эмоция девушки", en: "Girl's mood", v: "A lifestyle social post: a beautiful woman laughing candidly in soft natural light, a warm authentic mood and a short caption." },
        { ru: "Брутальный портрет", en: "Bold male portrait", v: "A striking square portrait of a brutal stylish man with an intense gaze, moody studio light and a short punchy caption." },
        { ru: "Анонс / новинка", en: "Announcement", v: "A bold announcement post with a big NEW headline, a simple product hint and energetic accent shapes." },
        { ru: "Большая цифра", en: "Big number fact", v: "A bold data post built around one huge number, a short caption and clean geometric accents." },
        { ru: "До / после", en: "Before / after", v: "A clean before/after social post showing a transformation, a divider line and a short result caption." },
        { ru: "Минимал-арт", en: "Minimal art", v: "A minimal aesthetic post: a single elegant object on lots of negative space, refined type and a calm palette." },
        { ru: "Праздник", en: "Celebration", v: "A festive greeting post with a warm celebratory mood, soft confetti accents and a heartfelt short message." },
        { ru: "Юмор / мем", en: "Fun / meme", v: "A playful humorous post with a punchy funny line, a bold expressive character and bright cheeky colors." },
    ],
    web_banner: [
        { ru: "SaaS-герой", en: "SaaS hero", v: "A clean landing hero: a bold value headline and CTA on the left, a sleek product UI mockup on a device on the right, airy whitespace." },
        { ru: "Команда / люди", en: "Team / people", v: "A warm landing hero with a friendly team smiling in soft daylight, a clear headline and a CTA button." },
        { ru: "Девушка-амбассадор", en: "Brand ambassador", v: "A bright hero banner: a beautiful confident woman as the brand ambassador on the right, a bold benefit headline and CTA on the left." },
        { ru: "Распродажа / промо", en: "Sale promo", v: "A high-contrast promo banner with a big discount headline, the product and an urgent CTA in an energetic palette." },
        { ru: "Мобильное приложение", en: "App showcase", v: "A modern app-launch hero: a phone mockup with a slick UI, short benefit copy and a download CTA over a clean gradient." },
        { ru: "Премиум-бренд", en: "Premium brand", v: "An elegant minimal hero: a premium product in soft light, a refined serif headline and generous negative space." },
        { ru: "Курс / вебинар", en: "Course / webinar", v: "An education hero: a confident expert on the right, a clear course headline and a sign-up CTA on the left." },
        { ru: "Тёмная тех-тема", en: "Dark tech", v: "A sleek dark-mode tech hero with glowing accents, a bold headline, a futuristic product render and a CTA." },
        { ru: "Эко / природа", en: "Eco / nature", v: "A fresh natural hero with organic textures and greenery, a calm headline and a soft CTA in an earthy palette." },
        { ru: "Чёрная пятница", en: "Black Friday", v: "A bold Black Friday hero: a dramatic dark background, a huge sale headline, a glowing product and an urgent CTA." },
    ],
    book_cover: [
        { ru: "Деловой бестселлер", en: "Business bestseller", v: "An elegant business book cover: a bold title on top, a striking conceptual illustration and the author name at the bottom." },
        { ru: "Героиня романа", en: "Novel heroine", v: "A dramatic novel cover featuring a beautiful woman with an emotional gaze, atmospheric lighting and elegant title typography." },
        { ru: "Брутальный триллер", en: "Thriller hero", v: "A tense thriller cover: a brutal man's silhouette in moody shadow with dramatic rim light and a bold ominous title." },
        { ru: "Фэнтези-мир", en: "Fantasy world", v: "An epic fantasy cover with a breathtaking magical landscape, a lone heroic figure and ornate title lettering." },
        { ru: "Саморазвитие", en: "Self-help", v: "A bright uplifting self-help cover: a bold motivational title, a clean symbolic illustration and a warm optimistic palette." },
        { ru: "Тёмный детектив", en: "Noir mystery", v: "A noir detective cover with a rain-soaked moody scene, a mysterious silhouette and classic dramatic typography." },
        { ru: "Любовный роман", en: "Romance", v: "A tender romance cover: a beautiful couple in a soft emotional embrace at golden hour and an elegant flowing title." },
        { ru: "Научпоп", en: "Science / non-fiction", v: "A clean science non-fiction cover with an elegant conceptual illustration, a confident modern title and a refined palette." },
        { ru: "Детская книга", en: "Children's book", v: "A charming children's book cover with a cute friendly character, a playful colorful illustration and a rounded title." },
        { ru: "Минимал-обложка", en: "Minimal cover", v: "A striking minimal book cover: one bold symbolic shape on a refined color field and elegant restrained typography." },
    ],
    logo: [
        { ru: "Чистый текстовый логотип", en: "Clean wordmark", v: "A clean centered wordmark logo with a small minimal icon above and a tagline below, balanced and legible." },
        { ru: "Геометрический знак", en: "Geometric mark", v: "A modern logo with a bold geometric icon mark, a confident sans-serif wordmark and generous whitespace." },
        { ru: "Винтажный значок", en: "Vintage badge", v: "A vintage emblem logo: a circular badge with classic lettering, a small crest icon and a refined retro feel." },
        { ru: "Премиум-монограмма", en: "Luxury monogram", v: "An elegant luxury monogram logo with a refined serif initial, a thin frame and a premium minimal palette." },
        { ru: "Игривый бренд", en: "Playful brand", v: "A friendly playful logo with a rounded wordmark, a cute simple mascot icon and cheerful colors." },
        { ru: "Тех-стартап", en: "Tech startup", v: "A sleek tech-startup logo with a clean geometric mark, a modern wordmark and a confident gradient accent." },
        { ru: "Кофейня / крафт", en: "Coffee / craft", v: "A cozy craft logo with a hand-drawn icon, warm rustic lettering and an artisanal vibe." },
        { ru: "Спортивный жирный", en: "Sport bold", v: "A bold athletic logo with a strong angular wordmark, a dynamic icon and high-energy contrast." },
        { ru: "Бьюти / эстетика", en: "Beauty / elegant", v: "A delicate beauty logo with a thin elegant wordmark, a subtle floral mark and a soft refined palette." },
        { ru: "Минимал-иконка", en: "Minimal icon", v: "A minimal logo: one clever simple icon mark with a quiet wordmark and lots of negative space." },
    ],
    music_cover: [
        { ru: "Атмосферный абстракт", en: "Atmospheric abstract", v: "A square album cover with striking abstract atmospheric art matching the genre mood, with the title and artist at the bottom." },
        { ru: "Портрет артистки", en: "Female artist portrait", v: "A moody album cover: an emotional close-up of a beautiful singer in dramatic light, with title and artist name below." },
        { ru: "Брутальный рэп", en: "Hip-hop / bold man", v: "A bold hip-hop cover: a brutal confident man with an intense presence, gritty urban texture and heavy title type." },
        { ru: "Неон-синтвейв", en: "Synthwave neon", v: "A retro synthwave cover with neon grids, a glowing sunset and chrome lettering in a nostalgic 80s mood." },
        { ru: "Лоу-фай уют", en: "Lo-fi cozy", v: "A cozy lo-fi cover: a calm illustrated scene with warm lamplight and rain and a soft nostalgic title." },
        { ru: "Тёмный метал", en: "Dark metal", v: "A dark metal cover with dramatic ominous art, harsh textures and bold aggressive lettering." },
        { ru: "Инди-акварель", en: "Indie watercolor", v: "A dreamy indie cover with a delicate watercolor illustration, soft pastels and a hand-lettered title." },
        { ru: "Танцевальный поп", en: "Dance pop", v: "A vibrant pop cover bursting with color and energy, a glamorous figure and a bold playful title." },
        { ru: "Джаз-нуар", en: "Jazz noir", v: "A classy jazz cover with a smoky noir scene, a warm spotlight and elegant vintage typography." },
        { ru: "Эмбиент-минимал", en: "Ambient minimal", v: "A minimal ambient cover: a single serene gradient field, a tiny title and a calm meditative mood." },
    ],
    product_card: [
        { ru: "Чистая студия", en: "Clean studio", v: "An e-commerce product card: the product centered on a seamless studio background, sharp focus, with name and a bold price." },
        { ru: "Косметика + модель", en: "Cosmetics + model", v: "A beauty product card: the cosmetic product with a beautiful model's glowing skin behind it in soft light, with name and price." },
        { ru: "Гаджет / техно", en: "Gadget / tech", v: "A sleek tech product card: a gadget floating with a soft reflection on a dark gradient, with a clean name and price." },
        { ru: "Еда / вкусно", en: "Tasty food", v: "A delicious food product card: the dish with fresh ingredients in appetizing light, with a bold name and price." },
        { ru: "Мода / одежда", en: "Fashion item", v: "A stylish fashion product card: a clothing item on a clean backdrop with editorial flair, name and price." },
        { ru: "Эко / натурально", en: "Eco / natural", v: "A natural product card: the product among organic textures and greenery in warm earthy light, with name and price." },
        { ru: "Премиум-люкс", en: "Premium luxury", v: "A luxury product card: the premium item in refined dramatic light on a dark field, with an elegant name and price." },
        { ru: "Скидка / акция", en: "On sale", v: "A promo product card: the product with a bold discount badge, an energetic accent and a striking sale price." },
        { ru: "Напиток / свежесть", en: "Drink / fresh", v: "A refreshing drink product card: the beverage with splashes and condensation in vivid fresh colors, with name and price." },
        { ru: "Хендмейд / крафт", en: "Handmade / craft", v: "A cozy handmade product card: the artisanal item on rustic textures in warm authentic light, with name and price." },
    ],
    packaging_label: [
        { ru: "Премиум-этикетка", en: "Premium label", v: "A premium packaging label: the brand on top, a refined bottle or box mockup, the product name and a small detail line." },
        { ru: "Крафт / эко", en: "Kraft / eco", v: "An eco packaging label with natural kraft textures, hand-drawn botanical accents and warm organic typography." },
        { ru: "Косметика-люкс", en: "Luxury cosmetics", v: "A luxury cosmetics label: an elegant jar or bottle in soft light, a refined serif brand and a delicate palette." },
        { ru: "Крафтовое пиво", en: "Craft beer", v: "A bold craft-beer label with a striking illustrated emblem, vintage lettering and a punchy color scheme." },
        { ru: "Кофе / зерно", en: "Coffee pack", v: "A warm coffee packaging label with a rich illustrated mark, cozy earthy tones and confident brand type." },
        { ru: "Снек / яркий", en: "Snack / bold", v: "A fun snack package label bursting with appetizing color, bold playful type and an energetic mascot." },
        { ru: "Парфюм / минимал", en: "Perfume minimal", v: "A minimal perfume label: an elegant flacon on a soft field, thin refined typography and a luxurious calm." },
        { ru: "Фарма / чистый", en: "Pharma / clean", v: "A clean clinical product label: a trustworthy bottle, precise legible typography and a fresh medical palette." },
        { ru: "Винтаж / аптека", en: "Vintage apothecary", v: "A vintage apothecary label with ornate frames, classic serif lettering and an aged refined texture." },
        { ru: "Спорт-питание", en: "Sports nutrition", v: "A bold sports-nutrition label: a powerful container with dynamic accents, strong type and high-energy contrast." },
    ],
    merch_print: [
        { ru: "Дерзкий слоган", en: "Bold slogan", v: "A bold merch print with a big punchy slogan, a striking central graphic and a small accent line in high contrast." },
        { ru: "Брутальный маскот", en: "Bold mascot", v: "A high-contrast t-shirt print with a fierce brutal mascot character, heavy lettering and an edgy streetwear vibe." },
        { ru: "Девушка / поп-арт", en: "Pop-art girl", v: "A pop-art merch print with a striking stylish woman illustration, bold outlines, halftone dots and punchy colors." },
        { ru: "Скейт / стрит", en: "Skate / street", v: "A gritty street-style print with a rebellious graphic, distressed textures and a bold slogan." },
        { ru: "Природа / горы", en: "Outdoor / mountains", v: "An outdoor adventure print with a scenic mountain illustration, vintage badge lettering and an earthy palette." },
        { ru: "Аниме-вайб", en: "Anime vibe", v: "An anime-style merch print with an expressive character, clean cel-shaded art and a bold catchphrase." },
        { ru: "Готика / тёмный", en: "Dark gothic", v: "A dark gothic print with an intricate ominous illustration, sharp blackletter type and a moody palette." },
        { ru: "Юмор / мем", en: "Funny meme", v: "A funny merch print with a quirky humorous character, a witty slogan and bright cheeky colors." },
        { ru: "Ретро 80-е", en: "Retro 80s", v: "A retro 80s print with a neon sunset, bold chrome lettering and a nostalgic synthwave mood." },
        { ru: "Минимал-лайн", en: "Minimal line art", v: "A minimal line-art merch print: a single elegant continuous-line illustration and a small refined slogan." },
    ],
};

// Which OBJECT_PRESET a character brief should drop onto the subject object, so
// the rendered figure matches the idea (e.g. "Bold girl" → a woman, not a man).
// Keyed by the brief's `en` label (unique across LAYOUT_BRIEFS). Briefs not
// listed here leave the layout's default object untouched.
export const BRIEF_SUBJECT = {
    "Shocked reaction": "brutal_man",
    "Bold confident girl": "bold_girl",
    "Money & success": "business_pro",
    "Before & after": "happy_smile",
    "Rage / drama": "brutal_man",
    "Amazed wow": "glamour_woman",
    "Gamer": "gamer",
    "Luxury lifestyle": "glamour_woman",
    "Model with product": "glamour_woman",
    "Brand hero (man)": "brutal_man",
    "Fitness energy": "athlete",
    "Girl's mood": "bold_girl",
    "Bold male portrait": "brutal_man",
    "Brand ambassador": "glamour_woman",
    "Novel heroine": "glamour_woman",
    "Thriller hero": "brutal_man",
    "Romance": "couple",
    "Female artist portrait": "glamour_woman",
    "Hip-hop / bold man": "brutal_man",
    "Cosmetics + model": "glamour_woman",
    "Pop-art girl": "bold_girl",
    "Bold mascot": "mascot",
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
        hld: "Main idea", aesthetics: "Mood & vibe",
        lighting: "Lighting", art_style: "Art style",
        image_palette: "Image colors (up to {n})", background: "Background", add_color: "Add color",
        hld_hint: "One sentence describing the whole image — the model leans on this most. E.g. a bold summer-sale poster for a sneaker brand.",
        aesthetics_hint: "The overall feel in a few words. Example: bold and punchy, calm and minimal, retro, luxurious. Safe to leave blank.",
        lighting_hint: "How the scene is lit. Example: bright daylight, soft studio light, moody shadows, neon glow. Safe to leave blank.",
        art_style_hint: "Shown for every image type except Photo — the drawing/rendering style. Example: flat vector, watercolor, low-poly 3D, bold poster graphics.",
        image_palette_hint: "Optional palette the whole image sticks to — add up to {n} colors. Leave empty to let Ideogram choose.",
        background_hint: "What sits behind everything — the backdrop behind your text and objects. Example: smooth orange-to-pink gradient, blurred city street, dark marble. Leave empty for a plain background.",
        block_text_title: "Text block", block_obj_title: "Object (obj)",
        text_literal: "Text (rendered literally)",
        font_preset: "Font preset (the description is the only real lever)",
        weight: "Weight", case: "Case", size_words: "Size (in words)", text_color: "Text color",
        legibility: "Legibility", leg_outline: "Outline", leg_contrast: "Contrast", leg_block: "Block",
        visual_only: "Manual text (visual-only — empty placeholder for a hand overlay)",
        override: "Extra description (override, appended last)",
        block_palette: "Block palette (up to {n})", desc_preview: "Final description (desc) for the model:",
        obj_desc: "Object description (desc)",
        select_block: "Select a block on the canvas, or add a new one (+ Text / + Object).",
        visual_only_preview: "(visual-only) this area becomes an empty placeholder — add the text by hand in Figma/Photoshop.",
        save_as_layout: "Save as template",
        export_btn: "⬇ Export", import_btn: "⬆ Import",
        import_done: "Imported: {n}", import_empty: "No valid presets in the file",
        preset_name_prompt: "Preset name:", custom_tag: "custom",
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
        tip_save_as_layout: "Save your current blocks, aspect ratio and background as a reusable template you can pick again later.",
        tip_export_layout: "Download this layout template as a JSON file to back it up or share it with someone else.",
        tip_import_layout: "Load layout template files from your computer so they show up in the template picker above.",
        tip_add_color: "Add a color: opens the color picker so you can pin one more shade to this palette.",
        tip_palette_swatch: "Click a color to remove it from the palette.",
        tip_obj_desc: "Describe what this graphic shows — e.g. a smiling sneaker, a coffee cup. Ideogram draws it here.",
        tip_text_literal: "Type the exact words you want to appear — these letters are drawn on the image as-is.",
        tip_font_preset: "Pick the lettering style for this text. A ⚠ means that font is shaky with Russian letters.",
        tip_weight: "How thick the letters look — from delicate Thin up to chunky Heavy.",
        tip_case: "How the letters are cased: as you typed, ALL CAPS, or Title Style.",
        tip_size_words: "How big and loud this text should feel — a tiny caption or a giant hero headline.",
        tip_text_color: "Pick the color of these letters — tap the swatch to choose any shade.",
        tip_leg_outline: "Adds a thin dark edge around the letters so they stand out on busy backgrounds.",
        tip_leg_contrast: "Asks Ideogram to make the text really pop against whatever is behind it.",
        tip_leg_block: "Puts the text on a solid color bar behind it, like a sticker — great for readability.",
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
        object_preset: "Object preset", text_preset: "Text style preset",
        tip_object_preset: "Pick a ready subject for this object — a character or a hero object. It replaces the description.",
        tip_text_preset: "Pick a ready lettering look — font, weight, case and color in one click.",
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
        hld: "Главная идея", aesthetics: "Настроение и вайб",
        lighting: "Освещение", art_style: "Художественный стиль",
        image_palette: "Цвета изображения (до {n})", background: "Фон", add_color: "Добавить цвет",
        hld_hint: "Одно предложение про всю картинку — модель опирается на него сильнее всего. Например: яркий постер летней распродажи кроссовок.",
        aesthetics_hint: "Общее ощущение в паре слов. Например: дерзко и сочно, спокойно и минимально, ретро, премиально. Можно оставить пустым.",
        lighting_hint: "Как освещена сцена. Например: яркий дневной свет, мягкий студийный, драматичные тени, неоновое свечение. Можно оставить пустым.",
        art_style_hint: "Показывается для всех типов, кроме «Фото» — стиль отрисовки. Например: плоский вектор, акварель, low-poly 3D, плакатная графика.",
        image_palette_hint: "Необязательная палитра, которой держится вся картинка — до {n} цветов. Оставьте пустым — Ideogram подберёт сам.",
        background_hint: "Что находится позади всего — задний фон под текстом и объектами. Например: плавный градиент из оранжевого в розовый, размытая улица, тёмный мрамор. Пусто — простой фон.",
        block_text_title: "Текстовый блок", block_obj_title: "Объект (obj)",
        text_literal: "Текст (рендерится буква-в-букву)",
        font_preset: "Шрифт-пресет (описание — единственный реальный рычаг)",
        weight: "Вес", case: "Регистр", size_words: "Размер (словами)", text_color: "Цвет текста",
        legibility: "Читаемость", leg_outline: "Обводка", leg_contrast: "Контраст", leg_block: "Плашка",
        visual_only: "Текст вручную (пустая плашка под ручной оверлей)",
        override: "Доп. описание (добавляется в конец)",
        block_palette: "Палитра блока (до {n})", desc_preview: "Итоговое описание для модели:",
        obj_desc: "Описание объекта (desc)",
        select_block: "Выберите блок на холсте или добавьте новый (+ Текст / + Объект).",
        visual_only_preview: "(visual-only) область станет пустой плашкой без текста — добавьте надпись вручную в Figma/Photoshop.",
        save_as_layout: "Сохранить как шаблон",
        export_btn: "⬇ Экспорт", import_btn: "⬆ Импорт",
        import_done: "Импортировано: {n}", import_empty: "В файле нет валидных пресетов",
        preset_name_prompt: "Имя пресета:", custom_tag: "свой",
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
        tip_save_as_layout: "Сохраните текущие блоки, формат и фон как свой шаблон, чтобы быстро применять его снова.",
        tip_export_layout: "Скачайте этот лейаут-шаблон в виде JSON-файла, чтобы сохранить про запас или передать коллеге.",
        tip_import_layout: "Загрузите файлы лейаут-шаблонов с компьютера — они появятся в списке шаблонов выше.",
        tip_add_color: "Добавить цвет: откроет палитру выбора, чтобы закрепить ещё один оттенок в наборе.",
        tip_palette_swatch: "Нажмите на цвет, чтобы убрать его из палитры.",
        tip_obj_desc: "Опишите, что здесь нарисовать — например, улыбающийся кроссовок или чашка кофе. Ideogram это и нарисует.",
        tip_text_literal: "Впишите точные слова, которые должны появиться — эти буквы рисуются на картинке буква-в-букву.",
        tip_font_preset: "Выберите стиль букв для этого текста. Значок ⚠ — шрифт плохо дружит с кириллицей.",
        tip_weight: "Насколько толстые буквы — от тонких до жирных и совсем мощных.",
        tip_case: "Как оформить буквы: как набрали, ВСЕ ЗАГЛАВНЫЕ или С Заглавных.",
        tip_size_words: "Насколько крупным и заметным будет текст — от мелкой подписи до огромного заголовка.",
        tip_text_color: "Выберите цвет этих букв — нажмите на квадратик и подберите любой оттенок.",
        tip_leg_outline: "Добавит тонкую тёмную обводку вокруг букв, чтобы они читались на пёстром фоне.",
        tip_leg_contrast: "Попросит Ideogram сделать текст хорошо заметным на фоне за ним.",
        tip_leg_block: "Положит текст на плотную цветную плашку, как на наклейке — отлично для читаемости.",
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
        object_preset: "Пресет объекта", text_preset: "Пресет стиля текста",
        tip_object_preset: "Выберите готовый объект — персонажа или герой-предмет. Заменяет описание.",
        tip_text_preset: "Выберите готовый вид надписи — шрифт, вес, регистр и цвет в один клик.",
        clear_confirm: "Очистить весь дизайн? Блоки, стиль, фон и цвета будут сброшены.",
        tip_layers: "Ваши блоки стопкой — верх списка это передний план картинки.",
        tip_layer_row: "Клик — выбрать; тяните вверх/вниз, чтобы менять перекрытие. Выше = ближе к переднему плану.",
        tip_layer_up: "На шаг к переднему плану.",
        tip_layer_down: "На шаг к заднему плану.",
    },
};

const SEG_LABELS = {
    weight: { en: { Thin: "Thin", Regular: "Regular", Bold: "Bold", Heavy: "Heavy" },
              ru: { Thin: "Тонкий", Regular: "Обычный", Bold: "Жирный", Heavy: "Чёрный" } },
    case: { en: { "As-typed": "As-typed", UPPERCASE: "UPPERCASE", Title: "Title" },
            ru: { "As-typed": "Как есть", UPPERCASE: "ВЕРХНИЙ", Title: "Заголовок" } },
    prominence: { en: { Caption: "Caption", Body: "Body", Headline: "Headline", Hero: "Hero" },
                  ru: { Caption: "Подпись", Body: "Текст", Headline: "Заголовок", Hero: "Гигант" } },
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
                desc: b.desc_en || "", role: b.role || "", color_palette: [],
            };
        }
        const text = lang === "en" ? (b.text_en ?? b.text_ru ?? "") : (b.text_ru ?? b.text_en ?? "");
        return {
            id: makeBlockId(), type: "text", rect, text,
            font_preset_id: b.font_preset_id || "grotesque_black",
            weight: b.weight || "Bold", case: b.case || "As-typed", prominence: b.prominence || "Headline",
            color: normHex(b.color) || "#FFFFFF",
            legibility: { outline: true, high_contrast: true, solid_block: false },
            visual_only: false, desc_override: "", role: b.role || "", color_palette: [],
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
