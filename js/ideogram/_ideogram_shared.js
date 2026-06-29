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
    material_typography: [
        { ru: "Ледяные буквы", en: "Frozen Ice", v: "A single giant short word fills the entire frame, its letters carved from translucent blue glacial ice with frost-feathered surfaces, sharp crystalline edges and tiny trapped air bubbles, faint cold mist curling at the bases, set against a clean dark teal void lit with a soft cold rim light." },
        { ru: "Сочные фрукты", en: "Ripe Fruit", v: "One bold short headline towers across the frame with each letter sculpted from fresh ripe fruit — glossy juicy strawberries, citrus segments and dewy berries pressed into the strokes — droplets of juice beading on the surfaces against a clean creamy backdrop lit like a bright food commercial." },
        { ru: "Расплавленное золото", en: "Molten Gold", v: "A massive short word dominates the frame rendered in flowing molten gold, mirror-bright liquid metal dripping and pooling along the letterforms with rich warm highlights and deep amber shadows, floating in a minimal jet-black studio space." },
        { ru: "Неоновое стекло", en: "Neon Glass", v: "One short punchy word fills the frame as glowing neon glass tubes bent into the letters, electric magenta and cyan light blooming with soft halos and gentle reflections on a dark wet floor against an otherwise empty midnight backdrop." },
        { ru: "Прозрачное стекло", en: "Clear Glass", v: "A giant short headline spans the frame built from thick transparent glass, the letters refracting and bending the light behind them with caustic glints and crisp bevelled edges, resting on a pale seamless gradient with delicate soft shadows." },
        { ru: "Цветущие буквы", en: "Blooming Flowers", v: "One large short word fills the frame formed entirely from blooming flowers and lush green foliage, dense rose and peony petals packed into each letter with tiny leaves spilling at the edges, photographed against a soft pastel sky." },
        { ru: "Пылающий огонь", en: "Blazing Fire", v: "A bold short word commands the frame sculpted from blazing fire and glowing embers, flames licking up the letterforms with bright orange cores fading to smoky edges and sparks drifting into a deep black night." },
        { ru: "Хром и жидкий металл", en: "Liquid Chrome", v: "One giant short headline fills the frame as flawless mirror chrome, the liquid-metal letters reflecting a soft studio gradient with smooth rounded curves and razor-sharp specular highlights, isolated on a minimal pearl-grey backdrop." },
        { ru: "Брызги воды", en: "Water Splash", v: "A massive short word spans the frame shaped from a splash of crystal-clear water, the letters frozen mid-motion with translucent flowing strokes, scattering droplets and glassy highlights against a fresh aqua-blue void." },
        { ru: "Мягкий пушистый мех", en: "Fluffy Fur", v: "One short cuddly headline fills the frame rendered in soft fluffy fur, each letter a plush mound of fine warm-toned hairs catching gentle light with a cozy depth-of-field blur, set on a clean muted beige background." },
    ],
    neon_sign: [
        { ru: "Открыто — розовый неон", en: "OPEN — pink neon", v: "A glowing pink neon-tube sign spelling the word OPEN in flowing cursive script, buzzing softly against a dark weathered red-brick wall at night, its rosy light bleeding warm halos across the mortar and casting a faint reflection in a wet sidewalk below." },
        { ru: "Бар — синяя вывеска", en: "BAR — electric blue", v: "An electric-blue neon sign reading BAR in bold blocky tubes, humming above a shadowed brick alley wall, cold cobalt light spilling across the rough masonry while a thin coil of cigarette smoke drifts through the glow." },
        { ru: "Открыто 24 часа", en: "OPEN 24 HOURS", v: "A vintage diner neon sign reading OPEN 24 HOURS, the words stacked in warm amber and ruby tubes with a small glowing arrow beneath, mounted on a grimy dark-brick facade slick with light drizzle that scatters the colored light into soft streaks." },
        { ru: "Розовое сердце", en: "Glowing heart", v: "A single oversized neon heart outlined in hot magenta tubes pulsing on a moody charcoal-brick wall, its pink radiance pooling into a soft circular bloom on the bricks while one tube flickers as if on the edge of burning out." },
        { ru: "Коктейли — мятный неон", en: "COCKTAILS — mint", v: "A retro neon sign spelling COCKTAILS in elegant minty-green script tubes beside a tilted glowing martini-glass icon, glowing on a dim textured brick wall, its cool emerald light catching every crack and chip in the old painted bricks." },
        { ru: "Закрыто — красный", en: "CLOSED — red", v: "A tired red neon sign reading CLOSED in slumping handwritten tubes, half its glow dimmed and one letter dark, fixed to a cold rain-darkened brick wall at night with a lonely crimson reflection trembling in a puddle below." },
        { ru: "Стрелка налево", en: "Neon arrow", v: "A bright yellow-orange neon arrow pointing left, its chasing tube segments suggesting motion, mounted on a dark soot-stained brick wall in a narrow night alley where the warm light rakes sharply across the rough brick texture." },
        { ru: "Мечтай — фиолетовый", en: "DREAM — purple", v: "A dreamy violet-and-pink neon sign spelling DREAM in soft rounded cursive tubes, glowing gently against a deep indigo-shadowed brick wall, the dual-tone light blending into a hazy lavender halo that softens the gritty masonry behind it." },
        { ru: "Кофе — оранжевый", en: "COFFEE — amber", v: "A cozy amber neon sign reading COFFEE in warm hand-script tubes above a tiny glowing steaming-cup icon, mounted on a dark espresso-brown brick wall at night, its honeyed light wrapping the bricks in an inviting golden warmth." },
        { ru: "Сломанный отель", en: "Flickering HOTEL", v: "A weathered turquoise neon sign reading HOTEL in tall narrow tubes, one letter buzzing and stuttering with a dying flicker, bolted to a grimy noir brick wall at night where the unsteady teal glow flares and fades across the damp stone." },
    ],
    food_typography_ad: [
        { ru: "Арбузный заголовок", en: "Watermelon Headline", v: "A bold summer headline spelling 'JUICY' built from glistening watermelon flesh studded with glossy black seeds and dripping pink juice, beside a tall sweating glass of watermelon cooler and a crisp price tag, on a sun-warmed coral backdrop scattered with chunks of green rind." },
        { ru: "Цитрусовый взрыв", en: "Citrus Splash", v: "A zesty headline reading 'FRESH' formed from vivid orange and lemon segments bursting with a fine citrus mist, surrounded by a chilled bottle of cold-pressed juice and a clean price callout, against a bright tangerine background flecked with droplets and glossy green leaves." },
        { ru: "Ягодный десерт", en: "Berry Indulgence", v: "A tempting headline spelling 'SWEET' sculpted from plump strawberries, blueberries and raspberries glazed with a sheen of cream, next to a layered berry parfait in a tall glass and a clean buy-now button, on a soft blush-pink studio surface dusted with fine sugar." },
        { ru: "Тропический микс", en: "Tropical Mix", v: "An exotic headline reading 'PARADISE' built from sliced mango, kiwi, dragonfruit and pineapple wedges glistening under bright sun, paired with a frosty smoothie bowl and a bold price banner, on a turquoise backdrop framed by monstera leaves and clinging water beads." },
        { ru: "Хрустящее зелёное яблоко", en: "Crisp Green Apple", v: "A snappy headline spelling 'CRISP' carved from crunchy green apple slices with bright dewy skin and a tart spritz of mist, beside a whole mirror-shiny apple and a chalkboard-style price, on a cool mint backdrop with crystalline droplets catching the light." },
        { ru: "Виноградная роскошь", en: "Grape Luxe", v: "An elegant headline reading 'PURE' formed from dewy clusters of deep-purple and emerald grapes wearing a frosted bloom, alongside a premium bottle of grape nectar and a refined price plate, on a moody plum backdrop bathed in soft vineyard light." },
        { ru: "Гранатовая энергия", en: "Pomegranate Power", v: "A vibrant headline spelling 'BOOST' built from glossy ruby pomegranate arils glistening like cut jewels with a few crimson splashes, next to a cracked-open pomegranate half and an energetic price burst, on a deep garnet backdrop streaked with running juice." },
        { ru: "Утренний завтрак", en: "Breakfast Fresh", v: "A cheerful headline reading 'MORNING' assembled from banana coins, peach slices and bright berries swirled into creamy yogurt, beside a wholesome granola bowl and a friendly price sticker, on a warm cream tabletop bathed in soft sunrise light." },
        { ru: "Манговое лето", en: "Mango Season", v: "A juicy headline spelling 'RIPE' sculpted from golden mango cubes oozing nectar with velvety skin highlights, paired with a creamy mango lassi and a summery price tag, on a saffron-yellow backdrop scattered with mint leaves and sticky droplets." },
        { ru: "Вишнёвый соблазн", en: "Cherry Temptation", v: "A playful headline reading 'YUM' formed from glossy red cherries on slender stems with mirror-bright skins and tiny clinging drips, beside a swirl of cherry sorbet in a cone and a bold sale price, on a candy-red backdrop sparkling with sugar specks." },
    ],
    logo_emblem: [
        { ru: "Кофейня — горный обжарщик", en: "Mountain Coffee Roastery", v: "A vintage circular coffee-roastery badge on a warm cream background, curved top text \"MOUNTAIN ROAST\" and bottom text \"EST. 1974\", a central engraved emblem of a steaming coffee cup framed by twin mountain peaks, double pinstripe rings and tiny star separators, rendered in deep espresso brown and burnt amber #3B2417 and #C8843A with subtle aged-ink texture." },
        { ru: "Крафтовая пивоварня", en: "Craft Brewery Seal", v: "A bold craft-brewery emblem on a deep navy background, arched top text \"IRON HOPS\" and lower banner \"BREWING CO.\", a centred icon of crossed wheat sheaves over a foaming beer barrel, surrounded by a beaded ring and hop-leaf flourishes, painted in antique gold and oxblood red #D4A017 and #7A1F1F with a hand-stamped letterpress feel." },
        { ru: "Барбершоп для джентльменов", en: "Barbershop Emblem", v: "A classic barbershop badge on a charcoal background, curved top text \"SHARP & CO.\" and bottom text \"GROOMING\", a central icon of crossed straight razor and comb beneath a striped barber pole, encircled by a fine rope border, rendered in ivory and brushed steel blue #EDE6D6 and #5C7A99 with a crisp engraved line style." },
        { ru: "Серфинг и океан", en: "Surf Coast Badge", v: "A sun-faded surf-club badge on a sandy teal background, arched top text \"WILD COAST\" and bottom text \"SURF CLUB\", a central emblem of a breaking wave cradling a vintage longboard with a rising sun behind, framed by a dotted ring and small palm motifs, in washed turquoise and coral #2E8B8B and #E8704A with a soft retro screen-print grain." },
        { ru: "Горный поход — заповедник", en: "Wilderness Outdoors Crest", v: "A rugged outdoor-adventure crest on a forest-green background, curved top text \"GREAT NORTH\" and lower text \"TRAIL CO.\", a central icon of a pine tree before a snow-capped mountain under a compass star, bordered by a notched ring and tiny arrowheads, rendered in cream and burnt orange #F2EAD3 and #C25B2C with a worn enamel-pin look." },
        { ru: "Ремесленная пекарня", en: "Artisan Bakery Mark", v: "A warm artisan-bakery badge on a soft wheat-cream background, arched top text \"GOLDEN CRUST\" and bottom text \"BAKED DAILY\", a central emblem of a crusty round loaf with a wheat stalk and rolling pin crossed beneath it, ringed by a delicate scalloped border, in toasted brown and honey gold #6B4226 and #E0A53B with a gentle vintage paper texture." },
        { ru: "Мотоклуб и гараж", en: "Motorcycle Garage Patch", v: "A tough motorcycle-garage emblem on a matte black background, curved top text \"ROUTE 66\" and bottom banner \"MOTOR WORKS\", a central icon of a winged engine piston with crossed wrenches and a single flame, encircled by a heavy chain-link ring, rendered in chrome silver and racing red #C0C0C0 and #B22222 with a distressed sticker finish." },
        { ru: "Винодельня и виноград", en: "Vineyard Estate Seal", v: "An elegant vineyard-estate seal on a deep burgundy background, arched top text \"VALLE D'ORO\" and bottom text \"WINE ESTATE\", a central emblem of a grape cluster draped over a curling vine with a sunlit hilltop villa, framed by a fine laurel wreath, in aged gold and dusty plum #C9A24B and #4E2A3E with a refined embossed-foil texture." },
        { ru: "Морской — компас и якорь", en: "Nautical Compass Badge", v: "A maritime navigation badge on a stormy slate-blue background, curved top text \"NORTH STAR\" and bottom text \"SAIL & SEA\", a central icon of a brass compass rose over a crossed anchor and oar, ringed by a twisted rope border with tiny ship wheels, rendered in pale sea-foam and weathered brass #DCE6E4 and #B08D4C with a nautical-chart engraving style." },
        { ru: "Острый соус — огненный перец", en: "Hot Sauce Firebrand", v: "A fiery hot-sauce brand badge on a deep crimson background, arched top text \"DRAGON HEAT\" and bottom text \"SMALL BATCH\", a central emblem of a flaming chili pepper wreathed in licking flames above a stylised skull, encircled by a jagged sunburst ring, in molten orange and charred black #F25C1E and #1A0E0A with a bold woodcut-poster grain." },
    ],
    logo_mascot: [
        { ru: "Лисёнок-обжарщик кофе", en: "Coffee-roaster fox", v: "A friendly round-cheeked fox mascot clutching a steaming espresso cup, centered above a clean bold wordmark whose letters are formed from rich glossy roasted coffee beans with deep oily highlights, plus a small tagline beneath, on a warm cream lockup with a palette of #2E1A0F, #C9702E and #F2E2C4." },
        { ru: "Робот-доставщик пиццы", en: "Pizza delivery robot", v: "A cheerful boxy delivery robot mascot balancing a fresh pizza slice, set above a confident chunky wordmark whose letters are built from melted stretchy mozzarella with golden toasted crust edges and oozing strings, plus a tiny tagline, on a punchy palette of #E2412B, #F4B12A and #FFF6E8." },
        { ru: "Сова-наставник для онлайн-школы", en: "Wise owl tutor", v: "A scholarly wide-eyed owl mascot perched with tiny round glasses, set above a crisp bold wordmark whose letters are carved from warm polished wood grain with soft golden highlights, plus a small tagline, on a trustworthy palette of #1B3A4B, #3F8E7A and #F1E7D0." },
        { ru: "Кит-космонавт для стартапа", en: "Astronaut whale startup", v: "A dreamy floating whale mascot in a glass space helmet with tiny stars drifting around it, hovering above a sleek bold wordmark whose letters are sculpted from glowing neon glass tubes ringed by a soft halo, plus a small tagline, on a cosmic palette of #0B1026, #6C5CE7 and #2BD4D9." },
        { ru: "Медведь-пекарь", en: "Baker bear", v: "A plump apron-wearing bear mascot proudly holding a fresh loaf, standing above a friendly rounded wordmark whose letters are shaped from golden braided bread dough dusted with fine flour, plus a small tagline, on a cozy bakery palette of #5A3420, #D99A4E and #FBF1DC." },
        { ru: "Ленивец для йога-студии", en: "Zen sloth yoga", v: "A serene smiling sloth mascot hanging in a calm meditation pose, placed above a soft bold wordmark whose letters bloom from delicate flowers and lush green foliage with dewy petals, plus a small tagline, on a tranquil palette of #2F4F3E, #88B49A and #F4F0E4." },
        { ru: "Дракончик-геймер", en: "Gamer dragon", v: "A spunky baby dragon mascot gripping a glowing game controller with sparks in its eyes, set above an edgy bold wordmark whose letters blaze with bright fire and drifting orange embers, plus a small tagline, on a high-energy palette of #14091F, #FF4D2E and #FFC93C." },
        { ru: "Пингвин-сёрфер", en: "Surfer penguin", v: "A cool sunglasses-wearing penguin mascot riding a tiny surfboard, positioned above a breezy bold wordmark whose letters are formed from a splash of clear curling water crowned with foamy white crests, plus a small tagline, on a fresh coastal palette of #073B4C, #06B6D4 and #FDFCDC." },
        { ru: "Лев-чемпион для фитнес-бренда", en: "Champion lion fitness", v: "A powerful flexing lion mascot with a proud golden mane, towering above a heavy bold wordmark whose letters are forged from polished mirror chrome with razor-sharp metallic highlights, plus a small tagline, on a strong gym palette of #14171C, #E63946 and #C0C5CE." },
        { ru: "Котик-садовод", en: "Gardener cat", v: "A gentle whiskered cat mascot in a straw hat cradling a little potted sprout, set above a wholesome bold wordmark whose letters are made of fresh green leaves and trailing vines beaded with dewdrops, plus a small tagline, on an earthy palette of #2D4A2B, #7FB069 and #FAF3DD." },
    ],
    art_hero: [
        { ru: "Воин с трещинами расплавленного золота", en: "Cracked-Gold Warrior Portrait", v: "A solemn close-up of an ancient warrior whose weathered skin is fractured by veins of molten gold, ember-bright eyes burning beneath a battered bronze helmet as dramatic chiaroscuro carves every scar against a deep void-black background." },
        { ru: "Кит из звёздной пыли", en: "Cosmic Whale of Stardust", v: "A colossal humpback whale drifting through deep space, its translucent body woven from swirling nebulae, glittering stardust and scattered constellations, fins trailing luminous galactic mist across an endless indigo cosmos." },
        { ru: "Балерина из жидкого стекла", en: "Liquid-Glass Ballerina", v: "A lone ballerina frozen mid-pirouette, her flowing gown sculpted entirely from clear shattering glass and arcing ribbons of splashing water, soft studio light refracting through every translucent fold against a misty pale-grey backdrop." },
        { ru: "Лис в пылающей осенней листве", en: "Autumn-Ember Fox", v: "A majestic red fox standing alert in a glowing autumn forest, its fur rendered in fiery amber and crimson, golden afternoon light streaming through falling maple leaves and drifting motes of dust around its sharp, watchful gaze." },
        { ru: "Парящий самурайский шлем", en: "Floating Samurai Helmet", v: "An ornate antique samurai helmet floating dead-centre against pure darkness, its lacquered black iron and gold inlay catching a single dramatic rim light while intricate dragon engravings and a deep-red horsehair crest glow with museum-grade detail." },
        { ru: "Богиня из цветущих лиан", en: "Goddess of Blooming Vines", v: "A serene forest goddess emerging from the gloom, her face and shoulders formed from blooming flowers, soft moss and curling green vines, dewdrops glistening on the petals as dappled sunlight filters through a deep verdant jungle canopy." },
        { ru: "Хрустальный колибри", en: "Crystal Hummingbird", v: "A single hummingbird hovering in mid-air, its body and outstretched wings carved from faceted prismatic crystal that scatters tiny rainbows, suspended before a soft teal-and-rose gradient flecked with delicate bokeh sparkles." },
        { ru: "Космонавт в океанской бездне", en: "Astronaut in the Deep", v: "A lone astronaut suspended in the silent dark of a bioluminescent ocean, glowing jellyfish and drifting plankton lighting the scratched helmet visor as cold blue light gleams off the worn white suit in haunting cinematic detail." },
        { ru: "Лев из расплавленной лавы", en: "Molten-Lava Lion", v: "A powerful lion's head emerging from darkness, its mane formed of cracking molten lava and rising embers, glowing orange fissures threading through obsidian-black skin as sparks drift upward into the smoke-filled void." },
        { ru: "Журавль-оригами в тумане", en: "Origami Crane in Mist", v: "A single elegant paper origami crane perched on a moss-covered stone, its crisp folded planes catching gentle morning light amid soft rolling mist and the faint blur of a distant pale-pink cherry-blossom branch." },
    ],
    art_composition: [
        { ru: "Маяк и одинокая чайка", en: "Lighthouse and lone gull", v: "A weathered white lighthouse stands on the lower-left third atop dark wet basalt rocks beneath a vast bruised twilight sky, while a single gull glides off toward the upper-right, the wide negative space filled with rolling sea mist and a thin cold horizon line." },
        { ru: "Чаепитие у дождливого окна", en: "Tea by the rainy window", v: "A steaming ceramic cup rests on a worn wooden sill in the lower-right third, its curl of vapour catching warm lamplight, while the upper-left two-thirds dissolve into a rain-streaked pane and the blurred amber glow of a city dusk beyond." },
        { ru: "Путник на горном гребне", en: "Wanderer on the ridge", v: "A tiny silhouetted hiker pauses on the lower-left third of a knife-edge ridge, dwarfed by a towering snow-dusted peak rising into the upper-right, layered blue valley haze receding behind in cool cinematic depth." },
        { ru: "Лодка на рассветном озере", en: "Boat on the dawn lake", v: "A small wooden rowboat drifts in the lower-right third of a glass-still alpine lake, its faint wake catching peach dawn light, while pine-dark mountains and a low ribbon of fog occupy the upper-left, mirrored softly in the mirror-calm water." },
        { ru: "Красный зонт в снегопаде", en: "Red umbrella in snowfall", v: "A single figure under a vivid red umbrella walks the lower-left third of an empty snow-blanketed boulevard, bare black trees and a faint grey lamppost anchoring the upper-right, fat snowflakes drifting through the muted hushed air." },
        { ru: "Кит под лучом света", en: "Whale beneath the light shaft", v: "A colossal humpback whale glides through the lower-left third of deep teal ocean, a single diver suspended small in the upper-right, golden shafts of sunlight piercing down through the silty water thick with drifting plankton." },
        { ru: "Скамья под цветущей сакурой", en: "Bench under the cherry tree", v: "An empty weathered park bench sits in the lower-right third of a quiet garden, a gnarled cherry tree heavy with pink blossom leaning in from the upper-left, scattered petals frozen mid-drift across the soft overcast light." },
        { ru: "Космонавт на алой пустоши", en: "Astronaut on red plains", v: "A lone astronaut stands small on the lower-left third of a rust-red rocky plain, casting a long shadow, while an immense pale crescent planet hangs in the upper-right of a deep starless violet sky, dust hazing the distant horizon." },
        { ru: "Лиса в зимнем лесу", en: "Fox in the winter wood", v: "A russet fox stands alert in the lower-right third of a snow-laden birch forest, its breath misting in the cold, slanting pale morning sun streaming through the bare trunks in the upper-left and dappling the untouched snow." },
        { ru: "Виолончелистка в пустом зале", en: "Cellist in the empty hall", v: "A solitary cellist sits bathed in a single warm spotlight in the lower-left third of a vast dim concert hall, rows of empty crimson velvet seats receding into shadow in the upper-right, dust motes suspended in the lone beam of light." },
    ],
    surreal_scene: [
        { ru: "Кит плывёт в облачном небе", en: "Whale in cloud sky", v: "A colossal humpback whale drifts weightlessly through a pastel dawn sky, trailing a long ribbon of golden migrating birds from its flukes while a tiny rowboat with a single lantern floats in its shadow far below, cinematic soft volumetric light, dreamlike fine-art surrealism." },
        { ru: "Дверь посреди пустыни", en: "Doorway in the desert", v: "A solitary antique wooden door stands open in the middle of an endless rippling sand desert, spilling a torrent of clear blue ocean water and darting silver fish through its frame onto the dry dunes, long evening shadows, surreal poetic contrast of two worlds, hyperreal painterly detail." },
        { ru: "Чаепитие на спине черепахи", en: "Tea party on a turtle", v: "An ancient mossy giant tortoise carries an entire miniature porcelain tea set with steaming cups and a tilted brass chandelier on its domed shell, wading slowly across a mirror-still lake at twilight as fireflies drift around it, whimsical surreal storybook atmosphere with rich warm light." },
        { ru: "Лестница в перевёрнутый океан", en: "Stairs to an upside-down ocean", v: "A spiral marble staircase rises out of a misty meadow and dissolves into an inverted ocean hanging overhead, where jellyfish float like glowing lanterns and a lone deer climbs toward the water-sky, surreal gravity-defying composition, ethereal blue and rose light, fine-art dreamscape." },
        { ru: "Город в стеклянном пузыре", en: "City in a glass bubble", v: "A delicate floating soap bubble cradles a tiny glowing miniature city of crooked towers inside it, balanced on the fingertip of a giant stone hand emerging from a sea of clouds, refractions and rainbow sheen sliding across the glass surface, surreal intimate scale play, luminous cinematic mood." },
        { ru: "Дерево с планетами вместо плодов", en: "Tree bearing planets", v: "A gnarled ancient tree grows on a small floating island and bears ripe glowing planets instead of fruit, one cracked open to reveal a swirling galaxy inside, while a child reaches up from a ladder built of stacked old books, surreal cosmic still life, deep indigo night and warm amber glow." },
        { ru: "Кит-аэростат над затопленным городом", en: "Airship-whale over a flooded city", v: "A translucent glass whale rigged like a floating airship drifts low over a half-submerged city of drowned rooftops, suspending a single illuminated greenhouse garden from its belly, soft reflections shimmering on the still flood water, melancholic surreal beauty, muted teal and gold palette." },
        { ru: "Человек-облако на скамейке", en: "Cloud-headed figure on a bench", v: "A lone figure in a vintage suit sits on a wrought-iron park bench with a swirling thunderstorm cloud where the head should be, gentle rain falling only over the bench while the rest of the autumn park stays sunlit, surreal melancholy portrait, soft cinematic chiaroscuro light." },
        { ru: "Кит из оригами и лунный свет", en: "Origami whale and moonlight", v: "An enormous paper origami whale folded from old maps floats above a calm midnight sea, its translucent body lit from within by a captured full moon while paper birds peel away from its fins and scatter into the stars, surreal monochrome blue dream, delicate luminous craft aesthetic." },
        { ru: "Музыкант с виолончелью из воды", en: "Cellist of liquid water", v: "A seated cellist plays an instrument made entirely of flowing transparent water that arcs and splashes mid-note into leaping silver fish, sheet music dissolving into a flock of moths drifting upward in a candlelit empty hall, surreal poetic synesthesia, rich warm chiaroscuro and glistening detail." },
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
