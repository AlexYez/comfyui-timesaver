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
        return (text || "")
            .split("\n")
            .map((line) => line.replace(/\w\S*/g, (t) => t.charAt(0).toUpperCase() + t.slice(1).toLowerCase()))
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
    const parts = String(aspect || DEFAULT_ASPECT_RATIO).split("x");
    const w = Number(parts[0]) || 16;
    const h = Number(parts[1]) || 9;
    return w / h;
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
        _presetsCache = { styles: [], fonts: [], aspect_ratios: ASPECT_RATIOS, default_aspect_ratio: DEFAULT_ASPECT_RATIO };
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

export async function fetchGraphReference(nodeId) {
    try {
        const response = await fetch(api.apiURL(`${ROUTE_BASE}/graph_ref?node_id=${encodeURIComponent(nodeId)}`));
        if (!response.ok) return null;
        const data = await response.json();
        return data?.filename || null;
    } catch {
        return null;
    }
}

// Build a /view URL for an input-dir image (reference underlay).
export function inputViewUrl(filename, subfolder = "", type = "input") {
    const params = new URLSearchParams({ filename: filename || "", type, subfolder: subfolder || "" });
    return api.apiURL(`/view?${params.toString()}`);
}

// Cyrillic guard analysis for a text block (soft warnings for the inspector).
export function cyrillicWarnings(block, allBlocks, fontsByIdMap, lang = DEFAULT_LANG) {
    const warnings = [];
    const text = block?.text || "";
    if (!hasCyrillic(text)) return warnings;

    const words = text.trim().split(/\s+/).filter(Boolean);
    if (words.length > 3 || text.replace(/\s/g, "").length > 20) warnings.push(t("cyr_long", lang));
    if (block.case !== "UPPERCASE" && /[а-яё]/.test(text)) warnings.push(t("cyr_upper", lang));
    const preset = fontsByIdMap?.[block.font_preset_id || ""];
    if (preset && preset.good_for_cyrillic === false) warnings.push(t("cyr_font", lang));
    const cyrBlocks = (allBlocks || []).filter((b) => b.type === "text" && !b.visual_only && hasCyrillic(b.text || ""));
    if (cyrBlocks.length >= 2) warnings.push(t("cyr_multi", lang));
    const latinBlocks = (allBlocks || []).filter(
        (b) => b.type === "text" && !b.visual_only && !hasCyrillic(b.text || "") && /[A-Za-z]/.test(b.text || ""),
    );
    if (latinBlocks.length > 0) warnings.push(t("cyr_mix", lang));
    return warnings;
}

// ── Localization ───────────────────────────────────────────────────────────── //
const I18N = {
    en: {
        editor_subtitle: "visual caption designer", language: "Language", clear: "Clear all", mp_label: "MP",
        add_text: "+ Text", add_obj: "+ Object", duplicate: "Duplicate", delete: "Delete block",
        reference: "🖼 Reference", clear_ref: "✕ ref", cancel: "Cancel", save: "Save",
        aspect: "Aspect ratio",
        card_template: "Template (what to make)", layout_preset: "Layout template",
        layout_none: "— pick a template —",
        card_style: "Style", style_preset: "Style preset", style_none: "— custom style —",
        card_global: "General style",
        card_overall: "1 · What you're making", card_look: "2 · How it should look", card_scene: "3 · What's in the scene",
        hld: "One-line brief", medium: "Image type", aesthetics: "Mood & vibe",
        lighting: "Lighting", art_style: "Art style", photo_label: "Camera & lens",
        image_palette: "Image colors (up to {n})", background: "Background", add_color: "Add color",
        hld_hint: "Describe the whole image in one sentence, as if telling a person what to make. The model leans on this most. Example: a bold summer-sale poster for a sneaker brand.",
        aesthetics_hint: "The overall feel in a few words. Example: bold and punchy, calm and minimal, retro, luxurious. Safe to leave blank.",
        lighting_hint: "How the scene is lit. Example: bright daylight, soft studio light, moody shadows, neon glow. Safe to leave blank.",
        photo_hint: "Shown only for the Photo image type — the shot's optics: lens, film, framing. Example: 85mm portrait, shallow depth, 35mm film, wide aerial.",
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
        cyr_banner: "Cyrillic in Ideogram 4 is less reliable than Latin. For print, generate the visual and add the Russian text by hand.",
        cyr_long: "Long Russian text renders unreliably — shorten to 1–3 words or enable 'manual text'.",
        cyr_upper: "Cyrillic comes out better in UPPERCASE — enable UPPERCASE.",
        cyr_font: "This font preset is not recommended for Cyrillic — pick a Cyrillic-safe one.",
        cyr_multi: "Multiple Cyrillic blocks reduce each one's accuracy — keep just one.",
        cyr_mix: "Cyrillic and Latin in one image — better split into separate generations.",
        cyr_badge: "Cyrillic hint added", cyr_warn_pill: "⚠ Cyrillic",
        save_as_layout: "Save as template", save_as_style: "Save as style",
        export_btn: "⬇ Export", import_btn: "⬆ Import",
        import_done: "Imported: {n}", import_empty: "No valid presets in the file", export_empty: "Nothing to export yet",
        medium_hint: "Sets the look of the WHOLE image, not the text. 'Photograph' fills the camera field below; every other type fills the art-style field.",
        preset_name_prompt: "Preset name:", custom_tag: "custom",
        summary: "{t} text · {o} obj", empty_hint: "Click \"✎ Edit design\"", edit_btn: "✎ Edit design",
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
        tip_style_preset: "Pick a saved style to instantly apply its colors, fonts and overall look to your design.",
        tip_save_as_style: "Save these style fields, palette and font as a reusable preset you can apply to future designs.",
        tip_export_style: "Download the current style as a JSON file to back it up or share it with others.",
        tip_import_style: "Load style preset files from your computer to add them to the style picker.",
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
    },
    ru: {
        editor_subtitle: "визуальный дизайнер капшена", language: "Язык", clear: "Очистить всё", mp_label: "МП",
        add_text: "+ Текст", add_obj: "+ Объект", duplicate: "Дублировать", delete: "Удалить выбранный блок",
        reference: "🖼 Референс", clear_ref: "✕ реф", cancel: "Отмена", save: "Сохранить",
        aspect: "Соотношение сторон",
        card_template: "Шаблон (что делаем)", layout_preset: "Шаблон-лейаут",
        layout_none: "— выберите шаблон —",
        card_style: "Стиль", style_preset: "Пресет стиля", style_none: "— свой стиль —",
        card_global: "Общий стиль",
        card_overall: "1 · Что вы создаёте", card_look: "2 · Как это должно выглядеть", card_scene: "3 · Что в сцене",
        hld: "Бриф одной строкой", medium: "Тип изображения", aesthetics: "Настроение и вайб",
        lighting: "Освещение", art_style: "Художественный стиль", photo_label: "Камера и оптика",
        image_palette: "Цвета изображения (до {n})", background: "Фон", add_color: "Добавить цвет",
        hld_hint: "Опишите всю картинку одним предложением, как будто объясняете человеку, что нарисовать. Модель опирается на него сильнее всего. Например: яркий постер летней распродажи кроссовок.",
        aesthetics_hint: "Общее ощущение в паре слов. Например: дерзко и сочно, спокойно и минимально, ретро, премиально. Можно оставить пустым.",
        lighting_hint: "Как освещена сцена. Например: яркий дневной свет, мягкий студийный, драматичные тени, неоновое свечение. Можно оставить пустым.",
        photo_hint: "Показывается только для типа «Фото» — оптика кадра: объектив, плёнка, ракурс. Например: портрет 85мм, малая глубина, плёнка 35мм, широкий аэроснимок.",
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
        cyr_banner: "Кириллица в Ideogram 4 менее надёжна латиницы. Для печати — генерируйте визуал и добавляйте русский текст вручную.",
        cyr_long: "Длинный русский текст рендерится ненадёжно — сократите до 1–3 слов или включите «текст вручную».",
        cyr_upper: "Кириллица лучше выходит в ВЕРХНЕМ регистре — включите UPPERCASE.",
        cyr_font: "Этот шрифт-пресет не рекомендуется для кириллицы — выберите Cyrillic-safe.",
        cyr_multi: "Несколько кириллических блоков снижают точность каждого — оставьте один.",
        cyr_mix: "Кириллица и латиница в одном изображении — лучше разнести на отдельные генерации.",
        cyr_badge: "кириллица: hint добавлен", cyr_warn_pill: "⚠ кириллица",
        save_as_layout: "Сохранить как шаблон", save_as_style: "Сохранить как стиль",
        export_btn: "⬇ Экспорт", import_btn: "⬆ Импорт",
        import_done: "Импортировано: {n}", import_empty: "В файле нет валидных пресетов", export_empty: "Пока нечего экспортировать",
        medium_hint: "Задаёт вид ВСЕЙ картинки, а не текста. «Фото» заполняет поле камеры ниже; остальные типы — поле художественного стиля.",
        preset_name_prompt: "Имя пресета:", custom_tag: "свой",
        summary: "{t} текст · {o} obj", empty_hint: "Нажмите «✎ Редактировать»", edit_btn: "✎ Редактировать",
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
        tip_style_preset: "Выберите сохранённый стиль — он сразу применит свои цвета, шрифты и общий вид к вашему дизайну.",
        tip_save_as_style: "Сохраните эти настройки стиля, палитру и шрифт как пресет, чтобы применять их к будущим дизайнам.",
        tip_export_style: "Скачайте текущий стиль в виде JSON-файла, чтобы сохранить на будущее или поделиться с другими.",
        tip_import_style: "Загрузите файлы стилей с компьютера, чтобы добавить их в список выбора стилей.",
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

// Localized display labels for the image-type (medium) <select>. The option
// VALUES stay the raw tokens (used by the caption); only the visible text changes.
const MEDIA_LABELS = {
    graphic_design: { en: "Graphic design", ru: "Графический дизайн" },
    photograph: { en: "Photograph", ru: "Фотография" },
    illustration: { en: "Illustration", ru: "Иллюстрация" },
    "3d_render": { en: "3D render", ru: "3D-рендер" },
    painting: { en: "Painting", ru: "Живопись" },
    digital_painting: { en: "Digital painting", ru: "Цифровая живопись" },
};
export function mediumLabel(token, lang = DEFAULT_LANG) {
    return MEDIA_LABELS[token]?.[lang] ?? MEDIA_LABELS[token]?.en ?? token;
}

export function localizedName(item, lang = DEFAULT_LANG) {
    return (item && (item[`name_${lang}`] || item.name_en || item.name_ru || item.id)) || "";
}

export function localizedDesc(item, lang = DEFAULT_LANG) {
    return (item && (item[`desc_${lang}`] || item.desc_en || item.desc_ru || "")) || "";
}

// ── Two-level presets ──────────────────────────────────────────────────────── //
export function layoutsList(presets) { return presets?.layouts || []; }
export function stylesList(presets) { return presets?.styles || []; }

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

// Resolve a style preset into the work.style object.
export function applyStyle(style) {
    return {
        preset_id: style?.id || "",
        aesthetics: style?.aesthetics || "",
        lighting: style?.lighting || "",
        medium: style?.medium || "graphic_design",
        photo: style?.photo || "",
        art_style: style?.art_style || "",
        color_palette: cleanPalette(style?.color_palette || [], IMAGE_PALETTE_CAP),
        font_preset_id: style?.font_preset_id || "",
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
