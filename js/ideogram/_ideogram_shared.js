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
export const WEIGHTS = ["Thin", "Regular", "Bold", "Heavy"];
export const CASES = ["As-typed", "UPPERCASE", "Title"];
export const PROMINENCE = ["Caption", "Body", "Headline", "Hero"];
export const PHOTO_MEDIUM = "photograph";
export const IMAGE_PALETTE_CAP = 16;
export const ELEMENT_PALETTE_CAP = 5;

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
        aspect_ratio: DEFAULT_ASPECT_RATIO,
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

// ── Backend calls ─────────────────────────────────────────────────────────── //
let _presetsCache = null;
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
export function cyrillicWarnings(block, allBlocks, fontsById) {
    const warnings = [];
    const text = block?.text || "";
    if (!hasCyrillic(text)) return warnings;

    const words = text.trim().split(/\s+/).filter(Boolean);
    if (words.length > 3 || text.replace(/\s/g, "").length > 20) {
        warnings.push("Длинный русский текст рендерится ненадёжно — сократите до 1–3 слов или включите «текст вручную».");
    }
    if (block.case !== "UPPERCASE" && /[а-яё]/.test(text)) {
        warnings.push("Кириллица лучше выходит в ВЕРХНЕМ регистре — включите UPPERCASE.");
    }
    const preset = fontsById?.[block.font_preset_id || ""];
    if (preset && preset.good_for_cyrillic === false) {
        warnings.push("Этот шрифт-пресет не рекомендуется для кириллицы — выберите Cyrillic-safe.");
    }
    const cyrBlocks = (allBlocks || []).filter((b) => b.type === "text" && !b.visual_only && hasCyrillic(b.text || ""));
    if (cyrBlocks.length >= 2) {
        warnings.push("Несколько кириллических блоков снижают точность каждого — оставьте один.");
    }
    const latinBlocks = (allBlocks || []).filter(
        (b) => b.type === "text" && !b.visual_only && !hasCyrillic(b.text || "") && /[A-Za-z]/.test(b.text || ""),
    );
    if (latinBlocks.length > 0) {
        warnings.push("Кириллица и латиница в одном изображении — лучше разнести на отдельные генерации.");
    }
    return warnings;
}
