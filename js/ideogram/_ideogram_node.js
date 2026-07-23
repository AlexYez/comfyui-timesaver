// In-node preview widget for TS_IdeogramDesigner.
//
// Renders a fluid, aspect-correct preview of the current design (reference
// underlay + block rectangles + labels), the shared "Open Interface" launcher
// that opens the full-screen modal editor, and a one-line summary. Fluid sizing + Nodes 1.0 /
// Nodes 2.0 (Vue) compatibility follow the verified sam_media_loader patterns:
// addDOMWidget with getMinHeight/getMaxHeight (no widget.computeSize), DPR
// canvas, ResizeObserver, syncDomSize, cleanup on removal.

import {
    DEFAULT_LANG,
    DESIGN_INPUT,
    MESH_POSITIONS,
    NODE_NAME,
    WEIGHT_CSS,
    applyCase,
    aspectFitBox,
    cleanPalette,
    dimsFromAspectMp,
    fetchGraphRef,
    fontFamilyForPreset,
    hexToRgba,
    hideWidget,
    inputViewUrl,
    loadPresets,
    localizedName,
    makeDefaultDesign,
    normHex,
    parseDesign,
    readPersistedDesign,
    setWidgetValue,
    stopPropagation,
    t,
} from "./_ideogram_shared.js";

import { openIdeogramEditor } from "./_ideogram_editor.js";

import {
    TS_UI_CLASS,
    createOpenInterfaceButton,
    ensureThemeStyles,
    getThemeColors,
    getUiLanguage,
    pickLocaleStrings,
    setOpenInterfaceLabel,
} from "../_theme.js";
import { api } from "/scripts/api.js";

const STYLE_ID = "ts-ideogram-node-styles";
const DOM_WIDGET_NAME = "ts_ideogram_node";
const DEFAULT_NODE_SIZE = [320, 300];
const MIN_NODE_WIDTH = 240;
const MIN_NODE_HEIGHT = 220;
const PAD = 10;
// Tall enough for the shared launcher button (which is bigger than the
// node's old inline button) plus breathing room, so it cannot spill over
// the artboard drawn underneath.
const TOOLBAR_H = 44;
// Mode switch row (Designer / Auto) above the toolbar.
const MODE_H = 30;
const TOP_CHROME = MODE_H + TOOLBAR_H;
const SUMMARY_H = 22;
const MODE_INPUT = "mode";
const AUTO_PROMPT_INPUT = "auto_prompt";
const AUTO_SEED_INPUT = "auto_seed";
const AUTO_CAPTION_INPUT = "auto_caption";
// Same engine and preset the SuperPrompt AI button uses — генерация идёт
// сразу по кнопке, без постановки workflow в очередь.
const ENHANCE_ROUTE = "/ts_super_prompt/enhance";
const ENHANCE_PRESET = "Ideogram Prompt Enhance";

// Node-chrome strings follow the ComfyUI UI locale (the design document keeps
// its own language; this is interface, not content).
const CHROME_STRINGS = {
    en: {
        modeDesigner: "Designer",
        modeAuto: "Auto",
        generate: "Generate Prompt",
        autoPlaceholder: "Describe your idea in plain words — the model writes the structured Ideogram JSON for you.",
        resultHint: "The generated JSON caption will appear here after a run.",
        running: "Generating...",
        done: "Caption ready.",
        needPrompt: "Type an idea first.",
        queueFailed: (m) => `Queue failed: ${m}`,
        runFailed: (m) => `Generation failed: ${m}`,
    },
    ru: {
        modeDesigner: "Дизайнер",
        modeAuto: "Авто",
        generate: "Создать промпт",
        autoPlaceholder: "Опишите идею обычными словами — модель сама напишет структурированный Ideogram JSON.",
        resultHint: "Сгенерированный JSON-капшен появится здесь после запуска.",
        running: "Генерация...",
        done: "Капшен готов.",
        needPrompt: "Сначала введите идею.",
        queueFailed: (m) => `Не удалось поставить в очередь: ${m}`,
        runFailed: (m) => `Генерация не удалась: ${m}`,
    },
};

function ensureStyles() {
    // Colours come from the shared --ts-* tokens in js/_theme.js; this
    // stylesheet is layout only. Never hard-code chrome colours here.
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-ideo-node{position:relative;width:100%;height:100%;min-height:0;box-sizing:border-box;color:var(--ts-text);font-family:var(--ts-font);background:var(--ts-checker);border:1px solid var(--ts-border-soft);border-radius:var(--ts-radius-lg);overflow:hidden;user-select:none}
.ts-ideo-node__canvas{position:absolute;inset:0;display:block;width:100%;height:100%}
.ts-ideo-node__toolbar{position:absolute;top:${MODE_H + 4}px;left:6px;right:6px;height:${TOOLBAR_H - 8}px;display:flex;align-items:center;justify-content:center;gap:6px;z-index:3}
.ts-ideo-node__pill{margin-left:auto;flex:0 0 auto;font-size:var(--ts-fs-xs);color:var(--ts-faint);white-space:nowrap;font-variant-numeric:tabular-nums}
.ts-ideo-node__summary{position:absolute;left:6px;right:6px;bottom:6px;height:${SUMMARY_H - 6}px;display:flex;align-items:center;gap:8px;font-size:var(--ts-fs-sm);color:var(--ts-muted);background:var(--ts-elevated);border:1px solid var(--ts-border-soft);border-radius:8px;padding:0 8px;z-index:3;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden}
.ts-ideo-node__warn{color:var(--ts-warning)}
.ts-ideo-node__modes{position:absolute;top:6px;left:6px;right:6px;height:${MODE_H - 8}px;display:flex;gap:4px;z-index:3}
.ts-ideo-node__modes .ts-ui-btn{flex:1 1 0;padding:3px 6px;font-size:var(--ts-fs-xs)}
.ts-ideo-node__auto{position:absolute;left:6px;right:6px;top:${MODE_H + 4}px;bottom:${SUMMARY_H + 4}px;display:none;flex-direction:column;gap:6px;z-index:3}
.ts-ideo-node__auto.is-active{display:flex}
.ts-ideo-node__auto-text{flex:1 1 55%;min-height:0;resize:none}
.ts-ideo-node__auto-result{flex:1 1 45%;min-height:0;overflow:auto;font-size:var(--ts-fs-xs);white-space:pre-wrap;word-break:break-word;background:var(--ts-sunken);border:1px solid var(--ts-border-soft);border-radius:var(--ts-radius-sm);padding:4px 6px;color:var(--ts-muted)}
.ts-ideo-node__auto-status{flex:0 0 auto;min-height:14px;font-size:var(--ts-fs-xs);text-align:center}
`;
    document.head.appendChild(style);
}

function removeDomWidget(node) {
    if (!Array.isArray(node?.widgets)) return;
    for (let i = node.widgets.length - 1; i >= 0; i -= 1) {
        if (node.widgets[i]?.name !== DOM_WIDGET_NAME) continue;
        (node.widgets[i].element || node.widgets[i].el || node.widgets[i].container)?.remove?.();
        node.widgets.splice(i, 1);
    }
}

// Canvas mirror of paletteGradientCss (_ideogram_shared.js): paint a palette
// onto a rect so the node preview matches the editor's WYSIWYG colors.
// mesh=true → layered radial blobs (artboard); mesh=false → diagonal gradient
// (block fills). Returns true if it painted anything.
function paintPaletteRect(ctx, colors, x, y, w, h, { alpha = 1, mesh = true } = {}) {
    const pal = cleanPalette(colors || [], 16);
    if (!pal.length) return false;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    if (pal.length === 1) {
        ctx.globalAlpha = alpha;
        ctx.fillStyle = pal[0];
        ctx.fillRect(x, y, w, h);
    } else {
        const g = ctx.createLinearGradient(x, y, x + w, y + h);
        pal.forEach((c, i) => g.addColorStop(i / (pal.length - 1), c));
        ctx.globalAlpha = alpha;
        ctx.fillStyle = g;
        ctx.fillRect(x, y, w, h);
        if (mesh) {
            const rad = Math.max(w, h) * 0.55;
            pal.slice(0, 6).forEach((c, i) => {
                const [px, py] = MESH_POSITIONS[i % MESH_POSITIONS.length];
                const cx = x + (px / 100) * w;
                const cy = y + (py / 100) * h;
                const rg = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
                rg.addColorStop(0, hexToRgba(c, alpha * 0.85));
                rg.addColorStop(1, hexToRgba(c, 0));
                ctx.fillStyle = rg;
                ctx.fillRect(x, y, w, h);
            });
        }
    }
    ctx.restore();
    return true;
}

// Word-wrap a single line to fit maxW (keeps an over-long word on its own line).
function wrapLine(ctx, line, maxW) {
    const words = line.split(" ");
    const out = [];
    let cur = "";
    for (const word of words) {
        const test = cur ? `${cur} ${word}` : word;
        if (!cur || ctx.measureText(test).width <= maxW) cur = test;
        else { out.push(cur); cur = word; }
    }
    if (cur) out.push(cur);
    return out;
}

// Canvas mirror of the editor's styled+auto-fitted block text: binary-search the
// font size so the word-wrapped lines fit the box on both axes, then draw
// centered with an outline or soft shadow — so the node preview matches the
// full editor WYSIWYG (editor uses white-space:pre-wrap + fitText).
function drawFittedText(ctx, text, x, y, w, h, { fontFamily, weight, color, outline, outlineColor, maxSize }) {
    if (!text || w < 3 || h < 3) return;
    const rawLines = String(text).split("\n");
    let lo = 4, hi = Math.max(4, Math.min(maxSize || h, h)), best = 4, bestLines = rawLines;
    for (let i = 0; i < 12 && lo <= hi; i += 1) {
        const mid = (lo + hi) >> 1;
        ctx.font = `${weight} ${mid}px ${fontFamily}`;
        let wrapped = [];
        for (const rl of rawLines) wrapped = wrapped.concat(wrapLine(ctx, rl, w));
        const totalH = wrapped.length * mid * 1.12;
        const fits = totalH <= h && wrapped.every((l) => ctx.measureText(l).width <= w);
        if (fits) { best = mid; bestLines = wrapped; lo = mid + 1; } else { hi = mid - 1; }
    }
    ctx.font = `${weight} ${best}px ${fontFamily}`;
    const lineH = best * 1.12;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    let ty = y + h / 2 - (bestLines.length * lineH) / 2 + lineH / 2;
    for (const line of bestLines) {
        if (outline) {
            ctx.lineWidth = Math.max(1, best * 0.1);
            ctx.lineJoin = "round";
            ctx.strokeStyle = outlineColor || "rgba(0,0,0,.85)";
            ctx.strokeText(line, x + w / 2, ty);
        } else {
            ctx.shadowColor = "rgba(0,0,0,.7)";
            ctx.shadowBlur = best * 0.08;
            ctx.shadowOffsetY = best * 0.04;
        }
        ctx.fillStyle = color;
        ctx.fillText(line, x + w / 2, ty);
        ctx.shadowColor = "transparent";
        ctx.shadowBlur = 0;
        ctx.shadowOffsetY = 0;
        ty += lineH;
    }
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
}

export function setupIdeogramNode(node) {
    if (!node || typeof node.addDOMWidget !== "function") return;
    if (typeof node._tsIdeoCleanup === "function") {
        try { node._tsIdeoCleanup(); } catch { /* ignore */ }
    }
    removeDomWidget(node);
    ensureStyles();
    hideWidget(node, DESIGN_INPUT);
    hideWidget(node, MODE_INPUT);
    hideWidget(node, AUTO_PROMPT_INPUT);
    hideWidget(node, AUTO_SEED_INPUT);
    hideWidget(node, AUTO_CAPTION_INPUT);
    const C = pickLocaleStrings(CHROME_STRINGS);
    const readPersisted = (name, fallback) => {
        const w = node.widgets?.find((x) => x?.name === name);
        const v = w?.value ?? node.properties?.[name];
        return v === undefined || v === null || v === "" ? fallback : v;
    };

    node.resizable = true;
    node.size = [
        Math.max(Number(node.size?.[0]) || DEFAULT_NODE_SIZE[0], MIN_NODE_WIDTH),
        Math.max(Number(node.size?.[1]) || DEFAULT_NODE_SIZE[1], MIN_NODE_HEIGHT),
    ];
    node.min_size = [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];

    const state = {
        design: parseDesign(readPersistedDesign(node)),
        presets: { styles: [], fonts: [] },
        refImg: null,
        refKey: "",
    };

    const container = document.createElement("div");
    container.className = `${TS_UI_CLASS} ts-ideo-node`;

    const canvas = document.createElement("canvas");
    canvas.className = "ts-ideo-node__canvas";

    const toolbar = document.createElement("div");
    toolbar.className = "ts-ideo-node__toolbar";
    // The launcher is chrome, so both its label and its tooltip follow the
    // ComfyUI UI locale via getUiLanguage(). Two TS nodes side by side never
    // disagree on the wording, and it never drifts into the design document's
    // language (which still drives the rest of this panel).
    const editBtn = createOpenInterfaceButton(() => openEditor(), {
        description: t("tip_edit_design", getUiLanguage()),
    });
    // Retained as a stable query hook (e2e test, user CSS); the look comes
    // from the shared launcher classes.
    editBtn.classList.add("ts-ideo-node__btn");
    const aspectPill = document.createElement("span");
    aspectPill.className = "ts-ideo-node__pill";
    aspectPill.textContent = "16x9";
    toolbar.append(editBtn);

    const summary = document.createElement("div");
    summary.className = "ts-ideo-node__summary";

    const modesRow = document.createElement("div");
    modesRow.className = "ts-ideo-node__modes";
    const designerBtn = document.createElement("button");
    designerBtn.type = "button";
    designerBtn.className = "ts-ui-btn";
    designerBtn.textContent = C.modeDesigner;
    const autoBtn = document.createElement("button");
    autoBtn.type = "button";
    autoBtn.className = "ts-ui-btn";
    autoBtn.textContent = C.modeAuto;
    modesRow.append(designerBtn, autoBtn);

    const autoPanel = document.createElement("div");
    autoPanel.className = "ts-ideo-node__auto";
    const autoText = document.createElement("textarea");
    autoText.className = "ts-ui-textarea ts-ideo-node__auto-text";
    autoText.placeholder = C.autoPlaceholder;
    autoText.value = String(readPersisted(AUTO_PROMPT_INPUT, "") || "");
    const generateBtn = document.createElement("button");
    generateBtn.type = "button";
    generateBtn.className = "ts-ui-btn ts-ui-btn--primary";
    generateBtn.textContent = C.generate;
    const autoStatus = document.createElement("div");
    autoStatus.className = "ts-ui-status ts-ideo-node__auto-status";
    const autoResult = document.createElement("div");
    autoResult.className = "ts-ideo-node__auto-result";
    autoResult.textContent = String(readPersisted(AUTO_CAPTION_INPUT, "") || "") || C.resultHint;
    autoPanel.append(autoText, generateBtn, autoStatus, autoResult);

    container.append(canvas, modesRow, toolbar, autoPanel, summary);
    stopPropagation(container, [
        "pointerdown", "pointerup", "pointermove", "mousedown", "mouseup",
        "wheel", "click", "dblclick", "contextmenu",
    ]);

    const widgetOptions = {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => MIN_NODE_HEIGHT - 30,
        getMaxHeight: () => 8192,
        afterResize: () => requestRedraw(),
    };
    const domWidget = node.addDOMWidget(DOM_WIDGET_NAME, "div", container, widgetOptions);
    const domWidgetEl = domWidget?.element || domWidget?.el || domWidget?.container;

    let mode = String(readPersisted(MODE_INPUT, "designer") || "designer");
    function applyMode(next, persist = true) {
        mode = next === "auto" ? "auto" : "designer";
        const isAuto = mode === "auto";
        designerBtn.classList.toggle("is-active", !isAuto);
        autoBtn.classList.toggle("is-active", isAuto);
        toolbar.style.display = isAuto ? "none" : "";
        canvas.style.display = isAuto ? "none" : "";
        autoPanel.classList.toggle("is-active", isAuto);
        if (persist) setWidgetValue(node, MODE_INPUT, mode);
    }
    designerBtn.addEventListener("click", (e) => { e.stopPropagation(); applyMode("designer"); });
    autoBtn.addEventListener("click", (e) => { e.stopPropagation(); applyMode("auto"); });

    let autoTextTimer = null;
    autoText.addEventListener("input", () => {
        if (autoTextTimer) clearTimeout(autoTextTimer);
        autoTextTimer = setTimeout(() => {
            autoTextTimer = null;
            setWidgetValue(node, AUTO_PROMPT_INPUT, autoText.value);
        }, 150);
    });

    const setAutoStatus = (text, kind = "") => {
        autoStatus.textContent = text || "";
        autoStatus.classList.toggle("is-error", kind === "error");
        autoStatus.classList.toggle("is-success", kind === "success");
    };

    async function queueGenerate() {
        if (!autoText.value.trim()) {
            setAutoStatus(C.needPrompt, "error");
            return;
        }
        // The button generates IMMEDIATELY through the pack's Qwen engine
        // (the SuperPrompt /enhance route) — no workflow queue involved, same
        // UX as the SuperPrompt AI button. The result is stored into the
        // hidden auto_caption widget, which execute() emits in Auto mode.
        setWidgetValue(node, AUTO_PROMPT_INPUT, autoText.value);
        const seed = Number(readPersisted(AUTO_SEED_INPUT, 0)) || 0;
        setWidgetValue(node, AUTO_SEED_INPUT, (seed + 1) & 0x7fffffff);
        setAutoStatus(C.running);
        generateBtn.disabled = true;
        try {
            const response = await api.fetchApi(ENHANCE_ROUTE, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: autoText.value,
                    system_preset: ENHANCE_PRESET,
                }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || payload?.error) {
                throw new Error(payload?.error || `HTTP ${response.status}`);
            }
            const caption = String(payload?.text || "").trim();
            if (!caption) throw new Error("empty reply");
            autoResult.textContent = caption;
            setWidgetValue(node, AUTO_CAPTION_INPUT, caption);
            setAutoStatus(C.done, "success");
        } catch (error) {
            setAutoStatus(C.runFailed(error?.message || error), "error");
        } finally {
            generateBtn.disabled = false;
        }
    }
    generateBtn.addEventListener("click", (e) => { e.stopPropagation(); queueGenerate(); });

    const prevOnExecuted = node.onExecuted;
    node.onExecuted = function onExecutedWithAuto(message) {
        const result = prevOnExecuted?.apply(this, arguments);
        const caption = message?.ts_ideo_auto?.[0];
        if (typeof caption === "string" && caption) {
            autoResult.textContent = caption;
            setAutoStatus(C.done, "success");
        }
        generateBtn.disabled = false;
        return result;
    };
    const onExecError = (event) => {
        if (String(event?.detail?.node_id) !== String(node.id)) return;
        generateBtn.disabled = false;
        setAutoStatus(C.runFailed(event?.detail?.exception_message || "error"), "error");
    };
    api.addEventListener("execution_error", onExecError);

    function syncDomSize() {
        if (domWidgetEl) {
            domWidgetEl.style.width = "100%";
            domWidgetEl.style.height = "100%";
            domWidgetEl.style.minHeight = "0";
            domWidgetEl.style.overflow = "hidden";
        }
        container.style.width = "100%";
        container.style.height = "100%";
        container.style.minHeight = "0";
    }

    function updateSummary() {
        const blocks = state.design.blocks || [];
        const texts = blocks.filter((b) => b.type === "text" && !b.visual_only).length;
        const placeholders = blocks.filter((b) => b.type === "text" && b.visual_only).length;
        const objs = blocks.filter((b) => b.type === "obj").length;
        const lang = state.design.language || DEFAULT_LANG;
        const styleObj = (state.presets.styles || []).find((s) => s.id === state.design.style?.preset_id);
        const styleName = styleObj ? localizedName(styleObj, lang) : (state.design.style?.preset_id || "—");
        setOpenInterfaceLabel(editBtn, undefined, t("tip_edit_design", getUiLanguage()));
        // Tooltips are chrome, not document content — they follow the UI
        // locale like the launcher (the summary counts keep the design's
        // own language: they describe the document).
        canvas.title = t("tip_node_canvas", getUiLanguage());
        const dims = dimsFromAspectMp(state.design.aspect_ratio, state.design.megapixels);
        aspectPill.textContent = `${dims.w}×${dims.h}`;
        aspectPill.title = t("tip_dims_pill", getUiLanguage());
        summary.innerHTML = "";
        const main = document.createElement("span");
        main.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis";
        const txtWord = lang === "en" ? "text" : "текст";
        const objWord = lang === "en" ? "obj" : "об.";
        main.textContent = `${texts} ${txtWord} · ${objs} ${objWord}${placeholders ? ` · ${placeholders}↳` : ""} · ${styleName}`;
        summary.append(main, aspectPill);
    }

    function ensureRefImage() {
        const ref = state.design.ref;
        const key = ref ? `${ref.filename}|${ref.subfolder || ""}|${ref.type || "input"}` : "";
        if (key === state.refKey) return;
        state.refKey = key;
        state.refImg = null;
        if (!ref?.filename) {
            requestRedraw();
            return;
        }
        const img = new Image();
        img.onload = () => { state.refImg = img; requestRedraw(); };
        // Repaint on failure too, or the canvas keeps showing the previous frame.
        img.onerror = () => { state.refImg = null; requestRedraw(); };
        img.src = inputViewUrl(ref.filename, ref.subfolder, ref.type);
    }

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, Math.floor(rect.width * dpr));
        const h = Math.max(1, Math.floor(rect.height * dpr));
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        return { rectWidth: rect.width, rectHeight: rect.height, dpr };
    }

    function draw() {
        const { rectWidth, rectHeight, dpr } = resizeCanvas();
        const ctx = canvas.getContext("2d");
        if (!ctx || rectWidth <= 0 || rectHeight <= 0) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rectWidth, rectHeight);

        const availW = rectWidth - PAD * 2;
        const availH = rectHeight - TOP_CHROME - SUMMARY_H - PAD;
        if (availW <= 4 || availH <= 4) return;
        const box = aspectFitBox(state.design.aspect_ratio, availW, availH);
        const ax = PAD + (availW - box.w) / 2;
        const ay = TOP_CHROME + (availH - box.h) / 2;

        // Artboard background: a reference underlay wins, else the style palette
        // mesh gradient (matches the editor artboard).
        const stylePal = state.design.style?.color_palette || [];
        // Canvas cannot read CSS variables, so the preview pulls resolved
        // theme colours instead (cached inside getThemeColors).
        const tsColors = getThemeColors();
        ctx.fillStyle = tsColors.sunken;
        ctx.fillRect(ax, ay, box.w, box.h);
        if (state.refImg) {
            ctx.save();
            ctx.globalAlpha = 0.65;
            ctx.drawImage(state.refImg, ax, ay, box.w, box.h);
            ctx.restore();
        } else {
            paintPaletteRect(ctx, stylePal, ax, ay, box.w, box.h, { alpha: 1, mesh: true });
        }
        ctx.strokeStyle = tsColors.border;
        ctx.lineWidth = 1;
        ctx.strokeRect(ax + 0.5, ay + 0.5, box.w - 1, box.h - 1);

        const lang = state.design.language || DEFAULT_LANG;
        for (const block of state.design.blocks || []) {
            const r = block.rect;
            if (!r) continue;
            const bx = ax + r.x * box.w;
            const by = ay + r.y * box.h;
            const bw = Math.max(2, r.w * box.w);
            const bh = Math.max(2, r.h * box.h);
            const isText = block.type === "text";
            const visualOnly = isText && block.visual_only;
            const accent = visualOnly ? tsColors.muted : isText ? tsColors.accent : tsColors.success;
            ctx.save();
            // A block's own palette tints its rectangle; else a faint type accent.
            if (!paintPaletteRect(ctx, block.color_palette, bx, by, bw, bh, { alpha: 0.5, mesh: false })) {
                ctx.globalAlpha = 0.12;
                ctx.fillStyle = accent;
                ctx.fillRect(bx, by, bw, bh);
            }
            ctx.globalAlpha = 1;
            ctx.strokeStyle = accent;
            ctx.lineWidth = 1.5;
            if (visualOnly) ctx.setLineDash([4, 3]);
            ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
            ctx.setLineDash([]);
            // WYSIWYG content — mirrors the editor's renderBlocks typography so the
            // node preview is a faithful small copy of the full editor.
            ctx.beginPath();
            ctx.rect(bx, by, bw, bh);
            ctx.clip();
            const pad = Math.max(2, Math.min(bw, bh) * 0.07);
            if (isText && !visualOnly) {
                const leg = block.legibility || {};
                if (leg.solid_block) {  // plate behind the text
                    ctx.fillStyle = normHex(block.plate_color) || "#1A1A1A";
                    ctx.fillRect(bx, by, bw, bh);
                }
                drawFittedText(ctx, applyCase(block.text || "", block.case),
                    bx + pad, by + pad, bw - pad * 2, bh - pad * 2, {
                        fontFamily: fontFamilyForPreset(block.font_preset_id),
                        weight: WEIGHT_CSS[block.weight] || 700,
                        color: normHex(block.color) || "#FFFFFF",
                        outline: !!leg.outline,
                        outlineColor: normHex(block.outline_color) || "#000000",
                    });
            } else if (block.type === "obj") {
                drawFittedText(ctx, block.desc || t("badge_obj", lang),
                    bx + pad, by + pad, bw - pad * 2, bh - pad * 2, {
                        fontFamily: "'Segoe UI',Tahoma,sans-serif", weight: 600,
                        color: "rgba(233,238,246,.88)", outline: false,
                    });
            } else if (visualOnly) {
                ctx.fillStyle = "rgba(154,166,184,.75)";
                ctx.font = `${Math.max(8, Math.min(bw, bh) * 0.4)}px 'Segoe UI',sans-serif`;
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("↳", bx + bw / 2, by + bh / 2);
                ctx.textAlign = "start";
                ctx.textBaseline = "alphabetic";
            }
            ctx.restore();
        }

        if (!(state.design.blocks || []).length) {
            ctx.fillStyle = tsColors.faint;
            ctx.font = "12px 'Segoe UI', sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(t("empty_hint", state.design.language || DEFAULT_LANG), ax + box.w / 2, ay + box.h / 2);
            ctx.textAlign = "start";
        }
    }

    let redrawScheduled = false;
    function requestRedraw() {
        if (redrawScheduled) return;
        redrawScheduled = true;
        requestAnimationFrame(() => {
            redrawScheduled = false;
            draw();
        });
    }

    function applyDesign(design) {
        state.design = design || makeDefaultDesign();
        const json = JSON.stringify(state.design);
        setWidgetValue(node, DESIGN_INPUT, json);
        if (node.setProperty) {
            node.setProperty(DESIGN_INPUT, json);
        } else {
            node.properties ||= {};
            node.properties[DESIGN_INPUT] = json;
        }
        ensureRefImage();
        updateSummary();
        requestRedraw();
        node.setDirtyCanvas(true, true);
    }

    async function openEditor() {
        if (!state.presets.fonts?.length && !state.presets.styles?.length) {
            state.presets = await loadPresets();
        }
        // Offer the graph IMAGE input (cached by execute) as a tracing underlay
        // when the design has no reference and the user hasn't explicitly
        // cleared one (ref_cleared).
        let graphRef = null;
        if (!state.design.ref?.filename && !state.design.ref_cleared) {
            graphRef = await fetchGraphRef(node.id);
        }
        openIdeogramEditor(node, {
            design: state.design,
            presets: state.presets,
            graphRef,
            onSave: (design) => applyDesign(design),
        });
    }

    // The launcher factory already wires the click (and stops propagation);
    // a second listener here opened the editor twice, stacking two overlays.
    canvas.addEventListener("dblclick", (e) => { e.stopPropagation(); openEditor(); });

    const prevOnResize = node.onResize;
    node.onResize = function onResize() {
        const r = prevOnResize?.apply(this, arguments);
        syncDomSize();
        requestRedraw();
        return r;
    };

    const resizeObserver = new ResizeObserver(() => requestRedraw());
    resizeObserver.observe(container);

    node._tsIdeoApplyDesign = applyDesign;
    node._tsIdeoSync = () => {
        state.design = parseDesign(readPersistedDesign(node));
        ensureRefImage();
        updateSummary();
        requestRedraw();
    };
    node._tsIdeoCleanup = () => {
        // Tear down an editor left open for this node (its close() also removes
        // the document-level keydown/paste listeners and the JSON poll timer).
        try { node._tsIdeoEditorClose?.(); } catch { /* ignore */ }
        resizeObserver.disconnect();
        api.removeEventListener("execution_error", onExecError);
        if (autoTextTimer) { clearTimeout(autoTextTimer); autoTextTimer = null; }
    };

    const prevOnRemoved = node.onRemoved;
    node.onRemoved = function onRemoved() {
        try { node._tsIdeoCleanup?.(); } catch { /* ignore */ }
        return prevOnRemoved?.apply(this, arguments);
    };

    // Initial load
    applyMode(mode, false);
    syncDomSize();
    updateSummary();
    ensureRefImage();
    requestRedraw();
    loadPresets().then((presets) => {
        state.presets = presets;
        updateSummary();
    });
}

export { DOM_WIDGET_NAME, NODE_NAME };
