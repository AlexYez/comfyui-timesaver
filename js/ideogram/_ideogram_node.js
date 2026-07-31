// In-node preview widget for TS_IdeogramDesigner.
//
// Renders a fluid, aspect-correct preview of the current design (reference
// underlay + block rectangles + labels), the shared "Open Interface" launcher
// that opens the full-screen modal editor, and a one-line summary. Fluid sizing + Nodes 1.0 /
// Nodes 2.0 (Vue) compatibility follow the verified sam_media_loader patterns:
// addDOMWidget with getMinHeight/getMaxHeight (no widget.computeSize), DPR
// canvas, ResizeObserver, syncDomSize, cleanup on removal.

import {
    ASPECT_RATIOS,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_LANG,
    DEFAULT_MEGAPIXELS,
    DESIGN_INPUT,
    MAX_MEGAPIXELS,
    MIN_MEGAPIXELS,
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
const AI_EVENT_PREFIX = "ts_super_prompt";

// The fullscreen editor drives megapixels with a continuous slider; the node
// panel has room for a dropdown, so offer the same range in 0.1 steps and snap
// whatever the editor left behind to the nearest one.
const MEGAPIXEL_STEPS = Array.from(
    { length: Math.round((MAX_MEGAPIXELS - MIN_MEGAPIXELS) * 10) + 1 },
    (_, i) => Math.round((MIN_MEGAPIXELS + i * 0.1) * 10) / 10,
);

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
        aspectTitle: "Aspect ratio of the generated image. Shared with the fullscreen editor.",
        mpTitle: "Output size in megapixels. Shared with the fullscreen editor.",
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
        aspectTitle: "Соотношение сторон. Общее с полноэкранным редактором.",
        mpTitle: "Размер вывода в мегапикселях. Общий с полноэкранным редактором.",
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
/* Every other child of .ts-ideo-node is position:absolute, so its
   min-content height is 0 and the Nodes 2.0 (Vue) layout collapses the
   widget to a 2px sliver at node-creation time — the toolbar then floats
   over the bare graph canvas and its launcher button is unclickable until a
   manual resize. This hidden in-flow spacer gives the container a real
   min-content height so Vue grants it space immediately. */
.ts-ideo-node__spacer{flex:0 0 auto;width:1px;height:${MIN_NODE_HEIGHT - TOP_CHROME}px;visibility:hidden;pointer-events:none}
.ts-ideo-node__modes{position:absolute;top:6px;left:6px;right:6px;height:${MODE_H - 8}px;display:flex;gap:4px;z-index:3}
.ts-ideo-node__modes .ts-ui-btn{flex:1 1 0;padding:3px 6px;font-size:var(--ts-fs-xs)}
.ts-ideo-node__auto{position:absolute;left:6px;right:6px;top:${MODE_H + 4}px;bottom:${SUMMARY_H + 4}px;display:none;flex-direction:column;gap:6px;z-index:3}
.ts-ideo-node__auto.is-active{display:flex}
.ts-ideo-node__auto-text{flex:1 1 55%;min-height:0;resize:none}
.ts-ideo-node__auto-result{flex:1 1 45%;min-height:0;overflow:auto;font-size:var(--ts-fs-xs);white-space:pre-wrap;word-break:break-word;background:var(--ts-sunken);border:1px solid var(--ts-border-soft);border-radius:var(--ts-radius-sm);padding:4px 6px;color:var(--ts-muted)}
.ts-ideo-node__auto-dims{flex:0 0 auto;display:flex;align-items:center;gap:6px}
.ts-ideo-node__auto-dims .ts-ui-select{flex:1 1 0;min-width:0;font-size:var(--ts-fs-sm);padding:2px 4px}
.ts-ideo-node__auto-dimslabel{flex:0 0 auto;font-size:var(--ts-fs-xs);color:var(--ts-faint);font-variant-numeric:tabular-nums;white-space:nowrap}
.ts-ideo-node__auto-status{flex:0 0 auto;min-height:14px;font-size:var(--ts-fs-sm);text-align:center}
.ts-ideo-node__auto-bar{flex:0 0 auto;height:3px;border-radius:2px;background:var(--ts-border-soft);overflow:hidden;display:none}
.ts-ideo-node__auto-bar.is-active{display:block}
.ts-ideo-node__auto-bar div{height:100%;width:0%;background:var(--ts-accent);transition:width .25s ease}
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
    const persist = (name, value) => {
        setWidgetValue(node, name, value);
        node.properties = node.properties || {};
        node.properties[name] = value;
    };
    const readPersisted = (name, fallback) => {
        // Both channels are checked in turn, not merged with ?? — a widget
        // holding its (empty) default would otherwise mask the properties
        // mirror that actually carries the restored value. Today hideWidget
        // removes the widget from node.widgets, so only the mirror is ever
        // found; this keeps working if that ever stops being true
        // (CLAUDE.md §12.5.13).
        const usable = (value) => value !== undefined && value !== null && value !== "";
        const widgetValue = node.widgets?.find((x) => x?.name === name)?.value;
        if (usable(widgetValue)) return widgetValue;
        const mirrored = node.properties?.[name];
        return usable(mirrored) ? mirrored : fallback;
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
    const autoBar = document.createElement("div");
    autoBar.className = "ts-ideo-node__auto-bar";
    const autoBarFill = document.createElement("div");
    autoBar.appendChild(autoBarFill);
    const autoResult = document.createElement("div");
    autoResult.className = "ts-ideo-node__auto-result";
    autoResult.textContent = String(readPersisted(AUTO_CAPTION_INPUT, "") || "") || C.resultHint;
    // Output size lives in the design document, which is also what the
    // fullscreen editor edits and what execute() reads through
    // dims_from_design. Editing it here therefore IS the synchronisation —
    // there is no second copy to keep in step.
    const dimsRow = document.createElement("div");
    dimsRow.className = "ts-ideo-node__auto-dims";

    const aspectSelect = document.createElement("select");
    aspectSelect.className = "ts-ui-select";
    aspectSelect.title = C.aspectTitle;
    for (const ratio of ASPECT_RATIOS) {
        const option = document.createElement("option");
        option.value = ratio;
        option.textContent = ratio.replace("x", ":");
        aspectSelect.appendChild(option);
    }

    const mpSelect = document.createElement("select");
    mpSelect.className = "ts-ui-select";
    mpSelect.title = C.mpTitle;
    for (const mp of MEGAPIXEL_STEPS) {
        const option = document.createElement("option");
        option.value = String(mp);
        option.textContent = `${mp} MP`;
        mpSelect.appendChild(option);
    }

    const dimsLabel = document.createElement("span");
    dimsLabel.className = "ts-ideo-node__auto-dimslabel";

    dimsRow.append(aspectSelect, mpSelect, dimsLabel);

    // Reflect the design — the editor may have changed it while this panel was
    // hidden, and a stale dropdown would lie about what the node will output.
    function syncAutoDims() {
        const aspect = String(state.design?.aspect_ratio || DEFAULT_ASPECT_RATIO);
        const mp = Number(state.design?.megapixels ?? DEFAULT_MEGAPIXELS);
        aspectSelect.value = ASPECT_RATIOS.includes(aspect) ? aspect : DEFAULT_ASPECT_RATIO;
        const nearest = MEGAPIXEL_STEPS.reduce(
            (best, step) => (Math.abs(step - mp) < Math.abs(best - mp) ? step : best),
            MEGAPIXEL_STEPS[0],
        );
        mpSelect.value = String(nearest);
        const [width, height] = dimsFromAspectMp(aspectSelect.value, Number(mpSelect.value));
        dimsLabel.textContent = `${width}×${height}`;
    }
    node._tsIdeoSyncAutoDims = syncAutoDims;

    aspectSelect.addEventListener("change", () => {
        applyDesign({ ...state.design, aspect_ratio: aspectSelect.value });
        syncAutoDims();
    });
    mpSelect.addEventListener("change", () => {
        applyDesign({ ...state.design, megapixels: Number(mpSelect.value) });
        syncAutoDims();
    });
    for (const control of [aspectSelect, mpSelect]) {
        control.addEventListener("pointerdown", stopPropagation);
    }

    autoPanel.append(autoText, dimsRow, generateBtn, autoBar, autoStatus, autoResult);

    const spacer = document.createElement("div");
    spacer.className = "ts-ideo-node__spacer";
    container.append(spacer, canvas, modesRow, toolbar, autoPanel, summary);
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
        if (persist) {
            setWidgetValue(node, MODE_INPUT, mode);
            node.properties = node.properties || {};
            node.properties[MODE_INPUT] = mode;
        }
    }
    designerBtn.addEventListener("click", (e) => { e.stopPropagation(); applyMode("designer"); });
    autoBtn.addEventListener("click", (e) => { e.stopPropagation(); applyMode("auto"); });

    let autoTextTimer = null;
    autoText.addEventListener("input", () => {
        if (autoTextTimer) clearTimeout(autoTextTimer);
        autoTextTimer = setTimeout(() => {
            autoTextTimer = null;
            persist(AUTO_PROMPT_INPUT, autoText.value);
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
        persist(AUTO_PROMPT_INPUT, autoText.value);
        // The bumped seed is SENT with the request. Bumping it alone only
        // invalidated the node's cache: the engine sampled from a fixed seed,
        // so pressing Generate twice on the same idea returned the identical
        // caption — exactly what the button promises not to do.
        const seed = Number(readPersisted(AUTO_SEED_INPUT, 0)) || 0;
        const nextSeed = (seed + 1) & 0x7fffffff;
        persist(AUTO_SEED_INPUT, nextSeed);
        setAutoStatus(C.running);
        generateBtn.disabled = true;
        // The SuperPrompt engine generates immediately (no workflow queue).
        // Route, preset and event names are a shared contract pinned by
        // tests/test_ideogram_superprompt_contract.py — edit them in sync.
        activeOperationId = `ts_ideo_${node.id}_${Math.random().toString(36).slice(2, 10)}`;
        setBar(true, 2);
        try {
            const response = await api.fetchApi(ENHANCE_ROUTE, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: autoText.value,
                    system_preset: ENHANCE_PRESET,
                    operation_id: activeOperationId,
                    seed: nextSeed,
                }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || payload?.error) {
                throw new Error(payload?.error || `HTTP ${response.status}`);
            }
            const caption = String(payload?.text || "").trim();
            if (!caption) throw new Error("empty reply");
            autoResult.textContent = caption;
            persist(AUTO_CAPTION_INPUT, caption);
            setAutoStatus(C.done, "success");
        } catch (error) {
            setAutoStatus(C.runFailed(error?.message || error), "error");
        } finally {
            generateBtn.disabled = false;
            activeOperationId = null;
            setBar(false);
        }
    }
    function setBar(active, percent) {
        autoBar.classList.toggle("is-active", Boolean(active));
        if (typeof percent === "number") autoBarFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    }
    // Real backend stages from the SuperPrompt engine (model download with
    // percentages, memory checks, generation), keyed by our operation id so a
    // SuperPrompt run elsewhere does not move this node's bar.
    let activeOperationId = null;
    const onAiProgress = (event) => {
        const d = event?.detail || {};
        if (!activeOperationId || d.operation_id !== activeOperationId) return;
        if (typeof d.percent === "number") setBar(true, d.percent);
        if (d.text) setAutoStatus(d.text);
    };
    api.addEventListener(`${AI_EVENT_PREFIX}.progress`, onAiProgress);
    generateBtn.addEventListener("click", (e) => { e.stopPropagation(); queueGenerate(); });

    const prevOnExecuted = node.onExecuted;
    node.onExecuted = function onExecutedWithAuto(message) {
        const result = prevOnExecuted?.apply(this, arguments);
        const caption = message?.ts_ideo_auto?.[0];
        if (typeof caption === "string" && caption) {
            autoResult.textContent = caption;
            persist(AUTO_CAPTION_INPUT, caption);
            setAutoStatus(C.done, "success");
        }
        generateBtn.disabled = false;
        setBar(false);
        return result;
    };
    const onExecError = (event) => {
        if (String(event?.detail?.node_id) !== String(node.id)) return;
        generateBtn.disabled = false;
        setBar(false);
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
        // The editor writes through here too, so the Auto panel's dropdowns
        // follow whatever it changed.
        node._tsIdeoSyncAutoDims?.();
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
        applyMode(String(readPersisted(MODE_INPUT, mode) || mode), false);
        const savedPrompt = String(readPersisted(AUTO_PROMPT_INPUT, "") || "");
        if (savedPrompt && autoText.value !== savedPrompt) autoText.value = savedPrompt;
        const savedCaption = String(readPersisted(AUTO_CAPTION_INPUT, "") || "");
        if (savedCaption) autoResult.textContent = savedCaption;
        state.design = parseDesign(readPersistedDesign(node));
        ensureRefImage();
        updateSummary();
        syncAutoDims();
        requestRedraw();
    };
    node._tsIdeoCleanup = () => {
        // Tear down an editor left open for this node (its close() also removes
        // the document-level keydown/paste listeners and the JSON poll timer).
        try { node._tsIdeoEditorClose?.(); } catch { /* ignore */ }
        resizeObserver.disconnect();
        api.removeEventListener("execution_error", onExecError);
        api.removeEventListener(`${AI_EVENT_PREFIX}.progress`, onAiProgress);
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
    syncAutoDims();
    ensureRefImage();
    requestRedraw();
    loadPresets().then((presets) => {
        state.presets = presets;
        updateSummary();
    });
}

export { DOM_WIDGET_NAME, NODE_NAME };
