// In-node preview widget for TS_IdeogramDesigner.
//
// Renders a fluid, aspect-correct preview of the current design (reference
// underlay + block rectangles + labels), an "Edit" button that opens the
// full-screen modal editor, and a one-line summary. Fluid sizing + Nodes 1.0 /
// Nodes 2.0 (Vue) compatibility follow the verified sam_media_loader patterns:
// addDOMWidget with getMinHeight/getMaxHeight (no widget.computeSize), DPR
// canvas, ResizeObserver, syncDomSize, cleanup on removal.

import {
    DESIGN_INPUT,
    NODE_NAME,
    aspectFitBox,
    fontsById,
    getWidgetValue,
    hideWidget,
    inputViewUrl,
    loadPresets,
    makeDefaultDesign,
    parseDesign,
    persistDesign,
    setWidgetValue,
    stopPropagation,
} from "./_ideogram_shared.js";

import { openIdeogramEditor } from "./_ideogram_editor.js";

const STYLE_ID = "ts-ideogram-node-styles";
const DOM_WIDGET_NAME = "ts_ideogram_node";
const DEFAULT_NODE_SIZE = [320, 300];
const MIN_NODE_WIDTH = 240;
const MIN_NODE_HEIGHT = 220;
const PAD = 10;
const TOOLBAR_H = 34;
const SUMMARY_H = 22;

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-ideo-node{--tsi-bg:#0e1218;--tsi-text:#e9eef6;--tsi-muted:#9aa6b8;--tsi-accent:#7aa2ff;--tsi-accent2:#3a72ff;position:relative;width:100%;height:100%;min-height:0;box-sizing:border-box;color:var(--tsi-text);font-family:"Segoe UI",Tahoma,sans-serif;background:repeating-conic-gradient(#171c26 0% 25%,#11151c 0% 50%) 50%/22px 22px;border:1px solid #28303c;border-radius:10px;overflow:hidden;user-select:none}
.ts-ideo-node__canvas{position:absolute;inset:0;display:block;width:100%;height:100%}
.ts-ideo-node__toolbar{position:absolute;top:6px;left:6px;right:6px;height:${TOOLBAR_H - 8}px;display:flex;align-items:center;gap:6px;z-index:3}
.ts-ideo-node__btn{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--tsi-accent2);background:linear-gradient(180deg,#7aa2ff,#3a72ff);color:#0b1530;border-radius:8px;padding:6px 12px;font-size:12px;font-weight:700;cursor:pointer;letter-spacing:.02em}
.ts-ideo-node__btn:hover{background:linear-gradient(180deg,#90b6ff,#5180ff)}
.ts-ideo-node__pill{margin-left:auto;font-size:10px;color:var(--tsi-muted);background:rgba(20,26,36,.85);border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:4px 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:55%}
.ts-ideo-node__summary{position:absolute;left:6px;right:6px;bottom:6px;height:${SUMMARY_H - 6}px;display:flex;align-items:center;gap:8px;font-size:11px;color:var(--tsi-muted);background:rgba(12,16,22,.7);border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:0 8px;z-index:3;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden}
.ts-ideo-node__warn{color:#ffcf6b}
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

export function setupIdeogramNode(node) {
    if (!node || typeof node.addDOMWidget !== "function") return;
    if (typeof node._tsIdeoCleanup === "function") {
        try { node._tsIdeoCleanup(); } catch { /* ignore */ }
    }
    removeDomWidget(node);
    ensureStyles();
    hideWidget(node, DESIGN_INPUT);

    node.resizable = true;
    node.size = [
        Math.max(Number(node.size?.[0]) || DEFAULT_NODE_SIZE[0], MIN_NODE_WIDTH),
        Math.max(Number(node.size?.[1]) || DEFAULT_NODE_SIZE[1], MIN_NODE_HEIGHT),
    ];
    node.min_size = [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];

    const state = {
        design: parseDesign(getWidgetValue(node, DESIGN_INPUT, "")),
        presets: { styles: [], fonts: [] },
        fontsById: {},
        refImg: null,
        refKey: "",
    };

    const container = document.createElement("div");
    container.className = "ts-ideo-node";

    const canvas = document.createElement("canvas");
    canvas.className = "ts-ideo-node__canvas";

    const toolbar = document.createElement("div");
    toolbar.className = "ts-ideo-node__toolbar";
    const editBtn = document.createElement("button");
    editBtn.className = "ts-ideo-node__btn";
    editBtn.textContent = "✎ Edit design";
    const aspectPill = document.createElement("div");
    aspectPill.className = "ts-ideo-node__pill";
    aspectPill.textContent = "16x9";
    toolbar.append(editBtn, aspectPill);

    const summary = document.createElement("div");
    summary.className = "ts-ideo-node__summary";

    container.append(canvas, toolbar, summary);
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
        const styleName = state.design.style?.preset_id || "—";
        aspectPill.textContent = state.design.aspect_ratio || "16x9";
        summary.innerHTML = "";
        const main = document.createElement("span");
        main.textContent = `${texts} текст · ${objs} obj${placeholders ? ` · ${placeholders} плашка` : ""} · ${styleName}`;
        summary.appendChild(main);
        if (blocks.some((b) => b.type === "text" && !b.visual_only && /[Ѐ-ӿ]/.test(b.text || ""))) {
            const warn = document.createElement("span");
            warn.className = "ts-ideo-node__warn";
            warn.textContent = "⚠ кириллица";
            warn.title = "Кириллица в Ideogram 4 менее надёжна латиницы. Для печати — visual-only + ручной оверлей.";
            summary.appendChild(warn);
        }
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
        img.onerror = () => { state.refImg = null; };
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
        const availH = rectHeight - TOOLBAR_H - SUMMARY_H - PAD;
        if (availW <= 4 || availH <= 4) return;
        const box = aspectFitBox(state.design.aspect_ratio, availW, availH);
        const ax = PAD + (availW - box.w) / 2;
        const ay = TOOLBAR_H + (availH - box.h) / 2;

        // Artboard
        ctx.fillStyle = "#0a0d12";
        ctx.fillRect(ax, ay, box.w, box.h);
        if (state.refImg) {
            ctx.save();
            ctx.globalAlpha = 0.65;
            ctx.drawImage(state.refImg, ax, ay, box.w, box.h);
            ctx.restore();
        }
        ctx.strokeStyle = "#3a4658";
        ctx.lineWidth = 1;
        ctx.strokeRect(ax + 0.5, ay + 0.5, box.w - 1, box.h - 1);

        for (const block of state.design.blocks || []) {
            const r = block.rect;
            if (!r) continue;
            const bx = ax + r.x * box.w;
            const by = ay + r.y * box.h;
            const bw = Math.max(2, r.w * box.w);
            const bh = Math.max(2, r.h * box.h);
            const isText = block.type === "text";
            const visualOnly = isText && block.visual_only;
            const accent = visualOnly ? "#9aa6b8" : isText ? "#7aa2ff" : "#82d6a8";
            ctx.save();
            ctx.globalAlpha = 0.18;
            ctx.fillStyle = accent;
            ctx.fillRect(bx, by, bw, bh);
            ctx.globalAlpha = 1;
            ctx.strokeStyle = accent;
            ctx.lineWidth = 1.5;
            if (visualOnly) ctx.setLineDash([4, 3]);
            ctx.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
            ctx.setLineDash([]);
            // Label
            const label = isText
                ? (block.visual_only ? "↳ вручную" : (block.text || "").split("\n")[0] || "текст")
                : (block.desc || "obj").slice(0, 24);
            ctx.font = "10px 'Segoe UI', sans-serif";
            ctx.textBaseline = "top";
            ctx.fillStyle = "#0b0e13";
            ctx.globalAlpha = 0.55;
            ctx.fillRect(bx, by, Math.min(bw, ctx.measureText(label).width + 10), 14);
            ctx.globalAlpha = 1;
            ctx.fillStyle = "#e9eef6";
            ctx.save();
            ctx.beginPath();
            ctx.rect(bx, by, bw, bh);
            ctx.clip();
            ctx.fillText(label, bx + 4, by + 2);
            ctx.restore();
            ctx.restore();
        }

        if (!(state.design.blocks || []).length) {
            ctx.fillStyle = "#6b7688";
            ctx.font = "12px 'Segoe UI', sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("Нажмите «Edit design»", ax + box.w / 2, ay + box.h / 2);
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
        persistDesign(json);
        ensureRefImage();
        updateSummary();
        requestRedraw();
        node.setDirtyCanvas(true, true);
    }

    async function openEditor() {
        if (!state.presets.fonts?.length && !state.presets.styles?.length) {
            state.presets = await loadPresets();
            state.fontsById = fontsById(state.presets);
        }
        openIdeogramEditor(node, {
            design: state.design,
            presets: state.presets,
            onSave: (design) => applyDesign(design),
        });
    }

    editBtn.addEventListener("click", (e) => { e.stopPropagation(); openEditor(); });
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
        state.design = parseDesign(getWidgetValue(node, DESIGN_INPUT, ""));
        ensureRefImage();
        updateSummary();
        requestRedraw();
    };
    node._tsIdeoCleanup = () => {
        resizeObserver.disconnect();
    };

    const prevOnRemoved = node.onRemoved;
    node.onRemoved = function onRemoved() {
        try { node._tsIdeoCleanup?.(); } catch { /* ignore */ }
        return prevOnRemoved?.apply(this, arguments);
    };

    // Initial load
    syncDomSize();
    updateSummary();
    ensureRefImage();
    requestRedraw();
    loadPresets().then((presets) => {
        state.presets = presets;
        state.fontsById = fontsById(presets);
        updateSummary();
    });
}

export { DOM_WIDGET_NAME, NODE_NAME };
