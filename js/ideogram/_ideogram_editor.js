// Full-screen modal editor for TS_IdeogramDesigner (v2).
//
// Mounted on document.body (fixed overlay) so it works identically in Nodes 1.0
// (LiteGraph) and Nodes 2.0 (Vue). Two-level presets: a layout template (what to
// make → instantiates placeholder blocks) and a style (aesthetic → palette +
// fonts + style_description). Localized RU/EN UI, double-click inline editing of
// text/object blocks, a stylized canvas preview, and custom-preset saving.
//
// openIdeogramEditor(node, { design, presets, onSave }) — onSave(newDesign).

import { api } from "/scripts/api.js";

import {
    CASES,
    DEFAULT_LANG,
    ELEMENT_PALETTE_CAP,
    IMAGE_PALETTE_CAP,
    LANGS,
    PHOTO_MEDIUM,
    PROMINENCE,
    ROUTE_BASE,
    WEIGHTS,
    applyCase,
    applyStyle,
    aspectFitBox,
    ASPECT_RATIOS,
    clamp,
    cleanPalette,
    composeTextDesc,
    DEFAULT_MEGAPIXELS,
    dimsFromAspectMp,
    MAX_MEGAPIXELS,
    MIN_MEGAPIXELS,
    fontFamilyForPreset,
    fontsById as buildFontsById,
    fracToBbox,
    instantiateLayout,
    inputViewUrl,
    invalidatePresetsCache,
    layoutsList,
    localizedDesc,
    localizedName,
    makeBlockId,
    normHex,
    paletteGradientCss,
    segLabel,
    stylesList,
    t,
} from "./_ideogram_shared.js";

const STYLE_ID = "ts-ideogram-editor-styles";
const MEDIA_OPTIONS = ["graphic_design", "photograph", "illustration", "3d_render", "painting", "digital_painting"];

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-ideoe-overlay{position:fixed;inset:0;z-index:11000;display:flex;flex-direction:column;background:rgba(6,9,13,.86);backdrop-filter:blur(3px);color:#e9eef6;font-family:"Segoe UI",Tahoma,sans-serif;font-size:13px}
.ts-ideoe-shell{position:absolute;inset:24px;display:flex;flex-direction:column;background:#0e1218;border:1px solid #283040;border-radius:14px;overflow:hidden;box-shadow:0 24px 80px rgba(0,0,0,.6)}
.ts-ideoe-header{display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid #1c2430;background:#10151d;flex:0 0 auto;flex-wrap:wrap}
.ts-ideoe-title{font-weight:700;font-size:14px;letter-spacing:.02em}
.ts-ideoe-title small{color:#8a93a3;font-weight:400;margin-left:8px}
.ts-ideoe-spacer{flex:1 1 auto}
.ts-ideoe-body{flex:1 1 auto;display:flex;min-height:0}
.ts-ideoe-stagewrap{flex:1 1 auto;position:relative;display:flex;align-items:center;justify-content:center;min-width:0;background:repeating-conic-gradient(#141a24 0% 25%,#0e131b 0% 50%) 50%/26px 26px;overflow:hidden}
.ts-ideoe-stage{position:relative;outline:none}
.ts-ideoe-artboard{position:absolute;left:0;top:0;background:#0a0d12;box-shadow:0 0 0 1px #38445a, 0 10px 40px rgba(0,0,0,.5);overflow:hidden}
.ts-ideoe-artboard .grid{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.05) 1px,transparent 1px);background-size:10% 10%;pointer-events:none}
.ts-ideoe-ref{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;pointer-events:none}
.ts-ideoe-block{position:absolute;box-sizing:border-box;border:1.5px solid #7aa2ff;background:rgba(122,162,255,.14);cursor:move;overflow:hidden;border-radius:2px}
.ts-ideoe-block.is-obj{border-color:#82d6a8;background:rgba(130,214,168,.16)}
.ts-ideoe-block.is-visual{border-style:dashed;border-color:#9aa6b8;background:rgba(154,166,184,.12)}
.ts-ideoe-block.is-selected{box-shadow:0 0 0 2px #ffd500, 0 0 0 4px rgba(255,213,0,.25);z-index:5}
.ts-ideoe-block__label{position:absolute;left:0;top:0;max-width:100%;padding:1px 5px;font-size:11px;font-weight:600;color:#0b0e13;background:rgba(255,255,255,.82);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;border-bottom-right-radius:4px}
.ts-ideoe-block__text{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:6px;font-weight:800;line-height:1.05;text-shadow:0 1px 2px rgba(0,0,0,.7);white-space:pre-wrap;overflow:hidden;pointer-events:none}
.ts-ideoe-block__obj{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-size:12px;font-weight:600;color:rgba(255,255,255,.85);overflow:hidden;pointer-events:none}
.ts-ideoe-handle{position:absolute;width:11px;height:11px;background:#ffd500;border:1px solid #1c1c1c;border-radius:2px;z-index:6}
.ts-ideoe-handle.nw{left:-6px;top:-6px;cursor:nwse-resize}.ts-ideoe-handle.ne{right:-6px;top:-6px;cursor:nesw-resize}
.ts-ideoe-handle.sw{left:-6px;bottom:-6px;cursor:nesw-resize}.ts-ideoe-handle.se{right:-6px;bottom:-6px;cursor:nwse-resize}
.ts-ideoe-handle.n{left:50%;top:-6px;transform:translateX(-50%);cursor:ns-resize}.ts-ideoe-handle.s{left:50%;bottom:-6px;transform:translateX(-50%);cursor:ns-resize}
.ts-ideoe-handle.w{left:-6px;top:50%;transform:translateY(-50%);cursor:ew-resize}.ts-ideoe-handle.e{right:-6px;top:50%;transform:translateY(-50%);cursor:ew-resize}
.ts-ideoe-inline{position:absolute;z-index:20;box-sizing:border-box;border:2px solid #ffd500;border-radius:3px;background:rgba(10,14,20,.94);color:#fff;font:700 14px/1.15 "Segoe UI",sans-serif;padding:4px 6px;resize:none;outline:none;text-align:center}
.ts-ideoe-inspector{flex:0 0 340px;display:flex;flex-direction:column;border-left:1px solid #1c2430;background:#0c1118;min-height:0}
.ts-ideoe-inspector__scroll{flex:1 1 auto;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:12px}
.ts-ideoe-card{border:1px solid #1f2937;border-radius:10px;background:#0f151d;padding:10px;display:flex;flex-direction:column;gap:8px}
.ts-ideoe-card h3{margin:0;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#8a93a3}
.ts-ideoe-row{display:flex;flex-direction:column;gap:4px}
.ts-ideoe-row label{font-size:11px;color:#9aa6b8}
.ts-ideoe-hint{font-size:11px;color:#8a93a3;line-height:1.4}
.ts-ideoe-banner{margin:0 12px;padding:8px 10px;border:1px solid #6b561f;background:rgba(255,207,107,.1);color:#ffcf6b;border-radius:8px;font-size:11px;line-height:1.4}
.ts-ideoe-warns{display:flex;flex-direction:column;gap:4px}
.ts-ideoe-warn{font-size:11px;color:#ffcf6b;background:rgba(255,207,107,.08);border:1px solid rgba(255,207,107,.25);border-radius:6px;padding:4px 7px}
.ts-ideoe input[type=text],.ts-ideoe textarea,.ts-ideoe-inspector input[type=text],.ts-ideoe-inspector textarea,.ts-ideoe-inspector select{width:100%;box-sizing:border-box;background:#0a0e14;border:1px solid #28323f;border-radius:6px;color:#e9eef6;padding:6px 8px;font-size:12px;font-family:inherit;outline:none}
.ts-ideoe-inspector textarea{resize:vertical;min-height:46px}
.ts-ideoe-inspector input:focus,.ts-ideoe-inspector textarea:focus,.ts-ideoe-inspector select:focus{border-color:#4da3ff}
.ts-ideoe-seg{display:flex;border:1px solid #28323f;border-radius:6px;overflow:hidden}
.ts-ideoe-seg button{flex:1 1 auto;background:#0a0e14;color:#9aa6b8;border:0;border-right:1px solid #1b232e;padding:5px 4px;font-size:11px;cursor:pointer}
.ts-ideoe-seg button:last-child{border-right:0}
.ts-ideoe-seg button.is-on{background:linear-gradient(180deg,#7aa2ff,#3a72ff);color:#0b1530;font-weight:700}
.ts-ideoe-langseg{display:inline-flex;border:1px solid #28323f;border-radius:8px;overflow:hidden}
.ts-ideoe-langseg button{background:#0a0e14;color:#9aa6b8;border:0;border-right:1px solid #1b232e;padding:6px 11px;font-size:12px;font-weight:700;cursor:pointer}
.ts-ideoe-langseg button:last-child{border-right:0}
.ts-ideoe-langseg button.is-on{background:linear-gradient(180deg,#7aa2ff,#3a72ff);color:#0b1530}
.ts-ideoe-checks{display:flex;flex-wrap:wrap;gap:8px}
.ts-ideoe-check{display:flex;align-items:center;gap:5px;font-size:11px;color:#cdd6e6;cursor:pointer}
.ts-ideoe-pal{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
.ts-ideoe-sw{width:20px;height:20px;border-radius:4px;border:1px solid rgba(255,255,255,.25);position:relative;cursor:pointer}
.ts-ideoe-sw:hover::after{content:"×";position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;background:rgba(0,0,0,.45);border-radius:4px}
.ts-ideoe-paladd{position:relative;overflow:hidden;display:inline-flex;align-items:center;gap:4px;height:20px;padding:0 9px;border:1px dashed #4a5568;border-radius:5px;background:#0f151d;color:#9fb0c8;font-size:11px;cursor:pointer;white-space:nowrap}
.ts-ideoe-paladd:hover{border-color:#7aa2ff;color:#cfe0ff}
.ts-ideoe-palinput{position:absolute;inset:0;width:100%;height:100%;margin:0;padding:0;border:0;opacity:0;cursor:pointer}
.ts-ideoe-descprev{font-size:11px;color:#9fe3c2;background:#08120d;border:1px solid #1c3a2c;border-radius:6px;padding:6px 8px;white-space:pre-wrap;word-break:break-word}
.ts-ideoe-bbox{font-size:10px;color:#7d899b;font-variant-numeric:tabular-nums}
.ts-ideoe-btn{display:inline-flex;align-items:center;gap:5px;border:1px solid #28323f;background:#161d27;color:#e9eef6;border-radius:8px;padding:6px 11px;font-size:12px;cursor:pointer;font-weight:600;white-space:nowrap}
.ts-ideoe-btn:hover{background:#1f2937}
.ts-ideoe-btn.primary{background:linear-gradient(180deg,#46d39a,#1fa97a);border-color:#1fa97a;color:#04130d}
.ts-ideoe-btn.primary:hover{background:linear-gradient(180deg,#5ee3ab,#27c08c)}
.ts-ideoe-btn.danger{background:#3a1d1d;border-color:#6b2f2f;color:#ffb4b1}
.ts-ideoe-btn.ghost{background:transparent}
.ts-ideoe-btn.small{padding:5px 9px;font-size:11px}
.ts-ideoe-select{background:#0a0e14;border:1px solid #28323f;border-radius:6px;color:#e9eef6;padding:6px 8px;font-size:12px}
.ts-ideoe-mp{display:inline-flex;align-items:center;gap:6px;border:1px solid #28323f;border-radius:8px;padding:3px 8px;background:#0a0e14}
.ts-ideoe-mplabel{font-size:11px;color:#9aa6b8}
.ts-ideoe-mpinput{width:54px;background:#10151d;border:1px solid #28323f;border-radius:5px;color:#e9eef6;padding:4px 6px;font-size:12px;text-align:center}
.ts-ideoe-dims{font-size:11px;color:#7aa2ff;font-variant-numeric:tabular-nums;min-width:74px;text-align:right}
.ts-ideoe-mprange{-webkit-appearance:none;appearance:none;width:92px;height:4px;border-radius:3px;background:#28323f;outline:none;cursor:pointer}
.ts-ideoe-mprange::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:14px;border-radius:50%;background:linear-gradient(180deg,#7aa2ff,#3a72ff);border:1px solid #2a4a8f;cursor:pointer}
.ts-ideoe-mprange::-moz-range-thumb{width:13px;height:13px;border:1px solid #2a4a8f;border-radius:50%;background:#5180ff;cursor:pointer}
.ts-ideoe-mpval{font-size:11px;color:#cdd6e6;font-variant-numeric:tabular-nums;min-width:24px;text-align:center}
.ts-ideoe-btnrow{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.ts-ideoe-hint{font-size:11px;color:#8a93a3;line-height:1.4;margin:2px 0 0}
.ts-ideoe-empty{color:#6b7688;font-size:12px;text-align:center;padding:24px 8px}
`;
    document.head.appendChild(style);
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
}

function buildSegmented(options, getValue, onPick, labelFn) {
    const wrap = el("div", "ts-ideoe-seg");
    const buttons = [];
    options.forEach((opt) => {
        const btn = el("button", null, labelFn ? labelFn(opt) : opt);
        btn.addEventListener("click", () => { onPick(opt); refresh(); });
        wrap.appendChild(btn);
        buttons.push([opt, btn]);
    });
    function refresh() {
        const cur = getValue();
        buttons.forEach(([opt, btn]) => btn.classList.toggle("is-on", opt === cur));
    }
    refresh();
    return wrap;
}

function buildPalette(getArr, setArr, cap, lang) {
    const wrap = el("div", "ts-ideoe-pal");
    function render() {
        wrap.innerHTML = "";
        const arr = getArr() || [];
        arr.forEach((hex, i) => {
            const sw = el("div", "ts-ideoe-sw");
            sw.style.background = hex;
            sw.title = `${hex}`;
            sw.addEventListener("click", () => {
                const next = arr.slice();
                next.splice(i, 1);
                setArr(next);
                render();
            });
            wrap.appendChild(sw);
        });
        if (arr.length < cap) {
            // A real <input type=color> overlaying the button: the native picker
            // anchors to the input's on-screen box, so it pops up right at the
            // button and tracks inspector scroll. A <label> wrapper activates the
            // input on click natively — the old zero-size hidden input + synthetic
            // .click() mis-anchored the popup (far below / off-screen when scrolled,
            // which read as "nothing happens" in the lower block palette).
            const add = el("label", "ts-ideoe-paladd");
            const picker = el("input", "ts-ideoe-palinput");
            picker.type = "color";
            picker.value = "#3A72FF";
            picker.addEventListener("change", () => {
                const hex = normHex(picker.value);
                if (hex) {
                    const next = (getArr() || []).slice();
                    if (!next.includes(hex)) next.push(hex);
                    setArr(cleanPalette(next, cap));
                }
                render();
            });
            // Keep the click from reaching inspector/stage handlers; never
            // preventDefault — the default action is what opens the picker.
            picker.addEventListener("click", (ev) => ev.stopPropagation());
            add.append(document.createTextNode(`＋ ${t("add_color", lang)}`), picker);
            wrap.appendChild(add);
        }
    }
    render();
    return wrap;
}

const WEIGHT_CSS = { Thin: 300, Regular: 400, Bold: 700, Heavy: 900 };
function weightToCss(w) { return WEIGHT_CSS[w] || 700; }

// Shrink a text element's font-size (binary search) so it fits its own box on
// both axes — the WYSIWYG auto-fit, since the node has no explicit font-size
// control. Must be called after the element is in the DOM and sized.
function fitText(textEl, iters = 12) {
    const boxW = textEl.clientWidth;
    const boxH = textEl.clientHeight;
    if (boxW < 4 || boxH < 4) return;
    let lo = 6, hi = Math.max(8, boxH), best = 6;
    for (let i = 0; i < iters && lo <= hi; i++) {
        const mid = (lo + hi) >> 1;
        textEl.style.fontSize = `${mid}px`;
        if (textEl.scrollWidth <= boxW && textEl.scrollHeight <= boxH) { best = mid; lo = mid + 1; }
        else { hi = mid - 1; }
    }
    textEl.style.fontSize = `${best}px`;
}

export function openIdeogramEditor(node, { design, presets, onSave }) {
    ensureStyles();

    const work = JSON.parse(JSON.stringify(design));
    work.blocks = Array.isArray(work.blocks) ? work.blocks : [];
    work.style = work.style || {};
    if (!LANGS.includes(work.language)) work.language = DEFAULT_LANG;

    const layouts = layoutsList(presets);
    const styles = stylesList(presets);
    const fontList = presets?.fonts || [];
    const fontMap = buildFontsById(presets);
    const tr = (key, vars) => t(key, work.language, vars);

    let selectedId = work.blocks[0]?.id || null;
    let lastClickInfo = { id: null, t: 0 };

    const overlay = el("div", "ts-ideoe-overlay ts-ideoe");
    const shell = el("div", "ts-ideoe-shell");

    // ── Header ──────────────────────────────────────────────────────────── //
    const header = el("div", "ts-ideoe-header");
    const title = el("div", "ts-ideoe-title", "TS Ideogram Designer");

    const addText = el("button", "ts-ideoe-btn");
    const addObj = el("button", "ts-ideoe-btn");
    const dupBtn = el("button", "ts-ideoe-btn");
    const delBtn = el("button", "ts-ideoe-btn danger");
    const clearBtn = el("button", "ts-ideoe-btn danger");

    const langSeg = el("div", "ts-ideoe-langseg");
    const langButtons = LANGS.map((lng) => {
        const b = el("button", null, lng.toUpperCase());
        b.addEventListener("click", () => setLanguage(lng));
        langSeg.appendChild(b);
        return [lng, b];
    });

    const aspectSel = el("select", "ts-ideoe-select");
    ASPECT_RATIOS.forEach((ar) => {
        const o = el("option", null, ar);
        o.value = ar;
        aspectSel.appendChild(o);
    });

    const mpWrap = el("div", "ts-ideoe-mp");
    const mpLabel = el("span", "ts-ideoe-mplabel");
    const mpInput = el("input", "ts-ideoe-mprange");
    mpInput.type = "range";
    mpInput.min = String(MIN_MEGAPIXELS);
    mpInput.max = String(MAX_MEGAPIXELS);
    mpInput.step = "0.1";
    const mpVal = el("span", "ts-ideoe-mpval");
    const dimsReadout = el("span", "ts-ideoe-dims");
    mpWrap.append(mpLabel, mpInput, mpVal, dimsReadout);

    const refBtn = el("button", "ts-ideoe-btn");
    const refClear = el("button", "ts-ideoe-btn ghost");
    const cancelBtn = el("button", "ts-ideoe-btn");
    const saveBtn = el("button", "ts-ideoe-btn primary");

    header.append(title, addText, addObj, dupBtn, delBtn, clearBtn, el("div", "ts-ideoe-spacer"),
        langSeg, aspectSel, mpWrap, refBtn, refClear, cancelBtn, saveBtn);

    function relabelHeader() {
        addText.textContent = tr("add_text");
        addObj.textContent = tr("add_obj");
        dupBtn.textContent = tr("duplicate");
        delBtn.textContent = tr("delete");
        refBtn.textContent = tr("reference");
        refClear.textContent = tr("clear_ref");
        cancelBtn.textContent = tr("cancel");
        saveBtn.textContent = tr("save");
        clearBtn.textContent = tr("clear");
        mpLabel.textContent = tr("mp_label");
        mpInput.value = String(work.megapixels ?? DEFAULT_MEGAPIXELS);
        mpVal.textContent = Number(work.megapixels ?? DEFAULT_MEGAPIXELS).toFixed(1);
        aspectSel.value = work.aspect_ratio;
        updateDimsReadout();
        langButtons.forEach(([lng, b]) => b.classList.toggle("is-on", lng === work.language));
    }

    function updateDimsReadout() {
        const d = dimsFromAspectMp(work.aspect_ratio, work.megapixels ?? DEFAULT_MEGAPIXELS);
        dimsReadout.textContent = `${d.w}×${d.h}`;
    }

    // ── Body ────────────────────────────────────────────────────────────── //
    const body = el("div", "ts-ideoe-body");
    const stageWrap = el("div", "ts-ideoe-stagewrap");
    const stage = el("div", "ts-ideoe-stage");
    const artboard = el("div", "ts-ideoe-artboard");
    const grid = el("div", "grid");
    let refImgEl = null;
    const blocksLayer = el("div");
    blocksLayer.style.cssText = "position:absolute;inset:0";
    artboard.append(grid, blocksLayer);
    stage.appendChild(artboard);
    stageWrap.appendChild(stage);

    const inspector = el("div", "ts-ideoe-inspector");
    const banner = el("div", "ts-ideoe-banner");
    banner.style.display = "none";
    const inspectorScroll = el("div", "ts-ideoe-inspector__scroll");
    inspector.append(banner, inspectorScroll);

    body.append(stageWrap, inspector);
    shell.append(header, body);
    overlay.appendChild(shell);
    document.body.appendChild(overlay);

    const fileInput = el("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.style.display = "none";
    overlay.appendChild(fileInput);

    const importInput = el("input");
    importInput.type = "file";
    importInput.accept = "application/json,.json";
    importInput.multiple = true;
    importInput.style.display = "none";
    overlay.appendChild(importInput);
    importInput.addEventListener("change", () => { handleImportFiles(Array.from(importInput.files || [])); });

    // ── Artboard sizing + style preview ─────────────────────────────────── //
    function layoutArtboard() {
        const availW = stageWrap.clientWidth - 32;
        const availH = stageWrap.clientHeight - 32;
        if (availW <= 0 || availH <= 0) return;
        const box = aspectFitBox(work.aspect_ratio, availW, availH);
        stage.style.width = `${box.w}px`;
        stage.style.height = `${box.h}px`;
        artboard.style.width = `${box.w}px`;
        artboard.style.height = `${box.h}px`;
    }
    function artboardSize() {
        const r = artboard.getBoundingClientRect();
        return { w: r.width || 1, h: r.height || 1, left: r.left, top: r.top };
    }
    function applyStylePreview() {
        if (work.ref) return;  // a reference underlay takes over the artboard background
        const css = paletteGradientCss(work.style?.color_palette || [], { alpha: 1, mesh: true });
        artboard.style.background = css || "#0a0d12";
    }

    function renderReference() {
        if (refImgEl) { refImgEl.remove(); refImgEl = null; }
        const ref = work.ref;
        if (ref?.filename) {
            refImgEl = el("img", "ts-ideoe-ref");
            refImgEl.src = inputViewUrl(ref.filename, ref.subfolder, ref.type);
            artboard.insertBefore(refImgEl, grid);
        }
        applyStylePreview();
    }

    async function uploadReference(file) {
        if (!file) return;
        try {
            const form = new FormData();
            form.append("image", file, file.name);
            form.append("overwrite", "true");
            const resp = await api.fetchApi("/upload/image", { method: "POST", body: form });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            work.ref = { filename: data.name, subfolder: data.subfolder || "", type: data.type || "input" };
            renderReference();
        } catch (error) {
            console.error("[TS Ideogram] reference upload failed:", error);
        }
    }

    // ── Blocks ──────────────────────────────────────────────────────────── //
    function getSelected() {
        return work.blocks.find((b) => b.id === selectedId) || null;
    }

    function renderBlocks() {
        applyStylePreview();
        blocksLayer.innerHTML = "";
        work.blocks.forEach((block) => {
            const r = block.rect || { x: 0.1, y: 0.1, w: 0.3, h: 0.2 };
            const div = el("div", "ts-ideoe-block");
            div.dataset.id = block.id;
            div.classList.toggle("is-obj", block.type === "obj");
            div.classList.toggle("is-visual", block.type === "text" && !!block.visual_only);
            div.classList.toggle("is-selected", block.id === selectedId);
            div.style.left = `${r.x * 100}%`;
            div.style.top = `${r.y * 100}%`;
            div.style.width = `${r.w * 100}%`;
            div.style.height = `${r.h * 100}%`;
            // WYSIWYG: a block's own palette tints/gradients its rectangle live.
            const bpal = cleanPalette(block.color_palette || [], ELEMENT_PALETTE_CAP);
            if (bpal.length) div.style.background = paletteGradientCss(bpal, { alpha: 0.55, mesh: false });

            let textEl = null;
            if (block.type === "text" && !block.visual_only) {
                textEl = el("div", "ts-ideoe-block__text");
                textEl.textContent = applyCase(block.text || "", block.case);
                textEl.style.color = normHex(block.color) || "#ffffff";
                textEl.style.fontFamily = fontFamilyForPreset(block.font_preset_id);
                textEl.style.fontWeight = weightToCss(block.weight);
                if (block.legibility && block.legibility.outline) {
                    textEl.style.webkitTextStroke = "1px rgba(0,0,0,.85)";
                    textEl.style.textShadow = "0 2px 5px rgba(0,0,0,.85)";
                } else {
                    textEl.style.webkitTextStroke = "0";
                    textEl.style.textShadow = "0 1px 2px rgba(0,0,0,.7)";
                }
                div.appendChild(textEl);
            } else if (block.type === "obj") {
                div.appendChild(el("div", "ts-ideoe-block__obj", block.desc || "obj"));
            }
            const label = el("div", "ts-ideoe-block__label",
                block.type === "obj" ? "OBJ" : (block.visual_only ? "↳" : "TXT"));
            div.appendChild(label);

            if (block.id === selectedId) {
                ["nw", "ne", "sw", "se", "n", "s", "e", "w"].forEach((dir) => {
                    const h = el("div", `ts-ideoe-handle ${dir}`);
                    h.addEventListener("pointerdown", (ev) => startDrag(ev, block, dir));
                    div.appendChild(h);
                });
            }

            // Single click → select + drag. Manual double-click detection (the
            // native dblclick is suppressed by startDrag's preventDefault).
            div.addEventListener("pointerdown", (ev) => {
                if (ev.target.classList.contains("ts-ideoe-handle")) return;
                const now = Date.now();
                if (lastClickInfo.id === block.id && now - lastClickInfo.t < 350) {
                    lastClickInfo = { id: null, t: 0 };
                    ev.preventDefault();
                    ev.stopPropagation();
                    startInlineEdit(block);
                    return;
                }
                lastClickInfo = { id: block.id, t: now };
                selectBlock(block.id);
                startDrag(ev, block, "move");
            });
            blocksLayer.appendChild(div);
            if (textEl) fitText(textEl);
        });
    }

    // ── Inline (double-click) editing on the canvas ─────────────────────── //
    let inlineEl = null;
    function startInlineEdit(block) {
        if (inlineEl) inlineEl.remove();
        if (block.type === "text" && block.visual_only) return;
        const r = block.rect || { x: 0.1, y: 0.1, w: 0.4, h: 0.2 };
        const ab = artboardSize();
        const ta = el("textarea", "ts-ideoe-inline");
        ta.value = block.type === "obj" ? (block.desc || "") : (block.text || "");
        ta.style.left = `${r.x * ab.w}px`;
        ta.style.top = `${r.y * ab.h}px`;
        ta.style.width = `${Math.max(60, r.w * ab.w)}px`;
        ta.style.height = `${Math.max(28, r.h * ab.h)}px`;
        if (block.type === "obj") { ta.style.fontWeight = "600"; ta.style.fontSize = "12px"; }
        artboard.appendChild(ta);
        inlineEl = ta;
        ta.focus();
        ta.select();
        const commit = () => {
            if (!inlineEl) return;
            const v = ta.value;
            if (block.type === "obj") block.desc = v; else block.text = v;
            inlineEl = null;
            ta.remove();
            renderBlocks();
            renderInspector();
        };
        const cancel = () => { inlineEl = null; ta.remove(); };
        ta.addEventListener("blur", commit);
        ta.addEventListener("keydown", (ev) => {
            ev.stopPropagation();
            if (ev.key === "Escape") { ev.preventDefault(); cancel(); }
            else if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); ta.blur(); }
        });
        ["pointerdown", "pointermove", "pointerup", "dblclick"].forEach((e) =>
            ta.addEventListener(e, (ev) => ev.stopPropagation()));
    }

    // ── Drag / resize ───────────────────────────────────────────────────── //
    function startDrag(ev, block, mode) {
        ev.preventDefault();
        ev.stopPropagation();
        const ab = artboardSize();
        const start = { mx: ev.clientX, my: ev.clientY, ...block.rect };
        function onMove(e) {
            const dx = (e.clientX - start.mx) / ab.w;
            const dy = (e.clientY - start.my) / ab.h;
            let { x, y, w, h } = start;
            if (mode === "move") {
                x = clamp(start.x + dx, 0, 1 - start.w);
                y = clamp(start.y + dy, 0, 1 - start.h);
            } else {
                if (mode.includes("e")) w = clamp(start.w + dx, 0.02, 1 - start.x);
                if (mode.includes("s")) h = clamp(start.h + dy, 0.02, 1 - start.y);
                if (mode.includes("w")) { const nx = clamp(start.x + dx, 0, start.x + start.w - 0.02); w = start.w + (start.x - nx); x = nx; }
                if (mode.includes("n")) { const ny = clamp(start.y + dy, 0, start.y + start.h - 0.02); h = start.h + (start.y - ny); y = ny; }
            }
            block.rect = { x, y, w, h };
            const div = blocksLayer.children[work.blocks.indexOf(block)];
            if (div) {
                div.style.left = `${x * 100}%`;
                div.style.top = `${y * 100}%`;
                div.style.width = `${w * 100}%`;
                div.style.height = `${h * 100}%`;
                if (mode !== "move") {
                    const txt = div.querySelector(".ts-ideoe-block__text");
                    if (txt) fitText(txt, 8);  // live re-fit while resizing
                }
            }
            updateBboxReadout();
        }
        function onUp() {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            renderBlocks();
        }
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
    }

    // ── Block CRUD ──────────────────────────────────────────────────────── //
    function addBlock(type) {
        const styleFont = work.style?.font_preset_id || fontList[0]?.id || "grotesque_black";
        const block = type === "text"
            ? {
                id: makeBlockId(), type: "text", rect: { x: 0.1, y: 0.1, w: 0.5, h: 0.18 },
                text: work.language === "en" ? "TEXT" : "ТЕКСТ", font_preset_id: styleFont,
                weight: "Bold", case: "UPPERCASE", prominence: "Headline",
                legibility: { outline: true, high_contrast: true, solid_block: false },
                visual_only: false, color: "#FFFFFF", desc_override: "", color_palette: [],
            }
            : {
                id: makeBlockId(), type: "obj", rect: { x: 0.55, y: 0.2, w: 0.4, h: 0.7 },
                desc: work.language === "en" ? "subject / graphic element" : "субъект / графика", color_palette: [],
            };
        work.blocks.push(block);
        selectBlock(block.id);
        renderBlocks();
    }

    function duplicateSelected() {
        const sel = getSelected();
        if (!sel) return;
        const copy = JSON.parse(JSON.stringify(sel));
        copy.id = makeBlockId();
        copy.rect = { ...sel.rect, x: clamp(sel.rect.x + 0.03, 0, 0.9), y: clamp(sel.rect.y + 0.03, 0, 0.9) };
        work.blocks.push(copy);
        selectBlock(copy.id);
        renderBlocks();
    }

    function deleteSelected() {
        const i = work.blocks.findIndex((b) => b.id === selectedId);
        if (i < 0) return;
        work.blocks.splice(i, 1);
        selectedId = work.blocks[Math.max(0, i - 1)]?.id || null;
        renderBlocks();
        renderInspector();
    }

    function selectBlock(id) {
        selectedId = id;
        renderBlocks();
        renderInspector();
    }

    // ── Inspector ───────────────────────────────────────────────────────── //
    let bboxReadoutEl = null;
    function updateBboxReadout() {
        const sel = getSelected();
        if (bboxReadoutEl && sel?.rect) {
            const b = fracToBbox(sel.rect.x, sel.rect.y, sel.rect.w, sel.rect.h);
            bboxReadoutEl.textContent = `bbox [y,x,y,x] = [${b.join(", ")}]  (0–1000)`;
        }
    }

    function row(labelKey) {
        const r = el("div", "ts-ideoe-row");
        r.appendChild(el("label", null, tr(labelKey)));
        return r;
    }

    function templateCard() {
        const card = el("div", "ts-ideoe-card");
        card.appendChild(el("h3", null, tr("card_template")));
        const sel = el("select");
        const none = el("option", null, tr("layout_none")); none.value = ""; sel.appendChild(none);
        layouts.forEach((L) => {
            const o = el("option", null, localizedName(L, work.language) + (L.custom ? ` (${tr("custom_tag")})` : ""));
            o.value = L.id;
            if (L.id === work.layout_id) o.selected = true;
            sel.appendChild(o);
        });
        sel.addEventListener("change", () => {
            const L = layouts.find((x) => x.id === sel.value);
            if (!L) { work.layout_id = ""; return; }
            const inst = instantiateLayout(L, work.language);
            work.layout_id = L.id;
            work.blocks = inst.blocks;
            work.aspect_ratio = inst.aspect_ratio;
            if (inst.background) work.background = inst.background;
            if (inst.high_level_description) work.high_level_description = inst.high_level_description;
            selectedId = work.blocks[0]?.id || null;
            layoutArtboard();
            renderBlocks();
            renderInspector();
            relabelHeader();
        });
        const r = el("div", "ts-ideoe-row");
        r.append(el("label", null, tr("layout_preset")), sel);
        card.appendChild(r);
        const cur = layouts.find((x) => x.id === work.layout_id);
        if (cur) card.appendChild(el("div", "ts-ideoe-hint", localizedDesc(cur, work.language)));
        const saveBtnL = el("button", "ts-ideoe-btn ghost small", tr("save_as_layout"));
        saveBtnL.addEventListener("click", () => saveCustomPreset("layout"));
        const expBtnL = el("button", "ts-ideoe-btn ghost small", tr("export_btn"));
        expBtnL.addEventListener("click", () => exportPreset("layout"));
        const impBtnL = el("button", "ts-ideoe-btn ghost small", tr("import_btn"));
        impBtnL.addEventListener("click", () => importPresets("layout"));
        const footL = el("div", "ts-ideoe-btnrow");
        footL.append(saveBtnL, expBtnL, impBtnL);
        card.appendChild(footL);
        return card;
    }

    function styleCard() {
        const card = el("div", "ts-ideoe-card");
        card.appendChild(el("h3", null, tr("card_style")));

        const styleRow = el("div", "ts-ideoe-row");
        const sel = el("select");
        const none = el("option", null, tr("style_none")); none.value = ""; sel.appendChild(none);
        styles.forEach((s) => {
            const o = el("option", null, localizedName(s, work.language) + (s.custom ? ` (${tr("custom_tag")})` : ""));
            o.value = s.id;
            if (s.id === work.style.preset_id) o.selected = true;
            sel.appendChild(o);
        });
        sel.addEventListener("change", () => {
            const s = styles.find((x) => x.id === sel.value);
            work.style = s ? applyStyle(s) : { ...work.style, preset_id: "" };
            renderInspector();
            renderBlocks();
        });
        styleRow.append(el("label", null, tr("style_preset")), sel);
        card.appendChild(styleRow);
        const curS = styles.find((x) => x.id === work.style.preset_id);
        if (curS) card.appendChild(el("div", "ts-ideoe-hint", localizedDesc(curS, work.language)));

        const hldRow = row("hld");
        const hld = el("textarea");
        hld.value = work.high_level_description || "";
        hld.addEventListener("input", () => { work.high_level_description = hld.value; });
        hldRow.appendChild(hld);
        card.appendChild(hldRow);

        const medRow = row("medium");
        const med = el("select");
        MEDIA_OPTIONS.forEach((m) => {
            const o = el("option", null, m); o.value = m;
            if (m === work.style.medium) o.selected = true;
            med.appendChild(o);
        });
        med.addEventListener("change", () => { work.style.medium = med.value; renderInspector(); });
        medRow.appendChild(med);
        card.appendChild(medRow);
        card.appendChild(el("div", "ts-ideoe-hint", tr("medium_hint")));

        const isPhoto = work.style.medium === PHOTO_MEDIUM;
        [["aesthetics", "aesthetics"], ["lighting", "lighting"],
         isPhoto ? ["photo", "photo_label"] : ["art_style", "art_style"]].forEach(([key, lblKey]) => {
            const rr = row(lblKey);
            const inp = el("textarea");
            inp.value = work.style[key] || "";
            inp.addEventListener("input", () => { work.style[key] = inp.value; });
            rr.appendChild(inp);
            card.appendChild(rr);
        });

        const palRow = row2("image_palette", { n: IMAGE_PALETTE_CAP });
        palRow.appendChild(buildPalette(() => work.style.color_palette, (n) => { work.style.color_palette = n; renderBlocks(); }, IMAGE_PALETTE_CAP, work.language));
        card.appendChild(palRow);

        const bgRow = row("background");
        const bg = el("textarea");
        bg.value = work.background || "";
        bg.addEventListener("input", () => { work.background = bg.value; });
        bgRow.appendChild(bg);
        card.appendChild(bgRow);

        const saveBtnS = el("button", "ts-ideoe-btn ghost small", tr("save_as_style"));
        saveBtnS.addEventListener("click", () => saveCustomPreset("style"));
        const expBtnS = el("button", "ts-ideoe-btn ghost small", tr("export_btn"));
        expBtnS.addEventListener("click", () => exportPreset("style"));
        const impBtnS = el("button", "ts-ideoe-btn ghost small", tr("import_btn"));
        impBtnS.addEventListener("click", () => importPresets("style"));
        const footS = el("div", "ts-ideoe-btnrow");
        footS.append(saveBtnS, expBtnS, impBtnS);
        card.appendChild(footS);
        return card;
    }

    function row2(labelKey, vars) {
        const r = el("div", "ts-ideoe-row");
        r.appendChild(el("label", null, tr(labelKey, vars)));
        return r;
    }

    function blockCard() {
        const sel = getSelected();
        if (!sel) return el("div", "ts-ideoe-empty", tr("select_block"));
        const card = el("div", "ts-ideoe-card");
        card.appendChild(el("h3", null, sel.type === "obj" ? tr("block_obj_title") : tr("block_text_title")));

        const bboxRow = el("div", "ts-ideoe-bbox");
        bboxReadoutEl = bboxRow;
        card.appendChild(bboxRow);
        updateBboxReadout();

        if (sel.type === "obj") {
            const dRow = row("obj_desc");
            const ta = el("textarea");
            ta.value = sel.desc || "";
            ta.addEventListener("input", () => { sel.desc = ta.value; renderBlocks(); });
            dRow.appendChild(ta);
            card.appendChild(dRow);
            const palRow = row2("block_palette", { n: ELEMENT_PALETTE_CAP });
            palRow.appendChild(buildPalette(() => sel.color_palette, (n) => { sel.color_palette = n; renderBlocks(); }, ELEMENT_PALETTE_CAP, work.language));
            card.appendChild(palRow);
            return card;
        }

        const textRow = row("text_literal");
        const ta = el("textarea");
        ta.value = sel.text || "";
        ta.addEventListener("input", () => { sel.text = ta.value; renderBlocks(); renderWarnings(); });
        textRow.appendChild(ta);
        card.appendChild(textRow);

        const fontRow = row("font_preset");
        const fontSel = el("select");
        const isCyr = /[Ѐ-ӿ]/.test(sel.text || "");
        const ordered = fontList.slice().sort((a, b) =>
            isCyr && a.good_for_cyrillic !== b.good_for_cyrillic ? (a.good_for_cyrillic ? -1 : 1) : 0);
        ordered.forEach((f) => {
            const o = el("option", null, (localizedName(f, work.language)) + (f.good_for_cyrillic ? "" : " ⚠"));
            o.value = f.id;
            if (f.id === sel.font_preset_id) o.selected = true;
            fontSel.appendChild(o);
        });
        fontSel.addEventListener("change", () => { sel.font_preset_id = fontSel.value; renderBlocks(); renderDescPreview(); renderWarnings(); });
        fontRow.appendChild(fontSel);
        card.appendChild(fontRow);

        const segRow = row("weight");
        segRow.appendChild(buildSegmented(WEIGHTS, () => sel.weight, (v) => { sel.weight = v; renderBlocks(); renderDescPreview(); }, (o) => segLabel("weight", o, work.language)));
        card.appendChild(segRow);

        const caseRow = row("case");
        caseRow.appendChild(buildSegmented(CASES, () => sel.case, (v) => { sel.case = v; renderBlocks(); renderDescPreview(); renderWarnings(); }, (o) => segLabel("case", o, work.language)));
        card.appendChild(caseRow);

        const promRow = row("size_words");
        promRow.appendChild(buildSegmented(PROMINENCE, () => sel.prominence, (v) => { sel.prominence = v; renderBlocks(); renderDescPreview(); }, (o) => segLabel("prominence", o, work.language)));
        card.appendChild(promRow);

        const colorRow = row("text_color");
        const color = el("input"); color.type = "color"; color.value = normHex(sel.color) || "#FFFFFF";
        color.addEventListener("input", () => { sel.color = color.value.toUpperCase(); renderBlocks(); renderDescPreview(); });
        colorRow.appendChild(color);
        card.appendChild(colorRow);

        const legRow = row("legibility");
        const checks = el("div", "ts-ideoe-checks");
        sel.legibility = sel.legibility || {};
        [["outline", "leg_outline"], ["high_contrast", "leg_contrast"], ["solid_block", "leg_block"]].forEach(([key, lblKey]) => {
            const c = el("label", "ts-ideoe-check");
            const cb = el("input"); cb.type = "checkbox"; cb.checked = !!sel.legibility[key];
            cb.addEventListener("change", () => { sel.legibility[key] = cb.checked; renderBlocks(); renderDescPreview(); });
            c.append(cb, document.createTextNode(tr(lblKey)));
            checks.appendChild(c);
        });
        legRow.appendChild(checks);
        card.appendChild(legRow);

        const voRow = el("label", "ts-ideoe-check");
        const vo = el("input"); vo.type = "checkbox"; vo.checked = !!sel.visual_only;
        vo.addEventListener("change", () => { sel.visual_only = vo.checked; renderBlocks(); renderDescPreview(); renderWarnings(); });
        voRow.append(vo, document.createTextNode(tr("visual_only")));
        card.appendChild(voRow);

        const ovRow = row("override");
        const ov = el("input"); ov.type = "text"; ov.value = sel.desc_override || "";
        ov.addEventListener("input", () => { sel.desc_override = ov.value; renderDescPreview(); });
        ovRow.appendChild(ov);
        card.appendChild(ovRow);

        const palRow = row2("block_palette", { n: ELEMENT_PALETTE_CAP });
        palRow.appendChild(buildPalette(() => sel.color_palette, (n) => { sel.color_palette = n; renderBlocks(); }, ELEMENT_PALETTE_CAP, work.language));
        card.appendChild(palRow);

        const prevRow = row("desc_preview");
        const prev = el("div", "ts-ideoe-descprev");
        prevRow.appendChild(prev);
        card.appendChild(prevRow);

        const warns = el("div", "ts-ideoe-warns");
        card.appendChild(warns);

        function renderDescPreview() {
            prev.textContent = sel.visual_only ? tr("visual_only_preview") : composeTextDesc(sel, fontMap);
        }
        function renderWarnings() {
            warns.innerHTML = "";  // Cyrillic warnings removed.
        }
        card._renderDescPreview = renderDescPreview;
        card._renderWarnings = renderWarnings;
        renderDescPreview();
        renderWarnings();
        return card;
    }

    let currentBlockCard = null;
    function renderDescPreview() { currentBlockCard?._renderDescPreview?.(); }
    function renderWarnings() { currentBlockCard?._renderWarnings?.(); updateBanner(); }
    function updateBanner() {
        banner.style.display = "none";  // Cyrillic warnings removed (per user: Cyrillic works fine).
    }

    function renderInspector() {
        inspectorScroll.innerHTML = "";
        inspectorScroll.appendChild(templateCard());
        inspectorScroll.appendChild(styleCard());
        currentBlockCard = blockCard();
        inspectorScroll.appendChild(currentBlockCard);
        updateBanner();
    }

    // ── Language ────────────────────────────────────────────────────────── //
    function setLanguage(lang) {
        if (!LANGS.includes(lang) || lang === work.language) return;
        work.language = lang;
        relabelHeader();
        renderInspector();
        renderBlocks();
    }

    // ── Custom presets: save / export / import ──────────────────────────── //
    function buildLayoutPreset(name, id) {
        return {
            id, name_en: name, name_ru: name, desc_en: "", desc_ru: "", custom: true,
            aspect_ratio: work.aspect_ratio,
            background_en: work.background || "",
            high_level_description_en: work.high_level_description || "",
            blocks: work.blocks.map((b) => b.type === "obj"
                ? { type: "obj", rect: b.rect, desc_en: b.desc || "", role: b.role || "" }
                : { type: "text", rect: b.rect, text_en: b.text || "", text_ru: b.text || "", font_preset_id: b.font_preset_id, weight: b.weight, case: b.case, prominence: b.prominence, color: b.color, role: b.role || "" }),
        };
    }
    function buildStylePreset(name, id) {
        return {
            id, name_en: name, name_ru: name, desc_en: "", desc_ru: "", custom: true,
            medium: work.style.medium, aesthetics: work.style.aesthetics || "", lighting: work.style.lighting || "",
            art_style: work.style.art_style || "", photo: work.style.photo || "",
            color_palette: work.style.color_palette || [], font_preset_id: work.style.font_preset_id || "",
        };
    }

    async function saveCustomPreset(kind) {
        const name = window.prompt(tr("preset_name_prompt"));
        if (!name || !name.trim()) return;
        const id = `user_${kind}_${Date.now().toString(36)}`;
        let preset;
        if (kind === "layout") { preset = buildLayoutPreset(name.trim(), id); layouts.push(preset); work.layout_id = id; }
        else { preset = buildStylePreset(name.trim(), id); styles.push(preset); work.style.preset_id = id; }
        try {
            await fetch(api.apiURL(`${ROUTE_BASE}/save_preset`), {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ kind, preset }),
            });
        } catch (e) { console.warn("[TS Ideogram] save_preset failed", e); }
        renderInspector();
    }

    // Export the current layout/style as a JSON file (browser save dialog).
    function downloadJson(filename, obj) {
        const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = el("a"); a.href = url; a.download = filename;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1500);
    }
    function exportPreset(kind) {
        const cur = kind === "layout"
            ? layouts.find((x) => x.id === work.layout_id)
            : styles.find((x) => x.id === work.style.preset_id);
        const name = (cur ? localizedName(cur, work.language) : "") || (kind === "layout" ? "layout" : "style");
        const id = `user_${kind}_${Date.now().toString(36)}`;
        const preset = kind === "layout" ? buildLayoutPreset(name, id) : buildStylePreset(name, id);
        const safe = (name || kind).replace(/[^\w\-]+/g, "_").slice(0, 40) || kind;
        downloadJson(`ts_ideogram_${kind}_${safe}.json`, preset);
    }

    // Import layouts/styles: pick JSON file(s), server copies them into the
    // node's user_presets/ folder, then they show up in the pickers.
    let importKind = "style";
    function importPresets(kind) { importKind = kind; importInput.value = ""; importInput.click(); }
    async function handleImportFiles(files) {
        let added = 0;
        for (const file of files) {
            let data;
            try { data = JSON.parse(await file.text()); } catch { continue; }
            const arr = Array.isArray(data) ? data
                : (importKind === "layout" && Array.isArray(data?.layouts)) ? data.layouts
                : (importKind === "style" && Array.isArray(data?.styles)) ? data.styles
                : [data];
            for (const raw of arr) {
                if (!raw || typeof raw !== "object") continue;
                try {
                    const resp = await fetch(api.apiURL(`${ROUTE_BASE}/import_preset`), {
                        method: "POST", headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ kind: importKind, preset: raw }),
                    });
                    const out = await resp.json();
                    if (out?.ok && out.preset) {
                        const list = importKind === "layout" ? layouts : styles;
                        const idx = list.findIndex((x) => x.id === out.preset.id);
                        if (idx >= 0) list[idx] = out.preset; else list.push(out.preset);
                        added += 1;
                    }
                } catch (e) { console.warn("[TS Ideogram] import failed", e); }
            }
        }
        invalidatePresetsCache();
        renderInspector();
        window.alert(added ? tr("import_done", { n: added }) : tr("import_empty"));
    }

    // ── Wiring ──────────────────────────────────────────────────────────── //
    addText.addEventListener("click", () => addBlock("text"));
    addObj.addEventListener("click", () => addBlock("obj"));
    dupBtn.addEventListener("click", duplicateSelected);
    delBtn.addEventListener("click", deleteSelected);
    aspectSel.addEventListener("change", () => { work.aspect_ratio = aspectSel.value; layoutArtboard(); renderBlocks(); updateDimsReadout(); });
    mpInput.addEventListener("input", () => {
        let mp = parseFloat(mpInput.value);
        if (!(mp > 0)) mp = DEFAULT_MEGAPIXELS;
        mp = Math.max(MIN_MEGAPIXELS, Math.min(MAX_MEGAPIXELS, mp));
        work.megapixels = mp;
        mpVal.textContent = mp.toFixed(1);
        updateDimsReadout();
    });
    clearBtn.addEventListener("click", () => {
        if (work.blocks.length && !window.confirm(tr("clear") + "?")) return;
        work.blocks = [];
        work.layout_id = "";
        selectedId = null;
        renderBlocks();
        renderInspector();
    });
    refBtn.addEventListener("click", () => fileInput.click());
    refClear.addEventListener("click", () => { work.ref = null; renderReference(); });
    fileInput.addEventListener("change", () => { uploadReference(fileInput.files?.[0]); fileInput.value = ""; });

    stageWrap.addEventListener("dragover", (e) => e.preventDefault());
    stageWrap.addEventListener("drop", (e) => {
        e.preventDefault();
        const file = Array.from(e.dataTransfer?.files || []).find((f) => f.type.startsWith("image/"));
        if (file) uploadReference(file);
    });
    function onPaste(e) {
        const item = Array.from(e.clipboardData?.items || []).find((i) => i.type.startsWith("image/"));
        const file = item?.getAsFile?.();
        if (file) uploadReference(file);
    }
    document.addEventListener("paste", onPaste);

    stageWrap.addEventListener("pointerdown", (e) => {
        if (e.target === stageWrap || e.target === stage || e.target === artboard || e.target === grid) selectBlock(null);
    });

    function close() {
        document.removeEventListener("paste", onPaste);
        document.removeEventListener("keydown", onKey);
        resizeObserver.disconnect();
        overlay.remove();
    }
    function commit() {
        if (inlineEl) inlineEl.blur();
        onSave?.(JSON.parse(JSON.stringify(work)));
        close();
    }
    cancelBtn.addEventListener("click", close);
    saveBtn.addEventListener("click", commit);
    function onKey(e) {
        if (inlineEl) return;
        if (e.key === "Escape") { e.stopPropagation(); close(); }
        else if ((e.key === "Delete" || e.key === "Backspace") && getSelected()
            && document.activeElement?.tagName !== "TEXTAREA" && document.activeElement?.tagName !== "INPUT") {
            deleteSelected();
        }
    }
    document.addEventListener("keydown", onKey);

    const resizeObserver = new ResizeObserver(() => { layoutArtboard(); renderBlocks(); });
    resizeObserver.observe(stageWrap);

    // ── Initial paint (synchronous + retry; robust to layout timing) ────── //
    function fullRender() { layoutArtboard(); renderReference(); renderBlocks(); }
    relabelHeader();
    renderInspector();
    fullRender();
    requestAnimationFrame(fullRender);
    let layoutAttempts = 0;
    (function ensureLaidOut() {
        if (stage.getBoundingClientRect().width >= 2 || layoutAttempts >= 40) return;
        layoutAttempts += 1;
        fullRender();
        setTimeout(ensureLaidOut, 50);
    })();
}
