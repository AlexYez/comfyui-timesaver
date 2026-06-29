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
    ROUTE_BASE,
    WEIGHTS,
    applyCase,
    BACKGROUND_PRESETS,
    LAYOUT_BRIEFS,
    LIGHTING_PRESETS,
    MOOD_PRESETS,
    OBJECT_PRESETS,
    aspectFitBox,
    ASPECT_RATIOS,
    clamp,
    cleanPalette,
    composeTextDesc,
    fetchCaptionPreview,
    designsList,
    saveDesignPreset,
    fetchDesignPreset,
    importDesignPreset,
    deleteDesignPreset,
    loadPresets,
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
    t,
} from "./_ideogram_shared.js";

const STYLE_ID = "ts-ideogram-editor-styles";

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
.ts-ideoe-block.is-selected{box-shadow:0 0 0 2px #ffd500, 0 0 0 4px rgba(255,213,0,.25)}
.ts-ideoe-block__label{position:absolute;left:0;top:0;max-width:100%;padding:1px 5px;font-size:11px;font-weight:600;color:#0b0e13;background:rgba(255,255,255,.82);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;border-bottom-right-radius:4px}
.ts-ideoe-block__text{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:6px;font-weight:800;line-height:1.05;text-shadow:0 1px 2px rgba(0,0,0,.7);white-space:pre-wrap;overflow:hidden;pointer-events:none}
.ts-ideoe-block__obj{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-size:12px;font-weight:600;color:rgba(255,255,255,.85);overflow:hidden;pointer-events:none}
.ts-ideoe-handle{position:absolute;width:11px;height:11px;background:#ffd500;border:1px solid #1c1c1c;border-radius:2px;z-index:6}
.ts-ideoe-handle.nw{left:-6px;top:-6px;cursor:nwse-resize}.ts-ideoe-handle.ne{right:-6px;top:-6px;cursor:nesw-resize}
.ts-ideoe-handle.sw{left:-6px;bottom:-6px;cursor:nesw-resize}.ts-ideoe-handle.se{right:-6px;bottom:-6px;cursor:nwse-resize}
.ts-ideoe-handle.n{left:50%;top:-6px;transform:translateX(-50%);cursor:ns-resize}.ts-ideoe-handle.s{left:50%;bottom:-6px;transform:translateX(-50%);cursor:ns-resize}
.ts-ideoe-handle.w{left:-6px;top:50%;transform:translateY(-50%);cursor:ew-resize}.ts-ideoe-handle.e{right:-6px;top:50%;transform:translateY(-50%);cursor:ew-resize}
.ts-ideoe textarea.ts-ideoe-inline{position:absolute;z-index:500;width:auto;box-sizing:border-box;border:2px solid #ffd500;border-radius:3px;background:#0c1016;color:#fff;caret-color:#ffd500;font:700 14px/1.2 "Segoe UI",sans-serif;padding:6px 8px;resize:none;outline:none;text-align:center;box-shadow:0 4px 18px rgba(0,0,0,.6)}
.ts-ideoe-inspector{flex:0 0 340px;display:flex;flex-direction:column;background:#0c1118;min-height:0;min-width:0}
.ts-ideoe-inspector__head{display:flex;align-items:center;padding:9px 12px;border-bottom:1px solid #1c2430;background:#10151d;flex:0 0 auto}
.ts-ideoe-inspector__title{font-weight:700;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#cdd6e6}
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
.ts-ideoe-tip{position:fixed;z-index:12000;max-width:300px;background:#0b1119;color:#e9eef6;border:1px solid #2a3950;border-radius:8px;padding:7px 10px;font-size:12px;line-height:1.45;box-shadow:0 8px 26px rgba(0,0,0,.6);pointer-events:none;opacity:0;transition:opacity .12s ease;white-space:normal}
.ts-ideoe-empty{color:#6b7688;font-size:12px;text-align:center;padding:24px 8px}
.ts-ideoe-layers{flex:0 0 252px;display:flex;flex-direction:column;background:#0c1118;min-height:0;min-width:0}
.ts-ideoe-blockpanel{flex:1 1 auto;min-height:0;overflow-y:auto;padding:10px;display:flex;flex-direction:column;gap:10px}
.ts-ideoe-resizer{flex:0 0 6px;cursor:col-resize;background:transparent;position:relative;z-index:6;align-self:stretch}
.ts-ideoe-resizer::after{content:"";position:absolute;left:2px;top:0;bottom:0;width:2px;background:#1c2430;transition:background .1s}
.ts-ideoe-resizer:hover::after,.ts-ideoe-resizer.is-drag::after{background:#4da3ff}
.ts-ideoe-vresizer{flex:0 0 7px;cursor:row-resize;background:transparent;position:relative}
.ts-ideoe-vresizer::after{content:"";position:absolute;top:2px;left:6px;right:6px;height:3px;border-radius:2px;background:#2b3850;transition:background .1s}
.ts-ideoe-vresizer:hover::after,.ts-ideoe-vresizer.is-drag::after{background:#4da3ff}
.ts-ideoe-jsonhead{display:flex;align-items:center;justify-content:space-between;gap:8px}
.ts-ideoe-json{margin:0;max-height:300px;overflow:auto;background:#080c12;border:1px solid #1c2733;border-radius:6px;padding:8px 10px;font-family:Consolas,'SF Mono','Courier New',monospace;font-size:11px;line-height:1.5;color:#8893a7;white-space:pre;tab-size:2}
.ts-ideoe-json .tsj-key{color:#7dd3fc}
.ts-ideoe-json .tsj-str{color:#a6e3a1}
.ts-ideoe-json .tsj-num{color:#fab387}
.ts-ideoe-json .tsj-bool{color:#cba6f7}
.ts-ideoe-json .tsj-null{color:#6b7688}
.ts-ideoe-copybtn{padding:4px 10px;font-size:11px}
.ts-ideoe-copybtn.ok{background:#1f7a4d;border-color:#1f7a4d;color:#eafff3}
.ts-ideoe-layers__head{display:flex;align-items:center;gap:7px;padding:9px 11px;border-bottom:1px solid #1c2430;background:#10151d;flex:0 0 auto}
.ts-ideoe-layers__title{font-weight:700;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#cdd6e6}
.ts-ideoe-layers__count{font-size:10px;color:#7d899b;background:rgba(255,255,255,.06);border-radius:8px;padding:1px 7px;font-variant-numeric:tabular-nums}
.ts-ideoe-layers__list{flex:0 0 auto;min-height:56px;overflow-y:auto;padding:6px;display:flex;flex-direction:column;gap:4px}
.ts-ideoe-layers__foot{flex:0 0 auto;display:grid;grid-template-columns:1fr 1fr;gap:5px;padding:8px;border-top:1px solid #1c2430;background:#0c1016}
.ts-ideoe-layers__foot .ts-ideoe-btn{justify-content:center;padding:6px 4px}
.ts-ideoe-lrow{display:flex;align-items:center;gap:7px;padding:5px 6px;border:1px solid #1d2532;border-radius:8px;background:#0f151d;cursor:grab;position:relative;touch-action:none}
.ts-ideoe-lrow:hover{border-color:#2b3850;background:#121a24}
.ts-ideoe-lrow.is-selected{border-color:#ffd500;background:#171a16;box-shadow:0 0 0 1px rgba(255,213,0,.35)}
.ts-ideoe-lrow.is-placeholder{opacity:.45;background:#0a0e14;border-style:dashed;border-color:#4da3ff}
.ts-ideoe-lrow.is-placeholder>*{visibility:hidden}
.ts-ideoe-ldrag{position:fixed;z-index:13000;pointer-events:none;margin:0;border-color:#4da3ff;background:#17304d;box-shadow:0 12px 30px rgba(0,0,0,.6);transform:scale(1.04);opacity:.97;cursor:grabbing}
.ts-ideoe-lchip{flex:0 0 auto;width:24px;height:24px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#0b0e13;box-shadow:inset 0 0 0 1px rgba(0,0,0,.2)}
.ts-ideoe-lbody{flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:1px;pointer-events:none}
.ts-ideoe-lname{font-size:12px;color:#e9eef6;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:600;line-height:1.25}
.ts-ideoe-lmeta{font-size:10px;color:#7d899b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.2}
.ts-ideoe-lacts{position:absolute;right:3px;top:50%;transform:translateY(-50%);display:flex;gap:1px;opacity:0;transition:opacity .1s ease;background:rgba(10,14,20,.92);border:1px solid #232d3b;border-radius:6px;padding:2px;pointer-events:none}
.ts-ideoe-lrow:hover .ts-ideoe-lacts,.ts-ideoe-lrow.is-selected .ts-ideoe-lacts{opacity:1;pointer-events:auto}
.ts-ideoe-lact{width:19px;height:19px;display:flex;align-items:center;justify-content:center;border:0;background:transparent;color:#9aa6b8;border-radius:4px;cursor:pointer;font-size:10px;line-height:1;padding:0}
.ts-ideoe-lact:hover{background:#26303f;color:#fff}
.ts-ideoe-lact.danger:hover{background:#5a2626;color:#ffb4b1}
`;
    document.head.appendChild(style);
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
}

// Attach a localized hover tooltip: store the i18n KEY (resolved to the current
// language at show time, so tooltips follow the RU/EN switch for free).
function tip(node, key, vars) {
    if (node && key) {
        node.dataset.tip = key;
        if (vars && vars.n != null) node.dataset.tipN = String(vars.n);
    }
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
            tip(sw, "tip_palette_swatch");
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
            tip(add, "tip_add_color");
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
    const stylePresets = presets?.styles || [];
    const MOOD_BY_ID = Object.fromEntries(MOOD_PRESETS.map((m) => [m.id, m]));
    const LIGHT_BY_ID = Object.fromEntries(LIGHTING_PRESETS.map((m) => [m.id, m]));
    const BG_BY_ID = Object.fromEntries(BACKGROUND_PRESETS.map((m) => [m.id, m]));
    let designs = designsList(presets);
    const fontList = presets?.fonts || [];
    const fontMap = buildFontsById(presets);
    const tr = (key, vars) => t(key, work.language, vars);

    let selectedId = work.blocks[0]?.id || null;
    let lastClickInfo = { id: null, t: 0 };
    // Block ids whose object the user set explicitly via the object-preset
    // dropdown — a later brief change must NOT clobber these (hierarchy:
    // explicit object preset > brief subject > layout default).
    const userSetSubjects = new Set();
    // Teardown for an in-progress pointer drag (layer reorder / panel resize), so
    // closing the editor mid-drag can't leak window listeners or a floating clone.
    let activeDragCleanup = null;

    const overlay = el("div", "ts-ideoe-overlay ts-ideoe");
    const shell = el("div", "ts-ideoe-shell");

    // ── Header ──────────────────────────────────────────────────────────── //
    const header = el("div", "ts-ideoe-header");
    const title = el("div", "ts-ideoe-title", "TS Ideogram Designer");

    const addText = el("button", "ts-ideoe-btn");
    const addObj = el("button", "ts-ideoe-btn");
    // Duplicate/Delete are icon buttons in the narrow layers footer (text labels
    // would force the panel wider); their meaning lives in the hover tooltips.
    const dupBtn = el("button", "ts-ideoe-btn", "⧉");
    const delBtn = el("button", "ts-ideoe-btn danger", "✕");
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

    // Block CRUD (+Text/+Object/Duplicate/Delete) lives in the layers panel
    // footer now; the header keeps only document-level controls.
    header.append(title, el("div", "ts-ideoe-spacer"),
        langSeg, aspectSel, mpWrap, refBtn, refClear, clearBtn, cancelBtn, saveBtn);

    // Hover tooltips for every toolbar control (localized at show time).
    tip(addText, "tip_add_text"); tip(addObj, "tip_add_obj"); tip(dupBtn, "tip_duplicate");
    tip(delBtn, "tip_delete"); tip(clearBtn, "tip_clear"); tip(aspectSel, "tip_aspect");
    tip(mpInput, "tip_megapixels"); tip(refBtn, "tip_reference"); tip(refClear, "tip_clear_ref");
    tip(cancelBtn, "tip_cancel"); tip(saveBtn, "tip_save");
    langButtons.forEach(([, b]) => tip(b, "tip_language"));

    function relabelHeader() {
        addText.textContent = tr("add_text");
        addObj.textContent = tr("add_obj");
        layersTitle.textContent = tr("layers_title");
        inspectorTitle.textContent = tr("general_settings");
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

    // Layers panel (left): the block stack, frontmost at the top — mirrors a
    // Photoshop layers list. Selection + drag-reorder drive the canvas z-order
    // (which is simply the work.blocks array order).
    const layersPanel = el("div", "ts-ideoe-layers");
    const layersHead = el("div", "ts-ideoe-layers__head");
    const layersTitle = el("div", "ts-ideoe-layers__title", "Layers");
    const layersCount = el("div", "ts-ideoe-layers__count", "0");
    tip(layersHead, "tip_layers");
    layersHead.append(layersTitle, layersCount);
    const layersList = el("div", "ts-ideoe-layers__list");
    const layersFoot = el("div", "ts-ideoe-layers__foot");
    layersFoot.append(addText, addObj, dupBtn, delBtn);
    // Selected-block settings live UNDER the layers list (same left column) so
    // switching blocks never means scrolling the right inspector.
    const blockPanel = el("div", "ts-ideoe-blockpanel");
    layersPanel.append(layersHead, layersList, layersFoot, blockPanel);

    const inspector = el("div", "ts-ideoe-inspector");
    const inspectorHead = el("div", "ts-ideoe-inspector__head");
    const inspectorTitle = el("div", "ts-ideoe-inspector__title", "General settings");
    inspectorHead.append(inspectorTitle);
    const banner = el("div", "ts-ideoe-banner");
    banner.style.display = "none";
    const inspectorScroll = el("div", "ts-ideoe-inspector__scroll");
    inspector.append(inspectorHead, banner, inspectorScroll);

    // Live JSON-prompt panel (bottom of the general-settings column): polls the
    // server-authoritative caption builder (/ts_ideogram/preview, no JS/Python
    // drift) and shows it pretty-printed with a copy button. Built once and
    // re-appended on every renderInspector so its <pre> ref stays stable.
    const jsonCardEl = el("div", "ts-ideoe-card");
    const jsonHead = el("div", "ts-ideoe-jsonhead");
    const jsonTitle = el("h3", null, "JSON");
    const copyBtn = el("button", "ts-ideoe-btn ghost small ts-ideoe-copybtn");
    tip(copyBtn, "tip_copy_json");
    jsonHead.append(jsonTitle, copyBtn);
    const jsonPre = el("pre", "ts-ideoe-json", "—");
    jsonCardEl.append(jsonHead, jsonPre);
    function relabelJson() {
        jsonTitle.textContent = tr("json_prompt");
        copyBtn.classList.remove("ok");
        copyBtn.textContent = tr("copy");
    }
    function prettyJson(s) { try { return JSON.stringify(JSON.parse(s), null, 2); } catch { return s; } }
    // Color-highlight a JSON string into SAFE html: keys / strings / numbers /
    // booleans / null each get a token class; punctuation keeps the base color.
    function escapeHtml(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
    function highlightJson(s) {
        return escapeHtml(s).replace(
            /("(?:\\.|[^"\\])*")(\s*:)?|(\b-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?\b)|(\btrue\b|\bfalse\b)|(\bnull\b)/g,
            (m, str, colon, num, bool, nul) => {
                if (str != null) return `<span class="tsj-${colon ? "key" : "str"}">${str}</span>${colon || ""}`;
                if (num != null) return `<span class="tsj-num">${num}</span>`;
                if (bool != null) return `<span class="tsj-bool">${bool}</span>`;
                if (nul != null) return `<span class="tsj-null">${nul}</span>`;
                return m;
            },
        );
    }
    let lastJsonSig = null;
    let jsonReqId = 0;
    async function refreshJson(force) {
        const dj = JSON.stringify(work);
        if (!force && dj === lastJsonSig) return;  // only re-fetch when the design changed
        const myReq = ++jsonReqId;
        const res = await fetchCaptionPreview(dj);
        if (myReq !== jsonReqId) return;  // a newer request superseded this one — drop the stale answer
        if (!res) return;  // transient failure: keep the last good panel + signature so the next tick retries
        lastJsonSig = dj;
        const text = res.json_prompt ? prettyJson(res.json_prompt) : "";
        if (text) jsonPre.innerHTML = highlightJson(text); else jsonPre.textContent = "—";
    }
    copyBtn.addEventListener("click", async () => {
        const text = jsonPre.textContent || "";
        let ok = false;
        try { await navigator.clipboard.writeText(text); ok = true; }
        catch {
            try {
                const ta = el("textarea"); ta.value = text;
                ta.style.position = "fixed"; ta.style.opacity = "0";
                document.body.appendChild(ta); ta.select();
                ok = document.execCommand("copy"); ta.remove();
            } catch { ok = false; }
        }
        if (ok) { copyBtn.classList.add("ok"); copyBtn.textContent = tr("copied"); setTimeout(relabelJson, 1200); }
    });
    const jsonTimer = setInterval(() => refreshJson(false), 500);

    // Resizable, fluid panels: drag the dividers to set the layers/inspector
    // widths (and the vertical divider for the layers-list height); the stage
    // flexes to fill the rest. All sizes persist per-user via readPanelSize.
    // The left column now holds the layers list AND the selected block's
    // settings, so it needs more room than the layers-only version (new key
    // invalidates the old narrow saved width).
    const LAYERS_W = { def: 252, min: 200, max: 400, key: "ts.ideoe.leftW" };
    const INSPECTOR_W = { def: 340, min: 268, max: 520, key: "ts.ideoe.inspectorW" };
    function readPanelSize(cfg) {
        try { const v = parseInt(localStorage.getItem(cfg.key), 10); if (Number.isFinite(v)) return clamp(v, cfg.min, cfg.max); } catch { /* ignore */ }
        return cfg.def;
    }
    function savePanelSize(cfg, w) { try { localStorage.setItem(cfg.key, String(Math.round(w))); } catch { /* ignore */ } }
    layersPanel.style.flex = `0 0 ${readPanelSize(LAYERS_W)}px`;
    inspector.style.flex = `0 0 ${readPanelSize(INSPECTOR_W)}px`;
    function makeResizer(panel, cfg, grow) {
        // grow = +1 when dragging right enlarges the panel (a LEFT-side panel),
        // -1 when dragging right shrinks it (a RIGHT-side panel).
        const divEl = el("div", "ts-ideoe-resizer");
        divEl.addEventListener("pointerdown", (ev) => {
            ev.preventDefault();
            const startX = ev.clientX;
            const startW = panel.getBoundingClientRect().width;
            divEl.classList.add("is-drag");
            function onMove(e) {
                const w = clamp(startW + grow * (e.clientX - startX), cfg.min, cfg.max);
                panel.style.flex = `0 0 ${w}px`;
                layoutArtboard();
                renderBlocks();
            }
            function onUp() {
                activeDragCleanup = null;
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
                divEl.classList.remove("is-drag");
                savePanelSize(cfg, panel.getBoundingClientRect().width);
            }
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
            activeDragCleanup = onUp;
        });
        return divEl;
    }
    // Vertical divider between the layers list and the block settings (same
    // column): drag to resize the layers area; the height persists.
    const LAYERS_LIST_H = { def: 200, min: 80, max: 640, key: "ts.ideoe.listH" };
    layersList.style.height = `${readPanelSize(LAYERS_LIST_H)}px`;
    function makeVResizer(targetEl, cfg) {
        const divEl = el("div", "ts-ideoe-vresizer");
        divEl.addEventListener("pointerdown", (ev) => {
            ev.preventDefault();
            const startY = ev.clientY;
            const startH = targetEl.getBoundingClientRect().height;
            divEl.classList.add("is-drag");
            function onMove(e) {
                targetEl.style.height = `${clamp(startH + (e.clientY - startY), cfg.min, cfg.max)}px`;
            }
            function onUp() {
                activeDragCleanup = null;
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
                divEl.classList.remove("is-drag");
                savePanelSize(cfg, targetEl.getBoundingClientRect().height);
            }
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
            activeDragCleanup = onUp;
        });
        return divEl;
    }
    layersPanel.insertBefore(makeVResizer(layersList, LAYERS_LIST_H), blockPanel);

    // Columns: general settings on the LEFT, layers + block settings on the
    // RIGHT (the stage flexes between). The grow sign flips with the side.
    body.append(inspector, makeResizer(inspector, INSPECTOR_W, +1), stageWrap,
        makeResizer(layersPanel, LAYERS_W, -1), layersPanel);
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
    importInput.addEventListener("change", () => { handleDesignImport(Array.from(importInput.files || [])); });

    // ── Tooltips: one floating bubble, shown on hover over any [data-tip] ───── //
    // The element stores the i18n KEY; the text is resolved in the current
    // language at show time, so tooltips localize together with the rest of the UI.
    const tipEl = el("div", "ts-ideoe-tip");
    overlay.appendChild(tipEl);
    let tipTimer = null;
    let tipTarget = null;
    function positionTip(target) {
        const r = target.getBoundingClientRect();
        const tr2 = tipEl.getBoundingClientRect();
        let left = r.left + r.width / 2 - tr2.width / 2;
        let top = r.bottom + 8;
        left = Math.max(8, Math.min(left, window.innerWidth - tr2.width - 8));
        if (top + tr2.height > window.innerHeight - 8) top = Math.max(8, r.top - tr2.height - 8);
        tipEl.style.left = `${left}px`;
        tipEl.style.top = `${top}px`;
    }
    function showTip(target) {
        const key = target.dataset.tip;
        if (!key) return;
        const vars = target.dataset.tipN != null ? { n: target.dataset.tipN } : undefined;
        tipEl.textContent = t(key, work.language, vars);
        tipEl.style.opacity = "1";
        positionTip(target);
    }
    function hideTip() { tipTarget = null; clearTimeout(tipTimer); tipEl.style.opacity = "0"; }
    overlay.addEventListener("mouseover", (e) => {
        const tEl = e.target.closest?.("[data-tip]");
        if (!tEl || tEl === tipTarget) return;
        tipTarget = tEl;
        clearTimeout(tipTimer);
        tipTimer = setTimeout(() => { if (tipTarget === tEl) showTip(tEl); }, 350);
    });
    overlay.addEventListener("mouseout", (e) => {
        const tEl = e.target.closest?.("[data-tip]");
        if (tEl && tEl === tipTarget) hideTip();
    });
    overlay.addEventListener("pointerdown", hideTip, true);

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
        // The artboard tint reflects the whole-image palette AND the background
        // palette, so switching either set of colours is visible on the canvas.
        const pal = [...(work.style?.color_palette || []), ...(work.background_palette || [])];
        const css = paletteGradientCss(pal, { alpha: 1, mesh: true });
        artboard.style.background = css || "#0a0d12";
    }

    // Any style/look/colour change re-tints the canvas and refreshes the live
    // JSON prompt immediately, so every preset switch is visibly reflected.
    function onStyleChange() {
        renderBlocks();
        refreshJson(true);
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
        work.blocks.forEach((block, i) => {
            const r = block.rect || { x: 0.1, y: 0.1, w: 0.3, h: 0.2 };
            const div = el("div", "ts-ideoe-block");
            div.dataset.id = block.id;
            // Stack by array order so the layers panel reorder is visible on the
            // canvas (last in the array = frontmost). Selection no longer forces
            // a block on top — z-order must reflect the real layer order.
            div.style.zIndex = String(i + 1);
            tip(div, "tip_block_rect");
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
            let fitEl = null;  // the content element auto-fitted to the block (text OR obj)
            if (block.type === "text" && !block.visual_only) {
                const leg = block.legibility || {};
                // Plate (solid_block) — a solid color block behind the text;
                // overrides the type tint so it shows live in the editor.
                if (leg.solid_block) div.style.background = normHex(block.plate_color) || "#1A1A1A";
                textEl = el("div", "ts-ideoe-block__text");
                textEl.textContent = applyCase(block.text || "", block.case);
                textEl.style.color = normHex(block.color) || "#ffffff";
                textEl.style.fontFamily = fontFamilyForPreset(block.font_preset_id);
                textEl.style.fontWeight = weightToCss(block.weight);
                if (leg.outline) {
                    const oc = normHex(block.outline_color) || "#000000";
                    textEl.style.webkitTextStroke = `1px ${oc}`;
                    textEl.style.textShadow = `0 1px 3px ${oc}`;
                } else {
                    textEl.style.webkitTextStroke = "0";
                    textEl.style.textShadow = "0 1px 2px rgba(0,0,0,.7)";
                }
                div.appendChild(textEl);
                fitEl = textEl;
            } else if (block.type === "obj") {
                const objEl = el("div", "ts-ideoe-block__obj", block.desc || "obj");
                div.appendChild(objEl);
                fitEl = objEl;
            }
            const label = el("div", "ts-ideoe-block__label",
                block.type === "obj" ? tr("badge_obj") : (block.visual_only ? "↳" : tr("badge_text")));
            div.appendChild(label);

            if (block.id === selectedId) {
                ["nw", "ne", "sw", "se", "n", "s", "e", "w"].forEach((dir) => {
                    const h = el("div", `ts-ideoe-handle ${dir}`);
                    tip(h, "tip_resize_handle");
                    h.addEventListener("pointerdown", (ev) => startDrag(ev, block, dir));
                    div.appendChild(h);
                });
            }

            // Single click → select + drag. Manual double-click detection (the
            // native dblclick is suppressed by startDrag's preventDefault).
            div.addEventListener("pointerdown", (ev) => {
                if (ev.target.classList.contains("ts-ideoe-handle")) return;
                // Alt+drag clones the block and drags the copy (original stays put).
                if (ev.altKey) {
                    ev.preventDefault();
                    ev.stopPropagation();
                    lastClickInfo = { id: null, t: 0 };
                    const copy = JSON.parse(JSON.stringify(block));
                    copy.id = makeBlockId();
                    work.blocks.push(copy);
                    selectBlock(copy.id);
                    startDrag(ev, copy, "move");
                    return;
                }
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
            if (fitEl) fitText(fitEl);
        });
    }

    // ── Layers panel (left) ─────────────────────────────────────────────── //
    function layerName(block) {
        if (block.type === "obj") return (block.desc || "").trim() || tr("layer_untitled");
        const text = (block.text || "").split("\n")[0].trim();
        return text || tr("layer_untitled");
    }
    function layerMeta(block) {
        if (block.type === "obj") return tr("badge_obj");
        if (block.visual_only) return `${tr("badge_text")} · ↳`;
        return tr("badge_text");
    }
    function layerChip(block) {
        const chip = el("div", "ts-ideoe-lchip");
        if (block.type === "obj") { chip.textContent = "▦"; chip.style.background = "#82d6a8"; }
        else if (block.visual_only) { chip.textContent = "↳"; chip.style.background = "#9aa6b8"; }
        else { chip.textContent = "T"; chip.style.background = "#7aa2ff"; }
        return chip;
    }

    function renderLayers() {
        layersCount.textContent = String(work.blocks.length);
        layersList.innerHTML = "";
        if (!work.blocks.length) {
            layersList.appendChild(el("div", "ts-ideoe-empty", tr("layers_empty")));
            return;
        }
        // Visual top→bottom = front→back, i.e. the reverse of the array (last = front).
        for (let i = work.blocks.length - 1; i >= 0; i -= 1) {
            const block = work.blocks[i];
            const rowEl = el("div", "ts-ideoe-lrow");
            rowEl.dataset.id = block.id;
            rowEl.classList.toggle("is-selected", block.id === selectedId);
            tip(rowEl, "tip_layer_row");
            const bodyEl = el("div", "ts-ideoe-lbody");
            bodyEl.append(
                el("div", "ts-ideoe-lname", layerName(block)),
                el("div", "ts-ideoe-lmeta", layerMeta(block)),
            );
            const acts = el("div", "ts-ideoe-lacts");
            const up = tip(el("button", "ts-ideoe-lact", "▲"), "tip_layer_up");
            up.addEventListener("click", (e) => { e.stopPropagation(); moveBlock(block.id, +1); });
            const down = tip(el("button", "ts-ideoe-lact", "▼"), "tip_layer_down");
            down.addEventListener("click", (e) => { e.stopPropagation(); moveBlock(block.id, -1); });
            const del = tip(el("button", "ts-ideoe-lact danger", "✕"), "tip_delete");
            del.addEventListener("click", (e) => { e.stopPropagation(); selectedId = block.id; deleteSelected(); });
            acts.append(up, down, del);
            rowEl.append(layerChip(block), bodyEl, acts);
            rowEl.addEventListener("pointerdown", (e) => startLayerDrag(e, block, rowEl));
            layersList.appendChild(rowEl);
        }
    }

    // Reorder by one step. dir +1 = toward the front (end of array), -1 = back.
    function moveBlock(id, dir) {
        const i = work.blocks.findIndex((b) => b.id === id);
        const j = i + dir;
        if (i < 0 || j < 0 || j >= work.blocks.length) return;
        const arr = work.blocks;
        [arr[i], arr[j]] = [arr[j], arr[i]];
        renderBlocks();
        renderLayers();
    }
    function moveToFront(id) {
        const i = work.blocks.findIndex((b) => b.id === id);
        if (i < 0) return;
        work.blocks.push(work.blocks.splice(i, 1)[0]);
        renderBlocks();
        renderLayers();
    }
    function moveToBack(id) {
        const i = work.blocks.findIndex((b) => b.id === id);
        if (i < 0) return;
        work.blocks.unshift(work.blocks.splice(i, 1)[0]);
        renderBlocks();
        renderLayers();
    }

    // Pointer-based drag reorder: live-move the row in the DOM, then read the
    // resulting order back on release (top→bottom = front→back → reverse to array).
    // A sub-threshold press is treated as a plain click → select.
    function startLayerDrag(ev, block, rowEl) {
        if (ev.target.closest(".ts-ideoe-lact")) return;  // action buttons handle themselves
        const startY = ev.clientY;
        const startRect = rowEl.getBoundingClientRect();
        const grabDX = ev.clientX - startRect.left;
        const grabDY = ev.clientY - startRect.top;
        let moved = false;
        let clone = null;  // the "picked up" row that follows the cursor
        function rowAfter(clientY) {
            const rows = [...layersList.querySelectorAll(".ts-ideoe-lrow")].filter((r) => r !== rowEl);
            for (const r of rows) {
                const rect = r.getBoundingClientRect();
                if (clientY < rect.top + rect.height / 2) return r;
            }
            return null;
        }
        function onMove(e) {
            if (!moved && Math.abs(e.clientY - startY) < 4) return;
            if (!moved) {
                moved = true;
                // Lift a floating clone that tracks the cursor; the original row
                // becomes a dashed placeholder marking the live drop slot.
                clone = rowEl.cloneNode(true);
                clone.classList.add("ts-ideoe-ldrag");
                clone.classList.remove("is-selected");
                clone.style.width = `${startRect.width}px`;
                document.body.appendChild(clone);
                rowEl.classList.add("is-placeholder");
            }
            e.preventDefault();
            clone.style.left = `${e.clientX - grabDX}px`;
            clone.style.top = `${e.clientY - grabDY}px`;
            const after = rowAfter(e.clientY);
            if (after === null) layersList.appendChild(rowEl);
            else if (after !== rowEl) layersList.insertBefore(rowEl, after);
        }
        let done = false;
        function endDrag(commit) {
            if (done) return;
            done = true;
            activeDragCleanup = null;
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            window.removeEventListener("pointercancel", onCancel);
            if (clone) { clone.remove(); clone = null; }
            rowEl.classList.remove("is-placeholder");
            if (!moved) { if (commit) selectBlock(block.id); return; }
            if (!commit) { renderLayers(); return; }  // cancelled mid-drag → restore order from data
            const ids = [...layersList.querySelectorAll(".ts-ideoe-lrow")].map((r) => r.dataset.id);
            const byId = new Map(work.blocks.map((b) => [b.id, b]));
            work.blocks = ids.map((id) => byId.get(id)).filter(Boolean).reverse();
            renderBlocks();
            renderLayers();
        }
        function onUp() { endDrag(true); }
        function onCancel() { endDrag(false); }
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onCancel);
        activeDragCleanup = () => endDrag(false);
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
        // Match the on-canvas rendered font size EXACTLY so the text never changes
        // size when entering/leaving edit mode — identical for every block type.
        const innerSel = block.type === "obj" ? ".ts-ideoe-block__obj" : ".ts-ideoe-block__text";
        // block.id may come from a loaded/imported workflow (untrusted) — escape it
        // so a metacharacter id can't throw a SyntaxError inside this handler.
        let renderedEl = null;
        try { renderedEl = blocksLayer.querySelector(`[data-id="${CSS.escape(String(block.id))}"] ${innerSel}`); } catch { renderedEl = null; }
        const rendered = renderedEl ? parseFloat(getComputedStyle(renderedEl).fontSize) || 0 : 0;
        const fs = Math.round(rendered || Math.max(18, Math.min(r.h * ab.h * 0.5, 64)));
        ta.style.fontSize = `${fs}px`;
        if (block.type === "obj") ta.style.fontWeight = "600";
        // Editor box: at least the block's size, but tall/wide enough for the font,
        // and kept inside the artboard.
        const w = Math.max(120, r.w * ab.w);
        const h = Math.max(fs + 28, r.h * ab.h);
        ta.style.width = `${w}px`;
        ta.style.height = `${h}px`;
        ta.style.left = `${clamp(r.x * ab.w, 0, Math.max(0, ab.w - w))}px`;
        ta.style.top = `${clamp(r.y * ab.h, 0, Math.max(0, ab.h - h))}px`;
        artboard.appendChild(ta);
        inlineEl = ta;
        ta.focus();
        ta.select();

        function onDocDown(ev) {
            if (!inlineEl) return;
            if (ta.contains(ev.target)) return;  // clicks inside the editor are fine
            commit();
        }
        function teardown() { document.removeEventListener("pointerdown", onDocDown, true); }
        const commit = () => {
            if (!inlineEl) return;
            const v = ta.value;
            inlineEl = null;
            teardown();
            if (block.type === "obj") block.desc = v; else block.text = v;
            ta.remove();
            renderBlocks();
            renderLayers();
            renderBlockPanel();
        };
        const cancel = () => {
            if (!inlineEl) return;
            inlineEl = null;
            teardown();
            ta.remove();
        };
        // Reliable outside-click commit: a capture-phase document listener fires
        // even when the clicked area (stage/artboard/inspector) is non-focusable,
        // which the textarea's blur alone does not catch. Registered synchronously
        // — the opening pointerdown already passed document's capture phase, so it
        // won't self-trigger.
        document.addEventListener("pointerdown", onDocDown, true);
        ta.addEventListener("blur", commit);
        ta.addEventListener("keydown", (ev) => {
            ev.stopPropagation();
            if (ev.key === "Escape") { ev.preventDefault(); cancel(); }
            else if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); commit(); }
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
                    const txt = div.querySelector(".ts-ideoe-block__text, .ts-ideoe-block__obj");
                    if (txt) fitText(txt, 8);  // live re-fit while resizing (text + obj)
                }
            }
            updateBboxReadout();
        }
        function onUp() {
            activeDragCleanup = null;
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            window.removeEventListener("pointercancel", onUp);
            renderBlocks();
        }
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onUp);
        activeDragCleanup = onUp;
    }

    // ── Block CRUD ──────────────────────────────────────────────────────── //
    function addBlock(type) {
        const styleFont = work.style?.font_preset_id || fontList[0]?.id || "grotesque_black";
        const block = type === "text"
            ? {
                id: makeBlockId(), type: "text", rect: { x: 0.1, y: 0.1, w: 0.5, h: 0.18 },
                text: work.language === "en" ? "TEXT" : "ТЕКСТ", font_preset_id: styleFont,
                weight: "Bold", case: "UPPERCASE",
                legibility: { outline: true, solid_block: false },
                visual_only: false, color: "#FFFFFF", outline_color: "#000000", plate_color: "#1A1A1A", desc_override: "",
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

    // Internal block clipboard (Ctrl+C / Ctrl+V).
    let clipboardBlock = null;
    function copySelected() {
        const sel = getSelected();
        if (sel) clipboardBlock = JSON.parse(JSON.stringify(sel));
    }
    function pasteBlock() {
        if (!clipboardBlock) return;
        const copy = JSON.parse(JSON.stringify(clipboardBlock));
        copy.id = makeBlockId();
        const rc = copy.rect || { x: 0.1, y: 0.1, w: 0.4, h: 0.2 };
        copy.rect = { ...rc, x: clamp((rc.x || 0) + 0.03, 0, 0.95), y: clamp((rc.y || 0) + 0.03, 0, 0.95) };
        work.blocks.push(copy);
        selectBlock(copy.id);
    }

    function deleteSelected() {
        const i = work.blocks.findIndex((b) => b.id === selectedId);
        if (i < 0) return;
        work.blocks.splice(i, 1);
        selectedId = work.blocks[Math.max(0, i - 1)]?.id || null;
        renderBlocks();
        renderLayers();
        renderBlockPanel();
    }

    function selectBlock(id) {
        selectedId = id;
        renderBlocks();
        renderLayers();
        renderBlockPanel();
    }

    // ── Inspector ───────────────────────────────────────────────────────── //
    let bboxReadoutEl = null;
    function updateBboxReadout() {
        const sel = getSelected();
        if (bboxReadoutEl && sel?.rect) {
            const b = fracToBbox(sel.rect.x, sel.rect.y, sel.rect.w, sel.rect.h);
            bboxReadoutEl.textContent = `${tr("bbox_label")} = [${b.join(", ")}]  (0–1000)`;
        }
    }

    function row(labelKey) {
        const r = el("div", "ts-ideoe-row");
        r.appendChild(el("label", null, tr(labelKey)));
        return r;
    }

    // Top-level "Design preset" card: save / load / export / import the WHOLE
    // design (work / design_json) — the one unified import/export.
    function designCard() {
        const card = el("div", "ts-ideoe-card");
        card.appendChild(tip(el("h3", null, tr("card_design")), "tip_design_card"));
        const r = el("div", "ts-ideoe-row");
        const sel = el("select");
        tip(sel, "tip_design_load");
        const none = el("option", null, tr("design_load")); none.value = ""; sel.appendChild(none);
        designs.forEach((d) => { const o = el("option", null, d.name || d.id); o.value = d.id; sel.appendChild(o); });
        r.append(el("label", null, tr("design_saved")), sel);
        card.appendChild(r);
        const applyBtn = tip(el("button", "ts-ideoe-btn ghost small", tr("design_apply")), "tip_design_apply");
        applyBtn.addEventListener("click", async () => {
            if (!sel.value) return;
            const res = await fetchDesignPreset(sel.value);
            if (res && res.design) loadDesignIntoEditor(res.design);
        });
        const saveBtn = tip(el("button", "ts-ideoe-btn ghost small", tr("design_save")), "tip_design_save");
        saveBtn.addEventListener("click", () => saveCurrentDesign());
        const expBtn = tip(el("button", "ts-ideoe-btn ghost small", tr("export_btn")), "tip_design_export");
        expBtn.addEventListener("click", () => exportCurrentDesign());
        const impBtn = tip(el("button", "ts-ideoe-btn ghost small", tr("import_btn")), "tip_design_import");
        impBtn.addEventListener("click", () => importDesignFiles());
        const delBtn = tip(el("button", "ts-ideoe-btn danger small", "🗑"), "tip_design_delete");
        delBtn.addEventListener("click", () => { if (sel.value && window.confirm(tr("design_delete_confirm"))) deleteDesign(sel.value); });
        const foot = el("div", "ts-ideoe-btnrow");
        foot.append(applyBtn, saveBtn, expBtn, impBtn, delBtn);
        card.appendChild(foot);
        return card;
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
            userSetSubjects.clear();  // fresh layout → fresh objects; drop stale ownership
            work.aspect_ratio = inst.aspect_ratio;
            if (inst.background) work.background = inst.background;
            // Default the Main idea to this layout's first curated brief (so the
            // dropdown opens on a strong, on-theme idea); fall back to the
            // layout's own high_level_description for layouts without briefs.
            const briefs = LAYOUT_BRIEFS[L.id];
            if (briefs?.length) applyBrief(briefs[0]);  // full default scene: look + objects + texts
            else if (inst.high_level_description) work.high_level_description = inst.high_level_description;
            selectedId = work.blocks[0]?.id || null;
            layoutArtboard();
            renderBlocks();
            renderLayers();
            renderBlockPanel();
            renderInspector();
            relabelHeader();
        });
        const r = el("div", "ts-ideoe-row");
        tip(sel, "tip_layout_preset");
        r.append(el("label", null, tr("layout_preset")), sel);
        card.appendChild(r);
        const cur = layouts.find((x) => x.id === work.layout_id);
        if (cur) card.appendChild(el("div", "ts-ideoe-hint", localizedDesc(cur, work.language)));
        briefRow(card);  // Main idea — sits right after the layout it belongs to
        return card;
    }

    // A labelled field row with the control + a one-line plain-language hint
    // underneath. `hintVars` feeds {n}-style placeholders in the hint string.
    function fieldRow(card, labelKey, hintKey, control, hintVars) {
        const r = row(labelKey);
        if (hintKey) tip(control, hintKey, hintVars);  // hover tooltip mirrors the inline hint
        r.appendChild(control);
        card.appendChild(r);
        if (hintKey) card.appendChild(el("div", "ts-ideoe-hint", tr(hintKey, hintVars)));
    }

    // A "none / presets… / custom" dropdown that writes a single style field.
    // The curated presets supply model-ready prose; "Custom…" reveals a text box
    // so any value (including one loaded from an older design) is still editable.
    function presetSelectRow(card, labelKey, tipKey, presets, getVal, setVal, onPreset) {
        const r = row(labelKey);
        const sel = el("select");
        tip(sel, tipKey);
        const optNone = el("option", null, tr("opt_none")); optNone.value = "__none__"; sel.appendChild(optNone);
        presets.forEach((p) => {
            const o = el("option", null, p[work.language] || p.en); o.value = p.id; sel.appendChild(o);
        });
        const optCustom = el("option", null, tr("opt_custom")); optCustom.value = "__custom__"; sel.appendChild(optCustom);
        const custom = el("input"); custom.type = "text"; custom.placeholder = tr("opt_custom");
        custom.style.marginTop = "5px";
        const cur = getVal();
        const match = presets.find((p) => p.v === cur);
        sel.value = !cur ? "__none__" : match ? match.id : "__custom__";
        custom.value = cur || "";
        custom.style.display = sel.value === "__custom__" ? "" : "none";
        sel.addEventListener("change", () => {
            if (sel.value === "__none__") { setVal(""); custom.style.display = "none"; onPreset?.(null); }
            else if (sel.value === "__custom__") { custom.style.display = ""; setVal(custom.value || ""); custom.focus(); onPreset?.(null); }
            else { const p = presets.find((x) => x.id === sel.value); setVal(p ? p.v : ""); custom.style.display = "none"; onPreset?.(p || null); }
            onStyleChange();
        });
        custom.addEventListener("input", () => { setVal(custom.value); onPreset?.(null); onStyleChange(); });
        r.append(sel, custom);
        card.appendChild(r);
    }

    // Apply one of the curated style presets to work.style (the LOOK only).
    // Shared by the style dropdown and by ideas (which reference a style by id).
    function applyStylePresetById(id) {
        const s = stylePresets.find((x) => x.id === id);
        if (!s) return false;
        work.style.preset_id = s.id;
        work.style.medium = s.medium || "graphic_design";
        // aesthetics + lighting are owned by their own named sub-presets (set by
        // the idea or their dropdowns) — the visual style drives only the rendering.
        work.style.photo = s.photo || "";
        work.style.art_style = s.art_style || "";
        work.style.color_palette = Array.isArray(s.color_palette) ? s.color_palette.slice() : [];
        if (s.font_preset_id) work.style.font_preset_id = s.font_preset_id;
        return true;
    }

    // Apply a full "main idea": its look (style preset reference) + environment +
    // the per-role object descriptions and headline texts — so the WHOLE scene
    // (objects, background, text, style, lighting, mood) appears on the canvas.
    // Blocks the user has manually overridden are left untouched.
    function applyBrief(brief) {
        if (!brief) return;
        if (brief.v) work.high_level_description = brief.v;
        if (brief.style_preset_id) applyStylePresetById(brief.style_preset_id);
        // Named sub-presets — set each field to the preset's own text/colours so the
        // dropdowns below show a REAL selection (not "Custom"), and the background
        // carries its baked-in palette (which tints the canvas).
        const mood = MOOD_BY_ID[brief.aesthetics_id];
        if (mood) work.style.aesthetics = mood.v;
        const light = LIGHT_BY_ID[brief.lighting_id];
        if (light) work.style.lighting = light.v;
        const bg = BG_BY_ID[brief.background_id];
        if (bg) {
            work.background = bg.v;
            work.background_palette = Array.isArray(bg.colors) ? bg.colors.slice() : [];
        } else if (brief.background) {
            work.background = brief.background;  // fallback for an un-mapped idea
        }
        const objByRole = {};
        (brief.objects || []).forEach((o) => { if (o && o.role) objByRole[o.role] = o.desc || ""; });
        const txtByRole = {};
        (brief.texts || []).forEach((t) => { if (t && t.role) txtByRole[t.role] = t; });
        work.blocks.forEach((b) => {
            if (userSetSubjects.has(b.id)) return;
            if (b.type === "obj" && Object.prototype.hasOwnProperty.call(objByRole, b.role)) {
                b.desc = objByRole[b.role];
            } else if (b.type === "text" && Object.prototype.hasOwnProperty.call(txtByRole, b.role)) {
                const t = txtByRole[b.role];
                b.text = (work.language === "en" ? (t.en ?? t.ru) : (t.ru ?? t.en)) || b.text;
            }
        });
    }

    // Main idea (high_level_description). A per-layout dropdown of curated briefs
    // when the active layout has them; choosing a character brief also syncs the
    // subject object (so the preview AND the final image match the idea). Falls
    // back to a free single-line input when the layout has no briefs.
    function briefRow(card) {
        const briefs = LAYOUT_BRIEFS[work.layout_id] || [];
        if (!briefs.length) {
            const hld = el("input");
            hld.type = "text";
            hld.value = work.high_level_description || "";
            hld.addEventListener("input", () => { work.high_level_description = hld.value; });
            fieldRow(card, "hld", "hld_hint", hld);
            return;
        }
        const r = row("hld");
        const sel = el("select");
        tip(sel, "hld_hint");
        const optNone = el("option", null, tr("opt_none")); optNone.value = "__none__"; sel.appendChild(optNone);
        briefs.forEach((b, i) => { const o = el("option", null, b[work.language] || b.en); o.value = String(i); sel.appendChild(o); });
        const optCustom = el("option", null, tr("opt_custom")); optCustom.value = "__custom__"; sel.appendChild(optCustom);
        const custom = el("input"); custom.type = "text"; custom.placeholder = tr("opt_custom"); custom.style.marginTop = "5px";
        const cur = work.high_level_description || "";
        const idx = briefs.findIndex((b) => b.v === cur);
        sel.value = !cur ? "__none__" : idx >= 0 ? String(idx) : "__custom__";
        custom.value = cur;
        custom.style.display = sel.value === "__custom__" ? "" : "none";
        sel.addEventListener("change", () => {
            if (sel.value === "__none__") { work.high_level_description = ""; custom.style.display = "none"; }
            else if (sel.value === "__custom__") { custom.style.display = ""; work.high_level_description = custom.value || ""; custom.focus(); }
            else {
                // A full scene: look (style preset) + background + per-role
                // objects & texts all populate the canvas.
                applyBrief(briefs[Number(sel.value)]);
                custom.style.display = "none";
            }
            renderBlocks();
            renderLayers();
            renderBlockPanel();
            renderInspector();   // reflect the idea's applied style/background + live prompt
        });
        custom.addEventListener("input", () => { work.high_level_description = custom.value; });
        r.append(sel, custom);
        card.appendChild(r);
    }

    // Merged "Style" card — everything about how the image LOOKS: the main idea,
    // then curated preset dropdowns (art style, mood, lighting, background) and
    // the image palette. Placement lives in the Layout card above; together they
    // are the design's two components.
    function styleCard() {
        const card = el("div", "ts-ideoe-card");
        card.appendChild(el("h3", null, tr("card_style")));

        stylePresetRow(card);  // "Visual style" — the 18 curated looks (merged with art-style)
        presetSelectRow(card, "aesthetics", "aesthetics_hint", MOOD_PRESETS,
            () => work.style.aesthetics, (v) => { work.style.aesthetics = v; });
        presetSelectRow(card, "lighting", "lighting_hint", LIGHTING_PRESETS,
            () => work.style.lighting, (v) => { work.style.lighting = v; });
        paletteRow(card, "lighting_colors", "lighting_colors_hint", ELEMENT_PALETTE_CAP,
            () => work.style.lighting_palette, (n) => { work.style.lighting_palette = n; onStyleChange(); });
        presetSelectRow(card, "background", "background_hint", BACKGROUND_PRESETS,
            () => work.background, (v) => { work.background = v; },
            (p) => { work.background_palette = p && Array.isArray(p.colors) ? p.colors.slice() : []; });
        paletteRow(card, "background_colors", "background_colors_hint", ELEMENT_PALETTE_CAP,
            () => work.background_palette, (n) => { work.background_palette = n; onStyleChange(); });

        // Colors for the WHOLE image (also tints the artboard preview).
        paletteRow(card, "image_palette", "image_palette_hint", IMAGE_PALETTE_CAP,
            () => work.style.color_palette, (n) => { work.style.color_palette = n; onStyleChange(); });
        return card;
    }

    // Ready-made STYLE presets (the curated look: medium + aesthetics + lighting +
    // photo/art_style + palette + font vibe). Picking one applies the whole look
    // and re-tints the canvas immediately; the sub-controls below stay editable.
    function stylePresetRow(card) {
        if (!stylePresets.length) return;
        const r = row("visual_style");
        const sel = el("select");
        tip(sel, "tip_visual_style");
        const none = el("option", null, tr("opt_none")); none.value = ""; sel.appendChild(none);
        stylePresets.forEach((s) => {
            const o = el("option", null, localizedName(s, work.language));
            o.value = s.id;
            if (s.id === work.style.preset_id) o.selected = true;
            sel.appendChild(o);
        });
        sel.addEventListener("change", () => {
            if (!sel.value) { work.style.preset_id = ""; onStyleChange(); return; }
            applyStylePresetById(sel.value);
            renderBlocks();      // re-tint the artboard from the style's palette
            renderInspector();   // reflect the applied look in the sub-controls + live prompt
        });
        r.appendChild(sel);
        card.appendChild(r);
        card.appendChild(el("div", "ts-ideoe-hint", tr("visual_style_hint")));
    }

    // A label + palette swatches + optional hint line, for an image / background /
    // lighting color set.
    function paletteRow(card, labelKey, hintKey, cap, getArr, setArr) {
        const r = row2(labelKey, { n: cap });
        r.appendChild(buildPalette(getArr, setArr, cap, work.language));
        card.appendChild(r);
        if (hintKey) card.appendChild(el("div", "ts-ideoe-hint", tr(hintKey, { n: cap })));
    }

    function row2(labelKey, vars) {
        const r = el("div", "ts-ideoe-row");
        r.appendChild(el("label", null, tr(labelKey, vars)));
        return r;
    }

    // Object preset dropdown — one-click popular subjects (characters + hero
    // objects). Replaces the obj desc and updates the canvas + layers live.
    function objectPresetRow(card, sel, descEl) {
        const r = row("object_preset");
        const dd = el("select");
        tip(dd, "tip_object_preset");
        const optNone = el("option", null, tr("opt_none")); optNone.value = "__none__"; dd.appendChild(optNone);
        OBJECT_PRESETS.forEach((p) => { const o = el("option", null, p[work.language] || p.en); o.value = p.id; dd.appendChild(o); });
        const match = OBJECT_PRESETS.find((p) => p.v === (sel.desc || ""));
        dd.value = match ? match.id : "__none__";
        dd.addEventListener("change", () => {
            const p = OBJECT_PRESETS.find((x) => x.id === dd.value);
            if (!p) return;
            sel.desc = p.v;
            descEl.value = p.v;
            userSetSubjects.add(sel.id);  // user owns this object now — briefs won't overwrite it
            renderBlocks();
            renderLayers();
        });
        r.appendChild(dd);
        card.appendChild(r);
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
            tip(ta, "tip_obj_desc");
            ta.value = sel.desc || "";
            ta.addEventListener("input", () => { sel.desc = ta.value; renderBlocks(); renderLayers(); });
            dRow.appendChild(ta);
            objectPresetRow(card, sel, ta);  // preset dropdown above the description
            card.appendChild(dRow);
            const palRow = row2("block_palette", { n: ELEMENT_PALETTE_CAP });
            palRow.appendChild(buildPalette(() => sel.color_palette, (n) => { sel.color_palette = n; renderBlocks(); }, ELEMENT_PALETTE_CAP, work.language));
            card.appendChild(palRow);
            return card;
        }

        // "Стиль текста" — the ONE styling dropdown. Each option is a complete
        // typographic style (font_preset_id → desc_snippet, the only lever
        // Ideogram reads); it always reflects the current selection. Weight /
        // case / size / color / legibility below are separate, orthogonal knobs.
        const fontRow = row("font_preset");
        const fontSel = el("select");
        fontList.forEach((f) => {
            const o = el("option", null, localizedName(f, work.language));
            o.value = f.id;
            if (f.id === sel.font_preset_id) o.selected = true;
            fontSel.appendChild(o);
        });
        fontSel.addEventListener("change", () => { sel.font_preset_id = fontSel.value; renderBlocks(); renderDescPreview(); });
        tip(fontSel, "tip_font_preset");
        fontRow.appendChild(fontSel);
        card.appendChild(fontRow);

        const textRow = row("text_literal");
        const ta = el("textarea");
        tip(ta, "tip_text_literal");
        ta.value = sel.text || "";
        ta.addEventListener("input", () => { sel.text = ta.value; renderBlocks(); renderLayers(); renderWarnings(); });
        textRow.appendChild(ta);
        card.appendChild(textRow);

        const segRow = row("weight");
        segRow.appendChild(tip(buildSegmented(WEIGHTS, () => sel.weight, (v) => { sel.weight = v; renderBlocks(); renderDescPreview(); }, (o) => segLabel("weight", o, work.language)), "tip_weight"));
        card.appendChild(segRow);

        const caseRow = row("case");
        caseRow.appendChild(tip(buildSegmented(CASES, () => sel.case, (v) => { sel.case = v; renderBlocks(); renderDescPreview(); renderWarnings(); }, (o) => segLabel("case", o, work.language)), "tip_case"));
        card.appendChild(caseRow);

        const colorRow = row("text_color");
        const color = el("input"); color.type = "color"; color.value = normHex(sel.color) || "#FFFFFF";
        tip(color, "tip_text_color");
        color.addEventListener("input", () => { sel.color = color.value.toUpperCase(); renderBlocks(); renderDescPreview(); });
        colorRow.appendChild(color);
        card.appendChild(colorRow);

        const legRow = row("legibility");
        const checks = el("div", "ts-ideoe-checks");
        sel.legibility = sel.legibility || {};
        [["outline", "leg_outline", "tip_leg_outline"], ["solid_block", "leg_block", "tip_leg_block"]].forEach(([key, lblKey, tipKey]) => {
            const c = el("label", "ts-ideoe-check");
            tip(c, tipKey);
            const cb = el("input"); cb.type = "checkbox"; cb.checked = !!sel.legibility[key];
            // Rebuild the card so the matching color picker shows/hides, and the
            // outline/plate now render live on the canvas.
            cb.addEventListener("change", () => { sel.legibility[key] = cb.checked; renderBlocks(); renderLayers(); renderBlockPanel(); });
            c.append(cb, document.createTextNode(tr(lblKey)));
            checks.appendChild(c);
        });
        legRow.appendChild(checks);
        card.appendChild(legRow);

        // Per-effect colors — only shown when the effect is enabled.
        if (sel.legibility.outline) {
            const r = row("outline_color");
            const ci = el("input"); ci.type = "color"; ci.value = normHex(sel.outline_color) || "#000000";
            tip(ci, "tip_outline_color");
            ci.addEventListener("input", () => { sel.outline_color = ci.value.toUpperCase(); renderBlocks(); renderDescPreview(); });
            r.appendChild(ci);
            card.appendChild(r);
        }
        if (sel.legibility.solid_block) {
            const r = row("plate_color");
            const ci = el("input"); ci.type = "color"; ci.value = normHex(sel.plate_color) || "#1A1A1A";
            tip(ci, "tip_plate_color");
            ci.addEventListener("input", () => { sel.plate_color = ci.value.toUpperCase(); renderBlocks(); renderDescPreview(); });
            r.appendChild(ci);
            card.appendChild(r);
        }

        const voRow = el("label", "ts-ideoe-check");
        tip(voRow, "tip_visual_only");
        const vo = el("input"); vo.type = "checkbox"; vo.checked = !!sel.visual_only;
        vo.addEventListener("change", () => { sel.visual_only = vo.checked; renderBlocks(); renderLayers(); renderDescPreview(); renderWarnings(); });
        voRow.append(vo, document.createTextNode(tr("visual_only")));
        card.appendChild(voRow);

        const ovRow = row("override");
        const ov = el("input"); ov.type = "text"; ov.value = sel.desc_override || "";
        tip(ov, "tip_override");
        ov.addEventListener("input", () => { sel.desc_override = ov.value; renderDescPreview(); });
        ovRow.appendChild(ov);
        card.appendChild(ovRow);

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
        inspectorScroll.appendChild(designCard());
        inspectorScroll.appendChild(templateCard());
        inspectorScroll.appendChild(styleCard());
        relabelJson();
        inspectorScroll.appendChild(jsonCardEl);  // live JSON prompt at the bottom
        refreshJson(true);
        updateBanner();
    }

    // The selected block's settings render in the LEFT column, under the layers
    // list — so switching blocks never scrolls the general-settings inspector.
    function renderBlockPanel() {
        blockPanel.innerHTML = "";
        currentBlockCard = blockCard();
        blockPanel.appendChild(currentBlockCard);
    }

    // ── Language ────────────────────────────────────────────────────────── //
    function setLanguage(lang) {
        if (!LANGS.includes(lang) || lang === work.language) return;
        work.language = lang;
        relabelHeader();
        renderInspector();
        renderLayers();
        renderBlockPanel();
        renderBlocks();
    }

    // ── Full-design presets (top level): save / load / export / import ────── //
    function downloadJson(filename, obj) {
        const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = el("a"); a.href = url; a.download = filename;
        document.body.appendChild(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1500);
    }

    // Replace the whole editor state with a loaded design (mutate `work` in place
    // so every closure keeps its reference), then repaint everything.
    function loadDesignIntoEditor(d) {
        if (!d || typeof d !== "object") return;
        const fresh = JSON.parse(JSON.stringify(d));
        Object.keys(work).forEach((k) => delete work[k]);
        Object.assign(work, fresh);
        work.blocks = Array.isArray(work.blocks) ? work.blocks : [];
        work.style = work.style || {};
        if (!LANGS.includes(work.language)) work.language = DEFAULT_LANG;
        userSetSubjects.clear();
        selectedId = work.blocks[0]?.id || null;
        relabelHeader();
        layoutArtboard();
        renderReference();
        renderBlocks();
        renderLayers();
        renderBlockPanel();
        renderInspector();
    }

    async function refreshDesigns() {
        invalidatePresetsCache();
        const p = await loadPresets();
        designs = designsList(p);
        renderInspector();
    }

    async function saveCurrentDesign() {
        const name = window.prompt(tr("design_name_prompt"));
        if (!name || !name.trim()) return;
        const res = await saveDesignPreset(name.trim(), JSON.parse(JSON.stringify(work)));
        if (res && res.ok) await refreshDesigns();
    }

    function exportCurrentDesign() {
        downloadJson(`ts_ideogram_design_${Date.now().toString(36)}.json`,
            { name: "design", version: 1, design: JSON.parse(JSON.stringify(work)) });
    }

    function importDesignFiles() { importInput.value = ""; importInput.click(); }
    async function handleDesignImport(files) {
        let added = 0;
        for (const file of files) {
            let data;
            try { data = JSON.parse(await file.text()); } catch { continue; }
            const arr = Array.isArray(data) ? data : [data];
            for (const raw of arr) {
                const out = await importDesignPreset(raw);
                if (out && out.ok) added += 1;
            }
        }
        await refreshDesigns();
        window.alert(added ? tr("import_done", { n: added }) : tr("import_empty"));
    }

    async function deleteDesign(id) {
        if (!id) return;
        const res = await deleteDesignPreset(id);
        if (res && res.ok) await refreshDesigns();
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
        const st = work.style || {};
        const hasContent = work.blocks.length || work.ref
            || (st.color_palette || []).length || work.background || work.high_level_description
            || st.aesthetics || st.lighting || st.photo || st.art_style || st.preset_id;
        if (hasContent && !window.confirm(tr("clear_confirm"))) return;
        // Wipe everything that is "content" — blocks, style, palette, background,
        // brief and the reference underlay. Keep the document settings the user
        // deliberately set: aspect ratio, megapixels and language.
        work.blocks = [];
        work.layout_id = "";
        work.background = "";
        work.high_level_description = "";
        work.ref = null;
        work.style = {
            preset_id: "", aesthetics: "", lighting: "", medium: "graphic_design",
            photo: "", art_style: "", color_palette: [], font_preset_id: "",
        };
        selectedId = null;
        renderReference();   // drops the underlay + repaints the (now empty) style preview
        renderBlocks();
        renderLayers();
        renderBlockPanel();
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
        if (inlineEl) return;  // an open inline editor commits via its own document listener
        if (e.target === stageWrap || e.target === stage || e.target === artboard || e.target === grid) selectBlock(null);
    });

    function close() {
        if (activeDragCleanup) { try { activeDragCleanup(); } catch { /* ignore */ } activeDragCleanup = null; }
        document.removeEventListener("paste", onPaste);
        document.removeEventListener("keydown", onKey);
        resizeObserver.disconnect();
        clearInterval(jsonTimer);
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
        if (e.key === "Escape") { e.stopPropagation(); close(); return; }
        // Don't hijack shortcuts while typing in a panel field — let native
        // copy/paste/delete work there.
        const tag = document.activeElement?.tagName;
        if (tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT") return;
        if (e.key === "Delete" || e.key === "Backspace") {
            if (getSelected()) { e.preventDefault(); deleteSelected(); }
            return;
        }
        const mod = e.ctrlKey || e.metaKey;
        if (!mod) return;
        // e.code is the physical key, so these work on any layout (incl. Cyrillic).
        if (e.code === "KeyC") { if (getSelected()) { copySelected(); e.preventDefault(); } }
        else if (e.code === "KeyV") { if (clipboardBlock) { pasteBlock(); e.preventDefault(); } }
        else if (e.code === "KeyD") { if (getSelected()) { duplicateSelected(); e.preventDefault(); } }
        // Photoshop-style z-order: Ctrl+]/[ one step, Ctrl+Shift+]/[ to front/back.
        else if (e.code === "BracketRight") { const s = getSelected(); if (s) { e.preventDefault(); e.shiftKey ? moveToFront(s.id) : moveBlock(s.id, +1); } }
        else if (e.code === "BracketLeft") { const s = getSelected(); if (s) { e.preventDefault(); e.shiftKey ? moveToBack(s.id) : moveBlock(s.id, -1); } }
    }
    document.addEventListener("keydown", onKey);

    const resizeObserver = new ResizeObserver(() => { layoutArtboard(); renderBlocks(); });
    resizeObserver.observe(stageWrap);

    // ── Initial paint (synchronous + retry; robust to layout timing) ────── //
    function fullRender() { layoutArtboard(); renderReference(); renderBlocks(); }
    relabelHeader();
    renderInspector();
    renderLayers();
    renderBlockPanel();
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
