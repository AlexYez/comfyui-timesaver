// Full-screen modal editor for TS_IdeogramDesigner.
//
// Mounted on document.body (a fixed overlay) so it is independent of the node
// render mode and works identically in Nodes 1.0 and Nodes 2.0 (Vue). Lets the
// user drag/resize text + object blocks on an aspect-correct artboard over an
// optional reference image, edit per-block font/style via presets, edit the
// global style/palette/background, and preview the emitted caption live.
//
// openIdeogramEditor(node, { design, presets, onSave }) — onSave(newDesign)
// is called with a fresh design object when the user clicks Save.

import { api } from "/scripts/api.js";

import {
    ASPECT_RATIOS,
    CASES,
    ELEMENT_PALETTE_CAP,
    IMAGE_PALETTE_CAP,
    PHOTO_MEDIUM,
    PROMINENCE,
    WEIGHTS,
    aspectFitBox,
    clamp,
    cleanPalette,
    composeTextDesc,
    cyrillicWarnings,
    fetchCaptionPreview,
    fontsById as buildFontsById,
    fracToBbox,
    inputViewUrl,
    makeBlockId,
    normHex,
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
.ts-ideoe-header{display:flex;align-items:center;gap:10px;padding:10px 14px;border-bottom:1px solid #1c2430;background:#10151d;flex:0 0 auto}
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
.ts-ideoe-block.is-obj{border-color:#82d6a8;background:rgba(130,214,168,.14)}
.ts-ideoe-block.is-visual{border-style:dashed;border-color:#9aa6b8;background:rgba(154,166,184,.12)}
.ts-ideoe-block.is-selected{box-shadow:0 0 0 2px #ffd500, 0 0 0 4px rgba(255,213,0,.25);z-index:5}
.ts-ideoe-block__label{position:absolute;left:0;top:0;max-width:100%;padding:1px 5px;font-size:11px;font-weight:600;color:#0b0e13;background:rgba(255,255,255,.82);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;border-bottom-right-radius:4px}
.ts-ideoe-block__text{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:6px;font-weight:800;line-height:1.05;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.7);white-space:pre-wrap;overflow:hidden;pointer-events:none}
.ts-ideoe-handle{position:absolute;width:11px;height:11px;background:#ffd500;border:1px solid #1c1c1c;border-radius:2px;z-index:6}
.ts-ideoe-handle.nw{left:-6px;top:-6px;cursor:nwse-resize}
.ts-ideoe-handle.ne{right:-6px;top:-6px;cursor:nesw-resize}
.ts-ideoe-handle.sw{left:-6px;bottom:-6px;cursor:nesw-resize}
.ts-ideoe-handle.se{right:-6px;bottom:-6px;cursor:nwse-resize}
.ts-ideoe-handle.n{left:50%;top:-6px;transform:translateX(-50%);cursor:ns-resize}
.ts-ideoe-handle.s{left:50%;bottom:-6px;transform:translateX(-50%);cursor:ns-resize}
.ts-ideoe-handle.w{left:-6px;top:50%;transform:translateY(-50%);cursor:ew-resize}
.ts-ideoe-handle.e{right:-6px;top:50%;transform:translateY(-50%);cursor:ew-resize}
.ts-ideoe-inspector{flex:0 0 340px;display:flex;flex-direction:column;border-left:1px solid #1c2430;background:#0c1118;min-height:0}
.ts-ideoe-inspector__scroll{flex:1 1 auto;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:12px}
.ts-ideoe-card{border:1px solid #1f2937;border-radius:10px;background:#0f151d;padding:10px;display:flex;flex-direction:column;gap:8px}
.ts-ideoe-card h3{margin:0;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#8a93a3}
.ts-ideoe-row{display:flex;flex-direction:column;gap:4px}
.ts-ideoe-row label{font-size:11px;color:#9aa6b8}
.ts-ideoe-banner{margin:0 12px 0;padding:8px 10px;border:1px solid #6b561f;background:rgba(255,207,107,.1);color:#ffcf6b;border-radius:8px;font-size:11px;line-height:1.4}
.ts-ideoe-warns{display:flex;flex-direction:column;gap:4px}
.ts-ideoe-warn{font-size:11px;color:#ffcf6b;background:rgba(255,207,107,.08);border:1px solid rgba(255,207,107,.25);border-radius:6px;padding:4px 7px}
.ts-ideoe input[type=text],.ts-ideoe textarea,.ts-ideoe-inspector input[type=text],.ts-ideoe-inspector textarea,.ts-ideoe-inspector select{width:100%;box-sizing:border-box;background:#0a0e14;border:1px solid #28323f;border-radius:6px;color:#e9eef6;padding:6px 8px;font-size:12px;font-family:inherit;outline:none}
.ts-ideoe-inspector textarea{resize:vertical;min-height:46px}
.ts-ideoe-inspector input:focus,.ts-ideoe-inspector textarea:focus,.ts-ideoe-inspector select:focus{border-color:#4da3ff}
.ts-ideoe-seg{display:flex;border:1px solid #28323f;border-radius:6px;overflow:hidden}
.ts-ideoe-seg button{flex:1 1 auto;background:#0a0e14;color:#9aa6b8;border:0;border-right:1px solid #1b232e;padding:5px 4px;font-size:11px;cursor:pointer}
.ts-ideoe-seg button:last-child{border-right:0}
.ts-ideoe-seg button.is-on{background:linear-gradient(180deg,#7aa2ff,#3a72ff);color:#0b1530;font-weight:700}
.ts-ideoe-checks{display:flex;flex-wrap:wrap;gap:8px}
.ts-ideoe-check{display:flex;align-items:center;gap:5px;font-size:11px;color:#cdd6e6;cursor:pointer}
.ts-ideoe-pal{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
.ts-ideoe-sw{width:20px;height:20px;border-radius:4px;border:1px solid rgba(255,255,255,.25);position:relative;cursor:pointer}
.ts-ideoe-sw:hover::after{content:"×";position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;background:rgba(0,0,0,.45);border-radius:4px}
.ts-ideoe-descprev{font-size:11px;color:#9fe3c2;background:#08120d;border:1px solid #1c3a2c;border-radius:6px;padding:6px 8px;white-space:pre-wrap;word-break:break-word}
.ts-ideoe-bbox{font-size:10px;color:#7d899b;font-variant-numeric:tabular-nums}
.ts-ideoe-btn{display:inline-flex;align-items:center;gap:5px;border:1px solid #28323f;background:#161d27;color:#e9eef6;border-radius:8px;padding:6px 11px;font-size:12px;cursor:pointer;font-weight:600;white-space:nowrap}
.ts-ideoe-btn:hover{background:#1f2937}
.ts-ideoe-btn.primary{background:linear-gradient(180deg,#46d39a,#1fa97a);border-color:#1fa97a;color:#04130d}
.ts-ideoe-btn.primary:hover{background:linear-gradient(180deg,#5ee3ab,#27c08c)}
.ts-ideoe-btn.danger{background:#3a1d1d;border-color:#6b2f2f;color:#ffb4b1}
.ts-ideoe-btn.ghost{background:transparent}
.ts-ideoe-select{background:#0a0e14;border:1px solid #28323f;border-radius:6px;color:#e9eef6;padding:6px 8px;font-size:12px}
.ts-ideoe-json{flex:0 0 auto;max-height:30%;border-top:1px solid #1c2430;background:#080b10;display:none;flex-direction:column}
.ts-ideoe-json.is-open{display:flex}
.ts-ideoe-json pre{margin:0;padding:10px 12px;overflow:auto;font-size:11px;color:#cdd6e6;font-family:ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word}
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

function buildSegmented(options, getValue, onPick) {
    const wrap = el("div", "ts-ideoe-seg");
    const buttons = [];
    options.forEach((opt) => {
        const btn = el("button", null, opt);
        btn.addEventListener("click", () => { onPick(opt); refresh(); });
        wrap.appendChild(btn);
        buttons.push([opt, btn]);
    });
    function refresh() {
        const cur = getValue();
        buttons.forEach(([opt, btn]) => btn.classList.toggle("is-on", opt === cur));
    }
    refresh();
    wrap._refresh = refresh;
    return wrap;
}

function buildPalette(getArr, setArr, cap) {
    const wrap = el("div", "ts-ideoe-pal");
    function render() {
        wrap.innerHTML = "";
        const arr = getArr() || [];
        arr.forEach((hex, i) => {
            const sw = el("div", "ts-ideoe-sw");
            sw.style.background = hex;
            sw.title = `${hex} (клик — удалить)`;
            sw.addEventListener("click", () => {
                const next = arr.slice();
                next.splice(i, 1);
                setArr(next);
                render();
            });
            wrap.appendChild(sw);
        });
        if (arr.length < cap) {
            const add = el("input");
            add.type = "color";
            add.className = "ts-ideoe-sw";
            add.style.padding = "0";
            add.title = "Добавить цвет";
            add.addEventListener("change", () => {
                const hex = normHex(add.value);
                if (hex) {
                    const next = (getArr() || []).slice();
                    if (!next.includes(hex)) next.push(hex);
                    setArr(cleanPalette(next, cap));
                }
                render();
            });
            wrap.appendChild(add);
        }
    }
    render();
    wrap._render = render;
    return wrap;
}

export function openIdeogramEditor(node, { design, presets, onSave }) {
    ensureStyles();

    // Work on a deep clone; commit only on Save.
    const work = JSON.parse(JSON.stringify(design));
    work.blocks = Array.isArray(work.blocks) ? work.blocks : [];
    const fonts = presets?.fonts || [];
    const styles = presets?.styles || [];
    const fontMap = buildFontsById(presets);

    let selectedId = work.blocks[0]?.id || null;
    let jsonOpen = false;

    const overlay = el("div", "ts-ideoe-overlay ts-ideoe");
    const shell = el("div", "ts-ideoe-shell");

    // ── Header ──────────────────────────────────────────────────────────── //
    const header = el("div", "ts-ideoe-header");
    const title = el("div", "ts-ideoe-title");
    title.innerHTML = `TS Ideogram Designer <small>визуальный дизайнер капшена</small>`;

    const addText = el("button", "ts-ideoe-btn", "+ Текст");
    const addObj = el("button", "ts-ideoe-btn", "+ Объект");
    const dupBtn = el("button", "ts-ideoe-btn", "Дублировать");
    const delBtn = el("button", "ts-ideoe-btn danger", "Удалить");
    const frontBtn = el("button", "ts-ideoe-btn ghost", "▲ слой");
    const backBtn = el("button", "ts-ideoe-btn ghost", "▼ слой");

    const aspectSel = el("select", "ts-ideoe-select");
    ASPECT_RATIOS.forEach((ar) => {
        const o = el("option", null, ar);
        o.value = ar;
        if (ar === work.aspect_ratio) o.selected = true;
        aspectSel.appendChild(o);
    });

    const refBtn = el("button", "ts-ideoe-btn", "🖼 Референс");
    const refClear = el("button", "ts-ideoe-btn ghost", "✕ реф");
    const jsonBtn = el("button", "ts-ideoe-btn", "{ } JSON");
    const cancelBtn = el("button", "ts-ideoe-btn", "Отмена");
    const saveBtn = el("button", "ts-ideoe-btn primary", "Сохранить");

    header.append(
        title, addText, addObj, dupBtn, delBtn, frontBtn, backBtn,
        el("div", "ts-ideoe-spacer"),
        aspectSel, refBtn, refClear, jsonBtn, cancelBtn, saveBtn,
    );

    // ── Body: stage + inspector ─────────────────────────────────────────── //
    const body = el("div", "ts-ideoe-body");
    const stageWrap = el("div", "ts-ideoe-stagewrap");
    const stage = el("div", "ts-ideoe-stage");
    const artboard = el("div", "ts-ideoe-artboard");
    const grid = el("div", "grid");
    let refImgEl = null;
    const blocksLayer = el("div");
    blocksLayer.style.position = "absolute";
    blocksLayer.style.inset = "0";
    artboard.append(grid, blocksLayer);
    stage.appendChild(artboard);
    stageWrap.appendChild(stage);

    const inspector = el("div", "ts-ideoe-inspector");
    const banner = el("div", "ts-ideoe-banner");
    banner.style.display = "none";
    banner.textContent = "Кириллица в Ideogram 4 менее надёжна латиницы. Для печати — генерируйте визуал и добавляйте русский текст вручную (тумблер «текст вручную»).";
    const inspectorScroll = el("div", "ts-ideoe-inspector__scroll");
    inspector.append(banner, inspectorScroll);

    body.append(stageWrap, inspector);

    // ── JSON preview panel ──────────────────────────────────────────────── //
    const jsonPanel = el("div", "ts-ideoe-json");
    const jsonPre = el("pre");
    jsonPre.textContent = "{ }";
    jsonPanel.appendChild(jsonPre);

    shell.append(header, body, jsonPanel);
    overlay.appendChild(shell);
    document.body.appendChild(overlay);

    // Hidden file input for reference upload
    const fileInput = el("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.style.display = "none";
    overlay.appendChild(fileInput);

    // ── Artboard sizing (keeps aspect; blocks use % so they scale) ──────── //
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

    // ── Reference image ─────────────────────────────────────────────────── //
    function renderReference() {
        if (refImgEl) { refImgEl.remove(); refImgEl = null; }
        const ref = work.ref;
        if (ref?.filename) {
            refImgEl = el("img", "ts-ideoe-ref");
            refImgEl.src = inputViewUrl(ref.filename, ref.subfolder, ref.type);
            artboard.insertBefore(refImgEl, grid);
        }
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
            schedulePreview();
        } catch (error) {
            console.error("[TS Ideogram] reference upload failed:", error);
        }
    }

    // ── Blocks rendering ────────────────────────────────────────────────── //
    function getSelected() {
        return work.blocks.find((b) => b.id === selectedId) || null;
    }

    function renderBlocks() {
        blocksLayer.innerHTML = "";
        work.blocks.forEach((block) => {
            const r = block.rect || { x: 0.1, y: 0.1, w: 0.3, h: 0.2 };
            const div = el("div", "ts-ideoe-block");
            div.classList.toggle("is-obj", block.type === "obj");
            div.classList.toggle("is-visual", block.type === "text" && !!block.visual_only);
            div.classList.toggle("is-selected", block.id === selectedId);
            div.style.left = `${r.x * 100}%`;
            div.style.top = `${r.y * 100}%`;
            div.style.width = `${r.w * 100}%`;
            div.style.height = `${r.h * 100}%`;

            if (block.type === "text" && !block.visual_only && (block.text || "").trim()) {
                const t = el("div", "ts-ideoe-block__text");
                t.textContent = block.text;
                t.style.color = normHex(block.color) || "#ffffff";
                t.style.fontSize = `${Math.max(10, r.h * artboardSize().h * 0.5)}px`;
                div.appendChild(t);
            }
            const label = el("div", "ts-ideoe-block__label",
                block.type === "obj" ? "OBJ" : (block.visual_only ? "↳ вручную" : "TXT"));
            div.appendChild(label);

            if (block.id === selectedId) {
                ["nw", "ne", "sw", "se", "n", "s", "e", "w"].forEach((dir) => {
                    const h = el("div", `ts-ideoe-handle ${dir}`);
                    h.addEventListener("pointerdown", (ev) => startDrag(ev, block, dir));
                    div.appendChild(h);
                });
            }

            div.addEventListener("pointerdown", (ev) => {
                if (ev.target.classList.contains("ts-ideoe-handle")) return;
                selectBlock(block.id);
                startDrag(ev, block, "move");
            });
            blocksLayer.appendChild(div);
        });
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
            }
            updateBboxReadout();
        }
        function onUp() {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            renderBlocks();
            schedulePreview();
        }
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
    }

    // ── Block CRUD ──────────────────────────────────────────────────────── //
    function addBlock(type) {
        const block = type === "text"
            ? {
                id: makeBlockId(), type: "text", rect: { x: 0.1, y: 0.1, w: 0.5, h: 0.18 },
                text: "ТЕКСТ", font_preset_id: fonts[0]?.id || "grotesque_black",
                weight: "Bold", case: "UPPERCASE", prominence: "Headline",
                legibility: { outline: true, high_contrast: true, solid_block: false },
                visual_only: false, color: "#FFFFFF", desc_override: "", color_palette: [],
            }
            : {
                id: makeBlockId(), type: "obj", rect: { x: 0.55, y: 0.2, w: 0.4, h: 0.7 },
                desc: "subject / graphic element", color_palette: [],
            };
        work.blocks.push(block);
        selectBlock(block.id);
        renderBlocks();
        schedulePreview();
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
        schedulePreview();
    }

    function deleteSelected() {
        const i = work.blocks.findIndex((b) => b.id === selectedId);
        if (i < 0) return;
        work.blocks.splice(i, 1);
        selectedId = work.blocks[Math.max(0, i - 1)]?.id || null;
        renderBlocks();
        renderInspector();
        schedulePreview();
    }

    function reorderSelected(delta) {
        const i = work.blocks.findIndex((b) => b.id === selectedId);
        if (i < 0) return;
        const j = clamp(i + delta, 0, work.blocks.length - 1);
        if (i === j) return;
        const [b] = work.blocks.splice(i, 1);
        work.blocks.splice(j, 0, b);
        renderBlocks();
        schedulePreview();
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

    function styleCard() {
        const card = el("div", "ts-ideoe-card");
        card.appendChild(el("h3", null, "Общий стиль"));

        const hldRow = el("div", "ts-ideoe-row");
        hldRow.appendChild(el("label", null, "Высокоуровневое описание (high_level_description)"));
        const hld = el("textarea");
        hld.value = work.high_level_description || "";
        hld.placeholder = "О чём всё изображение (1–2 предложения)…";
        hld.addEventListener("input", () => { work.high_level_description = hld.value; schedulePreview(); });
        hldRow.appendChild(hld);
        card.appendChild(hldRow);

        const presetRow = el("div", "ts-ideoe-row");
        presetRow.appendChild(el("label", null, "Пресет стиля"));
        const sel = el("select");
        const none = el("option", null, "— свой стиль —"); none.value = ""; sel.appendChild(none);
        styles.forEach((s) => {
            const o = el("option", null, s.name_ru || s.name_en || s.id);
            o.value = s.id;
            if (s.id === work.style.preset_id) o.selected = true;
            sel.appendChild(o);
        });
        sel.addEventListener("change", () => {
            const s = styles.find((x) => x.id === sel.value);
            if (s) {
                work.style = {
                    preset_id: s.id,
                    aesthetics: s.aesthetics || "",
                    lighting: s.lighting || "",
                    medium: s.medium || "graphic_design",
                    photo: s.photo || "",
                    art_style: s.art_style || "",
                    color_palette: cleanPalette(s.color_palette || [], IMAGE_PALETTE_CAP),
                };
            } else {
                work.style.preset_id = "";
            }
            renderInspector();
            schedulePreview();
        });
        presetRow.appendChild(sel);
        card.appendChild(presetRow);

        const mediumRow = el("div", "ts-ideoe-row");
        mediumRow.appendChild(el("label", null, "Medium"));
        const med = el("select");
        MEDIA_OPTIONS.forEach((m) => {
            const o = el("option", null, m); o.value = m;
            if (m === work.style.medium) o.selected = true;
            med.appendChild(o);
        });
        med.addEventListener("change", () => { work.style.medium = med.value; renderInspector(); schedulePreview(); });
        mediumRow.appendChild(med);
        card.appendChild(mediumRow);

        const isPhoto = work.style.medium === PHOTO_MEDIUM;
        [["aesthetics", "Aesthetics"], ["lighting", "Lighting"],
         isPhoto ? ["photo", "Photo (камера/оптика)"] : ["art_style", "Art style"]].forEach(([key, lbl]) => {
            const row = el("div", "ts-ideoe-row");
            row.appendChild(el("label", null, lbl));
            const inp = el("textarea");
            inp.value = work.style[key] || "";
            inp.addEventListener("input", () => { work.style[key] = inp.value; schedulePreview(); });
            row.appendChild(inp);
            card.appendChild(row);
        });

        const palRow = el("div", "ts-ideoe-row");
        palRow.appendChild(el("label", null, `Палитра изображения (до ${IMAGE_PALETTE_CAP})`));
        palRow.appendChild(buildPalette(
            () => work.style.color_palette,
            (next) => { work.style.color_palette = next; schedulePreview(); },
            IMAGE_PALETTE_CAP,
        ));
        card.appendChild(palRow);

        const bgRow = el("div", "ts-ideoe-row");
        bgRow.appendChild(el("label", null, "Фон (background)"));
        const bg = el("textarea");
        bg.value = work.background || "";
        bg.placeholder = "Описание фона/окружения…";
        bg.addEventListener("input", () => { work.background = bg.value; schedulePreview(); });
        bgRow.appendChild(bg);
        card.appendChild(bgRow);

        return card;
    }

    function blockCard() {
        const sel = getSelected();
        if (!sel) {
            const empty = el("div", "ts-ideoe-empty", "Выберите блок на холсте или добавьте новый (+ Текст / + Объект).");
            return empty;
        }
        const card = el("div", "ts-ideoe-card");
        card.appendChild(el("h3", null, sel.type === "obj" ? "Объект (obj)" : "Текстовый блок"));

        const bboxRow = el("div", "ts-ideoe-bbox");
        bboxReadoutEl = bboxRow;
        card.appendChild(bboxRow);
        updateBboxReadout();

        if (sel.type === "obj") {
            const descRow = el("div", "ts-ideoe-row");
            descRow.appendChild(el("label", null, "Описание объекта (desc)"));
            const ta = el("textarea");
            ta.value = sel.desc || "";
            ta.addEventListener("input", () => { sel.desc = ta.value; renderBlocks(); schedulePreview(); });
            descRow.appendChild(ta);
            card.appendChild(descRow);

            const palRow = el("div", "ts-ideoe-row");
            palRow.appendChild(el("label", null, `Палитра блока (до ${ELEMENT_PALETTE_CAP})`));
            palRow.appendChild(buildPalette(() => sel.color_palette, (n) => { sel.color_palette = n; schedulePreview(); }, ELEMENT_PALETTE_CAP));
            card.appendChild(palRow);
            return card;
        }

        // text block
        const textRow = el("div", "ts-ideoe-row");
        textRow.appendChild(el("label", null, "Текст (рендерится буква-в-букву)"));
        const ta = el("textarea");
        ta.value = sel.text || "";
        ta.placeholder = "Текст надписи…";
        ta.addEventListener("input", () => { sel.text = ta.value; renderBlocks(); renderWarnings(); schedulePreview(); });
        textRow.appendChild(ta);
        card.appendChild(textRow);

        const fontRow = el("div", "ts-ideoe-row");
        fontRow.appendChild(el("label", null, "Шрифт-пресет (описание — единственный реальный рычаг)"));
        const fontSel = el("select");
        const isCyr = /[Ѐ-ӿ]/.test(sel.text || "");
        const ordered = fonts.slice().sort((a, b) => {
            if (isCyr && (a.good_for_cyrillic !== b.good_for_cyrillic)) return a.good_for_cyrillic ? -1 : 1;
            return 0;
        });
        ordered.forEach((f) => {
            const safe = f.good_for_cyrillic ? "" : " ⚠";
            const o = el("option", null, `${f.name_ru || f.name_en}${safe}`);
            o.value = f.id;
            if (f.id === sel.font_preset_id) o.selected = true;
            fontSel.appendChild(o);
        });
        fontSel.addEventListener("change", () => { sel.font_preset_id = fontSel.value; renderDescPreview(); renderWarnings(); schedulePreview(); });
        fontRow.appendChild(fontSel);
        card.appendChild(fontRow);

        const segRow = el("div", "ts-ideoe-row");
        segRow.appendChild(el("label", null, "Вес"));
        segRow.appendChild(buildSegmented(WEIGHTS, () => sel.weight, (v) => { sel.weight = v; renderDescPreview(); schedulePreview(); }));
        card.appendChild(segRow);

        const caseRow = el("div", "ts-ideoe-row");
        caseRow.appendChild(el("label", null, "Регистр"));
        caseRow.appendChild(buildSegmented(CASES, () => sel.case, (v) => { sel.case = v; renderBlocks(); renderDescPreview(); renderWarnings(); schedulePreview(); }));
        card.appendChild(caseRow);

        const promRow = el("div", "ts-ideoe-row");
        promRow.appendChild(el("label", null, "Размер (словами)"));
        promRow.appendChild(buildSegmented(PROMINENCE, () => sel.prominence, (v) => { sel.prominence = v; renderDescPreview(); schedulePreview(); }));
        card.appendChild(promRow);

        const colorRow = el("div", "ts-ideoe-row");
        colorRow.appendChild(el("label", null, "Цвет текста"));
        const color = el("input");
        color.type = "color";
        color.value = normHex(sel.color) || "#FFFFFF";
        color.addEventListener("input", () => { sel.color = color.value.toUpperCase(); renderBlocks(); renderDescPreview(); schedulePreview(); });
        colorRow.appendChild(color);
        card.appendChild(colorRow);

        const legRow = el("div", "ts-ideoe-row");
        legRow.appendChild(el("label", null, "Читаемость"));
        const checks = el("div", "ts-ideoe-checks");
        sel.legibility = sel.legibility || {};
        [["outline", "Обводка"], ["high_contrast", "Контраст"], ["solid_block", "Плашка"]].forEach(([key, lbl]) => {
            const c = el("label", "ts-ideoe-check");
            const cb = el("input"); cb.type = "checkbox"; cb.checked = !!sel.legibility[key];
            cb.addEventListener("change", () => { sel.legibility[key] = cb.checked; renderDescPreview(); schedulePreview(); });
            c.append(cb, document.createTextNode(lbl));
            checks.appendChild(c);
        });
        legRow.appendChild(checks);
        card.appendChild(legRow);

        const voRow = el("label", "ts-ideoe-check");
        const vo = el("input"); vo.type = "checkbox"; vo.checked = !!sel.visual_only;
        vo.addEventListener("change", () => { sel.visual_only = vo.checked; renderBlocks(); renderDescPreview(); renderWarnings(); schedulePreview(); });
        voRow.append(vo, document.createTextNode("Текст вручную (visual-only — плашка под ручной оверлей)"));
        card.appendChild(voRow);

        const ovRow = el("div", "ts-ideoe-row");
        ovRow.appendChild(el("label", null, "Доп. описание (override, добавляется в конец)"));
        const ov = el("input"); ov.type = "text"; ov.value = sel.desc_override || "";
        ov.placeholder = "напр. slight upward tilt";
        ov.title = "Имена реальных шрифтов Ideogram не выбирают гарнитуру — только описания.";
        ov.addEventListener("input", () => { sel.desc_override = ov.value; renderDescPreview(); schedulePreview(); });
        ovRow.appendChild(ov);
        card.appendChild(ovRow);

        const palRow = el("div", "ts-ideoe-row");
        palRow.appendChild(el("label", null, `Палитра блока (до ${ELEMENT_PALETTE_CAP})`));
        palRow.appendChild(buildPalette(() => sel.color_palette, (n) => { sel.color_palette = n; schedulePreview(); }, ELEMENT_PALETTE_CAP));
        card.appendChild(palRow);

        const prevRow = el("div", "ts-ideoe-row");
        prevRow.appendChild(el("label", null, "Итоговое описание (desc) для модели:"));
        const prev = el("div", "ts-ideoe-descprev");
        prevRow.appendChild(prev);
        card.appendChild(prevRow);

        const warns = el("div", "ts-ideoe-warns");
        card.appendChild(warns);

        function renderDescPreview() {
            prev.textContent = sel.visual_only
                ? "(visual-only) область станет пустой плашкой без текста — добавьте надпись вручную в Figma/Photoshop."
                : composeTextDesc(sel, fontMap);
        }
        function renderWarnings() {
            warns.innerHTML = "";
            cyrillicWarnings(sel, work.blocks, fontMap).forEach((w) => warns.appendChild(el("div", "ts-ideoe-warn", w)));
            const anyCyr = work.blocks.some((b) => b.type === "text" && !b.visual_only && /[Ѐ-ӿ]/.test(b.text || ""));
            banner.style.display = anyCyr ? "block" : "none";
        }
        card._renderDescPreview = renderDescPreview;
        card._renderWarnings = renderWarnings;
        renderDescPreview();
        renderWarnings();
        return card;
    }

    let currentBlockCard = null;
    function renderDescPreview() { currentBlockCard?._renderDescPreview?.(); }
    function renderWarnings() {
        currentBlockCard?._renderWarnings?.();
        const anyCyr = work.blocks.some((b) => b.type === "text" && !b.visual_only && /[Ѐ-ӿ]/.test(b.text || ""));
        banner.style.display = anyCyr ? "block" : "none";
    }

    function renderInspector() {
        inspectorScroll.innerHTML = "";
        inspectorScroll.appendChild(styleCard());
        currentBlockCard = blockCard();
        inspectorScroll.appendChild(currentBlockCard);
        renderWarnings();
    }

    // ── Live caption preview (server-authoritative) ─────────────────────── //
    let previewTimer = null;
    function schedulePreview() {
        if (!jsonOpen) return;
        if (previewTimer) clearTimeout(previewTimer);
        previewTimer = setTimeout(async () => {
            const result = await fetchCaptionPreview(JSON.stringify(work));
            if (result) {
                try {
                    jsonPre.textContent = JSON.stringify(JSON.parse(result.json_prompt || "{}"), null, 2);
                } catch {
                    jsonPre.textContent = result.json_prompt || "{ }";
                }
            }
        }, 220);
    }

    // ── Wiring ──────────────────────────────────────────────────────────── //
    addText.addEventListener("click", () => addBlock("text"));
    addObj.addEventListener("click", () => addBlock("obj"));
    dupBtn.addEventListener("click", duplicateSelected);
    delBtn.addEventListener("click", deleteSelected);
    frontBtn.addEventListener("click", () => reorderSelected(1));
    backBtn.addEventListener("click", () => reorderSelected(-1));
    aspectSel.addEventListener("change", () => { work.aspect_ratio = aspectSel.value; layoutArtboard(); renderBlocks(); schedulePreview(); });
    refBtn.addEventListener("click", () => fileInput.click());
    refClear.addEventListener("click", () => { work.ref = null; renderReference(); schedulePreview(); });
    fileInput.addEventListener("change", () => { uploadReference(fileInput.files?.[0]); fileInput.value = ""; });
    jsonBtn.addEventListener("click", () => { jsonOpen = !jsonOpen; jsonPanel.classList.toggle("is-open", jsonOpen); if (jsonOpen) schedulePreview(); });

    // Drag & drop / paste reference onto the stage
    stageWrap.addEventListener("dragover", (e) => { e.preventDefault(); });
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

    // Click empty stage area deselects
    stageWrap.addEventListener("pointerdown", (e) => {
        if (e.target === stageWrap || e.target === stage || e.target === artboard || e.target === grid) {
            selectBlock(null);
        }
    });

    function close() {
        document.removeEventListener("paste", onPaste);
        document.removeEventListener("keydown", onKey);
        resizeObserver.disconnect();
        overlay.remove();
    }
    function commit() {
        onSave?.(JSON.parse(JSON.stringify(work)));
        close();
    }
    cancelBtn.addEventListener("click", close);
    saveBtn.addEventListener("click", commit);
    // No click-outside-to-close: avoids discarding edits by an accidental
    // click on the margin. Use Cancel / Esc to discard, Save to commit.
    function onKey(e) {
        if (e.key === "Escape") { e.stopPropagation(); close(); }
        else if ((e.key === "Delete" || e.key === "Backspace") && getSelected() && document.activeElement?.tagName !== "TEXTAREA" && document.activeElement?.tagName !== "INPUT") {
            deleteSelected();
        }
    }
    document.addEventListener("keydown", onKey);

    const resizeObserver = new ResizeObserver(() => { layoutArtboard(); renderBlocks(); });
    resizeObserver.observe(stageWrap);

    // ── Initial paint ───────────────────────────────────────────────────── //
    // Render synchronously first (reading clientWidth in layoutArtboard forces
    // a reflow, so the artboard gets sized immediately), then again on the next
    // frame, and retry on a short timer until the stage actually has width. This
    // does not rely solely on requestAnimationFrame firing and is robust to the
    // overlay not being laid out yet on the first tick.
    function fullRender() {
        layoutArtboard();
        renderReference();
        renderBlocks();
    }
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
