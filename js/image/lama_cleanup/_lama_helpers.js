// Shared setup for TS_LamaCleanup. The registerExtension call lives in
// ts-lama-cleanup.js so this module only owns the DOM widget logic.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

export const NODE_NAME = "TS_LamaCleanup";
const ROUTE_BASE = "/ts_lama_cleanup";
const STYLE_ID = "ts-lama-cleanup-styles";
export const DOM_WIDGET_NAME = "ts_lama_cleanup";
const INPUT_SOURCE_PATH = "source_path";
const INPUT_BRUSH_SIZE = "brush_size";
const INPUT_MAX_RESOLUTION = "max_resolution";
const INPUT_MASK_PADDING = "mask_padding";
const INPUT_FEATHER = "feather";
const INPUT_SESSION_ID = "session_id";
const INPUT_WORKING_PATH = "working_path";
const PROPERTY_SESSION_ID = "ts_lama_cleanup_session_id";
const DEFAULT_NODE_SIZE = [640, 520];
const MIN_NODE_WIDTH = 460;
const MIN_NODE_HEIGHT = 320;
const TITLE_BAR_HEIGHT = 30;
const STATUS_POLL_INTERVAL_MS = 1500;
const SOURCE_POLL_INTERVAL_MS = 300;
// Pixel margins inside the canvas reserved for the floating toolbar / status
// bar overlays. The image fit-letterboxes inside the area minus these so it
// never sits underneath the controls.
const IMAGE_PAD_TOP = 56;
const IMAGE_PAD_BOTTOM = 44;
const IMAGE_PAD_SIDE = 8;
// Cap how many edits the Undo stack remembers. Older entries are evicted FIFO
// and their backing temp files are removed via /cleanup_paths so disk usage
// stays bounded.
const MAX_HISTORY = 30;
const MEDIA_UPLOAD_ACCEPT = ["image/*", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"].join(",");

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-lama{--tslc-bg:#0e1218;--tslc-text:#e9eef6;--tslc-muted:#9aa6b8;--tslc-accent:#7aa2ff;--tslc-accent-strong:#3a72ff;--tslc-danger:#ef6f6c;--tslc-success:#82d6a8;--tslc-toolbar:rgba(12,16,22,.72);--tslc-toolbar-border:rgba(255,255,255,.08);position:relative;width:100%;height:100%;min-height:0;box-sizing:border-box;color:var(--tslc-text);font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif;background:repeating-conic-gradient(#1a2030 0% 25%,#0f141a 0% 50%) 50%/24px 24px;border:1px solid #28303c;border-radius:10px;overflow:hidden;user-select:none}
.ts-lama__canvas{position:absolute;inset:0;display:block;width:100%;height:100%;cursor:default;touch-action:none}
.ts-lama__canvas.has-image{cursor:none}
.ts-lama__empty{position:absolute;left:8px;right:8px;top:56px;bottom:44px;display:flex;align-items:center;justify-content:center;text-align:center;padding:16px;color:#cdd6e6;font-size:12px;pointer-events:none;background:linear-gradient(180deg,rgba(0,0,0,.45),rgba(0,0,0,.7));border-radius:6px}
.ts-lama__overlay{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:rgba(8,12,18,.6);backdrop-filter:blur(2px);color:var(--tslc-text);font-size:12px;pointer-events:none;flex-direction:column;gap:10px;z-index:5}
.ts-lama__overlay.is-active{display:flex}
.ts-lama__spinner{width:28px;height:28px;border-radius:999px;border:3px solid rgba(255,255,255,.14);border-top-color:var(--tslc-accent);animation:tslc-spin .9s linear infinite}
@keyframes tslc-spin{to{transform:rotate(360deg)}}
.ts-lama__toolbar{position:absolute;top:8px;left:8px;right:8px;display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:10px;background:var(--tslc-toolbar);border:1px solid var(--tslc-toolbar-border);backdrop-filter:blur(8px);z-index:6}
.ts-lama__group{display:flex;align-items:center;gap:6px}
.ts-lama__group--brush{flex:1 1 auto;min-width:0;justify-content:flex-start}
.ts-lama__btn{display:inline-flex;align-items:center;gap:5px;border:1px solid rgba(255,255,255,.12);background:rgba(20,26,36,.85);color:var(--tslc-text);border-radius:8px;padding:6px 11px;font-size:11px;cursor:pointer;font-weight:600;letter-spacing:.02em;white-space:nowrap}
.ts-lama__btn:hover{background:rgba(40,54,76,.95)}
.ts-lama__btn[disabled]{opacity:.4;cursor:not-allowed}
.ts-lama__btn--primary{background:linear-gradient(180deg,#7aa2ff,#3a72ff);border-color:#3a72ff;color:#0b1530}
.ts-lama__btn--primary:hover{background:linear-gradient(180deg,#90b6ff,#5180ff)}
.ts-lama__btn--icon{padding:6px 8px;width:30px;height:30px;justify-content:center}
.ts-lama__btn--icon svg{width:14px;height:14px;fill:currentColor;pointer-events:none}
.ts-lama__brush-label{font-size:10px;color:var(--tslc-muted);text-transform:uppercase;letter-spacing:.06em;white-space:nowrap}
.ts-lama__brush-slider{flex:1 1 auto;min-width:60px;max-width:200px;-webkit-appearance:none;appearance:none;height:4px;border-radius:999px;background:rgba(255,255,255,.18);outline:none;cursor:pointer}
.ts-lama__brush-slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:14px;border-radius:999px;background:var(--tslc-accent);border:2px solid #fff;cursor:pointer}
.ts-lama__brush-slider::-moz-range-thumb{width:14px;height:14px;border-radius:999px;background:var(--tslc-accent);border:2px solid #fff;cursor:pointer}
.ts-lama__brush-value{font-size:11px;color:var(--tslc-text);font-variant-numeric:tabular-nums;min-width:28px;text-align:right;font-weight:600}
.ts-lama__settings{position:absolute;top:50px;right:8px;width:240px;padding:10px;border-radius:10px;background:var(--tslc-toolbar);border:1px solid var(--tslc-toolbar-border);backdrop-filter:blur(8px);z-index:7;display:none;flex-direction:column;gap:10px;box-shadow:0 12px 32px rgba(0,0,0,.45)}
.ts-lama__settings.is-open{display:flex}
.ts-lama__settings-title{font-size:10px;color:var(--tslc-muted);text-transform:uppercase;letter-spacing:.06em;font-weight:700;margin-bottom:2px}
.ts-lama__field{display:flex;flex-direction:column;gap:4px}
.ts-lama__field-row{display:flex;align-items:center;justify-content:space-between;gap:6px;font-size:11px;color:var(--tslc-text)}
.ts-lama__field-name{color:var(--tslc-muted);font-size:10px;text-transform:uppercase;letter-spacing:.04em;font-weight:600}
.ts-lama__field-value{font-variant-numeric:tabular-nums;font-weight:600;font-size:11px}
.ts-lama__field-slider{-webkit-appearance:none;appearance:none;width:100%;height:4px;border-radius:999px;background:rgba(255,255,255,.18);outline:none;cursor:pointer}
.ts-lama__field-slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:12px;height:12px;border-radius:999px;background:var(--tslc-accent);border:2px solid #fff;cursor:pointer}
.ts-lama__field-slider::-moz-range-thumb{width:12px;height:12px;border-radius:999px;background:var(--tslc-accent);border:2px solid #fff;cursor:pointer}
.ts-lama__statusbar{position:absolute;left:8px;right:8px;bottom:8px;padding:6px 10px;font-size:11px;color:var(--tslc-muted);background:var(--tslc-toolbar);border:1px solid var(--tslc-toolbar-border);border-radius:8px;backdrop-filter:blur(6px);display:flex;justify-content:space-between;gap:10px;align-items:center;pointer-events:none;z-index:4}
.ts-lama__statusbar.is-error{color:var(--tslc-danger)}
.ts-lama__statusbar.is-success{color:var(--tslc-success)}
.ts-lama__statusbar-text{flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ts-lama__statusbar-meta{font-variant-numeric:tabular-nums;color:var(--tslc-muted);font-size:10px;white-space:nowrap}
.ts-lama__hidden-input{position:fixed;left:-9999px;top:-9999px;width:1px;height:1px;opacity:0;pointer-events:none}
.ts-lama__cursor{position:absolute;margin:0;padding:0;border-radius:50%;border:1.5px solid rgba(255,255,255,0.95);box-shadow:0 0 0 1px rgba(0,0,0,0.65);box-sizing:border-box;pointer-events:none;will-change:left,top,width,height;display:none;z-index:3}
.ts-lama__cursor.is-visible{display:block}
.ts-lama.is-drag-over{outline:2px dashed var(--tslc-accent);outline-offset:-4px;outline-style:dashed}
.ts-lama__drop-hint{position:absolute;inset:8px;display:none;align-items:center;justify-content:center;border:2px dashed var(--tslc-accent);border-radius:10px;background:rgba(122,162,255,.08);color:var(--tslc-text);font-size:13px;font-weight:600;pointer-events:none;z-index:8;text-shadow:0 1px 2px rgba(0,0,0,.6)}
.ts-lama.is-drag-over .ts-lama__drop-hint{display:flex}
`;
    document.head.appendChild(style);
}

function isNodesV2() {
    return Boolean(window?.comfyAPI?.domWidget?.DOMWidgetImpl);
}
function stopPropagation(element, events) {
    events.forEach((name) => element.addEventListener(name, (event) => event.stopPropagation()));
}
function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}
export function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || null;
}
function hideWidget(node, name) {
    const widget = getWidget(node, name);
    if (widget) {
        widget.hidden = true;
        widget.type = "hidden";
        widget.serialize = true;
        widget.options = { ...(widget.options || {}), hidden: true, serialize: true };
        widget.computeSize = () => [0, 0];
    }
    const input = node?.inputs?.find((item) => item?.name === name);
    if (input) input.hidden = true;
}
function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (widget) {
        widget.value = value;
        if (typeof widget.callback === "function") widget.callback(value);
    }
    node.properties ||= {};
    node.properties[name] = value;
}
function getWidgetValue(node, name, fallback) {
    return getWidget(node, name)?.value ?? fallback;
}
function removeDomWidget(node) {
    if (!Array.isArray(node?.widgets)) return;
    for (let index = node.widgets.length - 1; index >= 0; index -= 1) {
        const widget = node.widgets[index];
        if (widget?.name !== DOM_WIDGET_NAME) continue;
        (widget.element || widget.el || widget.container)?.remove?.();
        node.widgets.splice(index, 1);
    }
}
function ensureSessionId(node) {
    let sessionId = String(node?.properties?.[PROPERTY_SESSION_ID] || "").trim();
    if (!sessionId) {
        sessionId = String(getWidgetValue(node, INPUT_SESSION_ID, "") || "").trim();
    }
    if (!sessionId) {
        const cryptoObj = window.crypto || window.msCrypto;
        if (cryptoObj?.randomUUID) {
            sessionId = cryptoObj.randomUUID().replaceAll("-", "");
        } else {
            sessionId = `s${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
        }
    }
    sessionId = sessionId.replace(/[^a-zA-Z0-9_-]/g, "").slice(0, 64);
    node.properties ||= {};
    node.properties[PROPERTY_SESSION_ID] = sessionId;
    setWidgetValue(node, INPUT_SESSION_ID, sessionId);
    return sessionId;
}
function imageUrlForPath(filepath) {
    if (!filepath) return "";
    return api.apiURL(`${ROUTE_BASE}/view?filepath=${encodeURIComponent(filepath)}&t=${Date.now()}`);
}
function scheduleCanvasDirty() {
    app?.graph?.setDirtyCanvas?.(true, true);
}
function suppressDefaultImagePreview(node) {
    try {
        delete node.imgs;
    } catch {}
    try {
        Object.defineProperty(node, "imgs", {
            configurable: true,
            enumerable: true,
            get() {
                return [];
            },
            set() {
                /* swallow assignments from upload widget callbacks */
            },
        });
    } catch (error) {
        console.warn("[TS Lama Cleanup] Failed to suppress default image preview:", error);
    }
    try {
        node.imageIndex = null;
    } catch {}
}
function buildAnnotatedPath(uploadPayload) {
    const filename = String(uploadPayload?.name || "").trim();
    const uploadType = String(uploadPayload?.type || "input").trim() || "input";
    const subfolder = String(uploadPayload?.subfolder || "").trim().replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
    if (!filename) return "";
    return subfolder ? `${subfolder}/${filename} [${uploadType}]` : `${filename} [${uploadType}]`;
}
function gearIconSvg() {
    // Material Design "settings" icon (filled). Single clean path.
    return `<svg viewBox="0 0 24 24"><path d="M19.43 12.98c.04-.32.07-.64.07-.98 0-.34-.03-.66-.07-.98l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.3-.61-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98l-.38-2.65c-.04-.24-.24-.42-.49-.42h-4c-.25 0-.45.18-.49.42l-.38 2.65c-.61.25-1.17.59-1.69.98l-2.49-1c-.23-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64l2.11 1.65c-.04.32-.07.65-.07.98 0 .33.03.66.07.98l-2.11 1.65c-.19.15-.24.42-.12.64l2 3.46c.12.22.39.3.61.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.04.24.24.42.49.42h4c.25 0 .45-.18.49-.42l.38-2.65c.61-.25 1.17-.59 1.69-.98l2.49 1c.23.09.49 0 .61-.22l2-3.46c.12-.22.07-.49-.12-.64l-2.11-1.65zM12 15.5c-1.93 0-3.5-1.57-3.5-3.5s1.57-3.5 3.5-3.5 3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5z"/></svg>`;
}
function undoIconSvg() {
    // Material Design "undo" icon.
    return `<svg viewBox="0 0 24 24"><path d="M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z"/></svg>`;
}
function redoIconSvg() {
    // Material Design "redo" icon.
    return `<svg viewBox="0 0 24 24"><path d="M18.4 10.6C16.55 8.99 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z"/></svg>`;
}

function makeSlider({ min, max, step, value, onInput, className }) {
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    slider.value = String(value);
    slider.className = className;
    slider.addEventListener("input", (event) => {
        const next = Number(event.target.value);
        onInput(next);
    });
    return slider;
}

export function setupLamaCleanup(node) {
    if (!node || typeof node.addDOMWidget !== "function") return;
    if (typeof node._tsLamaCleanupCleanup === "function") {
        try { node._tsLamaCleanupCleanup(); } catch {}
    }
    removeDomWidget(node);
    ensureStyles();
    suppressDefaultImagePreview(node);
    // Hide every standard widget — we render our own controls inside the canvas.
    hideWidget(node, INPUT_SOURCE_PATH);
    hideWidget(node, INPUT_BRUSH_SIZE);
    hideWidget(node, INPUT_MAX_RESOLUTION);
    hideWidget(node, INPUT_MASK_PADDING);
    hideWidget(node, INPUT_FEATHER);
    hideWidget(node, INPUT_SESSION_ID);
    hideWidget(node, INPUT_WORKING_PATH);
    node.resizable = true;
    node.size = [
        Math.max(Number(node.size?.[0]) || DEFAULT_NODE_SIZE[0], MIN_NODE_WIDTH),
        Math.max(Number(node.size?.[1]) || DEFAULT_NODE_SIZE[1], MIN_NODE_HEIGHT),
    ];
    node.min_size = [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];

    const sessionId = ensureSessionId(node);

    const state = {
        sessionId,
        sourcePath: String(getWidgetValue(node, INPUT_SOURCE_PATH, "") || ""),
        workingPath: String(getWidgetValue(node, INPUT_WORKING_PATH, "") || ""),
        brushSize: Number(getWidgetValue(node, INPUT_BRUSH_SIZE, 40) || 40),
        maxResolution: Number(getWidgetValue(node, INPUT_MAX_RESOLUTION, 512) || 512),
        maskPadding: Number(getWidgetValue(node, INPUT_MASK_PADDING, 64) || 64),
        feather: Number(getWidgetValue(node, INPUT_FEATHER, 4) || 4),
        statusText: "",
        statusKind: "info",
        image: null,
        imageWidth: 0,
        imageHeight: 0,
        scale: 1,
        offsetX: 0,
        offsetY: 0,
        isProcessing: false,
        isModelLoading: false,
        modelStatusPollHandle: null,
        isDrawing: false,
        cursorImageX: 0,
        cursorImageY: 0,
        cursorVisible: false,
        // Last known mouse position in viewport coordinates. Used to place the
        // HTML cursor element exactly under the pointer regardless of any
        // border/padding/transform offsets between canvas and container.
        cursorClientX: 0,
        cursorClientY: 0,
        lastDrawImageX: 0,
        lastDrawImageY: 0,
        sourcePollHandle: null,
        settingsOpen: false,
        // Edit history for Undo/Redo. Each entry is a working_path string;
        // backend creates a fresh file per inpaint/seed so old steps survive.
        history: [],
        historyIndex: -1,
    };
    // Seed history from a previously serialised working path so reload of a
    // saved workflow keeps undo intact (only one step, but the picture loads).
    if (state.workingPath) {
        state.history = [state.workingPath];
        state.historyIndex = 0;
    }

    const maskCanvas = document.createElement("canvas");
    const maskCtx = maskCanvas.getContext("2d");
    // Offscreen canvas used to tint the mask before compositing it over the
    // image. Avoids source-in tricks that wiped the image when the mask was
    // empty.
    const tintedMaskCanvas = document.createElement("canvas");
    const tintedMaskCtx = tintedMaskCanvas.getContext("2d");
    // Pre-rendered image at display resolution so each mouse-move redraw
    // only needs to blit a small bitmap instead of downscaling the full
    // (potentially 4K+) source image. Rebuilt on image load and on resize.
    const imageCacheCanvas = document.createElement("canvas");
    const imageCacheCtx = imageCacheCanvas.getContext("2d");
    let imageCacheValid = false;

    const container = document.createElement("div");
    container.className = "ts-lama";

    const canvas = document.createElement("canvas");
    canvas.className = "ts-lama__canvas";

    const empty = document.createElement("div");
    empty.className = "ts-lama__empty";
    empty.textContent = "Click “Load Image” to begin.";

    const overlay = document.createElement("div");
    overlay.className = "ts-lama__overlay";
    const spinner = document.createElement("div");
    spinner.className = "ts-lama__spinner";
    const overlayLabel = document.createElement("div");
    overlayLabel.textContent = "Processing...";
    overlay.append(spinner, overlayLabel);

    // Toolbar
    const toolbar = document.createElement("div");
    toolbar.className = "ts-lama__toolbar";

    const leftGroup = document.createElement("div");
    leftGroup.className = "ts-lama__group";
    const loadButton = document.createElement("button");
    loadButton.className = "ts-lama__btn ts-lama__btn--primary";
    loadButton.textContent = "Load Image";
    const saveButton = document.createElement("button");
    saveButton.className = "ts-lama__btn";
    saveButton.textContent = "Save Image";
    saveButton.title = "Save the current cleaned image into the ComfyUI output folder.";
    const resetButton = document.createElement("button");
    resetButton.className = "ts-lama__btn";
    resetButton.textContent = "Reset";
    resetButton.title = "Discard local edits and restart from the loaded image.";
    const undoButton = document.createElement("button");
    undoButton.className = "ts-lama__btn ts-lama__btn--icon";
    undoButton.title = "Undo last edit";
    undoButton.innerHTML = undoIconSvg();
    undoButton.disabled = true;
    const redoButton = document.createElement("button");
    redoButton.className = "ts-lama__btn ts-lama__btn--icon";
    redoButton.title = "Redo edit";
    redoButton.innerHTML = redoIconSvg();
    redoButton.disabled = true;
    leftGroup.append(loadButton, saveButton, resetButton, undoButton, redoButton);

    const brushGroup = document.createElement("div");
    brushGroup.className = "ts-lama__group ts-lama__group--brush";
    const brushLabel = document.createElement("div");
    brushLabel.className = "ts-lama__brush-label";
    brushLabel.textContent = "Brush";
    const brushSlider = makeSlider({
        min: 1,
        max: 400,
        step: 1,
        value: state.brushSize,
        className: "ts-lama__brush-slider",
        onInput: (next) => {
            state.brushSize = next;
            brushValueLabel.textContent = String(Math.round(next));
            setWidgetValue(node, INPUT_BRUSH_SIZE, next);
            updateCursorElement();
        },
    });
    const brushValueLabel = document.createElement("div");
    brushValueLabel.className = "ts-lama__brush-value";
    brushValueLabel.textContent = String(Math.round(state.brushSize));
    brushGroup.append(brushLabel, brushSlider, brushValueLabel);

    const rightGroup = document.createElement("div");
    rightGroup.className = "ts-lama__group";
    const settingsButton = document.createElement("button");
    settingsButton.className = "ts-lama__btn ts-lama__btn--icon";
    settingsButton.title = "Advanced settings";
    settingsButton.innerHTML = gearIconSvg();
    rightGroup.append(settingsButton);

    toolbar.append(leftGroup, brushGroup, rightGroup);

    // Settings popover
    const settings = document.createElement("div");
    settings.className = "ts-lama__settings";

    const settingsTitle = document.createElement("div");
    settingsTitle.className = "ts-lama__settings-title";
    settingsTitle.textContent = "Advanced";
    settings.append(settingsTitle);

    function buildField(name, options, getter, setter, widgetKey) {
        const field = document.createElement("div");
        field.className = "ts-lama__field";
        const row = document.createElement("div");
        row.className = "ts-lama__field-row";
        const nameEl = document.createElement("div");
        nameEl.className = "ts-lama__field-name";
        nameEl.textContent = name;
        const valueEl = document.createElement("div");
        valueEl.className = "ts-lama__field-value";
        valueEl.textContent = String(Math.round(getter()));
        row.append(nameEl, valueEl);
        const slider = makeSlider({
            min: options.min,
            max: options.max,
            step: options.step,
            value: getter(),
            className: "ts-lama__field-slider",
            onInput: (next) => {
                setter(next);
                valueEl.textContent = String(Math.round(next));
                setWidgetValue(node, widgetKey, next);
            },
        });
        field.append(row, slider);
        return field;
    }

    settings.append(
        buildField("Max LaMa resolution", { min: 128, max: 2048, step: 64 }, () => state.maxResolution, (next) => { state.maxResolution = next; }, INPUT_MAX_RESOLUTION),
        buildField("Mask context padding", { min: 0, max: 512, step: 8 }, () => state.maskPadding, (next) => { state.maskPadding = next; }, INPUT_MASK_PADDING),
        buildField("Composite feather", { min: 0, max: 64, step: 1 }, () => state.feather, (next) => { state.feather = next; }, INPUT_FEATHER),
    );

    // Status bar
    const statusBar = document.createElement("div");
    statusBar.className = "ts-lama__statusbar";
    const statusText = document.createElement("div");
    statusText.className = "ts-lama__statusbar-text";
    statusText.textContent = "Click “Load Image” to begin.";
    const statusMeta = document.createElement("div");
    statusMeta.className = "ts-lama__statusbar-meta";
    statusBar.append(statusText, statusMeta);

    // Hidden file input for Load Image
    const fileInput = document.createElement("input");
    fileInput.className = "ts-lama__hidden-input";
    fileInput.type = "file";
    fileInput.accept = MEDIA_UPLOAD_ACCEPT;

    // Cursor circle as a real HTML element — moved with CSS transform so
    // mouse movement doesn't force a full canvas repaint of the image.
    const cursorElement = document.createElement("div");
    cursorElement.className = "ts-lama__cursor";

    // Visual hint shown while dragging an image file over the node.
    const dropHint = document.createElement("div");
    dropHint.className = "ts-lama__drop-hint";
    dropHint.textContent = "Drop image to load";

    container.append(canvas, empty, overlay, toolbar, settings, statusBar, fileInput, cursorElement, dropHint);

    stopPropagation(container, [
        "pointerdown", "pointerup", "pointermove",
        "mousedown", "mouseup", "mousemove",
        "wheel", "click", "dblclick", "contextmenu",
    ]);

    // ComfyUI core (>=1.34) routes DOM widgets through computeLayoutSize:
    //   distributeSpace gives them all the leftover node-body height bounded
    //   by [getMinHeight, getMaxHeight].
    // We must NOT set widget.computeSize — that pushes us into the fixed-size
    // branch and breaks the layout (creating runaway height in V2/Vue).
    const widgetOptions = {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 220,
        getMaxHeight: () => 8192,
        afterResize: () => { requestRedraw(); },
    };
    const domWidget = node.addDOMWidget(DOM_WIDGET_NAME, "div", container, widgetOptions);
    const domWidgetEl = domWidget?.element || domWidget?.el || domWidget?.container;

    function syncDomSize() {
        // Layout already allocates a slot for us; we just stretch the inner
        // element/container to fill it. No pixel math, no node.size feedback.
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

    function setStatus(message, kind = "info") {
        state.statusText = message || "";
        state.statusKind = kind;
        statusText.textContent = message || "";
        statusBar.classList.toggle("is-error", kind === "error");
        statusBar.classList.toggle("is-success", kind === "success");
    }

    function setOverlay(active, label = "Processing...") {
        overlay.classList.toggle("is-active", Boolean(active));
        overlayLabel.textContent = label;
    }

    function updateMeta() {
        const filename = state.sourcePath ? state.sourcePath.split(/[\\/]/).pop().replace(/\s\[input\]$/i, "") : "";
        const historyTag = state.history.length > 1
            ? ` • step ${state.historyIndex + 1}/${state.history.length}`
            : "";
        statusMeta.textContent = state.imageWidth && state.imageHeight
            ? `${filename || "image"} • ${state.imageWidth} × ${state.imageHeight}${historyTag}`
            : filename || "";
        empty.style.display = state.image ? "none" : "flex";
        // Hide the native cursor only while an image is loaded (we draw our
        // own brush circle then). Otherwise show the default arrow so the
        // mouse is visible over the empty canvas area.
        canvas.classList.toggle("has-image", Boolean(state.image));
        saveButton.disabled = !state.workingPath || state.isProcessing;
        resetButton.disabled = !state.workingPath || state.isProcessing;
        loadButton.disabled = state.isProcessing;
        settingsButton.disabled = state.isProcessing;
        undoButton.disabled = state.isProcessing || state.historyIndex <= 0;
        redoButton.disabled = state.isProcessing || state.historyIndex >= state.history.length - 1;
    }

    function pushHistory(path) {
        if (!path) return;
        // Dropped "future" entries (when user did Undo then made a new edit)
        // become orphan files; collect them so we can ask the backend to
        // remove them.
        const droppedFuture = state.history.slice(state.historyIndex + 1);
        state.history = state.history.slice(0, state.historyIndex + 1);
        state.history.push(path);
        state.historyIndex = state.history.length - 1;
        // FIFO eviction once the stack exceeds MAX_HISTORY.
        const overflow = state.history.length - MAX_HISTORY;
        const droppedOverflow = overflow > 0 ? state.history.splice(0, overflow) : [];
        if (overflow > 0) state.historyIndex -= overflow;
        const toCleanup = droppedFuture.concat(droppedOverflow).filter(Boolean);
        if (toCleanup.length) {
            cleanupPaths(toCleanup).catch(() => {});
        }
    }

    async function cleanupPaths(paths) {
        if (!paths || !paths.length) return;
        try {
            await api.fetchApi(`${ROUTE_BASE}/cleanup_paths`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: state.sessionId, paths }),
            });
        } catch {
            // Cleanup is best-effort; failures don't block editing.
        }
    }

    function resetHistoryTo(path) {
        if (!path) {
            state.history = [];
            state.historyIndex = -1;
            return;
        }
        state.history = [path];
        state.historyIndex = 0;
    }

    async function goToHistory(targetIndex) {
        if (state.isProcessing) return;
        if (targetIndex < 0 || targetIndex >= state.history.length) return;
        state.historyIndex = targetIndex;
        const path = state.history[targetIndex];
        state.workingPath = path;
        setWidgetValue(node, INPUT_WORKING_PATH, path);
        await refreshImage({ clearMask: true });
        updateMeta();
    }

    async function doUndo() {
        if (state.historyIndex <= 0) return;
        await goToHistory(state.historyIndex - 1);
        setStatus("Reverted previous edit.", "info");
    }

    async function doRedo() {
        if (state.historyIndex >= state.history.length - 1) return;
        await goToHistory(state.historyIndex + 1);
        setStatus("Restored next edit.", "info");
    }

    function ensureMaskCanvasSize() {
        if (!state.imageWidth || !state.imageHeight) return;
        if (maskCanvas.width !== state.imageWidth || maskCanvas.height !== state.imageHeight) {
            maskCanvas.width = state.imageWidth;
            maskCanvas.height = state.imageHeight;
        }
        if (tintedMaskCanvas.width !== state.imageWidth || tintedMaskCanvas.height !== state.imageHeight) {
            tintedMaskCanvas.width = state.imageWidth;
            tintedMaskCanvas.height = state.imageHeight;
        }
    }

    function clearMask() {
        if (maskCtx) maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        if (tintedMaskCtx) tintedMaskCtx.clearRect(0, 0, tintedMaskCanvas.width, tintedMaskCanvas.height);
    }

    function isMaskEmpty() {
        if (!maskCanvas.width || !maskCanvas.height) return true;
        const data = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
        for (let index = 3; index < data.length; index += 4) {
            if (data[index] > 8) return false;
        }
        return true;
    }

    // Paint the same brush stamp into BOTH canvases:
    //   maskCanvas      — white pixels, used as the /inpaint payload mask
    //   tintedMaskCanvas — solid dark pixels, used directly by redraw() for
    //                      display so we don't have to rebuild a tinted copy
    //                      on every frame (was the main cause of lag while
    //                      painting big images).
    function drawBrushAt(imageX, imageY, radius) {
        ensureMaskCanvasSize();
        if (radius <= 0) return;
        if (maskCtx) {
            maskCtx.fillStyle = "rgba(255,255,255,1)";
            maskCtx.beginPath();
            maskCtx.arc(imageX, imageY, radius, 0, Math.PI * 2);
            maskCtx.fill();
        }
        if (tintedMaskCtx) {
            tintedMaskCtx.fillStyle = "rgba(8,12,18,1)";
            tintedMaskCtx.beginPath();
            tintedMaskCtx.arc(imageX, imageY, radius, 0, Math.PI * 2);
            tintedMaskCtx.fill();
        }
    }

    function drawSegment(fromX, fromY, toX, toY, radius) {
        ensureMaskCanvasSize();
        if (radius <= 0) return;
        if (maskCtx) {
            maskCtx.strokeStyle = "rgba(255,255,255,1)";
            maskCtx.lineWidth = radius * 2;
            maskCtx.lineCap = "round";
            maskCtx.lineJoin = "round";
            maskCtx.beginPath();
            maskCtx.moveTo(fromX, fromY);
            maskCtx.lineTo(toX, toY);
            maskCtx.stroke();
        }
        if (tintedMaskCtx) {
            tintedMaskCtx.strokeStyle = "rgba(8,12,18,1)";
            tintedMaskCtx.lineWidth = radius * 2;
            tintedMaskCtx.lineCap = "round";
            tintedMaskCtx.lineJoin = "round";
            tintedMaskCtx.beginPath();
            tintedMaskCtx.moveTo(fromX, fromY);
            tintedMaskCtx.lineTo(toX, toY);
            tintedMaskCtx.stroke();
        }
    }

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = Math.max(1, Math.floor(rect.width * dpr));
        const height = Math.max(1, Math.floor(rect.height * dpr));
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
            imageCacheValid = false;
        }
        // Compute the usable image area inside canvas, accounting for the
        // toolbar (top) and statusbar (bottom) overlays so the image fit-
        // letterboxes between them. state.offsetX/Y are kept fresh between
        // redraws so cursor positioning, mask compositing and image cache
        // stay aligned.
        if (state.imageWidth > 0 && state.imageHeight > 0 && rect.width > 0 && rect.height > 0) {
            const usableWidth = Math.max(1, rect.width - IMAGE_PAD_SIDE * 2);
            const usableHeight = Math.max(1, rect.height - IMAGE_PAD_TOP - IMAGE_PAD_BOTTOM);
            const scale = Math.min(usableWidth / state.imageWidth, usableHeight / state.imageHeight);
            state.scale = scale;
            state.offsetX = IMAGE_PAD_SIDE + (usableWidth - state.imageWidth * scale) / 2;
            state.offsetY = IMAGE_PAD_TOP + (usableHeight - state.imageHeight * scale) / 2;
        }
        return { rectWidth: rect.width, rectHeight: rect.height, dpr };
    }

    function rebuildImageCache(rectWidth, rectHeight, dpr) {
        if (!state.image || !state.imageWidth || !state.imageHeight) {
            imageCacheValid = false;
            return;
        }
        const cacheW = Math.max(1, Math.floor(rectWidth * dpr));
        const cacheH = Math.max(1, Math.floor(rectHeight * dpr));
        if (imageCacheCanvas.width !== cacheW || imageCacheCanvas.height !== cacheH) {
            imageCacheCanvas.width = cacheW;
            imageCacheCanvas.height = cacheH;
        }
        imageCacheCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        imageCacheCtx.clearRect(0, 0, rectWidth, rectHeight);
        // Use the padded transform that resizeCanvas just stored on `state`,
        // so the cached image matches the placement assumed by mask blits and
        // pointer→image math.
        const drawWidth = state.imageWidth * state.scale;
        const drawHeight = state.imageHeight * state.scale;
        imageCacheCtx.drawImage(state.image, state.offsetX, state.offsetY, drawWidth, drawHeight);
        imageCacheValid = true;
    }

    function updateCursorElement() {
        if (!state.cursorVisible || state.isProcessing || !state.image) {
            cursorElement.classList.remove("is-visible");
            return;
        }
        const containerRect = container.getBoundingClientRect();
        if (!containerRect.width || !containerRect.height) {
            cursorElement.classList.remove("is-visible");
            return;
        }
        // CSS `left`/`top` and `width`/`height` are interpreted in the
        // container's LOCAL (pre-transform) pixel space, but `clientX`,
        // `clientY` and `getBoundingClientRect()` are in VIEWPORT (post-
        // transform) pixels. When LiteGraph or Vue applies a CSS scale to a
        // parent (graph zoom, node scaling), those two spaces diverge and the
        // cursor drifts away from the mouse. We detect the effective parent
        // scale from the ratio of rendered to layout size and convert.
        const layoutWidth = container.offsetWidth || containerRect.width;
        const parentScale = layoutWidth > 0 ? containerRect.width / layoutWidth : 1;
        const inverseScale = parentScale > 0.001 ? 1 / parentScale : 1;
        const xLocal = (state.cursorClientX - containerRect.left) * inverseScale - (container.clientLeft || 0);
        const yLocal = (state.cursorClientY - containerRect.top) * inverseScale - (container.clientTop || 0);
        const visualScale = state.scale || 1;
        const radius = Math.max(2, state.brushSize * visualScale * 0.5 * inverseScale);
        const size = radius * 2;
        cursorElement.style.width = `${size}px`;
        cursorElement.style.height = `${size}px`;
        cursorElement.style.left = `${xLocal - radius}px`;
        cursorElement.style.top = `${yLocal - radius}px`;
        cursorElement.classList.add("is-visible");
    }

    let pendingLayoutAttempts = 0;
    function redraw() {
        const { rectWidth, rectHeight, dpr } = resizeCanvas();
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        if ((rectWidth <= 0 || rectHeight <= 0) && pendingLayoutAttempts < 6) {
            pendingLayoutAttempts += 1;
            window.setTimeout(() => requestRedraw(), 40);
            return;
        }
        pendingLayoutAttempts = 0;
        // Clear canvas in raw pixel space — cheap full-canvas clear.
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!state.image || !state.imageWidth || !state.imageHeight) {
            updateCursorElement();
            return;
        }
        // Rebuild the cached display-resolution image only when the canvas
        // size or the loaded image changed. Subsequent redraws (e.g. mask
        // updates while painting) just blit the cache.
        if (!imageCacheValid) {
            rebuildImageCache(rectWidth, rectHeight, dpr);
        }
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.drawImage(imageCacheCanvas, 0, 0);
        if (tintedMaskCanvas.width && tintedMaskCanvas.height) {
            // tintedMaskCanvas is kept up to date by drawBrushAt/drawSegment,
            // so display only needs a single scaled blit.
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.save();
            ctx.globalAlpha = 0.72;
            const drawWidth = state.imageWidth * state.scale;
            const drawHeight = state.imageHeight * state.scale;
            ctx.drawImage(tintedMaskCanvas, state.offsetX, state.offsetY, drawWidth, drawHeight);
            ctx.restore();
        }
        // Cursor is rendered as an HTML element via updateCursorElement —
        // moving the cursor no longer requires touching the canvas.
        updateCursorElement();
    }

    let redrawScheduled = false;
    function requestRedraw() {
        if (redrawScheduled) return;
        redrawScheduled = true;
        requestAnimationFrame(() => {
            redrawScheduled = false;
            redraw();
            scheduleCanvasDirty();
        });
    }

    function pointerToImageCoords(event) {
        const rect = canvas.getBoundingClientRect();
        const xInCanvas = event.clientX - rect.left;
        const yInCanvas = event.clientY - rect.top;
        const scale = state.scale > 0 ? state.scale : 1;
        const imageX = (xInCanvas - state.offsetX) / scale;
        const imageY = (yInCanvas - state.offsetY) / scale;
        return {
            imageX: clamp(imageX, 0, Math.max(1, state.imageWidth - 1)),
            imageY: clamp(imageY, 0, Math.max(1, state.imageHeight - 1)),
            withinImage: imageX >= 0 && imageY >= 0 && imageX < state.imageWidth && imageY < state.imageHeight,
        };
    }

    async function loadImageElement(url) {
        if (!url) return null;
        const image = new Image();
        image.src = url;
        if (typeof image.decode === "function") {
            try {
                await image.decode();
                return image;
            } catch {
                // fall through
            }
        }
        await new Promise((resolve, reject) => {
            if (image.complete && image.naturalWidth > 0) {
                resolve();
                return;
            }
            image.onload = () => resolve();
            image.onerror = () => reject(new Error("Failed to load image."));
        });
        return image;
    }

    async function refreshImage(options = {}) {
        const path = state.workingPath || state.sourcePath;
        if (!path) {
            state.image = null;
            state.imageWidth = 0;
            state.imageHeight = 0;
            clearMask();
            updateMeta();
            requestRedraw();
            return;
        }
        const url = imageUrlForPath(path);
        try {
            const image = await loadImageElement(url);
            if (!image) return;
            state.image = image;
            state.imageWidth = image.naturalWidth || image.width || 0;
            state.imageHeight = image.naturalHeight || image.height || 0;
            imageCacheValid = false;
            ensureMaskCanvasSize();
            if (options.clearMask !== false) clearMask();
            updateMeta();
            requestRedraw();
        } catch (error) {
            setStatus(`Failed to load image preview: ${error?.message || error}`, "error");
        }
    }

    async function uploadFile(file) {
        const form = new FormData();
        form.append("image", file, file.name);
        form.append("type", "input");
        const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload?.error || payload?.message || "Upload failed.");
        return buildAnnotatedPath(payload);
    }

    async function chooseSourceFile(file) {
        if (!file) return;
        state.isProcessing = false;
        setOverlay(true, "Uploading image...");
        setStatus("Uploading...", "info");
        try {
            const annotated = await uploadFile(file);
            if (!annotated) throw new Error("Upload failed.");
            state.sourcePath = annotated;
            state.workingPath = "";
            setWidgetValue(node, INPUT_SOURCE_PATH, annotated);
            setWidgetValue(node, INPUT_WORKING_PATH, "");
            clearMask();
            await seedWorkingFile();
        } catch (error) {
            setStatus(error?.message || "Failed to load image.", "error");
        } finally {
            setOverlay(false);
            updateMeta();
        }
    }

    async function seedWorkingFile() {
        if (!state.sourcePath) return;
        try {
            const response = await api.fetchApi(`${ROUTE_BASE}/seed`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: state.sessionId, source_path: state.sourcePath }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                setStatus(payload?.error || "Failed to prepare working copy.", "error");
                return;
            }
            state.workingPath = String(payload?.working_path || "");
            setWidgetValue(node, INPUT_WORKING_PATH, state.workingPath);
            // Fresh source resets history to a single entry.
            resetHistoryTo(state.workingPath);
            await refreshImage({ clearMask: true });
            setStatus("Image loaded. Paint defects with the brush.", "info");
        } catch (error) {
            setStatus(error?.message || "Failed to prepare working copy.", "error");
        }
    }

    async function pollModelStatusWhileProcessing() {
        if (state.modelStatusPollHandle) return;
        let attempts = 0;
        const tick = async () => {
            attempts += 1;
            if (!state.isProcessing) {
                if (state.modelStatusPollHandle) {
                    window.clearInterval(state.modelStatusPollHandle);
                    state.modelStatusPollHandle = null;
                }
                return;
            }
            try {
                const response = await api.fetchApi(`${ROUTE_BASE}/model_status`);
                const status = await response.json();
                if (status?.loading) {
                    state.isModelLoading = true;
                    setOverlay(true, String(status?.message || "Loading model..."));
                } else if (status?.loaded) {
                    if (state.isModelLoading) {
                        state.isModelLoading = false;
                        setOverlay(true, "Inpainting...");
                    }
                }
            } catch {}
            if (attempts > 200 && state.modelStatusPollHandle) {
                window.clearInterval(state.modelStatusPollHandle);
                state.modelStatusPollHandle = null;
            }
        };
        state.modelStatusPollHandle = window.setInterval(tick, STATUS_POLL_INTERVAL_MS);
        tick();
    }

    async function maskCanvasToDataUrl() {
        if (!maskCanvas.width || !maskCanvas.height) return "";
        return maskCanvas.toDataURL("image/png");
    }

    async function runInpaint() {
        if (state.isProcessing) return;
        if (!state.image || !state.imageWidth || !state.imageHeight) return;
        if (isMaskEmpty()) return;
        state.isProcessing = true;
        updateMeta();
        setOverlay(true, "Inpainting...");
        setStatus("Sending region to LaMa...", "info");
        pollModelStatusWhileProcessing();
        try {
            const maskDataUrl = await maskCanvasToDataUrl();
            const response = await api.fetchApi(`${ROUTE_BASE}/inpaint`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: state.sessionId,
                    source_path: state.sourcePath,
                    working_path: state.workingPath,
                    mask: maskDataUrl,
                    max_resolution: state.maxResolution,
                    mask_padding: state.maskPadding,
                    feather: state.feather,
                }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload?.error || "Inpaint request failed.");
            }
            state.workingPath = String(payload?.working_path || "");
            setWidgetValue(node, INPUT_WORKING_PATH, state.workingPath);
            pushHistory(state.workingPath);
            await refreshImage({ clearMask: true });
            setStatus("Cleanup applied. Paint another area or press Save.", "success");
        } catch (error) {
            setStatus(error?.message || "Inpaint request failed.", "error");
        } finally {
            state.isProcessing = false;
            state.isModelLoading = false;
            setOverlay(false);
            if (state.modelStatusPollHandle) {
                window.clearInterval(state.modelStatusPollHandle);
                state.modelStatusPollHandle = null;
            }
            updateMeta();
            requestRedraw();
        }
    }

    async function saveToOutput() {
        if (!state.workingPath) {
            setStatus("Nothing to save yet.", "error");
            return;
        }
        if (state.isProcessing) return;
        const sourceName = state.sourcePath ? state.sourcePath.split(/[\\/]/).pop().replace(/\s\[input\]$/i, "") : "ts_lama_cleanup";
        try {
            const response = await api.fetchApi(`${ROUTE_BASE}/save`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    working_path: state.workingPath,
                    filename: sourceName,
                }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload?.error || "Save failed.");
            }
            setStatus(`Saved to output: ${payload?.filename || payload?.saved_path || "unknown"}`, "success");
        } catch (error) {
            setStatus(error?.message || "Save failed.", "error");
        }
    }

    async function resetToSource() {
        if (state.isProcessing) return;
        if (!state.sourcePath) {
            setStatus("No source image to reset to.", "error");
            return;
        }
        try {
            await api.fetchApi(`${ROUTE_BASE}/reset`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: state.sessionId }),
            });
        } catch {}
        state.workingPath = "";
        setWidgetValue(node, INPUT_WORKING_PATH, "");
        clearMask();
        await seedWorkingFile();
    }

    function toggleSettings(open) {
        state.settingsOpen = open !== undefined ? Boolean(open) : !state.settingsOpen;
        settings.classList.toggle("is-open", state.settingsOpen);
    }

    function onPointerDown(event) {
        if (state.isProcessing) return;
        if (!state.image) return;
        if (event.button !== 0) return;
        const coords = pointerToImageCoords(event);
        if (!coords.withinImage) return;
        state.isDrawing = true;
        state.lastDrawImageX = coords.imageX;
        state.lastDrawImageY = coords.imageY;
        state.cursorImageX = coords.imageX;
        state.cursorImageY = coords.imageY;
        state.cursorClientX = event.clientX;
        state.cursorClientY = event.clientY;
        state.cursorVisible = true;
        drawBrushAt(coords.imageX, coords.imageY, state.brushSize * 0.5);
        canvas.setPointerCapture?.(event.pointerId);
        requestRedraw();
    }

    function onPointerMove(event) {
        if (!state.image) return;
        const coords = pointerToImageCoords(event);
        state.cursorImageX = coords.imageX;
        state.cursorImageY = coords.imageY;
        state.cursorClientX = event.clientX;
        state.cursorClientY = event.clientY;
        state.cursorVisible = coords.withinImage;
        if (state.isDrawing && coords.withinImage) {
            drawSegment(state.lastDrawImageX, state.lastDrawImageY, coords.imageX, coords.imageY, state.brushSize * 0.5);
            state.lastDrawImageX = coords.imageX;
            state.lastDrawImageY = coords.imageY;
            updateCursorElement();
            requestRedraw();
        } else {
            // Cursor-only movement: just slide the HTML cursor circle.
            // No canvas redraw, no image rescale — should stay at 60fps even
            // for 4K+ images.
            updateCursorElement();
        }
    }

    function onPointerUp(event) {
        if (!state.isDrawing) return;
        state.isDrawing = false;
        canvas.releasePointerCapture?.(event.pointerId);
        runInpaint();
    }

    function onPointerLeave() {
        state.cursorVisible = false;
        updateCursorElement();
    }

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("pointerleave", onPointerLeave);
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());

    loadButton.addEventListener("click", (event) => {
        event.stopPropagation();
        try {
            fileInput.click();
        } catch (error) {
            console.error("[TS Lama Cleanup] fileInput.click failed:", error);
            setStatus(`Failed to open file picker: ${error?.message || error}`, "error");
        }
    });
    saveButton.addEventListener("click", (event) => { event.stopPropagation(); saveToOutput(); });
    resetButton.addEventListener("click", (event) => { event.stopPropagation(); resetToSource(); });
    undoButton.addEventListener("click", (event) => { event.stopPropagation(); doUndo(); });
    redoButton.addEventListener("click", (event) => { event.stopPropagation(); doRedo(); });
    settingsButton.addEventListener("click", (event) => {
        event.stopPropagation();
        toggleSettings();
    });
    fileInput.addEventListener("change", async () => {
        const [selectedFile] = Array.from(fileInput.files || []);
        try {
            await chooseSourceFile(selectedFile);
        } finally {
            fileInput.value = "";
        }
    });
    document.addEventListener("pointerdown", (event) => {
        if (!state.settingsOpen) return;
        if (settings.contains(event.target) || settingsButton.contains(event.target)) return;
        toggleSettings(false);
    });

    // ---------- Drag-and-drop image files onto the node ----------
    function dragHasImage(event) {
        const items = event?.dataTransfer?.items;
        if (items) {
            for (const item of items) {
                if (item?.kind === "file") {
                    if (!item.type || item.type.startsWith("image/")) return true;
                }
            }
        }
        const files = event?.dataTransfer?.files;
        return Boolean(files && files.length > 0);
    }
    function onContainerDragEnter(event) {
        if (state.isProcessing) return;
        if (!dragHasImage(event)) return;
        event.preventDefault();
        event.stopPropagation();
        container.classList.add("is-drag-over");
    }
    function onContainerDragOver(event) {
        if (state.isProcessing) return;
        if (!dragHasImage(event)) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = "copy";
        }
        container.classList.add("is-drag-over");
    }
    function onContainerDragLeave(event) {
        // Only clear when actually leaving the container (not when crossing
        // between children, which fires dragleave on the child).
        if (event.relatedTarget && container.contains(event.relatedTarget)) return;
        container.classList.remove("is-drag-over");
    }
    async function onContainerDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        container.classList.remove("is-drag-over");
        if (state.isProcessing) return;
        const files = Array.from(event.dataTransfer?.files || []);
        const file = files.find((f) => !f.type || f.type.startsWith("image/")) || files[0];
        if (!file) return;
        await chooseSourceFile(file);
    }
    container.addEventListener("dragenter", onContainerDragEnter);
    container.addEventListener("dragover", onContainerDragOver);
    container.addEventListener("dragleave", onContainerDragLeave);
    container.addEventListener("drop", onContainerDrop);

    // ---------- Paste image from clipboard ----------
    function pointerOverContainer() {
        const rect = container.getBoundingClientRect();
        return (
            state.cursorClientX >= rect.left
            && state.cursorClientX <= rect.right
            && state.cursorClientY >= rect.top
            && state.cursorClientY <= rect.bottom
        );
    }
    async function onDocumentPaste(event) {
        if (state.isProcessing) return;
        // Multiple Lama nodes could exist on the graph; only handle paste if
        // the user's mouse is currently hovering over THIS node's container.
        if (!pointerOverContainer()) return;
        const items = Array.from(event.clipboardData?.items || []);
        const imageItem = items.find((item) => item?.type && item.type.startsWith("image/"));
        const file = imageItem?.getAsFile?.();
        if (!file) return;
        event.preventDefault();
        await chooseSourceFile(file);
    }
    document.addEventListener("paste", onDocumentPaste);

    const previousOnResize = node.onResize;
    node.onResize = function onResize() {
        const result = previousOnResize?.apply(this, arguments);
        syncDomSize();
        requestRedraw();
        return result;
    };

    const resizeObserver = new ResizeObserver(() => requestRedraw());
    resizeObserver.observe(container);

    state.sourcePollHandle = window.setInterval(async () => {
        const nextSource = String(getWidgetValue(node, INPUT_SOURCE_PATH, "") || "");
        if (nextSource === state.sourcePath) return;
        state.sourcePath = nextSource;
        state.workingPath = "";
        setWidgetValue(node, INPUT_WORKING_PATH, "");
        clearMask();
        if (!nextSource) {
            state.image = null;
            state.imageWidth = 0;
            state.imageHeight = 0;
            updateMeta();
            requestRedraw();
            return;
        }
        await seedWorkingFile();
    }, SOURCE_POLL_INTERVAL_MS);

    node._tsLamaCleanupCleanup = () => {
        resizeObserver.disconnect();
        if (state.sourcePollHandle) window.clearInterval(state.sourcePollHandle);
        if (state.modelStatusPollHandle) window.clearInterval(state.modelStatusPollHandle);
        document.removeEventListener("paste", onDocumentPaste);
    };

    syncDomSize();
    updateMeta();
    requestRedraw();

    requestAnimationFrame(async () => {
        if (state.workingPath) {
            await refreshImage({ clearMask: true });
            setStatus("Loaded saved working state.", "info");
        } else if (state.sourcePath) {
            await seedWorkingFile();
        } else {
            setStatus("Click “Load Image” to begin.", "info");
        }
    });
}
