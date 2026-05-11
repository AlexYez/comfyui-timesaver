// Shared setup for TS_SAM_MediaLoader. The registerExtension call lives in
// ts-sam-media-loader.js so this module only owns the DOM widget logic.
//
// Patterns lifted from CLAUDE.md §12.5 (reference: TS_LamaCleanup):
//  - addDOMWidget with getMinHeight/getMaxHeight (no widget.computeSize)
//  - V2 Vue parent CSS scale compensation via container.offsetWidth ratio
//  - HTML cursor for sub-frame point hover, not canvas redraw
//  - image padding controlled via state.scale/offsetX/offsetY, canvas full-bleed
//  - suppress default node.imgs preview to avoid Vue rendering an extra <img>

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

export const NODE_NAME = "TS_SAM_MediaLoader";
const ROUTE_BASE = "/ts_sam_media_loader";
const STYLE_ID = "ts-sam-media-loader-styles";
export const DOM_WIDGET_NAME = "ts_sam_media_loader";

const INPUT_MODEL = "model";
const INPUT_SOURCE_PATH = "source_path";
const INPUT_MEDIA_TYPE = "media_type";
const INPUT_COORDS = "coordinates";
const INPUT_NEG_COORDS = "neg_coordinates";
const INPUT_SAM3_CHECKPOINT = "sam3_checkpoint";
const INPUT_MAX_FRAMES = "max_frames";
const INPUT_FRAME_STRIDE = "frame_stride";

const DEFAULT_NODE_SIZE = [560, 480];
const MIN_NODE_WIDTH = 420;
const MIN_NODE_HEIGHT = 300;
const IMAGE_PAD_TOP = 50;
const IMAGE_PAD_BOTTOM = 38;
const IMAGE_PAD_SIDE = 8;
// Image-space pixel distance under which a click on an existing point removes
// it instead of adding a new one.
const HIT_TEST_RADIUS = 12;
const POINT_RADIUS_BASE = 6;
const POINT_RADIUS_HOVER = 9;
// Debounce window between point edits and the SAM3 preview request. Each
// edit cancels the previous timer and arms a new one, mirroring LaMa's
// mouse-up-triggered run.
const PREVIEW_DEBOUNCE_MS = 220;
const SAM3_STATUS_POLL_MS = 1500;
// Walk no further than this many upstream nodes when looking for the
// checkpoint filename. Real-world SAM3 graphs are typically Loader -> Detect
// (one hop), but we leave headroom for short conversion chains.
const MAX_UPSTREAM_WALK = 8;
// Widget names that typically hold the SAM3 checkpoint filename on common
// loader nodes. ComfyUI core's CheckpointLoaderSimple uses ``ckpt_name``;
// UNet/diffusion model loaders use ``unet_name``; many community loaders
// mirror one of those conventions.
const CHECKPOINT_WIDGET_NAMES = ["ckpt_name", "unet_name", "model_name", "checkpoint_name"];
const MEDIA_UPLOAD_ACCEPT = [
    "image/*",
    "video/*",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif",
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg",
].join(",");

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-sml{--tsm-bg:#0e1218;--tsm-text:#e9eef6;--tsm-muted:#9aa6b8;--tsm-accent:#7aa2ff;--tsm-accent-strong:#3a72ff;--tsm-danger:#ef6f6c;--tsm-success:#82d6a8;--tsm-toolbar:rgba(12,16,22,.72);--tsm-toolbar-border:rgba(255,255,255,.08);position:relative;width:100%;height:100%;min-height:0;box-sizing:border-box;color:var(--tsm-text);font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif;background:repeating-conic-gradient(#1a2030 0% 25%,#0f141a 0% 50%) 50%/24px 24px;border:1px solid #28303c;border-radius:10px;overflow:hidden;user-select:none}
.ts-sml__canvas{position:absolute;inset:0;display:block;width:100%;height:100%;cursor:default;touch-action:none}
.ts-sml__canvas.has-image{cursor:crosshair}
.ts-sml__empty{position:absolute;left:8px;right:8px;top:50px;bottom:38px;display:flex;align-items:center;justify-content:center;text-align:center;padding:16px;color:#cdd6e6;font-size:12px;pointer-events:none;background:linear-gradient(180deg,rgba(0,0,0,.45),rgba(0,0,0,.7));border-radius:6px;line-height:1.6}
.ts-sml__overlay{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:rgba(8,12,18,.6);backdrop-filter:blur(2px);color:var(--tsm-text);font-size:12px;pointer-events:none;flex-direction:column;gap:10px;z-index:5}
.ts-sml__overlay.is-active{display:flex}
.ts-sml__spinner{width:28px;height:28px;border-radius:999px;border:3px solid rgba(255,255,255,.14);border-top-color:var(--tsm-accent);animation:tsm-spin .9s linear infinite}
@keyframes tsm-spin{to{transform:rotate(360deg)}}
.ts-sml__toolbar{position:absolute;top:8px;left:8px;right:8px;display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:10px;background:var(--tsm-toolbar);border:1px solid var(--tsm-toolbar-border);backdrop-filter:blur(8px);z-index:6}
.ts-sml__group{display:flex;align-items:center;gap:6px;min-width:0}
.ts-sml__btn{display:inline-flex;align-items:center;gap:5px;border:1px solid rgba(255,255,255,.12);background:rgba(20,26,36,.85);color:var(--tsm-text);border-radius:8px;padding:6px 11px;font-size:11px;cursor:pointer;font-weight:600;letter-spacing:.02em;white-space:nowrap}
.ts-sml__btn:hover{background:rgba(40,54,76,.95)}
.ts-sml__btn[disabled]{opacity:.4;cursor:not-allowed}
.ts-sml__btn--primary{background:linear-gradient(180deg,#7aa2ff,#3a72ff);border-color:#3a72ff;color:#0b1530}
.ts-sml__btn--primary:hover{background:linear-gradient(180deg,#90b6ff,#5180ff)}
.ts-sml__btn--danger{background:rgba(70,30,30,.85);border-color:rgba(239,111,108,.45);color:#ffb4b1}
.ts-sml__btn--danger:hover{background:rgba(99,40,40,.95)}
.ts-sml__counter{display:inline-flex;align-items:center;gap:8px;font-size:11px;color:var(--tsm-text);background:rgba(20,26,36,.85);border:1px solid rgba(255,255,255,.12);border-radius:8px;padding:5px 10px;font-variant-numeric:tabular-nums}
.ts-sml__counter-dot{display:inline-block;width:8px;height:8px;border-radius:50%;flex:0 0 auto}
.ts-sml__counter-dot--pos{background:#42d77c;box-shadow:0 0 0 1px rgba(0,0,0,.4)}
.ts-sml__counter-dot--neg{background:#ef6f6c;box-shadow:0 0 0 1px rgba(0,0,0,.4)}
.ts-sml__statusbar{position:absolute;left:8px;right:8px;bottom:8px;padding:6px 10px;font-size:11px;color:var(--tsm-muted);background:var(--tsm-toolbar);border:1px solid var(--tsm-toolbar-border);border-radius:8px;backdrop-filter:blur(6px);display:flex;justify-content:space-between;gap:10px;align-items:center;pointer-events:none;z-index:4}
.ts-sml__statusbar.is-error{color:var(--tsm-danger)}
.ts-sml__statusbar.is-success{color:var(--tsm-success)}
.ts-sml__statusbar-text{flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ts-sml__statusbar-meta{font-variant-numeric:tabular-nums;color:var(--tsm-muted);font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:65%;flex:0 1 auto}
.ts-sml__hidden-input{position:fixed;left:-9999px;top:-9999px;width:1px;height:1px;opacity:0;pointer-events:none}
.ts-sml.is-drag-over{outline:2px dashed var(--tsm-accent);outline-offset:-4px;outline-style:dashed}
.ts-sml__drop-hint{position:absolute;inset:8px;display:none;align-items:center;justify-content:center;border:2px dashed var(--tsm-accent);border-radius:10px;background:rgba(122,162,255,.08);color:var(--tsm-text);font-size:13px;font-weight:600;pointer-events:none;z-index:8;text-shadow:0 1px 2px rgba(0,0,0,.6)}
.ts-sml.is-drag-over .ts-sml__drop-hint{display:flex}
.ts-sml__preview-pill{display:inline-flex;align-items:center;gap:6px;font-size:10px;color:var(--tsm-muted);background:rgba(20,26,36,.85);border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:4px 8px;white-space:nowrap}
.ts-sml__preview-pill.is-loading{color:#cbd5ff;border-color:rgba(122,162,255,.55)}
.ts-sml__preview-pill.is-ready{color:#9fe3c2;border-color:rgba(130,214,168,.55)}
.ts-sml__preview-pill.is-error{color:#ffb4b1;border-color:rgba(239,111,108,.55)}
.ts-sml__preview-dot{width:6px;height:6px;border-radius:50%;background:#9aa6b8;flex:0 0 auto}
.ts-sml__preview-pill.is-loading .ts-sml__preview-dot{background:#7aa2ff;animation:tsm-blink 1s infinite}
.ts-sml__preview-pill.is-ready .ts-sml__preview-dot{background:#82d6a8}
.ts-sml__preview-pill.is-error .ts-sml__preview-dot{background:#ef6f6c}
@keyframes tsm-blink{0%,100%{opacity:1}50%{opacity:.3}}
`;
    document.head.appendChild(style);
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
function suppressDefaultImagePreview(node) {
    // ComfyUI core adds an <img> preview under nodes whose backend emits a
    // ``ui.images`` payload (which our ``IMAGE`` output does not, but the
    // upload combo widget on some core versions does). Make ``node.imgs``
    // permanently empty so V2 Vue does not render a duplicate preview below
    // our custom canvas.
    try { delete node.imgs; } catch {}
    try {
        Object.defineProperty(node, "imgs", {
            configurable: true,
            enumerable: true,
            get() { return []; },
            set() { /* swallow */ },
        });
    } catch (error) {
        console.warn("[TS SAM Media Loader] Failed to suppress default image preview:", error);
    }
    try { node.imageIndex = null; } catch {}
}

function scheduleCanvasDirty() {
    app?.graph?.setDirtyCanvas?.(true, true);
}

function formatBytes(bytes) {
    if (!bytes || bytes <= 0) return "";
    const units = ["B", "KB", "MB", "GB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit += 1;
    }
    return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`;
}

function buildMetaLine(state) {
    const parts = [];
    if (state.sourcePath) {
        const filename = state.sourcePath
            .split(/[\\/]/)
            .pop()
            .replace(/\s\[input\]$/i, "");
        if (filename) parts.push(filename);
    }
    if (state.imageWidth && state.imageHeight) {
        parts.push(`${state.imageWidth}×${state.imageHeight}`);
    }
    if (state.mediaType === "video") {
        if (state.frameCount > 0) parts.push(`${state.frameCount} frames`);
        if (state.fps > 0) parts.push(`${state.fps.toFixed(2)} fps`);
    } else if (state.mediaType === "image") {
        parts.push("image");
    }
    return parts.join(" · ");
}

function dataUrlForBase64(b64) {
    if (!b64) return "";
    return `data:image/jpeg;base64,${b64}`;
}

export function setupSamMediaLoader(node) {
    if (!node || typeof node.addDOMWidget !== "function") return;
    if (typeof node._tsSamMediaLoaderCleanup === "function") {
        try { node._tsSamMediaLoaderCleanup(); } catch {}
    }
    removeDomWidget(node);
    ensureStyles();
    suppressDefaultImagePreview(node);

    hideWidget(node, INPUT_SOURCE_PATH);
    hideWidget(node, INPUT_MEDIA_TYPE);
    hideWidget(node, INPUT_COORDS);
    hideWidget(node, INPUT_NEG_COORDS);
    hideWidget(node, INPUT_SAM3_CHECKPOINT);
    hideWidget(node, INPUT_MAX_FRAMES);
    hideWidget(node, INPUT_FRAME_STRIDE);

    node.resizable = true;
    node.size = [
        Math.max(Number(node.size?.[0]) || DEFAULT_NODE_SIZE[0], MIN_NODE_WIDTH),
        Math.max(Number(node.size?.[1]) || DEFAULT_NODE_SIZE[1], MIN_NODE_HEIGHT),
    ];
    node.min_size = [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];

    const state = {
        sourcePath: String(getWidgetValue(node, INPUT_SOURCE_PATH, "") || ""),
        mediaType: String(getWidgetValue(node, INPUT_MEDIA_TYPE, "") || ""),
        positivePoints: parsePointsJson(getWidgetValue(node, INPUT_COORDS, "[]")),
        negativePoints: parsePointsJson(getWidgetValue(node, INPUT_NEG_COORDS, "[]")),
        image: null,
        imageWidth: 0,
        imageHeight: 0,
        frameCount: 0,
        fps: 0,
        scale: 1,
        offsetX: 0,
        offsetY: 0,
        statusText: "",
        statusKind: "info",
        isUploading: false,
        hoverPoint: null,
        cursorClientX: 0,
        cursorClientY: 0,
        // Cleared on successful upload so we don't fight backend-detected size.
        pendingProbe: false,
        // SAM3 preview overlay state.
        checkpointName: String(getWidgetValue(node, INPUT_SAM3_CHECKPOINT, "") || ""),
        previewState: "idle",   // idle | loading | ready | error | disconnected
        previewMessage: "",
        previewMask: null,       // HTMLImageElement (grayscale PNG)
        previewMaskKey: "",      // signature of the mask currently displayed
        previewDebounceHandle: null,
        previewStatusPollHandle: null,
        previewRequestId: 0,     // monotonic — old responses ignored
        lastRequestSignature: "",
    };

    const container = document.createElement("div");
    container.className = "ts-sml";

    const canvas = document.createElement("canvas");
    canvas.className = "ts-sml__canvas";

    const empty = document.createElement("div");
    empty.className = "ts-sml__empty";
    empty.innerHTML = `
        <div>
            <div style="font-weight:600;font-size:13px;margin-bottom:8px;">Click "Load Image/Video"</div>
            <div style="color:#9aa6b8;">Left-click — positive point (green) · Shift / Right-click — negative point (red)</div>
            <div style="color:#9aa6b8;margin-top:4px;">Click on an existing point to remove it.</div>
        </div>
    `;

    const overlay = document.createElement("div");
    overlay.className = "ts-sml__overlay";
    const spinner = document.createElement("div");
    spinner.className = "ts-sml__spinner";
    const overlayLabel = document.createElement("div");
    overlayLabel.textContent = "Uploading...";
    overlay.append(spinner, overlayLabel);

    // Toolbar
    const toolbar = document.createElement("div");
    toolbar.className = "ts-sml__toolbar";

    const leftGroup = document.createElement("div");
    leftGroup.className = "ts-sml__group";

    const loadButton = document.createElement("button");
    loadButton.className = "ts-sml__btn ts-sml__btn--primary";
    loadButton.textContent = "Load Image/Video";

    const clearButton = document.createElement("button");
    clearButton.className = "ts-sml__btn ts-sml__btn--danger";
    clearButton.textContent = "Clear Points";
    clearButton.title = "Remove every positive and negative point.";
    clearButton.disabled = true;

    leftGroup.append(loadButton, clearButton);

    const counterGroup = document.createElement("div");
    counterGroup.className = "ts-sml__counter";
    const counterPosDot = document.createElement("span");
    counterPosDot.className = "ts-sml__counter-dot ts-sml__counter-dot--pos";
    const counterPos = document.createElement("span");
    counterPos.textContent = "0";
    const counterSep = document.createElement("span");
    counterSep.style.color = "#48526a";
    counterSep.textContent = "|";
    const counterNegDot = document.createElement("span");
    counterNegDot.className = "ts-sml__counter-dot ts-sml__counter-dot--neg";
    const counterNeg = document.createElement("span");
    counterNeg.textContent = "0";
    counterGroup.append(counterPosDot, counterPos, counterSep, counterNegDot, counterNeg);

    // Flex spacer pushes the counter + preview pill to the right edge while
    // the load/clear buttons stay on the left. Filename moved to the
    // status bar (bottom) so a long name no longer overflows the toolbar.
    const toolbarSpacer = document.createElement("div");
    toolbarSpacer.style.cssText = "flex:1 1 auto;min-width:0;";

    // Preview state pill (right of the counter): "Connect SAM3 model",
    // "Loading model...", "SAM3 preview", "Mask ready".
    const previewPill = document.createElement("div");
    previewPill.className = "ts-sml__preview-pill";
    const previewPillDot = document.createElement("span");
    previewPillDot.className = "ts-sml__preview-dot";
    const previewPillText = document.createElement("span");
    previewPillText.textContent = "Connect SAM3 model";
    previewPill.append(previewPillDot, previewPillText);
    previewPill.title = "Connect a SAM3 checkpoint loader to the 'model' input to see the live mask overlay.";

    toolbar.append(leftGroup, toolbarSpacer, counterGroup, previewPill);

    // Status bar
    const statusBar = document.createElement("div");
    statusBar.className = "ts-sml__statusbar";
    const statusText = document.createElement("div");
    statusText.className = "ts-sml__statusbar-text";
    statusText.textContent = "Load an image or a video to start placing SAM3 points.";
    const statusMeta = document.createElement("div");
    statusMeta.className = "ts-sml__statusbar-meta";
    statusBar.append(statusText, statusMeta);

    // Hidden file input
    const fileInput = document.createElement("input");
    fileInput.className = "ts-sml__hidden-input";
    fileInput.type = "file";
    fileInput.accept = MEDIA_UPLOAD_ACCEPT;

    const dropHint = document.createElement("div");
    dropHint.className = "ts-sml__drop-hint";
    dropHint.textContent = "Drop image or video to load";

    container.append(canvas, empty, overlay, toolbar, statusBar, fileInput, dropHint);

    stopPropagation(container, [
        "pointerdown", "pointerup", "pointermove",
        "mousedown", "mouseup", "mousemove",
        "wheel", "click", "dblclick", "contextmenu",
    ]);

    // V2-safe: getMinHeight/getMaxHeight only, never widget.computeSize.
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

    function setOverlay(active, label = "Uploading...") {
        overlay.classList.toggle("is-active", Boolean(active));
        overlayLabel.textContent = label;
    }

    function updateCounter() {
        counterPos.textContent = String(state.positivePoints.length);
        counterNeg.textContent = String(state.negativePoints.length);
        clearButton.disabled = state.positivePoints.length === 0 && state.negativePoints.length === 0;
    }

    function updateMeta() {
        const meta = buildMetaLine(state);
        statusMeta.textContent = meta;
        statusMeta.title = meta;
        empty.style.display = state.image ? "none" : "flex";
        canvas.classList.toggle("has-image", Boolean(state.image));
        loadButton.disabled = state.isUploading;
    }

    function persistPoints() {
        setWidgetValue(node, INPUT_COORDS, JSON.stringify(state.positivePoints));
        setWidgetValue(node, INPUT_NEG_COORDS, JSON.stringify(state.negativePoints));
        updateCounter();
        schedulePreview();
    }

    function setPreviewPill(kind, text) {
        state.previewState = kind;
        state.previewMessage = text || "";
        previewPill.classList.toggle("is-loading", kind === "loading");
        previewPill.classList.toggle("is-ready", kind === "ready");
        previewPill.classList.toggle("is-error", kind === "error");
        previewPillText.textContent = text || "";
        previewPill.title = text || "";
    }

    function findCheckpointNameUpstream(startNode, visited, depth) {
        if (!startNode || depth > MAX_UPSTREAM_WALK) return "";
        const id = startNode.id ?? startNode.uuid ?? Symbol.for("anon");
        if (visited.has(id)) return "";
        visited.add(id);
        // Look for a checkpoint widget on this node first.
        for (const widget of startNode.widgets || []) {
            if (!widget || typeof widget.value !== "string") continue;
            if (CHECKPOINT_WIDGET_NAMES.includes(widget.name)) {
                if (widget.value.trim()) return widget.value;
            }
        }
        // Otherwise walk upstream through this node's MODEL-like inputs.
        for (const input of startNode.inputs || []) {
            if (input.link == null) continue;
            const link = startNode.graph?.links?.[input.link];
            const src = link ? startNode.graph?.getNodeById(link.origin_id) : null;
            if (src) {
                const found = findCheckpointNameUpstream(src, visited, depth + 1);
                if (found) return found;
            }
        }
        return "";
    }

    function detectConnectedCheckpoint() {
        // Find the "model" socket input on our own node and walk upstream
        // until we hit a loader-shaped widget. If unconnected we get "".
        const modelInput = node.inputs?.find((i) => i?.name === INPUT_MODEL);
        const linkId = modelInput?.link;
        if (linkId == null) return "";
        const link = node.graph?.links?.[linkId];
        if (!link) return "";
        const src = node.graph?.getNodeById(link.origin_id);
        if (!src) return "";
        return findCheckpointNameUpstream(src, new Set(), 0);
    }

    function refreshCheckpointDetection() {
        const next = detectConnectedCheckpoint();
        if (next === state.checkpointName) return false;
        state.checkpointName = next;
        setWidgetValue(node, INPUT_SAM3_CHECKPOINT, next || "");
        return true;
    }

    function buildPreviewSignature() {
        return JSON.stringify({
            ckpt: state.checkpointName,
            src: state.sourcePath,
            pos: state.positivePoints,
            neg: state.negativePoints,
        });
    }

    function clearPreviewMask() {
        state.previewMask = null;
        state.previewMaskKey = "";
        requestRedraw();
    }

    function schedulePreview() {
        if (state.previewDebounceHandle) {
            window.clearTimeout(state.previewDebounceHandle);
            state.previewDebounceHandle = null;
        }
        state.previewDebounceHandle = window.setTimeout(() => {
            state.previewDebounceHandle = null;
            runPreview();
        }, PREVIEW_DEBOUNCE_MS);
    }

    async function runPreview() {
        const hasPoints = state.positivePoints.length > 0 || state.negativePoints.length > 0;
        if (!hasPoints || !state.image || !state.sourcePath) {
            clearPreviewMask();
            if (!state.checkpointName) {
                setPreviewPill("disconnected", "Connect SAM3 model");
            } else {
                setPreviewPill("idle", "Add points to preview");
            }
            return;
        }
        refreshCheckpointDetection();
        if (!state.checkpointName) {
            clearPreviewMask();
            setPreviewPill("disconnected", "Connect SAM3 model");
            return;
        }
        const signature = buildPreviewSignature();
        if (signature === state.lastRequestSignature && state.previewMask) return;
        state.lastRequestSignature = signature;
        const requestId = ++state.previewRequestId;
        setPreviewPill("loading", "Running SAM3...");

        try {
            const response = await api.fetchApi(`${ROUTE_BASE}/preview_mask`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    source_path: state.sourcePath,
                    checkpoint_name: state.checkpointName,
                    positive: state.positivePoints,
                    negative: state.negativePoints,
                    refine_iterations: 2,
                }),
            });
            const payload = await response.json().catch(() => ({}));
            if (requestId !== state.previewRequestId) return; // stale
            if (!response.ok || !payload?.ok) {
                if (payload?.needs_model) {
                    setPreviewPill("disconnected", "Connect SAM3 model");
                    clearPreviewMask();
                    return;
                }
                setPreviewPill("error", payload?.error || "Preview failed");
                return;
            }
            if (!payload.mask_b64) {
                clearPreviewMask();
                setPreviewPill("ready", "No mask");
                return;
            }
            const img = await loadImageElement(`data:image/png;base64,${payload.mask_b64}`);
            if (requestId !== state.previewRequestId) return;
            state.previewMask = img;
            state.previewMaskKey = `${requestId}:${state.positivePoints.length}p${state.negativePoints.length}n`;
            setPreviewPill("ready", "SAM3 preview");
            requestRedraw();
        } catch (error) {
            if (requestId !== state.previewRequestId) return;
            setPreviewPill("error", error?.message || "Preview failed");
        }
    }

    async function pollSam3Status() {
        try {
            const response = await api.fetchApi(`${ROUTE_BASE}/sam3_status`);
            const status = await response.json().catch(() => ({}));
            if (status?.loading) {
                setPreviewPill("loading", status?.message || "Loading SAM3 model...");
            }
        } catch {
            // Best-effort; the pill just stays as is until the next call.
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
        }
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
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!state.image || !state.imageWidth || !state.imageHeight) {
            return;
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const drawWidth = state.imageWidth * state.scale;
        const drawHeight = state.imageHeight * state.scale;
        ctx.drawImage(state.image, state.offsetX, state.offsetY, drawWidth, drawHeight);
        // Backend ships the mask as a blue-tinted RGBA PNG where alpha == mask.
        // We just blit it on top of the image with a global alpha to soften
        // the tint — no compositing tricks are needed.
        if (state.previewMask) {
            ctx.save();
            ctx.globalAlpha = 0.55;
            ctx.drawImage(state.previewMask, state.offsetX, state.offsetY, drawWidth, drawHeight);
            ctx.restore();
        }
        drawPoints(ctx);
    }

    function drawPoints(ctx) {
        const drawList = [
            { points: state.positivePoints, fill: "rgba(66,215,124,1)", stroke: "rgba(20,80,40,1)" },
            { points: state.negativePoints, fill: "rgba(239,111,108,1)", stroke: "rgba(80,20,20,1)" },
        ];
        for (const entry of drawList) {
            for (const point of entry.points) {
                const cx = state.offsetX + point.x * state.scale;
                const cy = state.offsetY + point.y * state.scale;
                const isHover = state.hoverPoint && state.hoverPoint.ref === point;
                const radius = isHover ? POINT_RADIUS_HOVER : POINT_RADIUS_BASE;
                ctx.save();
                ctx.lineWidth = 2;
                ctx.strokeStyle = "rgba(255,255,255,0.95)";
                ctx.fillStyle = entry.fill;
                ctx.beginPath();
                ctx.arc(cx, cy, radius, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
                ctx.lineWidth = 1;
                ctx.strokeStyle = entry.stroke;
                ctx.beginPath();
                ctx.arc(cx, cy, radius, 0, Math.PI * 2);
                ctx.stroke();
                ctx.restore();
            }
        }
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
        // rect is in viewport (post-transform) pixels; canvas CSS pixels match
        // because canvas is full-bleed inside container (per CLAUDE.md §12.5.5).
        const xInCanvas = event.clientX - rect.left;
        const yInCanvas = event.clientY - rect.top;
        const scale = state.scale > 0 ? state.scale : 1;
        const imageX = (xInCanvas - state.offsetX) / scale;
        const imageY = (yInCanvas - state.offsetY) / scale;
        const withinImage = imageX >= 0 && imageY >= 0
            && imageX < state.imageWidth && imageY < state.imageHeight;
        return {
            imageX: clamp(imageX, 0, Math.max(0, state.imageWidth - 1)),
            imageY: clamp(imageY, 0, Math.max(0, state.imageHeight - 1)),
            withinImage,
        };
    }

    function findPointAt(imageX, imageY) {
        const scale = state.scale > 0 ? state.scale : 1;
        // HIT_TEST_RADIUS is in CSS pixels at display scale; convert to image
        // pixels so dense images still allow a click target.
        const hitRadiusImage = HIT_TEST_RADIUS / scale;
        const candidates = [
            { kind: "positive", points: state.positivePoints },
            { kind: "negative", points: state.negativePoints },
        ];
        let best = null;
        let bestDist = hitRadiusImage * hitRadiusImage;
        for (const entry of candidates) {
            for (const point of entry.points) {
                const dx = point.x - imageX;
                const dy = point.y - imageY;
                const distSq = dx * dx + dy * dy;
                if (distSq <= bestDist) {
                    bestDist = distSq;
                    best = { kind: entry.kind, ref: point };
                }
            }
        }
        return best;
    }

    function onPointerDown(event) {
        if (state.isUploading) return;
        if (!state.image) return;
        if (event.button !== 0 && event.button !== 2) return;
        event.preventDefault();
        const coords = pointerToImageCoords(event);
        if (!coords.withinImage) return;
        const isNegative = event.button === 2 || event.shiftKey;
        const existing = findPointAt(coords.imageX, coords.imageY);
        if (existing) {
            const list = existing.kind === "positive"
                ? state.positivePoints
                : state.negativePoints;
            const index = list.indexOf(existing.ref);
            if (index >= 0) list.splice(index, 1);
            state.hoverPoint = null;
        } else {
            const point = { x: coords.imageX, y: coords.imageY };
            if (isNegative) {
                state.negativePoints.push(point);
            } else {
                state.positivePoints.push(point);
            }
        }
        persistPoints();
        requestRedraw();
    }

    function onPointerMove(event) {
        state.cursorClientX = event.clientX;
        state.cursorClientY = event.clientY;
        if (!state.image) return;
        const coords = pointerToImageCoords(event);
        if (!coords.withinImage) {
            if (state.hoverPoint) {
                state.hoverPoint = null;
                requestRedraw();
            }
            return;
        }
        const hover = findPointAt(coords.imageX, coords.imageY);
        if ((hover?.ref || null) !== (state.hoverPoint?.ref || null)) {
            state.hoverPoint = hover;
            requestRedraw();
        }
    }

    function onPointerLeave() {
        if (state.hoverPoint) {
            state.hoverPoint = null;
            requestRedraw();
        }
    }

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerleave", onPointerLeave);
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());

    loadButton.addEventListener("click", (event) => {
        event.stopPropagation();
        try {
            fileInput.click();
        } catch (error) {
            console.error("[TS SAM Media Loader] fileInput.click failed:", error);
            setStatus(`Failed to open file picker: ${error?.message || error}`, "error");
        }
    });

    clearButton.addEventListener("click", (event) => {
        event.stopPropagation();
        if (!state.positivePoints.length && !state.negativePoints.length) return;
        state.positivePoints = [];
        state.negativePoints = [];
        state.hoverPoint = null;
        // Clear preview ahead of persistPoints so the redraw triggered by the
        // change does not still show the stale mask for a few frames.
        clearPreviewMask();
        state.lastRequestSignature = "";
        persistPoints();
        requestRedraw();
        setStatus("Cleared all points.", "info");
    });

    fileInput.addEventListener("change", async () => {
        const [selectedFile] = Array.from(fileInput.files || []);
        try {
            await chooseSourceFile(selectedFile);
        } finally {
            fileInput.value = "";
        }
    });

    function dragHasMedia(event) {
        const items = event?.dataTransfer?.items;
        if (items) {
            for (const item of items) {
                if (item?.kind === "file") {
                    if (!item.type
                        || item.type.startsWith("image/")
                        || item.type.startsWith("video/")) return true;
                }
            }
        }
        const files = event?.dataTransfer?.files;
        return Boolean(files && files.length > 0);
    }
    function onContainerDragEnter(event) {
        if (state.isUploading) return;
        if (!dragHasMedia(event)) return;
        event.preventDefault();
        event.stopPropagation();
        container.classList.add("is-drag-over");
    }
    function onContainerDragOver(event) {
        if (state.isUploading) return;
        if (!dragHasMedia(event)) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = "copy";
        }
        container.classList.add("is-drag-over");
    }
    function onContainerDragLeave(event) {
        if (event.relatedTarget && container.contains(event.relatedTarget)) return;
        container.classList.remove("is-drag-over");
    }
    async function onContainerDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        container.classList.remove("is-drag-over");
        if (state.isUploading) return;
        const files = Array.from(event.dataTransfer?.files || []);
        const file = files.find((f) => !f.type || f.type.startsWith("image/") || f.type.startsWith("video/")) || files[0];
        if (!file) return;
        await chooseSourceFile(file);
    }
    container.addEventListener("dragenter", onContainerDragEnter);
    container.addEventListener("dragover", onContainerDragOver);
    container.addEventListener("dragleave", onContainerDragLeave);
    container.addEventListener("drop", onContainerDrop);

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
        if (state.isUploading) return;
        if (!pointerOverContainer()) return;
        const items = Array.from(event.clipboardData?.items || []);
        const mediaItem = items.find((item) => item?.type
            && (item.type.startsWith("image/") || item.type.startsWith("video/")));
        const file = mediaItem?.getAsFile?.();
        if (!file) return;
        event.preventDefault();
        await chooseSourceFile(file);
    }
    document.addEventListener("paste", onDocumentPaste);

    async function loadImageElement(src) {
        if (!src) return null;
        const image = new Image();
        image.src = src;
        if (typeof image.decode === "function") {
            try {
                await image.decode();
                return image;
            } catch { /* fall through */ }
        }
        await new Promise((resolve, reject) => {
            if (image.complete && image.naturalWidth > 0) {
                resolve();
                return;
            }
            image.onload = () => resolve();
            image.onerror = () => reject(new Error("Failed to decode preview frame."));
        });
        return image;
    }

    function applyPreviewPayload(payload) {
        state.sourcePath = String(payload?.source_path || state.sourcePath || "");
        state.mediaType = String(payload?.media_type || state.mediaType || "");
        state.imageWidth = Math.max(0, Number(payload?.width || 0));
        state.imageHeight = Math.max(0, Number(payload?.height || 0));
        state.frameCount = Math.max(0, Number(payload?.frame_count || 0));
        state.fps = Math.max(0, Number(payload?.fps || 0));
        setWidgetValue(node, INPUT_SOURCE_PATH, state.sourcePath);
        setWidgetValue(node, INPUT_MEDIA_TYPE, state.mediaType);
    }

    async function applyFirstFrame(b64) {
        const dataUrl = dataUrlForBase64(b64);
        if (!dataUrl) {
            state.image = null;
            requestRedraw();
            return;
        }
        try {
            const img = await loadImageElement(dataUrl);
            if (!img) {
                state.image = null;
            } else {
                state.image = img;
                if (!state.imageWidth || !state.imageHeight) {
                    state.imageWidth = img.naturalWidth || img.width || 0;
                    state.imageHeight = img.naturalHeight || img.height || 0;
                }
            }
        } catch (error) {
            console.warn("[TS SAM Media Loader] Failed to decode preview:", error);
            state.image = null;
        }
        updateMeta();
        requestRedraw();
    }

    async function chooseSourceFile(file) {
        if (!file) return;
        if (state.isUploading) return;
        state.isUploading = true;
        setOverlay(true, "Uploading...");
        setStatus(`Uploading ${file.name}...`, "info");
        updateMeta();
        try {
            const form = new FormData();
            form.append("file", file, file.name);
            const response = await api.fetchApi(`${ROUTE_BASE}/upload`, {
                method: "POST",
                body: form,
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || !payload?.ok) {
                throw new Error(payload?.error || "Upload failed.");
            }
            // Drop previous points — they were tied to the old image dimensions.
            state.positivePoints = [];
            state.negativePoints = [];
            state.hoverPoint = null;
            persistPoints();
            applyPreviewPayload(payload);
            await applyFirstFrame(payload.first_frame_b64);
            const sizeTag = formatBytes(payload.size_bytes || 0);
            const sizeNote = sizeTag ? ` (${sizeTag})` : "";
            const noun = state.mediaType === "video" ? "video" : "image";
            setStatus(`Loaded ${noun}${sizeNote}. Left-click — positive, Shift / right-click — negative.`, "success");
        } catch (error) {
            setStatus(error?.message || "Upload failed.", "error");
        } finally {
            state.isUploading = false;
            setOverlay(false);
            updateMeta();
        }
    }

    async function probeExistingSource() {
        if (!state.sourcePath) return;
        if (state.image) return;
        if (state.pendingProbe) return;
        state.pendingProbe = true;
        try {
            const response = await api.fetchApi(`${ROUTE_BASE}/probe`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source_path: state.sourcePath }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || !payload?.ok) {
                setStatus(payload?.error || "Could not restore previous media.", "error");
                return;
            }
            applyPreviewPayload(payload);
            await applyFirstFrame(payload.first_frame_b64);
            setStatus("Restored media from workflow.", "info");
        } catch (error) {
            setStatus(error?.message || "Could not restore previous media.", "error");
        } finally {
            state.pendingProbe = false;
        }
    }

    const previousOnResize = node.onResize;
    node.onResize = function onResize() {
        const result = previousOnResize?.apply(this, arguments);
        syncDomSize();
        requestRedraw();
        return result;
    };

    // LiteGraph fires onConnectionsChange when the user (dis)connects a link
    // to/from this node. We use it to refresh the detected checkpoint name
    // and re-run the preview when MODEL changes.
    const previousOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function onConnectionsChangeWrapper(
        type, slotIndex, isConnected, linkInfo, ioSlot
    ) {
        const result = previousOnConnectionsChange?.apply(this, arguments);
        // type === 1 (LiteGraph.INPUT) — we only care about input changes.
        if (type === 1) {
            const inputName = ioSlot?.name || node.inputs?.[slotIndex]?.name;
            if (inputName === INPUT_MODEL) {
                if (refreshCheckpointDetection()) {
                    state.lastRequestSignature = "";
                    // Trigger a fresh preview (or clear/disconnect message)
                    // once the link has actually been registered in graph.
                    schedulePreview();
                }
            }
        }
        return result;
    };

    const resizeObserver = new ResizeObserver(() => requestRedraw());
    resizeObserver.observe(container);

    // Some loader chains rename their checkpoint widget asynchronously after
    // the link change fires (e.g. after a "convert widget to input" flow).
    // A low-rate poller catches those late updates without blocking the UI.
    state.previewStatusPollHandle = window.setInterval(() => {
        if (refreshCheckpointDetection()) {
            state.lastRequestSignature = "";
            schedulePreview();
        } else if (state.previewState === "loading") {
            pollSam3Status();
        }
    }, SAM3_STATUS_POLL_MS);

    node._tsSamMediaLoaderCleanup = () => {
        resizeObserver.disconnect();
        document.removeEventListener("paste", onDocumentPaste);
        if (state.previewDebounceHandle) {
            window.clearTimeout(state.previewDebounceHandle);
            state.previewDebounceHandle = null;
        }
        if (state.previewStatusPollHandle) {
            window.clearInterval(state.previewStatusPollHandle);
            state.previewStatusPollHandle = null;
        }
        // Bump the request id so any in-flight /preview_mask response is
        // ignored once the node is gone.
        state.previewRequestId += 1;
    };

    const prevOnRemoved = node.onRemoved;
    node.onRemoved = function onRemovedWithCleanup() {
        try {
            node._tsSamMediaLoaderCleanup?.();
        } catch (err) {
            console.warn("[TS SAM Media Loader] cleanup on removal failed", err);
        }
        return prevOnRemoved?.apply(this, arguments);
    };

    syncDomSize();
    updateCounter();
    updateMeta();
    // Initial pill state. Graph may not be fully wired yet at this point;
    // onConnectionsChange / the poller will refine it once links resolve.
    refreshCheckpointDetection();
    setPreviewPill(
        state.checkpointName ? "idle" : "disconnected",
        state.checkpointName ? "Add points to preview" : "Connect SAM3 model",
    );
    requestRedraw();

    // Defer the restore probe so the layout settles first.
    requestAnimationFrame(async () => {
        if (state.sourcePath) {
            await probeExistingSource();
        }
        // Re-detect once layout/links are stable, then trigger a preview if
        // saved points were restored from the workflow.
        if (refreshCheckpointDetection()
            || state.positivePoints.length > 0
            || state.negativePoints.length > 0) {
            schedulePreview();
        }
    });
}

function parsePointsJson(raw) {
    if (typeof raw !== "string" || !raw.trim()) return [];
    try {
        const data = JSON.parse(raw);
        if (!Array.isArray(data)) return [];
        const out = [];
        for (const item of data) {
            if (item && typeof item.x === "number" && typeof item.y === "number") {
                out.push({ x: item.x, y: item.y });
            }
        }
        return out;
    } catch {
        return [];
    }
}
