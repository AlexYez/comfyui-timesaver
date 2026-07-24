// Shared setup for TS_LamaCleanup. The registerExtension call lives in
// ts-lama-cleanup.js so this module only owns the DOM widget logic.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import { TS_UI_CLASS, createOpenInterfaceButton, ensureThemeStyles, pickLocaleStrings } from "../../_theme.js";
import { hideWidget as sharedHideWidget } from "../../_dom_widget.js";
import { openFullscreenOverlay } from "../../_fullscreen.js";

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
// The node body is now a compact shell (preview + the launcher button); the painting
// UI lives in a fullscreen overlay, so the node no longer needs to be large.
// Existing workflows keep their serialised size (we only clamp upward to MIN).
const DEFAULT_NODE_SIZE = [320, 300];
const MIN_NODE_WIDTH = 240;
const MIN_NODE_HEIGHT = 160;
const TITLE_BAR_HEIGHT = 30;
const STATUS_POLL_INTERVAL_MS = 1500;
const SOURCE_POLL_INTERVAL_MS = 300;
// Pixel margins inside the canvas reserved for the floating toolbar / status
// bar overlays. The image fit-letterboxes inside the area minus these so it
// never sits underneath the controls.
const IMAGE_PAD_TOP = 56;
const IMAGE_PAD_BOTTOM = 44;
const IMAGE_PAD_SIDE = 8;
// User zoom (mouse-wheel on the image). zoomLevel = 1 == "fit-letterbox",
// zoomLevel > 1 zooms in past fit, < 1 zooms out below fit (we clamp to 1.0
// so the image never appears smaller than fit). Pan is offset added on top
// of the centred letterbox position and is also clamped so the image
// cannot be scrolled completely out of view.
const MIN_ZOOM_LEVEL = 1.0;
const MAX_ZOOM_LEVEL = 8.0;
const ZOOM_STEP = 1.15;
// How many viewport-pixels of the image must remain inside the usable area
// at all pan extremes. Keeps the picture findable when user pans aggressively.
const PAN_SLACK_PX = 120;
// Brush slider uses a log-scaled mapping (slider 0..100 → image-px 1..400)
// so small brushes get more granular control where it matters most.
const BRUSH_MIN_PX = 1;
const BRUSH_MAX_PX = 400;
const BRUSH_LOG_MIN = Math.log(BRUSH_MIN_PX);
const BRUSH_LOG_MAX = Math.log(BRUSH_MAX_PX);
// Cap how many edits the Undo stack remembers. Older entries are evicted FIFO
// and their backing temp files are removed via /cleanup_paths so disk usage
// stays bounded.
const MAX_HISTORY = 30;
// Painted-mask tint used for on-screen feedback only (the mask sent to the
// backend is pure white). A desaturated accent-family violet reads clearly as
// "selected region" over arbitrary imagery without the blue cast the old
// #080c12 tint gave. Kept as a literal because it is drawn to a canvas, where
// CSS variables cannot be used.
const MASK_TINT = "rgba(84, 74, 112, 1)";
const MEDIA_UPLOAD_ACCEPT = ["image/*", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"].join(",");

// User-visible UI strings. Resolved via pickLocaleStrings inside
// setupLamaCleanup (per-key ru→en fallback; locale switch reloads the page,
// so resolving once per setup call is correct). Log messages (console.*)
// intentionally stay English per project convention.
const STRINGS = {
    en: {
        clickToBegin: "Click “Load Image” to begin.",
        processing: "Processing...",
        load: "Load Image",
        save: "Save Image",
        saveTitle: "Save the current cleaned image into the ComfyUI output folder.",
        reset: "Reset",
        resetTitle: "Discard local edits and restart from the loaded image.",
        undoTitle: "Undo last edit",
        redoTitle: "Redo edit",
        brush: "Brush",
        fit: "Fit",
        fitTitle: "Fit image to view (resets zoom and pan).",
        oneToOne: "1:1",
        oneToOneTitle: "Show image at 1:1 (one image pixel per screen pixel).",
        settingsTitle: "Advanced settings",
        closeTitle: "Close editor (Esc)",
        advanced: "Advanced",
        maxResolutionField: "Max LaMa resolution",
        maskPaddingField: "Mask context padding",
        featherField: "Composite feather",
        dropHint: "Drop image to load",
        openEditorTitle: "Open the fullscreen editor",
        noImageYet: "No image yet — drop or paste one here.",
        imageFallback: "image",
        stepTag: (index, total) => ` • step ${index}/${total}`,
        undone: "Reverted previous edit.",
        redone: "Restored next edit.",
        failedLoadImage: "Failed to load image.",
        failedPreview: (message) => `Failed to load image preview: ${message}`,
        uploadFailed: "Upload failed.",
        uploadingImage: "Uploading image...",
        uploading: "Uploading...",
        failedWorkingCopy: "Failed to prepare working copy.",
        imageLoaded: "Image loaded. Paint defects with the brush.",
        loadingModel: "Loading model...",
        inpainting: "Inpainting...",
        sendingRegion: "Sending region to LaMa...",
        inpaintFailed: "Inpaint request failed.",
        cleanupApplied: "Cleanup applied. Paint another area or press Save.",
        nothingToSave: "Nothing to save yet.",
        saveFailed: "Save failed.",
        savedTo: (name) => `Saved to output: ${name}`,
        unknownFile: "unknown",
        noSource: "No source image to reset to.",
        filePickerFailed: (message) => `Failed to open file picker: ${message}`,
        loadedSaved: "Loaded saved working state.",
        ready: "Ready — open the interface to load an image.",
    },
    ru: {
        clickToBegin: "Нажмите «Загрузить изображение», чтобы начать.",
        processing: "Обработка...",
        load: "Загрузить изображение",
        save: "Сохранить изображение",
        saveTitle: "Сохранить текущее очищенное изображение в папку output ComfyUI.",
        reset: "Сброс",
        resetTitle: "Отменить локальные правки и начать заново с загруженного изображения.",
        undoTitle: "Отменить последнюю правку",
        redoTitle: "Вернуть правку",
        brush: "Кисть",
        fit: "Вписать",
        fitTitle: "Вписать изображение в окно (сбрасывает масштаб и смещение).",
        oneToOne: "1:1",
        oneToOneTitle: "Показать изображение 1:1 (один пиксель изображения на пиксель экрана).",
        settingsTitle: "Расширенные настройки",
        closeTitle: "Закрыть редактор (Esc)",
        advanced: "Расширенные",
        maxResolutionField: "Макс. разрешение LaMa",
        maskPaddingField: "Отступ контекста маски",
        featherField: "Растушёвка склейки",
        dropHint: "Отпустите изображение для загрузки",
        openEditorTitle: "Открыть полноэкранный редактор",
        noImageYet: "Изображения пока нет — перетащите или вставьте сюда.",
        imageFallback: "изображение",
        stepTag: (index, total) => ` • шаг ${index}/${total}`,
        undone: "Предыдущая правка отменена.",
        redone: "Следующая правка восстановлена.",
        failedLoadImage: "Не удалось загрузить изображение.",
        failedPreview: (message) => `Не удалось загрузить превью изображения: ${message}`,
        uploadFailed: "Не удалось загрузить файл.",
        uploadingImage: "Загрузка изображения...",
        uploading: "Загрузка...",
        failedWorkingCopy: "Не удалось подготовить рабочую копию.",
        imageLoaded: "Изображение загружено. Закрашивайте дефекты кистью.",
        loadingModel: "Загрузка модели...",
        inpainting: "Ретушь...",
        sendingRegion: "Отправка области в LaMa...",
        inpaintFailed: "Запрос ретуши не выполнен.",
        cleanupApplied: "Очистка применена. Закрасьте другую область или нажмите «Сохранить».",
        nothingToSave: "Пока нечего сохранять.",
        saveFailed: "Не удалось сохранить.",
        savedTo: (name) => `Сохранено в output: ${name}`,
        unknownFile: "неизвестно",
        noSource: "Нет исходного изображения для сброса.",
        filePickerFailed: (message) => `Не удалось открыть выбор файла: ${message}`,
        loadedSaved: "Загружено сохранённое рабочее состояние.",
        ready: "Готово — откройте интерфейс, чтобы загрузить изображение.",
    },
};

function ensureStyles() {
    // Shared tokens and component classes (buttons, sliders, panels, modal
    // shell) come from js/_theme.js; the rules below only add layout that is
    // specific to this node. Never hard-code colours here — use --ts-* tokens.
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
/* Layout only — every colour comes from the shared --ts-* tokens in
   js/_theme.js. Component look (buttons, sliders, panels, status bar, modal,
   spinner, drop target) comes from the ts-ui-* classes applied in JS. */
.ts-lama{position:relative;width:100%;height:100%;min-height:0;background:var(--ts-checker);
  border:1px solid var(--ts-border-soft);border-radius:var(--ts-radius-lg);overflow:hidden;user-select:none}
.ts-lama-modal .ts-lama{border:none;border-radius:0}
/* ---- Compact in-node shell ---- */
.ts-lama-shell{position:relative;width:100%;height:100%;min-height:0;display:flex;flex-direction:column;
  gap:6px;padding:6px;background:var(--ts-bg);border:1px solid var(--ts-border-soft);
  border-radius:var(--ts-radius-lg);overflow:hidden;user-select:none}
.ts-lama-shell__preview{position:relative;flex:1 1 auto;min-height:0;display:flex;align-items:center;
  justify-content:center;border-radius:var(--ts-radius);overflow:hidden;cursor:pointer;
  background:var(--ts-checker);border:1px solid var(--ts-border-soft)}
.ts-lama-shell__preview img{max-width:100%;max-height:100%;object-fit:contain;display:block}
.ts-lama-shell__placeholder{padding:10px;text-align:center;font-size:var(--ts-fs-sm);
  color:var(--ts-muted);pointer-events:none}
.ts-lama-shell__row{display:flex;align-items:center;gap:6px;flex:0 0 auto}
.ts-lama-shell__status{flex:0 0 auto;width:100%;min-width:0;font-size:var(--ts-fs-xs);
  color:var(--ts-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center}
/* ---- Editor canvas + floating chrome ---- */
.ts-lama__canvas{position:absolute;inset:0;display:block;width:100%;height:100%;cursor:default;touch-action:none}
.ts-lama__canvas.has-image{cursor:none}
.ts-lama__empty{position:absolute;left:8px;right:8px;top:56px;bottom:44px;display:flex;align-items:center;
  justify-content:center;text-align:center;padding:16px;color:var(--ts-muted);font-size:var(--ts-fs);
  pointer-events:none;background:var(--ts-scrim);border-radius:var(--ts-radius)}
.ts-lama__overlay{z-index:22}
/* right inset leaves the top-right corner free for the shared fullscreen close
   button (.ts-ui-fs-close). */
.ts-lama__toolbar{position:absolute;top:8px;left:8px;right:52px;z-index:6}
.ts-lama__group--brush{flex:1 1 auto;min-width:0;justify-content:flex-start}
.ts-lama__brush-slider{flex:1 1 auto;min-width:60px;max-width:200px}
.ts-lama__brush-value{font-size:var(--ts-fs-sm);color:var(--ts-text);font-variant-numeric:tabular-nums;
  min-width:28px;text-align:right;font-weight:600}
.ts-lama__settings{position:absolute;top:50px;right:8px;width:240px;z-index:7;display:none;
  flex-direction:column;gap:10px}
.ts-lama__settings.is-open{display:flex}
.ts-lama__statusbar{position:absolute;left:8px;right:8px;bottom:8px;pointer-events:none;z-index:4}
/* Brush ring: deliberately achromatic — it sits over arbitrary imagery, so it
   uses a white stroke with a dark halo for contrast rather than the accent. */
.ts-lama__cursor{position:absolute;margin:0;padding:0;border-radius:50%;
  border:1.5px solid rgba(255,255,255,.95);box-shadow:0 0 0 1px rgba(0,0,0,.65);box-sizing:border-box;
  pointer-events:none;will-change:left,top,width,height;display:none;z-index:3}
.ts-lama__cursor.is-visible{display:block}
/* Live brush-size preview (Photoshop-style): a centred ring the exact on-screen
   size of the brush plus a readable px chip, shown while the size slider or the
   [ / ] keys change the size. Achromatic like the cursor so it reads over any
   imagery. */
.ts-lama__brush-preview{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
  border-radius:50%;border:1.5px solid rgba(255,255,255,.95);box-shadow:0 0 0 1px rgba(0,0,0,.65);
  box-sizing:border-box;pointer-events:none;display:none;z-index:5}
.ts-lama__brush-preview.is-visible{display:block}
.ts-lama__brush-preview-label{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
  pointer-events:none;display:none;z-index:6;padding:2px 9px;border-radius:6px;
  background:rgba(0,0,0,.72);color:rgba(255,255,255,.96);font-size:12px;font-family:var(--ts-font);
  font-variant-numeric:tabular-nums;white-space:nowrap;box-shadow:0 1px 3px rgba(0,0,0,.5)}
.ts-lama__brush-preview-label.is-visible{display:block}
`;
    document.head.appendChild(style);
}

function isNodesV2() {
    // Keys off the CLASS (always present in modern builds), so this is
    // effectively always true and the widget always uses getMinHeight/getMaxHeight
    // — which size the pane correctly in BOTH renderers. Do NOT "fix" this to read
    // Comfy.VueNodes.Enabled: the computeSize branch it would enable in the classic
    // renderer fights node.size in a runaway feedback loop. See js/_dom_widget.js.
    return Boolean(window?.comfyAPI?.domWidget?.DOMWidgetImpl);
}
function stopPropagation(element, events) {
    events.forEach((name) => element.addEventListener(name, (event) => event.stopPropagation()));
}
function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}
// Slider resolution: one slider step per integer-image-pixel candidate in
// the BRUSH_MAX_PX range. With 400 steps over a log [1..400] mapping, every
// integer brush value in 1..200 round-trips cleanly through
// brushToSliderValue → sliderValueToBrush (a few values above ~390 collapse
// to the max). At 100 steps the round-trip drifts (e.g. 40 → 62 → 41) and
// silently mutates persisted workflow brush_size on first interaction.
const BRUSH_SLIDER_STEPS = BRUSH_MAX_PX;
function sliderValueToBrush(sliderValue) {
    const t = clamp(sliderValue / BRUSH_SLIDER_STEPS, 0, 1);
    const logSize = BRUSH_LOG_MIN + t * (BRUSH_LOG_MAX - BRUSH_LOG_MIN);
    return Math.max(BRUSH_MIN_PX, Math.round(Math.exp(logSize)));
}
function brushToSliderValue(brushPx) {
    const value = clamp(brushPx, BRUSH_MIN_PX, BRUSH_MAX_PX);
    const t = (Math.log(value) - BRUSH_LOG_MIN) / (BRUSH_LOG_MAX - BRUSH_LOG_MIN);
    return clamp(Math.round(t * BRUSH_SLIDER_STEPS), 0, BRUSH_SLIDER_STEPS);
}
export function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || node?._tsHiddenWidgets?.[name] || null;
}
function hideWidget(node, name) {
    return sharedHideWidget(node, name);
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
function readNumber(node, name, fallback) {
    // Distinct from `Number(getWidgetValue(...) || fallback)` which collapses
    // a legitimate stored 0 into the fallback. We preserve 0 (and any other
    // finite number) and only fall back when the value is NaN / null /
    // undefined / non-numeric string.
    const num = Number(getWidgetValue(node, name, fallback));
    return Number.isFinite(num) ? num : fallback;
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
    // Resolved here (not at module level) so the ComfyUI settings store is
    // ready when the locale is read.
    const L = pickLocaleStrings(STRINGS);
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
        // brushSize is clamped at load so a tampered or out-of-range
        // workflow value (e.g. 1000 from an older schema) cannot reach
        // drawSegment / drawBrushAt and paint a 500-image-px mask blob.
        brushSize: clamp(readNumber(node, INPUT_BRUSH_SIZE, 40), BRUSH_MIN_PX, BRUSH_MAX_PX),
        maxResolution: readNumber(node, INPUT_MAX_RESOLUTION, 512),
        maskPadding: readNumber(node, INPUT_MASK_PADDING, 64),
        feather: readNumber(node, INPUT_FEATHER, 4),
        statusText: "",
        statusKind: "info",
        image: null,
        imageWidth: 0,
        imageHeight: 0,
        // scale = fitScale * zoomLevel; fitScale recomputed each resize so the
        // image always letterboxes inside the canvas at zoomLevel = 1.
        fitScale: 1,
        scale: 1,
        zoomLevel: 1,
        panX: 0,
        panY: 0,
        offsetX: 0,
        offsetY: 0,
        // Mouse-driven pan state (middle-button drag).
        isPanning: false,
        panStartClientX: 0,
        panStartClientY: 0,
        panStartPanX: 0,
        panStartPanY: 0,
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
        // Fullscreen editor visibility. While false the painting container is
        // detached from the DOM, so redraws are skipped (zero-size rects).
        editorOpen: false,
        // Tracks whether the mouse is over the node shell, so clipboard paste
        // can be routed to the right node when several exist on the graph.
        pointerOverShell: false,
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
    container.className = `${TS_UI_CLASS} ts-lama`;

    const canvas = document.createElement("canvas");
    canvas.className = "ts-lama__canvas";

    const empty = document.createElement("div");
    empty.className = "ts-lama__empty";
    empty.textContent = L.clickToBegin;

    const overlay = document.createElement("div");
    overlay.className = "ts-ui-scrim ts-lama__overlay";
    const spinner = document.createElement("div");
    spinner.className = "ts-ui-spinner";
    const overlayLabel = document.createElement("div");
    overlayLabel.textContent = L.processing;
    overlay.append(spinner, overlayLabel);

    // Toolbar
    const toolbar = document.createElement("div");
    toolbar.className = "ts-ui-toolbar ts-lama__toolbar";

    const leftGroup = document.createElement("div");
    leftGroup.className = "ts-ui-group";
    const loadButton = document.createElement("button");
    loadButton.className = "ts-ui-btn ts-ui-btn--primary";
    loadButton.textContent = L.load;
    const saveButton = document.createElement("button");
    saveButton.className = "ts-ui-btn";
    saveButton.textContent = L.save;
    saveButton.title = L.saveTitle;
    const resetButton = document.createElement("button");
    resetButton.className = "ts-ui-btn";
    resetButton.textContent = L.reset;
    resetButton.title = L.resetTitle;
    const undoButton = document.createElement("button");
    undoButton.className = "ts-ui-btn ts-ui-btn--icon";
    undoButton.title = L.undoTitle;
    undoButton.innerHTML = undoIconSvg();
    undoButton.disabled = true;
    const redoButton = document.createElement("button");
    redoButton.className = "ts-ui-btn ts-ui-btn--icon";
    redoButton.title = L.redoTitle;
    redoButton.innerHTML = redoIconSvg();
    redoButton.disabled = true;
    leftGroup.append(loadButton, saveButton, resetButton, undoButton, redoButton);

    const brushGroup = document.createElement("div");
    brushGroup.className = "ts-ui-group ts-lama__group--brush";
    const brushLabel = document.createElement("div");
    brushLabel.className = "ts-ui-label";
    brushLabel.textContent = L.brush;
    // Brush slider uses a log scale (1 → 400 px in image-pixels) so small,
    // detail brushes get half the slider range instead of being squeezed into
    // the first few percent. The underlying widget stores the literal image-px
    // size so existing workflows keep working.
    const brushSlider = makeSlider({
        min: 0,
        max: BRUSH_SLIDER_STEPS,
        step: 1,
        value: brushToSliderValue(state.brushSize),
        className: "ts-ui-slider ts-lama__brush-slider",
        onInput: (sliderValue) => {
            const brushPx = sliderValueToBrush(sliderValue);
            state.brushSize = brushPx;
            brushValueLabel.textContent = String(brushPx);
            setWidgetValue(node, INPUT_BRUSH_SIZE, brushPx);
            updateCursorElement();
            showBrushPreview();
        },
    });
    const brushValueLabel = document.createElement("div");
    brushValueLabel.className = "ts-lama__brush-value";
    brushValueLabel.textContent = String(Math.round(state.brushSize));
    brushGroup.append(brushLabel, brushSlider, brushValueLabel);

    const zoomGroup = document.createElement("div");
    zoomGroup.className = "ts-ui-group";
    const fitButton = document.createElement("button");
    fitButton.className = "ts-ui-btn";
    fitButton.textContent = L.fit;
    fitButton.title = L.fitTitle;
    const oneToOneButton = document.createElement("button");
    oneToOneButton.className = "ts-ui-btn";
    oneToOneButton.textContent = L.oneToOne;
    oneToOneButton.title = L.oneToOneTitle;
    zoomGroup.append(fitButton, oneToOneButton);

    const rightGroup = document.createElement("div");
    rightGroup.className = "ts-ui-group";
    const settingsButton = document.createElement("button");
    settingsButton.className = "ts-ui-btn ts-ui-btn--icon";
    settingsButton.title = L.settingsTitle;
    settingsButton.innerHTML = gearIconSvg();
    // The close (×) control is provided by the shared fullscreen overlay
    // (openFullscreenOverlay), unified across all TS fullscreen editors.
    rightGroup.append(settingsButton);

    toolbar.append(leftGroup, brushGroup, zoomGroup, rightGroup);

    // Settings popover
    const settings = document.createElement("div");
    settings.className = "ts-ui-panel ts-lama__settings";

    const settingsTitle = document.createElement("div");
    settingsTitle.className = "ts-ui-title";
    settingsTitle.textContent = L.advanced;
    settings.append(settingsTitle);

    function buildField(name, options, getter, setter, widgetKey) {
        const field = document.createElement("div");
        field.className = "ts-ui-field";
        const row = document.createElement("div");
        row.className = "ts-ui-field__row";
        const nameEl = document.createElement("div");
        nameEl.className = "ts-ui-field__name";
        nameEl.textContent = name;
        const valueEl = document.createElement("div");
        valueEl.className = "ts-ui-field__value";
        valueEl.textContent = String(Math.round(getter()));
        row.append(nameEl, valueEl);
        const slider = makeSlider({
            min: options.min,
            max: options.max,
            step: options.step,
            value: getter(),
            className: "ts-ui-slider",
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
        buildField(L.maxResolutionField, { min: 128, max: 2048, step: 64 }, () => state.maxResolution, (next) => { state.maxResolution = next; }, INPUT_MAX_RESOLUTION),
        buildField(L.maskPaddingField, { min: 0, max: 512, step: 8 }, () => state.maskPadding, (next) => { state.maskPadding = next; }, INPUT_MASK_PADDING),
        buildField(L.featherField, { min: 0, max: 64, step: 1 }, () => state.feather, (next) => { state.feather = next; }, INPUT_FEATHER),
    );

    // Status bar
    const statusBar = document.createElement("div");
    statusBar.className = "ts-ui-statusbar ts-lama__statusbar";
    const statusText = document.createElement("div");
    statusText.className = "ts-ui-ellipsis";
    statusText.textContent = L.clickToBegin;
    const statusMeta = document.createElement("div");
    statusMeta.className = "ts-ui-meta";
    statusBar.append(statusText, statusMeta);

    // Hidden file input for Load Image
    const fileInput = document.createElement("input");
    fileInput.className = "ts-ui-file";
    fileInput.type = "file";
    fileInput.accept = MEDIA_UPLOAD_ACCEPT;

    // Cursor circle as a real HTML element — moved with CSS transform so
    // mouse movement doesn't force a full canvas repaint of the image.
    const cursorElement = document.createElement("div");
    cursorElement.className = "ts-lama__cursor";

    // Centred live brush-size preview (shown while the slider / [ ] keys move
    // the size — the pointer is off-canvas then, so the cursor ring is hidden).
    const brushPreviewRing = document.createElement("div");
    brushPreviewRing.className = "ts-lama__brush-preview";
    const brushPreviewLabel = document.createElement("div");
    brushPreviewLabel.className = "ts-lama__brush-preview-label";

    // Visual hint shown while dragging an image file over the node.
    const dropHint = document.createElement("div");
    dropHint.className = "ts-ui-drop";
    dropHint.textContent = L.dropHint;

    container.append(canvas, empty, overlay, toolbar, settings, statusBar, fileInput, cursorElement, brushPreviewRing, brushPreviewLabel, dropHint);

    // ---------- Compact in-node shell ----------
    // The node body only hosts a preview plus the shared launcher button; the
    // painting UI (container above) is mounted into a fullscreen overlay on
    // demand. Keeping the container in a variable (detached while closed)
    // preserves all canvas/mask/history state across open→close→open.
    const shell = document.createElement("div");
    shell.className = `${TS_UI_CLASS} ts-lama-shell`;

    const shellPreview = document.createElement("div");
    shellPreview.className = "ts-lama-shell__preview";
    shellPreview.title = L.openEditorTitle;
    const shellImage = document.createElement("img");
    shellImage.alt = "";
    shellImage.style.display = "none";
    const shellPlaceholder = document.createElement("div");
    shellPlaceholder.className = "ts-lama-shell__placeholder";
    // No button label quoted here: the launcher right below already carries
    // the wording, and it is localised — embedding it produced a mixed-language
    // sentence.
    shellPlaceholder.textContent = L.noImageYet;
    shellPreview.append(shellImage, shellPlaceholder);

    const shellRow = document.createElement("div");
    shellRow.className = "ts-ui-launchbar ts-lama-shell__row";
    const shellEditButton = createOpenInterfaceButton(() => openEditor());
    const shellStatus = document.createElement("div");
    shellStatus.className = "ts-ui-status ts-lama-shell__status";
    shellRow.append(shellEditButton);

    shell.append(shellPreview, shellRow, shellStatus);
    // Pointer traffic over the shell must not reach LiteGraph (node drag on
    // button press, graph zoom on wheel) — same guard the container gets.
    stopPropagation(shell, [
        "pointerdown", "pointerup", "pointermove",
        "mousedown", "mouseup", "mousemove",
        "wheel", "click", "dblclick", "contextmenu",
    ]);

    // "wheel" is stopped at the container level so wheel events fired over
    // the toolbar / brush slider / settings popover (children of container,
    // not children of canvas) do not bubble out to LiteGraph and zoom the
    // graph. The canvas has its own wheel listener registered below that
    // runs first because the canvas is the innermost target of any wheel
    // event over the image — that listener calls stopPropagation itself,
    // so wheel-over-image is consumed by in-node zoom and never reaches the
    // container handler.
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
        getMinHeight: () => 110,
        getMaxHeight: () => 8192,
        afterResize: () => { requestRedraw(); },
    };
    const domWidget = node.addDOMWidget(DOM_WIDGET_NAME, "div", shell, widgetOptions);
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
        shell.style.width = "100%";
        shell.style.height = "100%";
        shell.style.minHeight = "0";
    }

    function setStatus(message, kind = "info") {
        state.statusText = message || "";
        state.statusKind = kind;
        statusText.textContent = message || "";
        statusBar.classList.toggle("is-error", kind === "error");
        statusBar.classList.toggle("is-success", kind === "success");
        // Mirror into the node shell so progress/errors stay visible after the
        // fullscreen editor is closed (uploads and inpaints can finish there).
        shellStatus.textContent = message || "";
        shellStatus.title = message || "";
        shellStatus.classList.toggle("is-error", kind === "error");
        shellStatus.classList.toggle("is-success", kind === "success");
    }

    function setOverlay(active, label = L.processing) {
        overlay.classList.toggle("is-active", Boolean(active));
        overlayLabel.textContent = label;
    }

    // Node-shell preview. Keyed on the file path (the backend writes a fresh
    // versioned file per edit) so updateMeta — which runs on every pan/zoom
    // frame — never re-triggers a network fetch for an unchanged image.
    let shellPreviewPath = null;
    function updateShellPreview() {
        const path = state.workingPath || state.sourcePath || "";
        if (path === shellPreviewPath) return;
        shellPreviewPath = path;
        if (!path) {
            shellImage.removeAttribute("src");
            shellImage.style.display = "none";
            shellPlaceholder.style.display = "";
            return;
        }
        shellImage.onload = () => {
            shellImage.style.display = "block";
            shellPlaceholder.style.display = "none";
        };
        shellImage.onerror = () => {
            shellImage.style.display = "none";
            shellPlaceholder.style.display = "";
        };
        shellImage.src = imageUrlForPath(path);
    }

    function updateMeta() {
        const filename = state.sourcePath ? state.sourcePath.split(/[\\/]/).pop().replace(/\s\[input\]$/i, "") : "";
        const historyTag = state.history.length > 1
            ? L.stepTag(state.historyIndex + 1, state.history.length)
            : "";
        const zoomTag = state.image && state.scale > 0
            ? ` • ${Math.round(state.scale * 100)}%`
            : "";
        statusMeta.textContent = state.imageWidth && state.imageHeight
            ? `${filename || L.imageFallback} • ${state.imageWidth} × ${state.imageHeight}${historyTag}${zoomTag}`
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
        shellEditButton.disabled = state.isProcessing;
        updateShellPreview();
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
        setStatus(L.undone, "info");
    }

    async function doRedo() {
        if (state.historyIndex >= state.history.length - 1) return;
        await goToHistory(state.historyIndex + 1);
        setStatus(L.redone, "info");
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
            tintedMaskCtx.fillStyle = MASK_TINT;
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
            tintedMaskCtx.strokeStyle = MASK_TINT;
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
            // Assigning to canvas.width/height clears the bitmap. Pointer
            // events and LiteGraph graph-zoom can both call resizeCanvas
            // outside the redraw cycle (pointerToImageCoords on hover, wheel
            // handler before scale math). Without this requestRedraw the
            // image would disappear until the user nudges the node.
            requestRedraw();
        }
        // Image placement = letterbox fit, then multiplied by user zoom, then
        // shifted by user pan. The image cache holds the rendered image at
        // current scale/offset and must be invalidated when any of those
        // change (resize, wheel zoom, drag pan, Fit / 1:1 buttons).
        if (state.imageWidth > 0 && state.imageHeight > 0 && rect.width > 0 && rect.height > 0) {
            const usableWidth = Math.max(1, rect.width - IMAGE_PAD_SIDE * 2);
            const usableHeight = Math.max(1, rect.height - IMAGE_PAD_TOP - IMAGE_PAD_BOTTOM);
            const fitScale = Math.min(usableWidth / state.imageWidth, usableHeight / state.imageHeight);
            const newScale = fitScale * state.zoomLevel;
            const drawWidth = state.imageWidth * newScale;
            const drawHeight = state.imageHeight * newScale;
            // Clamp pan: overflow ÷ 2 keeps an aligned edge in view; PAN_SLACK
            // gives a small extra margin (only when the image actually
            // overflows the usable area in that axis) so the user can pan
            // just beyond the image edge to see context, but cannot drift
            // off-centre at fit-letterbox where overflow is 0 in both axes.
            // Previously PAN_SLACK_PX was added unconditionally, letting a
            // ±120px drift at zoomLevel=1 violate the "Fit" centered guarantee.
            const overflowX = Math.max(0, drawWidth - usableWidth);
            const overflowY = Math.max(0, drawHeight - usableHeight);
            const maxPanX = overflowX > 0 ? overflowX / 2 + PAN_SLACK_PX : 0;
            const maxPanY = overflowY > 0 ? overflowY / 2 + PAN_SLACK_PX : 0;
            state.panX = clamp(state.panX, -maxPanX, maxPanX);
            state.panY = clamp(state.panY, -maxPanY, maxPanY);
            const newOffsetX = IMAGE_PAD_SIDE + (usableWidth - drawWidth) / 2 + state.panX;
            const newOffsetY = IMAGE_PAD_TOP + (usableHeight - drawHeight) / 2 + state.panY;
            // Image cache stores the image at scale, blitted at runtime
            // offset, so only a scale change invalidates it. Offset changes
            // (pan) are handled by drawImage's dx/dy in redraw() without a
            // rebuild.
            if (Math.abs(newScale - state.scale) > 1e-4) {
                imageCacheValid = false;
            }
            state.fitScale = fitScale;
            state.scale = newScale;
            state.offsetX = newOffsetX;
            state.offsetY = newOffsetY;
        }
        return { rectWidth: rect.width, rectHeight: rect.height, dpr };
    }

    function rebuildImageCache(dpr) {
        if (!state.image || !state.imageWidth || !state.imageHeight) {
            imageCacheValid = false;
            return;
        }
        // Cache holds the image at its current display scale at the cache's
        // origin (no offset). redraw() blits the cache into the main canvas
        // at state.offsetX/Y. Decoupling cache size from the *displayed
        // position* means pan does not invalidate the cache (only zoom
        // does), eliminating the per-pointermove rebuild that previously
        // collapsed FPS on 4K+ images during a pan gesture.
        const drawWidth = state.imageWidth * state.scale;
        const drawHeight = state.imageHeight * state.scale;
        const cacheW = Math.max(1, Math.ceil(drawWidth * dpr));
        const cacheH = Math.max(1, Math.ceil(drawHeight * dpr));
        if (imageCacheCanvas.width !== cacheW || imageCacheCanvas.height !== cacheH) {
            imageCacheCanvas.width = cacheW;
            imageCacheCanvas.height = cacheH;
        }
        imageCacheCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        imageCacheCtx.clearRect(0, 0, drawWidth, drawHeight);
        imageCacheCtx.drawImage(state.image, 0, 0, drawWidth, drawHeight);
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
        // Hide the brush ring whenever the pointer hovers (or pointer-capture
        // drag-paints) over one of the on-canvas overlays — toolbar, status
        // bar, settings popover. Without this, painting strokes that drag
        // up into the toolbar leave a brush circle floating on top of the
        // buttons, and pointer-leave never fires while capture is held so
        // the ring stays planted until the user releases LMB.
        const overlays = [toolbar, statusBar, settings];
        for (const el of overlays) {
            const r = el.getBoundingClientRect();
            if (!r.width || !r.height) continue;
            if (state.cursorClientX >= r.left && state.cursorClientX <= r.right
                && state.cursorClientY >= r.top && state.cursorClientY <= r.bottom) {
                cursorElement.classList.remove("is-visible");
                return;
            }
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

    // Flash a centred ring of the brush's exact on-screen size (+ a px chip)
    // while the size is being changed from the slider or the [ / ] keys, then
    // fade it after a short idle. Mirrors updateCursorElement's diameter math so
    // the preview matches the actual paint size at any zoom.
    let brushPreviewTimer = 0;
    function showBrushPreview() {
        if (!state.image) return;
        const containerRect = container.getBoundingClientRect();
        if (!containerRect.width || !containerRect.height) return;
        const layoutWidth = container.offsetWidth || containerRect.width;
        const parentScale = layoutWidth > 0 ? containerRect.width / layoutWidth : 1;
        const inverseScale = parentScale > 0.001 ? 1 / parentScale : 1;
        const visualScale = state.scale || 1;
        const diameter = Math.max(4, state.brushSize * visualScale * inverseScale);
        brushPreviewRing.style.width = `${diameter}px`;
        brushPreviewRing.style.height = `${diameter}px`;
        brushPreviewLabel.textContent = `${Math.round(state.brushSize)} px`;
        brushPreviewRing.classList.add("is-visible");
        brushPreviewLabel.classList.add("is-visible");
        window.clearTimeout(brushPreviewTimer);
        brushPreviewTimer = window.setTimeout(() => {
            brushPreviewRing.classList.remove("is-visible");
            brushPreviewLabel.classList.remove("is-visible");
        }, 850);
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
        // Rebuild the cached display-resolution image only when the image
        // identity or scale changed. Pan (offsetX/offsetY only) reuses the
        // cache and just blits it at the new offset.
        if (!imageCacheValid) {
            rebuildImageCache(dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const cacheBlitW = state.imageWidth * state.scale;
        const cacheBlitH = state.imageHeight * state.scale;
        ctx.drawImage(imageCacheCanvas, state.offsetX, state.offsetY, cacheBlitW, cacheBlitH);
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
        // While the editor is closed the container is detached: every rect is
        // 0×0, so a redraw would only burn frames (and spin the
        // pendingLayoutAttempts retry). openEditor() repaints on mount.
        if (!state.editorOpen) return;
        if (redrawScheduled) return;
        redrawScheduled = true;
        requestAnimationFrame(() => {
            redrawScheduled = false;
            redraw();
            scheduleCanvasDirty();
        });
    }

    function pointerToImageCoords(event) {
        // Refresh placement state before reading it — node resize, wheel zoom
        // and pan can all leave state.scale/offset stale relative to the live
        // canvas rect. Without this, withinImage flickers and the brush jumps.
        resizeCanvas();
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
            image.onerror = () => reject(new Error(L.failedLoadImage));
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
            const newWidth = image.naturalWidth || image.width || 0;
            const newHeight = image.naturalHeight || image.height || 0;
            // Reset zoom and pan when the image dimensions actually change —
            // i.e. on a brand-new load. Inpaint refreshes (same w×h) preserve
            // the user's current zoom/pan so they can keep cleaning a region.
            if (newWidth !== state.imageWidth || newHeight !== state.imageHeight) {
                state.zoomLevel = 1;
                state.panX = 0;
                state.panY = 0;
            }
            state.image = image;
            state.imageWidth = newWidth;
            state.imageHeight = newHeight;
            imageCacheValid = false;
            ensureMaskCanvasSize();
            if (options.clearMask !== false) clearMask();
            updateMeta();
            requestRedraw();
        } catch (error) {
            setStatus(L.failedPreview(error?.message || error), "error");
        }
    }

    async function uploadFile(file) {
        const form = new FormData();
        form.append("image", file, file.name);
        form.append("type", "input");
        const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload?.error || payload?.message || L.uploadFailed);
        return buildAnnotatedPath(payload);
    }

    async function chooseSourceFile(file) {
        if (!file) return;
        state.isProcessing = false;
        setOverlay(true, L.uploadingImage);
        setStatus(L.uploading, "info");
        try {
            const annotated = await uploadFile(file);
            if (!annotated) throw new Error(L.uploadFailed);
            state.sourcePath = annotated;
            state.workingPath = "";
            setWidgetValue(node, INPUT_SOURCE_PATH, annotated);
            setWidgetValue(node, INPUT_WORKING_PATH, "");
            clearMask();
            await seedWorkingFile();
        } catch (error) {
            setStatus(error?.message || L.failedLoadImage, "error");
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
                setStatus(payload?.error || L.failedWorkingCopy, "error");
                return;
            }
            state.workingPath = String(payload?.working_path || "");
            setWidgetValue(node, INPUT_WORKING_PATH, state.workingPath);
            // Fresh source resets history to a single entry.
            resetHistoryTo(state.workingPath);
            await refreshImage({ clearMask: true });
            setStatus(L.imageLoaded, "info");
        } catch (error) {
            setStatus(error?.message || L.failedWorkingCopy, "error");
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
                    setOverlay(true, String(status?.message || L.loadingModel));
                } else if (status?.loaded) {
                    if (state.isModelLoading) {
                        state.isModelLoading = false;
                        setOverlay(true, L.inpainting);
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
        setOverlay(true, L.inpainting);
        setStatus(L.sendingRegion, "info");
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
                throw new Error(payload?.error || L.inpaintFailed);
            }
            state.workingPath = String(payload?.working_path || "");
            setWidgetValue(node, INPUT_WORKING_PATH, state.workingPath);
            pushHistory(state.workingPath);
            await refreshImage({ clearMask: true });
            setStatus(L.cleanupApplied, "success");
        } catch (error) {
            setStatus(error?.message || L.inpaintFailed, "error");
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
            setStatus(L.nothingToSave, "error");
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
                throw new Error(payload?.error || L.saveFailed);
            }
            setStatus(L.savedTo(payload?.filename || payload?.saved_path || L.unknownFile), "success");
        } catch (error) {
            setStatus(error?.message || L.saveFailed, "error");
        }
    }

    async function resetToSource() {
        if (state.isProcessing) return;
        if (!state.sourcePath) {
            setStatus(L.noSource, "error");
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
        // refreshImage only resets zoom/pan when image dimensions actually
        // change — for a Reset, the re-seeded source has the same dimensions
        // as what was on screen, so the dimensions-equal heuristic would keep
        // whatever zoom/pan the user had built up. That contradicts the Reset
        // button's documented "discard local edits and restart from the
        // loaded image" semantics; reset them here explicitly.
        state.zoomLevel = 1;
        state.panX = 0;
        state.panY = 0;
        await seedWorkingFile();
    }

    // ---------- Fullscreen editor ----------
    // The heavy editing UI (`container`, with its canvas, mask bitmaps and undo
    // history) is mounted into the shared fullscreen overlay on open and merely
    // re-parented out on close, so closing/reopening never loses the edit. All
    // the modal + focus-shield + Esc plumbing lives in js/_fullscreen.js.
    let editorHandle = null;

    function setBrushSize(nextPx) {
        const brushPx = clamp(Math.round(nextPx), BRUSH_MIN_PX, BRUSH_MAX_PX);
        state.brushSize = brushPx;
        brushSlider.value = String(brushToSliderValue(brushPx));
        brushValueLabel.textContent = String(brushPx);
        setWidgetValue(node, INPUT_BRUSH_SIZE, brushPx);
        updateCursorElement();
        showBrushPreview();
    }

    // Editor-scoped shortcuts. Esc-to-close and the focus shield that keeps
    // ComfyUI's capture-phase Ctrl+Z away from the graph (so it can't delete the
    // node under the open editor) are handled by the shared overlay. See
    // CLAUDE.md §12.5 / project_memory/reference_modal_hotkeys.md.
    function onEditorKey(event) {
        if (!state.editorOpen) return;
        const stop = () => { event.preventDefault(); event.stopPropagation(); };
        const active = document.activeElement;
        const tag = active?.tagName;
        // Real fields (the sliders) keep the keys; the parked key-anchor textarea
        // does not count as a field here.
        if ((tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT")
            && !active?.classList?.contains("ts-ui-keyanchor")) return;
        // Brush size on [ / ] — e.code is the physical key, so it also works on
        // a Cyrillic layout.
        if (event.code === "BracketLeft") { stop(); setBrushSize(state.brushSize / 1.2); return; }
        if (event.code === "BracketRight") { stop(); setBrushSize(state.brushSize * 1.2); return; }
        const mod = event.ctrlKey || event.metaKey;
        if (!mod) return;
        if (event.code === "KeyZ") { stop(); if (event.shiftKey) doRedo(); else doUndo(); }
        else if (event.code === "KeyY") { stop(); doRedo(); }
    }

    function openEditor() {
        if (state.editorOpen) return;
        state.editorOpen = true;
        resizeObserver.observe(container);
        editorHandle = openFullscreenOverlay(container, {
            closeTitle: L.closeTitle,
            onKey: onEditorKey,
            onOpen: () => {
                // The container was detached (zero-size rects) while closed, so
                // every cached layout value is stale.
                imageCacheValid = false;
                updateMeta();
                requestRedraw();
            },
            onClose: () => {
                editorHandle = null;
                state.editorOpen = false;
                toggleSettings(false);
                state.isDrawing = false;
                state.isPanning = false;
                state.cursorVisible = false;
                updateCursorElement();
                try { resizeObserver.unobserve(container); } catch { /* observer may be gone */ }
                updateMeta();
                scheduleCanvasDirty();
            },
        });
    }

    function closeEditor() {
        editorHandle?.close();
    }

    function toggleSettings(open) {
        state.settingsOpen = open !== undefined ? Boolean(open) : !state.settingsOpen;
        settings.classList.toggle("is-open", state.settingsOpen);
    }

    function onPointerDown(event) {
        if (state.isProcessing) return;
        if (!state.image) return;
        // Middle-mouse drag pans the image when zoomed in (and is a harmless
        // no-op at zoomLevel = 1 because the clamp keeps pan small there).
        if (event.button === 1) {
            event.preventDefault();
            state.isPanning = true;
            state.panStartClientX = event.clientX;
            state.panStartClientY = event.clientY;
            state.panStartPanX = state.panX;
            state.panStartPanY = state.panY;
            canvas.setPointerCapture?.(event.pointerId);
            canvas.style.cursor = "grabbing";
            return;
        }
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
        if (state.isPanning) {
            // panX/Y are stored in viewport-px (rect coords) like state.offsetX,
            // so the delta is the raw clientX/Y change. resizeCanvas() clamps
            // pan to keep the image findable. We deliberately do NOT mark the
            // image cache invalid here — the cache is sized to the image-at-
            // scale and blitted at runtime offset, so pan reuses it without
            // a rebuild.
            state.panX = state.panStartPanX + (event.clientX - state.panStartClientX);
            state.panY = state.panStartPanY + (event.clientY - state.panStartClientY);
            state.cursorClientX = event.clientX;
            state.cursorClientY = event.clientY;
            // If the user pressed MMB mid-stroke (LMB still held), keep
            // lastDrawImageX/Y synced with the current cursor position so
            // releasing MMB and continuing the LMB stroke resumes from the
            // pointer instead of jump-painting a segment from the stale
            // pre-pan position across the canvas. drawSegment is NOT called
            // here — pan does not paint, it just rebases the anchor.
            if (state.isDrawing) {
                const coords = pointerToImageCoords(event);
                state.lastDrawImageX = coords.imageX;
                state.lastDrawImageY = coords.imageY;
            }
            updateMeta();
            requestRedraw();
            return;
        }
        const coords = pointerToImageCoords(event);
        state.cursorImageX = coords.imageX;
        state.cursorImageY = coords.imageY;
        state.cursorClientX = event.clientX;
        state.cursorClientY = event.clientY;
        // Cursor is visible whenever the pointer is over the canvas and an
        // image is loaded — outside the image we still want feedback so the
        // user can see the brush approach the edge. Painting (drawSegment)
        // remains gated on withinImage so we never write outside bounds.
        state.cursorVisible = Boolean(state.image);
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
        if (state.isPanning && (event.button === 1 || event.type === "pointercancel")) {
            state.isPanning = false;
            canvas.releasePointerCapture?.(event.pointerId);
            canvas.style.cursor = "";
            return;
        }
        if (!state.isDrawing) return;
        state.isDrawing = false;
        canvas.releasePointerCapture?.(event.pointerId);
        runInpaint();
    }

    function onPointerLeave() {
        if (state.isPanning) {
            state.isPanning = false;
            canvas.style.cursor = "";
        }
        state.cursorVisible = false;
        updateCursorElement();
    }

    function onWheel(event) {
        // Image-level zoom: anchored on cursor, so the image pixel under the
        // mouse stays put while the picture grows or shrinks around it.
        event.preventDefault();
        event.stopPropagation();
        if (!state.image || !state.imageWidth || !state.imageHeight) return;
        if (state.isProcessing) return;
        resizeCanvas();
        const rect = canvas.getBoundingClientRect();
        const xInCanvas = event.clientX - rect.left;
        const yInCanvas = event.clientY - rect.top;
        const oldScale = state.scale;
        if (oldScale <= 0) return;
        const imageX = (xInCanvas - state.offsetX) / oldScale;
        const imageY = (yInCanvas - state.offsetY) / oldScale;
        // Normalise wheel delta across input devices. A desktop wheel notch
        // gives deltaY ~100 in deltaMode=0 (pixels); a macOS trackpad pinch
        // emits dozens of events with deltaY ~2-6 each. Firefox can use
        // deltaMode=1 (lines, deltaY ~3 per notch). Without normalisation a
        // single trackpad pinch saturates the zoom at MAX in one gesture.
        const deltaModeUnit = event.deltaMode === 1 ? 33 : event.deltaMode === 2 ? 400 : 1;
        const intensity = clamp(Math.abs(event.deltaY) * deltaModeUnit / 100, 0.05, 1);
        const factor = event.deltaY < 0
            ? Math.pow(ZOOM_STEP, intensity)
            : Math.pow(1 / ZOOM_STEP, intensity);
        const newZoomLevel = clamp(state.zoomLevel * factor, MIN_ZOOM_LEVEL, MAX_ZOOM_LEVEL);
        if (Math.abs(newZoomLevel - state.zoomLevel) < 1e-4) return;
        state.zoomLevel = newZoomLevel;
        // Solve pan so (imageX, imageY) maps back to (xInCanvas, yInCanvas)
        // under the new scale. Same math the user expects from Photoshop.
        const newScale = state.fitScale * state.zoomLevel;
        const newDrawWidth = state.imageWidth * newScale;
        const newDrawHeight = state.imageHeight * newScale;
        const usableWidth = Math.max(1, rect.width - IMAGE_PAD_SIDE * 2);
        const usableHeight = Math.max(1, rect.height - IMAGE_PAD_TOP - IMAGE_PAD_BOTTOM);
        const centeredOffsetX = IMAGE_PAD_SIDE + (usableWidth - newDrawWidth) / 2;
        const centeredOffsetY = IMAGE_PAD_TOP + (usableHeight - newDrawHeight) / 2;
        const desiredOffsetX = xInCanvas - imageX * newScale;
        const desiredOffsetY = yInCanvas - imageY * newScale;
        state.panX = desiredOffsetX - centeredOffsetX;
        state.panY = desiredOffsetY - centeredOffsetY;
        // Apply the new scale (and offsets) NOW so updateMeta's "XXX%" tag
        // and updateCursorElement's brush radius reflect the post-zoom value
        // in the current frame. resizeCanvas will reconfirm them on the next
        // requestAnimationFrame (and clamp panX/Y), but without writing them
        // here the status bar and cursor lag one frame behind every wheel
        // tick — visible as a hitch where the cursor size pops AFTER the
        // image redraws.
        state.scale = newScale;
        state.offsetX = desiredOffsetX;
        state.offsetY = desiredOffsetY;
        imageCacheValid = false;
        state.cursorClientX = event.clientX;
        state.cursorClientY = event.clientY;
        state.cursorVisible = Boolean(state.image);
        updateMeta();
        updateCursorElement();
        requestRedraw();
    }

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("pointerleave", onPointerLeave);
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    // passive:false because the handler calls preventDefault() to stop the
    // page from scrolling while the user zooms inside the node.
    canvas.addEventListener("wheel", onWheel, { passive: false });
    // Disable middle-click autoscroll on Windows — Chromium opens a vertical-
    // scroll cursor on middle-button-down by default, which fights our pan.
    canvas.addEventListener("auxclick", (event) => { if (event.button === 1) event.preventDefault(); });

    loadButton.addEventListener("click", (event) => {
        event.stopPropagation();
        try {
            fileInput.click();
        } catch (error) {
            console.error("[TS Lama Cleanup] fileInput.click failed:", error);
            setStatus(L.filePickerFailed(error?.message || error), "error");
        }
    });
    saveButton.addEventListener("click", (event) => { event.stopPropagation(); saveToOutput(); });
    resetButton.addEventListener("click", (event) => { event.stopPropagation(); resetToSource(); });
    fitButton.addEventListener("click", (event) => {
        event.stopPropagation();
        state.zoomLevel = 1;
        state.panX = 0;
        state.panY = 0;
        imageCacheValid = false;
        updateMeta();
        updateCursorElement();
        requestRedraw();
    });
    oneToOneButton.addEventListener("click", (event) => {
        event.stopPropagation();
        if (state.fitScale > 0) {
            state.zoomLevel = clamp(1 / state.fitScale, MIN_ZOOM_LEVEL, MAX_ZOOM_LEVEL);
            state.panX = 0;
            state.panY = 0;
            imageCacheValid = false;
            updateMeta();
            updateCursorElement();
            requestRedraw();
        }
    });
    shellPreview.addEventListener("click", (event) => { event.stopPropagation(); openEditor(); });
    shell.addEventListener("pointerenter", () => { state.pointerOverShell = true; });
    shell.addEventListener("pointerleave", () => { state.pointerOverShell = false; });
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
    function onDocumentPointerDownForSettings(event) {
        if (!state.settingsOpen) return;
        if (settings.contains(event.target) || settingsButton.contains(event.target)) return;
        toggleSettings(false);
    }
    document.addEventListener("pointerdown", onDocumentPointerDownForSettings);

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
    // Drop targets: the fullscreen editor AND the compact node shell, so an
    // image can be loaded without opening the editor first.
    function dropHost(event) {
        return event.currentTarget === shell ? shell : container;
    }
    function onContainerDragEnter(event) {
        if (state.isProcessing) return;
        if (!dragHasImage(event)) return;
        event.preventDefault();
        event.stopPropagation();
        dropHost(event).classList.add("is-drag-over");
    }
    function onContainerDragOver(event) {
        if (state.isProcessing) return;
        if (!dragHasImage(event)) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = "copy";
        }
        dropHost(event).classList.add("is-drag-over");
    }
    function onContainerDragLeave(event) {
        // Only clear when actually leaving the host (not when crossing between
        // children, which fires dragleave on the child).
        const host = dropHost(event);
        if (event.relatedTarget && host.contains(event.relatedTarget)) return;
        host.classList.remove("is-drag-over");
    }
    async function onContainerDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        dropHost(event).classList.remove("is-drag-over");
        if (state.isProcessing) return;
        const files = Array.from(event.dataTransfer?.files || []);
        const file = files.find((f) => !f.type || f.type.startsWith("image/")) || files[0];
        if (!file) return;
        await chooseSourceFile(file);
    }
    for (const host of [container, shell]) {
        host.addEventListener("dragenter", onContainerDragEnter);
        host.addEventListener("dragover", onContainerDragOver);
        host.addEventListener("dragleave", onContainerDragLeave);
        host.addEventListener("drop", onContainerDrop);
    }

    // ---------- Paste image from clipboard ----------
    function pasteTargetsThisNode() {
        // Multiple Lama nodes can exist on the graph. With the editor open it
        // covers the viewport and owns the clipboard; otherwise the paste goes
        // to whichever node's shell the mouse is hovering.
        if (state.editorOpen) return true;
        return state.pointerOverShell;
    }
    async function onDocumentPaste(event) {
        if (state.isProcessing) return;
        if (!pasteTargetsThisNode()) return;
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

    // Only observed while the fullscreen editor is mounted (openEditor /
    // closeEditor); the detached container never resizes.
    const resizeObserver = new ResizeObserver(() => requestRedraw());

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
        // The overlay lives on document.body, so it would survive node removal
        // or a workflow-tab switch unless we tear it down explicitly.
        closeEditor();
        resizeObserver.disconnect();
        if (state.sourcePollHandle) window.clearInterval(state.sourcePollHandle);
        if (state.modelStatusPollHandle) window.clearInterval(state.modelStatusPollHandle);
        document.removeEventListener("paste", onDocumentPaste);
        document.removeEventListener("pointerdown", onDocumentPointerDownForSettings);
    };

    // Wire cleanup into LiteGraph's onRemoved so polling intervals and the
    // document-level paste listener don't survive when the node is deleted
    // from the graph.
    const prevOnRemoved = node.onRemoved;
    node.onRemoved = function onRemovedWithLamaCleanup() {
        try {
            node._tsLamaCleanupCleanup?.();
        } catch (err) {
            console.warn("[TS LamaCleanup] cleanup on removal failed", err);
        }
        return prevOnRemoved?.apply(this, arguments);
    };

    syncDomSize();
    updateMeta();
    requestRedraw();

    requestAnimationFrame(async () => {
        if (state.workingPath) {
            await refreshImage({ clearMask: true });
            setStatus(L.loadedSaved, "info");
        } else if (state.sourcePath) {
            await seedWorkingFile();
        } else {
            setStatus(L.ready, "info");
        }
    });
}
