// Shared setup for TS_AudioLoader and TS_AudioPreview. Public exports at the
// bottom; the registerExtension calls live in the per-node entry points
// (./ts-audio-loader.js, ./ts-audio-preview.js) so each node owns its own
// stable extension ID.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

export const LOADER_NODE_NAME = "TS_AudioLoader";
export const PREVIEW_NODE_NAME = "TS_AudioPreview";
const ROUTE_BASE = "/ts_audio_loader";
const STYLE_ID = "ts-audio-loader-styles";
export const DOM_WIDGET_NAME = "ts_audio_loader";
export const PREVIEW_UI_KEY = "ts_audio_preview";
const INPUT_MODE = "mode";
const INPUT_SOURCE_PATH = "source_path";
const INPUT_CROP_START = "crop_start_seconds";
const INPUT_CROP_END = "crop_end_seconds";
const INPUT_PREVIEW_STATE = "preview_state_json";
const DEFAULT_NODE_SIZE = [560, 380];
const MIN_NODE_WIDTH = 440;
const MIN_NODE_HEIGHT = 360;
const HEADER_FOOTER_HEIGHT = 118;
const HANDLE_HITBOX = 10;
const MEDIA_UPLOAD_ACCEPT = [
    ".aac", ".aif", ".aiff", ".avi", ".flac", ".flv", ".m2ts", ".m4a", ".m4v", ".mkv", ".mov",
    ".mp3", ".mp4", ".mpeg", ".mpg", ".mts", ".ogg", ".opus", ".ts", ".wav", ".webm", ".wma",
    "audio/*", "video/*",
].join(",");
function makeCursor(svg, fallback = "ew-resize") {
    return `url("data:image/svg+xml;utf8,${encodeURIComponent(svg)}") 12 12, ${fallback}`;
}
const HANDLE_CURSOR = makeCursor(
    `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
        <rect x="4" y="4" width="2" height="16" rx="1" fill="#b9fff1"/>
        <rect x="18" y="4" width="2" height="16" rx="1" fill="#b9fff1"/>
        <path d="M9 12h6" stroke="#b9fff1" stroke-width="1.8" stroke-linecap="round"/>
    </svg>`,
);
const HANDLE_ACTIVE_CURSOR = makeCursor(
    `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
        <rect x="4" y="3" width="2" height="18" rx="1" fill="#d3fff6"/>
        <rect x="18" y="3" width="2" height="18" rx="1" fill="#d3fff6"/>
        <path d="M9 12h6" stroke="#d3fff6" stroke-width="2" stroke-linecap="round"/>
    </svg>`,
);
const DEFAULT_WAVE_CURSOR = "crosshair";

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-audio-loader{--tsal-bg:#12161c;--tsal-panel:#171d25;--tsal-panel-alt:#0f141a;--tsal-border:#28303c;--tsal-text:#e9eef6;--tsal-muted:#91a0b4;--tsal-accent:#29c7a3;--tsal-danger:#ef6f6c;width:100%;height:100%;min-height:0;box-sizing:border-box;padding:8px;display:flex;flex-direction:column;gap:8px;color:var(--tsal-text);font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif;background:radial-gradient(circle at top right,rgba(41,199,163,.15),transparent 32%),linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.01)),var(--tsal-bg);border:1px solid var(--tsal-border);border-radius:12px;overflow:hidden}
.ts-audio-loader__topbar{display:flex;align-items:center;justify-content:flex-end;gap:8px;flex-wrap:wrap}
.ts-audio-loader__actions{display:flex;gap:6px;flex-wrap:wrap}
.ts-audio-loader__button{border:1px solid var(--tsal-border);background:linear-gradient(180deg,#1f2732,#151b23);color:var(--tsal-text);border-radius:8px;padding:6px 12px;font-size:11px;cursor:pointer}
.ts-audio-loader__button.is-primary{background:linear-gradient(180deg,#31d9b1,#1ea98a);border-color:#1ea98a;color:#062018;font-weight:700}
.ts-audio-loader__button.is-danger{background:linear-gradient(180deg,#f07d79,#cf5f5c);border-color:#cf5f5c}
.ts-audio-loader__meta{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 8px;align-items:center}
.ts-audio-loader__file{min-width:0;font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ts-audio-loader__status,.ts-audio-loader__stats,.ts-audio-loader__timeline,.ts-audio-loader__crop{font-size:11px;color:var(--tsal-muted)}
.ts-audio-loader__stats{display:inline-flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}
.ts-audio-loader__wave-wrap{position:relative;flex:1 1 auto;min-height:110px;border-radius:12px;overflow:hidden;border:1px solid var(--tsal-border);background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,0)),repeating-linear-gradient(90deg,rgba(255,255,255,.035) 0 1px,transparent 1px 80px),linear-gradient(180deg,var(--tsal-panel),var(--tsal-panel-alt))}
.ts-audio-loader__canvas{width:100%;height:100%;display:block;cursor:crosshair}
.ts-audio-loader__empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:16px;color:var(--tsal-muted);font-size:12px;pointer-events:none}
.ts-audio-loader__bottom{display:grid;grid-template-columns:auto minmax(0,1fr) auto;gap:8px;align-items:center}
.ts-audio-loader__transport{display:flex;align-items:center;gap:6px}
.ts-audio-loader__play{width:34px;height:34px;border-radius:999px;border:1px solid var(--tsal-border);background:linear-gradient(180deg,#1f2732,#151b23);color:var(--tsal-text);cursor:pointer;display:inline-flex;align-items:center;justify-content:center}
.ts-audio-loader__play.is-active{background:linear-gradient(180deg,#31d9b1,#1ea98a);border-color:#1ea98a;color:#062018}
.ts-audio-loader__play svg{width:14px;height:14px;fill:currentColor;pointer-events:none}
.ts-audio-loader__play svg *{pointer-events:none}
.ts-audio-loader__hidden-media{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}`;
    document.head.appendChild(style);
}

function isNodesV2() { return Boolean(window?.comfyAPI?.domWidget?.DOMWidgetImpl); }
function stopPropagation(element, events) { events.forEach((name) => element.addEventListener(name, (event) => event.stopPropagation())); }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function formatSeconds(value) {
    const total = Math.max(0, Number(value) || 0);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const seconds = total % 60;
    const secondsText = seconds.toFixed(2).padStart(5, "0");
    return hours > 0 ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${secondsText}` : `${String(minutes).padStart(2, "0")}:${secondsText}`;
}
export function getWidget(node, name) { return node?.widgets?.find((widget) => widget?.name === name) || null; }
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
function removeDomWidget(node) {
    if (!Array.isArray(node?.widgets)) return;
    for (let index = node.widgets.length - 1; index >= 0; index -= 1) {
        const widget = node.widgets[index];
        if (widget?.name !== DOM_WIDGET_NAME) continue;
        (widget.element || widget.el || widget.container)?.remove?.();
        node.widgets.splice(index, 1);
    }
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
function getWidgetValue(node, name, fallback) { return getWidget(node, name)?.value ?? fallback; }
function scheduleCanvasDirty() { app?.graph?.setDirtyCanvas?.(true, true); }
function playIcon() { return `<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>`; }
function pauseIcon() { return `<svg viewBox="0 0 24 24"><path d="M6 19h4V5H6zm8-14v14h4V5z"/></svg>`; }
function loopIcon() { return `<svg viewBox="0 0 24 24"><path d="M7 7h9.5l-2-2L16 3l5 5-5 5-1.5-2 2-2H7a3 3 0 0 0-3 3v1H2v-1a5 5 0 0 1 5-5zm10 10H7.5l2 2L8 21l-5-5 5-5 1.5 2-2 2H17a3 3 0 0 0 3-3v-1h2v1a5 5 0 0 1-5 5z"/></svg>`; }
function isMediaPlaying(media) {
    return Boolean(media && !media.paused && !media.ended && media.currentTime >= 0);
}

export function setupAudioLoader(node) {
    if (!node || typeof node.addDOMWidget !== "function") return;
    const nodeName = node.type || node.comfyClass || "";
    const isPreviewNode = nodeName === PREVIEW_NODE_NAME;
    if (typeof node._tsAudioLoaderCleanup === "function") node._tsAudioLoaderCleanup();
    removeDomWidget(node);
    ensureStyles();
    hideWidget(node, INPUT_CROP_START);
    hideWidget(node, INPUT_CROP_END);
    if (!isPreviewNode) {
        hideWidget(node, INPUT_MODE);
    } else {
        hideWidget(node, INPUT_PREVIEW_STATE);
    }
    node.resizable = true;
    node.size = [
        Math.max(Number(node.size?.[0]) || DEFAULT_NODE_SIZE[0], MIN_NODE_WIDTH),
        Math.max(Number(node.size?.[1]) || DEFAULT_NODE_SIZE[1], MIN_NODE_HEIGHT),
    ];
    node.min_size = [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];

    const state = {
        mode: String(getWidgetValue(node, INPUT_MODE, "load") || "load"),
        sourcePath: String(getWidgetValue(node, INPUT_SOURCE_PATH, "") || ""),
        recordedPath: "",
        cropStart: Number(getWidgetValue(node, INPUT_CROP_START, 0) || 0),
        cropEnd: Number(getWidgetValue(node, INPUT_CROP_END, -1) || -1),
        isLooping: Boolean(node.properties?.ts_audio_loader_loop ?? false),
        duration: 0, sampleRate: 0, channels: 0, mediaType: "audio", peaks: [], filename: "",
        status: isPreviewNode ? "Connect audio and queue once to preview." : "Choose file to upload or record from microphone.",
        isLoading: false, isRecording: false, recordingObjectUrl: null,
        dragMode: null, dragAnchorSeconds: 0, dragStartLeft: 0, dragStartRight: 0, rafId: 0, mediaReady: false,
    };

    const container = document.createElement("div");
    container.className = "ts-audio-loader";
    const topbar = document.createElement("div");
    topbar.className = "ts-audio-loader__topbar";
    const actions = document.createElement("div");
    actions.className = "ts-audio-loader__actions";
    let loadButton = null;
    let recordButton = null;
    if (!isPreviewNode) {
        loadButton = document.createElement("button");
        loadButton.className = "ts-audio-loader__button is-primary";
        loadButton.textContent = "Load Audio";
        recordButton = document.createElement("button");
        recordButton.className = "ts-audio-loader__button";
        recordButton.textContent = "Start Record";
        actions.append(loadButton, recordButton);
    }
    const resetCropButton = document.createElement("button");
    resetCropButton.className = "ts-audio-loader__button";
    resetCropButton.textContent = "Reset Crop";
    actions.append(resetCropButton);
    topbar.append(actions);

    const meta = document.createElement("div");
    meta.className = "ts-audio-loader__meta";
    const fileLabel = document.createElement("div");
    fileLabel.className = "ts-audio-loader__file";
    fileLabel.textContent = "No file selected";
    const stats = document.createElement("div");
    stats.className = "ts-audio-loader__stats";
    const statusLabel = document.createElement("div");
    statusLabel.className = "ts-audio-loader__status";
    statusLabel.textContent = state.status;
    meta.append(fileLabel, stats, statusLabel);

    const waveWrap = document.createElement("div");
    waveWrap.className = "ts-audio-loader__wave-wrap";
    const canvas = document.createElement("canvas");
    canvas.className = "ts-audio-loader__canvas";
    const empty = document.createElement("div");
    empty.className = "ts-audio-loader__empty";
    empty.textContent = state.status;
    const audioEl = document.createElement("audio");
    audioEl.className = "ts-audio-loader__hidden-media";
    audioEl.preload = "metadata";
    const videoEl = document.createElement("video");
    videoEl.className = "ts-audio-loader__hidden-media";
    videoEl.preload = "metadata";
    const fileInput = !isPreviewNode ? document.createElement("input") : null;
    if (fileInput) {
        fileInput.className = "ts-audio-loader__hidden-media";
        fileInput.type = "file";
        fileInput.accept = MEDIA_UPLOAD_ACCEPT;
        waveWrap.append(canvas, empty, audioEl, videoEl, fileInput);
    } else {
        waveWrap.append(canvas, empty, audioEl, videoEl);
    }

    const bottom = document.createElement("div");
    bottom.className = "ts-audio-loader__bottom";
    const transport = document.createElement("div");
    transport.className = "ts-audio-loader__transport";
    const playButton = document.createElement("button");
    playButton.className = "ts-audio-loader__play";
    playButton.innerHTML = playIcon();
    const loopButton = document.createElement("button");
    loopButton.className = "ts-audio-loader__play";
    loopButton.innerHTML = loopIcon();
    loopButton.title = "Loop playback";
    const timelineLabel = document.createElement("div");
    timelineLabel.className = "ts-audio-loader__timeline";
    timelineLabel.textContent = "00:00.00 / 00:00.00";
    transport.append(playButton, loopButton, timelineLabel);
    const cropLabel = document.createElement("div");
    cropLabel.className = "ts-audio-loader__crop";
    cropLabel.textContent = "Crop: full";
    bottom.append(transport, document.createElement("div"), cropLabel);
    container.append(topbar, meta, waveWrap, bottom);

    stopPropagation(container, ["pointerdown", "pointerup", "mousedown", "mouseup", "mousemove", "wheel", "click", "dblclick", "contextmenu"]);
    const widgetOptions = { serialize: false, hideOnZoom: false };
    if (isNodesV2()) {
        widgetOptions.getMinHeight = () => MIN_NODE_HEIGHT - HEADER_FOOTER_HEIGHT;
        widgetOptions.afterResize = () => { syncDomSize(); drawWaveform(); };
    }
    const domWidget = node.addDOMWidget(DOM_WIDGET_NAME, "div", container, widgetOptions);
    const domWidgetEl = domWidget?.element || domWidget?.el || domWidget?.container;

    function getActiveMedia() { return state.mediaType === "video" ? videoEl : audioEl; }
    function getSelectionBounds() {
        const duration = Math.max(0, state.duration);
        const left = clamp(state.cropStart || 0, 0, duration);
        let right = Number.isFinite(state.cropEnd) && state.cropEnd > left ? state.cropEnd : duration;
        right = clamp(right, left, duration);
        return { left, right };
    }
    function syncDomSize() {
        const width = Math.max(MIN_NODE_WIDTH, Number(node.size?.[0]) || DEFAULT_NODE_SIZE[0]);
        const height = Math.max(MIN_NODE_HEIGHT, Number(node.size?.[1]) || DEFAULT_NODE_SIZE[1]);
        node.size = [width, height];
        const targetHeight = Math.max(132, height - 88);
        if (domWidgetEl) {
            domWidgetEl.style.width = "100%";
            domWidgetEl.style.height = `${targetHeight}px`;
            domWidgetEl.style.overflow = "hidden";
        }
        container.style.width = "100%";
        container.style.height = "100%";
    }
    function isAnyMediaPlaying() {
        return isMediaPlaying(audioEl) || isMediaPlaying(videoEl);
    }
    function pauseAllMedia() {
        [audioEl, videoEl].forEach((media) => {
            if (!media) return;
            try { media.pause(); } catch {}
        });
    }
    function updateModeButtons() {
        if (!recordButton) return;
        recordButton.classList.toggle("is-danger", state.isRecording);
        recordButton.textContent = state.isRecording ? "Stop Record" : "Start Record";
    }
    function updateCanvasCursor(pointerSeconds = null) {
        if (state.dragMode === "left" || state.dragMode === "right") {
            canvas.style.cursor = HANDLE_ACTIVE_CURSOR;
            return;
        }
        if (pointerSeconds != null && state.duration > 0 && hitTestHandle(pointerSeconds)) {
            canvas.style.cursor = HANDLE_CURSOR;
            return;
        }
        canvas.style.cursor = DEFAULT_WAVE_CURSOR;
    }
    function updateText() {
        fileLabel.textContent = state.filename || "No file selected";
        statusLabel.textContent = state.status;
        empty.textContent = state.isLoading ? "Loading waveform..." : state.status;
        empty.style.display = state.peaks.length > 0 ? "none" : "flex";
        const activeMedia = getActiveMedia();
        timelineLabel.textContent = `${formatSeconds(activeMedia?.currentTime || 0)} / ${formatSeconds(state.duration)}`;
        const bounds = getSelectionBounds();
        const cropDuration = Math.max(0, bounds.right - bounds.left);
        cropLabel.textContent = state.duration > 0
            ? `Crop: ${formatSeconds(bounds.left)} -> ${formatSeconds(bounds.right)} | Length: ${formatSeconds(cropDuration)}`
            : "Crop: full";
        stats.innerHTML = "";
        const items = [];
        if (state.mediaType) items.push(state.mediaType === "video" ? "video audio" : "audio");
        if (state.sampleRate) items.push(`${state.sampleRate} Hz`);
        if (state.channels) items.push(`${state.channels} ch`);
        if (state.duration) items.push(formatSeconds(state.duration));
        items.forEach((item) => {
            const span = document.createElement("span");
            span.textContent = item;
            stats.appendChild(span);
        });
        playButton.innerHTML = isAnyMediaPlaying() ? pauseIcon() : playIcon();
        loopButton.classList.toggle("is-active", state.isLooping);
        updateModeButtons();
    }
    function syncWidgets() {
        if (!isPreviewNode) {
            setWidgetValue(node, INPUT_MODE, state.mode);
            setWidgetValue(node, INPUT_SOURCE_PATH, state.sourcePath);
        }
        setWidgetValue(node, INPUT_CROP_START, Number(state.cropStart || 0));
        const bounds = getSelectionBounds();
        const storedEnd = state.duration > 0 && bounds.right >= state.duration - 0.01 ? -1 : Number(bounds.right);
        state.cropEnd = storedEnd;
        setWidgetValue(node, INPUT_CROP_END, storedEnd);
        node.properties.ts_audio_loader_loop = state.isLooping;
        scheduleCanvasDirty();
    }
    function persistPreviewState(payload) {
        if (!isPreviewNode) return;
        setWidgetValue(node, INPUT_PREVIEW_STATE, JSON.stringify(payload));
    }
    function clearRecordingObjectUrl() {
        if (!state.recordingObjectUrl) return;
        URL.revokeObjectURL(state.recordingObjectUrl);
        state.recordingObjectUrl = null;
    }
    function applyMediaPayload(payload, options = {}) {
        const mediaPath = String(payload?.preview_path || payload?.filepath || "");
        clearRecordingObjectUrl();
        state.duration = Number(payload?.duration_seconds || 0);
        state.sampleRate = Number(payload?.sample_rate || 0);
        state.channels = Number(payload?.channels || 0);
        state.peaks = Array.isArray(payload?.peaks) ? payload.peaks : [];
        state.filename = payload?.filename || (mediaPath ? mediaPath.split("/").pop() : "") || "Incoming audio";
        state.mediaType = payload?.media_type || "audio";
        state.status = state.peaks.length > 0 ? "Drag on waveform to crop. Double-click resets full range." : "Media loaded.";
        state.mediaReady = Boolean(mediaPath);
        if (!mediaPath) {
            audioEl.removeAttribute("src");
            videoEl.removeAttribute("src");
            audioEl.load();
            videoEl.load();
            updateText();
            drawWaveform();
            return;
        }
        const mediaUrl = api.apiURL(`${ROUTE_BASE}/view?filepath=${encodeURIComponent(mediaPath)}`);
        if (state.mediaType === "video") {
            videoEl.src = mediaUrl;
            videoEl.load();
            audioEl.removeAttribute("src");
            audioEl.load();
        } else {
            audioEl.src = mediaUrl;
            audioEl.load();
            videoEl.removeAttribute("src");
            videoEl.load();
        }
        if (!(state.cropEnd > state.cropStart)) {
            state.cropStart = 0;
            state.cropEnd = -1;
        }
        if (options.persist !== false) {
            persistPreviewState(payload);
        }
        syncWidgets();
        updateText();
        drawWaveform();
    }
    function restorePreviewState() {
        if (!isPreviewNode) return;
        const raw = String(getWidgetValue(node, INPUT_PREVIEW_STATE, "") || "");
        if (!raw) return;
        try {
            const payload = JSON.parse(raw);
            applyMediaPayload(payload, { persist: false });
        } catch {}
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
        return { width, height, dpr };
    }
    function drawWaveform() {
        const { width, height, dpr } = resizeCanvas();
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, width, height);
        ctx.scale(dpr, dpr);
        const drawWidth = width / dpr;
        const drawHeight = height / dpr;
        const midY = drawHeight / 2;
        ctx.fillStyle = "#121821";
        ctx.fillRect(0, 0, drawWidth, drawHeight);
        ctx.strokeStyle = "rgba(255,255,255,0.06)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(drawWidth, midY);
        ctx.stroke();
        if (!state.peaks.length || state.duration <= 0) {
            updateText();
            return;
        }
        const bounds = getSelectionBounds();
        const selectionStartX = (bounds.left / state.duration) * drawWidth;
        const selectionEndX = (bounds.right / state.duration) * drawWidth;
        ctx.fillStyle = "rgba(0, 0, 0, 0.28)";
        ctx.fillRect(0, 0, selectionStartX, drawHeight);
        ctx.fillRect(selectionEndX, 0, drawWidth - selectionEndX, drawHeight);
        const step = drawWidth / state.peaks.length;
        const playheadX = state.duration > 0 ? ((getActiveMedia()?.currentTime || 0) / state.duration) * drawWidth : 0;
        for (let index = 0; index < state.peaks.length; index += 1) {
            const peak = clamp(Number(state.peaks[index]) || 0, 0, 1);
            const x = index * step;
            const barWidth = Math.max(1, step - 1);
            const barHeight = Math.max(2, peak * (drawHeight * 0.46));
            const insideSelection = x + barWidth >= selectionStartX && x <= selectionEndX;
            ctx.fillStyle = insideSelection ? "#29c7a3" : "rgba(145, 160, 180, 0.38)";
            ctx.fillRect(x, midY - barHeight, barWidth, barHeight * 2);
        }
        ctx.fillStyle = "rgba(41, 199, 163, 0.14)";
        ctx.fillRect(selectionStartX, 0, Math.max(0, selectionEndX - selectionStartX), drawHeight);
        ctx.strokeStyle = "#b9fff1";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(selectionStartX, 0); ctx.lineTo(selectionStartX, drawHeight);
        ctx.moveTo(selectionEndX, 0); ctx.lineTo(selectionEndX, drawHeight);
        ctx.stroke();
        ctx.fillStyle = "#d3fff6";
        ctx.fillRect(selectionStartX - 2, 12, 4, drawHeight - 24);
        ctx.fillRect(selectionEndX - 2, 12, 4, drawHeight - 24);
        ctx.strokeStyle = "#f4fff9";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0); ctx.lineTo(playheadX, drawHeight);
        ctx.stroke();
        updateText();
    }
    function scheduleDrawLoop() {
        if (state.rafId) cancelAnimationFrame(state.rafId);
        const tick = () => {
            drawWaveform();
            if (getActiveMedia()?.paused === false) state.rafId = requestAnimationFrame(tick);
            else state.rafId = 0;
        };
        state.rafId = requestAnimationFrame(tick);
    }
    async function fetchMetadata(filepath) {
        if (!filepath) {
            state.peaks = [];
            state.duration = 0;
            state.sampleRate = 0;
            state.channels = 0;
            state.filename = "";
            state.status = isPreviewNode ? "Connect audio and queue once to preview." : "Choose file to upload or record from microphone.";
            state.mediaType = "audio";
            state.mediaReady = false;
            audioEl.removeAttribute("src");
            videoEl.removeAttribute("src");
            audioEl.load();
            videoEl.load();
            updateText();
            drawWaveform();
            return;
        }
        state.isLoading = true;
        state.status = "Loading waveform...";
        updateText();
        try {
            const response = await api.fetchApi(`${ROUTE_BASE}/metadata?filepath=${encodeURIComponent(filepath)}`);
            const payload = await response.json();
            if (!response.ok) throw new Error(payload?.error || "Failed to load media metadata.");
            applyMediaPayload(payload, { persist: false });
        } catch (error) {
            state.peaks = [];
            state.duration = 0;
            state.sampleRate = 0;
            state.channels = 0;
            state.filename = filepath.split("/").pop() || filepath;
            state.status = error?.message || "Failed to load media.";
        } finally {
            state.isLoading = false;
            updateText();
            drawWaveform();
        }
    }
    function buildAnnotatedPath(uploadPayload) {
        const filename = String(uploadPayload?.name || "").trim();
        const uploadType = String(uploadPayload?.type || "input").trim() || "input";
        const subfolder = String(uploadPayload?.subfolder || "").trim().replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
        if (!filename) return "";
        return subfolder ? `${subfolder}/${filename} [${uploadType}]` : `${filename} [${uploadType}]`;
    }
    async function uploadSelectedSource(file) {
        const form = new FormData();
        form.append("image", file, file.name);
        form.append("type", "input");
        const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload?.error || payload?.message || "Upload failed.");
        return buildAnnotatedPath(payload);
    }
    async function chooseSourceFile(file) {
        if (!fileInput) return;
        if (!file) return;
        state.isLoading = true;
        state.status = "Uploading media...";
        updateText();
        try {
            const uploadedPath = await uploadSelectedSource(file);
            if (!uploadedPath) throw new Error("Upload failed.");
            state.mode = "load";
            state.sourcePath = uploadedPath;
            state.recordedPath = "";
            state.cropStart = 0;
            state.cropEnd = -1;
            syncWidgets();
            await fetchMetadata(state.sourcePath);
        } catch (error) {
            state.status = error?.message || "Failed to upload media.";
            updateText();
        } finally {
            fileInput.value = "";
            state.isLoading = false;
            updateText();
        }
    }
    let mediaStream = null;
    let mediaRecorder = null;
    let recordChunks = [];
    async function decodeAudioBlob(blob) {
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) throw new Error("AudioContext unavailable.");
        const audioContext = new AudioContextCtor();
        try {
            const arrayBuffer = await blob.arrayBuffer();
            return await audioContext.decodeAudioData(arrayBuffer.slice(0));
        } finally {
            await audioContext.close().catch(() => {});
        }
    }
    function audioBufferToWavBlob(audioBuffer) {
        const channelCount = Math.max(1, audioBuffer.numberOfChannels || 1);
        const sampleRate = Math.max(1, audioBuffer.sampleRate || 44100);
        const frameCount = Math.max(0, audioBuffer.length || 0);
        const bytesPerSample = 2;
        const blockAlign = channelCount * bytesPerSample;
        const dataSize = frameCount * blockAlign;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);
        let offset = 0;
        const writeString = (value) => {
            for (let index = 0; index < value.length; index += 1) {
                view.setUint8(offset, value.charCodeAt(index));
                offset += 1;
            }
        };
        writeString("RIFF");
        view.setUint32(offset, 36 + dataSize, true); offset += 4;
        writeString("WAVE");
        writeString("fmt ");
        view.setUint32(offset, 16, true); offset += 4;
        view.setUint16(offset, 1, true); offset += 2;
        view.setUint16(offset, channelCount, true); offset += 2;
        view.setUint32(offset, sampleRate, true); offset += 4;
        view.setUint32(offset, sampleRate * blockAlign, true); offset += 4;
        view.setUint16(offset, blockAlign, true); offset += 2;
        view.setUint16(offset, bytesPerSample * 8, true); offset += 2;
        writeString("data");
        view.setUint32(offset, dataSize, true); offset += 4;
        const channels = [];
        for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
            channels.push(audioBuffer.getChannelData(channelIndex));
        }
        for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
            for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
                const sample = clamp(channels[channelIndex][frameIndex] || 0, -1, 1);
                const pcm = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
                view.setInt16(offset, Math.round(pcm), true);
                offset += 2;
            }
        }
        return new Blob([buffer], { type: "audio/wav" });
    }
    function buildPeaksFromAudioBuffer(audioBuffer, targetBins = 2048) {
        const channelCount = Math.max(1, audioBuffer.numberOfChannels || 1);
        const sampleCount = audioBuffer.length || 0;
        if (!sampleCount) return [0];
        const mono = new Float32Array(sampleCount);
        for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
            const channelData = audioBuffer.getChannelData(channelIndex);
            for (let index = 0; index < sampleCount; index += 1) {
                mono[index] += Math.abs(channelData[index]) / channelCount;
            }
        }
        if (mono.length <= targetBins) return Array.from(mono, (value) => clamp(value, 0, 1));
        const peaks = [];
        const samplesPerBin = Math.max(1, Math.ceil(mono.length / targetBins));
        for (let offset = 0; offset < mono.length; offset += samplesPerBin) {
            let peak = 0;
            const end = Math.min(mono.length, offset + samplesPerBin);
            for (let index = offset; index < end; index += 1) {
                if (mono[index] > peak) peak = mono[index];
            }
            peaks.push(clamp(peak, 0, 1));
        }
        return peaks;
    }
    async function previewRecording(blob, filename, decoded = null) {
        clearRecordingObjectUrl();
        const objectUrl = URL.createObjectURL(blob);
        state.recordingObjectUrl = objectUrl;
        state.mediaType = "audio";
        state.filename = filename;
        state.status = "Processing recording...";
        state.mediaReady = true;
        audioEl.src = objectUrl;
        audioEl.load();
        videoEl.removeAttribute("src");
        videoEl.load();
        try {
            const decodedBuffer = decoded || await decodeAudioBlob(blob);
            state.duration = Number(decodedBuffer.duration || 0);
            state.sampleRate = Number(decodedBuffer.sampleRate || 0);
            state.channels = Number(decodedBuffer.numberOfChannels || 0);
            state.peaks = buildPeaksFromAudioBuffer(decodedBuffer);
            state.status = "Drag on waveform to crop. Double-click resets full range.";
            state.cropStart = 0;
            state.cropEnd = -1;
            updateText();
            drawWaveform();
        } catch {
            state.status = "Recording saved. Loading waveform...";
            updateText();
        }
    }
    async function stopRecording() {
        if (!mediaRecorder) return;
        const recorder = mediaRecorder;
        mediaRecorder = null;
        state.isRecording = false;
        updateText();
        await new Promise((resolve) => { recorder.addEventListener("stop", resolve, { once: true }); recorder.stop(); });
        if (mediaStream) { mediaStream.getTracks().forEach((track) => track.stop()); mediaStream = null; }
    }
    async function uploadRecording(blob, filename) {
        const form = new FormData();
        form.append("audio", blob, filename);
        const response = await api.fetchApi(`${ROUTE_BASE}/upload_recording`, { method: "POST", body: form });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload?.error || "Failed to upload recording.");
        return payload?.path || "";
    }
    async function toggleRecording() {
        if (isPreviewNode) return;
        if (state.isRecording) { await stopRecording(); return; }
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            recordChunks = [];
            mediaRecorder = new MediaRecorder(mediaStream);
            const recordingMimeType = mediaRecorder.mimeType || "audio/webm";
            mediaRecorder.addEventListener("dataavailable", (event) => { if (event.data?.size) recordChunks.push(event.data); });
            mediaRecorder.addEventListener("stop", async () => {
                try {
                    const blob = new Blob(recordChunks, { type: recordingMimeType });
                    let uploadBlob = blob;
                    let decodedBuffer = null;
                    try {
                        decodedBuffer = await decodeAudioBlob(blob);
                        uploadBlob = audioBufferToWavBlob(decodedBuffer);
                    } catch {}
                    await previewRecording(uploadBlob, "Microphone Recording.wav", decodedBuffer);
                    const savedPath = await uploadRecording(uploadBlob, "recording.wav");
                    if (savedPath) {
                        state.mode = "record";
                        state.sourcePath = savedPath;
                        state.recordedPath = savedPath;
                        state.cropStart = 0;
                        state.cropEnd = -1;
                        syncWidgets();
                        await fetchMetadata(savedPath);
                    }
                } catch (error) {
                    state.status = error?.message || "Failed to save recording.";
                    updateText();
                }
            });
            mediaRecorder.start();
            state.isRecording = true;
            state.status = "Recording from microphone...";
            updateText();
        } catch (error) {
            state.isRecording = false;
            state.status = error?.message || "Microphone access denied.";
            updateText();
        }
    }
    function seekTo(seconds) {
        const media = getActiveMedia();
        if (!media || !state.mediaReady || state.duration <= 0) return;
        media.currentTime = clamp(seconds, 0, state.duration);
        drawWaveform();
    }
    async function togglePlay() {
        const media = getActiveMedia();
        if (!media || !state.mediaReady || !media.src) return;
        const bounds = getSelectionBounds();
        if (isAnyMediaPlaying()) {
            pauseAllMedia();
        } else {
            if (media.currentTime < bounds.left || media.currentTime > bounds.right) media.currentTime = bounds.left;
            try { await media.play(); scheduleDrawLoop(); } catch { state.status = "Browser blocked autoplay. Press play again."; }
        }
        updateText();
    }
    function toggleLoop() {
        state.isLooping = !state.isLooping;
        syncWidgets();
        updateText();
    }
    function updateSelectionFromSeconds(left, right) {
        const duration = Math.max(0, state.duration);
        if (duration <= 0) return;
        const clampedLeft = clamp(left, 0, duration);
        const clampedRight = clamp(right, clampedLeft, duration);
        state.cropStart = clampedLeft;
        state.cropEnd = clampedRight >= duration - 0.01 ? -1 : clampedRight;
        syncWidgets();
        drawWaveform();
    }
    function canvasSecondsFromPointer(event) {
        const rect = canvas.getBoundingClientRect();
        const x = clamp(event.clientX - rect.left, 0, rect.width);
        return (rect.width > 0 ? x / rect.width : 0) * state.duration;
    }
    function hitTestHandle(pointerSeconds) {
        const rect = canvas.getBoundingClientRect();
        const secondsPerPixel = rect.width > 0 ? state.duration / rect.width : state.duration;
        const hitSeconds = HANDLE_HITBOX * secondsPerPixel;
        const bounds = getSelectionBounds();
        if (Math.abs(pointerSeconds - bounds.left) <= hitSeconds) return "left";
        if (Math.abs(pointerSeconds - bounds.right) <= hitSeconds) return "right";
        return null;
    }
    function onCanvasPointerDown(event) {
        if (state.duration <= 0) return;
        const pointerSeconds = canvasSecondsFromPointer(event);
        const handle = hitTestHandle(pointerSeconds);
        const bounds = getSelectionBounds();
        state.dragMode = handle === "left" || handle === "right" ? handle : "pending-range";
        state.dragAnchorSeconds = pointerSeconds;
        state.dragStartLeft = bounds.left;
        state.dragStartRight = bounds.right;
        updateCanvasCursor(pointerSeconds);
        canvas.setPointerCapture?.(event.pointerId);
    }
    function onCanvasPointerMove(event) {
        const pointerSeconds = canvasSecondsFromPointer(event);
        if (!state.dragMode || state.duration <= 0) {
            updateCanvasCursor(pointerSeconds);
            return;
        }
        if (state.dragMode === "left") { updateSelectionFromSeconds(pointerSeconds, getSelectionBounds().right); return; }
        if (state.dragMode === "right") { updateSelectionFromSeconds(getSelectionBounds().left, pointerSeconds); return; }
        if (state.dragMode === "pending-range") {
            if (Math.abs(pointerSeconds - state.dragAnchorSeconds) >= (state.duration / Math.max(80, canvas.clientWidth || 80))) state.dragMode = "range";
            else return;
        }
        if (state.dragMode === "range") updateSelectionFromSeconds(Math.min(state.dragAnchorSeconds, pointerSeconds), Math.max(state.dragAnchorSeconds, pointerSeconds));
    }
    function onCanvasPointerUp(event) {
        if (!state.dragMode) return;
        const pointerSeconds = canvasSecondsFromPointer(event);
        if (state.dragMode === "pending-range") seekTo(pointerSeconds);
        else if (state.dragMode === "range") updateSelectionFromSeconds(Math.min(state.dragAnchorSeconds, pointerSeconds), Math.max(state.dragAnchorSeconds, pointerSeconds));
        state.dragMode = null;
        updateCanvasCursor(pointerSeconds);
        canvas.releasePointerCapture?.(event.pointerId);
    }
    function resetCrop() { state.cropStart = 0; state.cropEnd = -1; syncWidgets(); drawWaveform(); }
    canvas.addEventListener("pointerdown", onCanvasPointerDown);
    canvas.addEventListener("pointermove", onCanvasPointerMove);
    canvas.addEventListener("pointerup", onCanvasPointerUp);
    canvas.addEventListener("pointerleave", () => {
        if (!state.dragMode) updateCanvasCursor(null);
    });
    canvas.addEventListener("dblclick", () => resetCrop());
    if (loadButton && fileInput) {
        loadButton.addEventListener("click", () => { fileInput.click(); });
    }
    if (recordButton) {
        recordButton.addEventListener("click", async () => { await toggleRecording(); });
    }
    resetCropButton.addEventListener("click", () => resetCrop());
    playButton.addEventListener("click", async () => { await togglePlay(); });
    loopButton.addEventListener("click", () => { toggleLoop(); });
    if (fileInput) {
        fileInput.addEventListener("change", async () => {
            const [selectedFile] = Array.from(fileInput.files || []);
            await chooseSourceFile(selectedFile);
        });
    }
    [audioEl, videoEl].forEach((media) => {
        media.addEventListener("timeupdate", () => {
            const bounds = getSelectionBounds();
            if (bounds.right > bounds.left && media.currentTime >= bounds.right) {
                if (state.isLooping) {
                    media.currentTime = bounds.left;
                    if (media.paused) {
                        media.play().catch(() => {});
                    }
                } else {
                    media.pause();
                    media.currentTime = bounds.right;
                }
            }
            drawWaveform();
        });
        media.addEventListener("play", () => { scheduleDrawLoop(); updateText(); });
        media.addEventListener("pause", () => { updateText(); drawWaveform(); });
        media.addEventListener("loadedmetadata", () => drawWaveform());
        media.addEventListener("ended", () => {
            const bounds = getSelectionBounds();
            if (!state.isLooping) {
                updateText();
                drawWaveform();
                return;
            }
            media.currentTime = bounds.left;
            media.play().catch(() => {});
        });
        media.addEventListener("error", () => { state.status = "Browser preview is unavailable for this codec, but ffmpeg loading still works."; updateText(); });
    });
    const previousOnResize = node.onResize;
    node.onResize = function onResize() {
        const result = previousOnResize?.apply(this, arguments);
        syncDomSize();
        drawWaveform();
        return result;
    };
    const resizeObserver = new ResizeObserver(() => drawWaveform());
    resizeObserver.observe(container);
    const sourceWidgetPoll = !isPreviewNode ? window.setInterval(() => {
        const nextSourcePath = String(getWidgetValue(node, INPUT_SOURCE_PATH, "") || "");
        if (nextSourcePath === state.sourcePath) return;
        if (!nextSourcePath && state.mode === "record" && state.recordedPath) return;
        state.sourcePath = nextSourcePath;
        state.mode = "load";
        state.recordedPath = "";
        state.cropStart = 0;
        state.cropEnd = -1;
        syncWidgets();
        fetchMetadata(state.sourcePath);
    }, 300) : null;
    node._tsAudioLoaderCleanup = () => {
        resizeObserver.disconnect();
        if (sourceWidgetPoll) window.clearInterval(sourceWidgetPoll);
        if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = 0; }
        clearRecordingObjectUrl();
        [audioEl, videoEl].forEach((media) => { media.pause(); media.removeAttribute("src"); media.load(); });
        if (mediaRecorder && state.isRecording) { try { mediaRecorder.stop(); } catch {} }
        if (mediaStream) { mediaStream.getTracks().forEach((track) => track.stop()); mediaStream = null; }
    };
    node._tsAudioLoaderApplyPayload = (payload, persist = true) => {
        if (!payload) return;
        applyMediaPayload(payload, { persist });
    };
    requestAnimationFrame(() => {
        syncDomSize();
        updateText();
        drawWaveform();
        if (isPreviewNode) {
            restorePreviewState();
        } else if (state.sourcePath) {
            fetchMetadata(state.sourcePath);
        }
    });
}

