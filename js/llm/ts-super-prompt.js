// TS_SuperPrompt frontend — full DOM-rendered UI built around a single
// flex-grow textarea. The schema still exposes ``text``, ``high_quality``,
// ``system_preset`` and ``attached_image`` so workflows serialise correctly,
// but the standard ComfyUI widgets are hidden — every control lives inside
// the DOM widget so the visual stack stays consistent.
//
// Layout (top → bottom inside the DOM widget):
//   • compact toolbar (~30px): attach button (shows mini thumbnail when set),
//     high-quality toggle, preset select, record button, AI prompt button;
//   • prompt textarea (flex: 1, the main UI surface);
//   • thin status / progress strip (~16px).

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_ID = "ts.superPrompt";
const NODE_NAME = "TS_SuperPrompt";
const DOM_WIDGET_NAME = "ts_super_prompt_ui";

const VOICE_ROUTE_BASE = "/ts_voice_recognition";
const AI_ROUTE_BASE = "/ts_super_prompt";
const UPLOAD_ROUTE = "/upload/image";
const VOICE_EVENT_PREFIX = "ts_voice_recognition";
const AI_EVENT_PREFIX = "ts_super_prompt";

const TEXT_WIDGET = "text";
const HIGH_QUALITY_WIDGET = "high_quality";
const SYSTEM_PRESET_WIDGET = "system_preset";
const ATTACHED_IMAGE_WIDGET = "attached_image";

const DEFAULT_MODEL = "base";
const HIGH_QUALITY_MODEL = "turbo";
const AUDIO_BITS_PER_SECOND = 128_000;
const PROGRESS_CLEAR_DELAY_MS = 900;
const STATUS_RESET_DELAY_MS = 2400;

const IMAGE_ACCEPT = "image/*";
const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"]);
const MIME_CANDIDATES = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
];

const STYLE_ID = "ts-super-prompt-style";
const STYLE_TEXT = `
.ts-sp{position:absolute;inset:0;display:flex;flex-direction:column;gap:4px;padding:4px;
    background:rgba(20,24,32,.55);border:1px solid rgba(255,255,255,.06);border-radius:8px;
    color:#e6e9ef;font-family:inherit;font-size:11px;line-height:1.3;box-sizing:border-box;
    backdrop-filter:blur(4px);}
.ts-sp.is-drag-over{outline:2px dashed #7aa2ff;outline-offset:-3px}
.ts-sp__bar{display:flex;align-items:center;gap:6px;height:26px;flex:0 0 auto}
.ts-sp__group{display:inline-flex;align-items:center;gap:2px;flex:0 0 auto}
.ts-sp__textarea{flex:1 1 auto;min-height:0;width:100%;resize:none;box-sizing:border-box;
    padding:6px 8px;border-radius:6px;border:1px solid rgba(255,255,255,.08);
    background:rgba(0,0,0,.25);color:#e6e9ef;font-family:inherit;font-size:12px;line-height:1.4;
    outline:none;transition:border-color .15s,background .15s}
.ts-sp__textarea:focus{border-color:rgba(122,162,255,.55);background:rgba(0,0,0,.32)}
.ts-sp__textarea::placeholder{color:rgba(230,233,239,.4)}
.ts-sp__btn{display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto;
    width:26px;height:26px;padding:0;border-radius:6px;border:1px solid rgba(255,255,255,.1);
    background:rgba(255,255,255,.04);color:#e6e9ef;cursor:pointer;
    transition:background .15s,border-color .15s,color .15s,transform .08s;user-select:none}
.ts-sp__btn:hover:not(:disabled){background:rgba(122,162,255,.18);border-color:rgba(122,162,255,.45)}
.ts-sp__btn:active:not(:disabled){transform:translateY(1px)}
.ts-sp__btn:disabled{opacity:.5;cursor:not-allowed}
.ts-sp__btn svg{width:14px;height:14px;display:block;fill:none;stroke:currentColor;
    stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.ts-sp__btn--record.is-recording{background:#b3262e;border-color:#ff5a64;color:#fff;
    box-shadow:0 0 0 3px rgba(255,90,100,.18)}
.ts-sp__btn--record.is-recording:hover{background:#c63040}
.ts-sp__pill{display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto;
    height:26px;padding:0 9px;border-radius:6px;border:1px solid rgba(255,255,255,.1);
    background:rgba(255,255,255,.04);color:rgba(230,233,239,.75);font-size:10px;font-weight:700;
    letter-spacing:.4px;cursor:pointer;transition:background .15s,border-color .15s,color .15s,transform .08s;
    user-select:none;font-family:inherit}
.ts-sp__pill:hover:not(:disabled){background:rgba(122,162,255,.18);border-color:rgba(122,162,255,.45);color:#e6e9ef}
.ts-sp__pill:active:not(:disabled){transform:translateY(1px)}
.ts-sp__pill:disabled{opacity:.5;cursor:not-allowed}
.ts-sp__pill--toggle.is-on{background:rgba(122,162,255,.32);border-color:rgba(122,162,255,.6);color:#fff}
.ts-sp__pill--ai{letter-spacing:.6px}
.ts-sp__attach{position:relative;flex:0 0 auto;width:26px;height:26px;border-radius:6px;
    overflow:hidden;border:1px solid rgba(255,255,255,.1);background:rgba(255,255,255,.04);
    color:#e6e9ef;cursor:pointer;transition:background .15s,border-color .15s;padding:0;
    display:inline-flex;align-items:center;justify-content:center}
.ts-sp__attach:hover{background:rgba(122,162,255,.18);border-color:rgba(122,162,255,.45)}
.ts-sp__attach svg{width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:2;
    stroke-linecap:round;stroke-linejoin:round}
.ts-sp__attach.has-image{border-color:rgba(122,162,255,.65)}
.ts-sp__attach-thumb{position:absolute;inset:0;background-position:center;background-size:cover;
    background-repeat:no-repeat;display:none}
.ts-sp__attach.has-image .ts-sp__attach-thumb{display:block}
.ts-sp__attach.has-image .ts-sp__attach-icon{display:none}
.ts-sp__attach-clear{position:absolute;top:-3px;right:-3px;width:14px;height:14px;border-radius:50%;
    border:1px solid rgba(255,255,255,.15);background:#0a0d12;color:#fff;font-size:10px;line-height:1;
    cursor:pointer;display:none;align-items:center;justify-content:center;padding:0;
    box-shadow:0 1px 2px rgba(0,0,0,.5)}
.ts-sp__attach.has-image .ts-sp__attach-clear{display:flex}
.ts-sp__attach-clear:hover{background:#b3262e;border-color:#ff5a64}
.ts-sp__select{flex:1 1 auto;min-width:0;height:26px;padding:0 6px;border-radius:6px;
    border:1px solid rgba(255,255,255,.1);background:rgba(0,0,0,.25);color:#e6e9ef;
    font-size:11px;font-family:inherit;cursor:pointer;outline:none;
    -webkit-appearance:none;-moz-appearance:none;appearance:none;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23e6e9ef' d='M0 0l5 6 5-6z'/%3E%3C/svg%3E");
    background-repeat:no-repeat;background-position:right 6px center;padding-right:18px}
.ts-sp__select:focus{border-color:rgba(122,162,255,.55)}
.ts-sp__select option{background:#1a1f29;color:#e6e9ef}
.ts-sp__status{display:flex;align-items:center;gap:6px;min-height:14px;flex:0 0 auto;
    font-size:10px;color:rgba(230,233,239,.65)}
.ts-sp__status-text{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ts-sp__status.is-error .ts-sp__status-text{color:#ff6b6b}
.ts-sp__progress{flex:0 0 70px;height:3px;border-radius:2px;background:rgba(255,255,255,.08);
    overflow:hidden;display:none;position:relative}
.ts-sp__progress.is-active{display:block}
.ts-sp__progress-fill{height:100%;background:linear-gradient(90deg,#7aa2ff,#a8b8ff);width:0%;
    transition:width .2s ease-out}
.ts-sp__progress.is-indeterminate .ts-sp__progress-fill{
    width:35%;animation:ts-sp-indeterminate 1.1s linear infinite}
@keyframes ts-sp-indeterminate{
    0%{transform:translateX(-100%)}100%{transform:translateX(285%)}
}
.ts-sp__file{position:fixed;left:-9999px;top:-9999px}
`;

const SVG_ICON_MIC = `<svg viewBox="0 0 24 24"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 19v3"/></svg>`;
const SVG_ICON_STOP = `<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" stroke="none"/></svg>`;
const SVG_ICON_IMAGE = `<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="11" r="1.6" fill="currentColor" stroke="none"/><path d="M3 17l5-5 4 4 3-3 6 6"/></svg>`;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function ensureStylesInjected(doc) {
    if (!doc || doc.getElementById(STYLE_ID)) return;
    const styleEl = doc.createElement("style");
    styleEl.id = STYLE_ID;
    styleEl.textContent = STYLE_TEXT;
    doc.head.appendChild(styleEl);
}

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function getWidgetValue(node, name, fallback = null) {
    const widget = getWidget(node, name);
    return widget?.value ?? fallback;
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return false;
    if (widget.value === value) return true;
    widget.value = value;
    if (typeof widget.callback === "function") {
        try {
            widget.callback(value);
        } catch {
            // Some widgets are picky about callback signatures — ignore.
        }
    }
    return true;
}

function hideNativeWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.serializeValue = widget.serializeValue || (() => widget.value);
    // Mute draw + collapse layout slot. Litegraph reserves space for any
    // widget whose computeSize doesn't return ≤0, so we force ``[0, -4]`` —
    // matching the spacing fudge it applies between widget rows.
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    widget.type = widget.type === "hidden" ? widget.type : "hidden";
    if (widget.inputEl) {
        widget.inputEl.style.display = "none";
    }
    if (widget.element) {
        widget.element.style.display = "none";
    }
}

function toBoolean(value) {
    if (typeof value === "boolean") return value;
    if (typeof value === "string") {
        return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
    }
    return Boolean(value);
}

function setDirty(node) {
    node?.setDirtyCanvas?.(true, true);
    app?.graph?.setDirtyCanvas?.(true, true);
}

async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.error || response.statusText || `HTTP ${response.status}`);
    }
    return data;
}

function createAudioRecorder(stream, mimeType) {
    const options = { audioBitsPerSecond: AUDIO_BITS_PER_SECOND };
    if (mimeType) options.mimeType = mimeType;
    try {
        return new MediaRecorder(stream, options);
    } catch {
        return mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    }
}

function buildAnnotatedPath(uploadPayload) {
    const filename = String(uploadPayload?.name || "").trim();
    if (!filename) return "";
    const uploadType = String(uploadPayload?.type || "input").trim() || "input";
    const subfolder = String(uploadPayload?.subfolder || "")
        .trim()
        .replace(/\\/g, "/")
        .replace(/^\/+|\/+$/g, "");
    return subfolder ? `${subfolder}/${filename} [${uploadType}]` : `${filename} [${uploadType}]`;
}

function resolveAnnotatedThumbUrl(annotatedPath) {
    if (!annotatedPath) return "";
    const match = annotatedPath.match(/^(.+?)\s*\[([^\]]+)\]\s*$/);
    if (!match) return "";
    const rawPath = match[1].trim();
    const type = match[2].trim() || "input";
    const segments = rawPath.split("/").filter(Boolean);
    if (segments.length === 0) return "";
    const filename = segments.pop();
    const subfolder = segments.join("/");
    const params = new URLSearchParams({ filename, type });
    if (subfolder) params.set("subfolder", subfolder);
    params.set("t", String(Date.now()));
    return `/view?${params.toString()}`;
}

function fileExtensionOk(name) {
    const idx = String(name || "").lastIndexOf(".");
    if (idx < 0) return false;
    return IMAGE_EXTS.has(name.slice(idx).toLowerCase());
}

function clampPercent(value) {
    if (!Number.isFinite(Number(value))) return null;
    return Math.max(0, Math.min(100, Number(value)));
}

function getPresetOptions(node) {
    const widget = getWidget(node, SYSTEM_PRESET_WIDGET);
    const values = widget?.options?.values;
    if (Array.isArray(values) && values.length) return values;
    const current = widget?.value;
    return current ? [String(current)] : ["Prompts enhance"];
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

function setupSuperPrompt(node) {
    if (!node) return;
    if (typeof node._tsSuperPromptCleanup === "function") {
        node._tsSuperPromptCleanup();
    }

    // Hide every native widget — the DOM widget renders all controls.
    for (const widgetName of [TEXT_WIDGET, HIGH_QUALITY_WIDGET, SYSTEM_PRESET_WIDGET, ATTACHED_IMAGE_WIDGET]) {
        hideNativeWidget(getWidget(node, widgetName));
    }

    const doc = node?.graph?.canvas?.canvas?.ownerDocument || document;
    ensureStylesInjected(doc);

    const container = doc.createElement("div");
    container.className = "ts-sp";
    container.setAttribute("data-ts-super-prompt", "1");

    // -------- Toolbar --------
    const bar = doc.createElement("div");
    bar.className = "ts-sp__bar";

    // Attach button (paperclip / thumb).
    const attachBtn = doc.createElement("button");
    attachBtn.type = "button";
    attachBtn.className = "ts-sp__attach";
    attachBtn.title = "Прикрепить изображение (drop / paste / click)";
    const attachIcon = doc.createElement("span");
    attachIcon.className = "ts-sp__attach-icon";
    attachIcon.innerHTML = SVG_ICON_IMAGE;
    const attachThumb = doc.createElement("span");
    attachThumb.className = "ts-sp__attach-thumb";
    const attachClear = doc.createElement("button");
    attachClear.type = "button";
    attachClear.className = "ts-sp__attach-clear";
    attachClear.textContent = "×";
    attachClear.title = "Убрать изображение";
    attachBtn.append(attachIcon, attachThumb, attachClear);

    // ---- Voice group: HQ toggle + record button (both speak to Whisper). ----
    const voiceGroup = doc.createElement("div");
    voiceGroup.className = "ts-sp__group";

    const hqToggle = doc.createElement("button");
    hqToggle.type = "button";
    hqToggle.className = "ts-sp__pill ts-sp__pill--toggle";
    hqToggle.title =
        "High Quality voice: Whisper turbo (large-v3 turbo). Off: быстрая base.";
    hqToggle.textContent = "HQ";

    const recordBtn = doc.createElement("button");
    recordBtn.type = "button";
    recordBtn.className = "ts-sp__btn ts-sp__btn--record";
    recordBtn.title =
        "Запись с микрофона. Нажмите ещё раз во время записи, чтобы остановить и распознать.";
    recordBtn.innerHTML = SVG_ICON_MIC;

    // Mic first (primary action), HQ flag right next to it.
    voiceGroup.append(recordBtn, hqToggle);

    // ---- AI pill (text label, accent gradient). ----
    const aiBtn = doc.createElement("button");
    aiBtn.type = "button";
    aiBtn.className = "ts-sp__pill ts-sp__pill--ai";
    aiBtn.title =
        "Улучшает текст через Huihui-Qwen3.5-2B-abliterated. Если прикреплена картинка — она используется как референс.";
    aiBtn.textContent = "AI";

    // ---- Preset select (fills remaining toolbar space). ----
    const presetSelect = doc.createElement("select");
    presetSelect.className = "ts-sp__select";
    presetSelect.title = "Системный пресет для улучшения промпта.";
    for (const opt of getPresetOptions(node)) {
        const option = doc.createElement("option");
        option.value = String(opt);
        option.textContent = String(opt);
        presetSelect.appendChild(option);
    }

    // Order: [🎤 + HQ] voice · [🖼 attach] · [AI] · [preset ▼]
    // Mic is the primary input action so it leads; HQ sits with it as the
    // voice-quality flag. The image button visually groups with AI because
    // both feed the prompt-enhance pipeline. Preset stretches to fill.
    bar.append(voiceGroup, attachBtn, aiBtn, presetSelect);

    // -------- Textarea (main surface) --------
    const textarea = doc.createElement("textarea");
    textarea.className = "ts-sp__textarea";
    textarea.placeholder =
        "Промпт. Используйте микрофон, прикрепите картинку и нажмите Ai prompt для улучшения.";
    textarea.spellcheck = false;

    // -------- Status row --------
    const statusRow = doc.createElement("div");
    statusRow.className = "ts-sp__status";
    const statusText = doc.createElement("span");
    statusText.className = "ts-sp__status-text";
    statusText.textContent = "Ready";
    const progress = doc.createElement("div");
    progress.className = "ts-sp__progress";
    const progressFill = doc.createElement("div");
    progressFill.className = "ts-sp__progress-fill";
    progress.appendChild(progressFill);
    statusRow.append(statusText, progress);

    // Hidden file input for the attach picker.
    const fileInput = doc.createElement("input");
    fileInput.type = "file";
    fileInput.accept = IMAGE_ACCEPT;
    fileInput.className = "ts-sp__file";

    container.append(bar, textarea, statusRow, fileInput);

    // -----------------------------------------------------------------
    // State
    // -----------------------------------------------------------------
    let disposed = false;
    let mediaRecorder = null;
    let mediaStream = null;
    let chunks = [];
    let statusResetTimer = 0;
    let progressClearTimer = 0;

    const state = {
        activeModelName: DEFAULT_MODEL,
        isRecording: false,
        isVoiceBusy: false,
        isAiBusy: false,
        modelReady: false,
        missingDependencies: [],
        activeAiOperationId: "",
        attachedImage: String(getWidgetValue(node, ATTACHED_IMAGE_WIDGET, "") || ""),
    };

    // Initial values from hidden widgets.
    textarea.value = String(getWidgetValue(node, TEXT_WIDGET, "") || "");
    if (toBoolean(getWidgetValue(node, HIGH_QUALITY_WIDGET, false))) {
        hqToggle.classList.add("is-on");
    }
    const initialPreset = String(getWidgetValue(node, SYSTEM_PRESET_WIDGET, "") || "");
    if (initialPreset && Array.from(presetSelect.options).some((o) => o.value === initialPreset)) {
        presetSelect.value = initialPreset;
    }

    // -----------------------------------------------------------------
    // Sync helpers
    // -----------------------------------------------------------------
    function syncTextFromUi() {
        setWidgetValue(node, TEXT_WIDGET, textarea.value);
    }
    function syncHighQualityFromUi() {
        setWidgetValue(node, HIGH_QUALITY_WIDGET, hqToggle.classList.contains("is-on"));
    }
    function syncPresetFromUi() {
        setWidgetValue(node, SYSTEM_PRESET_WIDGET, presetSelect.value);
    }

    function isHighQualityEnabled() {
        return hqToggle.classList.contains("is-on");
    }
    function getActiveVoiceModel() {
        return isHighQualityEnabled() ? HIGH_QUALITY_MODEL : DEFAULT_MODEL;
    }
    function syncActiveVoiceModel() {
        const modelName = getActiveVoiceModel();
        if (state.activeModelName !== modelName) {
            state.activeModelName = modelName;
            state.modelReady = false;
        }
        return modelName;
    }

    function setProgress({ percent, active, error, indeterminate }) {
        window.clearTimeout(progressClearTimer);
        progress.classList.toggle("is-indeterminate", Boolean(indeterminate));
        if (error) {
            progress.classList.remove("is-active");
            progressFill.style.width = "0%";
            return;
        }
        if (active) {
            progress.classList.add("is-active");
            if (Number.isFinite(percent)) {
                progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
            }
            return;
        }
        if (Number.isFinite(percent) && percent >= 100) {
            progress.classList.add("is-active");
            progressFill.style.width = "100%";
            progressClearTimer = window.setTimeout(() => {
                progress.classList.remove("is-active");
                progressFill.style.width = "0%";
            }, PROGRESS_CLEAR_DELAY_MS);
            return;
        }
        progress.classList.remove("is-active");
        progressFill.style.width = "0%";
    }

    function setStatus(text, kind = "info", resetMs = 0) {
        window.clearTimeout(statusResetTimer);
        statusText.textContent = String(text || "");
        statusRow.classList.toggle("is-error", kind === "error");
        if (resetMs > 0) {
            statusResetTimer = window.setTimeout(() => {
                if (disposed) return;
                statusText.textContent = "Ready";
                statusRow.classList.remove("is-error");
            }, resetMs);
        }
    }

    function refreshRecordButton() {
        syncActiveVoiceModel();
        if (state.isRecording) {
            recordBtn.classList.add("is-recording");
            recordBtn.innerHTML = SVG_ICON_STOP;
            recordBtn.disabled = false;
            recordBtn.title = "Остановить запись и распознать";
            return;
        }
        recordBtn.classList.remove("is-recording");
        recordBtn.innerHTML = SVG_ICON_MIC;
        if (state.isVoiceBusy) {
            recordBtn.disabled = true;
            recordBtn.title = "Работаю...";
            return;
        }
        if (state.missingDependencies.length > 0) {
            recordBtn.disabled = true;
            recordBtn.title = `Не хватает зависимости: ${state.missingDependencies[0]}`;
            return;
        }
        recordBtn.disabled = state.isAiBusy;
        recordBtn.title = state.modelReady
            ? "Запись с микрофона"
            : "Скачать модель распознавания";
    }
    function refreshAiButton() {
        if (state.isAiBusy) {
            aiBtn.disabled = true;
            aiBtn.title = "AI работает...";
            return;
        }
        aiBtn.disabled = state.isRecording || state.isVoiceBusy;
        aiBtn.title = "Улучшить промпт через AI";
    }

    function renderAttached() {
        const annotated = state.attachedImage;
        if (annotated) {
            attachBtn.classList.add("has-image");
            const url = resolveAnnotatedThumbUrl(annotated);
            attachThumb.style.backgroundImage = url ? `url("${url}")` : "";
            const match = annotated.match(/^(.+?)\s*\[/);
            const path = match ? match[1] : annotated;
            const segments = path.split("/");
            attachBtn.title = `Прикреплено: ${segments[segments.length - 1]} (нажмите ×, чтобы убрать)`;
        } else {
            attachBtn.classList.remove("has-image");
            attachThumb.style.backgroundImage = "";
            attachBtn.title = "Прикрепить изображение (drop / paste / click)";
        }
    }

    function setAttachedImage(annotated) {
        const value = String(annotated || "");
        state.attachedImage = value;
        setWidgetValue(node, ATTACHED_IMAGE_WIDGET, value);
        renderAttached();
        setDirty(node);
    }

    // -----------------------------------------------------------------
    // WebSocket events
    // -----------------------------------------------------------------
    function matchesActiveModel(detail) {
        return !detail?.model || !state.activeModelName || detail.model === state.activeModelName;
    }
    function matchesActiveAiOperation(detail) {
        return (
            !detail?.operation_id ||
            !state.activeAiOperationId ||
            detail.operation_id === state.activeAiOperationId
        );
    }

    function onVoiceProgress(event) {
        if (!matchesActiveModel(event.detail)) return;
        const percent = Number(event.detail?.percent || 0);
        setStatus(`Voice model ${Math.round(percent)}%`);
        setProgress({ percent, active: true });
    }
    function onVoiceStatus(event) {
        if (!matchesActiveModel(event.detail) || !state.isVoiceBusy) return;
        const text = String(event.detail?.text || "Working");
        const percent = clampPercent(event.detail?.percent);
        setStatus(text);
        setProgress({ percent, active: true, indeterminate: percent === null });
    }
    function onVoiceDone(event) {
        if (!matchesActiveModel(event.detail)) return;
        state.modelReady = true;
        state.isVoiceBusy = false;
        setStatus("Voice model ready", "info", STATUS_RESET_DELAY_MS);
        setProgress({ percent: 100, active: false });
        refreshRecordButton();
        refreshAiButton();
    }
    function onVoiceError(event) {
        if (!matchesActiveModel(event.detail)) return;
        state.isVoiceBusy = false;
        setStatus(`Voice error: ${event.detail?.text || "failed"}`, "error", STATUS_RESET_DELAY_MS);
        setProgress({ active: false, error: true });
        refreshRecordButton();
        refreshAiButton();
    }
    function onAiProgress(event) {
        if (!matchesActiveAiOperation(event.detail)) return;
        const text = String(event.detail?.text || "AI Prompt");
        const percent = clampPercent(event.detail?.percent);
        setStatus(text);
        setProgress({ percent, active: true, indeterminate: percent === null });
    }
    function onAiDone(event) {
        if (!matchesActiveAiOperation(event.detail)) return;
        setStatus(String(event.detail?.text || "AI prompt ready"), "info", STATUS_RESET_DELAY_MS);
        setProgress({ percent: 100, active: false });
    }
    function onAiError(event) {
        if (!matchesActiveAiOperation(event.detail)) return;
        setStatus(`AI error: ${event.detail?.text || "failed"}`, "error", STATUS_RESET_DELAY_MS);
        setProgress({ active: false, error: true });
    }

    api.addEventListener(`${VOICE_EVENT_PREFIX}.progress`, onVoiceProgress);
    api.addEventListener(`${VOICE_EVENT_PREFIX}.status`, onVoiceStatus);
    api.addEventListener(`${VOICE_EVENT_PREFIX}.done`, onVoiceDone);
    api.addEventListener(`${VOICE_EVENT_PREFIX}.error`, onVoiceError);
    api.addEventListener(`${AI_EVENT_PREFIX}.progress`, onAiProgress);
    api.addEventListener(`${AI_EVENT_PREFIX}.done`, onAiDone);
    api.addEventListener(`${AI_EVENT_PREFIX}.error`, onAiError);

    // -----------------------------------------------------------------
    // Text manipulation (cursor-preserving insert + full replace)
    // -----------------------------------------------------------------
    let savedSelection = null;
    function rememberCursor() {
        if (doc.activeElement === textarea) {
            savedSelection = {
                start: textarea.selectionStart ?? textarea.value.length,
                end: textarea.selectionEnd ?? textarea.value.length,
            };
        } else {
            savedSelection = null;
        }
    }
    function insertRecognizedText(newText) {
        const text = String(newText || "").trim();
        if (!text) return false;
        const currentValue = textarea.value;
        let combined;
        let cursorPosition = null;
        if (savedSelection) {
            const start = Math.max(0, savedSelection.start);
            const end = Math.max(start, savedSelection.end);
            const before = currentValue.slice(0, start);
            const after = currentValue.slice(end);
            const prefix = before.length > 0 && !/\s$/.test(before) ? " " : "";
            const suffix = after.length > 0 && !/^\s/.test(after) ? " " : "";
            const inserted = `${prefix}${text}${suffix}`;
            combined = `${before}${inserted}${after}`;
            cursorPosition = start + inserted.length;
        } else {
            const separator = currentValue.length > 0 && !/\s$/.test(currentValue) ? " " : "";
            combined = `${currentValue}${separator}${text}`;
        }
        textarea.value = combined;
        syncTextFromUi();
        if (cursorPosition !== null) {
            textarea.selectionStart = cursorPosition;
            textarea.selectionEnd = cursorPosition;
            textarea.focus();
        }
        savedSelection = null;
        setDirty(node);
        return true;
    }
    function replaceText(newText) {
        const text = String(newText || "").trim();
        if (!text) return false;
        textarea.value = text;
        syncTextFromUi();
        setDirty(node);
        return true;
    }

    // -----------------------------------------------------------------
    // Voice
    // -----------------------------------------------------------------
    async function refreshStatus() {
        const modelName = syncActiveVoiceModel();
        const params = new URLSearchParams({
            model: modelName,
            high_quality: isHighQualityEnabled() ? "true" : "false",
        });
        try {
            const data = await fetchJson(`${VOICE_ROUTE_BASE}/status?${params.toString()}`);
            const info = data[state.activeModelName] || {};
            state.modelReady = Boolean(info.downloaded);
            state.missingDependencies = Array.isArray(info.missing_dependencies)
                ? info.missing_dependencies
                : [];
            setStatus("Ready");
            setProgress({ active: false });
        } catch {
            state.modelReady = false;
            state.missingDependencies = [];
            setStatus("Voice unavailable", "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
        }
        refreshRecordButton();
        refreshAiButton();
    }

    async function downloadVoiceModel(force = false) {
        syncActiveVoiceModel();
        if (state.missingDependencies.length > 0) {
            setStatus(`Missing ${state.missingDependencies[0]}`, "error");
            refreshRecordButton();
            return;
        }
        state.isVoiceBusy = true;
        setStatus(`Downloading ${state.activeModelName}...`);
        setProgress({ active: true, indeterminate: true });
        refreshRecordButton();
        try {
            const data = await fetchJson(`${VOICE_ROUTE_BASE}/preload`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: state.activeModelName,
                    high_quality: isHighQualityEnabled(),
                    force,
                }),
            });
            if (!data.ok) throw new Error(data.error || "preload failed");
            state.modelReady = true;
            state.isVoiceBusy = false;
            setStatus("Voice model ready", "info", STATUS_RESET_DELAY_MS);
            setProgress({ percent: 100, active: false });
        } catch (error) {
            state.isVoiceBusy = false;
            setStatus(`Voice error: ${error.message}`, "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
        }
        refreshRecordButton();
        refreshAiButton();
    }

    async function sendAudioToServer(blob) {
        syncActiveVoiceModel();
        const form = new FormData();
        form.append("model", state.activeModelName);
        form.append("high_quality", isHighQualityEnabled() ? "true" : "false");
        form.append("audio", blob, "recording.webm");
        setStatus("Recognizing speech...");
        setProgress({ active: true, indeterminate: true });
        try {
            const data = await fetchJson(`${VOICE_ROUTE_BASE}/transcribe`, {
                method: "POST",
                body: form,
            });
            state.isVoiceBusy = false;
            if (!insertRecognizedText(data.text)) {
                setStatus("No speech detected", "info", STATUS_RESET_DELAY_MS);
                setProgress({ active: false });
            } else {
                setStatus("Speech inserted", "info", STATUS_RESET_DELAY_MS);
                setProgress({ percent: 100, active: false });
            }
        } catch (error) {
            state.isVoiceBusy = false;
            setStatus(`Voice error: ${error.message}`, "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
        }
        refreshRecordButton();
        refreshAiButton();
    }

    async function startRecording() {
        if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
            setStatus("Microphone unsupported", "error", STATUS_RESET_DELAY_MS);
            return;
        }
        rememberCursor();
        setStatus("Opening microphone...");
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
            });
            const mimeType = MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m)) || "";
            mediaRecorder = createAudioRecorder(mediaStream, mimeType);
            chunks = [];
            mediaRecorder.ondataavailable = (event) => {
                if (event.data?.size > 0) chunks.push(event.data);
            };
            mediaRecorder.onstop = async () => {
                if (mediaStream) {
                    mediaStream.getTracks().forEach((track) => track.stop());
                    mediaStream = null;
                }
                if (disposed) return;
                const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
                if (blob.size <= 0) {
                    state.isVoiceBusy = false;
                    setStatus("No audio captured", "info", STATUS_RESET_DELAY_MS);
                    setProgress({ active: false });
                    refreshRecordButton();
                    refreshAiButton();
                    return;
                }
                await sendAudioToServer(blob);
            };
            mediaRecorder.start();
            state.isRecording = true;
            setStatus("Recording...");
            setProgress({ active: true, indeterminate: true });
            refreshRecordButton();
            refreshAiButton();
        } catch (error) {
            state.isRecording = false;
            state.isVoiceBusy = false;
            setStatus(`Mic error: ${error.message}`, "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
            refreshRecordButton();
        }
    }

    function stopRecording() {
        if (!mediaRecorder || !state.isRecording) return;
        state.isRecording = false;
        state.isVoiceBusy = true;
        setStatus("Preparing audio...");
        setProgress({ active: true, indeterminate: true });
        refreshRecordButton();
        try {
            mediaRecorder.stop();
        } catch (error) {
            state.isVoiceBusy = false;
            setStatus(`Voice error: ${error.message}`, "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
            refreshRecordButton();
        }
    }

    function onRecordClick(event) {
        if (state.isAiBusy) return;
        syncActiveVoiceModel();
        if (state.isRecording) {
            stopRecording();
            return;
        }
        if (state.isVoiceBusy) return;
        if (!state.modelReady) {
            downloadVoiceModel(Boolean(event?.shiftKey));
            return;
        }
        startRecording();
    }

    // -----------------------------------------------------------------
    // AI enhance
    // -----------------------------------------------------------------
    function buildAiPayload() {
        return {
            text: String(textarea.value || ""),
            system_preset: String(presetSelect.value || "Prompts enhance"),
            attached_image: String(state.attachedImage || ""),
            operation_id: state.activeAiOperationId,
        };
    }

    async function enhancePrompt() {
        if (state.isVoiceBusy || state.isAiBusy || state.isRecording) return;
        syncTextFromUi();
        syncPresetFromUi();
        const payload = buildAiPayload();
        if (!payload.text.trim() && !payload.attached_image) {
            setStatus("No prompt text or image", "info", STATUS_RESET_DELAY_MS);
            return;
        }
        state.isAiBusy = true;
        state.activeAiOperationId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
        payload.operation_id = state.activeAiOperationId;
        setStatus("Starting AI prompt...");
        setProgress({ active: true, indeterminate: true });
        refreshAiButton();
        refreshRecordButton();
        try {
            const data = await fetchJson(`${AI_ROUTE_BASE}/enhance`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!data.ok) throw new Error(data.error || "enhance failed");
            if (replaceText(data.text)) {
                const ref = data.used_image ? " (image)" : "";
                setStatus(`AI prompt ready${ref}`, "info", STATUS_RESET_DELAY_MS);
            } else {
                setStatus("Empty AI result", "info", STATUS_RESET_DELAY_MS);
            }
            setProgress({ percent: 100, active: false });
        } catch (error) {
            setStatus(`AI error: ${error.message}`, "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
        } finally {
            state.isAiBusy = false;
            if (!disposed) {
                refreshAiButton();
                refreshRecordButton();
            }
        }
    }

    // -----------------------------------------------------------------
    // Attach (file picker / drag-drop / paste)
    // -----------------------------------------------------------------
    async function uploadImageFile(file) {
        if (!file) return "";
        if (!file.type.startsWith("image/") && !fileExtensionOk(file.name)) {
            setStatus("Not an image file", "error", STATUS_RESET_DELAY_MS);
            return "";
        }
        setStatus("Uploading image...");
        setProgress({ active: true, indeterminate: true });
        try {
            const form = new FormData();
            form.append("image", file, file.name);
            const response = await api.fetchApi(UPLOAD_ROUTE, { method: "POST", body: form });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload?.error || payload?.message || `HTTP ${response.status}`);
            }
            const annotated = buildAnnotatedPath(payload);
            if (!annotated) throw new Error("Upload returned no filename");
            setAttachedImage(annotated);
            setStatus("Image attached", "info", STATUS_RESET_DELAY_MS);
            setProgress({ percent: 100, active: false });
            return annotated;
        } catch (error) {
            setStatus(`Upload error: ${error.message}`, "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
            return "";
        }
    }

    function pointerOverContainer(event) {
        const path = typeof event.composedPath === "function" ? event.composedPath() : [];
        return path.includes(container);
    }

    fileInput.addEventListener("change", async () => {
        const file = fileInput.files?.[0];
        fileInput.value = "";
        if (file) await uploadImageFile(file);
    });
    attachBtn.addEventListener("click", (event) => {
        if (event.target === attachClear) return;
        fileInput.click();
    });
    attachClear.addEventListener("click", (event) => {
        event.stopPropagation();
        setAttachedImage("");
        setStatus("Image removed", "info", STATUS_RESET_DELAY_MS);
    });

    function dragHasImage(event) {
        const dt = event.dataTransfer;
        if (!dt) return false;
        return Array.from(dt.types || []).includes("Files");
    }
    function onDragEnter(event) {
        if (!dragHasImage(event)) return;
        event.preventDefault();
        event.stopPropagation();
        container.classList.add("is-drag-over");
    }
    function onDragOver(event) {
        if (!dragHasImage(event)) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
        container.classList.add("is-drag-over");
    }
    function onDragLeave() {
        container.classList.remove("is-drag-over");
    }
    async function onDrop(event) {
        if (!dragHasImage(event)) return;
        event.preventDefault();
        event.stopPropagation();
        container.classList.remove("is-drag-over");
        const file = event.dataTransfer?.files?.[0];
        if (file) await uploadImageFile(file);
    }
    container.addEventListener("dragenter", onDragEnter);
    container.addEventListener("dragover", onDragOver);
    container.addEventListener("dragleave", onDragLeave);
    container.addEventListener("drop", onDrop);

    async function onDocumentPaste(event) {
        if (disposed) return;
        if (!pointerOverContainer(event)) return;
        const items = event.clipboardData?.items || [];
        for (const item of items) {
            if (item.kind === "file" && item.type.startsWith("image/")) {
                const file = item.getAsFile();
                if (file) {
                    event.preventDefault();
                    await uploadImageFile(file);
                    return;
                }
            }
        }
    }
    doc.addEventListener("paste", onDocumentPaste);

    // -----------------------------------------------------------------
    // Toolbar event wiring
    // -----------------------------------------------------------------
    textarea.addEventListener("input", () => syncTextFromUi());
    textarea.addEventListener("blur", () => syncTextFromUi());
    hqToggle.addEventListener("click", () => {
        hqToggle.classList.toggle("is-on");
        syncHighQualityFromUi();
        syncActiveVoiceModel();
        refreshStatus();
    });
    presetSelect.addEventListener("change", () => syncPresetFromUi());
    recordBtn.addEventListener("click", onRecordClick);
    aiBtn.addEventListener("click", () => enhancePrompt());

    // -----------------------------------------------------------------
    // Mount DOM widget
    // -----------------------------------------------------------------
    const domWidgetOptions = {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 110,
        getMaxHeight: () => 8192,
    };
    const domWidget = node.addDOMWidget(DOM_WIDGET_NAME, "div", container, domWidgetOptions);
    domWidget.__tsSuperPromptUi = true;

    // Make the node a touch taller out-of-the-box so the textarea has room.
    try {
        const minHeight = 170;
        if (Array.isArray(node.size) && node.size[1] < minHeight) {
            node.size[1] = minHeight;
        }
    } catch {
        // Some Litegraph forks store size differently — non-fatal.
    }

    // -----------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------
    function cleanup() {
        if (disposed) return;
        disposed = true;
        window.clearTimeout(statusResetTimer);
        window.clearTimeout(progressClearTimer);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.progress`, onVoiceProgress);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.status`, onVoiceStatus);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.done`, onVoiceDone);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.error`, onVoiceError);
        api.removeEventListener(`${AI_EVENT_PREFIX}.progress`, onAiProgress);
        api.removeEventListener(`${AI_EVENT_PREFIX}.done`, onAiDone);
        api.removeEventListener(`${AI_EVENT_PREFIX}.error`, onAiError);
        doc.removeEventListener("paste", onDocumentPaste);
        if (mediaRecorder && state.isRecording) {
            try {
                mediaRecorder.stop();
            } catch {
                // recorder may already be stopped — safe to ignore.
            }
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => track.stop());
            mediaStream = null;
        }
        if (Array.isArray(node.widgets)) {
            const idx = node.widgets.indexOf(domWidget);
            if (idx >= 0) node.widgets.splice(idx, 1);
        }
        container.remove();
    }

    node._tsSuperPromptCleanup = cleanup;
    if (!node._tsSuperPromptOriginalOnRemoved) {
        node._tsSuperPromptOriginalOnRemoved = node.onRemoved;
    }
    node.onRemoved = function onRemovedWrapper() {
        cleanup();
        return node._tsSuperPromptOriginalOnRemoved?.apply(this, arguments);
    };

    // Initial paint.
    renderAttached();
    refreshRecordButton();
    refreshAiButton();
    refreshStatus();
}

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedWrapper() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (!getWidget(this, DOM_WIDGET_NAME)) {
                setupSuperPrompt(this);
            }
            return result;
        };
    },
    loadedGraphNode(node) {
        if (![node?.type, node?.comfyClass].includes(NODE_NAME)) return;
        if (!getWidget(node, DOM_WIDGET_NAME)) {
            setupSuperPrompt(node);
        }
    },
});
