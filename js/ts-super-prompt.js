import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_ID = "ts.superPrompt";
const NODE_NAME = "TS_SuperPrompt";
const VOICE_ROUTE_BASE = "/ts_voice_recognition";
const AI_ROUTE_BASE = "/ts_super_prompt";
const VOICE_EVENT_PREFIX = "ts_voice_recognition";
const AI_EVENT_PREFIX = "ts_super_prompt";
const TEXT_WIDGET = "text";
const HIGH_QUALITY_WIDGET = "high_quality";
const SYSTEM_PRESET_WIDGET = "system_preset";
const WIDGET_TOOLTIPS = {
    [TEXT_WIDGET]: "Поле промпта: сюда попадает распознанная речь, а кнопка Ai prompt заменяет текст улучшенным промптом.",
    [HIGH_QUALITY_WIDGET]: "Включите, чтобы распознавать речь моделью Whisper turbo (large-v3 turbo). Выключено: используется быстрая base.",
    [SYSTEM_PRESET_WIDGET]: "Выберите системный пресет из qwen_3_vl_presets.json для улучшения промпта.",
};
const VOICE_BUTTON_TOOLTIP = "Запускает запись с микрофона. Во время записи нажмите еще раз, чтобы остановить и распознать аудио.";
const AI_BUTTON_TOOLTIP = "Улучшает текст через Huihui-Qwen3.5-2B-abliterated: при необходимости переводит на английский и делает качественный промпт для генерации.";
const DEFAULT_MODEL = "base";
const HIGH_QUALITY_MODEL = "turbo";
const LEGACY_DOM_WIDGET_NAME = "ts_super_prompt_progress";
const CURSOR_PATCH_FLAG = "__tsSuperPromptCursorPatch";
const PROGRESS_CLEAR_DELAY_MS = 900;
const AUDIO_BITS_PER_SECOND = 128000;

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function getWidgetValue(node, name, fallback = null) {
    const widget = getWidget(node, name);
    return widget?.value ?? fallback;
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

function setWidgetTooltip(widget, tooltip) {
    if (!widget || !tooltip) return;
    widget.tooltip = tooltip;
    widget.options ||= {};
    widget.options.tooltip = tooltip;
}

function applyWidgetTooltips(node) {
    for (const [name, tooltip] of Object.entries(WIDGET_TOOLTIPS)) {
        setWidgetTooltip(getWidget(node, name), tooltip);
    }
}

function removeControlWidgets(node) {
    if (!Array.isArray(node?.widgets)) return;
    for (let index = node.widgets.length - 1; index >= 0; index -= 1) {
        const widget = node.widgets[index];
        if (!widget?.__tsSuperPromptButton && widget?.name !== LEGACY_DOM_WIDGET_NAME) continue;
        const element = widget.element || widget.el || widget.container;
        element?.remove?.();
        node.widgets.splice(index, 1);
    }
}

function removeStaleConfigWidgets(node) {
    if (!Array.isArray(node?.widgets)) return;
    const allowed = new Set([TEXT_WIDGET, HIGH_QUALITY_WIDGET, SYSTEM_PRESET_WIDGET]);
    for (let index = node.widgets.length - 1; index >= 0; index -= 1) {
        const widget = node.widgets[index];
        if (!widget?.name || allowed.has(widget.name) || widget.__tsSuperPromptButton) continue;
        node.widgets.splice(index, 1);
    }
}

function getTextInput(widget) {
    return widget?.inputEl || widget?.element?.querySelector?.("textarea,input") || null;
}

function setTextValue(node, widget, value) {
    widget.value = value;
    const input = getTextInput(widget);
    if (input) input.value = value;
    if (typeof widget.callback === "function") widget.callback(value);
    node.properties ||= {};
    node.properties[TEXT_WIDGET] = value;
    setDirty(node);
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
    } catch (error) {
        return mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    }
}

function createProgressWidget(node) {
    let clearTimer = 0;

    const clearProgressLater = () => {
        window.clearTimeout(clearTimer);
        clearTimer = window.setTimeout(() => {
            if (node) {
                node.progress = undefined;
                setDirty(node);
            }
        }, PROGRESS_CLEAR_DELAY_MS);
    };

    return {
        set(update) {
            if (!node) return;
            window.clearTimeout(clearTimer);
            const rawPercent = update && typeof update.percent === "number" ? update.percent : undefined;
            const percent = rawPercent === undefined ? undefined : Math.max(0, Math.min(100, rawPercent));

            if (update?.error) {
                node.progress = undefined;
            } else if (update?.active) {
                node.progress = percent === undefined ? Math.max(node.progress || 0.02, 0.02) : percent / 100;
            } else if (percent !== undefined && percent > 0) {
                node.progress = percent / 100;
                if (percent >= 100) clearProgressLater();
            } else {
                node.progress = undefined;
            }
            setDirty(node);
        },
        dispose() {
            window.clearTimeout(clearTimer);
            if (node) {
                node.progress = undefined;
                setDirty(node);
            }
        },
    };
}

function isOverSuperPromptButton(node, graphPos) {
    if (!Array.isArray(node?.widgets) || !Array.isArray(graphPos)) return false;
    if (![node?.type, node?.comfyClass].includes(NODE_NAME)) return false;

    const x = graphPos[0] - node.pos[0];
    const y = graphPos[1] - node.pos[1];
    const width = node.size?.[0] || 0;

    for (const widget of node.widgets) {
        if (!widget?.__tsSuperPromptButton || widget.disabled || widget.last_y === undefined) continue;
        const height = widget.computeSize ? widget.computeSize(width)[1] : 20;
        const widgetWidth = widget.width || width;
        if (x >= 6 && x <= widgetWidth - 12 && y >= widget.last_y && y <= widget.last_y + height) {
            return true;
        }
    }
    return false;
}

function installCursorPatch() {
    const canvas = app?.canvas;
    const canvasEl = canvas?.canvas;
    if (!canvas || !canvasEl || canvasEl[CURSOR_PATCH_FLAG]) return;
    canvasEl[CURSOR_PATCH_FLAG] = true;

    const updateCursor = () => {
        if (isOverSuperPromptButton(canvas.node_over, canvas.graph_mouse)) {
            canvasEl.style.cursor = "pointer";
        }
    };

    canvasEl.addEventListener("pointermove", updateCursor, { passive: true });
    canvasEl.addEventListener("mousemove", updateCursor, { passive: true });
}

function setupSuperPrompt(node) {
    if (!node) return;
    if (typeof node._tsSuperPromptCleanup === "function") {
        node._tsSuperPromptCleanup();
    }
    removeControlWidgets(node);
    removeStaleConfigWidgets(node);
    applyWidgetTooltips(node);
    installCursorPatch();

    let disposed = false;
    let mediaRecorder = null;
    let mediaStream = null;
    let chunks = [];
    let voiceButton = null;
    let aiButton = null;
    let progressWidget = null;
    let statusTimer = 0;
    let savedCursor = null;

    const state = {
        activeModelName: DEFAULT_MODEL,
        isRecording: false,
        isVoiceBusy: false,
        isAiBusy: false,
        modelReady: false,
        missingDependencies: [],
        activeAiOperationId: "",
    };
    progressWidget = createProgressWidget(node);

    const isHighQualityEnabled = () => toBoolean(getWidgetValue(node, HIGH_QUALITY_WIDGET, false));

    const getActiveVoiceModel = () => {
        return isHighQualityEnabled() ? HIGH_QUALITY_MODEL : DEFAULT_MODEL;
    };

    const syncActiveVoiceModel = () => {
        const modelName = getActiveVoiceModel();
        if (state.activeModelName !== modelName) {
            state.activeModelName = modelName;
            state.modelReady = false;
        }
        return modelName;
    };

    const setVoiceLabel = (label) => {
        if (!voiceButton) return;
        voiceButton.name = label;
        setDirty(node);
    };

    const setAiLabel = (label) => {
        if (!aiButton) return;
        aiButton.name = label;
        setDirty(node);
    };

    const compactStatusLabel = (fallback, detail) => {
        const text = String(detail?.text || fallback);
        const rawPercent = detail?.percent;
        const percent = Number.isFinite(Number(rawPercent)) ? Math.round(Number(rawPercent)) : null;
        const shortText = text.length > 24 ? `${text.slice(0, 21)}...` : text;
        return percent === null ? shortText : `${shortText} ${percent}%`;
    };

    const scheduleRefresh = (delayMs = 3000) => {
        window.clearTimeout(statusTimer);
        statusTimer = window.setTimeout(() => {
            if (!disposed) refreshVoiceButton();
        }, delayMs);
    };

    const refreshVoiceButton = () => {
        syncActiveVoiceModel();
        if (state.isRecording) {
            setVoiceLabel("Stop Recording");
            return;
        }
        if (state.isVoiceBusy) return;
        if (state.missingDependencies.length > 0) {
            setVoiceLabel(`Missing ${state.missingDependencies[0]}`);
            return;
        }
        setVoiceLabel(state.modelReady ? "Start Recording" : "Download Voice Model");
    };

    const refreshAiButton = () => {
        if (!state.isAiBusy) setAiLabel("Ai prompt");
    };

    const matchesActiveModel = (detail) => {
        return !detail?.model || !state.activeModelName || detail.model === state.activeModelName;
    };

    const matchesActiveAiOperation = (detail) => {
        return !detail?.operation_id || !state.activeAiOperationId || detail.operation_id === state.activeAiOperationId;
    };

    const onProgress = (event) => {
        if (!matchesActiveModel(event.detail)) return;
        const percent = Number(event.detail?.percent || 0);
        setVoiceLabel(`Downloading ${percent}%`);
        progressWidget?.set({
            text: `Downloading voice model ${percent}%`,
            percent,
            active: true,
            error: false,
        });
    };

    const onStatus = (event) => {
        if (!matchesActiveModel(event.detail) || !state.isVoiceBusy) return;
        setVoiceLabel(String(event.detail?.text || "Working"));
        progressWidget?.set({
            text: String(event.detail?.text || "Working"),
            active: true,
            error: false,
        });
    };

    const onDone = (event) => {
        if (!matchesActiveModel(event.detail)) return;
        state.modelReady = true;
        state.isVoiceBusy = false;
        progressWidget?.set({ text: "Voice model ready", percent: 100, active: false, error: false });
        refreshVoiceButton();
    };

    const onError = (event) => {
        if (!matchesActiveModel(event.detail)) return;
        state.isVoiceBusy = false;
        setVoiceLabel(`Error: ${event.detail?.text || "failed"}`);
        progressWidget?.set({ text: `Voice error: ${event.detail?.text || "failed"}`, active: false, error: true });
        scheduleRefresh(4000);
    };

    const onAiProgress = (event) => {
        if (!matchesActiveAiOperation(event.detail)) return;
        const rawPercent = event.detail?.percent;
        const percent = Number.isFinite(Number(rawPercent)) ? Number(rawPercent) : undefined;
        setAiLabel(compactStatusLabel("AI Working", event.detail));
        progressWidget?.set({
            text: String(event.detail?.text || "AI Prompt"),
            percent,
            active: true,
            error: false,
        });
    };

    const onAiDone = (event) => {
        if (!matchesActiveAiOperation(event.detail)) return;
        progressWidget?.set({
            text: String(event.detail?.text || "AI prompt ready"),
            percent: 100,
            active: false,
            error: false,
        });
    };

    const onAiError = (event) => {
        if (!matchesActiveAiOperation(event.detail)) return;
        progressWidget?.set({
            text: `AI error: ${event.detail?.text || "failed"}`,
            active: false,
            error: true,
        });
    };

    api.addEventListener(`${VOICE_EVENT_PREFIX}.progress`, onProgress);
    api.addEventListener(`${VOICE_EVENT_PREFIX}.status`, onStatus);
    api.addEventListener(`${VOICE_EVENT_PREFIX}.done`, onDone);
    api.addEventListener(`${VOICE_EVENT_PREFIX}.error`, onError);
    api.addEventListener(`${AI_EVENT_PREFIX}.progress`, onAiProgress);
    api.addEventListener(`${AI_EVENT_PREFIX}.done`, onAiDone);
    api.addEventListener(`${AI_EVENT_PREFIX}.error`, onAiError);

    const cleanup = () => {
        disposed = true;
        window.clearTimeout(statusTimer);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.progress`, onProgress);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.status`, onStatus);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.done`, onDone);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.error`, onError);
        api.removeEventListener(`${AI_EVENT_PREFIX}.progress`, onAiProgress);
        api.removeEventListener(`${AI_EVENT_PREFIX}.done`, onAiDone);
        api.removeEventListener(`${AI_EVENT_PREFIX}.error`, onAiError);
        if (mediaRecorder && state.isRecording) {
            try {
                mediaRecorder.stop();
            } catch {}
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => track.stop());
            mediaStream = null;
        }
        progressWidget?.dispose?.();
        removeControlWidgets(node);
    };

    node._tsSuperPromptCleanup = cleanup;
    if (!node._tsSuperPromptOriginalOnRemoved) {
        node._tsSuperPromptOriginalOnRemoved = node.onRemoved;
    }
    node.onRemoved = function onRemovedWrapper() {
        cleanup();
        return node._tsSuperPromptOriginalOnRemoved?.apply(this, arguments);
    };

    const rememberCursor = () => {
        const textWidget = getWidget(node, TEXT_WIDGET);
        const input = getTextInput(textWidget);
        if (input && document.activeElement === input) {
            savedCursor = {
                input,
                start: input.selectionStart ?? input.value.length,
                end: input.selectionEnd ?? input.value.length,
            };
        } else {
            savedCursor = null;
        }
    };

    const insertText = (newText) => {
        const text = String(newText || "").trim();
        if (!text) return false;

        const textWidget = getWidget(node, TEXT_WIDGET);
        if (!textWidget) return false;

        const input = getTextInput(textWidget);
        const currentValue = String(textWidget.value || input?.value || "");
        let combined = "";
        let cursorPosition = null;

        if (savedCursor && savedCursor.input === input) {
            const start = Math.max(0, savedCursor.start);
            const end = Math.max(start, savedCursor.end);
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

        setTextValue(node, textWidget, combined);
        if (input && cursorPosition !== null) {
            input.selectionStart = cursorPosition;
            input.selectionEnd = cursorPosition;
            input.focus();
        }
        savedCursor = null;
        return true;
    };

    const replaceText = (newText) => {
        const text = String(newText || "").trim();
        if (!text) return false;
        const textWidget = getWidget(node, TEXT_WIDGET);
        if (!textWidget) return false;
        setTextValue(node, textWidget, text);
        return true;
    };

    const refreshStatus = async () => {
        const modelName = syncActiveVoiceModel();
        const params = new URLSearchParams({
            model: modelName,
            high_quality: isHighQualityEnabled() ? "true" : "false",
        });
        try {
            const data = await fetchJson(`${VOICE_ROUTE_BASE}/status?${params.toString()}`);
            const info = data[state.activeModelName] || {};
            state.modelReady = Boolean(info.downloaded);
            state.missingDependencies = Array.isArray(info.missing_dependencies) ? info.missing_dependencies : [];
        } catch (error) {
            state.modelReady = false;
            state.missingDependencies = [];
            setVoiceLabel("Voice Unavailable");
            progressWidget?.set({ text: "Voice unavailable", active: false, error: true });
            scheduleRefresh(5000);
            return;
        }
        progressWidget?.set({ text: "Ready", percent: 0, active: false, error: false });
        refreshVoiceButton();
    };

    const downloadVoiceModel = async (force = false) => {
        syncActiveVoiceModel();
        if (state.missingDependencies.length > 0) {
            progressWidget?.set({
                text: `Missing ${state.missingDependencies[0]}`,
                active: false,
                error: true,
            });
            refreshVoiceButton();
            return;
        }
        state.isVoiceBusy = true;
        setVoiceLabel(`Downloading ${state.activeModelName}`);
        progressWidget?.set({ text: `Downloading ${state.activeModelName}`, active: true, error: false });
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
            progressWidget?.set({ text: "Voice model ready", percent: 100, active: false, error: false });
            refreshVoiceButton();
        } catch (error) {
            state.isVoiceBusy = false;
            setVoiceLabel(`Error: ${error.message}`);
            progressWidget?.set({ text: `Voice error: ${error.message}`, active: false, error: true });
            scheduleRefresh(4000);
        }
    };

    const sendAudioToServer = async (blob) => {
        syncActiveVoiceModel();
        const form = new FormData();
        form.append("model", state.activeModelName);
        form.append("high_quality", isHighQualityEnabled() ? "true" : "false");
        form.append("audio", blob, "recording.webm");

        try {
            progressWidget?.set({ text: "Recognizing speech", active: true, error: false });
            const data = await fetchJson(`${VOICE_ROUTE_BASE}/transcribe`, {
                method: "POST",
                body: form,
            });
            state.isVoiceBusy = false;
            if (!insertText(data.text)) {
                setVoiceLabel("No Speech Detected");
                progressWidget?.set({ text: "No speech detected", active: false, error: false });
                scheduleRefresh(2500);
                return;
            }
            progressWidget?.set({ text: "Speech inserted", percent: 100, active: false, error: false });
            refreshVoiceButton();
        } catch (error) {
            state.isVoiceBusy = false;
            setVoiceLabel(`Error: ${error.message}`);
            progressWidget?.set({ text: `Voice error: ${error.message}`, active: false, error: true });
            scheduleRefresh(4000);
        }
    };

    const startRecording = async () => {
        if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
            setVoiceLabel("Mic Unsupported");
            progressWidget?.set({ text: "Microphone unsupported", active: false, error: true });
            scheduleRefresh(4000);
            return;
        }

        rememberCursor();
        try {
            progressWidget?.set({ text: "Opening microphone", active: true, error: false });
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });

            const mimeCandidates = [
                "audio/webm;codecs=opus",
                "audio/webm",
                "audio/ogg;codecs=opus",
                "audio/mp4",
            ];
            const mimeType = mimeCandidates.find((mime) => MediaRecorder.isTypeSupported(mime)) || "";
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
                    setVoiceLabel("No Audio Captured");
                    progressWidget?.set({ text: "No audio captured", active: false, error: false });
                    scheduleRefresh(2500);
                    return;
                }
                await sendAudioToServer(blob);
            };

            mediaRecorder.start();
            state.isRecording = true;
            progressWidget?.set({ text: "Recording voice", active: true, error: false });
            refreshVoiceButton();
        } catch (error) {
            state.isRecording = false;
            state.isVoiceBusy = false;
            setVoiceLabel(`Mic Error: ${error.message}`);
            progressWidget?.set({ text: `Mic error: ${error.message}`, active: false, error: true });
            scheduleRefresh(4000);
        }
    };

    const stopRecording = () => {
        if (!mediaRecorder || !state.isRecording) return;
        state.isRecording = false;
        state.isVoiceBusy = true;
        setVoiceLabel("Recognizing");
        progressWidget?.set({ text: "Preparing audio", active: true, error: false });
        try {
            mediaRecorder.stop();
        } catch (error) {
            state.isVoiceBusy = false;
            setVoiceLabel(`Error: ${error.message}`);
            progressWidget?.set({ text: `Voice error: ${error.message}`, active: false, error: true });
            scheduleRefresh(4000);
        }
    };

    const handleVoiceClick = (event) => {
        if (state.isVoiceBusy || state.isAiBusy) return;
        syncActiveVoiceModel();
        if (state.isRecording) {
            stopRecording();
            return;
        }
        if (!state.modelReady) {
            downloadVoiceModel(Boolean(event?.shiftKey));
            return;
        }
        startRecording();
    };

    const installRecordButtonStyle = (widget) => {
        widget.type = "ts_super_prompt_record_button";
        widget.computeSize ||= (width) => [width, globalThis.LiteGraph?.NODE_WIDGET_HEIGHT || 20];
        widget.mouse = (event) => {
            if (!String(event?.type || "").endsWith("down")) return false;
            widget.clicked = true;
            handleVoiceClick(event);
            return true;
        };
        widget.draw = (ctx, _node, widgetWidth, y, height) => {
            const margin = 15;
            const width = widgetWidth - margin * 2;
            const isRecording = state.isRecording;
            const background = isRecording
                ? "#b3262e"
                : widget.clicked
                    ? "#aaa"
                    : (globalThis.LiteGraph?.WIDGET_BGCOLOR || "#333");
            const outline = isRecording ? "#ff5a64" : (globalThis.LiteGraph?.WIDGET_OUTLINE_COLOR || "#666");
            const text = isRecording ? "#fff" : (globalThis.LiteGraph?.WIDGET_TEXT_COLOR || "#ddd");

            widget.clicked = false;
            ctx.save();
            ctx.fillStyle = background;
            ctx.strokeStyle = outline;
            ctx.fillRect(margin, y, width, height);
            ctx.strokeRect(margin, y, width, height);
            ctx.textAlign = "center";
            ctx.fillStyle = text;
            ctx.fillText(widget.label || widget.name, widgetWidth * 0.5, y + height * 0.7);
            ctx.restore();
        };
    };

    const buildAiPayload = () => {
        const textWidget = getWidget(node, TEXT_WIDGET);
        return {
            text: String(textWidget?.value || ""),
            system_preset: String(getWidgetValue(node, SYSTEM_PRESET_WIDGET, "Prompts enhance") || "Prompts enhance"),
            operation_id: state.activeAiOperationId,
        };
    };

    const highQualityWidget = getWidget(node, HIGH_QUALITY_WIDGET);
    if (highQualityWidget && !highQualityWidget.__tsSuperPromptPatched) {
        highQualityWidget.label = "Hight Quality";
        const originalCallback = highQualityWidget.callback;
        highQualityWidget.callback = function highQualityCallback(value) {
            const result = originalCallback?.apply(this, arguments);
            if (!disposed && !state.isRecording && !state.isVoiceBusy) {
                syncActiveVoiceModel();
                refreshStatus();
            }
            return result;
        };
        highQualityWidget.__tsSuperPromptPatched = true;
    }

    const enhancePrompt = async () => {
        if (state.isVoiceBusy || state.isAiBusy || state.isRecording) return;
        const payload = buildAiPayload();
        if (!payload.text.trim()) {
            setAiLabel("No Prompt");
            progressWidget?.set({ text: "No prompt text", active: false, error: false });
            window.setTimeout(refreshAiButton, 2200);
            return;
        }

        state.isAiBusy = true;
        state.activeAiOperationId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
        payload.operation_id = state.activeAiOperationId;
        setAiLabel("AI Working");
        progressWidget?.set({ text: "Starting AI Prompt", active: true, error: false });
        let delayedRefresh = false;
        try {
            const data = await fetchJson(`${AI_ROUTE_BASE}/enhance`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!data.ok) throw new Error(data.error || "enhance failed");
            if (!replaceText(data.text)) {
                setAiLabel("Empty AI Result");
                progressWidget?.set({ text: "Empty AI result", active: false, error: false });
                window.setTimeout(refreshAiButton, 2500);
                delayedRefresh = true;
            }
        } catch (error) {
            setAiLabel(`AI Error: ${error.message}`);
            progressWidget?.set({ text: `AI error: ${error.message}`, active: false, error: true });
            window.setTimeout(refreshAiButton, 5000);
            delayedRefresh = true;
        } finally {
            state.isAiBusy = false;
            if (!disposed && !delayedRefresh) refreshAiButton();
        }
    };

    voiceButton = node.addWidget("button", "Loading Voice Model", null, function superPromptVoiceButton() {
        const event = Array.from(arguments).find((arg) => arg instanceof Event);
        handleVoiceClick(event);
    });
    voiceButton.__tsSuperPromptButton = true;
    setWidgetTooltip(voiceButton, VOICE_BUTTON_TOOLTIP);
    installRecordButtonStyle(voiceButton);

    aiButton = node.addWidget("button", "Ai prompt", null, function superPromptAiButton() {
        enhancePrompt();
    });
    aiButton.__tsSuperPromptButton = true;
    setWidgetTooltip(aiButton, AI_BUTTON_TOOLTIP);

    refreshStatus();
    refreshAiButton();
}

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedWrapper() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setupSuperPrompt(this);
            return result;
        };
    },
    loadedGraphNode(node) {
        if (![node?.type, node?.comfyClass].includes(NODE_NAME)) return;
        setupSuperPrompt(node);
    },
});
