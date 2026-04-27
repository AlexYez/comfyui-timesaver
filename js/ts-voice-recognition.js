import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_ID = "ts.voiceRecognition";
const NODE_NAME = "TS_VoiceRecognition";
const ROUTE_BASE = "/ts_voice_recognition";
const EVENT_PREFIX = "ts_voice_recognition";
const TEXT_WIDGET = "text";
const TRANSLATE_WIDGET = "translate_to_english";
const DEFAULT_MODEL = "base";

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function setDirty(node) {
    node?.setDirtyCanvas?.(true, true);
    app?.graph?.setDirtyCanvas?.(true, true);
}

function removeControlWidgets(node) {
    if (!Array.isArray(node?.widgets)) return;
    for (let index = node.widgets.length - 1; index >= 0; index -= 1) {
        const widget = node.widgets[index];
        if (!widget?.__tsVoiceRecognitionButton) continue;
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

function setupVoiceRecognition(node) {
    if (!node) return;
    if (typeof node._tsVoiceRecognitionCleanup === "function") {
        node._tsVoiceRecognitionCleanup();
    }
    removeControlWidgets(node);

    let disposed = false;
    let mediaRecorder = null;
    let mediaStream = null;
    let chunks = [];
    let button = null;
    let statusTimer = 0;
    let savedCursor = null;

    const state = {
        activeModelName: DEFAULT_MODEL,
        isRecording: false,
        isBusy: false,
        modelReady: false,
        missingDependencies: [],
    };

    const setButtonLabel = (label) => {
        if (!button) return;
        button.name = label;
        setDirty(node);
    };

    const scheduleRefresh = (delayMs = 3000) => {
        window.clearTimeout(statusTimer);
        statusTimer = window.setTimeout(() => {
            if (!disposed) refreshButton();
        }, delayMs);
    };

    const refreshButton = () => {
        if (state.isRecording) {
            setButtonLabel("Stop Recording");
            return;
        }
        if (state.isBusy) return;
        if (state.missingDependencies.length > 0) {
            setButtonLabel(`Missing ${state.missingDependencies[0]}`);
            return;
        }
        setButtonLabel(state.modelReady ? "Record Voice" : "Download Model");
    };

    const matchesActiveModel = (detail) => {
        return !detail?.model || !state.activeModelName || detail.model === state.activeModelName;
    };

    const onProgress = (event) => {
        if (!matchesActiveModel(event.detail)) return;
        const percent = Number(event.detail?.percent || 0);
        setButtonLabel(`Downloading ${percent}%`);
    };

    const onStatus = (event) => {
        if (!matchesActiveModel(event.detail) || !state.isBusy) return;
        setButtonLabel(String(event.detail?.text || "Working"));
    };

    const onDone = (event) => {
        if (!matchesActiveModel(event.detail)) return;
        state.modelReady = true;
        state.isBusy = false;
        refreshButton();
    };

    const onError = (event) => {
        if (!matchesActiveModel(event.detail)) return;
        state.isBusy = false;
        setButtonLabel(`Error: ${event.detail?.text || "failed"}`);
        scheduleRefresh(4000);
    };

    api.addEventListener(`${EVENT_PREFIX}.progress`, onProgress);
    api.addEventListener(`${EVENT_PREFIX}.status`, onStatus);
    api.addEventListener(`${EVENT_PREFIX}.done`, onDone);
    api.addEventListener(`${EVENT_PREFIX}.error`, onError);

    const cleanup = () => {
        disposed = true;
        window.clearTimeout(statusTimer);
        api.removeEventListener(`${EVENT_PREFIX}.progress`, onProgress);
        api.removeEventListener(`${EVENT_PREFIX}.status`, onStatus);
        api.removeEventListener(`${EVENT_PREFIX}.done`, onDone);
        api.removeEventListener(`${EVENT_PREFIX}.error`, onError);
        if (mediaRecorder && state.isRecording) {
            try {
                mediaRecorder.stop();
            } catch {}
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => track.stop());
            mediaStream = null;
        }
        removeControlWidgets(node);
    };

    node._tsVoiceRecognitionCleanup = cleanup;
    if (!node._tsVoiceRecognitionOriginalOnRemoved) {
        node._tsVoiceRecognitionOriginalOnRemoved = node.onRemoved;
    }
    node.onRemoved = function onRemovedWrapper() {
        cleanup();
        return node._tsVoiceRecognitionOriginalOnRemoved?.apply(this, arguments);
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

    const refreshStatus = async () => {
        try {
            const data = await fetchJson(`${ROUTE_BASE}/status`);
            const names = Object.keys(data || {});
            state.activeModelName = names[0] || DEFAULT_MODEL;
            const info = data[state.activeModelName] || {};
            state.modelReady = Boolean(info.downloaded);
            state.missingDependencies = Array.isArray(info.missing_dependencies) ? info.missing_dependencies : [];
        } catch (error) {
            state.modelReady = false;
            state.missingDependencies = [];
            setButtonLabel("Voice Unavailable");
            scheduleRefresh(5000);
            return;
        }
        refreshButton();
    };

    const downloadModel = async (force = false) => {
        if (state.missingDependencies.length > 0) {
            refreshButton();
            return;
        }
        state.isBusy = true;
        setButtonLabel(`Downloading ${state.activeModelName}`);
        try {
            const data = await fetchJson(`${ROUTE_BASE}/preload`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: state.activeModelName, force }),
            });
            if (!data.ok) throw new Error(data.error || "preload failed");
            state.modelReady = true;
            state.isBusy = false;
            refreshButton();
        } catch (error) {
            state.isBusy = false;
            setButtonLabel(`Error: ${error.message}`);
            scheduleRefresh(4000);
        }
    };

    const sendToServer = async (blob) => {
        const translate = getWidget(node, TRANSLATE_WIDGET)?.value === true;
        const form = new FormData();
        form.append("audio", blob, "recording.webm");
        const params = new URLSearchParams({ translate: translate ? "true" : "false" });

        try {
            const data = await fetchJson(`${ROUTE_BASE}/transcribe?${params}`, {
                method: "POST",
                body: form,
            });
            state.isBusy = false;
            if (!insertText(data.text)) {
                setButtonLabel("No Speech Detected");
                scheduleRefresh(2500);
                return;
            }
            refreshButton();
        } catch (error) {
            state.isBusy = false;
            setButtonLabel(`Error: ${error.message}`);
            scheduleRefresh(4000);
        }
    };

    const startRecording = async () => {
        if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
            setButtonLabel("Mic Unsupported");
            scheduleRefresh(4000);
            return;
        }

        rememberCursor();
        try {
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
            mediaRecorder = mimeType ? new MediaRecorder(mediaStream, { mimeType }) : new MediaRecorder(mediaStream);
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
                    state.isBusy = false;
                    setButtonLabel("No Audio Captured");
                    scheduleRefresh(2500);
                    return;
                }
                await sendToServer(blob);
            };

            mediaRecorder.start();
            state.isRecording = true;
            refreshButton();
        } catch (error) {
            state.isRecording = false;
            state.isBusy = false;
            setButtonLabel(`Mic Error: ${error.message}`);
            scheduleRefresh(4000);
        }
    };

    const stopRecording = () => {
        if (!mediaRecorder || !state.isRecording) return;
        state.isRecording = false;
        state.isBusy = true;
        setButtonLabel("Recognizing");
        try {
            mediaRecorder.stop();
        } catch (error) {
            state.isBusy = false;
            setButtonLabel(`Error: ${error.message}`);
            scheduleRefresh(4000);
        }
    };

    const handleClick = (event) => {
        if (state.isBusy) return;
        if (state.isRecording) {
            stopRecording();
            return;
        }
        if (!state.modelReady) {
            downloadModel(Boolean(event?.shiftKey));
            return;
        }
        startRecording();
    };

    button = node.addWidget("button", "Loading Voice Model", null, function voiceRecognitionButton() {
        const event = Array.from(arguments).find((arg) => arg instanceof Event);
        handleClick(event);
    });
    button.__tsVoiceRecognitionButton = true;

    refreshStatus();
}

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedWrapper() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setupVoiceRecognition(this);
            return result;
        };
    },
    loadedGraphNode(node) {
        if (![node?.type, node?.comfyClass].includes(NODE_NAME)) return;
        setupVoiceRecognition(node);
    },
});
