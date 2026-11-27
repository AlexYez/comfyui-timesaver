// TS_AudioPreview frontend entry point. Shares ./_audio_helpers.js with
// TS_AudioLoader; the helper detects preview-mode internally via node.type.

import { app } from "/scripts/app.js";

import {
    DOM_WIDGET_NAME,
    PREVIEW_NODE_NAME,
    PREVIEW_UI_KEY,
    SAVER_NODE_NAME,
    getWidget,
    setupAudioLoader,
} from "./_audio_helpers.js";

const EXTENSION_ID = "ts.audioPreview";

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (![PREVIEW_NODE_NAME, SAVER_NODE_NAME].includes(nodeData.name)) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedWrapper() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (nodeData.name === SAVER_NODE_NAME) {
                const prefix = getWidget(this, "filename_prefix");
                if (prefix) prefix.serializeValue = () =>
                    typeof prefix.value === "string"
                        ? window.comfyAPI.utils.applyTextReplacements(app, prefix.value)
                        : prefix.value;
            }
            if (!getWidget(this, DOM_WIDGET_NAME)) {
                setupAudioLoader(this);
            }
            return result;
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function onExecutedWrapper(message) {
            const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message && message[PREVIEW_UI_KEY]?.[0]) {
                this._tsAudioLoaderApplyPayload?.(message[PREVIEW_UI_KEY][0], true);
            }
            return result;
        };
    },
    loadedGraphNode(node) {
        if (![PREVIEW_NODE_NAME, SAVER_NODE_NAME].includes(node?.comfyClass || node?.type)) return;
        if (!getWidget(node, DOM_WIDGET_NAME)) {
            setupAudioLoader(node);
            return;
        }
        if (typeof node._tsAudioLoaderRehydrate === "function") {
            node._tsAudioLoaderRehydrate();
        }
    },
});
