// TS_AudioPreview frontend entry point. Shares ./_audio_helpers.js with
// TS_AudioLoader; the helper detects preview-mode internally via node.type.

import { app } from "/scripts/app.js";

import {
    DOM_WIDGET_NAME,
    PREVIEW_NODE_NAME,
    PREVIEW_UI_KEY,
    getWidget,
    setupAudioLoader,
} from "./_audio_helpers.js";

const EXTENSION_ID = "ts.audioPreview";

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== PREVIEW_NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedWrapper() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
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
        if (node?.type !== PREVIEW_NODE_NAME && node?.comfyClass !== PREVIEW_NODE_NAME) return;
        if (!getWidget(node, DOM_WIDGET_NAME)) {
            setupAudioLoader(node);
            return;
        }
        if (typeof node._tsAudioLoaderRehydrate === "function") {
            node._tsAudioLoaderRehydrate();
        }
    },
});
