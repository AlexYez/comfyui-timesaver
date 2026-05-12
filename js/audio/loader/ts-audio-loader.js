// TS_AudioLoader frontend entry point. UI logic lives in ./_audio_helpers.js
// (shared with TS_AudioPreview); this module only registers the extension.

import { app } from "/scripts/app.js";

import {
    DOM_WIDGET_NAME,
    LOADER_NODE_NAME,
    getWidget,
    setupAudioLoader,
} from "./_audio_helpers.js";

const EXTENSION_ID = "ts.audioLoader";

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== LOADER_NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedWrapper() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (!getWidget(this, DOM_WIDGET_NAME)) {
                setupAudioLoader(this);
            }
            return result;
        };
    },
    loadedGraphNode(node) {
        if (node?.type !== LOADER_NODE_NAME && node?.comfyClass !== LOADER_NODE_NAME) return;
        if (!getWidget(node, DOM_WIDGET_NAME)) {
            setupAudioLoader(node);
            return;
        }
        if (typeof node._tsAudioLoaderRehydrate === "function") {
            node._tsAudioLoaderRehydrate();
        }
    },
});
