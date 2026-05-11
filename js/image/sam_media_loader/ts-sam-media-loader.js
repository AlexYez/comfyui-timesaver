// TS_SAM_MediaLoader frontend entry point. UI logic lives in
// ./_sam_media_helpers.js; this module only registers the extension under a
// stable ID and wires the DOM widget on node creation / workflow restore.

import { app } from "/scripts/app.js";

import {
    DOM_WIDGET_NAME,
    NODE_NAME,
    getWidget,
    setupSamMediaLoader,
} from "./_sam_media_helpers.js";

const EXTENSION_ID = "ts.samMediaLoader";

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedWrapper() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (!getWidget(this, DOM_WIDGET_NAME)) {
                setupSamMediaLoader(this);
            }
            return result;
        };
    },
    loadedGraphNode(node) {
        if (node?.type !== NODE_NAME && node?.comfyClass !== NODE_NAME) return;
        if (!getWidget(node, DOM_WIDGET_NAME)) {
            setupSamMediaLoader(node);
        }
    },
});
