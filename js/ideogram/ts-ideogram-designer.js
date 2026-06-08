// TS_IdeogramDesigner front-end entry point. UI logic lives in
// ./_ideogram_node.js (in-node preview) and ./_ideogram_editor.js (modal
// editor); this module only registers the extension under a stable ID and
// wires the DOM widget on node creation / workflow restore. Pattern mirrors
// js/image/sam_media_loader/ts-sam-media-loader.js.

import { app } from "/scripts/app.js";

import { NODE_NAME, setupIdeogramNode } from "./_ideogram_node.js";

const EXTENSION_ID = "ts.ideogramDesigner";

function isTargetNode(node) {
    return node?.comfyClass === NODE_NAME || node?.type === NODE_NAME;
}

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedWrapper() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (!this._tsIdeoCleanup) {
                setupIdeogramNode(this);
            }
            return result;
        };
    },
    loadedGraphNode(node) {
        if (!isTargetNode(node)) return;
        if (!node._tsIdeoCleanup) {
            setupIdeogramNode(node);
        } else {
            node._tsIdeoSync?.();
        }
    },
});
