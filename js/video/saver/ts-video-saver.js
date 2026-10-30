// Точка входа TS Video Saver. Только регистрация — плеер в `_video_saver.js`.

import { app } from "/scripts/app.js";

import { getWidget } from "../../_dom_widget.js";
import { DOM_WIDGET, NODE_TYPE, UI_KEY, setupVideoSaver } from "./_video_saver.js";

// A background workflow has no live node to receive onExecuted. ComfyUI keeps
// its outputs in the workflow's change tracker and restores them after loading
// the graph, later than loadedGraphNode. Restore our custom player at that hook.
function restoreOutputs(outputs) {
    const root = app.rootGraph || app.graph;
    if (!root || !outputs) return;
    const workflow = app.extensionManager?.workflow;
    const graphs = [root, ...(root.subgraphs?.values?.() || [])];
    for (const graph of graphs) {
        for (const node of graph._nodes || []) {
            if (node.comfyClass !== NODE_TYPE && node.type !== NODE_TYPE) continue;
            const locator = workflow?.nodeToNodeLocatorId?.(node) ?? String(node.id);
            const payload = outputs[locator]?.[UI_KEY]?.[0];
            if (payload) node._tsVideoSaverApply?.(payload);
        }
    }
}

app.registerExtension({
    name: "ts.videoSaver",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function tsVideoSaverCreated(...args) {
            const result = onCreated?.apply(this, args);
            setupVideoSaver(this);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function tsVideoSaverExecuted(message, ...rest) {
            const payload = message?.[UI_KEY]?.[0];
            if (payload) this._tsVideoSaverApply?.(payload);
            return onExecuted?.apply(this, [message, ...rest]);
        };
    },
    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_TYPE && node?.type !== NODE_TYPE) return;
        // Виджет не пересоздаём — в Nodes 2.0 это двоит шапку ноды (§12.5.12).
        if (!getWidget(node, DOM_WIDGET)) setupVideoSaver(node);
        else node._tsVideoSaverRehydrate?.();
    },
    onNodeOutputsUpdated(outputs) {
        restoreOutputs(outputs);
    },
});
