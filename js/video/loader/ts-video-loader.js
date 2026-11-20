// Точка входа TS Video Loader. Только регистрация — вся работа в `_video_loader.js`.

import { app } from "/scripts/app.js";

import { getWidget } from "../../_dom_widget.js";
import { DOM_WIDGET, NODE_TYPE, setupVideoLoader } from "./_video_loader.js";

// LiteGraph can restore the output array saved by an older workflow/clipboard.
// Only append the new slot; existing objects and their links stay untouched.
function appendCropOutput(node) {
    const type = node?.comfyClass || node?.type;
    if (type === NODE_TYPE && node.outputs?.length === 4) node.addOutput("full_images", "IMAGE");
    if (type === "TS_VideoInfo" && node.outputs?.length === 13) node.addOutput("crop_data", "CROP_DATA");
}

app.registerExtension({
    name: "ts.videoLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (![NODE_TYPE, "TS_VideoInfo"].includes(nodeData?.name)) return;
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function tsVideoCropConfigured(...args) {
            const result = onConfigure?.apply(this, args);
            appendCropOutput(this);
            return result;
        };
        if (nodeData.name !== NODE_TYPE) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function tsVideoLoaderCreated(...args) {
            const result = onCreated?.apply(this, args);
            setupVideoLoader(this);
            return result;
        };
    },
    loadedGraphNode(node) {
        appendCropOutput(node);
        if (node?.comfyClass !== NODE_TYPE && node?.type !== NODE_TYPE) return;
        // ⚠️ Виджет не пересобираем: в Nodes 2.0 повторная регистрация DOM-виджета
        // двоит верхнюю часть ноды (§12.5.12). Только перечитываем состояние.
        if (!getWidget(node, DOM_WIDGET)) setupVideoLoader(node);
        else node._tsVideoLoaderRehydrate?.();
    },
});
