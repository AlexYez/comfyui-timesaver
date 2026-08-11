// Точка входа TS Video Loader. Только регистрация — вся работа в `_video_loader.js`.

import { app } from "/scripts/app.js";

import { getWidget } from "../../_dom_widget.js";
import { DOM_WIDGET, NODE_TYPE, setupVideoLoader } from "./_video_loader.js";

app.registerExtension({
    name: "ts.videoLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function tsVideoLoaderCreated(...args) {
            const result = onCreated?.apply(this, args);
            setupVideoLoader(this);
            return result;
        };
    },
    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_TYPE && node?.type !== NODE_TYPE) return;
        // ⚠️ Виджет не пересобираем: в Nodes 2.0 повторная регистрация DOM-виджета
        // двоит верхнюю часть ноды (§12.5.12). Только перечитываем состояние.
        if (!getWidget(node, DOM_WIDGET)) setupVideoLoader(node);
        else node._tsVideoLoaderRehydrate?.();
    },
});
