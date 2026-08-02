// TS Image Studio — extension entry.
//
// Registers the node UI: an "Open Interface" launcher on TS_ImageStudio and
// the hidden state widgets (session_id, result_path) mirrored to
// node.properties (CLAUDE.md §12.5.13). The application itself lives in
// _app.js and mounts on demand.

import { app } from "/scripts/app.js";
import {
    createOpenInterfaceButton,
    ensureThemeStyles,
    TS_UI_CLASS,
} from "../../_theme.js";
import { openStudio } from "./_app.js";

const NODE_ID = "TS_ImageStudio";
const W_SESSION = "session_id";
const W_RESULT = "result_path";

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (widget) {
        widget.value = value;
        widget.callback?.(value);
    }
    node.properties ||= {};
    node.properties[name] = value;
}

function readPersisted(node, name) {
    const widget = getWidget(node, name)?.value;
    if (widget !== undefined && widget !== null && widget !== "") return String(widget);
    const prop = node?.properties?.[name];
    if (prop !== undefined && prop !== null && prop !== "") return String(prop);
    return "";
}

function setupStudioNode(node) {
    ensureThemeStyles();
    for (const name of [W_SESSION, W_RESULT]) {
        const widget = getWidget(node, name);
        if (widget) widget.type = "hidden";
    }

    const host = document.createElement("div");
    host.className = `${TS_UI_CLASS} ts-istudio-launch`;
    host.style.cssText = "display:flex;align-items:center;justify-content:center;padding:6px";
    const button = createOpenInterfaceButton(() => {
        openStudio(node, {
            sessionId: readPersisted(node, W_SESSION),
            setSessionId: (id) => setWidgetValue(node, W_SESSION, id),
            setResultPath: (path) => {
                setWidgetValue(node, W_RESULT, path);
                node.graph?.setDirtyCanvas(true, true);
            },
        }).catch((err) => console.error("[TS Studio] failed to open", err));
    });
    host.appendChild(button);
    node.addDOMWidget("ts_studio_launch", "div", host, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 44,
        getMaxHeight: () => 44,
    });

    node._tsStudioRehydrate = () => {
        // Values restored from the workflow land AFTER onNodeCreated; mirror
        // them into properties so both channels agree (§12.5.12).
        node.properties ||= {};
        node.properties[W_SESSION] = readPersisted(node, W_SESSION);
        node.properties[W_RESULT] = readPersisted(node, W_RESULT);
    };
}

app.registerExtension({
    name: "ts.imageStudio",
    nodeCreated(node) {
        if (node?.comfyClass === NODE_ID) setupStudioNode(node);
    },
    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_ID) return;
        if (!getWidget(node, "ts_studio_launch")) setupStudioNode(node);
        node._tsStudioRehydrate?.();
    },
});
