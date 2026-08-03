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
    pickLocaleStrings,
    TS_UI_CLASS,
} from "../../_theme.js";
import { publishAssetAction } from "../../_studio/_asset_actions.js";
import { studioStateFromPng } from "../../_studio/_pnginfo.js";
import { openStudio, openStudioInstance } from "./_app.js";

const NODE_ID = "TS_ImageStudio";
const W_SESSION = "session_id";
const W_RESULT = "result_path";

const ACTION_STRINGS = {
    en: {
        restore: "Restore studio session",
        noSession: "This image was not made in TS Image Studio.",
        unreadable: (reason) => `Could not read the image: ${reason}`,
        restored: "Session restored in TS Image Studio.",
    },
    ru: {
        restore: "Восстановить сессию студии",
        noSession: "Это изображение сделано не в TS Image Studio.",
        unreadable: (reason) => `Не удалось прочитать изображение: ${reason}`,
        restored: "Сессия восстановлена в TS Image Studio.",
    },
};

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

/** How a studio instance stores its session: in its node when it has one. */
function persistFor(node) {
    if (!node) return detachedPersist();
    return {
        sessionId: readPersisted(node, W_SESSION),
        setSessionId: (id) => setWidgetValue(node, W_SESSION, id),
        setResultPath: (path) => {
            setWidgetValue(node, W_RESULT, path);
            node.graph?.setDirtyCanvas(true, true);
        },
    };
}

// A studio opened from an asset browser may have no node on the graph. It
// still needs somewhere to keep its session id, so results of that sitting
// land together — but adding a node to someone's workflow uninvited is not
// ours to do, so the session lives here instead, for as long as the page does.
let detachedSession = "";

function detachedPersist() {
    return {
        sessionId: detachedSession,
        setSessionId: (id) => { detachedSession = id; },
        setResultPath: () => {},
    };
}

function findStudioNode() {
    const nodes = app.graph?._nodes || app.graph?.nodes || [];
    return nodes.find((candidate) => candidate?.comfyClass === NODE_ID) || null;
}

/**
 * Rebuild the session an image was made in.
 *
 * Called by asset browsers through the shared action registry, so it takes a
 * plain asset descriptor and answers with a message rather than touching the
 * caller's UI. An already-open studio is reused: restoring a session must
 * never leave the user with two studios on screen.
 *
 * @param {{url: string, filename?: string}} asset
 * @returns {Promise<{ok: boolean, message: string}>}
 */
async function recreateFromAsset(asset) {
    const t = pickLocaleStrings(ACTION_STRINGS);
    const url = String(asset?.url || "");
    if (!url) return { ok: false, message: t.unreadable("no URL") };
    let blob;
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        blob = await response.blob();
    } catch (err) {
        return { ok: false, message: t.unreadable(String(err?.message || err)) };
    }
    let found;
    try {
        found = await studioStateFromPng(blob);
    } catch (err) {
        return { ok: false, message: t.unreadable(String(err?.message || err)) };
    }
    if (!found?.state) return { ok: false, message: t.noSession };
    try {
        const host = findStudioNode();
        const studio = openStudioInstance() || await openStudio(host, persistFor(host));
        await studio.applyStudioState(found.state);
    } catch (err) {
        console.error("[TS Studio] restoring a session failed", err);
        return { ok: false, message: String(err?.message || err) };
    }
    return { ok: true, message: t.restored };
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
        openStudio(node, persistFor(node))
            .catch((err) => console.error("[TS Studio] failed to open", err));
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
    setup() {
        // Published once the extension loads, not once the studio opens: an
        // asset browser builds its menu long before anyone opens the studio.
        publishAssetAction({
            id: "ts-image-studio.recreate",
            label: { en: ACTION_STRINGS.en.restore, ru: ACTION_STRINGS.ru.restore },
            order: 20,
            // Only PNGs carry the snapshot chunk. Whether this particular PNG
            // has one is answered by run(), which has the file in hand — a
            // menu must not wait on a download to decide what to show.
            supports: (asset) => asset?.type === "image"
                && /\.png$/i.test(String(asset.extension || asset.filename || asset.url || "")),
            run: (asset) => recreateFromAsset(asset),
        });
    },
    nodeCreated(node) {
        if (node?.comfyClass === NODE_ID) setupStudioNode(node);
    },
    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_ID) return;
        if (!getWidget(node, "ts_studio_launch")) setupStudioNode(node);
        node._tsStudioRehydrate?.();
    },
});
