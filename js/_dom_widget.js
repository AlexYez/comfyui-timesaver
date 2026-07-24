// Shared DOM-widget mounting for TS nodes that host their own resizable UI.
//
// Every such node needs the SAME sizing plumbing, and getting it wrong is the
// pack's most repeated bug: measuring getBoundingClientRect() (viewport pixels,
// scaled by canvas zoom) and writing it back into layout pixels skews the UI at
// any zoom other than 1 (CLAUDE.md §12.5.3), and the Nodes 1.0 vs Nodes 2.0
// (Vue / DOMWidgetImpl) renderers size DOM widgets through DIFFERENT hooks
// (§12.5.1). This module encodes the one correct approach once so a node just
// hands over its element and its bounds.
//
// The rule it enforces: NO JavaScript geometry. Heights come from node.size
// (graph units == layout pixels, zoom-independent) via computeSize in the
// legacy renderer and getMinHeight/getMaxHeight in Vue; the element's own CSS
// flexbox fills whatever slot it is granted. Because the sizing never reads
// rendered rects, canvas zoom cannot distort it.

import { app } from "/scripts/app.js";

/**
 * True when the DOMWidgetImpl (Nodes 2.0) widget implementation is present.
 *
 * NOTE: this deliberately keys off the CLASS, which exists in every modern build
 * even when Vue nodes are toggled OFF — so it is effectively always true today,
 * and this helper's DOM widgets ALWAYS take the getMinHeight/getMaxHeight path.
 * That is intentional and load-bearing: those bounds size the pane correctly in
 * BOTH the Vue and the classic canvas renderers for helper users. The computeSize
 * branch below is legacy (pre-DOMWidgetImpl) and would fight node.size in a
 * feedback loop if it ever ran, so we must NOT switch this to read the
 * Comfy.VueNodes.Enabled setting here. (The audio nodes DO read the setting —
 * they need computeSize in the classic renderer because their pane must fill an
 * exact height; they carry their own non-feedback computeSize for it.)
 */
export function isNodesV2() {
    if (typeof window === "undefined") return false;
    return Boolean(window.comfyAPI?.domWidget?.DOMWidgetImpl);
}

function clamp(value, min) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.max(n, min) : min;
}

/**
 * Find a node's widget by name — INCLUDING widgets the pack has hidden by
 * removing them from node.widgets (see hideWidget). Node code must use this
 * instead of node.widgets.find() so reads/writes of hidden state still work.
 *
 * @param {object} node LiteGraph node.
 * @param {string} name Widget name.
 * @returns {object|null}
 */
export function getWidget(node, name) {
    const live = node?.widgets?.find((w) => w?.name === name);
    if (live) return live;
    return node?._tsHiddenWidgets?.[name] || null;
}

// Wrap app.graphToPrompt ONCE so the values of widgets we removed from
// node.widgets (to kill the Nodes 2.0 phantom rows) are injected back into each
// node's prompt inputs. The value is read from the stashed widget itself — it
// stays live because the node's setValue path still finds it via getWidget().
// Without this, removed widgets would silently drop to their Python defaults.
let promptInjectorInstalled = false;
function installPromptInjector() {
    if (promptInjectorInstalled || !app || typeof app.graphToPrompt !== "function") return;
    promptInjectorInstalled = true;
    const original = app.graphToPrompt.bind(app);
    app.graphToPrompt = async function tsGraphToPrompt(...args) {
        const result = await original(...args);
        try {
            for (const node of app.graph?._nodes || []) {
                const stash = node?._tsHiddenWidgets;
                const entry = result?.output?.[String(node.id)];
                if (!stash || !entry?.inputs) continue;
                for (const name of Object.keys(stash)) {
                    // Never clobber a real wired connection to this input.
                    if (entry.inputs[name] === undefined) {
                        const value = stash[name]?.value;
                        if (value !== undefined) entry.inputs[name] = value;
                    }
                }
            }
        } catch (err) {
            console.warn("[TS DomWidget] prompt injection failed", err);
        }
        return result;
    };
}

// Keep stashed widget values and node.properties in sync across the two events
// that matter once a widget is out of node.widgets:
//   • onSerialize (SAVE): copy each stashed widget's live value into the node's
//     serialized properties so it survives save/reload (widgets_values no longer
//     carries it). onSerialize runs after LiteGraph populated `o`, so write into
//     o.properties directly.
//   • onConfigure (RELOAD): properties are restored AFTER onNodeCreated, so the
//     value hideWidget seeded at creation is stale. Re-seed each stashed widget
//     from the freshly restored properties here, before the node's own
//     onConfigure (e.g. SuperPrompt's syncUiFromWidgets) reads it.
function installNodeHooks(node) {
    if (node._tsNodeHooksInstalled) return;
    node._tsNodeHooksInstalled = true;

    const prevSerialize = node.onSerialize;
    node.onSerialize = function tsOnSerialize(o) {
        const result = prevSerialize?.apply(this, arguments);
        try {
            const stash = node._tsHiddenWidgets;
            if (stash) {
                node.properties = node.properties || {};
                if (o) o.properties = o.properties || {};
                for (const name of Object.keys(stash)) {
                    const value = stash[name]?.value;
                    if (value === undefined) continue;
                    node.properties[name] = value;
                    if (o?.properties) o.properties[name] = value;
                }
            }
        } catch (err) {
            console.warn("[TS DomWidget] serialize sync failed", err);
        }
        return result;
    };

    const prevConfigure = node.onConfigure;
    node.onConfigure = function tsOnConfigure() {
        const result = prevConfigure?.apply(this, arguments);
        try {
            const stash = node._tsHiddenWidgets;
            if (stash && node.properties) {
                for (const name of Object.keys(stash)) {
                    const value = node.properties[name];
                    if (value === undefined || value === null || value === "") continue;
                    try { stash[name].value = value; } catch { /* non-writable */ }
                }
            }
        } catch (err) {
            console.warn("[TS DomWidget] configure re-seed failed", err);
        }
        return result;
    };
}

/**
 * Hide a native widget whose value a node's own DOM UI manages.
 *
 * This is the pack's ONE correct way to hide a state-carrier widget — every TS
 * GUI node uses it, so this whole class of bug cannot recur per node.
 *
 * WHY REMOVAL, not type="hidden": in Nodes 2.0 (Vue) the node body is a CSS grid
 * that renders ONE row per entry in node.widgets. A widget marked hidden (or even
 * type="hidden") is NOT dropped from that grid — it renders an empty row whose
 * track ABSORBS the node's spare height. So on resize the empty gap grew instead
 * of the node's own UI (the exact bug users hit: "растягивается пустота, а не
 * окно"). getLayoutWidgets filters hidden for the HEIGHT calc, but the render
 * list does not, and there is no widget flag that removes the row. The only
 * reliable fix is to take the widget out of node.widgets entirely.
 *
 * Removal would normally break two things; both are handled here:
 *  - Value read/write: the widget object is stashed on node._tsHiddenWidgets and
 *    getWidget() (this module) still finds it. Its `value` is redefined to mirror
 *    to node.properties, so every write survives and can be re-injected.
 *  - Serialisation: node.properties is saved with the node (workflow save) and
 *    the prompt injector (installPromptInjector) puts the value back into the
 *    node's prompt inputs at queue time. Verified across save/reload + execute.
 *
 * Also removes the widget's converted-input SLOT (unconnected only) so no socket
 * row remains either.
 *
 * @param {object} node LiteGraph node.
 * @param {string} name Widget/input name to hide.
 */
export function hideWidget(node, name) {
    installPromptInjector();
    const widget = (node?.widgets || []).find((w) => w?.name === name);
    if (widget) {
        node.properties = node.properties || {};
        // Restore the value from properties when present (widgets_values dropped
        // it on reload because the widget was removed before save); otherwise
        // seed properties from the widget's current value. The stashed widget
        // then carries the live value — the node's setValue path keeps it in
        // sync, and onSerialize mirrors it back to properties at save time.
        const persisted = node.properties[name];
        if (persisted !== undefined && persisted !== null && persisted !== "") {
            try { widget.value = persisted; } catch { /* non-writable — ignore */ }
        } else {
            node.properties[name] = widget.value;
        }
        widget.hidden = true;
        widget.serialize = true;
        // Stash so getWidget() keeps finding it, then remove from node.widgets so
        // the Vue grid renders NO row for it (kills the height-stealing phantom).
        node._tsHiddenWidgets = node._tsHiddenWidgets || {};
        node._tsHiddenWidgets[name] = widget;
        const wi = node.widgets.indexOf(widget);
        if (wi >= 0) node.widgets.splice(wi, 1);
        installNodeHooks(node);
    }
    if (Array.isArray(node?.inputs)) {
        const idx = node.inputs.findIndex((i) => i?.name === name);
        if (idx >= 0 && node.inputs[idx] && !node.inputs[idx].link) {
            node.inputs.splice(idx, 1);
        }
    }
}

/**
 * Mount a resizable DOM widget with correct sizing in both renderers.
 *
 * @param {object} node LiteGraph node.
 * @param {HTMLElement} element The widget's root element (its CSS should be a
 *   flex/absolute layout that fills 100% height — this helper never sets inner
 *   geometry, only the widget slot).
 * @param {object} [options]
 * @param {string} [options.name="ts_dom_widget"] Widget name.
 * @param {number} [options.minWidth=200] Lower bound for node width.
 * @param {number} [options.minHeight=200] Lower bound for node height.
 * @param {number} [options.defaultWidth] Width applied when the node has none.
 * @param {number} [options.defaultHeight] Height applied when the node has none.
 * @param {number} [options.chromeHeight=0] Pixels above the widget (title bar +
 *   visible stock widgets) that the legacy renderer must leave for chrome.
 * @param {number} [options.minWidgetHeight=120] Lower bound for the widget's own height.
 * @param {() => void} [options.onResize] Extra callback after each resize.
 * @returns {{domWidget:object, element:HTMLElement, isV2:boolean, sanitize:() => void}}
 */
export function addResizableDomWidget(node, element, options = {}) {
    const {
        name = "ts_dom_widget",
        minWidth = 200,
        minHeight = 200,
        defaultWidth = minWidth,
        defaultHeight = minHeight,
        chromeHeight = 0,
        minWidgetHeight = 120,
        onResize,
    } = options;

    const sanitize = () => {
        const width = clamp(node.size?.[0] || defaultWidth, minWidth);
        const height = clamp(node.size?.[1] || defaultHeight, minHeight);
        node.size = [width, height];
        node.min_size = [minWidth, minHeight];
    };

    // Widget height derived from node.size — never from a measured rect.
    const widgetHeight = () =>
        Math.max(minWidgetHeight, clamp(node.size?.[1] || defaultHeight, minHeight) - chromeHeight);

    node.resizable = true;
    sanitize();

    const isV2 = isNodesV2();
    const widgetOptions = {
        serialize: false,
        // Keep the UI visible when zoomed out — hiding it reads as the node
        // losing its content.
        hideOnZoom: false,
    };
    if (isV2) {
        // Vue distributes height between these bounds; the CSS flex column
        // absorbs it. The ceiling MUST track the node height, or a tall
        // content block makes the layout grant all of it and the node balloons
        // (the fix is keeping the scrollable region out of flow in the node's
        // own CSS — see the style selector — so its natural height is small).
        widgetOptions.getMinHeight = () => minWidgetHeight;
        widgetOptions.getMaxHeight = () => widgetHeight();
    }

    const domWidget = node.addDOMWidget(name, "div", element, widgetOptions);
    const domWidgetEl = domWidget?.element || domWidget?.el || domWidget?.container || element;
    if (domWidgetEl) {
        domWidgetEl.style.width = "100%";
        domWidgetEl.style.overflow = "hidden";
    }

    // Legacy renderer sizes the widget through computeSize; assigning it in Vue
    // would force the fixed-size branch of computeLayoutSize (§12.5.1).
    if (!isV2) {
        domWidget.computeSize = function computeSize(width) {
            const safeWidth = clamp(width || node.size?.[0] || defaultWidth, minWidth);
            return [safeWidth, widgetHeight()];
        };
    }

    const previousOnResize = node.onResize;
    const onResizeWrapped = function onResizeWrapped() {
        const result = previousOnResize?.apply(this, arguments);
        sanitize();
        onResize?.();
        return result;
    };
    node.onResize = onResizeWrapped;

    // Nodes 2.0 grants the widget its height from the element's own min-content,
    // so the mounted element MUST have in-flow content (a flex column, a grid).
    // An element whose children are all position:absolute has zero min-content
    // and collapses to a sliver until a manual resize — give it a hidden in-flow
    // spacer of minWidgetHeight in that case (see js/ideogram/_ideogram_node.js).
    return { domWidget, element: domWidgetEl, isV2, sanitize };
}
