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
/** Widgets that occupy a slot in the positional `widgets_values` array. */
function isSerializableWidget(widget) {
    if (!widget) return false;
    if (widget.options?.serialize === false) return false;
    if (widget.serialize === false) return false;
    return true;
}

function serializableWidgetCount(node) {
    return (node?.widgets || []).filter(isSerializableWidget).length;
}

/**
 * The value kind a widget accepts, from its STABLE LiteGraph type.
 *
 * Deliberately not `typeof widget.value`: by the time onConfigure runs the
 * value may already be a mis-shifted one of the wrong type, which is exactly
 * what we are trying to detect.
 */
function expectedKind(widget) {
    const type = String(widget?.type || "").toLowerCase();
    if (type === "number" || type === "slider") return "number";
    if (type === "toggle" || type === "boolean") return "boolean";
    if (type === "combo") return "combo";
    if (type === "text" || type === "string" || type === "customtext" || type === "textarea") return "string";
    return "unknown";
}

function widgetAccepts(widget, value) {
    switch (expectedKind(widget)) {
        case "number": return typeof value === "number";
        case "boolean": return typeof value === "boolean";
        case "string": return typeof value === "string";
        case "combo": {
            const values = widget?.options?.values;
            if (Array.isArray(values) && values.length) return values.includes(value);
            return typeof value === "string";
        }
        default: return false; // unknown: neutral, scores for no layout
    }
}

function resolveWidget(node, name) {
    return node?.widgets?.find((w) => w?.name === name) || node?._tsHiddenWidgets?.[name] || null;
}

/**
 * How well a positional value array fits a candidate widget layout.
 *
 * DOM widgets are skipped: they carry no meaningful value and appear in only
 * one of the two layouts, so counting them would bias the comparison.
 */
function layoutFitScore(node, names, values) {
    let score = 0;
    const limit = Math.min(names.length, values.length);
    for (let i = 0; i < limit; i += 1) {
        const widget = resolveWidget(node, names[i]);
        if (!widget || !isSerializableWidget(widget)) continue;
        if (widgetAccepts(widget, values[i])) score += 1;
    }
    return score;
}

/**
 * Re-apply a workflow saved BEFORE this module started removing hidden widgets.
 *
 * LiteGraph serialises widget values POSITIONALLY (`widgets_values` is a plain
 * array, one slot per serialisable widget, in node.widgets order). Older builds
 * hid state-carrier widgets with `widget.hidden = true`, which KEPT them in
 * node.widgets — so their values are in that array. Now hideWidget removes them
 * (the only way to kill the Vue phantom row), which makes the array one slot
 * shorter than the save: LiteGraph then shifts every value into the wrong
 * widget and the hidden ones silently fall back to their Python defaults.
 * (Symptom users hit: TS Resolution Selector always emitting a square, because
 * `aspect_ratio` kept its "1:1" default while `resolution` got "16:9".)
 *
 * `node._tsWidgetOrder` records the ORIGINAL order captured before the first
 * removal, so a legacy array can be mapped back by name.
 *
 * Telling the two formats apart by LENGTH does not work: LiteGraph pushes a
 * slot for EVERY widget including the node's own DOM widget, so a current save
 * can be exactly as long as a legacy one. Instead both candidate layouts are
 * scored by how well the values fit each widget's declared type, and the array
 * is only remapped when the original layout fits strictly better. A modern
 * workflow always fits its own layout best, so it is never touched; a tie
 * (ambiguous) also leaves the data alone.
 *
 * @param {object} node LiteGraph node.
 * @param {object} info Serialized node data passed to onConfigure.
 */
function restoreLegacyWidgetValues(node, info) {
    const order = node?._tsWidgetOrder;
    const stash = node?._tsHiddenWidgets;
    if (!Array.isArray(order) || !stash) return;
    const values = info?.widgets_values;
    // Object-keyed saves address widgets by name and cannot shift.
    if (!Array.isArray(values) || !values.length) return;
    // A legacy array carries a slot for EVERY original widget. Anything shorter
    // is a current-format save (whose only slots may be the node's own DOM
    // widget) — remapping that would overwrite values already restored from
    // node.properties with the DOM widget's empty placeholder.
    const currentWidgets = node.widgets || [];
    if (values.length < order.length) return;
    if (values.length < currentWidgets.length) return;

    const currentNames = currentWidgets.map((w) => w?.name);
    const legacyScore = layoutFitScore(node, order, values);
    const currentScore = layoutFitScore(node, currentNames, values);

    if (values.length === currentWidgets.length) {
        // Same length: only a strict type-fit win proves this is the old
        // layout. When the current layout has no scoreable widget at all (every
        // widget is hidden, so the single slot belongs to the DOM widget) the
        // comparison is meaningless — leave the data alone, because
        // node.properties already carries the truth for such saves.
        const scoreableCurrent = currentWidgets.filter(isSerializableWidget).length;
        if (!scoreableCurrent || legacyScore <= currentScore) return;
    } else if (legacyScore < currentScore) {
        return; // more slots than this layout holds, but it still fits better
    }

    for (let i = 0; i < values.length; i += 1) {
        const name = order[i];
        const value = values[i];
        if (!name || value === undefined) continue;
        const widget = node.widgets?.find((w) => w?.name === name) || stash[name];
        if (!widget) continue;
        try {
            widget.value = value;
        } catch {
            continue; // non-writable widget — leave it alone
        }
        if (stash[name]) {
            // Mirror to properties too: hideWidget seeded them with the default
            // at creation, and the re-seed below would otherwise clobber us.
            node.properties = node.properties || {};
            node.properties[name] = value;
        }
    }
}

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
    node.onConfigure = function tsOnConfigure(info) {
        // FIRST: repair a pre-removal (legacy) widgets_values array, so the
        // node's own onConfigure below already reads correct values.
        try {
            restoreLegacyWidgetValues(node, info);
        } catch (err) {
            console.warn("[TS DomWidget] legacy widget restore failed", err);
        }
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
        // Snapshot the widget order BEFORE the first removal. This is the layout
        // every previously saved workflow used for its positional
        // `widgets_values`, and restoreLegacyWidgetValues() maps them back by it.
        if (!node._tsWidgetOrder) {
            node._tsWidgetOrder = (node.widgets || [])
                .filter(isSerializableWidget)
                .map((w) => w.name);
        }
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
            // removeInput(), not splice(): every link stores the INDEX of the
            // input it lands on, and LiteGraph renumbers those links when a slot
            // is removed through the API. A bare splice left the links of every
            // input after this one pointing one slot too far — a wire attached
            // to the next input silently moved to its neighbour.
            if (typeof node.removeInput === "function") {
                node.removeInput(idx);
            } else {
                node.inputs.splice(idx, 1);
            }
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
