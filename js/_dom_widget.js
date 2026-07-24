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

/** True when the Vue (Nodes 2.0) DOM-widget implementation is present. */
export function isNodesV2() {
    if (typeof window === "undefined") return false;
    return Boolean(window.comfyAPI?.domWidget?.DOMWidgetImpl);
}

function clamp(value, min) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.max(n, min) : min;
}

/**
 * Hide a native widget whose value a node's own DOM UI manages.
 *
 * This is the pack's ONE correct way to hide a state-carrier widget so it shows
 * in NEITHER renderer and stacks NO empty rows — every TS GUI node uses it.
 *
 * Two row sources have to be silenced:
 *  1. The widget itself. Nodes 2.0 (Vue) drops it from the node grid when
 *     widget.hidden is true (getLayoutWidgets filters on it); Nodes 1.0 is
 *     collapsed via computeSize + a no-op draw. We deliberately do NOT set
 *     widget.type = "hidden": that type is skipped by workflow value
 *     serialization (widgets_values), which would drop the user's text/state on
 *     reload for nodes that don't also mirror to node.properties.
 *  2. The widget-input SLOT. A converted-widget input renders a ~22px slot row
 *     in the Nodes 2.0 grid, and marking it hidden does NOT collapse it — five
 *     such rows stacked ~120px of dead space above the node's UI. Removing the
 *     input entry collapses the row for good, and the widget keeps serialising
 *     its value into the prompt (verified via graphToPrompt), so execution is
 *     unaffected. Only unconnected slots are removed, so a user's wire is never
 *     orphaned.
 *
 * @param {object} node LiteGraph node.
 * @param {string} name Widget/input name to hide.
 */
export function hideWidget(node, name) {
    const widget = (node?.widgets || []).find((w) => w?.name === name);
    if (widget) {
        widget.hidden = true;
        // type="hidden" is what actually collapses the widget in the Nodes 2.0
        // grid (widget.hidden alone leaves the native row at full height). We
        // guard the value against the hidden-type serialization skip by forcing
        // serialize + a serializeValue that returns the live value, so save/reload
        // keeps the user's state even for nodes that don't mirror to properties.
        widget.type = "hidden";
        widget.serialize = true;
        widget.serializeValue = widget.serializeValue || (() => widget.value);
        widget.options = { ...(widget.options || {}), hidden: true, serialize: true };
        widget.computeSize = () => [0, -4];
        widget.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
        widget.draw = () => {};
        const el = widget.element || widget.el || widget.container;
        if (el?.style) el.style.display = "none";
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
