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
