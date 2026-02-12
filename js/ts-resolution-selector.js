import { app } from "/scripts/app.js";

const EXTENSION_ID = "ts.resolutionselector";
const NODE_NAME = "TS_ResolutionSelector";
const INPUT_RATIO = "aspect_ratio";
const STYLE_ID = "ts-resolution-selector-styles";
const DEFAULT_WIDGET_HEIGHT = 250;
const MIN_WIDGET_HEIGHT = 220;
const MIN_NODE_WIDTH = 300;
const MIN_NODE_HEIGHT = 260;

const RATIO_PRESETS = [
    { label: "1:1", value: "1:1" },
    { label: "4:3", value: "4:3" },
    { label: "3:2", value: "3:2" },
    { label: "16:9", value: "16:9" },
    { label: "21:9", value: "21:9" },
    { label: "3:4", value: "3:4" },
    { label: "2:3", value: "2:3" },
    { label: "9:16", value: "9:16" },
    { label: "9:21", value: "9:21" },
];

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-reso-selector {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 6px;
    box-sizing: border-box;
    overflow: hidden;
    min-height: 0;
    height: 100%;
    color: #e6e7ea;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    pointer-events: auto;
}
.ts-reso-grid {
    display: grid;
    grid-template-columns: repeat(3, var(--ts-reso-cell, 56px));
    grid-template-rows: repeat(3, var(--ts-reso-cell, 56px));
    gap: 5px;
    flex: 1 1 auto;
    min-height: 0;
    width: 100%;
    height: 100%;
    align-content: center;
    justify-content: center;
    overflow: hidden;
}
.ts-reso-card {
    border: 1px solid #2d343f;
    border-radius: 8px;
    background: #14171c;
    padding: 4px 3px 5px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    cursor: pointer;
    transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    color: inherit;
    width: 100%;
    height: 100%;
}
.ts-reso-card:hover {
    border-color: #465267;
    box-shadow: 0 0 0 1px rgba(80, 110, 170, 0.18);
}
.ts-reso-card.is-selected {
    border-color: #27d8b2;
    box-shadow: 0 0 0 1px rgba(39, 216, 178, 0.4);
    background: #152424;
}
.ts-reso-icon-wrap {
    width: 78%;
    height: 58%;
    display: flex;
    align-items: center;
    justify-content: center;
}
.ts-reso-icon {
    height: 70%;
    aspect-ratio: var(--ts-reso-ratio, 1 / 1);
    width: auto;
    max-width: 100%;
    border-radius: 4px;
    border: 1px solid #848c9d;
    background: linear-gradient(135deg, #2c3442 0%, #1b2028 100%);
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.06);
}
.ts-reso-label {
    font-size: clamp(8px, 1.2vh, 10px);
    letter-spacing: 0.02em;
    color: #d6dae2;
}
`;
    document.head.appendChild(style);
}

function stopPropagation(element, events) {
    events.forEach((eventName) => {
        element.addEventListener(eventName, (event) => {
            event.stopPropagation();
        });
    });
}

function parseRatio(value) {
    if (!value) {
        return [1, 1];
    }
    const parts = String(value).split(":");
    if (parts.length !== 2) {
        return [1, 1];
    }
    const w = Number(parts[0]);
    const h = Number(parts[1]);
    if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
        return [1, 1];
    }
    return [w, h];
}

function isTargetNode(node) {
    return node?.comfyClass === NODE_NAME || node?.type === NODE_NAME;
}

function isNodesV2() {
    if (typeof window === "undefined") {
        return false;
    }
    return Boolean(window.comfyAPI?.domWidget?.DOMWidgetImpl);
}

function setupResolutionSelector(node) {
    if (!node || node._tsResolutionSelectorInitialized) {
        return;
    }
    node._tsResolutionSelectorInitialized = true;

    if (typeof node.addDOMWidget !== "function") {
        return;
    }

    ensureStyles();

    node.resizable = true;
    if (!node.size || node.size.length < 2) {
        node.size = [Math.max(320, MIN_NODE_WIDTH), Math.max(320, MIN_NODE_HEIGHT)];
    } else {
        node.size = [Math.max(MIN_NODE_WIDTH, node.size[0]), Math.max(MIN_NODE_HEIGHT, node.size[1])];
    }

    const ratioWidget = node.widgets?.find((widget) => widget.name === INPUT_RATIO);
    if (ratioWidget) {
        ratioWidget.hidden = true;
        ratioWidget.computeSize = () => [0, -4];
    }

    const container = document.createElement("div");
    container.className = "ts-reso-selector";
    const isV2 = isNodesV2();
    let widgetHeight = DEFAULT_WIDGET_HEIGHT;
    let baseHeight = null;
    let domWidgetEl = null;
    let layoutRaf = null;

    const grid = document.createElement("div");
    grid.className = "ts-reso-grid";
    stopPropagation(grid, ["wheel"]);
    container.appendChild(grid);

    const buttons = new Map();
    RATIO_PRESETS.forEach((item) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-reso-card";
        button.dataset.value = item.value;

        const iconWrap = document.createElement("div");
        iconWrap.className = "ts-reso-icon-wrap";
        const icon = document.createElement("div");
        icon.className = "ts-reso-icon";
        const [rw, rh] = parseRatio(item.value);
        icon.style.setProperty("--ts-reso-ratio", `${rw} / ${rh}`);
        iconWrap.appendChild(icon);

        const label = document.createElement("div");
        label.className = "ts-reso-label";
        label.textContent = item.label;

        button.appendChild(iconWrap);
        button.appendChild(label);
        grid.appendChild(button);
        buttons.set(item.value, button);

        stopPropagation(button, ["pointerdown", "mousedown", "mouseup", "dblclick", "contextmenu"]);
    });

    stopPropagation(container, [
        "pointerdown",
        "pointerup",
        "mousedown",
        "mouseup",
        "wheel",
        "dblclick",
        "contextmenu",
    ]);

    const widgetOptions = {
        serialize: false,
        hideOnZoom: true,
    };
    if (isV2) {
        widgetOptions.getMinHeight = () => widgetHeight;
        widgetOptions.getMaxHeight = () => widgetHeight;
    }
    const domWidget = node.addDOMWidget("ts_resolution_selector", "div", container, widgetOptions);
    domWidgetEl = domWidget?.element || domWidget?.el || domWidget?.container;
    if (domWidgetEl) {
        domWidgetEl.style.overflow = "hidden";
    }

    domWidget.computeSize = function (width) {
        return [width, widgetHeight];
    };

    const ensureBaseHeight = () => {
        if (isV2) {
            return;
        }
        if (baseHeight !== null) {
            return;
        }
        const nodeHeight = node.size?.[1];
        if (!nodeHeight) {
            return;
        }
        baseHeight = Math.max(0, nodeHeight - widgetHeight);
    };

    const syncWidgetHeightFromNode = () => {
        if (isV2) {
            return;
        }
        ensureBaseHeight();
        if (baseHeight === null) {
            return;
        }
        const nodeHeight = node.size?.[1];
        if (!nodeHeight) {
            return;
        }
        widgetHeight = Math.max(MIN_WIDGET_HEIGHT, nodeHeight - baseHeight);
    };

    const updateGridLayout = () => {
        const host = domWidgetEl || container;
        const width = host?.clientWidth || node.size?.[0] || 0;
        const height = host?.clientHeight || (isV2 ? 0 : widgetHeight) || 0;
        if (!width || !height) {
            return;
        }
        const padding = 12;
        const gap = 5;
        const availableW = Math.max(0, width - padding - gap * 2);
        const fallbackH = height || availableW;
        const availableH = Math.max(0, fallbackH - padding - gap * 2);
        const cellSize = Math.max(32, Math.floor(Math.min(availableW / 3, availableH / 3)));
        grid.style.setProperty("--ts-reso-cell", `${cellSize}px`);
    };

    const scheduleLayout = () => {
        if (layoutRaf) {
            return;
        }
        layoutRaf = requestAnimationFrame(() => {
            layoutRaf = null;
            updateGridLayout();
        });
    };

    const state = {
        selected: "",
    };

    const applySelection = (value, trigger = true) => {
        if (!value) {
            return;
        }
        state.selected = value;
        buttons.forEach((button, key) => {
            button.classList.toggle("is-selected", key === value);
        });
        if (ratioWidget && trigger) {
            ratioWidget.value = value;
            ratioWidget.callback?.(value);
        }
        if (node.setProperty) {
            node.setProperty(INPUT_RATIO, value);
        } else {
            node.properties ||= {};
            node.properties[INPUT_RATIO] = value;
        }
        node.setDirtyCanvas(true, true);
    };

    buttons.forEach((button, value) => {
        button.addEventListener("click", (event) => {
            event.preventDefault();
            applySelection(value, true);
        });
    });

    const syncSelection = () => {
        const stored = ratioWidget?.value || node.properties?.[INPUT_RATIO];
        const defaultValue = stored || RATIO_PRESETS[0].value;
        applySelection(defaultValue, false);
    };

    node._tsResolutionSelectorSync = () => {
        syncSelection();
        scheduleLayout();
    };

    if (typeof ResizeObserver === "function") {
        const observer = new ResizeObserver(() => scheduleLayout());
        observer.observe(container);
        node._tsResolutionSelectorObserver = observer;
    }

    const prevOnResize = node.onResize;
    node.onResize = function () {
        const result = prevOnResize?.apply(this, arguments);
        syncWidgetHeightFromNode();
        scheduleLayout();
        return result;
    };

    const prevOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        if (layoutRaf) {
            cancelAnimationFrame(layoutRaf);
            layoutRaf = null;
        }
        if (node._tsResolutionSelectorObserver) {
            node._tsResolutionSelectorObserver.disconnect();
            node._tsResolutionSelectorObserver = null;
        }
        return prevOnRemoved?.apply(this, arguments);
    };

    requestAnimationFrame(() => {
        syncWidgetHeightFromNode();
        scheduleLayout();
    });
    syncSelection();
}

app.registerExtension({
    name: EXTENSION_ID,
    nodeCreated(node) {
        if (!isTargetNode(node)) {
            return;
        }
        setupResolutionSelector(node);
    },
    loadedGraphNode(node) {
        if (!isTargetNode(node)) {
            return;
        }
        if (!node._tsResolutionSelectorInitialized) {
            setupResolutionSelector(node);
        }
        node._tsResolutionSelectorSync?.();
    },
});
