import { app } from "/scripts/app.js";

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";
import { addResizableDomWidget } from "../_dom_widget.js";

const EXTENSION_ID = "ts.resolutionselector";
const NODE_NAME = "TS_ResolutionSelector";
const INPUT_RATIO = "aspect_ratio";
const STYLE_ID = "ts-resolution-selector-styles";
const DEFAULT_NODE_WIDTH = 250;
const DEFAULT_NODE_HEIGHT = 340;
const MIN_NODE_WIDTH = 180;
const MIN_NODE_HEIGHT = 260;
// Node title bar + slot rows above the DOM widget (legacy sizing only).
const WIDGET_CHROME_HEIGHT = 60;
const MIN_WIDGET_HEIGHT = 160;

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
    // Colours come from the shared --ts-* tokens (js/_theme.js); keep this
    // stylesheet to layout only.
    ensureThemeStyles();
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
    color: var(--ts-text);
    font-family: var(--ts-font);
    pointer-events: auto;
}
.ts-reso-grid {
    /* 3x3 of equal cells that fill the widget — no fixed pixel size, so the
       grid scales with the node in both renderers and at any canvas zoom
       (pure CSS, never JS geometry: see CLAUDE.md §12.5.3). */
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(3, 1fr);
    gap: 5px;
    flex: 1 1 auto;
    min-height: 0;
    width: 100%;
    /* A square cell would leave one axis short on a non-square node; instead
       cells fill the grid and the icon inside keeps the aspect proportions. */
    justify-items: stretch;
    align-items: stretch;
    overflow: hidden;
}
.ts-reso-card {
    border: 1px solid var(--ts-border-soft);
    border-radius: var(--ts-radius);
    background: var(--ts-surface);
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
    border-color: var(--ts-border-strong);
    background: var(--ts-surface-hover);
}
.ts-reso-card.is-selected {
    border-color: var(--ts-accent);
    box-shadow: 0 0 0 1px var(--ts-accent-line);
    background: var(--ts-accent-soft);
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
    border: 1px solid var(--ts-muted);
    background: var(--ts-elevated);
}
.ts-reso-label {
    font-size: var(--ts-fs-xs);
    letter-spacing: 0.02em;
    color: var(--ts-muted);
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




function hideRatioWidget(node) {
    // The card grid is the visible control for aspect_ratio, so the stock combo
    // must be fully hidden. type="hidden" is the part the Vue (Nodes 2.0)
    // renderer honours; the bare .hidden flag alone leaves it on screen and it
    // doubles the grid.
    const widget = node?.widgets?.find((item) => item.name === INPUT_RATIO);
    if (widget) {
        widget.hidden = true;
        widget.type = "hidden";
        widget.serialize = true;
        widget.options = { ...(widget.options || {}), hidden: true, serialize: true };
        widget.computeSize = () => [0, 0];
    }
    const input = node?.inputs?.find((item) => item?.name === INPUT_RATIO);
    if (input) input.hidden = true;
    return widget;
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

    const ratioWidget = hideRatioWidget(node);

    const container = document.createElement("div");
    container.className = `${TS_UI_CLASS} ts-reso-selector`;

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

    addResizableDomWidget(node, container, {
        name: "ts_resolution_selector",
        minWidth: MIN_NODE_WIDTH,
        minHeight: MIN_NODE_HEIGHT,
        defaultWidth: DEFAULT_NODE_WIDTH,
        defaultHeight: DEFAULT_NODE_HEIGHT,
        chromeHeight: WIDGET_CHROME_HEIGHT,
        minWidgetHeight: MIN_WIDGET_HEIGHT,
    });

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
    };

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
        } else {
            hideRatioWidget(node);
        }
        node._tsResolutionSelectorSync?.();
    },
});
