import { app } from "/scripts/app.js";

import { TS_UI_CLASS, createRatioCards, ensureThemeStyles } from "../_theme.js";
import { addResizableDomWidget, hideWidget as sharedHideWidget, getWidget as sharedGetWidget } from "../_dom_widget.js";

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
/* The cards themselves are the pack's shared control (the .ts-ui-ratio family
   in js/_theme.js) — this node only says how they fill ITS widget: the node can be
   resized, so the grid stretches to the full height instead of keeping the
   card's own compact box. Everything else — the frame, the label, the selected
   state — is the same object the studio draws.
   NOTE: no backticks in this comment — the whole stylesheet is one template
   literal, and one backtick would end it. */
.ts-reso-selector .ts-ui-ratios {
    grid-template-rows: repeat(3, 1fr);
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
}
.ts-reso-selector .ts-ui-ratio {
    height: 100%;
}
.ts-reso-selector .ts-ui-ratio__wrap {
    /* The node can be dragged, so the square grows with the card instead of
       staying at the shared token's size: height comes from the stretched grid
       row, aspect-ratio derives the width from it. Still a SQUARE — the frame's
       per-cent sides depend on that (js/_theme.js). */
    width: auto;
    height: 100%;
    max-width: 100%;
    aspect-ratio: 1 / 1;
    flex: 1 1 auto;
    min-height: 0;
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
function isTargetNode(node) {
    return node?.comfyClass === NODE_NAME || node?.type === NODE_NAME;
}




function hideRatioWidget(node) {
    // The card grid is the visible control for aspect_ratio, so the stock combo
    // is hidden via the shared helper (collapses in both renderers + drops the
    // converted-input row from the Nodes 2.0 grid).
    sharedHideWidget(node, INPUT_RATIO);
    return sharedGetWidget(node, INPUT_RATIO);
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

    const cards = createRatioCards({
        values: RATIO_PRESETS.map((item) => item.value),
        onSelect: (value) => applySelection(value, true),
    });
    const grid = cards.element;
    stopPropagation(grid, ["wheel"]);
    container.appendChild(grid);

    const buttons = cards.buttons;
    for (const button of buttons.values()) {
        stopPropagation(button, ["pointerdown", "mousedown", "mouseup", "dblclick", "contextmenu"]);
    }

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
        cards.select(value);
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
