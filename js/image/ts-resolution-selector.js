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




const RESOLUTION_DEFAULT = 1.5;
const CUSTOM_RATIO_DEFAULT = "0:0";
const RATIO_PATTERN = /^\s*\d+(?:\.\d+)?\s*:\s*\d+(?:\.\d+)?\s*$/;

/**
 * Put every widget back into a value it can actually hold.
 *
 * ⚠️ Графы, открытые со сдвигом значений до исправления в _dom_widget.js,
 * успели так и сохраниться: у владельца лежит `[1.5, 1.5, false, ""]` —
 * `custom_ratio` получил число от сдвинутого `resolution`. Такой граф ошибки не
 * даёт (строка «1.5» без двоеточия молча читается как 1:1), а «16:9» в
 * `resolution` роняет очередь: «couldn't be converted to FLOAT». Легаси-разбор
 * их уже не спасёт — в массиве нет правильных значений. Поэтому нода сама
 * приводит поля к допустимым: число из строки, если оно там есть, иначе
 * умолчание схемы. Меняются только невозможные значения — правильные не
 * трогаются никогда.
 *
 * @returns {string[]} names of the widgets that were repaired.
 */
function healWidgetValues(node) {
    const repaired = [];
    const set = (name, value) => {
        const widget = sharedGetWidget(node, name);
        if (!widget) return;
        widget.value = value;
        repaired.push(name);
    };

    const resolution = sharedGetWidget(node, "resolution");
    if (resolution) {
        const min = Number(resolution.options?.min ?? 0.1);
        const max = Number(resolution.options?.max ?? 3.0);
        let value = resolution.value;
        if (typeof value !== "number" || !Number.isFinite(value)) {
            const parsed = typeof value === "string" ? Number(value) : NaN;
            set("resolution", Number.isFinite(parsed) ? Math.min(max, Math.max(min, parsed)) : RESOLUTION_DEFAULT);
        } else if (value < min || value > max) {
            set("resolution", Math.min(max, Math.max(min, value)));
        }
    }

    // Только НЕ строка (число, оставшееся от сдвига). Любая строка — законное
    // значение строкового поля: её разбирает сервер, и решать за человека,
    // что он имел в виду, нода не вправе.
    const custom = sharedGetWidget(node, "custom_ratio");
    if (custom && typeof custom.value !== "string") {
        const text = String(custom.value ?? "");
        set("custom_ratio", RATIO_PATTERN.test(text) ? text : CUSTOM_RATIO_DEFAULT);
    }

    const original = sharedGetWidget(node, "original_aspect");
    if (original && typeof original.value !== "boolean") {
        set("original_aspect", original.value === "true" || original.value === 1);
    }

    const ratio = sharedGetWidget(node, INPUT_RATIO);
    const presets = RATIO_PRESETS.map((item) => item.value);
    if (ratio && !presets.includes(ratio.value)) {
        const stored = node.properties?.[INPUT_RATIO];
        set(INPUT_RATIO, presets.includes(stored) ? stored : presets[0]);
        node.properties ||= {};
        node.properties[INPUT_RATIO] = ratio.value;
    }

    if (repaired.length) {
        console.warn(`[TS Resolution Selector] node ${node.id}: repaired values left by an old save: ${repaired.join(", ")}`);
    }
    return repaired;
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
        // После legacy-разбора (он идёт в onConfigure, раньше этого хука) —
        // и ДО синхронизации карточек: они должны показать уже исправленное.
        try {
            healWidgetValues(node);
        } catch (error) {
            console.warn("[TS Resolution Selector] could not repair saved values", error);
        }
        node._tsResolutionSelectorSync?.();
    },
});
