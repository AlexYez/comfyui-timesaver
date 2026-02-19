import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_ID = "ts_suite.style_prompt_selector";
const NODE_NAME = "TS_StylePromptSelector";
const STYLE_INPUT = "style_id";
const STYLE_CSS_ID = "ts-style-selector-styles";

const NODE_WIDTH = 250;
const NODE_HEIGHT = 300;
const WIDGET_HEIGHT = 240;
const GRID_GAP = 4;

function ensureStyles() {
    if (document.getElementById(STYLE_CSS_ID)) {
        return;
    }
    const style = document.createElement("style");
    style.id = STYLE_CSS_ID;
    style.textContent = `
.ts-style-selector {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 6px;
    box-sizing: border-box;
    overflow: hidden;
    height: 100%;
    min-height: 0;
    width: 100%;
    color: #e6e7ea;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    pointer-events: auto;
}
.ts-style-search {
    width: 100%;
    box-sizing: border-box;
    padding: 4px 6px;
    background: #141414;
    border: 1px solid #333;
    border-radius: 6px;
    color: #e8e8e8;
    outline: none;
    font-size: 11px;
}
.ts-style-search::placeholder {
    color: #8a8a8a;
}
.ts-style-grid {
    display: grid;
    grid-template-columns: repeat(3, var(--ts-card-size, 1fr));
    grid-auto-rows: var(--ts-card-size, auto);
    gap: 4px;
    flex: 1 1 auto;
    min-height: 0;
    width: 100%;
    align-items: start;
    align-content: start;
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 2px;
    padding-bottom: 15px;
    box-sizing: border-box;
}
.ts-style-card {
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    border: 1px solid #2d343f;
    border-radius: 6px;
    background: #14171c;
    padding: 0;
    cursor: pointer;
    overflow: hidden;
}
.ts-style-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.ts-style-card.is-selected {
    border-color: #4da3ff;
    box-shadow: 0 0 0 1px rgba(77, 163, 255, 0.4);
}
.ts-style-label {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 3px 4px;
    font-size: 9px;
    text-align: center;
    background: rgba(0, 0, 0, 0.55);
    color: #f0f0f0;
    box-sizing: border-box;
    pointer-events: none;
}
.ts-style-grid.has-selection .ts-style-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.75);
    pointer-events: none;
}
.ts-style-grid.has-selection .ts-style-card.is-selected::after {
    background: transparent;
}
.ts-style-empty {
    font-size: 11px;
    color: #9a9a9a;
    padding: 4px 2px;
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

function makePreviewUrl(relPath) {
    return api.apiURL(`/ts_styles/preview?path=${encodeURIComponent(relPath)}`);
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

function hideStyleWidget(node) {
    const widget = node?.widgets?.find((item) => item.name === STYLE_INPUT);
    if (widget) {
        widget.hidden = true;
        widget.type = "hidden";
        widget.serialize = true;
        widget.options = { ...(widget.options || {}), hidden: true, serialize: true };
        widget.computeSize = () => [0, -4];
    }

    const input = node?.inputs?.find((item) => item?.name === STYLE_INPUT);
    if (input) {
        input.hidden = true;
    }
}

function setupStyleSelector(node) {
    if (!node || node._tsStyleSelectorInitialized) {
        return;
    }
    node._tsStyleSelectorInitialized = true;

    if (typeof node.addDOMWidget !== "function") {
        return;
    }

    ensureStyles();
    hideStyleWidget(node);

    node.resizable = false;
    node.size = [NODE_WIDTH, NODE_HEIGHT];
    node.min_size = [NODE_WIDTH, NODE_HEIGHT];
    node.max_size = [NODE_WIDTH, NODE_HEIGHT];

    const styleWidget = node.widgets?.find((widget) => widget.name === STYLE_INPUT);
    if (styleWidget) {
        styleWidget.hidden = true;
        styleWidget.computeSize = () => [0, -4];
    }

    const container = document.createElement("div");
    container.className = "ts-style-selector";

    const search = document.createElement("input");
    search.type = "text";
    search.className = "ts-style-search";
    search.placeholder = "Search styles...";

    const grid = document.createElement("div");
    grid.className = "ts-style-grid";

    const empty = document.createElement("div");
    empty.className = "ts-style-empty";
    empty.textContent = "Loading styles...";

    container.appendChild(search);
    container.appendChild(grid);
    container.appendChild(empty);

    const isV2 = isNodesV2();
    const widgetOptions = {
        serialize: false,
        hideOnZoom: true,
    };
    if (isV2) {
        widgetOptions.getMinHeight = () => WIDGET_HEIGHT;
        widgetOptions.getMaxHeight = () => WIDGET_HEIGHT;
    }

    const domWidget = node.addDOMWidget("ts_style_selector", "div", container, widgetOptions);
    const domWidgetEl = domWidget?.element || domWidget?.el || domWidget?.container;
    if (domWidgetEl) {
        domWidgetEl.style.overflow = "hidden";
        domWidgetEl.style.height = `${WIDGET_HEIGHT}px`;
        domWidgetEl.style.minHeight = `${WIDGET_HEIGHT}px`;
        domWidgetEl.style.maxHeight = `${WIDGET_HEIGHT}px`;
    }
    container.style.height = `${WIDGET_HEIGHT}px`;
    container.style.minHeight = `${WIDGET_HEIGHT}px`;
    container.style.maxHeight = `${WIDGET_HEIGHT}px`;

    domWidget.computeSize = function () {
        return [NODE_WIDTH, WIDGET_HEIGHT];
    };

    const state = {
        styles: [],
        filtered: [],
        selectedValue: "",
        loading: true,
    };
    let layoutRaf = null;

    const updateLayout = () => {
        const containerHeight = container.clientHeight || WIDGET_HEIGHT;
        const searchHeight = search.getBoundingClientRect().height || 0;
        const gridHeight = Math.max(0, containerHeight - searchHeight - 6);
        grid.style.height = `${gridHeight}px`;
        grid.style.minHeight = `${gridHeight}px`;
        const gridWidth = grid.clientWidth || NODE_WIDTH;
        const available = Math.max(0, gridWidth - GRID_GAP * 2);
        const card = Math.max(24, Math.floor(available / 3));
        grid.style.setProperty("--ts-card-size", `${card}px`);
    };

    const scheduleLayout = () => {
        if (layoutRaf) {
            return;
        }
        layoutRaf = requestAnimationFrame(() => {
            layoutRaf = null;
            updateLayout();
        });
    };

    const styleValue = (style) => (style.name || style.id || "").trim();

    const matchesSelection = (style, value) => {
        if (!value) {
            return false;
        }
        return value === style.id || value === style.name || value === styleValue(style);
    };

    const setSelection = (value, trigger = true) => {
        state.selectedValue = value || "";
        grid.classList.toggle("has-selection", Boolean(state.selectedValue));
        grid.querySelectorAll(".ts-style-card").forEach((card) => {
            const isSelected = card.dataset.value === state.selectedValue;
            card.classList.toggle("is-selected", isSelected);
        });
        if (styleWidget && trigger) {
            styleWidget.value = state.selectedValue;
            styleWidget.callback?.(state.selectedValue);
        }
        if (node.setProperty) {
            node.setProperty(STYLE_INPUT, state.selectedValue);
        } else {
            node.properties ||= {};
            node.properties[STYLE_INPUT] = state.selectedValue;
        }
        node.setDirtyCanvas(true, true);
    };

    const renderGrid = () => {
        grid.innerHTML = "";

        if (state.loading) {
            empty.textContent = "Loading styles...";
            empty.style.display = "block";
            return;
        }

        if (!state.filtered.length) {
            empty.textContent = "No styles found.";
            empty.style.display = "block";
            return;
        }

        empty.style.display = "none";
        grid.classList.toggle("has-selection", Boolean(state.selectedValue));

        state.filtered.forEach((style) => {
            const value = styleValue(style);
            if (!value) {
                return;
            }
            const card = document.createElement("button");
            card.type = "button";
            card.className = "ts-style-card";
            card.dataset.value = value;
            if (matchesSelection(style, state.selectedValue)) {
                card.classList.add("is-selected");
            }
            card.title = style.description || style.prompt || style.name || style.id || "";

            if (style.preview) {
                const img = document.createElement("img");
                img.alt = style.name || style.id || "style";
                img.src = makePreviewUrl(style.preview);
                img.onerror = () => {
                    img.remove();
                };
                card.appendChild(img);
            }

            const label = document.createElement("div");
            label.className = "ts-style-label";
            label.textContent = style.name || style.id || "";
            card.appendChild(label);

            card.addEventListener("click", (event) => {
                event.preventDefault();
                const nextValue = value === state.selectedValue ? "" : value;
                setSelection(nextValue, true);
                renderGrid();
            });

            grid.appendChild(card);
        });
        scheduleLayout();
    };

    const applyFilter = () => {
        const query = search.value.trim().toLowerCase();
        if (!query) {
            state.filtered = state.styles.slice();
        } else {
            state.filtered = state.styles.filter((style) => {
                const haystack = [style.id, style.name, style.description, style.prompt]
                    .filter(Boolean)
                    .join(" ")
                    .toLowerCase();
                return haystack.includes(query);
            });
        }
        renderGrid();
    };

    const syncSelection = () => {
        const stored = styleWidget?.value || node.properties?.[STYLE_INPUT] || "";
        setSelection(stored, false);
    };

    const loadStyles = async () => {
        state.loading = true;
        renderGrid();
        try {
            const response = await fetch(api.apiURL("/ts_styles"));
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const payload = await response.json();
            state.styles = Array.isArray(payload.styles) ? payload.styles : [];
            state.loading = false;
            syncSelection();
            applyFilter();
            scheduleLayout();
        } catch (error) {
            state.loading = false;
            state.filtered = [];
            empty.textContent = "Failed to load styles.";
            empty.style.display = "block";
            console.error("[TS Style Prompt Selector] Failed to load styles:", error);
        }
    };

    search.addEventListener("input", applyFilter);

    stopPropagation(container, [
        "pointerdown",
        "pointerup",
        "mousedown",
        "mouseup",
        "wheel",
        "dblclick",
        "contextmenu",
    ]);
    stopPropagation(grid, ["wheel"]);

    node._tsStyleSelectorSync = () => {
        syncSelection();
        applyFilter();
        scheduleLayout();
    };

    const prevOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        if (layoutRaf) {
            cancelAnimationFrame(layoutRaf);
            layoutRaf = null;
        }
        return prevOnRemoved?.apply(this, arguments);
    };

    renderGrid();
    loadStyles();
    scheduleLayout();
}

app.registerExtension({
    name: EXTENSION_ID,
    nodeCreated(node) {
        if (!isTargetNode(node)) {
            return;
        }
        setupStyleSelector(node);
    },
    loadedGraphNode(node) {
        if (!isTargetNode(node)) {
            return;
        }
        if (!node._tsStyleSelectorInitialized) {
            setupStyleSelector(node);
        }
        node.resizable = false;
        node.size = [NODE_WIDTH, NODE_HEIGHT];
        node.min_size = [NODE_WIDTH, NODE_HEIGHT];
        node.max_size = [NODE_WIDTH, NODE_HEIGHT];
        hideStyleWidget(node);
        node._tsStyleSelectorSync?.();
    },
});
