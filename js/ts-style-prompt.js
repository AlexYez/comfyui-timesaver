import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT_NAME = "ts_suite.style_prompt_selector";
const NODE_NAME = "TS_StylePromptSelector";
const STYLE_INPUT = "style_id";
const STYLE_CSS_ID = "ts-style-selector-styles";

function ensureStylesheet() {
    if (document.getElementById(STYLE_CSS_ID)) {
        return;
    }
    const style = document.createElement("style");
    style.id = STYLE_CSS_ID;
    style.textContent = `
.ts-style-selector {
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  color: #e8e8e8;
  box-sizing: border-box;
  padding-bottom: 12px;
  pointer-events: none;
}
.ts-style-search {
  width: 100%;
  box-sizing: border-box;
  padding: 6px 8px;
  background: #141414;
  border: 1px solid #333;
  border-radius: 6px;
  color: #e8e8e8;
  outline: none;
  pointer-events: auto;
}
.ts-style-search::placeholder {
  color: #8a8a8a;
}
.ts-style-grid {
  display: grid;
  --ts-card-size: 88px;
  grid-template-columns: repeat(auto-fit, minmax(var(--ts-card-size), 1fr));
  grid-auto-rows: var(--ts-card-size);
  gap: 8px;
  max-height: 300px;
  overflow: auto;
  padding: 4px 4px 4px 6px;
  align-content: start;
  justify-content: start;
  pointer-events: auto;
}
.ts-style-card {
  position: relative;
  width: 100%;
  aspect-ratio: 1 / 1;
  border: 1px solid #2a2a2a;
  border-radius: 8px;
  overflow: hidden;
  background: #1a1a1a;
  cursor: pointer;
  padding: 0;
  pointer-events: auto;
}
.ts-style-card img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.ts-style-card.is-selected {
  outline: 2px solid #4da3ff;
  box-shadow: 0 0 0 2px rgba(77, 163, 255, 0.35);
}
.ts-style-label {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 4px 6px;
  font-size: 10px;
  text-align: center;
  background: rgba(0, 0, 0, 0.55);
  color: #f0f0f0;
}
.ts-style-empty {
  font-size: 12px;
  color: #9a9a9a;
  padding: 4px 2px;
  pointer-events: auto;
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

function isNearResizeEdge(event, container, threshold = 12) {
    if (!container) {
        return false;
    }
    const rect = container.getBoundingClientRect();
    if (!rect.width || !rect.height) {
        return false;
    }
    const x = event.clientX;
    const y = event.clientY;
    return x >= rect.right - threshold || y >= rect.bottom - threshold;
}

function isTargetNode(node) {
    return node?.comfyClass === NODE_NAME || node?.type === NODE_NAME;
}

function setupStyleSelector(node) {
    if (!node || node._tsStyleSelectorInitialized) {
        return;
    }
    node._tsStyleSelectorInitialized = true;

    if (typeof node.addDOMWidget !== "function") {
        return;
    }

    ensureStylesheet();

    node.resizable = true;
    if (!node.size || node.size.length < 2) {
        node.size = [360, 420];
    } else {
        node.size = [Math.max(node.size[0], 360), Math.max(node.size[1], 420)];
    }

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
    stopPropagation(search, [
        "pointerdown",
        "pointerup",
        "mousedown",
        "mouseup",
        "wheel",
        "dblclick",
        "contextmenu",
    ]);

    const grid = document.createElement("div");
    grid.className = "ts-style-grid";
    stopPropagation(grid, ["wheel"]);

    const empty = document.createElement("div");
    empty.className = "ts-style-empty";
    empty.textContent = "Loading styles...";

    container.appendChild(search);
    container.appendChild(grid);
    container.appendChild(empty);

    const domWidget = node.addDOMWidget("ts_style_selector", "ts_style_selector", container, {
        serialize: false,
        hideOnZoom: true,
    });

    domWidget.computeSize = function (width) {
        const height = Math.max(240, (node.size?.[1] || 420) - 80);
        return [width, height];
    };

    const state = {
        styles: [],
        filtered: [],
        selectedId: "",
    };

    const updateLayout = () => {
        const width = Math.max(200, node.size?.[0] || container.clientWidth || 360);
        const height = Math.max(240, node.size?.[1] || container.clientHeight || 420);
        const gridWidth = grid.clientWidth || width;
        const horizontalPadding = 16;
        const available = Math.max(160, gridWidth - horizontalPadding);
        const gap = 8;
        const minSize = 72;
        const maxSize = 140;
        let columns = Math.max(2, Math.floor((available + gap) / (minSize + gap)));
        columns = Math.min(columns, 6);
        const rawSize = Math.floor((available - gap * (columns - 1)) / columns);
        const cardSize = Math.max(minSize, Math.min(maxSize, rawSize));
        grid.style.setProperty("--ts-card-size", `${cardSize}px`);
        grid.style.maxHeight = `${Math.max(140, height - 130)}px`;
    };
    let layoutFrame = null;
    const scheduleLayout = () => {
        if (layoutFrame) {
            return;
        }
        layoutFrame = requestAnimationFrame(() => {
            layoutFrame = null;
            updateLayout();
        });
    };
    const ensureLayoutReady = () => {
        let attempts = 0;
        const tick = () => {
            attempts += 1;
            scheduleLayout();
            const ready =
                container.isConnected &&
                grid.clientWidth > 0 &&
                grid.clientHeight > 0 &&
                (node.size?.[0] || 0) > 0 &&
                (node.size?.[1] || 0) > 0;
            if (!ready && attempts < 60) {
                requestAnimationFrame(tick);
            }
        };
        requestAnimationFrame(tick);
    };

    const syncSelection = () => {
        if (styleWidget?.value) {
            state.selectedId = styleWidget.value;
        } else if (node.properties && node.properties[STYLE_INPUT]) {
            state.selectedId = node.properties[STYLE_INPUT];
        }
    };

    const setSelected = (style) => {
        if (!style || !style.id) {
            return;
        }
        const nextId = style.id === state.selectedId ? "" : style.id;
        state.selectedId = nextId;
        if (styleWidget) {
            styleWidget.value = nextId;
            styleWidget.callback?.(nextId);
        }
        if (node.setProperty) {
            node.setProperty(STYLE_INPUT, nextId);
        } else {
            node.properties ||= {};
            node.properties[STYLE_INPUT] = nextId;
        }
        renderGrid();
        node.setDirtyCanvas(true, true);
    };

    const renderGrid = () => {
        grid.innerHTML = "";
        if (!state.filtered.length) {
            empty.textContent = state.styles.length ? "No styles found." : "No styles available.";
            empty.style.display = "block";
            return;
        }
        empty.style.display = "none";
        state.filtered.forEach((style) => {
            const card = document.createElement("button");
            card.type = "button";
            card.className = "ts-style-card";
            if (style.id === state.selectedId) {
                card.classList.add("is-selected");
            }
            if (style.description) {
                card.title = style.description;
            } else if (style.prompt) {
                card.title = style.prompt;
            }

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
            label.textContent = style.name || style.id || "Style";
            card.appendChild(label);

            const guardResize = (event) => {
                if (isNearResizeEdge(event, container)) {
                    return;
                }
                event.stopPropagation();
            };
            card.addEventListener("pointerdown", guardResize);
            card.addEventListener("mousedown", guardResize);
            card.addEventListener("contextmenu", (event) => {
                event.stopPropagation();
            });
            card.addEventListener("click", (event) => {
                event.preventDefault();
                setSelected(style);
            });

            grid.appendChild(card);
        });
    };

    const applyFilter = () => {
        const query = search.value.trim().toLowerCase();
        if (!query) {
            state.filtered = state.styles.slice();
        } else {
            state.filtered = state.styles.filter((style) => {
                const haystack = [
                    style.id,
                    style.name,
                    style.description,
                    style.prompt,
                ]
                    .filter(Boolean)
                    .join(" ")
                    .toLowerCase();
                return haystack.includes(query);
            });
        }
        renderGrid();
    };

    search.addEventListener("input", () => {
        applyFilter();
    });

    const loadStyles = async () => {
        empty.textContent = "Loading styles...";
        try {
            const response = await fetch(api.apiURL("/ts_styles"));
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const payload = await response.json();
            state.styles = Array.isArray(payload.styles) ? payload.styles : [];
            syncSelection();
            if (!state.selectedId && state.styles.length) {
                setSelected(state.styles[0]);
            } else {
                applyFilter();
            }
            ensureLayoutReady();
        } catch (error) {
            empty.textContent = "Failed to load styles.";
            console.error("[TS Style Prompt Selector] Failed to load styles:", error);
        }
    };

    node._tsStyleSelectorSync = () => {
        syncSelection();
        applyFilter();
        scheduleLayout();
    };

    if (typeof ResizeObserver === "function") {
        const observer = new ResizeObserver(() => scheduleLayout());
        observer.observe(container);
        node._tsStyleSelectorObserver = observer;
    }

    const prevOnResize = node.onResize;
    node.onResize = function () {
        const result = prevOnResize?.apply(this, arguments);
        scheduleLayout();
        return result;
    };

    ensureLayoutReady();

    loadStyles();
    scheduleLayout();
    node._tsStyleEnsureLayout = ensureLayoutReady;
}

app.registerExtension({
    name: EXT_NAME,
    async nodeCreated(node) {
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
        node._tsStyleSelectorSync?.();
        node._tsStyleEnsureLayout?.();
    },
});
