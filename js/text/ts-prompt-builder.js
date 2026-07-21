import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import { TS_UI_CLASS, ensureThemeStyles, pickLocaleStrings } from "../_theme.js";

const TS_PROMPT_BUILDER_EXTENSION_ID = "ts.prompt_builder";
const TS_PROMPT_BUILDER_NODE_NAME = "TS_PromptBuilder";
const TS_PROMPT_BUILDER_CONFIG_INPUT = "config_json";
const TS_PROMPT_BUILDER_STYLE_ID = "ts-prompt-builder-styles";
const TS_PROMPT_BUILDER_NODE_WIDTH = 260;
const TS_PROMPT_BUILDER_NODE_HEIGHT = 340;
const TS_PROMPT_BUILDER_WIDGET_HEIGHT = 260;

const STRINGS = {
    en: {
        hint: "Click to toggle. Drag handle to reorder.",
        loading: "Loading prompt blocks...",
        noFiles: "No prompt files found.",
        loadFailed: "Failed to load prompt files.",
    },
    ru: {
        hint: "Клик — вкл/выкл. Тяните за ручку для порядка.",
        loading: "Загрузка блоков промпта...",
        noFiles: "Файлы промптов не найдены.",
        loadFailed: "Не удалось загрузить файлы промптов.",
    },
};

function tsEnsureStyles() {
    // Colours come from the shared --ts-* tokens (js/_theme.js); keep this
    // stylesheet to layout only.
    ensureThemeStyles();
    if (document.getElementById(TS_PROMPT_BUILDER_STYLE_ID)) {
        return;
    }
    const tsStyle = document.createElement("style");
    tsStyle.id = TS_PROMPT_BUILDER_STYLE_ID;
    tsStyle.textContent = `
.ts-prompt-builder {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 6px;
    box-sizing: border-box;
    height: 100%;
    min-height: 0;
    width: 100%;
    color: var(--ts-text);
    font-family: var(--ts-font);
    pointer-events: auto;
}
.ts-prompt-list {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
    overflow-y: auto;
    padding-right: 2px;
}
.ts-prompt-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 6px;
    border: 1px solid var(--ts-border-soft);
    border-radius: var(--ts-radius-sm);
    background: var(--ts-surface);
    cursor: pointer;
    user-select: none;
    transition: border-color 0.15s ease, background 0.15s ease, opacity 0.15s ease;
}
.ts-prompt-item.is-disabled {
    opacity: 0.45;
}
.ts-prompt-item.is-drop-target {
    border-color: var(--ts-accent);
    box-shadow: 0 0 0 1px var(--ts-accent-line);
}
.ts-prompt-handle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid var(--ts-border-soft);
    background: var(--ts-sunken);
    color: var(--ts-muted);
    font-size: var(--ts-fs-xs);
    line-height: 1;
    cursor: grab;
}
.ts-prompt-label {
    flex: 1 1 auto;
    font-size: var(--ts-fs);
    color: var(--ts-text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ts-prompt-toggle {
    width: 10px;
    height: 10px;
    border-radius: 999px;
    border: 1px solid var(--ts-border);
    background: var(--ts-sunken);
}
.ts-prompt-item.is-enabled .ts-prompt-toggle {
    background: var(--ts-accent);
    border-color: var(--ts-accent);
}
.ts-prompt-hint {
    font-size: var(--ts-fs-xs);
    color: var(--ts-muted);
}
.ts-prompt-empty {
    font-size: var(--ts-fs-sm);
    color: var(--ts-muted);
    padding: 2px 0;
}
`;
    document.head.appendChild(tsStyle);
}

function tsStopPropagation(tsElement, tsEvents) {
    tsEvents.forEach((tsEventName) => {
        tsElement.addEventListener(tsEventName, (tsEvent) => {
            tsEvent.stopPropagation();
        });
    });
}

function tsIsTargetNode(tsNode) {
    return tsNode?.comfyClass === TS_PROMPT_BUILDER_NODE_NAME || tsNode?.type === TS_PROMPT_BUILDER_NODE_NAME;
}

function tsIsNodesV2() {
    if (typeof window === "undefined") {
        return false;
    }
    return Boolean(window.comfyAPI?.domWidget?.DOMWidgetImpl);
}

function tsHideConfigWidget(tsNode) {
    const tsWidget = tsNode?.widgets?.find((tsItem) => tsItem.name === TS_PROMPT_BUILDER_CONFIG_INPUT);
    if (tsWidget) {
        tsWidget.hidden = true;
        tsWidget.type = "hidden";
        tsWidget.serialize = true;
        tsWidget.options = { ...(tsWidget.options || {}), hidden: true, serialize: true };
        tsWidget.computeSize = () => [0, -4];
    }
    const tsInput = tsNode?.inputs?.find((tsItem) => tsItem?.name === TS_PROMPT_BUILDER_CONFIG_INPUT);
    if (tsInput) {
        tsInput.hidden = true;
    }
}

function tsMakeLabel(tsFileName) {
    return String(tsFileName || "").replace(/\.txt$/i, "");
}

function tsBuildItems(tsBlocks, tsFiles) {
    const tsItems = [];
    const tsSeen = new Set();
    if (Array.isArray(tsBlocks)) {
        tsBlocks.forEach((tsEntry) => {
            const tsName = tsEntry?.file || tsEntry?.name;
            if (!tsName || tsSeen.has(tsName)) {
                return;
            }
            tsSeen.add(tsName);
            tsItems.push({
                name: tsName,
                label: tsMakeLabel(tsName),
                enabled: tsEntry?.enabled !== false,
            });
        });
    }
    if (Array.isArray(tsFiles)) {
        tsFiles.forEach((tsName) => {
            if (!tsName || tsSeen.has(tsName)) {
                return;
            }
            tsSeen.add(tsName);
            tsItems.push({
                name: tsName,
                label: tsMakeLabel(tsName),
                enabled: true,
            });
        });
    }
    return tsItems;
}

function tsSetupPromptBuilder(tsNode) {
    if (!tsNode || tsNode._tsPromptBuilderInitialized) {
        return;
    }
    tsNode._tsPromptBuilderInitialized = true;

    if (typeof tsNode.addDOMWidget !== "function") {
        return;
    }

    const L = pickLocaleStrings(STRINGS);
    tsEnsureStyles();
    tsHideConfigWidget(tsNode);

    tsNode.resizable = false;
    tsNode.size = [TS_PROMPT_BUILDER_NODE_WIDTH, TS_PROMPT_BUILDER_NODE_HEIGHT];
    tsNode.min_size = [TS_PROMPT_BUILDER_NODE_WIDTH, TS_PROMPT_BUILDER_NODE_HEIGHT];
    tsNode.max_size = [TS_PROMPT_BUILDER_NODE_WIDTH, TS_PROMPT_BUILDER_NODE_HEIGHT];

    const tsContainer = document.createElement("div");
    tsContainer.className = `${TS_UI_CLASS} ts-prompt-builder`;

    const tsList = document.createElement("div");
    tsList.className = "ts-prompt-list";

    const tsHint = document.createElement("div");
    tsHint.className = "ts-prompt-hint";
    tsHint.textContent = L.hint;

    const tsEmpty = document.createElement("div");
    tsEmpty.className = "ts-prompt-empty";
    tsEmpty.textContent = L.loading;

    tsContainer.appendChild(tsList);
    tsContainer.appendChild(tsHint);
    tsContainer.appendChild(tsEmpty);

    const tsIsV2 = tsIsNodesV2();
    const tsWidgetOptions = {
        serialize: false,
        hideOnZoom: true,
    };
    if (tsIsV2) {
        tsWidgetOptions.getMinHeight = () => TS_PROMPT_BUILDER_WIDGET_HEIGHT;
        tsWidgetOptions.getMaxHeight = () => TS_PROMPT_BUILDER_WIDGET_HEIGHT;
    }

    const tsDomWidget = tsNode.addDOMWidget("ts_prompt_builder", "div", tsContainer, tsWidgetOptions);
    const tsDomWidgetEl = tsDomWidget?.element || tsDomWidget?.el || tsDomWidget?.container;
    if (tsDomWidgetEl) {
        tsDomWidgetEl.style.overflow = "hidden";
        tsDomWidgetEl.style.height = `${TS_PROMPT_BUILDER_WIDGET_HEIGHT}px`;
        tsDomWidgetEl.style.minHeight = `${TS_PROMPT_BUILDER_WIDGET_HEIGHT}px`;
        tsDomWidgetEl.style.maxHeight = `${TS_PROMPT_BUILDER_WIDGET_HEIGHT}px`;
    }
    tsContainer.style.height = `${TS_PROMPT_BUILDER_WIDGET_HEIGHT}px`;
    tsContainer.style.minHeight = `${TS_PROMPT_BUILDER_WIDGET_HEIGHT}px`;
    tsContainer.style.maxHeight = `${TS_PROMPT_BUILDER_WIDGET_HEIGHT}px`;

    // V1 (LiteGraph) only: fixed-size legacy layout. In V2 the Vue renderer
    // sizes DOM widgets through getMinHeight/getMaxHeight (set above) — an
    // assigned computeSize would push the widget into the legacy fixed branch
    // of computeLayoutSize() (see CLAUDE.md §12.5.1).
    if (!tsIsV2) {
        tsDomWidget.computeSize = function () {
            return [TS_PROMPT_BUILDER_NODE_WIDTH, TS_PROMPT_BUILDER_WIDGET_HEIGHT];
        };
    }

    const tsConfigWidget = tsNode.widgets?.find((tsItem) => tsItem.name === TS_PROMPT_BUILDER_CONFIG_INPUT);
    const tsState = {
        items: [],
        loading: true,
        dragIndex: null,
    };

    const tsSerializeItems = (tsItems) =>
        JSON.stringify(
            tsItems.map((tsItem) => ({
                file: tsItem.name,
                enabled: Boolean(tsItem.enabled),
            })),
        );

    const tsSyncConfig = () => {
        const tsJson = tsSerializeItems(tsState.items);
        if (tsConfigWidget) {
            tsConfigWidget.value = tsJson;
            tsConfigWidget.callback?.(tsJson);
        }
        if (tsNode.setProperty) {
            tsNode.setProperty(TS_PROMPT_BUILDER_CONFIG_INPUT, tsJson);
        } else {
            tsNode.properties ||= {};
            tsNode.properties[TS_PROMPT_BUILDER_CONFIG_INPUT] = tsJson;
        }
        tsNode.setDirtyCanvas(true, true);
    };

    const tsPersistConfig = async () => {
        try {
            const tsPayload = {
                blocks: tsState.items.map((tsItem) => ({
                    file: tsItem.name,
                    enabled: Boolean(tsItem.enabled),
                })),
            };
            const tsResponse = await fetch(api.apiURL("/ts_prompt_builder/config"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(tsPayload),
            });
            if (!tsResponse.ok) {
                throw new Error(`HTTP ${tsResponse.status}`);
            }
            const tsData = await tsResponse.json();
            const tsBlocks = tsBuildItems(tsData.blocks, tsData.files);
            if (tsBlocks.length) {
                tsState.items = tsBlocks;
                tsRenderList();
                tsSyncConfig();
            }
        } catch (tsError) {
            console.error("[TS Prompt Builder] Failed to save config:", tsError);
        }
    };

    const tsMoveItem = (tsFromIndex, tsToIndex) => {
        if (tsFromIndex === tsToIndex) {
            return;
        }
        const tsItems = tsState.items.slice();
        const [tsMoved] = tsItems.splice(tsFromIndex, 1);
        tsItems.splice(tsToIndex, 0, tsMoved);
        tsState.items = tsItems;
        tsRenderList();
        tsSyncConfig();
        tsPersistConfig();
    };

    const tsToggleItem = (tsIndex) => {
        const tsItems = tsState.items.slice();
        const tsItem = tsItems[tsIndex];
        if (!tsItem) {
            return;
        }
        tsItem.enabled = !tsItem.enabled;
        tsState.items = tsItems;
        tsRenderList();
        tsSyncConfig();
        tsPersistConfig();
    };

    const tsRenderList = () => {
        tsList.innerHTML = "";

        if (tsState.loading) {
            tsEmpty.textContent = L.loading;
            tsEmpty.style.display = "block";
            return;
        }

        if (!tsState.items.length) {
            tsEmpty.textContent = L.noFiles;
            tsEmpty.style.display = "block";
            return;
        }

        tsEmpty.style.display = "none";

        tsState.items.forEach((tsItem, tsIndex) => {
            const tsRow = document.createElement("div");
            tsRow.className = "ts-prompt-item";
            tsRow.classList.toggle("is-disabled", !tsItem.enabled);
            tsRow.classList.toggle("is-enabled", Boolean(tsItem.enabled));

            const tsHandle = document.createElement("div");
            tsHandle.className = "ts-prompt-handle";
            tsHandle.textContent = "::";
            tsHandle.setAttribute("draggable", "true");

            const tsLabel = document.createElement("div");
            tsLabel.className = "ts-prompt-label";
            tsLabel.textContent = tsItem.label;

            const tsToggle = document.createElement("div");
            tsToggle.className = "ts-prompt-toggle";

            tsRow.appendChild(tsHandle);
            tsRow.appendChild(tsLabel);
            tsRow.appendChild(tsToggle);

            tsRow.addEventListener("click", (tsEvent) => {
                if (tsEvent.target?.closest(".ts-prompt-handle")) {
                    return;
                }
                tsToggleItem(tsIndex);
            });

            tsHandle.addEventListener("dragstart", (tsEvent) => {
                tsState.dragIndex = tsIndex;
                tsEvent.dataTransfer.effectAllowed = "move";
                tsEvent.dataTransfer.setData("text/plain", String(tsIndex));
                tsRow.classList.add("is-dragging");
            });

            tsHandle.addEventListener("dragend", () => {
                tsState.dragIndex = null;
                tsRow.classList.remove("is-dragging");
                tsList.querySelectorAll(".ts-prompt-item.is-drop-target").forEach((tsEl) => {
                    tsEl.classList.remove("is-drop-target");
                });
            });

            tsRow.addEventListener("dragover", (tsEvent) => {
                tsEvent.preventDefault();
                tsRow.classList.add("is-drop-target");
            });

            tsRow.addEventListener("dragleave", () => {
                tsRow.classList.remove("is-drop-target");
            });

            tsRow.addEventListener("drop", (tsEvent) => {
                tsEvent.preventDefault();
                tsRow.classList.remove("is-drop-target");
                const tsFromIndex = Number(tsEvent.dataTransfer.getData("text/plain"));
                const tsResolvedFrom = Number.isFinite(tsFromIndex) ? tsFromIndex : tsState.dragIndex;
                if (Number.isFinite(tsResolvedFrom)) {
                    tsMoveItem(tsResolvedFrom, tsIndex);
                }
            });

            tsStopPropagation(tsRow, [
                "pointerdown",
                "pointerup",
                "mousedown",
                "mouseup",
                "dblclick",
                "contextmenu",
            ]);
            tsStopPropagation(tsHandle, [
                "pointerdown",
                "mousedown",
                "mouseup",
                "dblclick",
                "contextmenu",
            ]);

            tsList.appendChild(tsRow);
        });
    };

    const tsLoadState = async () => {
        tsState.loading = true;
        tsRenderList();
        try {
            const tsResponse = await fetch(api.apiURL("/ts_prompt_builder/state"));
            if (!tsResponse.ok) {
                throw new Error(`HTTP ${tsResponse.status}`);
            }
            const tsPayload = await tsResponse.json();
            tsState.items = tsBuildItems(tsPayload.blocks, tsPayload.files);
            tsState.loading = false;
            tsRenderList();
            tsSyncConfig();
        } catch (tsError) {
            tsState.loading = false;
            tsState.items = [];
            tsRenderList();
            tsEmpty.textContent = L.loadFailed;
            tsEmpty.style.display = "block";
            console.error("[TS Prompt Builder] Failed to load prompt files:", tsError);
        }
    };

    tsStopPropagation(tsContainer, [
        "pointerdown",
        "pointerup",
        "mousedown",
        "mouseup",
        "wheel",
        "dblclick",
        "contextmenu",
    ]);
    tsStopPropagation(tsList, ["wheel"]);

    tsNode._tsPromptBuilderSync = () => {
        tsLoadState();
    };

    tsRenderList();
    tsLoadState();
}

app.registerExtension({
    name: TS_PROMPT_BUILDER_EXTENSION_ID,
    nodeCreated(tsNode) {
        if (!tsIsTargetNode(tsNode)) {
            return;
        }
        tsSetupPromptBuilder(tsNode);
    },
    loadedGraphNode(tsNode) {
        if (!tsIsTargetNode(tsNode)) {
            return;
        }
        if (!tsNode._tsPromptBuilderInitialized) {
            tsSetupPromptBuilder(tsNode);
        }
        tsNode.resizable = false;
        tsNode.size = [TS_PROMPT_BUILDER_NODE_WIDTH, TS_PROMPT_BUILDER_NODE_HEIGHT];
        tsNode.min_size = [TS_PROMPT_BUILDER_NODE_WIDTH, TS_PROMPT_BUILDER_NODE_HEIGHT];
        tsNode.max_size = [TS_PROMPT_BUILDER_NODE_WIDTH, TS_PROMPT_BUILDER_NODE_HEIGHT];
        tsHideConfigWidget(tsNode);
        tsNode._tsPromptBuilderSync?.();
    },
});
