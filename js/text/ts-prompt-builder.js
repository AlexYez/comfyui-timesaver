import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { TS_UI_CLASS, ensureThemeStyles, pickLocaleStrings, createOpenInterfaceButton } from "../_theme.js";
import { addResizableDomWidget, hideWidget, getWidget } from "../_dom_widget.js";
import { openFullscreenOverlay } from "../_fullscreen.js";
import { parseConfig as tsParseConfig, buildItems as tsBuildItems, serializeItems, moveItem } from "./_prompt_builder_state.js";

const NODE = "TS_PromptBuilder", CONFIG = "config_json";
const DRAG_TYPE = "application/x-ts-prompt-block";
const STRINGS = {
    en: {
        title: "Prompt Builder", close: "Close (Esc)", fullscreen: "Arrange prompt blocks in full screen",
        hint: "Drag a handle to reorder. Keyboard: Alt + arrows on a block toggle.",
        loading: "Loading prompt blocks…", noFiles: "No blocks yet. Add your own text.",
        loadFailed: "Could not reload files. Your blocks are kept.", saveFailed: "Could not save default file order. The workflow keeps your edits.",
        addCustom: "+ Your text", customText: "Your text", placeholder: "Write a prompt fragment…",
        remove: "Remove text block", reload: "Reload", reloadTitle: "Reload wildcard files", move: "Drag to reorder",
        enable: "Enable block", disable: "Disable block", blocks: "Enabled blocks",
    },
    ru: {
        title: "Конструктор промпта", close: "Закрыть (Esc)", fullscreen: "Разложить блоки промпта на весь экран",
        hint: "Тяните за ручку для порядка. С клавиатуры: Alt + стрелки на переключателе блока.",
        loading: "Загрузка блоков…", noFiles: "Блоков пока нет. Добавьте свой текст.",
        loadFailed: "Не удалось обновить файлы. Ваши блоки сохранены.", saveFailed: "Не удалось сохранить порядок файлов по умолчанию. Правки остаются в workflow.",
        addCustom: "+ Свой текст", customText: "Свой текст", placeholder: "Впишите фрагмент промпта…",
        remove: "Удалить текстовый блок", reload: "Обновить", reloadTitle: "Обновить файлы wildcard", move: "Тяните для перестановки · Alt + стрелки",
        enable: "Включить блок", disable: "Выключить блок", blocks: "Включено блоков",
    },
};

function styles() {
    ensureThemeStyles();
    if (document.getElementById("ts-prompt-builder-styles")) return;
    const sheet = document.createElement("style");
    sheet.id = "ts-prompt-builder-styles";
    sheet.textContent = `
.ts-prompt-builder-shell{width:100%;height:100%;min-height:0;display:flex}
.ts-prompt-builder{display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;
    height:100%;min-height:0;width:100%;color:var(--ts-text);font-family:var(--ts-font);pointer-events:auto}
.ts-prompt-toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:5px;flex:0 0 auto}
.ts-prompt-toolbar .ts-ui-btn{font-size:var(--ts-fs-sm)}
.ts-prompt-toolbar .ts-ui-launch{margin-left:auto}
.ts-prompt-body{position:relative;flex:1 1 0;min-height:0}
.ts-prompt-list{position:absolute;inset:0;display:flex;flex-direction:column;gap:4px;overflow:auto;padding:2px}
.ts-prompt-item{display:flex;flex-direction:column;gap:6px;padding:6px;border:1px solid var(--ts-border-soft);
    border-radius:var(--ts-radius-sm);background:var(--ts-surface);cursor:pointer;user-select:none;flex:0 0 auto}
.ts-prompt-item.is-disabled{opacity:.5}
.ts-prompt-item.is-drop-target{border-color:var(--ts-accent);box-shadow:0 0 0 1px var(--ts-accent-line)}
.ts-prompt-item.is-dragging{opacity:.5}
.ts-prompt-item-head{display:flex;gap:6px;align-items:center;min-width:0}
.ts-prompt-handle{display:inline-flex;align-items:center;justify-content:center;flex:0 0 22px;
    width:22px;height:24px;border:1px solid var(--ts-border-soft);border-radius:4px;
    background:var(--ts-sunken);color:var(--ts-muted);cursor:grab;font-size:var(--ts-fs-sm)}
.ts-prompt-handle:active{cursor:grabbing}
.ts-prompt-label{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:var(--ts-fs)}
.ts-prompt-number{font-size:var(--ts-fs-xs);color:var(--ts-muted)}
.ts-prompt-toggle{width:18px;height:18px;flex:0 0 18px;padding:0;border-radius:50%;
    border:1px solid var(--ts-border);background:var(--ts-sunken);cursor:pointer}
.ts-prompt-item.is-enabled .ts-prompt-toggle{background:var(--ts-accent);border-color:var(--ts-accent)}
.ts-prompt-remove{flex:0 0 auto}
.ts-prompt-custom{box-sizing:border-box;width:100%;min-height:86px;resize:vertical;padding:7px;
    border:1px solid var(--ts-border-soft);border-radius:var(--ts-radius-sm);background:var(--ts-sunken);
    color:var(--ts-text);font:inherit;font-size:var(--ts-fs);line-height:1.5;cursor:text;user-select:text}
.ts-prompt-hint{font-size:var(--ts-fs-xs);color:var(--ts-muted);flex:0 0 auto}
.ts-prompt-empty{font-size:var(--ts-fs-sm);color:var(--ts-muted);padding:8px}
.ts-prompt-counter{font-size:var(--ts-fs-sm);color:var(--ts-muted);margin-left:auto;white-space:nowrap}
.ts-prompt-builder.is-fullscreen{padding:12px 18px;gap:10px;font-size:var(--ts-fs)}
.ts-prompt-builder.is-fullscreen .ts-prompt-list{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));
    gap:10px;align-content:start;align-items:start;padding:4px 6px 12px}
.ts-prompt-builder.is-fullscreen .ts-prompt-item{min-height:84px;padding:12px;justify-content:center}
.ts-prompt-builder.is-fullscreen .ts-prompt-label{white-space:normal;overflow-wrap:anywhere;line-height:1.4;font-size:var(--ts-fs-lg)}
.ts-prompt-builder.is-fullscreen .is-custom{grid-column:span 2;justify-content:flex-start}
.ts-prompt-builder.is-fullscreen .ts-prompt-custom{min-height:112px}
@media(max-width:520px){.ts-prompt-builder.is-fullscreen .is-custom{grid-column:span 1}}
`;
    document.head.appendChild(sheet);
}

function stopEvents(element, events) {
    for (const name of events) element.addEventListener(name, event => event.stopPropagation());
}

function setup(node) {
    if (node._tsPromptBuilderInitialized || typeof node.addDOMWidget !== "function") return;
    node._tsPromptBuilderInitialized = true;
    styles();
    hideWidget(node, CONFIG);
    const L = pickLocaleStrings(STRINGS);
    const state = { items: [], loading: true, removed: false, fullscreen: null, dragName: null, loadId: 0 };
    const dragOwner = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
    const shell = document.createElement("div");
    shell.className = "ts-prompt-builder-shell";
    const root = document.createElement("div");
    root.className = `${TS_UI_CLASS} ts-prompt-builder`;
    const bar = document.createElement("div"); bar.className = "ts-prompt-toolbar";
    const body = document.createElement("div"); body.className = "ts-prompt-body";
    const list = document.createElement("div"); list.className = "ts-prompt-list";
    const empty = document.createElement("div"); empty.className = "ts-prompt-empty";
    const hint = document.createElement("div"); hint.className = "ts-prompt-hint"; hint.textContent = L.hint;
    const counter = document.createElement("span"); counter.className = "ts-prompt-counter"; counter.title = L.blocks;
    body.append(list, empty); root.append(bar, body, hint); shell.append(root);
    node.size = [300, 380];
    addResizableDomWidget(node, shell, { name: "ts_prompt_builder", minWidth: 260,
        minHeight: 240, defaultWidth: 300, defaultHeight: 380, chromeHeight: 60, minWidgetHeight: 160 });

    function sync() {
        const value = serializeItems(state.items);
        const widget = getWidget(node, CONFIG);
        if (widget) { widget.value = value; widget.callback?.(value); }
        if (node.setProperty) node.setProperty(CONFIG, value);
        else { node.properties ||= {}; node.properties[CONFIG] = value; }
        counter.textContent = `${state.items.filter(item => item.enabled).length} / ${state.items.length}`;
        node.setDirtyCanvas?.(true, true);
    }

    // Shared defaults contain file choices only. HTTP responses never replace
    // newer local edits, and rapid default writes run in order.
    let pendingDefaults = null, saveLoop = null, lastQueued = null;
    function persistDefaults() {
        const payload = serializeItems(state.items, { filesOnly: true });
        if (payload === lastQueued || state.removed) return;
        lastQueued = payload; pendingDefaults = payload;
        if (saveLoop) return;
        saveLoop = (async () => {
            while (pendingDefaults !== null && !state.removed) {
                const next = pendingDefaults; pendingDefaults = null;
                try {
                    const response = await fetch(api.apiURL("/ts_prompt_builder/config"), {
                        method: "POST", headers: { "Content-Type": "application/json" }, body: next,
                    });
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                } catch (error) {
                    lastQueued = null;
                    if (!state.removed) hint.textContent = L.saveFailed;
                    console.warn("[TS Prompt Builder] Default save failed", error);
                }
            }
        })().finally(() => { saveLoop = null; });
    }

    function edit(change, { renderList = true, persist = true } = {}) {
        node.graph?.beforeChange?.(); change();
        if (renderList) render();
        sync(); node.graph?.afterChange?.();
        if (persist) persistDefaults();
    }

    function move(from, to) {
        const next = moveItem(state.items, from, to);
        if (next !== state.items) {
            const changedFiles = serializeItems(next, { filesOnly: true }) !== serializeItems(state.items, { filesOnly: true });
            edit(() => { state.items = next; }, { persist: changedFiles });
        }
    }

    function button(label, className = "") {
        const result = document.createElement("button"); result.type = "button";
        result.className = `ts-ui-btn ${className}`; result.title = label; result.setAttribute("aria-label", label);
        result.textContent = label;
        return result;
    }

    function render() {
        list.replaceChildren();
        empty.hidden = Boolean(state.items.length);
        empty.textContent = state.loading ? L.loading : L.noFiles;
        if (state.loading && !state.items.length) return;
        for (const [index, item] of state.items.entries()) {
            const row = document.createElement("div"); row.className = "ts-prompt-item";
            row.classList.toggle("is-custom", item.type === "custom");
            row.classList.toggle("is-disabled", !item.enabled); row.classList.toggle("is-enabled", item.enabled);
            row.dataset.blockId = `${item.type === "custom" ? "custom" : "file"}:${item.name}`;
            const head = document.createElement("div"); head.className = "ts-prompt-item-head";
            // A focusable mouse drag handle is re-focused by the shared modal
            // before dragstart, cancelling the native drag. Keep the grip
            // non-focusable; keyboard reorder lives on the block toggle.
            const handle = document.createElement("div"); handle.className = "ts-prompt-handle";
            handle.title = L.move; handle.textContent = "::"; handle.draggable = true;
            const label = document.createElement("span"); label.className = "ts-prompt-label";
            label.textContent = item.type === "custom" ? L.customText : item.label; label.title = label.textContent;
            const number = document.createElement("span"); number.className = "ts-prompt-number"; number.textContent = index + 1;
            const toggle = button(item.enabled ? L.disable : L.enable, "ts-prompt-toggle");
            toggle.textContent = "";
            toggle.setAttribute("aria-pressed", String(item.enabled));
            toggle.addEventListener("click", () => edit(() => { item.enabled = !item.enabled; }, { persist: item.type !== "custom" }));
            head.append(handle, label, number, toggle); row.append(head);
            if (item.type === "custom") {
                const remove = button(L.remove, "ts-ui-btn--icon ts-prompt-remove"); remove.textContent = "×";
                remove.addEventListener("click", () => edit(() => { state.items = state.items.filter(entry => entry !== item); }, { persist: false }));
                head.append(remove);
                const input = document.createElement("textarea"); input.className = "ts-prompt-custom";
                input.placeholder = L.placeholder; input.setAttribute("aria-label", L.customText); input.value = item.text;
                input.addEventListener("input", () => edit(() => { item.text = input.value; }, { renderList: false, persist: false }));
                stopEvents(input, ["keydown", "keyup", "copy", "cut", "paste"]);
                row.append(input);
            }
            row.addEventListener("click", event => {
                if (!event.target.closest("button,textarea,.ts-prompt-handle")) edit(() => { item.enabled = !item.enabled; }, { persist: item.type !== "custom" });
            });
            toggle.addEventListener("keydown", event => {
                if (!event.altKey || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
                event.preventDefault(); event.stopPropagation();
                move(index, index + (["ArrowLeft", "ArrowUp"].includes(event.key) ? -1 : 1));
                [...list.children].find(el => el.dataset.blockId === row.dataset.blockId)?.querySelector(".ts-prompt-toggle")?.focus();
            });
            handle.addEventListener("dragstart", event => {
                state.dragName = row.dataset.blockId;
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData(DRAG_TYPE, JSON.stringify({ owner: dragOwner, id: state.dragName }));
                row.classList.add("is-dragging");
            });
            handle.addEventListener("dragend", () => {
                state.dragName = null; row.classList.remove("is-dragging");
                list.querySelectorAll(".is-drop-target").forEach(el => el.classList.remove("is-drop-target"));
            });
            row.addEventListener("dragover", event => {
                if (!state.dragName || !event.dataTransfer.types.includes(DRAG_TYPE)) return;
                event.preventDefault(); event.stopPropagation(); event.dataTransfer.dropEffect = "move";
                row.classList.add("is-drop-target");
            });
            row.addEventListener("dragleave", event => {
                if (!row.contains(event.relatedTarget)) row.classList.remove("is-drop-target");
            });
            row.addEventListener("drop", event => {
                row.classList.remove("is-drop-target");
                if (!state.dragName || !event.dataTransfer.types.includes(DRAG_TYPE)) return;
                event.preventDefault(); event.stopPropagation();
                try {
                    const data = JSON.parse(event.dataTransfer.getData(DRAG_TYPE));
                    if (data.owner !== dragOwner || data.id !== state.dragName) return;
                    const from = [...list.children].findIndex(el => el.dataset.blockId === data.id);
                    move(from, index);
                } catch { /* External drags never rearrange the workflow. */ }
                state.dragName = null;
            });
            list.append(row);
        }
    }

    async function load() {
        const loadId = ++state.loadId;
        state.loading = true; render();
        try {
            const response = await fetch(api.apiURL("/ts_prompt_builder/state"));
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const tsPayload = await response.json();
            if (state.removed || loadId !== state.loadId) return;
            // Read after async loading, when configure/undo has restored state.
            const value = getWidget(node, CONFIG)?.value;
            const tsRaw = value || node.properties?.[CONFIG];
            const tsSaved = tsParseConfig(tsRaw);
            state.items = tsBuildItems(tsPayload.blocks, tsPayload.files, tsSaved);
            state.loading = false; hint.textContent = L.hint; render(); sync();
        } catch (error) {
            if (state.removed || loadId !== state.loadId) return;
            state.loading = false; render(); hint.textContent = L.loadFailed;
            console.warn("[TS Prompt Builder] File reload failed", error);
        }
    }

    const add = button(L.addCustom, "ts-prompt-add");
    add.addEventListener("click", () => {
        const name = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
        edit(() => state.items.push({ type: "custom", name, text: "", enabled: true }), { persist: false });
        const inputs = list.querySelectorAll("textarea"); inputs.item(inputs.length - 1)?.focus();
    });
    const reload = button(L.reload, "ts-prompt-reload");
    reload.title = L.reloadTitle; reload.setAttribute("aria-label", L.reloadTitle); reload.addEventListener("click", load);
    const open = createOpenInterfaceButton(() => {
        if (state.fullscreen?.isOpen()) return;
        root.classList.add("is-fullscreen");
        state.fullscreen = openFullscreenOverlay(root, { label: L.title, closeTitle: L.close, trigger: open,
            onClose: () => {
                root.classList.remove("is-fullscreen"); state.fullscreen = null;
                if (!state.removed) shell.append(root);
                node.setDirtyCanvas?.(true, true);
            } });
    }, { description: L.fullscreen });
    open.classList.add("ts-prompt-fullscreen");
    bar.append(add, reload, counter, open);
    stopEvents(root, ["pointerdown", "pointerup", "mousedown", "mouseup", "wheel", "dblclick", "contextmenu"]);
    node._tsPromptBuilderSync = load;
    const previousRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
        state.removed = true; state.loadId++; pendingDefaults = null; state.fullscreen?.close();
        return previousRemoved?.apply(this, args);
    };
    render(); load();
}

app.registerExtension({
    name: "ts.prompt_builder",
    nodeCreated(node) { if (node.comfyClass === NODE || node.type === NODE) setup(node); },
    loadedGraphNode(node) {
        if (node.comfyClass !== NODE && node.type !== NODE) return;
        setup(node); hideWidget(node, CONFIG); node._tsPromptBuilderSync?.();
    },
});
