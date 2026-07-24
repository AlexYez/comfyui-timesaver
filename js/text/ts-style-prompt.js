// TS Style Prompt Selector — searchable, categorised grid of style thumbnails.
//
// Layout is pure CSS (flex column + auto-fill grid + aspect-ratio cards).
// Deliberately NO JavaScript geometry: the previous implementation measured
// getBoundingClientRect() (viewport pixels, post-zoom-transform) and wrote the
// result into style.height (local pixels, pre-transform), so at any graph zoom
// other than 1 the grid height came out wrong — the classic coordinate-space
// pitfall from CLAUDE.md §12.5.3. CSS flex sizing is immune to the transform.
//
// The cards are built ONCE per library load; search and the category filter
// only toggle the `hidden` attribute, so thumbnails are never re-fetched and
// the browse position is restored when the filter is cleared.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import { TS_UI_CLASS, ensureThemeStyles, getUiLanguage, pickLocaleStrings, createOpenInterfaceButton } from "../_theme.js";
import { hideWidget as sharedHideWidget, getWidget as sharedGetWidget } from "../_dom_widget.js";

const EXTENSION_ID = "ts_suite.style_prompt_selector";
const NODE_NAME = "TS_StylePromptSelector";
const STYLE_INPUT = "style_id";
const STYLE_CSS_ID = "ts-style-selector-styles";
const DOM_WIDGET_NAME = "ts_style_selector";

const DEFAULT_NODE_WIDTH = 250;
const DEFAULT_NODE_HEIGHT = 340;
const MIN_NODE_WIDTH = 240;
const MIN_NODE_HEIGHT = 280;
const MIN_WIDGET_HEIGHT = 180;
// Legacy (pre-DOMWidgetImpl frontends) only: node title bar + the search and
// category rows that live inside the widget's own chrome.
const WIDGET_CHROME_HEIGHT = 56;
const ALL_CATEGORIES = "__all__";
const SEARCH_DEBOUNCE_MS = 100;

const STRINGS = {
    en: {
        searchPlaceholder: "Search styles...",
        allCategories: "All categories",
        loading: "Loading styles...",
        noStyles: "No styles found.",
        loadFailed: "Failed to load styles.",
        modalTitle: "Style Library",
        modalSearch: "Search by name, category or prompt...",
        selectedNone: "No style selected",
        selected: (name) => `Selected: ${name}`,
        clear: "Clear",
        done: "Done",
        closeTitle: "Close (Esc)",
        collapseHint: "Click to shrink",
    },
    ru: {
        searchPlaceholder: "Поиск стилей...",
        allCategories: "Все категории",
        loading: "Загрузка стилей...",
        noStyles: "Стили не найдены.",
        loadFailed: "Не удалось загрузить стили.",
        modalTitle: "Библиотека стилей",
        modalSearch: "Поиск по названию, категории или промпту...",
        selectedNone: "Стиль не выбран",
        selected: (name) => `Выбрано: ${name}`,
        clear: "Сбросить",
        done: "Готово",
        closeTitle: "Закрыть (Esc)",
        collapseHint: "Клик — свернуть",
    },
};

function ensureStyles() {
    // Colours come from the shared --ts-* tokens (js/_theme.js); keep this
    // stylesheet to layout only.
    ensureThemeStyles();
    if (document.getElementById(STYLE_CSS_ID)) {
        return;
    }
    const style = document.createElement("style");
    style.id = STYLE_CSS_ID;
    style.textContent = `
.ts-style-selector {
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding: 6px;
    box-sizing: border-box;
    contain: layout paint;
    overflow: hidden;
    height: 100%;
    min-height: 0;
    width: 100%;
    color: var(--ts-text);
    font-family: var(--ts-font);
    pointer-events: auto;
}
.ts-style-search {
    flex: 0 0 auto;
    width: 100%;
    box-sizing: border-box;
    padding: 4px 6px;
    background: var(--ts-sunken);
    border: 1px solid var(--ts-border-soft);
    border-radius: var(--ts-radius-sm);
    color: var(--ts-text);
    outline: none;
    font-size: var(--ts-fs-sm);
}
.ts-style-search:focus {
    border-color: var(--ts-accent-line);
}
.ts-style-search::placeholder {
    color: var(--ts-faint);
}
.ts-style-cat {
    flex: 0 0 auto;
    width: 100%;
    padding: 3px 6px;
    font-size: var(--ts-fs-sm);
}
.ts-style-body {
    /* The scroll host is absolutely positioned INSIDE this box (same trick as
       the Lama canvas): otherwise the sections' natural height (~3200px for
       113 styles) is what the V2 layout measures, the node grows to fit it,
       and a node-derived ceiling then feeds that growth back into itself. */
    position: relative;
    flex: 1 1 0;
    min-height: 0;
}
.ts-style-scroll {
    position: absolute;
    inset: 0;
    overflow-y: auto;
    overflow-x: hidden;
    /* Reserve the gutter so the scrollbar never overlays the last column and
       the columns don't shift when content stops overflowing. */
    scrollbar-gutter: stable;
    padding-right: 2px;
    box-sizing: border-box;
}
.ts-style-section[hidden] {
    display: none;
}
.ts-style-header {
    /* Sticky lives on a plain block inside the scroll host, NOT on a grid
       item: a sticky grid item is clamped to its own grid area, which made
       the headers pile up over the cards. */
    position: sticky;
    top: 0;
    z-index: 2;
    padding: 4px 2px 3px;
    font-size: var(--ts-fs-xs);
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--ts-muted);
    background: var(--ts-bg);
    border-bottom: 1px solid var(--ts-border-soft);
}
.ts-style-header[hidden] {
    display: none;
}
.ts-style-grid {
    display: grid;
    /* Column count adapts to node width; cards keep a square shape via
       aspect-ratio. align-items:start is REQUIRED — the default stretch
       overrides aspect-ratio, collapsing the auto row to ~2px while the card
       still painted at full size and spilled over the next rows. */
    grid-template-columns: repeat(auto-fill, minmax(72px, 1fr));
    align-items: start;
    gap: 4px;
    padding: 4px 0 8px;
    box-sizing: border-box;
}
.ts-style-card {
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    border: 1px solid var(--ts-border-soft);
    border-radius: var(--ts-radius-sm);
    background: var(--ts-surface);
    padding: 0;
    cursor: pointer;
    overflow: hidden;
}
.ts-style-card[hidden] {
    display: none;
}
.ts-style-card img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.ts-style-card.is-selected {
    border-color: var(--ts-accent);
    box-shadow: 0 0 0 1px var(--ts-accent-line);
}
.ts-style-label {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    padding: 3px 4px;
    font-size: 9px;
    text-align: center;
    /* Deliberate hard-coded colours: the plate sits over an arbitrary
       thumbnail and must stay readable in any theme. */
    background: rgba(0, 0, 0, 0.58);
    color: #f2f2f2;
    box-sizing: border-box;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    pointer-events: none;
}
.ts-style-scroll.has-selection .ts-style-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.75);
    pointer-events: none;
}
.ts-style-scroll.has-selection .ts-style-card.is-selected::after {
    background: transparent;
}
.ts-style-empty {
    flex: 0 0 auto;
    font-size: var(--ts-fs-sm);
    color: var(--ts-muted);
    padding: 4px 2px;
}
/* Expanded preview: fills the body (the browsing area) with a big view of the
   selected style; click anywhere on it to collapse back to the grid. Lives
   inside .ts-style-body (position:relative) above the scroll host. */
.ts-style-expand {
    position: absolute;
    inset: 0;
    z-index: 5;
    display: none;
    flex-direction: column;
    background: var(--ts-bg);
    border-radius: var(--ts-radius-sm);
    overflow: hidden;
    cursor: zoom-out;
}
.ts-style-expand.is-active { display: flex; }
.ts-style-expand__img {
    flex: 1 1 auto;
    min-height: 0;
    width: 100%;
    object-fit: contain;
    background: var(--ts-sunken);
    display: block;
}
.ts-style-expand__bar {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 9px;
    border-top: 1px solid var(--ts-border-soft);
    background: var(--ts-elevated);
}
.ts-style-expand__name {
    flex: 1 1 auto;
    min-width: 0;
    font-size: var(--ts-fs-sm);
    font-weight: 600;
    color: var(--ts-text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.ts-style-expand__hint {
    flex: 0 0 auto;
    font-size: var(--ts-fs-xs);
    color: var(--ts-muted);
}
/* Compact top bar in-node: search fills, launcher sits to the right. */
.ts-style-topbar {
    display: flex;
    align-items: center;
    gap: 5px;
    flex: 0 0 auto;
}
.ts-style-topbar .ts-style-search {
    flex: 1 1 auto;
    width: auto;
}
.ts-style-topbar .ts-ui-launch {
    flex: 0 0 auto;
    padding: 6px 9px;
}
/* Compact in-node launcher: icon only (the shared button keeps its "Open
   Interface" tooltip); the full label lives in the fullscreen header. */
.ts-style-topbar .ts-ui-launch span { display: none; }
.ts-style-topbar .ts-ui-launch svg { margin: 0; }
/* ── Fullscreen style library ──────────────────────────────────────────── */
.ts-style-modal { align-items: center; justify-content: center; padding: 24px; }
.ts-style-modal__panel {
    display: flex;
    flex-direction: column;
    width: min(1680px, 96vw);
    height: min(92vh, 100%);
    min-height: 0;
    background: var(--ts-bg);
    border: 1px solid var(--ts-border-soft);
    border-radius: var(--ts-radius-lg);
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.55);
    overflow: hidden;
}
.ts-style-modal__head {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--ts-border-soft);
    background: var(--ts-elevated);
}
.ts-style-modal__title { font-size: var(--ts-fs-lg); font-weight: 600; white-space: nowrap; }
.ts-style-modal__search {
    flex: 1 1 auto;
    min-width: 0;
    max-width: 460px;
    box-sizing: border-box;
    padding: 7px 10px;
    background: var(--ts-sunken);
    border: 1px solid var(--ts-border-soft);
    border-radius: var(--ts-radius-sm);
    color: var(--ts-text);
    outline: none;
    font-size: var(--ts-fs);
}
.ts-style-modal__search:focus { border-color: var(--ts-accent-line); }
.ts-style-modal__cat { flex: 0 0 auto; width: 220px; min-width: 180px; }
.ts-style-modal__selinfo {
    flex: 1 1 auto;
    min-width: 0;
    text-align: right;
    font-size: var(--ts-fs-sm);
    color: var(--ts-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ts-style-modal__scroll {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-gutter: stable;
    padding: 14px 18px 24px;
}
.ts-style-modal__section { margin-bottom: 10px; }
.ts-style-modal__header {
    position: sticky;
    top: 0;
    z-index: 2;
    padding: 8px 2px;
    font-size: var(--ts-fs-sm);
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: var(--ts-muted);
    background: var(--ts-bg);
    border-bottom: 1px solid var(--ts-border-soft);
}
.ts-style-modal__header[hidden] { display: none; }
.ts-style-modal__section[hidden] { display: none; }
.ts-style-modal__grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 14px;
    padding: 12px 0 4px;
}
.ts-style-modal__card {
    display: flex;
    flex-direction: column;
    border: 1px solid var(--ts-border-soft);
    border-radius: var(--ts-radius-md);
    background: var(--ts-surface);
    padding: 0;
    cursor: pointer;
    overflow: hidden;
    text-align: left;
    transition: border-color .12s, box-shadow .12s, transform .08s;
}
.ts-style-modal__card:hover { border-color: var(--ts-accent-line); transform: translateY(-2px); }
.ts-style-modal__card[hidden] { display: none; }
.ts-style-modal__card.is-selected {
    border-color: var(--ts-accent);
    box-shadow: 0 0 0 2px var(--ts-accent-line);
}
.ts-style-modal__thumb {
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    background: var(--ts-sunken);
    overflow: hidden;
}
.ts-style-modal__thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.ts-style-modal__check {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 24px;
    height: 24px;
    border-radius: 999px;
    display: none;
    align-items: center;
    justify-content: center;
    /* Deliberate: badge floats over an arbitrary thumbnail. */
    background: var(--ts-accent);
    color: var(--ts-accent-contrast);
    font-size: 14px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
}
.ts-style-modal__card.is-selected .ts-style-modal__check { display: flex; }
.ts-style-modal__meta { padding: 8px 10px 10px; }
.ts-style-modal__name {
    font-size: var(--ts-fs);
    font-weight: 600;
    color: var(--ts-text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.ts-style-modal__desc {
    margin-top: 3px;
    font-size: var(--ts-fs-xs);
    color: var(--ts-muted);
    line-height: 1.35;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.ts-style-modal__empty {
    padding: 40px;
    text-align: center;
    color: var(--ts-muted);
    font-size: var(--ts-fs);
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
    // Keys off the CLASS (always present in modern builds) on purpose, so the
    // widget always uses getMinHeight/getMaxHeight — correct in BOTH renderers.
    // Do NOT switch to reading Comfy.VueNodes.Enabled: the syncLegacyHeight /
    // computeSize branch it would enable fights node.size and runs the node away.
    return Boolean(window.comfyAPI?.domWidget?.DOMWidgetImpl);
}

// Only a lower bound: the auto-fill grid rewards wide nodes with more columns,
// so the old 520px width cap is gone.
function sanitizeNodeSize(node) {
    const width = Math.max(Number(node?.size?.[0]) || DEFAULT_NODE_WIDTH, MIN_NODE_WIDTH);
    const height = Math.max(Number(node?.size?.[1]) || DEFAULT_NODE_HEIGHT, MIN_NODE_HEIGHT);
    node.size = [width, height];
    node.min_size = [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];
}

function getWidgetHeight(node) {
    const safeHeight = Math.max(Number(node?.size?.[1]) || DEFAULT_NODE_HEIGHT, MIN_NODE_HEIGHT);
    return Math.max(MIN_WIDGET_HEIGHT, safeHeight - WIDGET_CHROME_HEIGHT);
}

function hideStyleWidget(node) {
    sharedHideWidget(node, STYLE_INPUT);
}

function removeDomWidgets(node) {
    if (!Array.isArray(node?.widgets)) {
        return;
    }
    for (let index = node.widgets.length - 1; index >= 0; index -= 1) {
        const widget = node.widgets[index];
        if (widget?.name !== DOM_WIDGET_NAME) {
            continue;
        }
        const element = widget.element || widget.el || widget.container;
        element?.remove?.();
        node.widgets.splice(index, 1);
    }
}

function setupStyleSelector(node) {
    if (!node || typeof node.addDOMWidget !== "function") {
        return;
    }

    if (typeof node._tsStyleSelectorCleanup === "function") {
        node._tsStyleSelectorCleanup();
    }
    removeDomWidgets(node);

    const L = pickLocaleStrings(STRINGS);
    ensureStyles();
    hideStyleWidget(node);
    sanitizeNodeSize(node);

    node.resizable = true;
    node._tsStyleSelectorInitialized = true;

    // hideStyleWidget() above already removed this widget from the grid; keep a
    // reference (from the hidden-widget stash) for reading the persisted value.
    const styleWidget = sharedGetWidget(node, STYLE_INPUT);

    const container = document.createElement("div");
    container.className = `${TS_UI_CLASS} ts-style-selector`;

    const search = document.createElement("input");
    search.type = "search";
    search.className = "ts-style-search";
    search.placeholder = L.searchPlaceholder;

    const categorySelect = document.createElement("select");
    categorySelect.className = "ts-ui-select ts-style-cat";

    const body = document.createElement("div");
    body.className = "ts-style-body";
    const scroll = document.createElement("div");
    scroll.className = "ts-style-scroll";
    body.appendChild(scroll);

    // Big-preview overlay: clicking a card enlarges its thumbnail to fill the
    // body; clicking the enlarged view collapses back and scrolls to the card.
    const expand = document.createElement("div");
    expand.className = "ts-style-expand";
    const expandImg = document.createElement("img");
    expandImg.className = "ts-style-expand__img";
    const expandBar = document.createElement("div");
    expandBar.className = "ts-style-expand__bar";
    const expandName = document.createElement("div");
    expandName.className = "ts-style-expand__name";
    const expandHint = document.createElement("div");
    expandHint.className = "ts-style-expand__hint";
    expandHint.textContent = L.collapseHint;
    expandBar.append(expandName, expandHint);
    expand.append(expandImg, expandBar);
    body.appendChild(expand);

    const empty = document.createElement("div");
    empty.className = "ts-style-empty";
    empty.textContent = L.loading;

    // Launcher opens the big fullscreen library; it shares the same style data
    // and selection so browsing large or in-node stays in sync.
    const launchBtn = createOpenInterfaceButton(() => openStyleLibrary());
    const topbar = document.createElement("div");
    topbar.className = "ts-style-topbar";
    topbar.appendChild(search);
    topbar.appendChild(launchBtn);

    container.appendChild(topbar);
    container.appendChild(categorySelect);
    container.appendChild(body);
    container.appendChild(empty);

    const isV2 = isNodesV2();
    const widgetOptions = {
        serialize: false,
        // Keep the grid visible at any graph zoom — hiding it while zoomed out
        // read as "the node lost its content".
        hideOnZoom: false,
    };
    if (isV2) {
        // Height is distributed by the frontend's own layout between
        // getMinHeight/getMaxHeight (CLAUDE.md §12.5.1); the CSS flex column
        // absorbs whatever it receives, so no getHeight/afterResize hooks.
        // The ceiling MUST follow the node's own height: the sections have a
        // real content height (~3200px for 113 styles), so an effectively
        // unbounded getMaxHeight made the layout grant all of it and the node
        // ballooned instead of scrolling.
        widgetOptions.getMinHeight = () => MIN_WIDGET_HEIGHT;
        widgetOptions.getMaxHeight = () => 8192;
    }

    const domWidget = node.addDOMWidget(DOM_WIDGET_NAME, "div", container, widgetOptions);
    const domWidgetEl = domWidget?.element || domWidget?.el || domWidget?.container;
    if (domWidgetEl) {
        domWidgetEl.style.overflow = "hidden";
        domWidgetEl.style.width = "100%";
    }

    // Pre-DOMWidgetImpl frontends size DOM widgets through computeSize and
    // leave the element height to us. node.size is in graph units — identical
    // to layout pixels regardless of canvas zoom, so this stays correct where
    // rect-based math was not.
    const syncLegacyHeight = () => {
        if (isV2) {
            return;
        }
        const height = getWidgetHeight(node);
        if (domWidgetEl) {
            domWidgetEl.style.height = `${height}px`;
        }
        container.style.height = `${height}px`;
    };
    if (!isV2) {
        domWidget.computeSize = function (width) {
            const safeWidth = Math.max(Number(width) || node.size?.[0] || DEFAULT_NODE_WIDTH, MIN_NODE_WIDTH);
            return [safeWidth, getWidgetHeight(node)];
        };
    }

    const state = {
        styles: [],
        selectedValue: "",
        category: ALL_CATEGORIES,
        loading: true,
    };

    // One record per card, built once per library load. Filtering toggles the
    // `hidden` attribute so scroll position and loaded thumbnails survive.
    const cardRecords = [];
    const sectionRecords = [];
    let searchTimer = null;

    const styleValue = (style) => (style.name || style.id || "").trim();

    // The library ships both an English and a Russian name per style; show the
    // one matching the ComfyUI locale, English being the fallback. Descriptions
    // exist in Russian only, so the English tooltip falls back to the prompt,
    // which is the more useful text there anyway.
    const isRu = getUiLanguage() === "ru";
    const styleLabel = (style) =>
        (isRu ? style.name_ru || style.name : style.name) || style.id || "";
    const categoryLabel = (style) =>
        (isRu ? style.category_ru || style.category : style.category) || "";
    const styleTooltip = (style) => {
        const parts = isRu
            ? [style.description, style.prompt]
            : [style.prompt, style.description];
        return parts.filter(Boolean)[0] || styleLabel(style);
    };

    const matchesSelection = (style, value) => {
        if (!value) {
            return false;
        }
        return value === style.id || value === style.name || value === styleValue(style);
    };

    // Set by openStyleLibrary while the fullscreen modal is mounted; lets
    // setSelection keep the big grid + selection banner in sync with the node.
    let modalRefresh = null;
    let modalRoot = null;

    const setSelection = (value, trigger = true) => {
        const nextValue = value || "";
        const changed = nextValue !== state.selectedValue;
        state.selectedValue = nextValue;
        scroll.classList.toggle("has-selection", Boolean(state.selectedValue));
        cardRecords.forEach((record) => {
            record.el.classList.toggle(
                "is-selected",
                record.value === state.selectedValue || matchesSelection(record.style, state.selectedValue),
            );
        });
        modalRefresh?.();

        if (styleWidget && trigger && changed) {
            styleWidget.value = state.selectedValue;
            styleWidget.callback?.(state.selectedValue);
        }

        if (changed) {
            if (node.setProperty) {
                node.setProperty(STYLE_INPUT, state.selectedValue);
            } else {
                node.properties ||= {};
                node.properties[STYLE_INPUT] = state.selectedValue;
            }
            node.setDirtyCanvas(true, true);
        }

        // Keep the in-node big preview in lock-step with the selection, so the
        // pick looks identical no matter where it came from — an in-node card,
        // the fullscreen gallery (on Done), or a reloaded workflow: a selected
        // style shows expanded, no selection returns to the grid.
        if (state.selectedValue) {
            const rec = cardRecords.find(
                (record) => record.value === state.selectedValue
                    || matchesSelection(record.style, state.selectedValue),
            );
            if (rec && expandedRecord?.value !== rec.value) {
                expandPreview(rec);
            }
        } else {
            collapsePreview();
        }
    };

    // The card record currently blown up to fill the body, or null.
    let expandedRecord = null;
    const expandPreview = (record) => {
        if (!record?.style?.preview) return;
        expandImg.src = makePreviewUrl(record.style.preview);
        expandImg.alt = styleLabel(record.style);
        expandName.textContent = styleLabel(record.style);
        expandName.title = styleTooltip(record.style);
        expand.classList.add("is-active");
        expandedRecord = record;
    };
    const collapsePreview = () => {
        if (!expandedRecord) return;
        const record = expandedRecord;
        expand.classList.remove("is-active");
        expandedRecord = null;
        // Return the user to the spot in the grid where that thumbnail lives.
        requestAnimationFrame(() => {
            try { record.el.scrollIntoView({ block: "center", behavior: "auto" }); } catch { /* ignore */ }
        });
    };
    expand.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        // Turn the style OFF: setSelection("") collapses the preview back to the
        // grid AND clears the selection, so the node applies no style.
        setSelection("", true);
    });

    const rebuildCategorySelect = () => {
        categorySelect.innerHTML = "";
        const counts = new Map(); // canonical (en) category -> {label, count}
        state.styles.forEach((style) => {
            const key = style.category || "";
            if (!key) {
                return;
            }
            const entry = counts.get(key) || { label: categoryLabel(style), count: 0 };
            entry.count += 1;
            counts.set(key, entry);
        });

        const allOption = document.createElement("option");
        allOption.value = ALL_CATEGORIES;
        allOption.textContent = `${L.allCategories} (${state.styles.length})`;
        categorySelect.appendChild(allOption);

        counts.forEach((entry, key) => {
            const option = document.createElement("option");
            option.value = key;
            option.textContent = `${entry.label} (${entry.count})`;
            categorySelect.appendChild(option);
        });
        categorySelect.value = state.category;
    };

    const buildGrid = () => {
        scroll.innerHTML = "";
        cardRecords.length = 0;
        sectionRecords.length = 0;

        // One section per category: a sticky header plus its own card grid.
        const byCategory = new Map();
        state.styles.forEach((style) => {
            if (!styleValue(style)) return;
            const key = style.category || "";
            if (!byCategory.has(key)) byCategory.set(key, []);
            byCategory.get(key).push(style);
        });

        byCategory.forEach((styles, category) => {
            const section = document.createElement("div");
            section.className = "ts-style-section";

            const header = document.createElement("div");
            header.className = "ts-style-header";
            header.textContent = category ? categoryLabel(styles[0]) : "";
            if (!category) header.hidden = true;
            section.appendChild(header);

            const grid = document.createElement("div");
            grid.className = "ts-style-grid";
            section.appendChild(grid);

            const records = [];
            styles.forEach((style) => {
                const value = styleValue(style);
                const card = document.createElement("button");
                card.type = "button";
                card.className = "ts-style-card";
                card.dataset.value = value;
                card.title = styleTooltip(style);

                if (style.preview) {
                    const img = document.createElement("img");
                    img.alt = styleLabel(style) || "style";
                    img.loading = "lazy";
                    img.src = makePreviewUrl(style.preview);
                    img.onerror = () => { img.remove(); };
                    card.appendChild(img);
                }

                const label = document.createElement("div");
                label.className = "ts-style-label";
                label.textContent = styleLabel(style);
                card.appendChild(label);

                card.addEventListener("click", (event) => {
                    event.preventDefault();
                    // Select the style; setSelection blows its thumbnail up to
                    // fill the body (clicking the enlarged view clears it again).
                    setSelection(value, true);
                });

                grid.appendChild(card);
                const record = {
                    style,
                    value,
                    el: card,
                    haystack: [
                        style.id, style.name, style.name_ru,
                        style.category, style.category_ru,
                        style.description, style.prompt,
                    ].filter(Boolean).join(" ").toLowerCase(),
                };
                cardRecords.push(record);
                records.push(record);
            });

            scroll.appendChild(section);
            sectionRecords.push({ category, el: section, header, records });
        });
    };

    // Filtering shortens the content, so the browser clamps scrollTop to 0.
    // Remember where the user was browsing and put them back when the filter
    // is cleared, instead of dumping them at the top of 113 styles.
    let browsePosition = 0;
    let wasFiltered = false;

    const applyFilter = () => {
        if (state.loading) {
            empty.textContent = L.loading;
            empty.style.display = "block";
            return;
        }

        const query = search.value.trim().toLowerCase();
        const filtered = Boolean(query) || state.category !== ALL_CATEGORIES;
        if (filtered && !wasFiltered) {
            browsePosition = scroll.scrollTop;
        }

        let visibleCount = 0;
        sectionRecords.forEach((section) => {
            const categoryOk = state.category === ALL_CATEGORIES || section.category === state.category;
            let sectionVisible = 0;
            section.records.forEach((record) => {
                const show = categoryOk && (!query || record.haystack.includes(query));
                record.el.hidden = !show;
                if (show) sectionVisible += 1;
            });
            // An empty section would otherwise leave its header floating.
            section.el.hidden = sectionVisible === 0;
            visibleCount += sectionVisible;
        });

        empty.textContent = L.noStyles;
        empty.style.display = visibleCount ? "none" : "block";

        if (!filtered && wasFiltered && browsePosition) {
            // The rows only exist again after layout, hence the next frame.
            requestAnimationFrame(() => { scroll.scrollTop = browsePosition; });
        }
        wasFiltered = filtered;
    };

    const syncSelection = () => {
        const stored = styleWidget?.value || node.properties?.[STYLE_INPUT] || "";
        setSelection(stored, false);
    };

    const scrollSelectionIntoView = () => {
        const selected = cardRecords.find((record) => record.el.classList.contains("is-selected"));
        selected?.el.scrollIntoView({ block: "nearest" });
    };

    const loadStyles = async () => {
        state.loading = true;
        applyFilter();
        try {
            const response = await fetch(api.apiURL("/ts_styles"));
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const payload = await response.json();
            state.styles = Array.isArray(payload.styles) ? payload.styles : [];
            state.loading = false;
            rebuildCategorySelect();
            buildGrid();
            syncSelection();
            applyFilter();
            scrollSelectionIntoView();
        } catch (error) {
            state.loading = false;
            empty.textContent = L.loadFailed;
            empty.style.display = "block";
            console.error("[TS Style Prompt Selector] Failed to load styles:", error);
        }
    };

    // ── Fullscreen library ────────────────────────────────────────────────
    // A big, comfortable browser over the SAME style data + selection. Built
    // fresh on open (cheap: cards are plain <img loading="lazy">), torn down on
    // close so no listeners leak.
    const openStyleLibrary = () => {
        if (modalRoot) return;
        const doc = node?.graph?.canvas?.canvas?.ownerDocument || document;

        const overlay = doc.createElement("div");
        overlay.className = `${TS_UI_CLASS} ts-ui-modal ts-style-modal`;

        const keyAnchor = doc.createElement("textarea");
        keyAnchor.className = "ts-ui-keyanchor";
        keyAnchor.setAttribute("aria-hidden", "true");

        const panel = doc.createElement("div");
        panel.className = "ts-style-modal__panel";

        const head = doc.createElement("div");
        head.className = "ts-style-modal__head";

        const title = doc.createElement("div");
        title.className = "ts-style-modal__title";
        title.textContent = L.modalTitle;

        const mSearch = doc.createElement("input");
        mSearch.type = "search";
        mSearch.className = "ts-style-modal__search";
        mSearch.placeholder = L.modalSearch;

        const mCat = doc.createElement("select");
        mCat.className = "ts-ui-select ts-style-modal__cat";

        const selInfo = doc.createElement("div");
        selInfo.className = "ts-style-modal__selinfo";

        const clearBtn = doc.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "ts-ui-btn";
        clearBtn.textContent = L.clear;

        const doneBtn = doc.createElement("button");
        doneBtn.type = "button";
        doneBtn.className = "ts-ui-btn ts-ui-btn--primary";
        doneBtn.textContent = L.done;

        head.append(title, mSearch, mCat, selInfo, clearBtn, doneBtn);

        const mScroll = doc.createElement("div");
        mScroll.className = "ts-style-modal__scroll";

        const mEmpty = doc.createElement("div");
        mEmpty.className = "ts-style-modal__empty";
        mEmpty.textContent = L.noStyles;
        mEmpty.style.display = "none";

        panel.append(head, mScroll, mEmpty);
        overlay.append(keyAnchor, panel);

        // Category dropdown (mirrors the in-node one).
        const counts = new Map();
        state.styles.forEach((style) => {
            const key = style.category || "";
            if (!key) return;
            const entry = counts.get(key) || { label: categoryLabel(style), count: 0 };
            entry.count += 1;
            counts.set(key, entry);
        });
        const allOpt = doc.createElement("option");
        allOpt.value = ALL_CATEGORIES;
        allOpt.textContent = `${L.allCategories} (${state.styles.length})`;
        mCat.appendChild(allOpt);
        counts.forEach((entry, key) => {
            const opt = doc.createElement("option");
            opt.value = key;
            opt.textContent = `${entry.label} (${entry.count})`;
            mCat.appendChild(opt);
        });
        mCat.value = ALL_CATEGORIES;

        // Cards, grouped into sticky-header sections like the in-node grid.
        const mRecords = [];
        const mSections = [];
        const byCategory = new Map();
        state.styles.forEach((style) => {
            if (!styleValue(style)) return;
            const key = style.category || "";
            if (!byCategory.has(key)) byCategory.set(key, []);
            byCategory.get(key).push(style);
        });
        byCategory.forEach((styles, category) => {
            const section = doc.createElement("div");
            section.className = "ts-style-modal__section";
            const header = doc.createElement("div");
            header.className = "ts-style-modal__header";
            header.textContent = category ? categoryLabel(styles[0]) : "";
            if (!category) header.hidden = true;
            const grid = doc.createElement("div");
            grid.className = "ts-style-modal__grid";
            section.append(header, grid);

            const records = [];
            styles.forEach((style) => {
                const value = styleValue(style);
                const card = doc.createElement("button");
                card.type = "button";
                card.className = "ts-style-modal__card";
                card.title = styleTooltip(style);

                const thumb = doc.createElement("div");
                thumb.className = "ts-style-modal__thumb";
                if (style.preview) {
                    const img = doc.createElement("img");
                    img.alt = styleLabel(style) || "style";
                    img.loading = "lazy";
                    img.src = makePreviewUrl(style.preview);
                    img.onerror = () => { img.remove(); };
                    thumb.appendChild(img);
                }
                const check = doc.createElement("div");
                check.className = "ts-style-modal__check";
                check.textContent = "✓";
                thumb.appendChild(check);

                const meta = doc.createElement("div");
                meta.className = "ts-style-modal__meta";
                const nm = doc.createElement("div");
                nm.className = "ts-style-modal__name";
                nm.textContent = styleLabel(style);
                meta.appendChild(nm);
                const descText = (isRu ? style.description : style.prompt) || style.description || style.prompt;
                if (descText) {
                    const desc = doc.createElement("div");
                    desc.className = "ts-style-modal__desc";
                    desc.textContent = descText;
                    meta.appendChild(desc);
                }
                card.append(thumb, meta);

                card.addEventListener("click", (event) => {
                    event.preventDefault();
                    const next = value === state.selectedValue ? "" : value;
                    setSelection(next, true);
                });

                grid.appendChild(card);
                const record = {
                    style,
                    value,
                    el: card,
                    haystack: [
                        style.id, style.name, style.name_ru,
                        style.category, style.category_ru,
                        style.description, style.prompt,
                    ].filter(Boolean).join(" ").toLowerCase(),
                };
                mRecords.push(record);
                records.push(record);
            });
            mScroll.appendChild(section);
            mSections.push({ category, el: section, records });
        });

        const applyModalFilter = () => {
            const query = mSearch.value.trim().toLowerCase();
            const cat = mCat.value || ALL_CATEGORIES;
            let visible = 0;
            mSections.forEach((section) => {
                const catOk = cat === ALL_CATEGORIES || section.category === cat;
                let n = 0;
                section.records.forEach((record) => {
                    const show = catOk && (!query || record.haystack.includes(query));
                    record.el.hidden = !show;
                    if (show) n += 1;
                });
                section.el.hidden = n === 0;
                visible += n;
            });
            mEmpty.style.display = visible ? "none" : "block";
        };

        const refresh = () => {
            mRecords.forEach((record) => {
                record.el.classList.toggle(
                    "is-selected",
                    record.value === state.selectedValue || matchesSelection(record.style, state.selectedValue),
                );
            });
            const sel = mRecords.find((r) => r.el.classList.contains("is-selected"));
            selInfo.textContent = sel ? L.selected(styleLabel(sel.style)) : L.selectedNone;
        };

        const close = () => {
            if (!modalRoot) return;
            doc.removeEventListener("keydown", onKey, true);
            overlay.remove();
            modalRoot = null;
            modalRefresh = null;
        };
        const onKey = (event) => {
            if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); close(); }
        };

        mSearch.addEventListener("input", applyModalFilter);
        mCat.addEventListener("change", applyModalFilter);
        clearBtn.addEventListener("click", () => setSelection("", true));
        doneBtn.addEventListener("click", close);
        // Click on the dimmed backdrop (outside the panel) closes.
        overlay.addEventListener("pointerdown", (event) => {
            if (event.target === overlay) close();
        });
        // The overlay lives over the graph canvas; stop events leaking to it.
        ["pointerdown", "pointerup", "wheel", "click", "dblclick", "contextmenu", "keydown"].forEach((ev) => {
            panel.addEventListener(ev, (event) => event.stopPropagation());
        });
        doc.addEventListener("keydown", onKey, true);

        doc.body.appendChild(overlay);
        modalRoot = overlay;
        modalRefresh = refresh;
        applyModalFilter();
        refresh();
        // Focus the search for immediate typing; keyAnchor parks graph hotkeys.
        requestAnimationFrame(() => mSearch.focus());
    };
    node._tsStyleSelectorCloseLibrary = () => { if (modalRoot) modalRoot.remove(); modalRoot = null; modalRefresh = null; };

    search.addEventListener("input", () => {
        if (searchTimer) {
            clearTimeout(searchTimer);
        }
        searchTimer = setTimeout(() => {
            searchTimer = null;
            applyFilter();
        }, SEARCH_DEBOUNCE_MS);
    });

    categorySelect.addEventListener("change", () => {
        state.category = categorySelect.value || ALL_CATEGORIES;
        applyFilter();
        if (state.category !== ALL_CATEGORIES) scroll.scrollTop = 0;
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

    node._tsStyleSelectorSync = () => {
        syncSelection();
        applyFilter();
    };

    const previousOnResize = node.onResize;
    const onResizeWrapped = function () {
        const result = previousOnResize?.apply(this, arguments);
        sanitizeNodeSize(this);
        syncLegacyHeight();
        return result;
    };
    node.onResize = onResizeWrapped;

    const teardown = () => {
        node._tsStyleSelectorCloseLibrary?.();
        node._tsStyleSelectorCloseLibrary = null;
        if (searchTimer) {
            clearTimeout(searchTimer);
            searchTimer = null;
        }
        if (node.onResize === onResizeWrapped) {
            node.onResize = previousOnResize;
        }
        node._tsStyleSelectorSync = null;
        node._tsStyleSelectorInitialized = false;
    };

    const prevOnRemoved = node.onRemoved;
    const onRemovedWrapped = function () {
        teardown();
        node._tsStyleSelectorCleanup = null;
        return prevOnRemoved?.apply(this, arguments);
    };
    node.onRemoved = onRemovedWrapped;

    node._tsStyleSelectorCleanup = () => {
        teardown();
        if (node.onRemoved === onRemovedWrapped) {
            node.onRemoved = prevOnRemoved;
        }
        removeDomWidgets(node);
    };

    syncLegacyHeight();
    applyFilter();
    loadStyles();
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
        hideStyleWidget(node);
        sanitizeNodeSize(node);
        const hasWidget = node.widgets?.some((widget) => widget?.name === DOM_WIDGET_NAME);
        if (!hasWidget || !node._tsStyleSelectorSync) {
            setupStyleSelector(node);
        }
        node.resizable = true;
        node._tsStyleSelectorSync?.();
    },
});
