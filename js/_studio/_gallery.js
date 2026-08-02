// TS Studio kit — session gallery panel (ui-kit layer).
//
// The right panel's "Session" tab: a grid of this session's results, newest
// first. Selecting a card is the act that feeds the bridge node's output.
// Type-specific card rendering is injected (images now; video/audio studios
// pass their own), the grid mechanics stay shared.

import { resultViewUrl } from "./_session.js";

const STYLE_ID = "ts-studio-gallery-styles";

export function ensureGalleryStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-studio__gallery{display:flex;flex-direction:column;min-height:0;flex:1}
.ts-studio__gallerytabs{display:flex;gap:2px;padding:8px 8px 0}
.ts-studio__gallerytab{flex:1;padding:5px 0;border:none;background:none;color:var(--ts-muted);
    cursor:pointer;font-size:var(--ts-fs-sm);border-bottom:2px solid transparent}
.ts-studio__gallerytab.is-active{color:var(--ts-text);border-bottom-color:var(--ts-accent)}
.ts-studio__gallerygrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;
    padding:8px;overflow-y:auto;align-content:start;flex:1}
.ts-studio__card{position:relative;border:2px solid transparent;border-radius:var(--ts-radius);
    overflow:hidden;cursor:pointer;padding:0;background:var(--ts-sunken);aspect-ratio:1}
.ts-studio__card img{width:100%;height:100%;object-fit:cover;display:block}
.ts-studio__card.is-selected{border-color:var(--ts-accent)}
.ts-studio__card:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
.ts-studio__galleryempty{padding:16px 10px;color:var(--ts-muted);font-size:var(--ts-fs-sm);
    text-align:center}
`;
    document.head.appendChild(style);
}

/**
 * @param {object} options
 * @param {(result: {image: object, params: object|null}) => void} options.onSelect
 * @param {object} options.t Locale strings.
 * @returns {{element, add, setAll, selectLast, count}}
 */
export function createGallery(options) {
    ensureGalleryStyles();
    const element = document.createElement("div");
    element.className = "ts-studio__gallery";

    const tabs = document.createElement("div");
    tabs.className = "ts-studio__gallerytabs";
    const sessionTab = document.createElement("button");
    sessionTab.type = "button";
    sessionTab.className = "ts-studio__gallerytab is-active";
    sessionTab.textContent = options.t.tabSession;
    const libraryTab = document.createElement("button");
    libraryTab.type = "button";
    libraryTab.className = "ts-studio__gallerytab";
    libraryTab.textContent = options.t.tabLibrary;
    libraryTab.disabled = true; // asset providers arrive in phase 2
    libraryTab.title = options.t.tabLibrarySoon;
    tabs.append(sessionTab, libraryTab);

    const grid = document.createElement("div");
    grid.className = "ts-studio__gallerygrid";
    grid.setAttribute("role", "listbox");
    const empty = document.createElement("div");
    empty.className = "ts-studio__galleryempty";
    empty.textContent = options.t.galleryEmpty;

    element.append(tabs, grid, empty);

    const results = [];
    let selectedCard = null;

    function syncEmpty() {
        empty.style.display = results.length ? "none" : "";
    }

    function add(result) {
        results.push(result);
        const card = document.createElement("button");
        card.type = "button";
        card.className = "ts-studio__card";
        card.setAttribute("role", "option");
        const img = document.createElement("img");
        img.loading = "lazy";
        img.alt = result.image.filename;
        img.src = resultViewUrl(result.image);
        card.appendChild(img);
        card.addEventListener("click", () => select(card, result));
        grid.prepend(card);
        syncEmpty();
        return { card, result };
    }

    function select(card, result) {
        selectedCard?.classList.remove("is-selected");
        selectedCard = card;
        card.classList.add("is-selected");
        card.setAttribute("aria-selected", "true");
        options.onSelect?.(result);
    }

    function setAll(list) {
        results.length = 0;
        grid.textContent = "";
        selectedCard = null;
        for (const result of list) add(result);
    }

    function selectLast() {
        const card = grid.firstElementChild;
        if (card && results.length) select(card, results[results.length - 1]);
    }

    return { element, add, setAll, selectLast, count: () => results.length };
}
