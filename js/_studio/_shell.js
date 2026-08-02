// TS Studio kit — the application shell (ui-kit layer).
//
// Frame of every TS studio app (image now; video/audio later): mode rail,
// left control deck, center stage, right collapsible panel, all mounted in
// the pack's fullscreen overlay. The shell knows NOTHING about image modes —
// rail tabs come from data, panes come from the caller. Styling: theme
// tokens only, hairline separators, compact density (plan §9).

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";
import { openFullscreenOverlay } from "../_fullscreen.js";

const STYLE_ID = "ts-studio-shell-styles";

export function ensureShellStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    // Layout only; hairlines via border tokens, surfaces via theme tokens.
    style.textContent = `
.ts-studio{display:grid;grid-template-columns:44px minmax(280px,340px) minmax(0,1fr) auto;
    width:100%;height:100%;min-height:0;background:var(--ts-bg);color:var(--ts-text);
    font-size:var(--ts-fs)}
.ts-studio__rail{display:flex;flex-direction:column;align-items:center;gap:4px;
    padding:8px 0;border-right:1px solid var(--ts-border);background:var(--ts-elevated)}
.ts-studio__railbtn{width:32px;height:32px;display:flex;align-items:center;justify-content:center;
    border-radius:var(--ts-radius);border:none;background:none;color:var(--ts-muted);cursor:pointer}
.ts-studio__railbtn:hover{background:var(--ts-border-soft);color:var(--ts-text)}
.ts-studio__railbtn.is-active{background:var(--ts-accent-soft);color:var(--ts-accent)}
.ts-studio__railbtn:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
.ts-studio__railspacer{flex:1}
.ts-studio__deck{display:flex;flex-direction:column;min-height:0;overflow-y:auto;
    gap:10px;padding:10px;border-right:1px solid var(--ts-border);background:var(--ts-elevated)}
.ts-studio__stage{position:relative;min-width:0;min-height:0;background:var(--ts-sunken)}
.ts-studio__side{position:relative;display:flex;flex-direction:column;min-height:0;width:280px;
    border-left:1px solid var(--ts-border);background:var(--ts-elevated)}
.ts-studio__side.is-collapsed{width:26px}
.ts-studio__side.is-collapsed>*:not(.ts-studio__sidegrip){display:none}
.ts-studio__sidegrip{position:absolute;left:0;top:50%;transform:translateY(-50%);z-index:3;
    width:14px;height:56px;display:flex;align-items:center;justify-content:center;
    border:1px solid var(--ts-border);border-left:none;border-radius:0 var(--ts-radius) var(--ts-radius) 0;
    background:var(--ts-elevated);color:var(--ts-muted);cursor:pointer;padding:0}
.ts-studio__sidegrip:hover{color:var(--ts-text)}
.ts-studio__section{display:flex;flex-direction:column;gap:5px}
.ts-studio__sectionhead{font-size:var(--ts-fs-xs);font-weight:700;letter-spacing:.05em;
    text-transform:uppercase;color:var(--ts-muted)}
.ts-studio__deckfoot{margin-top:auto;display:flex;flex-direction:column;gap:5px}
`;
    document.head.appendChild(style);
}

/**
 * @param {object} options
 * @param {{id: string, title: string, icon: string}[]} options.modes Rail tabs
 *   (icon = inline SVG string). Data-driven: the video studio registers its
 *   own list here without touching the shell.
 * @param {(modeId: string) => void} options.onMode
 * @param {() => void} options.onClose
 * @param {string} options.label Dialog aria-label.
 * @param {string} options.closeTitle
 * @param {string} options.collapseTitle
 * @returns Shell handle: {root, deck, stage, side, setMode, setSideCollapsed, close}.
 */
export function createShell(options) {
    ensureShellStyles();
    const root = document.createElement("div");
    root.className = `${TS_UI_CLASS} ts-studio`;

    const rail = document.createElement("div");
    rail.className = "ts-studio__rail";
    rail.setAttribute("role", "tablist");
    const railButtons = new Map();
    for (const mode of options.modes) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-studio__railbtn";
        button.title = mode.title;
        button.setAttribute("role", "tab");
        button.setAttribute("aria-label", mode.title);
        button.innerHTML = mode.icon;
        button.addEventListener("click", () => options.onMode?.(mode.id));
        rail.appendChild(button);
        railButtons.set(mode.id, button);
    }
    const spacer = document.createElement("div");
    spacer.className = "ts-studio__railspacer";
    rail.appendChild(spacer);

    const deck = document.createElement("div");
    deck.className = "ts-studio__deck";
    const stage = document.createElement("div");
    stage.className = "ts-studio__stage";
    const side = document.createElement("div");
    side.className = "ts-studio__side";

    const grip = document.createElement("button");
    grip.type = "button";
    grip.className = "ts-studio__sidegrip";
    grip.title = options.collapseTitle;
    grip.setAttribute("aria-label", options.collapseTitle);
    grip.textContent = "›";
    grip.addEventListener("click", () => setSideCollapsed(!side.classList.contains("is-collapsed")));
    side.appendChild(grip);

    root.append(rail, deck, stage, side);

    function setMode(modeId) {
        for (const [id, button] of railButtons) {
            button.classList.toggle("is-active", id === modeId);
            button.setAttribute("aria-selected", id === modeId ? "true" : "false");
        }
    }

    function setSideCollapsed(collapsed) {
        side.classList.toggle("is-collapsed", collapsed);
        grip.textContent = collapsed ? "‹" : "›";
    }

    const overlay = openFullscreenOverlay(root, {
        label: options.label,
        closeTitle: options.closeTitle,
        onClose: options.onClose,
        onKey: options.onKey,
    });

    return {
        root, deck, stage, side, rail,
        setMode, setSideCollapsed,
        parkFocus: overlay.parkFocus,
        close: overlay.close,
        isOpen: overlay.isOpen,
    };
}

/** A titled deck section: header + body container. */
export function deckSection(title) {
    const section = document.createElement("div");
    section.className = "ts-studio__section";
    if (title) {
        const head = document.createElement("div");
        head.className = "ts-studio__sectionhead";
        head.textContent = title;
        section.appendChild(head);
    }
    return section;
}
