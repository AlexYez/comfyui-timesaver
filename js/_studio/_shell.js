// TS Studio kit — the application shell (ui-kit layer).
//
// Frame of every TS studio app (image now; video/audio later): mode rail,
// collapsible asset panel, control deck, stage — left to right, mirroring
// ComfyUI's own sidebar order so the browser sits where the eye expects it.
// The shell knows NOTHING about image modes — rail tabs come from data, panes
// come from the caller. Styling: theme tokens only, hairline separators,
// compact density (plan §9).

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
.ts-studio{display:grid;
    grid-template-columns:44px auto var(--ts-studio-deck-w,340px) minmax(0,1fr);
    width:100%;height:100%;min-height:0;background:var(--ts-bg);color:var(--ts-text);
    font-size:var(--ts-fs)}
.ts-studio__rail{display:flex;flex-direction:column;align-items:center;gap:4px;
    padding:8px 0;border-right:1px solid var(--ts-border);background:var(--ts-elevated)}
.ts-studio__railbtn{position:relative;width:32px;height:32px;display:flex;align-items:center;
    justify-content:center;
    border-radius:var(--ts-radius);border:none;background:none;color:var(--ts-muted);cursor:pointer}
.ts-studio__railbtn:hover{background:var(--ts-border-soft);color:var(--ts-text)}
.ts-studio__railbtn.is-active{background:var(--ts-accent-soft);color:var(--ts-accent)}
.ts-studio__railbtn:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
.ts-studio__railspacer{flex:1}
/* The deck is a frame: only its inner body is rebuilt per backend, so the
   resizer strip and any chrome survive a deck rebuild. */
.ts-studio__deck{position:relative;display:flex;flex-direction:column;min-height:0;min-width:0;
    border-right:1px solid var(--ts-border);background:var(--ts-elevated)}
.ts-studio__deckbody{display:flex;flex-direction:column;min-height:0;overflow-y:auto;
    gap:10px;padding:10px;flex:1}
/* Column resizers: invisible strips over the hairlines. The layout is fluid —
   widths are CSS variables, everything else is minmax/percent. */
.ts-studio__resizer{position:absolute;top:0;bottom:0;width:7px;z-index:9;cursor:col-resize}
.ts-studio__resizer:hover,.ts-studio__resizer.is-active{background:var(--ts-accent-soft)}
.ts-studio__stage{position:relative;min-width:0;min-height:0;background:var(--ts-sunken)}
.ts-studio__side{position:relative;display:flex;flex-direction:column;min-height:0;
    width:var(--ts-studio-side-w,280px);
    border-right:1px solid var(--ts-border);background:var(--ts-elevated)}
.ts-studio__side.is-collapsed{width:26px}
.ts-studio__side.is-collapsed>*:not(.ts-studio__sidegrip){display:none}
.ts-studio__sidegrip{position:absolute;right:0;top:50%;transform:translateY(-50%);z-index:10;
    width:14px;height:56px;display:flex;align-items:center;justify-content:center;
    border:1px solid var(--ts-border);border-right:none;border-radius:var(--ts-radius) 0 0 var(--ts-radius);
    background:var(--ts-elevated);color:var(--ts-muted);cursor:pointer;padding:0}
/* The asset browser can live on either edge (Settings). Mirroring is a matter
   of column order and which side owns the divider and the grip — nothing in
   the panel itself changes. */
.ts-studio--side-right{grid-template-columns:44px var(--ts-studio-deck-w,340px) minmax(0,1fr) auto}
.ts-studio--side-right .ts-studio__side{order:3;border-right:none;
    border-left:1px solid var(--ts-border)}
.ts-studio--side-right .ts-studio__deck{order:1}
.ts-studio--side-right .ts-studio__stage{order:2}
.ts-studio--side-right .ts-studio__sidegrip{right:auto;left:0;border-right:1px solid var(--ts-border);
    border-left:none;border-radius:0 var(--ts-radius) var(--ts-radius) 0}
.ts-studio--side-right .ts-studio__sidegrip.is-collapsed{transform:translateY(-50%) scaleX(-1)}
/* On this side the panel's own tab strip runs into the corner the fullscreen
   close button occupies, so it yields the same reserved room the other
   top-edge bars do. */
.ts-studio--side-right .ts-studio__side>*:first-child,
.ts-studio--side-right .ts-studio__gallerytabs{padding-right:var(--ts-fs-safe-right)}
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
    const deckBody = document.createElement("div");
    deckBody.className = "ts-studio__deckbody";
    deck.appendChild(deckBody);
    const stage = document.createElement("div");
    stage.className = "ts-studio__stage";
    const side = document.createElement("div");
    side.className = "ts-studio__side";

    const grip = document.createElement("button");
    grip.type = "button";
    grip.className = "ts-studio__sidegrip";
    grip.title = options.collapseTitle;
    grip.setAttribute("aria-label", options.collapseTitle);
    grip.textContent = "‹";
    grip.addEventListener("click", () => setSideCollapsed(!side.classList.contains("is-collapsed")));
    side.appendChild(grip);

    root.append(rail, side, deck, stage);

    function setMode(modeId) {
        for (const [id, button] of railButtons) {
            button.classList.toggle("is-active", id === modeId);
            button.setAttribute("aria-selected", id === modeId ? "true" : "false");
        }
    }

    function setSideCollapsed(collapsed) {
        side.classList.toggle("is-collapsed", collapsed);
        // The chevron points the way the panel will move, which flips with the
        // side the browser lives on.
        const rightSide = root.classList.contains("ts-studio--side-right");
        grip.textContent = collapsed === rightSide ? "‹" : "›";
    }

    /**
     * Which edge the asset browser occupies.
     *
     * @param {"left"|"right"} placement
     */
    function setSidePlacement(placement) {
        const right = placement === "right";
        root.classList.toggle("ts-studio--side-right", right);
        sideResizer.setEdge(right ? "left" : "right");
        setSideCollapsed(side.classList.contains("is-collapsed"));
    }

    // ── fluid columns: draggable widths, persisted per install ──────────── //
    const WIDTH_KEY = "ts-studio.columns";
    let widths = { deck: 340, side: 280 };
    try {
        widths = { ...widths, ...JSON.parse(localStorage.getItem(WIDTH_KEY) || "{}") };
    } catch { /* defaults stand */ }
    function applyWidths() {
        root.style.setProperty("--ts-studio-deck-w",
            `${Math.min(560, Math.max(240, widths.deck))}px`);
        root.style.setProperty("--ts-studio-side-w",
            `${Math.min(520, Math.max(200, widths.side))}px`);
    }
    applyWidths();

    function makeResizer(host, edge, get, set) {
        const grip = document.createElement("div");
        grip.className = "ts-studio__resizer";
        // The edge a resizer lives on is not fixed: the browser column swaps
        // sides, and a grip left on the far edge would widen the panel when
        // dragged toward it.
        let liveEdge = edge;
        grip.style[liveEdge] = "-4px";
        grip.setEdge = (next) => {
            if (next === liveEdge) return;
            grip.style[liveEdge] = "";
            liveEdge = next;
            grip.style[liveEdge] = "-4px";
        };
        let startX = 0;
        let startW = 0;
        grip.addEventListener("pointerdown", (event) => {
            event.preventDefault();
            grip.setPointerCapture(event.pointerId);
            grip.classList.add("is-active");
            startX = event.clientX;
            startW = get();
        });
        grip.addEventListener("pointermove", (event) => {
            if (!grip.classList.contains("is-active")) return;
            const delta = event.clientX - startX;
            set(startW + (liveEdge === "right" ? delta : -delta));
            applyWidths();
        });
        const stop = () => {
            if (!grip.classList.contains("is-active")) return;
            grip.classList.remove("is-active");
            try {
                localStorage.setItem(WIDTH_KEY, JSON.stringify(widths));
            } catch { /* private mode */ }
        };
        grip.addEventListener("pointerup", stop);
        grip.addEventListener("pointercancel", stop);
        host.appendChild(grip);
        return grip;
    }
    // Each grip lives on the right edge of the column it sizes. The deck hosts
    // its own because only deckBody is wiped on a rebuild — an earlier version
    // put the grip among the rebuilt children and it vanished (measured).
    makeResizer(deck, "right", () => widths.deck, (w) => { widths.deck = w; });
    const sideResizer = makeResizer(side, "right", () => widths.side,
                                    (w) => { widths.side = w; });

    const overlay = openFullscreenOverlay(root, {
        label: options.label,
        closeTitle: options.closeTitle,
        onClose: options.onClose,
        onKey: options.onKey,
    });

    return {
        // `deck` is the rebuildable body; `deckFrame` is the column that keeps
        // the resizer across rebuilds.
        root, deck: deckBody, deckFrame: deck, stage, side, rail,
        setMode, setSideCollapsed, setSidePlacement,
        isSideCollapsed: () => side.classList.contains("is-collapsed"),
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
