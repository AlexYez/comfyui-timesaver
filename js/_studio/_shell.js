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
/* Коробка вкладок режимов: существует, только чтобы их можно было
   пересобрать; на раскладку не влияет. */
.ts-studio__railmodes{display:contents}
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
.ts-studio__side.is-collapsed .ts-studio__resizer{pointer-events:none}
.ts-studio__stage{position:relative;min-width:0;min-height:0;background:var(--ts-sunken)}
.ts-studio__side{position:relative;display:flex;flex-direction:column;min-height:0;
    width:var(--ts-studio-side-w,280px);
    border-right:1px solid var(--ts-border);background:var(--ts-elevated)}
/* Свёрнутая панель — ровно та линия, что отделяла её от сцены. Ни культи со
   своей рамкой (читалась как вторая полоса), ни ярлычка на разделителе: он
   неизбежно выбирал между второй линией и налезанием на холст. Разворачивает
   панель обычная иконка в рельсе — там ей ничто не мешает и она одинаково
   работает при любом положении браузера. */
.ts-studio__side.is-collapsed{width:0}
.ts-studio__side.is-collapsed>*{display:none}
/* The asset browser can live on either edge (Settings). Mirroring is a matter
   of column order and which side owns the divider — nothing in the panel
   itself changes. */
.ts-studio--side-right{grid-template-columns:44px var(--ts-studio-deck-w,340px) minmax(0,1fr) auto}
.ts-studio--side-right .ts-studio__side{order:3;border-right:none;
    border-left:1px solid var(--ts-border)}
.ts-studio--side-right .ts-studio__deck{order:1}
.ts-studio--side-right .ts-studio__stage{order:2}
/* On this side the panel's own tab strip runs into the corner the fullscreen
   close button occupies, so it yields the same reserved room the other
   top-edge bars do. */
.ts-studio--side-right .ts-studio__side>*:first-child,
.ts-studio--side-right .ts-studio__gallerytabs{padding-right:var(--ts-fs-safe-right)}
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
    // Вкладки режимов живут в своей коробке, чтобы их можно было пересобрать,
    // не тронув всё остальное в рельсе (настройки, наборы, справка, ярлычок
    // панели добавляются приложением ПОСЛЕ распорки). `display:contents`
    // оставляет кнопки прямыми детьми рельса для раскладки.
    const modeHost = document.createElement("div");
    modeHost.className = "ts-studio__railmodes";
    rail.appendChild(modeHost);

    /**
     * Пересобрать вкладки режимов.
     *
     * Нужно потому, что набор разделов зависит от установленных моделей:
     * выключили пак — раздел, который держался только на нём, обязан исчезнуть
     * сразу, а не «после переоткрытия студии».
     *
     * @param {{id: string, title: string, icon: string}[]} modes
     */
    function setModes(modes) {
        modeHost.replaceChildren();
        railButtons.clear();
        for (const mode of modes || []) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "ts-studio__railbtn";
            button.title = mode.title;
            button.setAttribute("role", "tab");
            button.setAttribute("aria-label", mode.title);
            button.innerHTML = mode.icon;
            button.addEventListener("click", () => options.onMode?.(mode.id));
            modeHost.appendChild(button);
            railButtons.set(mode.id, button);
        }
    }
    setModes(options.modes);
    const spacer = document.createElement("div");
    spacer.className = "ts-studio__railspacer";
    rail.appendChild(spacer);
    // Кнопка сворачивания браузера добавляется ниже, сразу после распорки.

    const deck = document.createElement("div");
    deck.className = "ts-studio__deck";
    const deckBody = document.createElement("div");
    deckBody.className = "ts-studio__deckbody";
    deck.appendChild(deckBody);
    const stage = document.createElement("div");
    stage.className = "ts-studio__stage";
    const side = document.createElement("div");
    side.className = "ts-studio__side";

    // Переключатель браузера ассетов живёт в рельсе, а не на разделителе.
    //
    // Ярлычок на линии между панелью и сценой не имеет хорошего положения: с
    // одной стороны он даёт свёрнутой панели вторую полосу, с другой лезет на
    // холст. В рельсе он просто иконка среди иконок — не мешает ничему и
    // ведёт себя одинаково, на каком бы краю ни жил сам браузер.
    const sideToggle = document.createElement("button");
    sideToggle.type = "button";
    sideToggle.className = "ts-studio__railbtn ts-studio__sidetoggle";
    sideToggle.title = options.collapseTitle;
    sideToggle.setAttribute("aria-label", options.collapseTitle);
    sideToggle.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none"'
        + ' stroke="currentColor" stroke-width="1.7"><rect x="3" y="4" width="18"'
        + ' height="16" rx="2"/><path d="M9 4v16"/></svg>';
    sideToggle.addEventListener("click",
        () => setSideCollapsed(!side.classList.contains("is-collapsed")));
    rail.appendChild(sideToggle);

    root.append(rail, side, deck, stage);

    function setMode(modeId) {
        for (const [id, button] of railButtons) {
            button.classList.toggle("is-active", id === modeId);
            button.setAttribute("aria-selected", id === modeId ? "true" : "false");
        }
    }

    function setSideCollapsed(collapsed, remember = true) {
        side.classList.toggle("is-collapsed", collapsed);
        // Сворачивание — такая же настройка рабочего места, как ширина
        // колонок: свернул браузер ассетов, закрыл студию, открыл — он обязан
        // остаться свёрнутым. Раньше он каждый раз разворачивался сам.
        if (remember) rememberColumns({ sideCollapsed: collapsed });
        // Кнопка подсвечена, когда панель открыта — как остальные в рельсе.
        sideToggle.classList.toggle("is-active", !collapsed);
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
    }

    // ── fluid columns: draggable widths, persisted per install ──────────── //
    const WIDTH_KEY = "ts-studio.columns";
    let widths = { deck: 340, side: 280, sideCollapsed: false, panelTab: "" };
    try {
        widths = { ...widths, ...JSON.parse(localStorage.getItem(WIDTH_KEY) || "{}") };
    } catch { /* defaults stand */ }

    function rememberColumns(patch) {
        widths = { ...widths, ...patch };
        try {
            localStorage.setItem(WIDTH_KEY, JSON.stringify(widths));
        } catch { /* private mode */ }
    }
    function applyWidths() {
        root.style.setProperty("--ts-studio-deck-w",
            `${Math.min(560, Math.max(240, widths.deck))}px`);
        root.style.setProperty("--ts-studio-side-w",
            `${Math.min(520, Math.max(200, widths.side))}px`);
    }
    applyWidths();
    // Безусловно: кнопка обязана отражать состояние и при открытой панели.
    setSideCollapsed(Boolean(widths.sideCollapsed), false);

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
            rememberColumns({});
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
        setMode, setModes, setSideCollapsed, setSidePlacement,
        /** Какая вкладка боковой панели была открыта в прошлый раз. */
        panelTab: () => widths.panelTab || "",
        rememberPanelTab: (which) => rememberColumns({ panelTab: String(which || "") }),
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
