// The panel: a list of group names with a checkbox each. Nothing else.
//
// Everything adjustable — the filters, the order, the one-at-a-time rule — is a
// node property, edited through the node's own Properties Panel, and the bulk
// actions live in its right-click menu. The body of the node stays what it is
// meant to be: the groups of this workflow, and whether each one is on.
//
// This module renders and forwards clicks. It decides nothing — what a click
// means is _groups_model.js, and what the graph looks like is _groups_watch.js.

import { TS_UI_CLASS, ensureThemeStyles, pickLocaleStrings } from "../../_theme.js";
import { STATE_EMPTY, STATE_MIXED, STATE_ON } from "./_groups_model.js";

const STYLE_ID = "ts-group-bypasser-styles";

// The node sizes itself to the number of groups, so these three numbers are
// shared arithmetic rather than styling: the entry file adds them up to decide
// how tall the node should be. A row is a fixed height for exactly that reason
// — measuring one back out of the DOM would be JS geometry, which is how
// sizing goes wrong at other zoom levels (CLAUDE.md 12.5.3).
export const ROW_HEIGHT = 26;
export const ROW_GAP = 2;
export const LIST_PADDING = 8;
// Floor for the panel's own min-content: exactly one row. Never zero — zero is
// what collapses a DOM widget to a sliver in Nodes 2.0 — but no larger either,
// because a workflow with a single group should give a single-row node.
export const MIN_CONTENT_HEIGHT = ROW_HEIGHT + LIST_PADDING;
// Room for the "no groups" / "nothing matches" line, which needs more than a
// row to read.
export const MESSAGE_HEIGHT = ROW_HEIGHT * 3;

// Space the node's body spends on anything that is not the panel. Two numbers,
// and telling them apart is the whole fix — a fourth group never fitted and the
// list carried a scrollbar it should never need.
//
// ⚠️ WIDGET_CHROME_HEIGHT CANCELS ITSELF OUT. It is added here and subtracted
// again by the shared helper, which offers the widget `node.size[1] - chrome`
// as its height. So raising it moves the node's edge and grants the panel
// exactly nothing — measured: 10, 30 and 50 all left the same 126 px of panel
// under five groups. The first attempt at this bug did just that.
export const WIDGET_CHROME_HEIGHT = 30;
// ⚠️ THIS is the missing height. The element is laid out into the widget's
// renderArea rather than into the height the layout computed for it, and the
// difference is a fixed inset — 20 px, measured at 1, 2, 4 and 5 groups alike.
// Nothing subtracts it later, so adding it here is what actually reaches the
// list. Nobody may fold it into the constant above; that would cancel it too.
export const WIDGET_LAYOUT_INSET = 20;
// What the panel never gets, whatever the node's height.
export const WIDGET_OVERHEAD = WIDGET_CHROME_HEIGHT + WIDGET_LAYOUT_INSET;
export const MIN_NODE_HEIGHT = WIDGET_OVERHEAD + MIN_CONTENT_HEIGHT;
// Past this the list scrolls instead. A panel taller than this stops being a
// control and becomes a wall.
const MAX_VISIBLE_ROWS = 14;

/**
 * How tall the node wants to be for this many groups.
 *
 * The node sizes itself to its content — a workflow with two groups gets a
 * two-row node, not a two-row list with a hand's width of nothing under it.
 * Rows are a fixed height so this is arithmetic rather than measurement; see
 * the constants in _groups_view.js.
 */
export function heightForRows(count) {
    const rows = Math.max(0, Math.min(Number(count) || 0, MAX_VISIBLE_ROWS));
    const content = rows > 0
        ? rows * ROW_HEIGHT + (rows - 1) * ROW_GAP + LIST_PADDING
        // No groups at all: room for the line that says so.
        : MESSAGE_HEIGHT + LIST_PADDING;
    return Math.max(MIN_NODE_HEIGHT, WIDGET_OVERHEAD + content);
}

export function ensureStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    // Layout only. Every colour comes from the --ts-* tokens, except a group's
    // own dot, which is the colour the person chose on the canvas and is set
    // from data rather than written here.
    style.textContent = `
/* Two opposite hazards in Nodes 2.0, and both are answered here. Vue grants the
   widget a height from the element's own min-content: a list that asks for all
   its rows balloons the node down the canvas, and one that asks for nothing
   collapses it to a sliver. So the rows ask for nothing (the zero flex basis
   below) and the panel itself carries a floor of one row. A hidden spacer
   element would do the same job, but only if it is allowed to take up room —
   ours was cancelling itself out with a negative margin, and the panel's own
   height measured 8 px instead of a row. min-height needs no such care.
   See js/_dom_widget.js and CLAUDE.md 12.5.1.
   NOTE: no backticks in this comment — the whole stylesheet is one template
   literal, and one backtick would end it. */
.ts-gb{display:flex;flex-direction:column;padding:${LIST_PADDING / 2}px;box-sizing:border-box;
  height:100%;min-height:${MIN_CONTENT_HEIGHT}px;overflow:hidden;color:var(--ts-text);
  font-family:var(--ts-font);font-size:var(--ts-fs-sm);pointer-events:auto}
.ts-gb__list{flex:1 1 0;min-height:0;overflow-y:auto;overflow-x:hidden;
  display:flex;flex-direction:column;gap:${ROW_GAP}px;padding-right:2px}
.ts-gb__list::-webkit-scrollbar{width:8px}
.ts-gb__list::-webkit-scrollbar-thumb{background:var(--ts-scrollbar);border-radius:4px}
.ts-gb__list::-webkit-scrollbar-thumb:hover{background:var(--ts-scrollbar-hover)}
.ts-gb__item{display:flex;align-items:center;gap:8px;padding:0 7px;flex:0 0 auto;
  height:${ROW_HEIGHT}px;box-sizing:border-box;
  border-radius:var(--ts-radius-sm);background:var(--ts-sunken);cursor:pointer;user-select:none}
.ts-gb__item:hover{background:var(--ts-surface-hover)}
.ts-gb__item.is-empty{cursor:default;opacity:.45}
.ts-gb__box{width:15px;height:15px;flex:0 0 auto;border-radius:4px;
  border:1px solid var(--ts-border-strong);background:var(--ts-bg);position:relative}
.ts-gb__box::after{content:"";position:absolute;inset:3px;border-radius:2px;background:transparent}
.ts-gb__item[data-state="on"] .ts-gb__box{border-color:var(--ts-accent)}
.ts-gb__item[data-state="on"] .ts-gb__box::after{background:var(--ts-accent)}
/* Half-filled: neither wholly on nor wholly off. A tick or an empty box would
   have to claim one of them, and be wrong. */
.ts-gb__item[data-state="mixed"] .ts-gb__box{border-color:var(--ts-accent-line)}
.ts-gb__item[data-state="mixed"] .ts-gb__box::after{background:var(--ts-accent-dim);inset:6px 3px}
/* Shown only for groups the person actually coloured, so an uncoloured
   workflow stays a clean column of names. */
.ts-gb__dot{width:7px;height:7px;flex:0 0 auto;border-radius:50%}
.ts-gb__name{flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ts-gb__empty{padding:10px 8px;color:var(--ts-muted);text-align:center;line-height:1.5}
`;
    document.head.appendChild(style);
}

/**
 * Build the panel.
 *
 * The words are resolved here, once, at build time: changing ComfyUI's language
 * reloads the page, so there is nothing to redraw for (doc/THEME.md 5.5).
 *
 * @param {object} deps
 * @param {(action: {type: string, key?: string}) => void} deps.onAction
 * @returns {{element: HTMLElement, render: (view: object) => void, strings: object}}
 */
export function createGroupsPanel({ onAction }) {
    ensureStyles();
    const t = pickLocaleStrings(STRINGS);

    const host = document.createElement("div");
    host.className = `${TS_UI_CLASS} ts-gb`;

    const list = document.createElement("div");
    list.className = "ts-gb__list";
    host.append(list);

    const placeholder = document.createElement("div");
    placeholder.className = "ts-gb__empty";

    // Rows are reused between renders so scrolling and hover survive a refresh.
    const rows = new Map();

    function buildRow(key) {
        const item = document.createElement("div");
        item.className = "ts-gb__item";
        item.dataset.key = key;

        const box = document.createElement("span");
        box.className = "ts-gb__box";
        const dot = document.createElement("span");
        dot.className = "ts-gb__dot";
        const name = document.createElement("span");
        name.className = "ts-gb__name";
        item.append(box, dot, name);

        item.addEventListener("click", () => {
            if (item.dataset.state === STATE_EMPTY) return;
            onAction({ type: "toggle", key });
        });
        // Getting to a group is a thing you want now and then, not a button you
        // want to look at every day — so it is a double-click, named in the
        // row's tooltip rather than taking up room in the list.
        item.addEventListener("dblclick", (event) => {
            event.preventDefault();
            onAction({ type: "reveal", key });
        });

        const parts = { item, dot, name };
        rows.set(key, parts);
        return parts;
    }

    /**
     * Draw a finished view model.
     *
     * @param {object} view
     * @param {Array<{key,title,color,state}>} view.rows Filtered and sorted.
     * @param {boolean} view.filterValid
     * @param {number} view.totalGroups Groups before filtering.
     */
    function render(view) {
        const wanted = view.rows || [];
        for (const row of wanted) {
            const parts = rows.get(row.key) || buildRow(row.key);
            parts.item.dataset.state = row.state;
            parts.item.classList.toggle("is-empty", row.state === STATE_EMPTY);
            parts.name.textContent = row.title || t.untitled;
            parts.item.title = rowHint(row);
            parts.dot.style.background = row.color || "transparent";
            parts.dot.style.display = row.color ? "" : "none";
            list.appendChild(parts.item);
        }
        const keep = new Set(wanted.map((row) => row.key));
        for (const [key, parts] of rows) {
            if (keep.has(key)) continue;
            parts.item.remove();
            rows.delete(key);
        }

        if (!wanted.length) {
            placeholder.textContent = view.totalGroups
                ? (view.filterValid === false ? t.filterInvalid : t.nothingMatches)
                : t.noGroups;
            list.appendChild(placeholder);
        } else if (placeholder.parentNode) {
            placeholder.remove();
        }
    }

    function rowHint(row) {
        if (row.state === STATE_EMPTY) return t.emptyHint;
        const state = row.state === STATE_MIXED
            ? t.mixedHint
            : (row.state === STATE_ON ? t.onHint : t.offHint);
        return `${state}\n${t.revealHint}`;
    }

    return { element: host, render, strings: t };
}

/** en/ru for everything this node says, in the panel and in its menu. */
export const STRINGS = {
    en: {
        untitled: "(untitled)",
        onHint: "On — click to bypass",
        offHint: "Bypassed — click to switch on",
        mixedHint: "Partly bypassed — click to switch the whole group on",
        emptyHint: "This group holds no nodes",
        revealHint: "Double-click to show it on the canvas",
        noGroups: "This workflow has no groups yet",
        nothingMatches: "Nothing matches the filter set in the properties",
        filterInvalid: "The filter in the properties cannot be read",
        menuEnableAll: "Switch on every group shown",
        menuDisableAll: "Bypass every group shown",
        menuInvert: "Invert every group shown",
        menuTally: (on, total) => `${on} of ${total} groups on`,
    },
    ru: {
        untitled: "(без названия)",
        onHint: "Включена — клик выключит",
        offHint: "Выключена — клик включит",
        mixedHint: "Включена частично — клик включит группу целиком",
        emptyHint: "В этой группе нет нод",
        revealHint: "Двойной клик покажет её на канвасе",
        noGroups: "В этом workflow пока нет групп",
        nothingMatches: "Под фильтр из свойств ничего не подошло",
        filterInvalid: "Фильтр в свойствах не читается",
        menuEnableAll: "Включить все показанные группы",
        menuDisableAll: "Выключить все показанные группы",
        menuInvert: "Инвертировать показанные группы",
        menuTally: (on, total) => `включено ${on} из ${total} групп`,
    },
};
