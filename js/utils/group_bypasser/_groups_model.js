// Pure logic for TS Group Bypasser: no DOM, no `app`, no LiteGraph.
//
// Everything here works on plain records, which is what makes it testable
// without a browser (tests/test_group_bypasser_model.py runs it under node).
// The reader in _groups_watch.js turns real LGraphGroups into these records;
// the view in _groups_view.js turns the answers back into pixels.
//
// The one idea worth holding on to: THE GRAPH IS THE TRUTH. A group is "off"
// because the nodes inside it are bypassed, not because a checkbox says so.
// Nothing here stores state — every answer is derived from the modes handed in.
// That is why a node muted by hand, or shared with another group, can never
// make the list lie: it simply reads as "mixed".

/** LiteGraph node modes we care about. */
export const MODE_ALWAYS = 0;
export const MODE_NEVER = 2; // muted by hand; this node never sets it
export const MODE_BYPASS = 4;

export const STATE_ON = "on";
export const STATE_OFF = "off";
export const STATE_MIXED = "mixed";
export const STATE_EMPTY = "empty";

export const SORT_POSITION = "position";
export const SORT_TITLE = "title";
export const SORT_COLOR = "color";

export const RESTRICT_NONE = "none";
export const RESTRICT_MAX_ONE = "max-one";
export const RESTRICT_ALWAYS_ONE = "always-one";

/**
 * @typedef {object} GroupRecord
 * @property {string} key    Identity for reconciliation (group id, or a fallback).
 * @property {string} title
 * @property {string} color  Normalised `#rrggbb`, or "" for a group with no colour.
 * @property {number} x      Canvas position, for reading order.
 * @property {number} y
 * @property {number[]} modes Modes of the nodes this group manages, in graph order.
 */

/**
 * A colour in one shape: lower-case `#rrggbb`, or "" when there is none.
 *
 * Groups carry whatever was written into the workflow — `#3f789e`, `#3F789E`,
 * a three-digit `#39e`, or nothing at all. Comparing those as typed would put
 * the same colour in two different filter buckets.
 */
export function normaliseColour(value) {
    let text = String(value ?? "").trim().toLowerCase();
    if (!text) return "";
    if (text.startsWith("#")) text = text.slice(1);
    if (!/^[0-9a-f]{3,8}$/.test(text)) return "";
    if (text.length === 3) text = text.replace(/(.)(.)(.)/, "$1$1$2$2$3$3");
    // Longer forms carry alpha or are simply unexpected; the first six digits
    // are the colour, and that is all a filter compares.
    return `#${text.slice(0, 6)}`;
}

/**
 * What the checkbox should show for a group.
 *
 * "mixed" is not a nicety. Two things produce it in ordinary use: a node that
 * belongs to two overlapping groups, and a node the person muted by hand.
 * Without a third state the row would have to claim one of them, and be wrong.
 */
export function groupState(modes) {
    if (!modes || modes.length === 0) return STATE_EMPTY;
    if (modes.every((mode) => mode === MODE_ALWAYS)) return STATE_ON;
    if (modes.every((mode) => mode === MODE_BYPASS)) return STATE_OFF;
    return STATE_MIXED;
}

/**
 * Compile the title filter.
 *
 * Plain text is a case-insensitive substring — that is what a person typing
 * into a filter box expects. `/…/` is a regular expression, for when it isn't
 * enough. An unusable expression is reported rather than silently matching
 * nothing, so the field can say so instead of the list going mysteriously
 * empty.
 *
 * @param {string} query
 * @returns {{valid: boolean, empty: boolean, test: (title: string) => boolean}}
 */
export function compileTitleFilter(query) {
    const text = String(query ?? "").trim();
    if (!text) return { valid: true, empty: true, test: () => true };
    const asRegex = text.length > 2 && text.startsWith("/") && text.endsWith("/");
    if (asRegex) {
        try {
            const pattern = new RegExp(text.slice(1, -1), "i");
            return { valid: true, empty: false, test: (title) => pattern.test(String(title ?? "")) };
        } catch {
            return { valid: false, empty: false, test: () => false };
        }
    }
    const needle = text.toLowerCase();
    return {
        valid: true,
        empty: false,
        test: (title) => String(title ?? "").toLowerCase().includes(needle),
    };
}

/**
 * Does a group pass the colour filter?
 *
 * An empty selection means "every colour" rather than "no colour" — a filter
 * nobody has touched must not hide anything. "" in the selection is the
 * bucket for groups with no colour of their own.
 */
export function matchesColour(colour, selected) {
    const wanted = Array.isArray(selected) ? selected : [...(selected || [])];
    if (!wanted.length) return true;
    return wanted.map(normaliseColour).includes(normaliseColour(colour));
}

/** The colours actually present, so the filter offers only what exists. */
export function paletteOf(groups) {
    const seen = [];
    for (const group of groups || []) {
        const colour = normaliseColour(group.color);
        if (!seen.includes(colour)) seen.push(colour);
    }
    // Colourless last: it is the catch-all, not a colour anyone chose.
    return seen.filter(Boolean).sort().concat(seen.includes("") ? [""] : []);
}

/** Groups left after both filters; they combine with AND. */
export function filterGroups(groups, { title = "", colours = [] } = {}) {
    const matcher = compileTitleFilter(title);
    if (!matcher.valid) return [];
    return (groups || []).filter(
        (group) => matcher.test(group.title) && matchesColour(group.color, colours),
    );
}

/**
 * Order for the list.
 *
 * Position is the default because it matches how the graph is read: top to
 * bottom, then left to right. A list ordered by anything else forces the eye
 * to search the canvas for what a row refers to.
 */
export function sortGroups(groups, mode = SORT_POSITION) {
    const rows = [...(groups || [])];
    const byTitle = (a, b) => String(a.title ?? "").localeCompare(String(b.title ?? ""));
    if (mode === SORT_TITLE) return rows.sort(byTitle);
    if (mode === SORT_COLOR) {
        return rows.sort((a, b) => {
            const left = normaliseColour(a.color);
            const right = normaliseColour(b.color);
            // Colourless groups sink to the bottom rather than sorting before
            // every real colour, which is what an empty string would do.
            if (!left !== !right) return left ? -1 : 1;
            return left === right ? byTitle(a, b) : left.localeCompare(right);
        });
    }
    return rows.sort((a, b) => {
        const dy = (a.y ?? 0) - (b.y ?? 0);
        if (Math.abs(dy) > 1) return dy;
        const dx = (a.x ?? 0) - (b.x ?? 0);
        return dx !== 0 ? dx : byTitle(a, b);
    });
}

/** How many rows are in each state, for the counter under the list. */
export function countStates(groups) {
    const tally = { on: 0, off: 0, mixed: 0, empty: 0, total: 0 };
    for (const group of groups || []) {
        tally.total += 1;
        const state = groupState(group.modes);
        if (state === STATE_ON) tally.on += 1;
        else if (state === STATE_OFF) tally.off += 1;
        else if (state === STATE_MIXED) tally.mixed += 1;
        else tally.empty += 1;
    }
    return tally;
}

// --------------------------------------------------------------------------- //
// Deciding what to change
// --------------------------------------------------------------------------- //
//
// Every action answers with a list of decisions — {key, on} — rather than
// touching anything. Applying them needs LiteGraph; choosing them does not,
// and keeping the two apart is what makes the restriction rules checkable.

const isOn = (group) => groupState(group.modes) === STATE_ON;
const hasNodes = (group) => groupState(group.modes) !== STATE_EMPTY;

function decisions(groups, wanted) {
    // Only what actually changes, so an "enable all" over an already-enabled
    // list is one no-op instead of a graph-wide write.
    return groups
        .filter(hasNodes)
        .filter((group) => wanted(group) !== isOn(group))
        .map((group) => ({ key: group.key, on: wanted(group) }));
}

/**
 * Clicking one row.
 *
 * A "mixed" group is brought to order rather than pushed further into it: the
 * click turns everything in it back on. Normal is the state you can reason
 * about, and the second click still turns the whole group off.
 *
 * @param {GroupRecord[]} groups The rows on screen — the restriction rules
 *   apply to what this node manages, not to groups it is filtering out.
 */
export function planToggle(groups, key, restriction = RESTRICT_NONE) {
    const rows = groups || [];
    const target = rows.find((group) => group.key === key);
    if (!target || !hasNodes(target)) return [];
    const state = groupState(target.modes);
    const turningOn = state !== STATE_ON;

    if (!turningOn && restriction === RESTRICT_ALWAYS_ONE) {
        const othersOn = rows.filter((group) => group.key !== key && isOn(group));
        // Nothing else is holding the fort, so this one stays: "always one"
        // means the list is never empty, and a click that would empty it is
        // simply not an instruction we can carry out.
        if (!othersOn.length) return [];
    }
    const exclusive = turningOn
        && (restriction === RESTRICT_MAX_ONE || restriction === RESTRICT_ALWAYS_ONE);
    return decisions(rows, (group) =>
        group.key === key ? turningOn : (exclusive ? false : isOn(group)));
}

/**
 * The buttons under the list. They act on the rows on screen — which is the
 * whole point of having filters.
 *
 * @param {"enable"|"disable"|"invert"} action
 */
export function planBulk(groups, action, restriction = RESTRICT_NONE) {
    const rows = (groups || []).filter(hasNodes);
    const limited = restriction === RESTRICT_MAX_ONE || restriction === RESTRICT_ALWAYS_ONE;

    if (action === "enable") {
        // Under a one-at-a-time rule "enable all" cannot mean all; the first
        // row in the shown order is the one that survives.
        const chosen = limited ? rows.slice(0, 1) : rows;
        return decisions(rows, (group) => chosen.includes(group));
    }
    if (action === "disable") {
        if (restriction === RESTRICT_ALWAYS_ONE && rows.length) {
            const keep = rows.find(isOn) || rows[0];
            return decisions(rows, (group) => group === keep);
        }
        return decisions(rows, () => false);
    }
    if (action === "invert") {
        if (limited) {
            // Inverting cannot honour "one at a time" — the result would be
            // most of the list enabled. Fall back to enabling the first row
            // that is currently off, which is what inverting was reaching for.
            const chosen = rows.find((group) => !isOn(group));
            return chosen ? planToggle(rows, chosen.key, restriction) : [];
        }
        return decisions(rows, (group) => !isOn(group));
    }
    return [];
}
