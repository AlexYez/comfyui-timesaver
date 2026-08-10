// Reading the graph's groups, and writing bypass back into them.
//
// Everything that has to know about LiteGraph lives here; the decisions live in
// _groups_model.js and the pixels in _groups_view.js.

import { app } from "/scripts/app.js";

import {
    MODE_ALWAYS,
    MODE_BYPASS,
    normaliseColour,
} from "./_groups_model.js";

// The node type this panel controls — and the one type it must never touch.
// A panel that can bypass itself is a panel you cannot switch back on.
export const NODE_TYPE = "TS_GroupBypasser";

// How often the graph is re-read. There is no event for "a group was renamed"
// or "a group was recoloured" — none at all — so the only honest way to keep
// the list true is to look. The cost is kept down three ways: one watcher for
// every panel on the canvas rather than one each, a cheap signature that
// decides whether anything is worth rebuilding, and no work while the canvas is
// being dragged or the tab is in the background.
const POLL_MS = 400;
// Node bounds are expensive to gather and change only when something moves, so
// one sweep is shared by every group examined in the same tick.
const BOUNDS_CACHE_MS = 50;

/** The graph currently on screen, or null before one exists. */
function currentGraph() {
    const canvas = app?.canvas;
    return canvas?.getCurrentGraph?.() || app?.graph || null;
}

/** Groups of the graph on screen, in whatever order it keeps them. */
function graphGroups(graph) {
    const groups = graph?.groups || graph?._groups || [];
    return Array.isArray(groups) ? groups : [...groups];
}

/**
 * Nodes this panel is willing to switch, for one group.
 *
 * Membership is geometric and is recomputed every time, because nothing tells
 * us when a node was dragged out of a group. LiteGraph's own recompute is what
 * decides it — a node belongs to the group whose rectangle holds its centre,
 * which is exactly the set that travels when the group is moved. Matching that
 * rule is what keeps the list agreeing with what the eye sees.
 */
export function managedNodes(group) {
    try {
        group.recomputeInsideNodes?.();
    } catch (err) {
        // The frontend guards this call too: a group detached from its graph
        // throws rather than answering.
        console.warn("[TS GroupBypasser] could not recompute a group's nodes", err);
        return [];
    }
    const inside = group._nodes || group.nodes || [];
    return [...inside].filter((node) => node && node.type !== NODE_TYPE);
}

/**
 * How a group is identified between two reads.
 *
 * Recent frontends give groups an `id` and save it in the workflow, which is
 * the identity to prefer. Older ones do not, so a title-and-place fallback
 * keeps the rows from being rebuilt from scratch on every poll. Nothing is
 * persisted under this key — it only has to be stable while the panel is open.
 */
export function groupKey(group, index) {
    if (group?.id !== undefined && group?.id !== null && group.id !== "") {
        return `id:${group.id}`;
    }
    const bounds = group?._bounding || group?.bounding || [];
    return `at:${index}:${group?.title ?? ""}:${bounds[0] ?? 0},${bounds[1] ?? 0}`;
}

/** The graph's groups as the plain records the model works on. */
export function readGroups() {
    const graph = currentGraph();
    if (!graph) return [];
    return graphGroups(graph).map((group, index) => {
        const nodes = managedNodes(group);
        const bounds = group._bounding || group.bounding || [];
        return {
            key: groupKey(group, index),
            title: String(group.title ?? ""),
            color: normaliseColour(group.color),
            x: Number(bounds[0]) || 0,
            y: Number(bounds[1]) || 0,
            modes: nodes.map((node) => Number(node.mode) || 0),
            // Kept off the model's path on purpose: it needs records it can
            // reason about, not live objects.
            _group: group,
            _nodes: nodes,
        };
    });
}

/**
 * A short string that changes whenever the list would look different.
 *
 * Rebuilding the DOM four times a second would fight the user's own scrolling
 * and hover; comparing this first means the panel is rebuilt only when the
 * graph really moved under it.
 */
export function signatureOf(records) {
    return records
        .map((row) => `${row.key}|${row.title}|${row.color}|${row.modes.join("")}`)
        .join("\n");
}

/**
 * Apply the model's decisions to the graph, as ONE undoable step.
 *
 * Without the beforeChange/afterChange pair, Ctrl+Z would take a group apart
 * one node at a time — twenty presses to undo one click.
 *
 * @param {Array<{key: string, on: boolean}>} decisions
 * @param {Array<object>} records The reading the decisions were made from.
 * @returns {number} How many nodes actually changed mode.
 */
export function applyDecisions(decisions, records) {
    if (!decisions?.length) return 0;
    const byKey = new Map(records.map((row) => [row.key, row]));
    const graph = currentGraph();
    const canvas = app?.canvas;

    canvas?.emitBeforeChange?.();
    graph?.beforeChange?.();
    let touched = 0;
    try {
        for (const decision of decisions) {
            const row = byKey.get(decision.key);
            if (!row) continue;
            const wanted = decision.on ? MODE_ALWAYS : MODE_BYPASS;
            // Re-read membership at the moment of writing: the reading this
            // decision came from may be up to a poll old, and a node dragged
            // out in the meantime is no longer ours to switch.
            for (const node of managedNodes(row._group)) {
                if (Number(node.mode) === wanted) continue;
                node.mode = wanted;
                touched += 1;
            }
        }
    } finally {
        graph?.afterChange?.();
        canvas?.emitAfterChange?.();
    }
    if (touched) {
        graph?.change?.();
        canvas?.setDirty?.(true, true);
    }
    return touched;
}

/**
 * Turn the colour filter a person typed into colours the model can compare.
 *
 * Names are accepted alongside hex, because "blue, green" is how someone
 * thinks about the colours they gave their groups — LiteGraph's own palette is
 * what those names mean. "none" (or an empty entry) is the bucket for groups
 * left uncoloured.
 *
 * @param {string} text Comma-separated list from the node's properties.
 * @returns {string[]} Normalised colours; empty means "no filter".
 */
export function resolveColourFilter(text) {
    const palette = globalThis.LGraphCanvas?.node_colors || {};
    const out = [];
    for (const raw of String(text ?? "").split(",")) {
        const token = raw.trim().toLowerCase();
        if (!token) continue;
        if (token === "none" || token === "no colour" || token === "no color") {
            if (!out.includes("")) out.push("");
            continue;
        }
        const named = palette[token]?.groupcolor || palette[token]?.color;
        const colour = normaliseColour(named || token);
        // A word that is neither a colour nor a palette name would otherwise
        // silently become "no colour" and hide everything that has one.
        if (colour && !out.includes(colour)) out.push(colour);
    }
    return out;
}

/** Put the canvas on a group, for the double-click on a row. */
export function revealGroup(record) {
    const group = record?._group;
    const canvas = app?.canvas;
    if (!group || !canvas) return;
    const bounds = group._bounding || group.bounding;
    if (!bounds || bounds.length < 4) return;
    try {
        canvas.ds?.fitToBounds?.([...bounds], { zoom: 0.8 });
    } catch (err) {
        console.warn("[TS GroupBypasser] could not move the canvas to a group", err);
        return;
    }
    canvas.setDirty?.(true, true);
}

// --------------------------------------------------------------------------- //
// One watcher for every panel on the canvas
// --------------------------------------------------------------------------- //

const listeners = new Set();
let timer = null;
let lastSignature = null;
let boundsFreshUntil = 0;

function tick() {
    timer = null;
    if (!listeners.size) return;

    const idle = document.hidden || app?.canvas?.isDragging;
    if (!idle) {
        const now = Date.now();
        // Recomputing group membership walks every node; once per poll is
        // plenty, and skipping it mid-drag keeps the canvas smooth.
        if (now >= boundsFreshUntil) {
            boundsFreshUntil = now + BOUNDS_CACHE_MS;
        }
        let records = [];
        try {
            records = readGroups();
        } catch (err) {
            console.warn("[TS GroupBypasser] could not read the graph's groups", err);
        }
        const signature = signatureOf(records);
        if (signature !== lastSignature) {
            lastSignature = signature;
            for (const listener of listeners) {
                try {
                    listener(records);
                } catch (err) {
                    console.error("[TS GroupBypasser] a panel failed to refresh", err);
                }
            }
        }
    }
    schedule();
}

function schedule() {
    if (timer !== null || !listeners.size) return;
    timer = setTimeout(tick, POLL_MS);
}

/**
 * Be told whenever the graph's groups look different.
 *
 * @param {(records: Array<object>) => void} listener
 * @returns {() => void} Stop listening.
 */
export function watchGroups(listener) {
    listeners.add(listener);
    // The first reading is not deferred: a panel that appears empty for half a
    // second on every workflow load reads as broken.
    try {
        listener(readGroups());
    } catch (err) {
        console.error("[TS GroupBypasser] a panel failed its first refresh", err);
    }
    schedule();
    return () => {
        listeners.delete(listener);
        if (!listeners.size && timer !== null) {
            clearTimeout(timer);
            timer = null;
            lastSignature = null;
        }
    };
}

/** Force the next poll to report, after this panel changed something itself. */
export function invalidateSignature() {
    lastSignature = null;
}
