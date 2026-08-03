// What the person set stays set — across tabs, models and sittings.
//
// A studio rebuilds its whole deck whenever the mode or the model changes, so
// without this every switch would hand back the file's defaults. That is the
// difference between a tool and a form you fill in again each time.
//
// Two scopes, because they answer different questions:
//
//   shared     the prompt, the seed, the frame, the LoRA chain. These belong
//              to what the person is making, not to which graph renders it,
//              so they follow across models and modes.
//   per-graph  everything else — denoise, steps, scale, presets. "Repaint
//              strength" means a different number in Inpaint than in Upscale,
//              and returning to a mode should find it as it was left.
//
// Values are written as they change, not only on teardown: a closed tab, a
// reload or a crash must not cost the last thing that was touched.

const STORAGE_KEY = "ts.studio.values";
const VERSION = 1;

/** Kinds whose value belongs to the work, not to the graph. */
const SHARED_KINDS = new Set(["prompt", "seed", "size", "loras", "styles"]);

// A design or a long prompt is worth keeping; a runaway value is not. Anything
// larger is remembered for this sitting only, so storage cannot be filled by
// one control.
const MAX_VALUE_BYTES = 64 * 1024;
// Graphs come and go as packs are installed; keep the recently used ones.
const MAX_GRAPHS = 40;

const WRITE_DELAY_MS = 400;

let cache = null;
let writeTimer = null;

function empty() {
    return { v: VERSION, shared: {}, graphs: {}, order: [] };
}

function load() {
    if (cache) return cache;
    try {
        const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
        cache = raw && raw.v === VERSION && typeof raw === "object"
            ? { v: VERSION, shared: raw.shared || {}, graphs: raw.graphs || {},
                order: Array.isArray(raw.order) ? raw.order : [] }
            : empty();
    } catch {
        // Private mode, a full disk, or someone else's key in this origin.
        cache = empty();
    }
    return cache;
}

function scheduleWrite() {
    if (writeTimer) return;
    writeTimer = setTimeout(() => {
        writeTimer = null;
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(cache));
        } catch (err) {
            // Storage refused us. The sitting still works from the cache; only
            // the next launch forgets, which is worth a line but not a crash.
            console.warn("[TS Studio] settings were not saved", err);
        }
    }, WRITE_DELAY_MS);
}

function tooBig(value) {
    // Strings count too: a pasted wall of text is the likeliest way one value
    // grows past what the store should carry.
    if (value == null || (typeof value !== "object" && typeof value !== "string")) {
        return false;
    }
    try {
        return JSON.stringify(value).length > MAX_VALUE_BYTES;
    } catch {
        return true;                    // cyclic or otherwise unstorable
    }
}

function touchGraph(store, graphId) {
    store.order = store.order.filter((id) => id !== graphId);
    store.order.push(graphId);
    while (store.order.length > MAX_GRAPHS) {
        const dropped = store.order.shift();
        delete store.graphs[dropped];
    }
}

/**
 * Remember one control's value.
 *
 * @param {string} graphId Backend id, e.g. "krea2/inpaint".
 * @param {string} param Control parameter name.
 * @param {string} kind Control kind — decides the scope.
 * @param {*} value Whatever the control's get() returns.
 */
export function remember(graphId, param, kind, value) {
    if (!param || value === undefined) return;
    const store = load();
    const shared = SHARED_KINDS.has(kind);
    const bucket = shared ? store.shared : (store.graphs[graphId] ||= {});
    bucket[param] = value;
    if (!shared) touchGraph(store, graphId);
    if (tooBig(value)) {
        // Kept in the live cache for this sitting, dropped from what we write.
        const persisted = JSON.parse(JSON.stringify(cache));
        const target = shared ? persisted.shared : (persisted.graphs[graphId] || {});
        delete target[param];
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted)); }
        catch { /* see scheduleWrite */ }
        return;
    }
    scheduleWrite();
}

/**
 * What this control was left at, or undefined when it was never touched.
 *
 * A per-graph value wins over a shared one: the graph's own history is more
 * specific than anything carried across.
 */
export function recall(graphId, param, kind) {
    if (!param) return undefined;
    const store = load();
    const own = store.graphs[graphId];
    if (own && own[param] !== undefined) return own[param];
    if (SHARED_KINDS.has(kind) && store.shared[param] !== undefined) {
        return store.shared[param];
    }
    return undefined;
}

/** Everything remembered for one graph, merged with the shared values. */
export function recallAll(graphId) {
    const store = load();
    return { ...store.shared, ...(store.graphs[graphId] || {}) };
}

/** Forget one graph's values — used when a control set changes shape. */
export function forgetGraph(graphId) {
    const store = load();
    delete store.graphs[graphId];
    store.order = store.order.filter((id) => id !== graphId);
    scheduleWrite();
}

/** Forget everything. Offered in Settings, and useful in tests. */
export function forgetAll() {
    cache = empty();
    try { localStorage.removeItem(STORAGE_KEY); }
    catch { /* see scheduleWrite */ }
}

/** Write now rather than on the timer — for teardown paths. */
export function flush() {
    if (!writeTimer) return;
    clearTimeout(writeTimer);
    writeTimer = null;
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(cache)); }
    catch { /* see scheduleWrite */ }
}

/** For tests and for the Settings row that reports how much is kept. */
export function stats() {
    const store = load();
    return {
        graphs: Object.keys(store.graphs).length,
        shared: Object.keys(store.shared).length,
    };
}

export const SHARED_KINDS_FOR_TEST = SHARED_KINDS;
