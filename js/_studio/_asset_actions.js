// Asset actions — a registry that lets any asset browser offer commands this
// pack provides, without either side importing the other.
//
// The problem it solves: the Artius browser owns the right-click menu over a
// user's images; the studio owns the ability to rebuild the session an image
// was made in. Wiring one into the other's internals would mean every update
// on either side risks breaking the other. So the studio publishes an action,
// the browser lists whatever actions are published, and neither has to know
// what the other is.
//
// The registry is a plain array on `window` on purpose: publisher and consumer
// are separate ComfyUI extensions with no shared module graph, and an array is
// the one contract that survives both being loaded in either order.
//
// Contract for one action:
//
//   {
//     id:      "ts-image-studio.recreate",         // unique, stable
//     label:   { en: "…", ru: "…" } | "…",         // menu text
//     order:   20,                                 // lower sorts first
//     supports(asset) -> boolean,                  // fast, synchronous
//     run(asset) -> Promise<{ok, message?}>|void,  // does the work
//   }
//
// The `asset` a browser passes in:
//
//   { id, filename, url, type, extension }
//
// `url` must be fetchable as-is. `supports` is called while a menu is being
// built, so it stays synchronous and cheap — anything that needs the file
// itself belongs in `run`.

const REGISTRY_KEY = "tsAssetActions";

/** The shared registry, created on first use. */
function registry() {
    if (!Array.isArray(window[REGISTRY_KEY])) window[REGISTRY_KEY] = [];
    return window[REGISTRY_KEY];
}

/**
 * Publish an action for asset browsers to offer.
 *
 * Re-publishing the same id replaces the previous entry: an extension that
 * reloads must not leave a stale closure behind in the menu.
 *
 * @param {object} action See the contract above.
 * @returns {() => void} Removes the action again.
 */
export function publishAssetAction(action) {
    if (!action?.id || typeof action.run !== "function") {
        console.warn("[TS Asset Actions] ignored an action without an id or run()");
        return () => {};
    }
    const list = registry();
    const existing = list.findIndex((entry) => entry?.id === action.id);
    if (existing >= 0) list.splice(existing, 1);
    list.push(action);
    list.sort((a, b) => (a?.order ?? 50) - (b?.order ?? 50));
    return () => {
        const index = registry().findIndex((entry) => entry?.id === action.id);
        if (index >= 0) registry().splice(index, 1);
    };
}

/**
 * Actions that apply to one asset — the call an asset browser makes.
 *
 * A throwing `supports` drops that action rather than the menu: a broken
 * third-party entry must not cost the user their right-click.
 *
 * @param {object} asset {id, filename, url, type, extension}
 * @returns {object[]} Applicable actions, in display order.
 */
export function assetActionsFor(asset) {
    return registry().filter((action) => {
        if (typeof action?.run !== "function") return false;
        if (typeof action.supports !== "function") return true;
        try {
            return Boolean(action.supports(asset));
        } catch (err) {
            console.warn(`[TS Asset Actions] '${action.id}' supports() failed`, err);
            return false;
        }
    });
}

/**
 * Menu text for an action in the given locale, falling back to English and
 * then to the id — a label is never allowed to render as "undefined".
 *
 * @param {object} action
 * @param {string} locale "en" | "ru"
 */
export function assetActionLabel(action, locale) {
    const label = action?.label;
    if (typeof label === "string") return label;
    return label?.[locale] || label?.en || action?.id || "";
}
