// TS Studio kit — the one observable session store (core layer, no DOM).
//
// Architecture (doc/IMAGE_STUDIO_PLAN.md §3.5): all studio state lives here;
// mutations go through dispatch(action); UI subscribes selectively. The undo
// stack is a subscriber of this store, not a patch inside every handler.
// Core modules must stay DOM-free so they unit-test under Node.

/**
 * @template S
 * @param {S} initialState
 * @param {(state: S, action: {type: string}) => S} reducer Pure: returns a NEW
 *   object when anything changed (reference equality is the dirty check).
 */
export function createStore(initialState, reducer) {
    let state = initialState;
    let subscribers = [];

    function getState() {
        return state;
    }

    function dispatch(action) {
        const next = reducer(state, action);
        if (next === state) return;
        const prev = state;
        state = next;
        for (const sub of subscribers.slice()) {
            const slice = sub.selector(state);
            if (slice !== sub.last) {
                sub.last = slice;
                try {
                    sub.callback(slice, prev, action);
                } catch (err) {
                    console.error("[TS Studio] store subscriber failed", err);
                }
            }
        }
    }

    /**
     * @param {(state: S) => any} selector Return the same reference for "no
     *   change" — the callback fires only when the selected slice differs.
     * @returns {() => void} unsubscribe
     */
    function subscribe(selector, callback) {
        const sub = { selector, callback, last: selector(state) };
        subscribers.push(sub);
        return () => {
            subscribers = subscribers.filter((s) => s !== sub);
        };
    }

    return { getState, dispatch, subscribe };
}
