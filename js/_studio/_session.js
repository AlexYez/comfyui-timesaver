// TS Studio kit — session identity and result history (core layer, no DOM).
//
// A session is one node instance's body of work. Results are written by the
// TS_StudioOutput marker into output/images/<model>/, named for the model and
// the moment — a tree a person can browse without the app. The session itself
// travels in the PNG snapshot (ts_studio), so the gallery can rebuild a
// sitting after a reload without owning the folder layout.

export function newSessionId() {
    const stamp = Date.now().toString(36);
    const noise = Math.random().toString(36).slice(2, 8);
    return `s${stamp}_${noise}`;
}

/**
 * Where one run's file goes: a folder per model, a name carrying the model
 * and the local date and time. Mirrors the layout the owner's own workflows
 * save into, so studio work sits beside everything else.
 *
 * @param {string} family Backend family id, e.g. "krea2".
 * @param {Date} [now] Injected in tests.
 */
export function outputPrefix(family, now = new Date()) {
    const pad = (n) => String(n).padStart(2, "0");
    const date = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
    const time = `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    // The family id comes from a pack manifest, so it is not ours to trust:
    // separators are folded away, and a name made only of dots would still be
    // a step up the tree.
    let model = String(family || "studio").replace(/[^\w.-]+/g, "-");
    if (!model || /^\.+$/.test(model)) model = "studio";
    return `images/${model}/${model}_${date}_${time}`;
}

export function resultRelPath(image) {
    const folder = String(image.subfolder || "").replace(/\\/g, "/");
    return folder ? `${folder}/${image.filename}` : image.filename;
}

export function resultViewUrl(image) {
    const params = new URLSearchParams({
        filename: image.filename,
        subfolder: image.subfolder || "",
        type: image.type || "output",
    });
    return `/view?${params}`;
}

/**
 * Rebuild this session's gallery from the server history (page reload path).
 *
 * @param {(url: string) => Promise<Response>} fetcher
 * @param {string} sessionId
 * @returns {Promise<{image: object, params: object|null}[]>} oldest first
 */
export async function restoreResults(fetcher, sessionId) {
    const response = await fetcher("/history?max_items=512");
    if (!response.ok) return [];
    const history = await response.json();
    const found = [];
    // Which sitting a picture belongs to is written in its run snapshot, not in
    // its path: the output tree is organised for a person (images/<model>/), so
    // the folder says nothing about the session. History comes back oldest
    // first, which is the order the gallery wants.
    for (const entry of Object.values(history)) {
        if (!entry?.status?.completed) continue;
        if (!sessionId || readSessionId(entry) !== sessionId) continue;
        for (const output of Object.values(entry.outputs || {})) {
            for (const image of output.images || []) {
                if ((image.type || "output") !== "output") continue;
                found.push({ image, params: readRunParams(entry) });
            }
        }
    }
    return found;
}

/** The session a history entry belongs to, or "" when it carries no snapshot. */
function readSessionId(historyEntry) {
    try {
        const snapshot = historyEntry.prompt?.[3]?.extra_pnginfo?.ts_studio;
        const state = typeof snapshot === "string" ? JSON.parse(snapshot) : snapshot;
        return String(state?.session || "");
    } catch {
        return "";
    }
}

// The parameters of a run are recoverable from the stored prompt: marker
// nodes carry their values. This is what makes "reuse seed/params" honest —
// it reads what actually ran, not what the UI remembers.
function readRunParams(historyEntry) {
    try {
        const prompt = historyEntry.prompt?.[2];
        if (!prompt) return null;
        const params = {};
        for (const node of Object.values(prompt)) {
            const name = node?.inputs?.param_name;
            if (typeof name === "string" && "value" in node.inputs) {
                params[name] = node.inputs.value;
            }
        }
        return params;
    } catch {
        return null;
    }
}
