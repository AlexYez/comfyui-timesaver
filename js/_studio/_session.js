// TS Studio kit — session identity and result history (core layer, no DOM).
//
// A session is one node instance's body of work. Results are files under
// output/ts_studio/<session>/ written by the TS_StudioOutput marker; history
// survives a page reload by re-reading /history and keeping only entries
// whose images were saved under this session's prefix.

export function newSessionId() {
    const stamp = Date.now().toString(36);
    const noise = Math.random().toString(36).slice(2, 8);
    return `s${stamp}_${noise}`;
}

export function sessionPrefix(sessionId) {
    return `ts_studio/${sessionId}/result`;
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
    const marker = `ts_studio/${sessionId}/`;
    const found = [];
    for (const entry of Object.values(history)) {
        const stamp = entry?.status?.completed ? 1 : 0;
        if (!stamp) continue;
        for (const output of Object.values(entry.outputs || {})) {
            for (const image of output.images || []) {
                const folder = String(image.subfolder || "").replace(/\\/g, "/") + "/";
                if (folder.startsWith(marker)) {
                    found.push({ image, params: readRunParams(entry) });
                }
            }
        }
    }
    found.sort((a, b) => resultRelPath(a.image).localeCompare(resultRelPath(b.image)));
    return found;
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
