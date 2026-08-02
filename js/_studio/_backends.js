// TS Studio kit — backend registry (core layer, no DOM).
//
// Loads the built-in workflow files shipped under WEB_DIRECTORY, inspects
// their markers, resolves model files against the live /object_info and
// groups everything by family for the UI. A user-tier (userdata) merge joins
// in phase 4: same id overrides built-in.

import { inspectBackend, resolveModelFile } from "./_markers.js";

const BUILTIN_BASE = "/extensions/comfyui-timesaver/image/studio/workflows";

/**
 * @typedef {object} Backend
 * @property {string} id
 * @property {object} manifest
 * @property {object} graph      Pristine API-format JSON.
 * @property {object} spec       inspectBackend result.
 * @property {boolean} available
 * @property {string[]} problems Human-readable reasons when not available.
 * @property {Record<string, string>} modelFiles Resolved "studio:*" -> combo value.
 */

async function fetchJson(fetcher, url) {
    const response = await fetcher(url);
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
    return response.json();
}

/**
 * Load every built-in backend. Never throws for a single bad file — a broken
 * backend appears with available=false and its reasons, so the UI can show it
 * grey instead of vanishing it (plan §4).
 *
 * @param {(url: string) => Promise<Response>} fetcher api.fetchApi-compatible.
 * @param {object} objectInfo Full /object_info payload.
 */
export async function loadBackends(fetcher, objectInfo) {
    const index = await fetchJson(fetcher, `${BUILTIN_BASE}/index.json`);
    const backends = [];
    for (const rel of index.workflows || []) {
        const entry = { id: rel, manifest: null, graph: null, spec: null,
                        available: false, problems: [], modelFiles: {} };
        backends.push(entry);
        try {
            entry.graph = await fetchJson(fetcher, `${BUILTIN_BASE}/${rel}`);
            entry.spec = inspectBackend(entry.graph);
            entry.manifest = entry.spec.manifest;
            entry.id = entry.manifest.id;
        } catch (err) {
            entry.problems.push(String(err?.message || err));
            continue;
        }
        validateAgainstServer(entry, objectInfo);
        entry.available = entry.problems.length === 0;
    }
    return backends;
}

function comboOptionsFor(objectInfo, cls, inputName) {
    const spec = objectInfo[cls];
    if (!spec) return null;
    const declared = { ...(spec.input?.required || {}), ...(spec.input?.optional || {}) };
    const decl = declared[inputName];
    if (Array.isArray(decl?.[0])) return decl[0];
    if (Array.isArray(decl?.[1]?.options)) return decl[1].options;
    return null;
}

function validateAgainstServer(entry, objectInfo) {
    for (const [nodeId, node] of Object.entries(entry.graph)) {
        if (!objectInfo[node.class_type]) {
            const dep = (entry.manifest.dependencies || [])
                .find((d) => (d.nodes || []).includes(node.class_type));
            entry.problems.push(dep
                ? `needs the '${dep.pack}' pack (${node.class_type})`
                : `node ${node.class_type} (${nodeId}) is not installed`);
        }
    }
    for (const model of entry.manifest.models || []) {
        const marker = entry.spec.models.find((m) => m.title === model.title);
        if (!marker) {
            entry.problems.push(`manifest names '${model.title}' but no node carries that title`);
            continue;
        }
        const node = entry.graph[marker.nodeId];
        const options = comboOptionsFor(objectInfo, node.class_type, model.input);
        if (!options) continue; // node missing: already reported above
        const current = node.inputs[model.input];
        if (typeof current === "string" && options.includes(current)) {
            entry.modelFiles[model.title] = current;
            continue;
        }
        const resolved = resolveModelFile(options, model.match);
        if (resolved) {
            entry.modelFiles[model.title] = resolved;
        } else {
            entry.problems.push(`no installed file matches '${model.match}' for ${model.title}`);
        }
    }
}

/** Group available backends by family for the model switcher. */
export function groupByFamily(backends) {
    const families = new Map();
    for (const backend of backends) {
        const family = backend.manifest?.family;
        if (!family) continue;
        if (!families.has(family)) {
            families.set(family, {
                family,
                label: backend.manifest.family_label || family,
                modes: new Map(),
            });
        }
        families.get(family).modes.set(backend.manifest.mode, backend);
    }
    return families;
}
