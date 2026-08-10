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
    // ⚠️ ALWAYS revalidate. ComfyUI serves these files with an ETag but no
    // Cache-Control, so the browser is free to reuse its copy without asking —
    // and the studio then runs a graph the repository no longer contains. That
    // is a silent wrong answer: the panel looks right, the numbers are last
    // week's. `no-cache` means "ask, with the ETag" — the server replies 304 for
    // an unchanged file, so the cost is one conditional request per backend.
    const response = await fetcher(url, { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
    return response.json();
}

/**
 * Load every built-in backend. Never throws for a single bad file — a broken
 * backend appears with available=false and its reasons, so the UI can show it
 * grey instead of vanishing it (plan §4).
 *
 * @param {(url: string, options?: object) => Promise<Response>} fetcher
 *   api.fetchApi-compatible. MUST forward its options — the manifests are
 *   requested with `cache: "no-cache"` (see fetchJson).
 * @param {object} objectInfo Full /object_info payload.
 */
export async function loadBackends(fetcher, objectInfo, apiFetcher = null) {
    const index = await fetchJson(fetcher, `${BUILTIN_BASE}/index.json`);
    const sources = (index.workflows || []).map((rel) => ({
        rel, url: `${BUILTIN_BASE}/${rel}`, tier: "builtin",
    }));
    // User tier: ComfyUI userdata under user/default/ts-studio/workflows.
    // A user file whose manifest id matches a built-in OVERRIDES it (plan §4).
    if (apiFetcher) {
        try {
            const res = await apiFetcher("/userdata?dir=ts-studio%2Fworkflows&recurse=true");
            if (res.ok) {
                for (const name of await res.json()) {
                    if (!String(name).endsWith(".json")) continue;
                    // An installed pack leaves a stamp beside its graphs; it is
                    // bookkeeping, not a backend, and would read as a broken one.
                    if (String(name).endsWith("pack.json")) continue;
                    const encoded = encodeURIComponent(`ts-studio/workflows/${name}`);
                    sources.push({ rel: `user:${name}`, tier: "user",
                                   url: `/userdata/${encoded}`, viaApi: true });
                }
            }
        } catch (err) {
            console.warn("[TS Studio] userdata listing failed", err);
        }
    }
    const backends = [];
    for (const source of sources) {
        const entry = { id: source.rel, tier: source.tier, manifest: null, graph: null,
                        spec: null, available: false, problems: [], modelFiles: {} };
        backends.push(entry);
        try {
            entry.graph = await fetchJson(source.viaApi ? apiFetcher : fetcher, source.url);
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
    // User override: same manifest id keeps only the user version.
    //
    // Перекрытие обязано быть слышным. Установленный набор молча заменяет
    // встроенный граф того же id, и студия начинает считать по нему — включая
    // случай, когда встроенный новее. Час замеров однажды ушёл именно сюда:
    // правки в графе не действовали, потому что работал не он.
    const byId = new Map();
    for (const backend of backends) {
        const key = backend.manifest?.id || backend.id;
        const existing = byId.get(key);
        if (!existing) { byId.set(key, backend); continue; }
        if (backend.tier !== "user") continue;
        backend.shadows = existing.id;
        console.warn(`[TS Studio] installed pack '${backend.id}' overrides the built-in `
            + `'${existing.id}' (${key}) — the studio runs the installed graph`);
        byId.set(key, backend);
    }
    return [...byId.values()];
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

/**
 * Семейства, которых в студии быть не должно, — и почему.
 *
 * Спрятать семейство могут две разные вещи, и путать их нельзя: выключенный
 * человеком пак предлагает вернуть переключатель, а пак выше потолка уровня
 * предлагает сам пак. Поэтому ответ — причина, а не «да/нет».
 *
 * Потолок существует потому, что вся сборка сейчас едет одним куском: без него
 * автор физически не может увидеть студию глазами бесплатного пользователя —
 * экран паков был бы честным, а сам список моделей нет.
 *
 * @param {Map} families результат groupByFamily
 * @param {{packs: object[], disabled: string[], viewTier: number|null}} state
 * @returns {{families: Map, hidden: object[]}} hidden — то, что убрано, в виде
 *          записей для серого списка: {family, label, modes, packId, why}
 */
export function applyPackState(families, state) {
    const disabled = new Set(state?.disabled || []);
    const ceiling = state?.viewTier;
    const kept = new Map(families);
    const hidden = [];
    for (const pack of state?.packs || []) {
        const tier = Number(pack.tier || 0);
        let why = "";
        if (disabled.has(pack.id)) why = "off";
        else if (ceiling !== null && ceiling !== undefined && tier > Number(ceiling)) why = "tier";
        if (!why) continue;
        for (const name of pack.families || []) {
            const key = typeof name === "string" ? name : name?.family;
            const family = kept.get(key);
            if (!family) continue;
            kept.delete(key);
            hidden.push({
                family: key,
                label: family.label,
                modes: [...family.modes.keys()],
                packId: pack.id,
                tier,
                why,
            });
        }
    }
    return { families: kept, hidden };
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
                // What a subscription pass must open for this family. 0 is
                // free; the studio never asks for a pass to run those.
                tier: Number(backend.manifest.tier || 0),
                modes: new Map(),
            });
        }
        const entry = families.get(family);
        // A family is as paid as its most restricted backend.
        entry.tier = Math.max(entry.tier, Number(backend.manifest.tier || 0));
        entry.modes.set(backend.manifest.mode, backend);
    }
    return families;
}
