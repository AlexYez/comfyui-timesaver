// TS Studio kit — marker discovery and prompt patching (core layer, no DOM).
//
// A backend workflow is an API-format prompt JSON whose entry/exit points are
// the pack's marker nodes (nodes/image/studio/markers/). This module finds
// them, validates the file and produces a patched copy ready for /prompt.
// Pure functions over plain objects — unit-testable under Node.

const INPUT_MARKERS = new Set([
    "TS_StudioInputText",
    "TS_StudioInputNumber",
    "TS_StudioInputSeed",
    "TS_StudioInputImage",
    "TS_StudioInputMask",
]);

/** Parsed view of one backend file. Throws with a readable reason on defects. */
export function inspectBackend(graph) {
    const params = new Map();
    const models = [];
    let output = null;
    let manifest = null;
    let loraStack = null;

    for (const [nodeId, node] of Object.entries(graph)) {
        const cls = node.class_type;
        const inputs = node.inputs || {};
        if (INPUT_MARKERS.has(cls)) {
            const name = String(inputs.param_name || "").trim();
            if (!name) throw new Error(`marker ${nodeId} (${cls}) has an empty param_name`);
            if (params.has(name)) throw new Error(`duplicate param_name '${name}' (${nodeId})`);
            params.set(name, { nodeId, cls, label: String(inputs.label || "") });
        } else if (cls === "TS_StudioOutput") {
            if (output) throw new Error("more than one TS_StudioOutput");
            output = nodeId;
        } else if (cls === "TS_StudioManifest") {
            if (manifest) throw new Error("more than one TS_StudioManifest");
            manifest = JSON.parse(inputs.manifest || "{}");
        } else if (cls === "TS_StudioLoraStack") {
            if (loraStack) throw new Error("more than one TS_StudioLoraStack");
            loraStack = nodeId;
        }
        const title = node._meta?.title;
        if (title && title.startsWith("studio:")) {
            models.push({ title, nodeId, cls });
        }
    }
    if (!output) throw new Error("no TS_StudioOutput marker");
    if (!manifest) throw new Error("no TS_StudioManifest marker");
    if (!manifest.id || !manifest.family || !manifest.mode) {
        throw new Error("manifest needs id, family and mode");
    }
    // manifest.literals: {param: {node|title, input}} — values written straight
    // into a named node input (booleans, combos), no marker node required.
    const literals = new Map(Object.entries(manifest.literals || {}));
    return { params, models, output, manifest, loraStack, literals };
}

/**
 * Patched deep copy of a backend graph, ready to submit.
 *
 * @param {object} graph API-format prompt JSON (left untouched).
 * @param {object} spec  inspectBackend(graph) result.
 * @param {object} run
 * @param {Record<string, string|number>} [run.values] param_name -> value.
 * @param {Record<string, string>} [run.modelFiles] "studio:model" -> combo value.
 * @param {{name: string, strength: number}[]} [run.loras] Chain order = array order.
 * @param {string} [run.filenamePrefix] Session save prefix for the output marker.
 * @param {string[]} [run.dropParams] Optional image params left unset: their
 *   marker AND everything that only feeds through it is removed, so optional
 *   reference branches disappear instead of failing on an empty filename.
 * @param {(cls: string, input: string) => boolean} [run.isOptionalInput]
 *   Whether a class input is optional (from /object_info). A consumer of a
 *   dropped branch survives when the link sat on an optional input (the link
 *   is simply removed); a required input dooms the consumer too.
 */
export function patchBackend(graph, spec, run) {
    const out = structuredClone(graph);

    for (const [name, value] of Object.entries(run.values || {})) {
        const literal = spec.literals?.get(name);
        if (literal) {
            const targetId = literal.node
                || spec.models.find((m) => m.title === literal.title)?.nodeId;
            if (targetId && out[targetId]) out[targetId].inputs[literal.input] = value;
            continue;
        }
        const param = spec.params.get(name);
        if (!param) throw new Error(`backend has no param '${name}'`);
        out[param.nodeId].inputs.value = value;
    }

    for (const [title, file] of Object.entries(run.modelFiles || {})) {
        const model = spec.models.find((m) => m.title === title);
        if (!model) continue;
        const inputs = out[model.nodeId].inputs;
        for (const key of ["unet_name", "clip_name", "vae_name", "ckpt_name", "model_name"]) {
            if (key in inputs) {
                inputs[key] = file;
                break;
            }
        }
    }

    if (run.filenamePrefix) {
        out[spec.output].inputs.filename_prefix = run.filenamePrefix;
    }

    expandLoraStack(out, spec, run.loras || []);
    const dropRoots = (run.dropParams || [])
        .map((name) => spec.params.get(name)?.nodeId)
        .filter(Boolean);
    if (dropRoots.length) {
        removeBranches(out, dropRoots, run.isOptionalInput || (() => false));
    }
    return out;
}

// The stack marker is MODEL-only passthrough; the user's stack becomes a chain
// of native LoraLoaderModelOnly nodes spliced in front of it. The marker node
// itself stays (a passthrough at execution time), so downstream links and the
// empty-stack case need no rewiring at all.
function expandLoraStack(graph, spec, loras) {
    if (!spec.loraStack || !loras.length) return;
    const marker = graph[spec.loraStack];
    let upstream = marker.inputs.model;
    loras.forEach((lora, index) => {
        const id = `ts_lora_${index}`;
        graph[id] = {
            class_type: "LoraLoaderModelOnly",
            inputs: {
                model: upstream,
                lora_name: lora.name,
                strength_model: Number(lora.strength),
            },
        };
        upstream = [id, 0];
    });
    marker.inputs.model = upstream;
}

// Remove unset optional branches. Two directions, in order:
//
// DOWNSTREAM: a consumer of a removed node cannot execute without that input.
// If the link sits on an OPTIONAL input (per /object_info), the link is
// removed and the consumer lives — this is how Qwen-Edit-style multi-image
// encoders keep working with 1 of 3 references. A link on a REQUIRED input
// dooms the consumer, and the wave continues from it.
//
// UPSTREAM: after the wave, ancestors whose every consumer died (a private
// VAEEncode of the dropped reference) are garbage-collected; shared nodes
// (the family's VAE loader) keep their other consumers and stay.
function removeBranches(graph, rootIds, isOptionalInput) {
    const doomed = new Set(rootIds);
    const queue = [...rootIds];
    while (queue.length) {
        const dead = queue.shift();
        for (const [nodeId, node] of Object.entries(graph)) {
            if (doomed.has(nodeId)) continue;
            for (const [inputName, value] of Object.entries(node.inputs || {})) {
                if (!Array.isArray(value) || value[0] !== dead) continue;
                if (isOptionalInput(node.class_type, inputName)) {
                    delete node.inputs[inputName];
                } else if (!doomed.has(nodeId)) {
                    doomed.add(nodeId);
                    queue.push(nodeId);
                }
            }
        }
    }
    // Mark-and-sweep from the graph's real roots: whatever the surviving
    // output nodes and the manifest cannot reach upstream is orphaned
    // scaffolding of the dropped branch.
    const alive = new Set();
    const stack = Object.entries(graph)
        .filter(([nodeId, node]) => !doomed.has(nodeId)
            && (node.class_type === "TS_StudioOutput" || node.class_type === "TS_StudioManifest"))
        .map(([nodeId]) => nodeId);
    while (stack.length) {
        const nodeId = stack.pop();
        if (alive.has(nodeId) || doomed.has(nodeId)) continue;
        alive.add(nodeId);
        for (const value of Object.values(graph[nodeId].inputs || {})) {
            if (Array.isArray(value)) stack.push(value[0]);
        }
    }
    for (const nodeId of Object.keys(graph)) {
        if (!alive.has(nodeId)) delete graph[nodeId];
    }
}

/** Model files named by the graph that a live combo listing does not offer. */
export function missingModelValues(graph, objectInfo) {
    const missing = [];
    for (const [nodeId, node] of Object.entries(graph)) {
        const spec = objectInfo[node.class_type];
        if (!spec) {
            missing.push({ nodeId, kind: "node", value: node.class_type });
            continue;
        }
        const declared = { ...(spec.input?.required || {}), ...(spec.input?.optional || {}) };
        for (const [name, value] of Object.entries(node.inputs || {})) {
            const spec_item = declared[name];
            const options = Array.isArray(spec_item?.[0]) ? spec_item[0]
                : Array.isArray(spec_item?.[1]?.options) ? spec_item[1].options : null;
            if (options && typeof value === "string" && !options.includes(value)) {
                missing.push({ nodeId, kind: "value", input: name, value });
            }
        }
    }
    return missing;
}

/** First combo option matching the manifest's regex, or null. */
export function resolveModelFile(options, matchPattern) {
    if (!matchPattern) return null;
    const re = new RegExp(matchPattern, "i");
    return options.find((option) => re.test(option.replace(/\\/g, "/"))) ?? null;
}
