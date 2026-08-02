// TS Studio kit — reproducibility from PNG metadata (core layer, no DOM).
//
// Every studio result already embeds the full API prompt (ComfyUI's standard
// tEXt "prompt" chunk, written by the save helper). This module reads it
// back: drop a studio PNG anywhere later — even from a file browser — and
// the exact backend, prompt, seed, sizes and LoRA chain are recoverable.
// Pure functions: blob in, plain objects out.

/** Parse PNG tEXt/iTXt chunks; returns {keyword: text}. Not a PNG? -> {}. */
export async function readPngText(blob) {
    const buffer = new Uint8Array(await blob.arrayBuffer());
    const out = {};
    const MAGIC = [0x89, 0x50, 0x4E, 0x47];
    if (buffer.length < 8 || MAGIC.some((b, i) => buffer[i] !== b)) return out;
    const view = new DataView(buffer.buffer);
    let offset = 8;
    const ascii = (start, end) => String.fromCharCode(...buffer.subarray(start, end));
    while (offset + 8 <= buffer.length) {
        const length = view.getUint32(offset);
        const type = ascii(offset + 4, offset + 8);
        const dataStart = offset + 8;
        if (type === "tEXt") {
            const data = buffer.subarray(dataStart, dataStart + length);
            const zero = data.indexOf(0);
            if (zero > 0) {
                const keyword = ascii(dataStart, dataStart + zero);
                out[keyword] = new TextDecoder("latin1").decode(data.subarray(zero + 1));
            }
        } else if (type === "iTXt") {
            const data = buffer.subarray(dataStart, dataStart + length);
            const zero = data.indexOf(0);
            if (zero > 0) {
                const keyword = ascii(dataStart, dataStart + zero);
                // keyword \0 compression flag+method \0 lang \0 translated \0 text
                let cursor = zero + 3;
                for (let fields = 0; fields < 2 && cursor < data.length; cursor += 1) {
                    if (data[cursor] === 0) fields += 1;
                }
                out[keyword] = new TextDecoder("utf-8").decode(data.subarray(cursor));
            }
        } else if (type === "IEND") {
            break;
        }
        offset = dataStart + length + 4;
    }
    return out;
}

/**
 * Recover the studio run from an embedded API prompt.
 *
 * @returns {null | {backendId: string, values: Record<string, any>,
 *                   loras: {name: string, strength: number}[]}}
 *   null when the prompt is not a studio backend (no manifest marker).
 */
export function extractStudioRun(promptJson) {
    let manifest = null;
    const values = {};
    const loras = [];
    for (const node of Object.values(promptJson || {})) {
        const inputs = node?.inputs || {};
        if (node.class_type === "TS_StudioManifest") {
            try {
                manifest = JSON.parse(inputs.manifest || "{}");
            } catch {
                manifest = null;
            }
        } else if (typeof inputs.param_name === "string" && "value" in inputs
                   && !Array.isArray(inputs.value)) {
            values[inputs.param_name] = inputs.value;
        } else if (node.class_type === "LoraLoaderModelOnly"
                   && typeof inputs.lora_name === "string") {
            loras.push({ name: inputs.lora_name,
                         strength: Number(inputs.strength_model ?? 1) });
        }
    }
    if (!manifest?.id) return null;
    return { backendId: manifest.id, mode: manifest.mode, family: manifest.family,
             values, loras };
}

/** Convenience: blob -> studio run (or null). */
export async function studioRunFromPng(blob) {
    const text = await readPngText(blob);
    if (!text.prompt) return null;
    try {
        return extractStudioRun(JSON.parse(text.prompt));
    } catch {
        return null;
    }
}
