// TS Studio kit — reproducibility from PNG metadata (core layer, no DOM).
//
// A studio result carries two independent records, and this module owns both
// ends of the trip:
//
//   • `ts_studio` — the studio's OWN snapshot chunk (see buildStudioState).
//     It holds everything needed to recreate the session: rail tab, backend,
//     every control value, the LoRA chain, the styles and the annotated names
//     of the source image / mask / references. Written through the output
//     marker's extra_pnginfo, so it sits BESIDE ComfyUI's `prompt` and
//     `workflow` chunks rather than competing with them — ComfyUI keeps
//     reading its metadata and image browsers keep reading theirs.
//   • the standard `prompt` graph — the fallback for images made before the
//     snapshot existed, read back through the marker nodes.
//
// Pure functions: blob in, plain objects out.

export const STUDIO_STATE_CHUNK = "ts_studio";
export const STUDIO_STATE_VERSION = 1;

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

/**
 * Snapshot of everything the studio needs to rebuild a session.
 *
 * Deliberately flat and self-describing: a future studio (video, audio) or a
 * later schema reads `v` and decides. Values are primitives and plain arrays
 * only — no DOM handles, no blobs, no absolute paths. Image references travel
 * as ComfyUI annotated names ("sub/name.png [input]"), which resolve through
 * /view on any install.
 */
export function buildStudioState(run) {
    const state = {
        v: STUDIO_STATE_VERSION,
        app: "ts-image-studio",
        backend: run.backendId,
        family: run.family,
        family_label: run.familyLabel || run.family,
        mode: run.mode,
        ui_mode: run.uiMode || run.mode,
        // Which sitting produced this. The gallery restores a session from
        // this rather than from the folder, which leaves the output tree free
        // to be organised for a person: output/images/<model>/.
        session: run.sessionId || "",
        values: {},
        loras: (run.loras || []).map((l) => ({ name: l.name, strength: Number(l.strength) })),
        styles: (run.styles || []).map((s) => String(s)),
        sources: {},
    };
    for (const [key, value] of Object.entries(run.values || {})) {
        if (value === null || value === undefined) continue;
        if (typeof value === "object") continue;
        state.values[key] = value;
    }
    for (const [key, value] of Object.entries(run.sources || {})) {
        if (typeof value === "string" && value) state.sources[key] = value;
    }
    if (run.size?.aspect) state.size = { aspect: run.size.aspect, mp: Number(run.size.mp) };
    return state;
}

/** True when the object looks like a snapshot this build can apply. */
export function isStudioState(value) {
    return Boolean(value && typeof value === "object"
        && value.app === "ts-image-studio"
        && Number(value.v) >= 1 && Number(value.v) <= STUDIO_STATE_VERSION
        && value.backend);
}

/**
 * Read a studio session out of a PNG: the snapshot chunk when present, the
 * prompt graph otherwise.
 *
 * @returns {Promise<null | {source: "snapshot"|"prompt", state: object}>}
 */
export async function studioStateFromPng(blob) {
    const text = await readPngText(blob);
    const raw = text[STUDIO_STATE_CHUNK];
    if (raw) {
        try {
            const parsed = JSON.parse(raw);
            if (isStudioState(parsed)) return { source: "snapshot", state: parsed };
        } catch {
            // Fall through to the prompt graph — a damaged chunk is not fatal.
        }
    }
    if (!text.prompt) return null;
    let legacy = null;
    try {
        legacy = extractStudioRun(JSON.parse(text.prompt));
    } catch {
        return null;
    }
    if (!legacy) return null;
    return {
        source: "prompt",
        state: buildStudioState({
            backendId: legacy.backendId,
            family: legacy.family,
            mode: legacy.mode,
            values: legacy.values,
            loras: legacy.loras,
        }),
    };
}

/**
 * Промпт, записанный в картинке. Первое найденное — оно и есть.
 *
 * Порядок опроса — от самого точного к самому общему:
 *
 *   1. `ts_studio` — снимок самой студии: ровно то, что человек написал.
 *   2. `prompt` — граф ComfyUI: берётся текст той ноды, которая кормит
 *      положительное обусловливание. Отрицательный промпт сюда попасть не
 *      должен, поэтому узлы со словом `negative` в связях отбрасываются.
 *   3. `parameters` — формат A1111: первая строка до «Negative prompt:».
 *
 * ⚠️ Пусто — это законный ответ. Картинка могла быть снята телефоном, и
 * выдумывать за неё промпт нельзя: в пачке это обернулось бы десятком кадров
 * не о том.
 *
 * @param {Record<string, string>} text чанки из `readPngText`
 * @returns {string} промпт или пустая строка
 */
export function promptFromPngText(text = {}) {
    const studio = text[STUDIO_STATE_CHUNK];
    if (studio) {
        try {
            const state = JSON.parse(studio);
            const found = state?.values?.prompt ?? state?.prompt;
            if (typeof found === "string" && found.trim()) return found.trim();
        } catch { /* чанк битый — идём дальше */ }
    }

    if (text.prompt) {
        try {
            const graph = JSON.parse(text.prompt);
            const found = positivePromptFromGraph(graph);
            if (found) return found;
        } catch { /* не JSON — идём дальше */ }
    }

    if (typeof text.parameters === "string" && text.parameters.trim()) {
        const [positive] = text.parameters.split(/\r?\n?Negative prompt:/);
        if (positive.trim()) return positive.trim();
    }
    return "";
}

/**
 * Положительный промпт из графа ComfyUI.
 *
 * Как отличить его от отрицательного без запуска графа: смотрим, во ЧТО узел
 * включён. Сэмплеры называют свои входы `positive` и `negative`, и этого
 * достаточно. Если сэмплера в графе нет (чужая сборка), берём первый текст —
 * ошибиться тут дешевле, чем не найти ничего.
 *
 * @param {object} graph граф в формате API
 * @returns {string}
 */
export function positivePromptFromGraph(graph) {
    const nodes = Object.entries(graph || {});
    const positive = new Set();
    const negative = new Set();
    for (const [, node] of nodes) {
        for (const [name, link] of Object.entries(node?.inputs || {})) {
            if (!Array.isArray(link) || typeof link[0] !== "string") continue;
            if (/negative/i.test(name)) negative.add(link[0]);
            else if (/positive|conditioning/i.test(name)) positive.add(link[0]);
        }
    }
    const textOf = (id) => {
        const value = graph?.[id]?.inputs?.text;
        return typeof value === "string" ? value.trim() : "";
    };
    for (const id of positive) {
        if (negative.has(id)) continue;
        const text = textOf(id);
        if (text) return text;
    }
    // Сэмплера не нашлось: берём первый текстовый энкодер, не помеченный
    // отрицательным.
    for (const [id, node] of nodes) {
        if (!/CLIPTextEncode|TextEncode/i.test(String(node?.class_type || ""))) continue;
        if (negative.has(id)) continue;
        const text = textOf(id);
        if (text) return text;
    }
    return "";
}

/** Удобство: картинка → промпт. */
export async function promptFromPng(blob) {
    return promptFromPngText(await readPngText(blob));
}
