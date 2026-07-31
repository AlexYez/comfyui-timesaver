// TS Files Downloader (Ultimate) — "Get models from workflow" button.
//
// Fills the node's `file_list` with every model the CURRENT workflow needs, so a
// graph handed to someone else pulls its own weights on the first run.
//
// Two questions, two different sources:
//
//   WHERE does it go?  The loader's own widget value, always. It stores a path
//      relative to its category folder, and that path carries the subfolder the
//      user filed the model under (`qwen/qwen_image_vae.safetensors` -> the
//      loader reads `models/vae/qwen/`). Template metadata knows nothing about
//      that, so a download aimed at the bare `vae/` stays invisible to the
//      loader forever. The widget therefore OVERRIDES every other source.
//
//   WHERE do I get it?  `node.properties.models` first — ComfyUI stamps
//      {name, url, directory} onto each loader that came from a template — then
//      a MarkdownNote's `[file](url)` list to fill gaps and cross-check.
//
// Both are matched to the loader by FILE NAME, never by full path, and template
// metadata goes stale: a user who swaps in their own model leaves the old
// {name, url} behind, so an entry no loader selects is flagged, not trusted.
//
// Separators come from whichever OS saved the graph (`a\b` on Windows, `a/b`
// elsewhere, sometimes mixed), so every path is normalised to POSIX before it
// is compared or written.
//
// CRITICAL: loaders usually live INSIDE subgraphs. Walking the root node list
// finds nothing on a modern template — the scan must descend into
// `definitions.subgraphs[].nodes`. In the Anima LLLite template that ships with
// ComfyUI, 5 of 5 models sit one level down and the root holds none.
//
// The button is frontend-only and marked non-serialising on purpose:
// `widgets_values` is a POSITIONAL array, so a serialising widget would shift
// all 11 saved values of every workflow that already contains this node
// (CLAUDE.md §12.5, project_memory/reference_widgets_values_positional.md).

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import { TS_UI_CLASS, ensureThemeStyles, pickLocaleStrings } from "../_theme.js";
import { openFullscreenOverlay } from "../_fullscreen.js";

const NODE_TYPE = "TS Files Downloader";
const FILE_LIST_WIDGET = "file_list";
const STYLE_ID = "ts-files-downloader-styles";

// Extensions that identify a model file rather than an image or a config.
const MODEL_EXT = /\.(safetensors|ckpt|pt|pth|bin|gguf|onnx|sft|safetensor)$/i;

// Loader node type -> the models/ subfolder its file belongs in. Used when an
// entry carries a URL but no directory, and to name the folder for models that
// were found by filename alone. Derived from ComfyUI's own folder registry
// names, which is exactly what the node's target resolver accepts.
const TYPE_TO_FOLDER = {
    CheckpointLoader: "checkpoints",
    CheckpointLoaderSimple: "checkpoints",
    unCLIPCheckpointLoader: "checkpoints",
    ImageOnlyCheckpointLoader: "checkpoints",
    UNETLoader: "diffusion_models",
    DiffusionModelLoader: "diffusion_models",
    CLIPLoader: "text_encoders",
    DualCLIPLoader: "text_encoders",
    TripleCLIPLoader: "text_encoders",
    QuadrupleCLIPLoader: "text_encoders",
    VAELoader: "vae",
    LoraLoader: "loras",
    LoraLoaderModelOnly: "loras",
    ControlNetLoader: "controlnet",
    DiffControlNetLoader: "controlnet",
    ModelPatchLoader: "model_patches",
    CLIPVisionLoader: "clip_vision",
    StyleModelLoader: "style_models",
    GLIGENLoader: "gligen",
    UpscaleModelLoader: "upscale_models",
    PhotoMakerLoader: "photomaker",
    HypernetworkLoader: "hypernetworks",
};

const STRINGS = {
    en: {
        button: "Get models from workflow",
        title: "Models found in this workflow",
        summary: (add, total) => `${add} to add · ${total} found in total`,
        colAdd: "Will be added",
        colPresent: "Already in the list",
        colMismatch: "Pointing at the wrong folder",
        colNoLink: "No download link",
        colOrphan: "Left over from a template",
        srcGraph: "node",
        srcNote: "note",
        srcBoth: "node + note",
        srcWidget: "filename only",
        srcUnused: "no loader uses it",
        merge: "Append",
        fix: "Fix folders",
        replace: "Replace list",
        cancel: "Cancel",
        close: "Close (Esc)",
        empty: "No models found in this workflow.",
        emptyHint:
            "Model links live in each loader's properties, or in a Markdown note. " +
            "This workflow has neither.",
        noLinkHint:
            "The workflow uses these but named no URL. Search Hugging Face for them, " +
            "or add the links by hand.",
        appended: (n) => `Appended ${n} line(s) to file_list.`,
        fixedPaths: (n) => `Corrected the folder on ${n} line(s).`,
        mismatchHint:
            "The list points these at a folder the loader does not read from, so they " +
            "would download and still count as missing.",
        orphanHint:
            "Named by a loader's stored metadata, but no node in the graph uses the file. " +
            "Usually a model that was swapped out. Not offered for download.",
        replaced: (n) => `file_list replaced with ${n} line(s).`,
        nothing: "Nothing to add — every model is already in the list.",
        hfSearch: "Find on Hugging Face",
        hfSearching: "Searching Hugging Face...",
        hfNothing: "No repository on Hugging Face contains these files under that exact name.",
        hfFound: (n) => `Found ${n} of them — adding to the list.`,
        hfFailed: (m) => `Search failed: ${m}`,
    },
    ru: {
        button: "Взять модели из workflow",
        title: "Модели, найденные в этом workflow",
        summary: (add, total) => `${add} к добавлению · всего найдено ${total}`,
        colAdd: "Будут добавлены",
        colPresent: "Уже в списке",
        colMismatch: "Указана не та папка",
        colNoLink: "Ссылка не найдена",
        colOrphan: "Осталось от шаблона",
        srcGraph: "нода",
        srcNote: "заметка",
        srcBoth: "нода + заметка",
        srcWidget: "только имя файла",
        srcUnused: "ни один лоудер не использует",
        merge: "Дополнить",
        fix: "Исправить папки",
        replace: "Заменить список",
        cancel: "Отмена",
        close: "Закрыть (Esc)",
        empty: "В этом workflow моделей не найдено.",
        emptyHint:
            "Ссылки на модели лежат в свойствах нод-загрузчиков либо в Markdown-заметке. " +
            "Здесь нет ни того, ни другого.",
        noLinkHint:
            "Эти модели workflow использует, но ссылок не назвал. Найдите их на Hugging Face " +
            "или добавьте ссылки вручную.",
        appended: (n) => `В file_list дописано строк: ${n}.`,
        fixedPaths: (n) => `Папка исправлена в строках: ${n}.`,
        mismatchHint:
            "В списке они нацелены не в ту папку, из которой читает лоудер, — скачаются, " +
            "но всё равно будут считаться отсутствующими.",
        orphanHint:
            "Записаны в метаданных лоудера, но файл не использует ни одна нода графа. " +
            "Обычно это подменённая модель. К скачиванию не предлагается.",
        replaced: (n) => `file_list заменён, строк: ${n}.`,
        nothing: "Добавлять нечего — все модели уже в списке.",
        hfSearch: "Найти на Hugging Face",
        hfSearching: "Ищем на Hugging Face...",
        hfNothing: "Ни в одном репозитории Hugging Face нет файлов с такими именами.",
        hfFound: (n) => `Найдено: ${n} — добавляем в список.`,
        hfFailed: (m) => `Поиск не удался: ${m}`,
    },
};

/* ------------------------------------------------------------------ scanning */

/** Collect every node of a serialised graph, descending into subgraph defs. */
function collectNodes(container, out = [], seen = new Set()) {
    if (!container || typeof container !== "object" || seen.has(container)) return out;
    seen.add(container);
    for (const node of container.nodes || []) {
        if (node && typeof node === "object") out.push(node);
    }
    for (const subgraph of container.definitions?.subgraphs || []) {
        collectNodes(subgraph, out, seen);
    }
    return out;
}

/** Filename part of a URL, without query or fragment. */
function fileNameFromUrl(url) {
    const raw = String(url || "");
    let path = raw;
    try {
        path = new URL(raw).pathname;
    } catch {
        path = raw.split(/[?#]/)[0];
    }
    const last = path.split("/").filter(Boolean).pop() || "";
    try {
        return decodeURIComponent(last);
    } catch {
        return last;
    }
}

function keyOf(name) {
    return String(name || "").trim().toLowerCase();
}

/**
 * Split a loader's stored value into its file name and its subfolder.
 *
 * Loaders store a path RELATIVE to their category folder, and users routinely
 * organise models into subfolders: `qwen\qwen_image_vae.safetensors` means
 * `models/vae/qwen/qwen_image_vae.safetensors`. Dropping the `qwen/` part
 * downloads the file next to the folder the loader actually reads from, so it
 * stays "missing" no matter how many times it is fetched.
 */
// A workflow is an untrusted document: it is shared, downloaded and opened from
// strangers. The paths stored in it may therefore name a folder BELOW a model
// directory and nothing else. These segments do not name a folder — they name a
// LOCATION, and the backend would expand them into an absolute path outside
// models/: a drive letter (`C:`), a home reference (`~`) or an environment
// variable (`%APPDATA%`, `$HOME`, `${HOME}`). One such segment poisons the whole
// value, so the target it came from is dropped rather than partially cleaned.
const LOCATION_SEGMENT = /:|^~|%[^%]+%|\$\{?[A-Za-z_]/;

/** Split a path into safe folder segments, or null when it names a location. */
function safeSegments(value) {
    const parts = String(value || "")
        .replace(/\\/g, "/")
        .split("/")
        .map((part) => part.trim())
        .filter((part) => part && part !== "." && part !== "..");
    for (const part of parts) {
        if (LOCATION_SEGMENT.test(part)) {
            console.warn(`[TS FilesDownloader] ignoring non-relative target from the workflow: ${value}`);
            return null;
        }
    }
    return parts;
}

function splitRelativePath(value) {
    // A workflow saved on Windows stores `krea2\file.safetensors`, the same
    // graph saved on macOS or Linux stores `krea2/file.safetensors`, and a
    // hand-edited one can mix both. Normalise to POSIX and drop the noise
    // segments so the two spellings produce an identical result.
    const parts = String(value || "")
        .replace(/\\/g, "/")
        .split("/")
        .map((part) => part.trim())
        .filter((part) => part && part !== "." && part !== "..");
    // The file name is kept verbatim — it identifies the model and is only ever
    // matched, never written (the backend derives the saved name itself). Only
    // the subfolder, which DOES become part of the download target, is vetted.
    const base = parts.pop() || "";
    const sub = safeSegments(parts.join("/"));
    return { base, sub: sub ? sub.join("/") : "" };
}

/**
 * Join a category folder with an optional subfolder, always POSIX-style.
 * `file_list` is read by people and by a backend that normalises separators
 * itself, so forward slashes travel between operating systems unchanged.
 */
function joinTarget(folder, sub) {
    // `folder` may itself come from the workflow (a template's `directory`), so
    // both halves are vetted. A poisoned folder yields an empty target, which
    // classify() reports as "no download link" instead of downloading anywhere.
    const folderParts = safeSegments(folder);
    const subParts = safeSegments(sub);
    if (folderParts === null) return "";
    return [...folderParts, ...(subParts || [])].join("/");
}

// ComfyUI reads two directories for some categories and keeps the old name
// working (folder_paths.map_legacy): `models/clip` and `models/text_encoders`
// are both searched for a text encoder, `models/unet` and
// `models/diffusion_models` for a UNET. A list aimed at either spelling works,
// so the two must not be reported as a disagreement.
const FOLDER_ALIASES = {
    unet: "diffusion_models",
    clip: "text_encoders",
    t2i_adapter: "controlnet",
};

/** Canonical form of a target folder: separators, case, aliases, `models/`. */
function canonTarget(value) {
    const parts = String(value || "")
        .replace(/\\/g, "/")
        .split("/")
        .map((part) => part.trim().toLowerCase())
        .filter((part) => part && part !== "." && part !== "..");
    if (parts[0] === "models") parts.shift();
    if (parts.length) parts[0] = FOLDER_ALIASES[parts[0]] || parts[0];
    return parts.join("/");
}

/** Compare two target folders written by a human: separators and case vary. */
function sameTarget(a, b) {
    return canonTarget(a) === canonTarget(b);
}

/**
 * Which spelling of an aliased category this list already uses.
 *
 * `clip` and `text_encoders` are equally valid names that ComfyUI has read for
 * years, and each is a real directory. Whichever one a user has settled on is
 * the right place to keep putting their files, so follow the list instead of
 * imposing the newer name on it.
 */
function preferredAliases(existingText) {
    const preferred = new Map();
    for (const row of parseFileList(existingText)) {
        const [first] = String(row.target || "")
            .replace(/\\/g, "/")
            .split("/")
            .map((part) => part.trim())
            .filter(Boolean);
        if (!first) continue;
        const spelled = first.toLowerCase();
        const canonical = FOLDER_ALIASES[spelled];
        if (canonical && !preferred.has(canonical)) preferred.set(canonical, first);
    }
    return preferred;
}

function applyPreferredAlias(directory, preferred) {
    if (!directory || !preferred.size) return directory;
    const parts = String(directory).split("/");
    const canonical = FOLDER_ALIASES[parts[0]?.toLowerCase()] || parts[0]?.toLowerCase();
    const spelling = preferred.get(canonical);
    if (!spelling) return directory;
    parts[0] = spelling;
    return parts.join("/");
}

/**
 * Parse a MarkdownNote body.
 *
 * Templates write the links as a list under a bold folder heading:
 *
 *   **text_encoders**
 *   - [qwen_3_06b_base.safetensors](https://…/qwen_3_06b_base.safetensors)
 *
 * so the most recent bold line before a link names its folder.
 */
function parseNote(text) {
    const found = [];
    let folder = "";
    const linkRe = /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g;
    const bareRe = /(https?:\/\/[^\s)<>"']+)/g;

    for (const line of String(text || "").split(/\r?\n/)) {
        // A folder heading is the bold (or ##) word that OPENS the line. The old
        // pattern demanded the line end right after the bold run, so a heading
        // that carries a qualifier — "**diffusion_models** (Mage-Flow-Turbo)",
        // which is how ComfyUI's own templates write them — matched nothing and
        // every link under it lost its folder.
        const heading = line.match(/^\s*(?:#+\s*)?\*\*([^*]+)\*\*.*$/) || line.match(/^\s*#+\s+([A-Za-z0-9_]+)\s*$/);
        if (heading) {
            folder = heading[1].trim();
            continue;
        }
        const seenUrls = new Set();
        linkRe.lastIndex = 0;
        let match;
        while ((match = linkRe.exec(line)) !== null) {
            seenUrls.add(match[2]);
            const label = match[1].trim();
            const name = MODEL_EXT.test(label) ? label : fileNameFromUrl(match[2]);
            if (MODEL_EXT.test(name)) found.push({ name, url: match[2], directory: folder });
        }
        bareRe.lastIndex = 0;
        while ((match = bareRe.exec(line)) !== null) {
            if (seenUrls.has(match[1])) continue;
            const name = fileNameFromUrl(match[1]);
            if (MODEL_EXT.test(name)) found.push({ name, url: match[1], directory: folder });
        }
    }
    return found;
}

/** Widget names on this node whose value is driven from outside by a link. */
function externallyDrivenWidgets(node) {
    return new Set(
        (node.inputs || [])
            .filter((input) => input?.widget?.name && input.link !== null && input.link !== undefined)
            .map((input) => String(input.widget.name)),
    );
}

/**
 * A node's widget values as {name: value}, EXCLUDING the ones a link overrides.
 *
 * A widget whose input is connected no longer shows its stored value — the
 * value comes down the wire. Inside a subgraph that is the normal case: the
 * loader keeps whatever default it was saved with while the real choice lives
 * on the subgraph's own widget, one level up. Reading the stale default is how
 * a model nobody uses ended up in the download list.
 */
function liveWidgetValues(node) {
    const driven = externallyDrivenWidgets(node);
    const named = node.widgets_values_named;
    if (named && typeof named === "object" && !Array.isArray(named)) {
        return Object.fromEntries(
            Object.entries(named).filter(([name, value]) => typeof value === "string" && !driven.has(name)),
        );
    }
    // Older graphs have no name map; positional values cannot be matched to an
    // input, so they are all kept (the pre-subgraph behaviour).
    const out = {};
    (node.widgets_values || []).forEach((value, index) => {
        if (typeof value === "string") out[`#${index}`] = value;
    });
    return out;
}

/**
 * Every model filename the workflow ACTUALLY uses, with the folder it belongs
 * in when that can be resolved.
 *
 * Three sources, all of them "what the user sees on the canvas":
 *   - a loader's own widget, when no link overrides it;
 *   - a widget promoted onto a subgraph instance, mapped back through the
 *     subgraph's links to the loader that consumes it (that is where the folder
 *     comes from);
 *   - any other node's widget that names a model file, so custom loaders count.
 */
export function visibleModelValues(serialised) {
    const definitions = new Map(
        (serialised.definitions?.subgraphs || []).map((def) => [String(def.id), def]),
    );
    const found = [];

    const walk = (container) => {
        for (const node of container.nodes || []) {
            if (!node || typeof node !== "object") continue;
            const definition = definitions.get(String(node.type));
            if (definition) {
                // A subgraph instance: its widgets are the promoted inputs, in
                // the order the definition declares them.
                const values = liveWidgetValues(node);
                const positional = Object.keys(values).every((key) => key.startsWith("#"));
                const inputs = definition.inputs || [];
                const nodeById = new Map((definition.nodes || []).map((n) => [Number(n.id), n]));
                const targetBySlot = new Map();
                for (const link of definition.links || []) {
                    // -10 is the subgraph's input node: slot -> the node it feeds.
                    if (String(link.origin_id) !== "-10") continue;
                    targetBySlot.set(Number(link.origin_slot), Number(link.target_id));
                }
                const entries = positional
                    ? inputs.map((input, index) => [String(input.name), values[`#${index}`]])
                    : Object.entries(values);
                for (const [name, value] of entries) {
                    if (typeof value !== "string" || !MODEL_EXT.test(value)) continue;
                    const slot = inputs.findIndex((input) => String(input.name) === name);
                    const consumer = slot >= 0 ? nodeById.get(targetBySlot.get(slot)) : null;
                    found.push({ value, folder: consumer ? TYPE_TO_FOLDER[consumer.type] : undefined });
                }
                continue;
            }
            const folder = TYPE_TO_FOLDER[node.type];
            for (const value of Object.values(liveWidgetValues(node))) {
                if (MODEL_EXT.test(value)) found.push({ value, folder });
            }
        }
        for (const sub of container.definitions?.subgraphs || []) walk(sub);
    };

    walk(serialised);
    return found;
}

/**
 * Walk the current workflow and merge the three sources into one map keyed by
 * filename. Returns entries shaped {name, url, directory, sources:Set}.
 */
export function scanWorkflow(graph) {
    let serialised;
    try {
        serialised = graph.serialize();
    } catch (err) {
        console.error("[TS FilesDownloader] graph.serialize() failed", err);
        return [];
    }

    const nodes = collectNodes(serialised);
    const byName = new Map();

    // Keyed by FILE NAME only. A loader stores `qwen/qwen_image_vae.safetensors`
    // while the template metadata says `qwen_image_vae.safetensors`; keying on
    // the full relative path would treat one file as two.
    const upsert = (name, source, patch) => {
        const { base } = splitRelativePath(name);
        const key = keyOf(base);
        if (!key) return null;
        let entry = byName.get(key);
        if (!entry) {
            entry = {
                name: base,
                url: "",
                directory: "",
                sources: new Set(),
                fromLoader: false,
            };
            byName.set(key, entry);
        }
        entry.sources.add(source);
        // First writer wins: sources are visited in order of trust.
        if (patch.url && !entry.url) entry.url = patch.url;
        if (patch.directory && !entry.directory) entry.directory = joinTarget(patch.directory, "");
        return entry;
    };

    // 1. properties.models — the authoritative source.
    for (const node of nodes) {
        const models = node.properties?.models;
        if (!Array.isArray(models)) continue;
        for (const model of models) {
            if (!model || typeof model !== "object") continue;
            const name = model.name || fileNameFromUrl(model.url);
            if (!MODEL_EXT.test(String(name || ""))) continue;
            upsert(name, "graph", { url: model.url, directory: model.directory });
        }
    }

    // 2. Markdown notes — fill the gaps, and confirm what source 1 already knows.
    for (const node of nodes) {
        if (node.type !== "MarkdownNote" && node.type !== "Note") continue;
        for (const value of node.widgets_values || []) {
            if (typeof value !== "string") continue;
            for (const hit of parseNote(value)) {
                upsert(hit.name, "note", { url: hit.url, directory: hit.directory });
            }
        }
    }

    // 3. What the canvas actually selects. This pass runs LAST and OVERRIDES the
    //    folder, because the widget is the only statement of where the workflow
    //    really looks: it carries the subfolder the user organised the model
    //    into, and the template metadata above knows nothing about it.
    //    Downloading to the metadata's bare folder leaves the file invisible.
    const visible = visibleModelValues(serialised);
    for (const { value, folder } of visible) {
        if (!folder) continue;
        const { base, sub } = splitRelativePath(value);
        const entry = upsert(base, "widget", {});
        if (!entry) continue;
        entry.directory = joinTarget(folder, sub);
        entry.fromLoader = true;
    }

    // 4. Anything the graph actually uses. Template metadata outlives the model
    //    it described — swap in your own checkpoint and the old {name, url}
    //    stays behind forever — so an entry no visible widget selects is a
    //    leftover, not a requirement. `visible` deliberately excludes a value a
    //    link overrides: inside a subgraph that stale default is exactly the
    //    "left over" model, and counting it made the node offer a download for
    //    a file the workflow never loads.
    const referenced = new Set(visible.map(({ value }) => keyOf(splitRelativePath(value).base)));
    for (const [key, entry] of byName) entry.referenced = referenced.has(key);

    return [...byName.values()].sort((a, b) =>
        (a.directory || "").localeCompare(b.directory || "") || a.name.localeCompare(b.name),
    );
}

/* --------------------------------------------------------------- file_list IO */

function parseFileList(text) {
    return String(text || "")
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line && !line.startsWith("#"))
        .map((line) => {
            const parts = line.split(/\s+/);
            return { url: parts[0] || "", target: parts.slice(1).join(" ") };
        });
}

/**
 * Classify each entry against what the node already lists.
 *
 * A model that is already downloaded is NOT filtered out: the point of the list
 * is to travel with the workflow, so the next person gets the file too.
 */
export function classify(entries, existingText) {
    // The list itself is a source of URLs: anything already in it was curated
    // by hand, and for a model no template describes it is the ONLY link there
    // is. Ignoring it reported hand-added models as "no download link".
    const listed = new Map();
    for (const row of parseFileList(existingText)) {
        listed.set(keyOf(fileNameFromUrl(row.url)), row);
    }
    const preferred = preferredAliases(existingText);
    const add = [];
    const present = [];
    const mismatch = [];
    const noLink = [];
    const orphan = [];
    for (const raw of entries) {
        const key = keyOf(raw.name);
        const row = listed.get(key);
        const entry = {
            ...raw,
            directory: applyPreferredAlias(raw.directory, preferred),
            // Fall back to the link the user already wrote down for this file.
            url: raw.url || (row ? row.url : ""),
        };
        if (!entry.referenced) {
            // Named only by stale template metadata: downloading it would fetch
            // a file this graph never loads.
            orphan.push(entry);
            continue;
        }
        if (!entry.url || !entry.directory) {
            noLink.push(entry);
            continue;
        }
        if (!row) {
            add.push(entry);
        } else if (sameTarget(row.target, entry.directory)) {
            present.push(entry);
        } else {
            // Listed, but aimed at a folder the loader does not read from —
            // typically a line written before subfolders were understood.
            mismatch.push({ ...entry, currentTarget: row.target });
        }
    }
    return { add, present, mismatch, noLink, orphan };
}

/** Rewrite only the target of lines whose model is aimed at the wrong folder. */
/**
 * The target as this node writes it into the list: `models/<folder>/<sub>`.
 *
 * Everything the scan produces is a MODEL folder, so spelling the full path
 * says where the file goes without the reader having to know that a bare
 * `diffusion_models` is resolved against the models directory rather than
 * ComfyUI's root. The backend accepts every spelling, but only one of them
 * should be the one we generate.
 */
export function displayTarget(directory) {
    const parts = String(directory || "")
        .replace(/\\/g, "/")
        .split("/")
        .map((part) => part.trim())
        .filter(Boolean);
    if (!parts.length) return "";
    if (parts[0].toLowerCase() === "models") parts.shift();
    if (!parts.length) return "models";
    return ["models", ...parts].join("/");
}

function fixTargets(text, entries) {
    const wanted = new Map(entries.map((e) => [keyOf(e.name), displayTarget(e.directory)]));
    let fixed = 0;
    const lines = String(text || "").split(/\r?\n/).map((line) => {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith("#")) return line;
        const parts = trimmed.split(/\s+/);
        const url = parts[0];
        const target = wanted.get(keyOf(fileNameFromUrl(url)));
        if (!target || sameTarget(parts.slice(1).join(" "), target)) return line;
        fixed += 1;
        return `${url} ${target}`;
    });
    return { text: lines.join("\n"), fixed };
}

export function toLine(entry) {
    return `${entry.url} ${displayTarget(entry.directory)}`;
}

/**
 * Lines of the current list that a rewrite must not throw away: comments, and
 * downloads for models this scan is not rewriting (a utility model no loader
 * names, an archive, a link the user curated by hand). "Replace list" used to
 * emit only what the scan found, silently deleting the rest.
 */
function unclaimedLines(text, written) {
    const claimed = new Set(written.map((entry) => keyOf(entry.name)));
    return String(text || "")
        .split(/\r?\n/)
        .filter((line) => {
            const trimmed = line.trim();
            if (!trimmed) return false;
            if (trimmed.startsWith("#")) return true;
            return !claimed.has(keyOf(fileNameFromUrl(trimmed.split(/\s+/)[0])));
        })
        .map((line) => line.trim());
}

function writeFileList(node, widget, text) {
    widget.value = text;
    try {
        widget.callback?.(text);
    } catch (err) {
        console.warn("[TS FilesDownloader] file_list callback failed", err);
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

/* ------------------------------------------------------------------- reporting */

function ensureStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    // Layout only — every colour comes from the --ts-* tokens.
    style.textContent = `
.ts-fdl{display:flex;flex-direction:column;gap:12px;width:min(760px,92vw);max-height:82vh;
    padding:18px;border-radius:var(--ts-radius-lg);background:var(--ts-elevated);
    border:1px solid var(--ts-border);box-shadow:var(--ts-shadow);color:var(--ts-text)}
/* Right inset keeps the title and the summary clear of the overlay's close
   button, which sits on the panel's own corner in the centred variant. */
.ts-fdl__head{display:flex;flex-direction:column;gap:4px;padding-right:38px}
.ts-fdl__scroll{overflow:auto;display:flex;flex-direction:column;gap:14px;padding-right:4px}
.ts-fdl__group{display:flex;flex-direction:column;gap:6px}
.ts-fdl__grouphead{display:flex;align-items:baseline;gap:8px;
    font-size:var(--ts-fs-sm);font-weight:700;color:var(--ts-muted);
    text-transform:uppercase;letter-spacing:.04em}
.ts-fdl__row{display:grid;grid-template-columns:minmax(120px,auto) 1fr auto;gap:10px;
    align-items:baseline;padding:6px 8px;border-radius:8px;background:var(--ts-sunken)}
.ts-fdl__dir{font-weight:700;color:var(--ts-accent);font-size:var(--ts-fs-sm)}
.ts-fdl__name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ts-fdl__src{font-size:var(--ts-fs-sm);color:var(--ts-muted);white-space:nowrap}
.ts-fdl__action{display:flex;align-items:center;gap:10px;margin:2px 0 8px;flex-wrap:wrap}
.ts-fdl__bar{flex:1 1 120px;min-width:100px;height:3px;border-radius:2px;background:var(--ts-border-soft);overflow:hidden;display:none}
.ts-fdl__bar.is-active{display:block}
.ts-fdl__bar div{height:100%;width:0%;background:var(--ts-accent);transition:width .25s ease}
.ts-fdl__hint{font-size:var(--ts-fs-sm);color:var(--ts-muted)}
.ts-fdl__foot{display:flex;justify-content:flex-end;gap:8px;flex-wrap:wrap}
`;
    document.head.appendChild(style);
}

function sourceLabel(entry, t) {
    const hasGraph = entry.sources.has("graph");
    const hasNote = entry.sources.has("note");
    let origin = t.srcWidget;
    if (hasGraph && hasNote) origin = t.srcBoth;
    else if (hasGraph) origin = t.srcGraph;
    else if (hasNote) origin = t.srcNote;
    // Template metadata outlives the model it described: if no loader in the
    // graph selects this file, say so instead of quietly proposing it.
    if (!entry.fromLoader && (hasGraph || hasNote)) return `${origin} · ${t.srcUnused}`;
    return origin;
}

function buildGroup(title, entries, t, hint, action) {
    if (!entries.length) return null;
    const group = document.createElement("div");
    group.className = "ts-fdl__group";

    const head = document.createElement("div");
    head.className = "ts-fdl__grouphead";
    head.textContent = `${title} — ${entries.length}`;
    group.appendChild(head);

    if (action) group.appendChild(action);

    if (hint) {
        const note = document.createElement("div");
        note.className = "ts-fdl__hint";
        note.textContent = hint;
        group.appendChild(note);
    }

    for (const entry of entries) {
        const row = document.createElement("div");
        row.className = "ts-fdl__row";

        const dir = document.createElement("span");
        dir.className = "ts-fdl__dir";
        dir.textContent = entry.currentTarget
            ? `${entry.currentTarget} → ${entry.directory}`
            : entry.directory || "—";

        const name = document.createElement("span");
        name.className = "ts-fdl__name";
        name.textContent = entry.name;
        name.title = entry.url || entry.name;

        const src = document.createElement("span");
        src.className = "ts-fdl__src";
        src.textContent = sourceLabel(entry, t);

        row.append(dir, name, src);
        group.appendChild(row);
    }
    return group;
}

/**
 * "Find on Hugging Face" for the models the graph names but carries no link
 * for. The backend matches the exact filename inside candidate repos and
 * builds the download URL itself; this only presents what came back.
 */
function buildHfSearchAction(entries, t, onFound) {
    const wrap = document.createElement("div");
    wrap.className = "ts-fdl__action";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "ts-ui-btn";
    button.textContent = t.hfSearch;

    const status = document.createElement("span");
    status.className = "ts-fdl__hint";

    // Looking through several repositories takes tens of seconds. The backend
    // reports where it is, keyed by an operation id so two nodes searching at
    // once cannot move each other's bar.
    const bar = document.createElement("div");
    bar.className = "ts-fdl__bar";
    const barFill = document.createElement("div");
    bar.appendChild(barFill);

    let operationId = null;
    const onProgress = (event) => {
        const detail = event?.detail || {};
        if (!operationId || detail.operation_id !== operationId) return;
        if (detail.text) status.textContent = String(detail.text);
        if (typeof detail.percent === "number") {
            barFill.style.width = `${Math.max(0, Math.min(100, detail.percent))}%`;
        }
    };
    api.addEventListener("ts_downloader.search_progress", onProgress);
    // Closing the report must take the subscription with it, or every scan
    // leaves another listener (and this whole closure) alive.
    wrap._tsTeardown = () => api.removeEventListener("ts_downloader.search_progress", onProgress);

    button.addEventListener("click", async () => {
        button.disabled = true;
        status.textContent = t.hfSearching;
        operationId = `ts_hf_${Math.random().toString(36).slice(2, 10)}`;
        bar.classList.add("is-active");
        barFill.style.width = "2%";
        try {
            const response = await api.fetchApi("/ts_downloader/hf_search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filenames: entries.map((e) => e.name),
                    operation_id: operationId,
                }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok || payload?.error) throw new Error(payload?.error || `HTTP ${response.status}`);

            const results = payload.results || {};
            const found = [];
            for (const entry of entries) {
                const best = (results[entry.name] || [])[0];
                if (!best?.url) continue;
                found.push({ ...entry, url: best.url, hfRepo: best.repo, hfSize: best.size });
            }
            if (!found.length) {
                status.textContent = t.hfNothing;
                bar.classList.remove("is-active");
                button.disabled = false;
                operationId = null;
                return;
            }
            status.textContent = t.hfFound(found.length);
            onFound(found);
        } catch (error) {
            status.textContent = t.hfFailed(error?.message || error);
            bar.classList.remove("is-active");
            button.disabled = false;
        } finally {
            operationId = null;
        }
    });

    wrap.append(button, status, bar);
    return wrap;
}

// Only ever one report on screen: a second press while the first is still open
// would stack overlays, and the buttons of the stale one would write a diff that
// no longer matches the list.
let openReport = null;

function showReport(node, widget, buckets, t) {
    ensureStyles();
    if (openReport?.isOpen()) openReport.close();

    const panel = document.createElement("div");
    panel.className = `${TS_UI_CLASS} ts-fdl`;

    const head = document.createElement("div");
    head.className = "ts-fdl__head";
    const title = document.createElement("div");
    title.className = "ts-ui-title";
    title.textContent = t.title;
    const summary = document.createElement("div");
    summary.className = "ts-fdl__hint";
    const total =
        buckets.add.length + buckets.mismatch.length + buckets.present.length +
        buckets.noLink.length + buckets.orphan.length;
    summary.textContent = total
        ? t.summary(buckets.add.length, total)
        : t.empty;
    head.append(title, summary);
    panel.appendChild(head);

    const scroll = document.createElement("div");
    scroll.className = "ts-fdl__scroll";
    const groups = [
        buildGroup(t.colAdd, buckets.add, t),
        buildGroup(t.colMismatch, buckets.mismatch, t, t.mismatchHint),
        buildGroup(t.colPresent, buckets.present, t),
        buildGroup(
            t.colNoLink, buckets.noLink, t, t.noLinkHint,
            buckets.noLink.length
                ? buildHfSearchAction(buckets.noLink, t, (found) => {
                    const current = String(widget.value || "");
                    const prefix = current && !current.endsWith("\n") ? "\n" : "";
                    writeFileList(node, widget, current + prefix + found.map(toLine).join("\n") + "\n");
                    toast("success", t.appended(found.length));
                    openReport?.close();
                })
                : null,
        ),
        buildGroup(t.colOrphan, buckets.orphan, t, t.orphanHint),
    ].filter(Boolean);
    if (!groups.length) {
        const empty = document.createElement("div");
        empty.className = "ts-fdl__hint";
        empty.textContent = t.emptyHint;
        scroll.appendChild(empty);
    } else {
        groups.forEach((g) => scroll.appendChild(g));
    }
    panel.appendChild(scroll);

    const foot = document.createElement("div");
    foot.className = "ts-fdl__foot";
    panel.appendChild(foot);

    const overlay = openFullscreenOverlay(panel, {
        // A report holds no unsaved work, so dismissing it costs nothing.
        closeOnBackdrop: true,
        // It is a dialog, not an editor: the user clicked a node in the middle
        // of the canvas, so the panel belongs in the middle of the screen.
        center: true,
        extraClass: "ts-fdl-overlay",
        closeTitle: t.close,
        onClose: () => {
            for (const el of panel.querySelectorAll(".ts-fdl__action")) {
                try { el._tsTeardown?.(); } catch { /* already gone */ }
            }
            if (openReport === overlay) openReport = null;
        },
    });
    openReport = overlay;

    const addButton = (label, primary, handler) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = primary ? "ts-ui-btn ts-ui-btn--primary" : "ts-ui-btn";
        button.textContent = label;
        button.addEventListener("click", () => {
            try {
                handler();
            } finally {
                overlay.close();
            }
        });
        foot.appendChild(button);
        return button;
    };

    const ready = [...buckets.add, ...buckets.mismatch, ...buckets.present];
    if (buckets.mismatch.length) {
        addButton(t.fix, !buckets.add.length, () => {
            const result = fixTargets(widget.value, buckets.mismatch);
            writeFileList(node, widget, result.text);
            toast("success", t.fixedPaths(result.fixed));
        });
    }
    if (buckets.add.length) {
        addButton(t.merge, true, () => {
            const current = String(widget.value || "");
            const prefix = current && !current.endsWith("\n") ? "\n" : "";
            writeFileList(node, widget, current + prefix + buckets.add.map(toLine).join("\n") + "\n");
            toast("success", t.appended(buckets.add.length));
        });
    }
    if (ready.length) {
        addButton(t.replace, !buckets.add.length, () => {
            const lines = [...ready.map(toLine), ...unclaimedLines(widget.value, ready)];
            writeFileList(node, widget, lines.join("\n") + "\n");
            toast("success", t.replaced(lines.length));
        });
    }
    addButton(t.cancel, false, () => {});
}

function toast(severity, detail) {
    try {
        app.extensionManager?.toast?.add({ severity, summary: "TS Files Downloader", detail, life: 4000 });
    } catch {
        /* toasts are a nicety; never let one break the flow */
    }
}

/* ------------------------------------------------------------------ extension */

function getWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name) || null;
}

function attachButton(node) {
    if (node.__tsFdlButton) return;
    const t = pickLocaleStrings(STRINGS);

    const button = node.addWidget("button", t.button, null, () => {
        const widget = getWidget(node, FILE_LIST_WIDGET);
        if (!widget) {
            console.warn(`[TS FilesDownloader] ${FILE_LIST_WIDGET} widget not found`);
            return;
        }
        const entries = scanWorkflow(app.graph);
        const buckets = classify(entries, widget.value);
        if (!buckets.add.length && !buckets.present.length && !buckets.noLink.length) {
            toast("info", t.empty);
        }
        if (!buckets.add.length && buckets.present.length && !buckets.noLink.length) {
            toast("info", t.nothing);
        }
        showReport(node, widget, buckets, t);
    });

    // Keep this widget OUT of widgets_values: that array is positional, and one
    // extra slot would shift all 11 saved values of every existing workflow.
    button.serialize = false;
    button.options = { ...(button.options || {}), serialize: false };
    node.__tsFdlButton = button;
}

app.registerExtension({
    name: "ts.filesDownloader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                attachButton(this);
            } catch (err) {
                console.error("[TS FilesDownloader] failed to add the scan button", err);
            }
            return result;
        };
    },
});
