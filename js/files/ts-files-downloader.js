// TS Files Downloader (Ultimate) — "Get models from workflow" button.
//
// Fills the node's `file_list` with every model the CURRENT workflow needs, so a
// graph handed to someone else pulls its own weights on the first run.
//
// Where the models come from, in order of trust:
//
//   1. `node.properties.models` — ComfyUI stamps {name, url, directory} onto each
//      loader that came from a template. Machine-readable and exact, so this is
//      the primary source, NOT the note.
//   2. A MarkdownNote's model list — the same facts as prose, `[file](url)`
//      grouped under **folder** headings. Fills gaps and cross-checks (1).
//   3. A loader's own widget value — yields the filename and, through the node
//      type, the folder, but no URL. Reported so the user sees what is missing;
//      looking those up on HuggingFace is a separate, later pass.
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
        colNoLink: "No download link",
        srcGraph: "node",
        srcNote: "note",
        srcBoth: "node + note",
        srcWidget: "filename only",
        merge: "Append",
        replace: "Replace list",
        cancel: "Cancel",
        close: "Close (Esc)",
        empty: "No models found in this workflow.",
        emptyHint:
            "Model links live in each loader's properties, or in a Markdown note. " +
            "This workflow has neither.",
        noLinkHint:
            "These are used by the workflow but no URL was found. Add them by hand, " +
            "or wait for the HuggingFace lookup.",
        appended: (n) => `Appended ${n} line(s) to file_list.`,
        replaced: (n) => `file_list replaced with ${n} line(s).`,
        nothing: "Nothing to add — every model is already in the list.",
    },
    ru: {
        button: "Взять модели из workflow",
        title: "Модели, найденные в этом workflow",
        summary: (add, total) => `${add} к добавлению · всего найдено ${total}`,
        colAdd: "Будут добавлены",
        colPresent: "Уже в списке",
        colNoLink: "Ссылка не найдена",
        srcGraph: "нода",
        srcNote: "заметка",
        srcBoth: "нода + заметка",
        srcWidget: "только имя файла",
        merge: "Дополнить",
        replace: "Заменить список",
        cancel: "Отмена",
        close: "Закрыть (Esc)",
        empty: "В этом workflow моделей не найдено.",
        emptyHint:
            "Ссылки на модели лежат в свойствах нод-загрузчиков либо в Markdown-заметке. " +
            "Здесь нет ни того, ни другого.",
        noLinkHint:
            "Эти модели workflow использует, но ссылки на них не нашлось. Добавьте вручную " +
            "или дождитесь поиска по HuggingFace.",
        appended: (n) => `В file_list дописано строк: ${n}.`,
        replaced: (n) => `file_list заменён, строк: ${n}.`,
        nothing: "Добавлять нечего — все модели уже в списке.",
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
        const bold = line.match(/^\s*(?:#+\s*)?\*\*([^*]+)\*\*\s*$/);
        if (bold) {
            folder = bold[1].trim();
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

/**
 * Walk the current workflow and merge the three sources into one map keyed by
 * filename. Returns entries shaped {name, url, directory, sources:Set}.
 */
function scanWorkflow(graph) {
    let serialised;
    try {
        serialised = graph.serialize();
    } catch (err) {
        console.error("[TS FilesDownloader] graph.serialize() failed", err);
        return [];
    }

    const nodes = collectNodes(serialised);
    const byName = new Map();

    const upsert = (name, source, patch) => {
        const key = keyOf(name);
        if (!key) return null;
        let entry = byName.get(key);
        if (!entry) {
            entry = { name: String(name).trim(), url: "", directory: "", sources: new Set() };
            byName.set(key, entry);
        }
        entry.sources.add(source);
        // First writer wins: sources are visited in order of trust.
        if (patch.url && !entry.url) entry.url = patch.url;
        if (patch.directory && !entry.directory) entry.directory = String(patch.directory).trim();
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

    // 3. Loader widgets — filename and folder only, no link.
    for (const node of nodes) {
        const folder = TYPE_TO_FOLDER[node.type];
        if (!folder) continue;
        for (const value of node.widgets_values || []) {
            if (typeof value !== "string" || !MODEL_EXT.test(value)) continue;
            upsert(value, "widget", { directory: folder });
        }
    }

    // A URL with no folder can still be placed if some loader type claims it.
    for (const entry of byName.values()) {
        if (entry.directory) continue;
        for (const node of nodes) {
            const folder = TYPE_TO_FOLDER[node.type];
            if (!folder) continue;
            const used = (node.widgets_values || []).some(
                (v) => typeof v === "string" && keyOf(v) === keyOf(entry.name),
            );
            if (used) {
                entry.directory = folder;
                break;
            }
        }
    }

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
function classify(entries, existingText) {
    const listed = new Set(parseFileList(existingText).map((row) => keyOf(fileNameFromUrl(row.url))));
    const add = [];
    const present = [];
    const noLink = [];
    for (const entry of entries) {
        if (!entry.url || !entry.directory) noLink.push(entry);
        else if (listed.has(keyOf(entry.name))) present.push(entry);
        else add.push(entry);
    }
    return { add, present, noLink };
}

function toLine(entry) {
    return `${entry.url} ${entry.directory}`;
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
.ts-fdl__head{display:flex;flex-direction:column;gap:4px}
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
.ts-fdl__hint{font-size:var(--ts-fs-sm);color:var(--ts-muted)}
.ts-fdl__foot{display:flex;justify-content:flex-end;gap:8px;flex-wrap:wrap}
`;
    document.head.appendChild(style);
}

function sourceLabel(entry, t) {
    const hasGraph = entry.sources.has("graph");
    const hasNote = entry.sources.has("note");
    if (hasGraph && hasNote) return t.srcBoth;
    if (hasGraph) return t.srcGraph;
    if (hasNote) return t.srcNote;
    return t.srcWidget;
}

function buildGroup(title, entries, t, hint) {
    if (!entries.length) return null;
    const group = document.createElement("div");
    group.className = "ts-fdl__group";

    const head = document.createElement("div");
    head.className = "ts-fdl__grouphead";
    head.textContent = `${title} — ${entries.length}`;
    group.appendChild(head);

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
        dir.textContent = entry.directory || "—";

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
    const total = buckets.add.length + buckets.present.length + buckets.noLink.length;
    summary.textContent = total
        ? t.summary(buckets.add.length, total)
        : t.empty;
    head.append(title, summary);
    panel.appendChild(head);

    const scroll = document.createElement("div");
    scroll.className = "ts-fdl__scroll";
    const groups = [
        buildGroup(t.colAdd, buckets.add, t),
        buildGroup(t.colPresent, buckets.present, t),
        buildGroup(t.colNoLink, buckets.noLink, t, t.noLinkHint),
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
        extraClass: "ts-fdl-overlay",
        closeTitle: t.close,
        onClose: () => {
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

    const ready = [...buckets.add, ...buckets.present];
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
            writeFileList(node, widget, ready.map(toLine).join("\n") + "\n");
            toast("success", t.replaced(ready.length));
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
