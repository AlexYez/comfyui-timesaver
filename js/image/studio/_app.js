// TS Image Studio — application composition (app layer).
//
// Wiring only (plan §3.5): builds the shell, loads backends, renders the
// deck from the active backend's manifest through the control registry,
// submits runs through the runner and feeds the gallery. No business logic
// beyond composition lives here.

import { api } from "/scripts/api.js";
import { pickLocaleStrings } from "../../_theme.js";
import { createShell, deckSection } from "../../_studio/_shell.js";
import { ensureControlStyles, getControlRenderer, randomSeed } from "../../_studio/_controls.js";
import { createGallery } from "../../_studio/_gallery.js";
import { createRunner } from "../../_studio/_runner.js";
import { loadBackends, groupByFamily } from "../../_studio/_backends.js";
import { patchBackend } from "../../_studio/_markers.js";
import { newSessionId, sessionPrefix, resultRelPath, restoreResults } from "../../_studio/_session.js";

const ICONS = {
    generate: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 3l1.8 4.7L18.5 9l-4.7 1.8L12 15.5l-1.8-4.7L5.5 9l4.7-1.3zM19 15l.9 2.3 2.1.7-2.1.9L19 21l-.9-2.1-2.1-.9 2.1-.7z"/></svg>',
};

const STRINGS = {
    en: {
        appLabel: "TS Image Studio",
        close: "Close (Esc)",
        collapse: "Collapse or expand the asset panel (Tab)",
        model: "Model",
        prompt: "Prompt",
        negativePrompt: "Negative prompt",
        promptPlaceholder: "Describe the image…",
        format: "Format",
        resolution: "Resolution",
        seed: "Seed",
        randomize: "randomize",
        randomizeTip: "New random seed on every run. Type a seed to pin it.",
        advanced: "Advanced",
        run: "Run",
        runHint: "Ctrl+Enter",
        queued: (n) => `${n} in queue`,
        cancel: "Cancel",
        generating: (p) => `Generating… ${p}%`,
        tabSession: "Session",
        tabLibrary: "Library",
        tabLibrarySoon: "Asset browser arrives in the next phase",
        galleryEmpty: "Results of this session appear here.",
        stageEmpty: "Describe the image and press Run.",
        backendBroken: "unavailable",
        noBackends: "No backend workflows are available for any installed model.",
        runFailed: (m) => `Run failed: ${m}`,
        modes: { t2i: "Generate", edit: "Edit", inpaint: "Inpaint", upscale: "Upscale" },
    },
    ru: {
        appLabel: "TS Image Studio",
        close: "Закрыть (Esc)",
        collapse: "Свернуть или развернуть панель ассетов (Tab)",
        model: "Модель",
        prompt: "Промпт",
        negativePrompt: "Негативный промпт",
        promptPlaceholder: "Опишите изображение…",
        format: "Формат",
        resolution: "Разрешение",
        seed: "Seed",
        randomize: "случайный",
        randomizeTip: "Новый случайный сид на каждый запуск. Введите сид, чтобы закрепить.",
        advanced: "Дополнительно",
        run: "Run",
        runHint: "Ctrl+Enter",
        queued: (n) => `в очереди: ${n}`,
        cancel: "Отмена",
        generating: (p) => `Генерация… ${p}%`,
        tabSession: "Сессия",
        tabLibrary: "Библиотека",
        tabLibrarySoon: "Браузер ассетов появится в следующей фазе",
        galleryEmpty: "Результаты этой сессии появятся здесь.",
        stageEmpty: "Опишите изображение и нажмите Run.",
        backendBroken: "недоступен",
        noBackends: "Нет доступных workflow ни для одной установленной модели.",
        runFailed: (m) => `Ошибка запуска: ${m}`,
        modes: { t2i: "Генерация", edit: "Редактирование", inpaint: "Inpaint", upscale: "Upscale" },
    },
};

const APP_STYLE_ID = "ts-image-studio-app-styles";

function ensureAppStyles() {
    if (document.getElementById(APP_STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = APP_STYLE_ID;
    style.textContent = `
.ts-istudio__stagefit{position:absolute;inset:12px;display:flex;align-items:center;justify-content:center}
.ts-istudio__stagefit img{max-width:100%;max-height:100%;object-fit:contain;border-radius:4px}
.ts-istudio__stageempty{color:var(--ts-muted);font-size:var(--ts-fs-lg)}
.ts-istudio__caption{position:absolute;left:10px;bottom:10px;padding:3px 8px;font-size:var(--ts-fs-sm);
    color:var(--ts-muted);background:var(--ts-elevated);border:1px solid var(--ts-border);
    border-radius:var(--ts-radius-sm)}
.ts-istudio__modelrow{display:flex;align-items:center;gap:6px}
.ts-istudio__modelrow select{flex:1}
.ts-istudio__runwrap{display:flex;flex-direction:column;gap:4px}
.ts-istudio__runhint{text-align:center;color:var(--ts-muted);font-size:var(--ts-fs-xs)}
.ts-istudio__progress{height:3px;border-radius:2px;background:var(--ts-border-soft);overflow:hidden;display:none}
.ts-istudio__progress.is-active{display:block}
.ts-istudio__progress div{height:100%;width:0%;background:var(--ts-accent);transition:width .2s ease}
`;
    document.head.appendChild(style);
}

/** Open the studio for one node instance. Returns the overlay handle. */
export async function openStudio(node, persist) {
    ensureAppStyles();
    ensureControlStyles();
    const t = pickLocaleStrings(STRINGS);
    const locale = STRINGS.ru === t ? "ru" : "en";

    const objectInfo = await (await api.fetchApi("/object_info")).json();
    // Backend workflow files are WEB_DIRECTORY statics: /extensions/* lives
    // OUTSIDE the /api prefix that api.fetchApi prepends, so plain fetch.
    const backends = await loadBackends((url) => fetch(url), objectInfo);
    const families = groupByFamily(backends);
    const sessionId = persist.sessionId || newSessionId();
    persist.setSessionId(sessionId);

    const optionalIndex = buildOptionalIndex(objectInfo);
    const runner = createRunner(api);
    const values = {};      // param -> value for the active backend
    let activeBackend = null;
    let queueCount = 0;

    const modeIds = [...new Set([...families.values()].flatMap((f) => [...f.modes.keys()]))];
    const shell = createShell({
        label: t.appLabel,
        closeTitle: t.close,
        collapseTitle: t.collapse,
        modes: modeIds.map((id) => ({ id, title: t.modes[id] || id, icon: ICONS.generate })),
        onMode: (id) => selectMode(id),
        onClose: () => runner.destroy(),
        onKey: (event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                run();
            } else if (event.key === "Tab") {
                event.preventDefault();
                shell.setSideCollapsed(!shell.side.classList.contains("is-collapsed"));
            }
        },
    });

    // ── stage ───────────────────────────────────────────────────────────── //
    const stageFit = document.createElement("div");
    stageFit.className = "ts-istudio__stagefit";
    const stageEmpty = document.createElement("div");
    stageEmpty.className = "ts-istudio__stageempty";
    stageEmpty.textContent = t.stageEmpty;
    const stageImg = document.createElement("img");
    stageImg.style.display = "none";
    stageImg.alt = "";
    const caption = document.createElement("div");
    caption.className = "ts-istudio__caption";
    caption.style.display = "none";
    stageFit.append(stageEmpty, stageImg);
    shell.stage.append(stageFit, caption);

    function showResult(result) {
        stageImg.src = `/view?filename=${encodeURIComponent(result.image.filename)}` +
            `&subfolder=${encodeURIComponent(result.image.subfolder || "")}&type=output`;
        stageImg.style.display = "";
        stageEmpty.style.display = "none";
        const params = result.params || {};
        const bits = [];
        if (params.width && params.height) bits.push(`${params.width} × ${params.height}`);
        if (params.seed !== undefined) bits.push(`seed ${params.seed}`);
        caption.textContent = bits.join(" · ");
        caption.style.display = bits.length ? "" : "none";
    }

    function showPreviewBlob(blob) {
        stageImg.src = URL.createObjectURL(blob);
        stageImg.style.display = "";
        stageEmpty.style.display = "none";
    }

    // ── gallery (right panel) ───────────────────────────────────────────── //
    const gallery = createGallery({
        t,
        onSelect: (result) => {
            showResult(result);
            persist.setResultPath(resultRelPath(result.image));
        },
    });
    shell.side.appendChild(gallery.element);
    restoreResults((url) => api.fetchApi(url), sessionId).then((restored) => {
        if (restored.length) {
            gallery.setAll(restored);
            gallery.selectLast();
        }
    });

    // ── deck ────────────────────────────────────────────────────────────── //
    let seedControl = null;

    function buildDeck(backend) {
        shell.deck.textContent = "";
        seedControl = null;
        for (const key of Object.keys(values)) delete values[key];

        const modelSection = deckSection(t.model);
        const modelRow = document.createElement("div");
        modelRow.className = "ts-istudio__modelrow";
        const select = document.createElement("select");
        select.className = "ts-ui-select";
        for (const family of families.values()) {
            const candidate = family.modes.get(backend.manifest.mode);
            if (!candidate) continue;
            const option = document.createElement("option");
            option.value = family.family;
            option.textContent = candidate.available
                ? family.label
                : `${family.label} — ${t.backendBroken}`;
            option.disabled = !candidate.available;
            option.selected = family.family === backend.manifest.family;
            select.appendChild(option);
        }
        select.addEventListener("change", () => {
            const next = families.get(select.value)?.modes.get(backend.manifest.mode);
            if (next?.available) selectBackend(next);
        });
        modelRow.appendChild(select);
        modelSection.appendChild(modelRow);
        shell.deck.appendChild(modelSection);

        const advanced = [];
        for (const control of backend.manifest.controls || []) {
            const renderer = getControlRenderer(control.kind);
            if (!renderer) {
                console.warn(`[TS Studio] no renderer for control kind '${control.kind}' — skipped`);
                continue;
            }
            const instance = renderer(control, {
                t, locale,
                onChange: (param, value) => { values[param] = value; },
            });
            if (control.kind === "seed") seedControl = instance;
            if (control.advanced) advanced.push(instance.element);
            else shell.deck.appendChild(instance.element);
        }
        if (advanced.length) {
            const toggle = document.createElement("button");
            toggle.type = "button";
            toggle.className = "ts-studio__advanced";
            toggle.textContent = `▸ ${t.advanced}`;
            const holder = document.createElement("div");
            holder.style.display = "none";
            holder.append(...advanced);
            toggle.addEventListener("click", () => {
                const open = holder.style.display === "none";
                holder.style.display = open ? "" : "none";
                toggle.textContent = `${open ? "▾" : "▸"} ${t.advanced}`;
            });
            shell.deck.append(toggle, holder);
        }

        const foot = document.createElement("div");
        foot.className = "ts-studio__deckfoot";
        const progress = document.createElement("div");
        progress.className = "ts-istudio__progress";
        const progressFill = document.createElement("div");
        progress.appendChild(progressFill);
        const runWrap = document.createElement("div");
        runWrap.className = "ts-istudio__runwrap";
        const runButton = document.createElement("button");
        runButton.type = "button";
        runButton.className = "ts-ui-btn ts-ui-btn--primary";
        runButton.textContent = t.run;
        runButton.addEventListener("click", () => run());
        const hint = document.createElement("div");
        hint.className = "ts-istudio__runhint";
        hint.textContent = t.runHint;
        runWrap.append(runButton, progress, hint);
        foot.appendChild(runWrap);
        shell.deck.appendChild(foot);
        deckWidgets = { runButton, progress, progressFill, hint };
    }

    let deckWidgets = null;

    function updateHint() {
        if (deckWidgets) {
            deckWidgets.hint.textContent = queueCount > 0
                ? `${t.runHint} · ${t.queued(queueCount)}` : t.runHint;
        }
    }

    function selectBackend(backend) {
        activeBackend = backend;
        buildDeck(backend);
    }

    function selectMode(modeId) {
        shell.setMode(modeId);
        const current = activeBackend?.manifest.family;
        const inFamily = families.get(current)?.modes.get(modeId);
        const fallback = [...families.values()]
            .map((f) => f.modes.get(modeId)).find((b) => b?.available);
        const next = inFamily?.available ? inFamily : fallback;
        if (next) selectBackend(next);
    }

    // ── run ─────────────────────────────────────────────────────────────── //
    async function run() {
        if (!activeBackend) return;
        const seedState = values.seed || { value: 0, randomize: true };
        const seed = seedState.randomize ? randomSeed() : Number(seedState.value || 0);
        seedControl?.showSeed(seed);

        const runValues = {};
        for (const [param, value] of Object.entries(values)) {
            if (param === "seed") continue;
            if (typeof value === "object" && value !== null) continue;
            runValues[param] = value;
        }
        runValues.seed = seed;

        let patched;
        try {
            patched = patchBackend(activeBackend.graph, activeBackend.spec, {
                values: runValues,
                modelFiles: activeBackend.modelFiles,
                filenamePrefix: sessionPrefix(sessionId),
                isOptionalInput: optionalIndex,
            });
        } catch (err) {
            setStatus(t.runFailed(err.message));
            return;
        }
        try {
            queueCount += 1;
            updateHint();
            await runner.submit(patched, {
                onProgress: (value, max) => {
                    const pct = max ? Math.round((value / max) * 100) : 0;
                    deckWidgets.progress.classList.add("is-active");
                    deckWidgets.progressFill.style.width = `${pct}%`;
                },
                onPreview: (blob) => showPreviewBlob(blob),
                onDone: (images) => {
                    queueCount -= 1;
                    updateHint();
                    deckWidgets.progress.classList.remove("is-active");
                    for (const image of images) {
                        const result = { image, params: { ...runValues } };
                        gallery.add(result);
                        showResult(result);
                        persist.setResultPath(resultRelPath(image));
                    }
                },
                onError: (message) => {
                    queueCount -= 1;
                    updateHint();
                    deckWidgets.progress.classList.remove("is-active");
                    setStatus(t.runFailed(message));
                },
                onCancelled: () => {
                    queueCount -= 1;
                    updateHint();
                    deckWidgets.progress.classList.remove("is-active");
                },
            });
        } catch (err) {
            queueCount -= 1;
            updateHint();
            setStatus(t.runFailed(err.message));
        }
    }

    function setStatus(message) {
        caption.textContent = message;
        caption.style.display = "";
        console.warn("[TS Studio]", message);
    }

    // ── boot ────────────────────────────────────────────────────────────── //
    const firstAvailable = backends.find((b) => b.available);
    if (firstAvailable) {
        shell.setMode(firstAvailable.manifest.mode);
        selectBackend(firstAvailable);
    } else {
        const note = document.createElement("div");
        note.className = "ts-studio__galleryempty";
        note.textContent = t.noBackends;
        shell.deck.appendChild(note);
        for (const backend of backends) {
            console.warn(`[TS Studio] backend ${backend.id}:`, backend.problems.join("; "));
        }
    }
    return shell;
}

function buildOptionalIndex(objectInfo) {
    return (cls, inputName) =>
        Boolean(objectInfo[cls]?.input?.optional
            && inputName in objectInfo[cls].input.optional);
}
