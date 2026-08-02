// TS Image Studio — application composition (app layer).
//
// Wiring only (plan §3.5): builds the shell, loads backends, renders the
// deck from the active backend's manifest through the control registry,
// submits runs through the runner and feeds the gallery. No business logic
// beyond composition lives here.

import { api } from "/scripts/api.js";
import { TS_UI_CLASS, ensureThemeStyles, pickLocaleStrings } from "../../_theme.js";
import { createShell, deckSection } from "../../_studio/_shell.js";
import { ensureControlStyles, getControlRenderer, randomSeed } from "../../_studio/_controls.js";
import { createGallery } from "../../_studio/_gallery.js";
import { createRunner } from "../../_studio/_runner.js";
import { loadBackends, groupByFamily } from "../../_studio/_backends.js";
import { patchBackend } from "../../_studio/_markers.js";
import { newSessionId, sessionPrefix, resultRelPath, restoreResults } from "../../_studio/_session.js";
import { mountPromptTools } from "../../_studio/_prompt_tools.js";
import { pickAssetProvider } from "../../_studio/_assets.js";
import { createInpaintMode } from "./_modes_inpaint.js";
import { createDownloadPanel } from "../../_studio/_downloads.js";
import { createHelpPanel } from "../../_studio/_help.js";
import { uploadImage, makeDropZone } from "../../_studio/_dnd.js";
import { studioRunFromPng } from "../../_studio/_pnginfo.js";

const ICONS = {
    t2i: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 3l1.8 4.7L18.5 9l-4.7 1.8L12 15.5l-1.8-4.7L5.5 9l4.7-1.3zM19 15l.9 2.3 2.1.7-2.1.9L19 21l-.9-2.1-2.1-.9 2.1-.7z"/></svg>',
    edit: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M4 20l4.5-1 11-11a2.1 2.1 0 0 0-3-3l-11 11zM13.5 6.5l3 3"/></svg>',
    inpaint: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 21c-4 0-7-2.5-7-6 0-4 4-5 5-9 .4-1.6 2.6-1.6 3 0 1 4 6 5 6 9 0 3.5-3 6-7 6z"/></svg>',
    upscale: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M4 20v-5m0 5h5m-5 0l6-6M20 4v5m0-5h-5m5 0l-6 6"/></svg>',
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
        libraryHint: "Recent server results. Drag into a reference slot; double-click to view.",
        libraryPickTip: "Drag into a slot · double-click to view on the stage",
        libraryEmpty: "No recent images on this server yet.",
        galleryEmpty: "Results of this session appear here.",
        stageEmpty: "Describe the image and press Run.",
        backendBroken: "unavailable",
        noBackends: "No backend workflows are available for any installed model.",
        runFailed: (m) => `Run failed: ${m}`,
        upscaleNeedsImage: "Select an image in the Session gallery first, then Run.",
        requiresMissing: (p) => `Add the required image first (${p}).`,
        pngRestored: (f) => `Settings restored from the image (${f}).`,
        pngNotStudio: "This image carries no studio settings.",
        pngNoBackend: (id) => `The image was made by backend '${id}', which is not available here.`,
        modes: { t2i: "Generate", edit: "Edit", inpaint: "Inpaint", upscale: "Upscale" },
        references: "References",
        refSlotTip: (n) => `Reference ${n}: drop an image here, or click to pick a file`,
        refClear: "Remove this reference",
        loraAdd: "+ Add LoRA",
        loraSearch: "Search LoRAs…",
        loraDrag: "Drag to reorder — the chain applies top to bottom",
        loraStrength: "Strength (negative values invert the effect)",
        loraRemove: "Remove this LoRA",
        loraNone: "No LoRA files installed",
        help: {
            helpHeader: "Help",
            hintsToggle: "Teaching tooltips on every control",
            closeLabel: "Close",
            missing: "Help pages are not available on this install.",
            open: "Help (F1)",
        },
        dl: {
            dlHeader: "Missing models",
            get: "Download",
            stop: "Stop",
            searching: "Searching on Hugging Face…",
            notFound: "Not found on Hugging Face — add the file manually.",
            waiting: "Queued…",
            verifying: "Verifying SHA256…",
            total: (n, d, s) => `${n} downloading · ${d} of ${s}`,
            status: (s) => ({done: "Done", error: "Failed", cancelled: "Cancelled"}[s] || s),
            doneHint: "Model downloaded — switch models to re-check.",
        },
        inp: {
            cleanup: "Cleanup", repaint: "Repaint",
            cleanupTip: "Instant object removal (LaMa): paint and release — no prompt needed",
            repaintTip: "Diffusion repaint: paint the mask, describe the change, press Run",
            brush: "Brush size ([ and ])",
            eraser: "Eraser — paint to remove mask",
            clear: "Clear the mask",
            undo: "Undo (Ctrl+Z)", redo: "Redo (Ctrl+Y)",
            empty: "Drop an image here, pick a session result, or drag from the Library.",
            cleaning: "Cleaning…",
            cleaned: (s) => `Cleaned in ${s} s`,
            repainted: "Repainted — the result is on the canvas and in the gallery.",
            needImage: "Add an image to inpaint first.",
            needMask: "Paint a mask first.",
            paintFailed: (m) => `Failed: ${m}`,
        },
        pt: {
            mic: "Dictate the prompt (click to start/stop)",
            hq: "High-quality voice model (slower, more accurate)",
            micDenied: "Microphone unavailable or denied.",
            transcribing: "Transcribing…",
            attach: "Attach an image — AI will combine it with your text",
            detach: "Remove the attached image",
            uploading: "Uploading image…",
            preset: "Enhance preset",
            enhance: "Enhance the prompt with AI",
            enhancing: "Enhancing the prompt…",
            enhancingImage: "Reading the image and combining with your text…",
            noSuperPrompt: "TS SuperPrompt is not available on this server",
            styles: "Style library",
            styleSearch: "Search styles…",
            removeStyle: "Click to remove this style",
            opFailed: (m) => `Failed: ${m}`,
        },
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
        libraryHint: "Недавние результаты сервера. Тяните в слот референса; двойной клик — просмотр.",
        libraryPickTip: "Тяните в слот · двойной клик — показать на холсте",
        libraryEmpty: "На сервере пока нет недавних изображений.",
        galleryEmpty: "Результаты этой сессии появятся здесь.",
        stageEmpty: "Опишите изображение и нажмите Run.",
        backendBroken: "недоступен",
        noBackends: "Нет доступных workflow ни для одной установленной модели.",
        runFailed: (m) => `Ошибка запуска: ${m}`,
        upscaleNeedsImage: "Сначала выберите изображение в галерее сессии, затем Run.",
        requiresMissing: (p) => `Сначала добавьте обязательное изображение (${p}).`,
        pngRestored: (f) => `Настройки восстановлены из изображения (${f}).`,
        pngNotStudio: "В этом изображении нет настроек студии.",
        pngNoBackend: (id) => `Изображение сделано бэкендом '${id}', он здесь недоступен.`,
        modes: { t2i: "Генерация", edit: "Редактирование", inpaint: "Inpaint", upscale: "Upscale" },
        references: "Референсы",
        refSlotTip: (n) => `Референс ${n}: перетащите картинку или кликните для выбора файла`,
        refClear: "Убрать этот референс",
        loraAdd: "+ Добавить LoRA",
        loraSearch: "Поиск LoRA…",
        loraDrag: "Перетащите, чтобы изменить порядок — цепочка применяется сверху вниз",
        loraStrength: "Сила (отрицательные значения инвертируют эффект)",
        loraRemove: "Убрать эту LoRA",
        loraNone: "Файлы LoRA не установлены",
        help: {
            helpHeader: "Справка",
            hintsToggle: "Обучающие подсказки на каждом контроле",
            closeLabel: "Закрыть",
            missing: "Страницы справки недоступны в этой установке.",
            open: "Справка (F1)",
        },
        dl: {
            dlHeader: "Недостающие модели",
            get: "Скачать",
            stop: "Стоп",
            searching: "Поиск на Hugging Face…",
            notFound: "Не найдено на Hugging Face — добавьте файл вручную.",
            waiting: "В очереди…",
            verifying: "Проверка SHA256…",
            total: (n, d, s) => `скачивается: ${n} · ${d} из ${s}`,
            status: (s) => ({done: "Готово", error: "Ошибка", cancelled: "Отменено"}[s] || s),
            doneHint: "Модель скачана — переключите модель для перепроверки.",
        },
        inp: {
            cleanup: "Cleanup", repaint: "Repaint",
            cleanupTip: "Мгновенное удаление объектов (LaMa): закрасьте и отпустите — промпт не нужен",
            repaintTip: "Диффузионная перерисовка: маска + описание изменения + Run",
            brush: "Размер кисти ([ и ])",
            eraser: "Ластик — стирает маску",
            clear: "Очистить маску",
            undo: "Отменить (Ctrl+Z)", redo: "Вернуть (Ctrl+Y)",
            empty: "Перетащите изображение, выберите результат сессии или тяните из Библиотеки.",
            cleaning: "Очистка…",
            cleaned: (s) => `Очищено за ${s} с`,
            repainted: "Перерисовано — результат на холсте и в галерее.",
            needImage: "Сначала добавьте изображение.",
            needMask: "Сначала нарисуйте маску.",
            paintFailed: (m) => `Ошибка: ${m}`,
        },
        pt: {
            mic: "Надиктовать промпт (клик — старт/стоп)",
            hq: "Качественная модель голоса (медленнее, точнее)",
            micDenied: "Микрофон недоступен или доступ запрещён.",
            transcribing: "Распознавание…",
            attach: "Приложить картинку — ИИ объединит её с вашим текстом",
            detach: "Убрать приложенную картинку",
            uploading: "Загрузка картинки…",
            preset: "Пресет улучшения",
            enhance: "Улучшить промпт ИИ",
            enhancing: "Улучшение промпта…",
            enhancingImage: "Читаю картинку и объединяю с вашим текстом…",
            noSuperPrompt: "TS SuperPrompt недоступен на этом сервере",
            styles: "Библиотека стилей",
            styleSearch: "Поиск стилей…",
            removeStyle: "Клик — убрать стиль",
            opFailed: (m) => `Ошибка: ${m}`,
        },
    },
};

const APP_STYLE_ID = "ts-image-studio-app-styles";

function ensureAppStyles() {
    ensureThemeStyles();
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
    const backends = await loadBackends((url) => fetch(url), objectInfo, (url) => api.fetchApi(url));
    const families = groupByFamily(backends);
    const sessionId = persist.sessionId || newSessionId();
    persist.setSessionId(sessionId);

    const optionalIndex = buildOptionalIndex(objectInfo);
    const runner = createRunner(api);
    const values = {};      // param -> value for the active backend
    let activeBackend = null;
    let queueCount = 0;

    const MODE_ORDER = ["t2i", "edit", "inpaint", "upscale"];
    const present = new Set([...families.values()].flatMap((f) => [...f.modes.keys()]));
    const modeIds = [...MODE_ORDER.filter((m) => present.has(m)),
                     ...[...present].filter((m) => !MODE_ORDER.includes(m))];
    const shell = createShell({
        label: t.appLabel,
        closeTitle: t.close,
        collapseTitle: t.collapse,
        modes: modeIds.map((id) => ({ id, title: t.modes[id] || id, icon: ICONS[id] || ICONS.t2i })),
        onMode: (id) => selectMode(id),
        onClose: () => {
            stageDropTeardown?.();
            gallery.teardown?.();
            helpPanel.teardown?.();
            inpaintMode?.teardown();
            for (const instance of controlInstances) instance.teardown?.();
            promptTools?.teardown();
            runner.destroy();
        },
        onKey: (event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                run();
            } else if (activeModeId === "inpaint" && inpaintMode
                       && (event.ctrlKey || event.metaKey)
                       && (event.key === "z" || event.key === "Z")) {
                event.preventDefault();
                inpaintMode.undo();
            } else if (activeModeId === "inpaint" && inpaintMode
                       && (event.ctrlKey || event.metaKey)
                       && (event.key === "y" || event.key === "Y")) {
                event.preventDefault();
                inpaintMode.redo();
            } else if (activeModeId === "inpaint" && inpaintMode
                       && (event.key === "[" || event.key === "]")) {
                event.preventDefault();
                inpaintMode.brushDelta(event.key === "]" ? 6 : -6);
            } else if (event.key === "F1") {
                event.preventDefault();
                helpPanel.toggle();
            } else if (event.key === "Tab") {
                event.preventDefault();
                shell.setSideCollapsed(!shell.side.classList.contains("is-collapsed"));
            }
        },
    });

    // ── stage ───────────────────────────────────────────────────────────── //
    const stageFit = document.createElement("div");
    // Own token scope (nested scopes are harmless): stage chrome keeps its
    // colours even if a mode ever hoists it out of the shell.
    stageFit.className = `${TS_UI_CLASS} ts-istudio__stagefit`;
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

    let selectedResult = null;

    function showResult(result) {
        selectedResult = result;
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

    function showLibraryAsset(asset) {
        stageImg.src = asset.url;
        stageImg.style.display = "";
        stageEmpty.style.display = "none";
        caption.textContent = asset.name || "";
        caption.style.display = asset.name ? "" : "none";
    }

    function showPreviewBlob(blob) {
        stageImg.src = URL.createObjectURL(blob);
        stageImg.style.display = "";
        stageEmpty.style.display = "none";
    }

    const helpPanel = createHelpPanel({
        stage: shell.stage, t, locale,
        studioRoot: shell.root,
        pagesBase: "/extensions/comfyui-timesaver/image/studio/help",
    });
    const helpButton = document.createElement("button");
    helpButton.type = "button";
    helpButton.className = "ts-studio__railbtn";
    helpButton.title = t.help.open;
    helpButton.setAttribute("aria-label", t.help.open);
    helpButton.textContent = "?";
    helpButton.addEventListener("click", () => helpPanel.toggle());
    shell.rail.appendChild(helpButton);

    // ── gallery (right panel) ───────────────────────────────────────────── //
    const gallery = createGallery({
        t,
        onSelect: (result) => {
            showResult(result);
            persist.setResultPath(resultRelPath(result.image));
        },
        mountLibrary: (host) => {
            let handle = null;
            pickAssetProvider().then((provider) => {
                if (!provider) return;
                handle = provider.mount(host, {
                    api, t,
                    onPick: (asset) => showLibraryAsset(asset),
                });
            });
            return { unmount: () => handle?.unmount?.() };
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
    let promptTools = null;
    let controlInstances = [];
    let controlsByParam = new Map();
    const loraOptions = readLoraOptions(objectInfo);

    function buildDeck(backend) {
        for (const instance of controlInstances) instance.teardown?.();
        controlInstances = [];
        controlsByParam = new Map();
        shell.deck.textContent = "";
        seedControl = null;
        promptTools?.teardown();
        promptTools = null;
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

        const controls = [...(backend.manifest.controls || [])];
        // Reference slots come from the manifest's refs block; insert the
        // control right after the prompt unless the author placed one.
        const refsMax = Number(backend.manifest.refs?.max || 0);
        if (refsMax > 0 && !controls.some((c) => c.kind === "refs")) {
            const promptIndex = controls.findIndex((c) => c.kind === "prompt");
            controls.splice(promptIndex + 1, 0, { kind: "refs", max: refsMax, param: "__refs" });
        }

        const advanced = [];
        for (const control of controls) {
            const renderer = getControlRenderer(control.kind);
            if (!renderer) {
                console.warn(`[TS Studio] no renderer for control kind '${control.kind}' — skipped`);
                continue;
            }
            const instance = renderer(control, {
                t, locale, loraOptions,
                uploadImage: (blob, name) => uploadImage(api, blob, name),
                onChange: (param, value) => {
                    values[param] = value;
                    // Replace ON implies 100% strength: grey the slider out.
                    if (param === "replace") {
                        controlsByParam.get("denoise")?.setDisabled?.(Boolean(value));
                    }
                },
            });
            controlInstances.push(instance);
            if (control.param) controlsByParam.set(control.param, instance);
            if (control.kind === "seed") seedControl = instance;
            if (control.advanced) advanced.push(instance.element);
            else shell.deck.appendChild(instance.element);
            if (control.kind === "prompt" && control.superprompt) {
                const slot = instance.element.querySelector(
                    `[data-ts-slot="prompt-toolbar:${control.param}"]`);
                const textarea = instance.element.querySelector("textarea");
                if (slot && textarea) {
                    promptTools = mountPromptTools({
                        textarea, slot, api, objectInfo, t, locale,
                    });
                }
            }
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

        const problems = [];
        for (const candidate of backends) {
            if (candidate.available || candidate.manifest?.mode !== backend.manifest.mode) continue;
            for (const reason of candidate.problems) {
                const match = /no installed file matches '([^']+)' for (\S+)/.exec(reason);
                if (!match) continue;
                const modelSpec = (candidate.manifest.models || [])
                    .find((m) => m.title === match[2]);
                problems.push({
                    familyLabel: candidate.manifest.family_label || candidate.manifest.family,
                    filenameHint: match[1].replace(/[^\w.-]+/g, "_"),
                    folder: modelSpec?.folder || "checkpoints",
                });
            }
        }
        let downloadPanel = null;
        if (problems.length) {
            downloadPanel = createDownloadPanel({
                api, t, problems,
                onResolved: () => setStatus(t.dl.doneHint),
            });
            controlInstances.push(downloadPanel);
            shell.deck.appendChild(downloadPanel.element);
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

    let inpaintMode = null;
    let activeModeId = null;

    function ensureInpaintMounted() {
        if (!inpaintMode) {
            inpaintMode = createInpaintMode({
                api, t, sessionId,
                onEngineChange: () => {},
            });
            shell.stage.appendChild(inpaintMode.element);
            const selectedUrl = stageImg.src && stageImg.style.display !== "none" ? stageImg.src : "";
            if (selectedUrl) {
                inpaintMode.setImageFromUrl(selectedUrl).catch(() => {});
            }
        }
        inpaintMode.element.style.display = "";
        stageFit.style.display = "none";
        caption.style.display = "none";
    }

    function leaveInpaint() {
        if (inpaintMode) inpaintMode.element.style.display = "none";
        stageFit.style.display = "";
    }

    function selectMode(modeId) {
        shell.setMode(modeId);
        activeModeId = modeId;
        if (modeId === "inpaint") ensureInpaintMounted();
        else leaveInpaint();
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
            if (param === "seed" || param === "loras" || param === "__refs") continue;
            if (typeof value === "object" && value !== null) continue;
            runValues[param] = value;
        }
        runValues.seed = seed;
        if (typeof values.replace === "boolean") {
            runValues.replace = values.replace;
            if (values.replace) runValues.denoise = 1.0;
        }
        // Styles append to the prompt the same way the selector node does —
        // what runs is exactly what the gallery params will replay.
        const styleTail = promptTools?.getStylePrompts().join(", ");
        if (styleTail && typeof runValues.prompt === "string") {
            const base = runValues.prompt.trim().replace(/[,\s]+$/, "");
            runValues.prompt = base ? `${base}, ${styleTail}` : styleTail;
        }
        // Filled reference slots become image params; empty ones become
        // dropParams so the patcher removes their optional branches.
        const dropParams = [];
        for (const [name, annotated] of Object.entries(values.__refs || {})) {
            if (!activeBackend.spec.params.has(name)) continue;
            if (annotated) runValues[name] = annotated;
            else dropParams.push(name);
        }

        if (activeModeId === "inpaint" && inpaintMode) {
            try {
                const collected = await inpaintMode.collectRunValues();
                Object.assign(runValues, collected);
            } catch (err) {
                setStatus(String(err.message || err));
                return;
            }
        }
        if (activeModeId === "upscale") {
            if (!selectedResult) {
                setStatus(t.upscaleNeedsImage);
                return;
            }
            try {
                const url = "/view?" + new URLSearchParams({
                    filename: selectedResult.image.filename,
                    subfolder: selectedResult.image.subfolder || "",
                    type: "output",
                });
                const blob = await (await fetch(url)).blob();
                runValues.source_image = await uploadImage(api, blob, selectedResult.image.filename);
            } catch (err) {
                setStatus(t.runFailed(err.message));
                return;
            }
        }
        // Required params are judged AFTER mode-specific collection — the
        // inpaint surface contributes source_image/mask right above.
        for (const required of activeBackend.manifest.requires || []) {
            const value = runValues[required] ?? values.__refs?.[required];
            if (!value) {
                setStatus(t.requiresMissing(required));
                return;
            }
        }

        let patched;
        try {
            patched = patchBackend(activeBackend.graph, activeBackend.spec, {
                values: runValues,
                modelFiles: activeBackend.modelFiles,
                loras: Array.isArray(values.loras) ? values.loras : [],
                dropParams,
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
                onPreview: (blob) => {
                    if (activeModeId === "inpaint" && inpaintMode) inpaintMode.showPreview(blob);
                    else showPreviewBlob(blob);
                },
                onDone: (images) => {
                    queueCount -= 1;
                    updateHint();
                    deckWidgets.progress.classList.remove("is-active");
                    for (const image of images) {
                        const result = { image, params: { ...runValues } };
                        gallery.add(result);
                        showResult(result);
                        persist.setResultPath(resultRelPath(image));
                        if (activeModeId === "inpaint" && inpaintMode) {
                            inpaintMode.hidePreview();
                            const url = "/view?" + new URLSearchParams({
                                filename: image.filename,
                                subfolder: image.subfolder || "",
                                type: "output",
                            });
                            inpaintMode.acceptRepaintResult(url, image.filename)
                                .catch(() => {});
                        }
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

    // ── reproducibility: drop a studio PNG to restore its run ───────────── //
    async function restoreFromPng(blob) {
        const run = await studioRunFromPng(blob);
        if (!run) return false;
        const backend = backends.find((b) => b.manifest?.id === run.backendId && b.available)
            || backends.find((b) => b.manifest?.family === run.family
                && b.manifest?.mode === run.mode && b.available);
        if (!backend) {
            setStatus(t.pngNoBackend(run.backendId));
            return true;
        }
        selectMode(backend.manifest.mode);
        selectBackend(backend);
        for (const [param, value] of Object.entries(run.values)) {
            if (param === "seed") {
                controlsByParam.get("seed")?.set({ value: Number(value), randomize: false });
            } else if (param === "width" || param === "height") {
                continue; // handled below through the size control
            } else {
                controlsByParam.get(param)?.set(value);
                if (param in values || controlsByParam.has(param)) values[param] = value;
            }
        }
        const width = Number(run.values.width);
        const height = Number(run.values.height);
        if (width > 0 && height > 0) {
            const divisor = ((a, b) => { while (b) { [a, b] = [b, a % b]; } return a; })(width, height);
            controlsByParam.get("size")?.set({
                aspect: `${width / divisor}:${height / divisor}`,
                mp: (width * height) / 1e6,
            });
            values.width = width;
            values.height = height;
        }
        controlsByParam.get("loras")?.set(run.loras);
        if (run.loras.length) values.loras = run.loras;
        setStatus(t.pngRestored(backend.manifest.family_label || backend.manifest.family));
        return true;
    }

    const stageDropTeardown = makeDropZone(stageFit, {
        max: 1,
        onDrop: async ([item]) => {
            try {
                const blob = await item.getBlob();
                if (await restoreFromPng(blob)) return;
                setStatus(t.pngNotStudio);
            } catch (err) {
                setStatus(String(err?.message || err));
            }
        },
    });

    // ── boot ────────────────────────────────────────────────────────────── //
    const available = backends.filter((b) => b.available)
        .sort((a, b2) => (a.manifest.order || 99) - (b2.manifest.order || 99));
    const firstAvailable = available.find((b) => b.manifest.mode === modeIds[0]) || available[0];
    if (firstAvailable) {
        selectMode(firstAvailable.manifest.mode);
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

function readLoraOptions(objectInfo) {
    const spec = objectInfo?.LoraLoaderModelOnly?.input?.required?.lora_name;
    if (Array.isArray(spec?.[0])) return spec[0];
    if (Array.isArray(spec?.[1]?.options)) return spec[1].options;
    return [];
}
