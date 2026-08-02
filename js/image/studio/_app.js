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
import { uploadImage, makeDropZone, annotatedImageUrl } from "../../_studio/_dnd.js";
import { buildStudioState, studioStateFromPng } from "../../_studio/_pnginfo.js";
import { createQueuePanel } from "../../_studio/_queue.js";

// Rail tabs are UI modes, not backend modes. "Generate" covers both t2i and
// edit: the same act with or without reference images, so the user picks a
// model and the references appear when that model can use them (plan §9).
const UI_MODES = [
    { id: "generate", backendModes: ["t2i", "edit"] },
    { id: "inpaint", backendModes: ["inpaint"] },
    { id: "upscale", backendModes: ["upscale"] },
];

// Control kinds whose value belongs to the user, not to the backend file:
// they survive a model or mode switch. The prompt above all — a switch must
// never cost the text someone just wrote.
const STICKY_KINDS = new Set(["prompt", "seed", "size", "loras", "refs"]);

const ICONS = {
    generate: '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M12 3l1.8 4.7L18.5 9l-4.7 1.8L12 15.5l-1.8-4.7L5.5 9l4.7-1.3zM19 15l.9 2.3 2.1.7-2.1.9L19 21l-.9-2.1-2.1-.9 2.1-.7z"/></svg>',

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
        promptExpanding: "Preparing the prompt for this model…",
        designerReady: "A design is ready — it drives this render",
        designerEmpty: "No design yet — the prompt is used instead",
        designerMissing: "This editor is not installed",
        designerNeedsInput: "Write a prompt or open the designer first.",
        promptExpandFailed: (m) => `Could not prepare the prompt: ${m}`,
        format: "Format",
        sizeFromReference: "Size follows the reference image",
        resolution: "Resolution",
        seed: "Seed",
        randomize: "randomize",
        randomizeTip: "Random seed on every run — click to pin the current one",
        seedFixedTip: "Pinned seed — click to randomise on every run",
        seedDice: "Roll a new seed now and pin it",
        seedFieldTip: "Type a seed to reproduce an image exactly",
        seedHintRandom: "New seed every run",
        seedHintFixed: "This seed is used every run",
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
        modes: { generate: "Generate", t2i: "Generate", edit: "Edit",
                 inpaint: "Inpaint", upscale: "Upscale" },
        tabQueue: "Queue",
        recreate: "Recreate",
        recreateTip: "Restore the mode, settings and source image this result was made with",
        recreated: (f) => `Session recreated (${f}).`,
        sourceSet: "Image loaded as the source.",
        queue: {
            foreign: "Job from the graph",
            queueEmpty: "The queue is empty.",
            count: (r, p) => (r ? `running 1 · queued ${p}` : `queued ${p}`),
            stopRunning: "Stop",
            stopRunningTip: "Interrupt the job that is running now",
            clearPending: "Clear",
            clearPendingTip: "Remove every queued job except the running one",
            dropTip: "Remove this job from the queue",
            reorderTip: "Drag to reorder the queue",
        },
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
            stylesLoading: "Loading the style library…",
            stylesEmpty: "No styles match.",
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
        promptExpanding: "Готовлю промпт под эту модель…",
        designerReady: "Дизайн готов — рендер идёт по нему",
        designerEmpty: "Дизайна нет — используется промпт",
        designerMissing: "Этот редактор не установлен",
        designerNeedsInput: "Напишите промпт или откройте дизайнер.",
        promptExpandFailed: (m) => `Не удалось подготовить промпт: ${m}`,
        format: "Формат",
        sizeFromReference: "Размер берётся от референса",
        resolution: "Разрешение",
        seed: "Seed",
        randomize: "случайный",
        randomizeTip: "Случайный сид на каждый запуск — клик закрепит текущий",
        seedFixedTip: "Сид закреплён — клик включит случайный на каждый запуск",
        seedDice: "Сгенерировать новый сид и закрепить его",
        seedFieldTip: "Введите сид, чтобы точно повторить изображение",
        seedHintRandom: "Новый сид на каждый запуск",
        seedHintFixed: "Этот сид используется на каждом запуске",
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
        modes: { generate: "Генерация", t2i: "Генерация", edit: "Редактирование",
                 inpaint: "Inpaint", upscale: "Upscale" },
        tabQueue: "Очередь",
        recreate: "Повторить",
        recreateTip: "Восстановить режим, настройки и исходник, с которыми сделан результат",
        recreated: (f) => `Сессия восстановлена (${f}).`,
        sourceSet: "Изображение загружено как исходник.",
        queue: {
            foreign: "Задача из графа",
            queueEmpty: "Очередь пуста.",
            count: (r, p) => (r ? `выполняется 1 · в очереди ${p}` : `в очереди ${p}`),
            stopRunning: "Стоп",
            stopRunningTip: "Прервать выполняющуюся задачу",
            clearPending: "Очистить",
            clearPendingTip: "Убрать из очереди все задачи, кроме текущей",
            dropTip: "Убрать эту задачу из очереди",
            reorderTip: "Перетащите, чтобы изменить порядок",
        },
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
            stylesLoading: "Загружаю библиотеку стилей…",
            stylesEmpty: "Стили не найдены.",
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
.ts-istudio__caption{position:absolute;left:10px;bottom:10px;padding:3px 5px 3px 8px;
    display:flex;align-items:center;gap:8px;font-size:var(--ts-fs-sm);
    color:var(--ts-muted);background:var(--ts-elevated);border:1px solid var(--ts-border);
    border-radius:var(--ts-radius-sm)}
.ts-istudio__recreate{border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);
    background:none;color:var(--ts-accent);cursor:pointer;padding:1px 7px;
    font-size:var(--ts-fs-xs)}
.ts-istudio__recreate:hover{border-color:var(--ts-accent-line);background:var(--ts-accent-soft)}
.ts-istudio__recreate:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
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

    const present = new Set([...families.values()].flatMap((f) => [...f.modes.keys()]));
    const uiModes = UI_MODES.filter((m) => m.backendModes.some((b) => present.has(b)));
    const modeIds = uiModes.map((m) => m.id);
    const backendModesOf = (uiMode) =>
        UI_MODES.find((m) => m.id === uiMode)?.backendModes || [uiMode];

    /** Families offering any backend of this UI mode, with their roles. */
    function familiesForMode(uiMode) {
        const modes = backendModesOf(uiMode);
        const out = new Map();
        for (const family of families.values()) {
            const found = modes.map((m) => family.modes.get(m)).filter(Boolean);
            if (!found.length) continue;
            // The primary backend runs when no reference is filled in; the
            // edit backend takes over as soon as one is. Only Generate has
            // that pairing — Inpaint and Upscale must not sprout reference
            // slots just because the family can also edit (measured).
            const primary = family.modes.get(modes[0]) || found[0];
            const edit = modes.includes("edit") ? (family.modes.get("edit") || null) : null;
            out.set(family.family, { family, primary, edit, label: family.label });
        }
        return out;
    }

    const shell = createShell({
        label: t.appLabel,
        closeTitle: t.close,
        collapseTitle: t.collapse,
        modes: modeIds.map((id) => ({ id, title: t.modes[id] || id, icon: ICONS[id] || ICONS.generate })),
        onMode: (id) => selectMode(id),
        onClose: () => {
            stageDropTeardown?.();
            gallery.teardown?.();
            queuePanel.teardown();
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
                shell.setSideCollapsed(!shell.isSideCollapsed());
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
    const captionText = document.createElement("span");
    // Recreate lives with the image it describes, and only appears when the
    // studio actually knows how that image was made.
    const recreateButton = document.createElement("button");
    recreateButton.type = "button";
    recreateButton.className = "ts-istudio__recreate";
    recreateButton.textContent = t.recreate;
    recreateButton.title = t.recreateTip;
    recreateButton.style.display = "none";
    caption.append(captionText, recreateButton);
    stageFit.append(stageEmpty, stageImg);
    shell.stage.append(stageFit, caption);

    let selectedResult = null;
    // An image dropped into Upscale outranks the gallery selection.
    let upscaleSource = "";

    function setCaption(text, state) {
        captionText.textContent = text || "";
        recreateButton.style.display = state ? "" : "none";
        recreateButton.onclick = state ? () => {
            applyStudioState(state).catch((err) => setStatus(String(err?.message || err)));
        } : null;
        caption.style.display = (text || state) ? "" : "none";
    }

    function showResult(result) {
        selectedResult = result;
        upscaleSource = "";
        stageImg.src = `/view?filename=${encodeURIComponent(result.image.filename)}` +
            `&subfolder=${encodeURIComponent(result.image.subfolder || "")}&type=output`;
        stageImg.style.display = "";
        stageEmpty.style.display = "none";
        const params = result.params || {};
        const bits = [];
        if (params.width && params.height) bits.push(`${params.width} × ${params.height}`);
        if (params.seed !== undefined) bits.push(`seed ${params.seed}`);
        setCaption(bits.join(" · "), result.state || null);
    }

    function showLibraryAsset(asset) {
        stageImg.src = asset.url;
        stageImg.style.display = "";
        stageEmpty.style.display = "none";
        setCaption(asset.name || "", null);
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

    // ── asset panel: session results, library, queue ────────────────────── //
    const queuePanel = createQueuePanel({ api, t });
    const gallery = createGallery({
        t,
        extraTabs: [{
            id: "queue",
            label: t.tabQueue,
            element: queuePanel.element,
            onVisible: (visible) => queuePanel.setVisible(visible),
        }],
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

    // Values the user owns, carried across deck rebuilds. Captured from the
    // live controls just before they are torn down, so nothing is lost when a
    // model or a mode changes under the same deck.
    const sticky = new Map();       // param -> {kind, value}
    let stylesSticky = [];

    function captureSticky() {
        for (const [param, instance] of controlsByParam) {
            if (!STICKY_KINDS.has(instance.kind)) continue;
            try {
                sticky.set(param, { kind: instance.kind, value: instance.get() });
            } catch (err) {
                console.warn(`[TS Studio] could not keep '${param}'`, err);
            }
        }
        if (promptTools) stylesSticky = promptTools.getSelectedStyles();
    }

    function buildDeck(backend) {
        captureSticky();
        for (const instance of controlInstances) instance.teardown?.();
        controlInstances = [];
        controlsByParam = new Map();
        shell.deck.textContent = "";
        seedControl = null;
        promptTools?.teardown();
        promptTools = null;
        for (const key of Object.keys(values)) delete values[key];

        const roles = familiesForMode(activeModeId || backend.manifest.mode);
        const editBackend = roles.get(backend.manifest.family)?.edit || null;

        const modelSection = deckSection(t.model);
        const modelRow = document.createElement("div");
        modelRow.className = "ts-istudio__modelrow";
        const select = document.createElement("select");
        select.className = "ts-ui-select";
        for (const role of roles.values()) {
            const option = document.createElement("option");
            option.value = role.family.family;
            const usable = role.primary.available || role.edit?.available;
            option.textContent = usable ? role.label : `${role.label} — ${t.backendBroken}`;
            option.disabled = !usable;
            option.selected = role.family.family === backend.manifest.family;
            select.appendChild(option);
        }
        select.addEventListener("change", () => {
            const role = familiesForMode(activeModeId).get(select.value);
            const next = role?.primary?.available ? role.primary
                : (role?.edit?.available ? role.edit : null);
            if (next) selectBackend(next);
        });
        modelRow.appendChild(select);
        modelSection.appendChild(modelRow);
        shell.deck.appendChild(modelSection);

        const controls = [...(backend.manifest.controls || [])];
        // Reference slots follow the model's ability, not the current backend
        // file: a family with an edit backend shows them even while its
        // text-to-image backend is the one loaded. Filling a slot is what
        // switches the run over to the edit graph.
        const refsMax = Number(backend.manifest.refs?.max
            || editBackend?.manifest.refs?.max || 0);
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
                // A designer editor opens seeded with what the deck shows.
                getPrompt: () => controlsByParam.get("prompt")?.get() || "",
                getSize: () => controlsByParam.get("size")?.get(),
                onChange: (param, value) => {
                    values[param] = value;
                    // Replace ON implies 100% strength: grey the slider out.
                    if (param === "replace") {
                        controlsByParam.get("denoise")?.setDisabled?.(Boolean(value));
                    }
                    // A filled reference sends the run to the edit graph,
                    // which takes its frame from that image.
                    if (param === "__refs") {
                        const used = Object.values(value || {}).some(Boolean);
                        controlsByParam.get("size")?.setDisabled?.(used);
                    }
                },
            });
            instance.kind = control.kind;
            controlInstances.push(instance);
            if (control.param) controlsByParam.set(control.param, instance);
            const kept = sticky.get(control.param);
            if (kept && kept.kind === control.kind) {
                // What the user set outlives the file's default.
                instance.set(kept.value);
            } else if ((control.kind === "number" || control.kind === "slider"
                        || control.kind === "prompt")
                       && backend.spec.params.has(control.param)) {
                // Seed the control with the backend file's own default so the
                // deck never shows an empty field lying about what will run.
                const markerId = backend.spec.params.get(control.param).nodeId;
                const fileDefault = backend.graph[markerId]?.inputs?.value;
                if (fileDefault !== undefined && fileDefault !== "") {
                    instance.set(fileDefault);
                }
            }
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
                        initialStyles: stylesSticky,
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
        const roles = familiesForMode(modeId);
        const current = roles.get(activeBackend?.manifest.family);
        const pick = (role) => (role?.primary?.available ? role.primary
            : (role?.edit?.available ? role.edit : null));
        const next = pick(current) || [...roles.values()].map(pick).find(Boolean);
        if (next) selectBackend(next);
    }

    /**
     * Which graph a run actually submits. In Generate the answer depends on
     * the reference slots: empty means text-to-image, filled means the
     * family's edit graph — one rail tab, two backends.
     */
    function runBackend() {
        if (activeModeId !== "generate" || !activeBackend) return activeBackend;
        const role = familiesForMode("generate").get(activeBackend.manifest.family);
        const hasRef = Object.values(values.__refs || {}).some(Boolean);
        if (hasRef && role?.edit?.available) return role.edit;
        if (!hasRef && role?.primary?.available) return role.primary;
        return activeBackend;
    }

    // ── run ─────────────────────────────────────────────────────────────── //
    async function run() {
        if (!activeBackend) return;
        const target = runBackend();
        if (!target) return;

        const seedState = values.seed || { value: 0, randomize: true };
        const seed = seedState.randomize ? randomSeed() : Number(seedState.value || 0);
        seedControl?.showSeed(seed);

        const runValues = {};
        for (const [param, value] of Object.entries(values)) {
            if (param === "seed" || param === "loras" || param === "__refs") continue;
            if (typeof value === "object" && value !== null) continue;
            // The deck is built from one backend but a run may go to another
            // (Generate → edit): only params that graph actually declares.
            if (!target.spec.params.has(param) && !target.spec.literals?.has(param)) continue;
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
        // What the user typed — the metadata and the snapshot keep this, not
        // whatever a model-specific pipeline expands it into.
        const authoredPrompt = typeof runValues.prompt === "string" ? runValues.prompt : "";
        // Families whose node owns the prompt format (Ideogram's designer)
        // get their own path: the design the editor produced, or — when the
        // user only typed text — the node's Auto mode, fed by the same
        // SuperPrompt preset its editor uses.
        const designer = target.manifest.designer;
        if (designer && authoredPrompt !== undefined) {
            const prepared = await prepareDesignerRun(designer, authoredPrompt, runValues);
            if (!prepared) return;                  // status already explains
        }
        // Filled reference slots become image params; empty ones become
        // dropParams so the patcher removes their optional branches.
        const dropParams = [];
        for (const [name, annotated] of Object.entries(values.__refs || {})) {
            if (!target.spec.params.has(name)) continue;
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
            // A dropped image wins over the gallery selection: it is the more
            // deliberate act of the two.
            if (upscaleSource) {
                runValues.source_image = upscaleSource;
            } else if (selectedResult) {
                try {
                    const url = "/view?" + new URLSearchParams({
                        filename: selectedResult.image.filename,
                        subfolder: selectedResult.image.subfolder || "",
                        type: "output",
                    });
                    const blob = await (await fetch(url)).blob();
                    runValues.source_image =
                        await uploadImage(api, blob, selectedResult.image.filename);
                } catch (err) {
                    setStatus(t.runFailed(err.message));
                    return;
                }
            } else {
                setStatus(t.upscaleNeedsImage);
                return;
            }
        }
        // Required params are judged AFTER mode-specific collection — the
        // inpaint surface contributes source_image/mask right above.
        for (const required of target.manifest.requires || []) {
            const value = runValues[required] ?? values.__refs?.[required];
            if (!value) {
                setStatus(t.requiresMissing(required));
                return;
            }
        }

        const loras = Array.isArray(values.loras) ? values.loras : [];
        // The snapshot rides in the PNG so this exact session can be recreated
        // from the image alone — sources included.
        const studioState = buildStudioState({
            backendId: target.manifest.id,
            family: target.manifest.family,
            familyLabel: target.manifest.family_label,
            mode: target.manifest.mode,
            uiMode: activeModeId,
            values: { ...runValues, prompt: authoredPrompt || runValues.prompt },
            loras,
            styles: promptTools?.getStyleNames() || [],
            size: controlsByParam.get("size")?.get(),
            sources: {
                source_image: runValues.source_image,
                mask: runValues.mask,
                ...(values.__refs || {}),
            },
        });

        let patched;
        try {
            patched = patchBackend(target.graph, target.spec, {
                values: runValues,
                modelFiles: target.modelFiles,
                loras,
                dropParams,
                filenamePrefix: sessionPrefix(sessionId),
                isOptionalInput: optionalIndex,
                promptText: authoredPrompt || (typeof runValues.prompt === "string"
                    ? runValues.prompt : ""),
                studioState,
            });
        } catch (err) {
            setStatus(t.runFailed(err.message));
            return;
        }
        try {
            queueCount += 1;
            updateHint();
            await runner.submit(patched, {
                // The snapshot travels as extra_pnginfo so the saver writes
                // the ts_studio chunk even though the studio sends no
                // LiteGraph workflow of its own.
                pngInfo: { ts_studio: studioState },
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
                        const result = { image, params: { ...runValues }, state: studioState };
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
        setCaption(message, null);
        console.warn("[TS Studio]", message);
    }

    /**
     * Ideogram 4 reads a structured caption, not prose — TS_IdeogramDesigner
     * is the node that builds one, so the graph runs on its output. Two ways
     * in: a design from its editor (Designer mode), or the typed text turned
     * into a caption by the same SuperPrompt preset the editor's own Generate
     * button uses (Auto mode). Either way the node does the converting.
     *
     * @returns {Promise<boolean>} false when the run must not proceed.
     */
    async function prepareDesignerRun(designer, authoredPrompt, runValues) {
        const design = controlsByParam.get(designer.param)?.get();
        const size = controlsByParam.get("size")?.get();
        if (design) {
            const { applyFrameToDesign } = await import("../../_studio/_editors.js");
            runValues[designer.param] = JSON.stringify(
                applyFrameToDesign(design, size?.aspect, size?.mp));
            runValues[designer.mode_param || "mode"] = "designer";
            return true;
        }
        if (!authoredPrompt.trim()) {
            setStatus(t.designerNeedsInput);
            return false;
        }
        setStatus(t.promptExpanding);
        try {
            const response = await api.fetchApi("/ts_super_prompt/enhance", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: authoredPrompt, system_preset: designer.preset,
                                       seed: Date.now() }),
            });
            const payload = await response.json();
            if (!response.ok || payload.error) {
                throw new Error(payload.error || `HTTP ${response.status}`);
            }
            const caption = String(payload.text || "").trim();
            if (!caption) throw new Error("empty result");
            runValues[designer.caption_param || "auto_caption"] = caption;
            runValues[designer.mode_param || "mode"] = "auto";
            setStatus("");
            return true;
        } catch (err) {
            setStatus(t.promptExpandFailed(err.message));
            return false;
        }
    }

    // ── recreate: rebuild a whole session from a result ─────────────────── //
    //
    // The snapshot is the source of truth (mode, backend, every control, the
    // LoRA chain, the styles and the annotated sources). Applying it walks the
    // same paths a person would: pick the rail tab, pick the model, set the
    // controls, then put the sources back where that mode expects them.
    async function applyStudioState(state) {
        const backend = backends.find((b) => b.manifest?.id === state.backend && b.available)
            || backends.find((b) => b.manifest?.family === state.family
                && b.manifest?.mode === state.mode && b.available);
        if (!backend) {
            setStatus(t.pngNoBackend(state.backend));
            return true;
        }
        const uiMode = UI_MODES.find((m) => m.id === state.ui_mode)
            || UI_MODES.find((m) => m.backendModes.includes(backend.manifest.mode));
        // Sticky values must not fight the snapshot: it replaces them wholesale.
        sticky.clear();
        stylesSticky = [];
        selectMode(uiMode?.id || backend.manifest.mode);
        selectBackend(backend);

        for (const [param, value] of Object.entries(state.values || {})) {
            if (param === "width" || param === "height") continue;   // via size
            if (param === "seed") {
                controlsByParam.get("seed")?.set({ value: Number(value), randomize: false });
                continue;
            }
            controlsByParam.get(param)?.set(value);
            if (controlsByParam.has(param)) values[param] = value;
        }
        const width = Number(state.values?.width);
        const height = Number(state.values?.height);
        if (state.size?.aspect) {
            controlsByParam.get("size")?.set(state.size);
        } else if (width > 0 && height > 0) {
            const divisor = ((a, b) => { while (b) { [a, b] = [b, a % b]; } return a; })(width, height);
            controlsByParam.get("size")?.set({
                aspect: `${width / divisor}:${height / divisor}`,
                mp: (width * height) / 1e6,
            });
        }
        if (width > 0 && height > 0) { values.width = width; values.height = height; }
        controlsByParam.get("loras")?.set(state.loras || []);
        values.loras = state.loras || [];

        const sources = state.sources || {};
        const refs = ["ref_1", "ref_2", "ref_3", "ref_4", "ref_5", "ref_6"]
            .map((key) => sources[key] || "");
        if (refs.some(Boolean)) controlsByParam.get("__refs")?.set(refs);
        await restoreSources(sources, uiMode?.id || backend.manifest.mode);
        setStatus(t.recreated(backend.manifest.family_label || backend.manifest.family));
        return true;
    }

    /** Put the run's source image back into the surface its mode works on. */
    async function restoreSources(sources, uiMode) {
        const source = sources.source_image;
        if (!source) return;
        const url = annotatedImageUrl(source);
        if (!url) return;
        if (uiMode === "inpaint") {
            ensureInpaintMounted();
            await inpaintMode.setImageFromUrl(url).catch(() => {});
        } else if (uiMode === "upscale") {
            upscaleSource = source;
            stageImg.src = url;
            stageImg.style.display = "";
            stageEmpty.style.display = "none";
        }
    }

    /** A dropped image either recreates its session or becomes the source. */
    async function acceptDroppedImage(item) {
        const blob = await item.getBlob();
        const found = await studioStateFromPng(blob);
        if (found) return applyStudioState(found.state);
        const annotated = await uploadImage(api, blob, item.name || "dropped.png");
        if (activeModeId === "inpaint") {
            ensureInpaintMounted();
            await inpaintMode.setImageFromBlob(blob, item.name || "dropped.png");
        } else if (activeModeId === "upscale") {
            upscaleSource = annotated;
            stageImg.src = URL.createObjectURL(blob);
            stageImg.style.display = "";
            stageEmpty.style.display = "none";
        } else {
            const current = controlsByParam.get("__refs")?.get() || [];
            const slot = current.findIndex((v) => !v);
            if (slot < 0) return false;
            current[slot] = annotated;
            controlsByParam.get("__refs")?.set(current);
        }
        setStatus(t.sourceSet);
        return true;
    }

    const stageDropTeardown = makeDropZone(stageFit, {
        max: 1,
        onDrop: async ([item]) => {
            try {
                if (!await acceptDroppedImage(item)) setStatus(t.pngNotStudio);
            } catch (err) {
                setStatus(String(err?.message || err));
            }
        },
    });

    // ── boot ────────────────────────────────────────────────────────────── //
    const available = backends.filter((b) => b.available)
        .sort((a, b2) => (a.manifest.order || 99) - (b2.manifest.order || 99));
    const bootModes = backendModesOf(modeIds[0]);
    const firstAvailable = available.find((b) => bootModes.includes(b.manifest.mode))
        || available[0];
    if (firstAvailable) {
        selectMode(modeIds[0]);
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
