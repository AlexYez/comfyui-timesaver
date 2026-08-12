// TS Image Studio — application composition (app layer).
//
// Wiring only (plan §3.5): builds the shell, loads backends, renders the
// deck from the active backend's manifest through the control registry,
// submits runs through the runner and feeds the gallery. No business logic
// beyond composition lives here.

import { api } from "/scripts/api.js";
import { TS_UI_CLASS, ensureThemeStyles, getUiLanguage, pickLocaleStrings } from "../../_theme.js";
import { createShell, deckSection } from "../../_studio/_shell.js";
import { ensureControlStyles, getControlRenderer, randomSeed } from "../../_studio/_controls.js";
import { createGallery } from "../../_studio/_gallery.js";
import { splitterGrid } from "../../_studio/_tilegrid.js";
import { createStage } from "./_stage.js";
import { createRunProgress } from "./_progress.js";
import { backendForRun, buildOffers, offersForModes, packReadiness,
    rolesForModes } from "./_catalog.js";
import { createRunner } from "../../_studio/_runner.js";
import { createHistory } from "../../_studio/_history.js";
import { createIterations, ITERATIONS_CSS } from "../../_studio/_iterate.js";
import { createBatch, seedFor } from "../../_studio/_batch.js";
import { createBatchPanel, BATCH_CSS } from "../../_studio/_batchpanel.js";
import { promptFromPng } from "../../_studio/_pnginfo.js";
import { applyPackState, loadBackends, groupByFamily } from "../../_studio/_backends.js";
import { patchBackend } from "../../_studio/_markers.js";
import { newSessionId, outputPrefix, resultAnnotated, resultRelPath, resultViewUrl,
         restoreResults } from "../../_studio/_session.js";
import { mountPromptTools } from "../../_studio/_prompt_tools.js";
import { promptPresetsFor } from "../../_studio/_prompt_presets.js";
import { parseFrameAspect } from "../../_studio/_outframe.js";
import { pickAssetProvider } from "../../_studio/_assets.js";
import { createInpaintMode } from "./_modes_inpaint.js";
import { createDownloadPanel } from "../../_studio/_downloads.js";
import { createHelpPanel } from "../../_studio/_help.js";
import { commandsFor, createContextMenu, registerStudioStageCommands }
    from "../../_studio/_ctxmenu.js";
import { extractFrame, frameChoiceItems, isVideoItem }
    from "../../_studio/_videoframe.js";
import { createSettingsPanel, readSetting, settingsStrings }
    from "../../_studio/_settings.js";
import { uploadImage, makeDropZone, annotatedImageUrl } from "../../_studio/_dnd.js";
import { buildStudioState } from "../../_studio/_pnginfo.js";
import { loadWorkspace, saveWorkspace } from "../../_studio/_workspace.js";
import { applyStrengthRule, collectDeckValues, collectRefs,
    withStyles } from "./_runvalues.js";
import { createQueuePanel } from "../../_studio/_queue.js";
import { createGate } from "../../_studio/_gate.js";
import { createShowcase } from "../../_studio/_showcase.js";
import * as memory from "../../_studio/_memory.js";

// Rail tabs are UI modes, not backend modes. "Generate" covers both t2i and
// edit: the same act with or without reference images, so the user picks a
// model and the references appear when that model can use them (plan §9).
// Сколько первых латентных превью не показывать. Первые шаги — почти чистый
// шум: быстрый декодер показывает его честно, и поверх лица это выглядит
// пугающе, а полезного там ещё нет.
const PREVIEW_SKIP_STEPS = 2;
// В расширении хватает одного: со второго шага уже видно, что дорисовывается.
const OUTPAINT_SKIP_STEPS = 1;

// Разделы интерфейса. Раздел появляется в рельсе, только если хотя бы одно
// живое семейство умеет его режим: пустых вкладок в студии не бывает — модели
// приезжают паками, и нечем занять раздел, для которого пак не установлен.
const UI_MODES = [
    { id: "generate", backendModes: ["t2i", "edit"] },
    // Перерисовать картинку целиком просят чаще, чем что-либо кроме генерации:
    // без нарезки, без увеличения — только денойз и разрешение под модель.
    { id: "img2img", backendModes: ["img2img"] },
    { id: "inpaint", backendModes: ["inpaint"] },
    { id: "outpaint", backendModes: ["outpaint"] },
    { id: "upscale", backendModes: ["upscale"] },
    // Пачка идёт последней: это не ещё один способ сделать кадр, а способ
    // сделать их двадцать. Работает на той же модели, что и генерация.
    { id: "batch", backendModes: ["t2i"] },
];

// Режимы, работающие НАД картинкой на сцене: им нужен исходник, и они его
// показывают. Отличаются от генерации, которая начинается с пустого места.
const SOURCE_MODES = new Set(["img2img", "upscale", "outpaint"]);
// Разделы, где человек ИТЕРИРУЕТ: прогнал — посмотрел — вернулся — прогнал
// иначе. Их результаты идут во временную папку и попадают в библиотеку только
// по кнопке Save. Инпэйнт живёт по тому же правилу, но своей панелью: у него
// холст, маска и собственный откат мазков.
const ITERATIVE_MODES = new Set(["upscale", "img2img"]);

// Marks a picker entry that stands for a model the catalogue offers but this
// machine does not have. Prefixed so it can never collide with a family id.
const GHOST_PREFIX = "offer:";

// Rail icons. All drawn on the same 24 grid, same stroke weight, round joins —
// each shape is symmetric about its own centre so nothing reads as skewed at
// 17px. Generate: a four-point spark. Inpaint: a pencil, because that is what
// the mode does. Upscale: corners pushed outward. Settings: sliders, which
// stay legible at this size where a toothed gear turns to mush.
const ICON_ATTRS = 'viewBox="0 0 24 24" width="17" height="17" fill="none" '
    + 'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
    + 'stroke-linejoin="round"';

const ICONS = {
    generate: `<svg ${ICON_ATTRS}><path d="M11 3.5l1.7 4.3 4.3 1.7-4.3 1.7L11 15.5 9.3 11.2 5 9.5l4.3-1.7z"/><path d="M18 14.5l.85 2.15 2.15.85-2.15.85L18 20.5l-.85-2.15L15 17.5l2.15-.85z"/></svg>`,
    // Image to Image: кадр перерисовывается в кадр — две рамки и стрелка между
    // ними. Ни увеличения, ни нарезки в знаке нет, потому что их нет и в режиме.
    img2img: `<svg ${ICON_ATTRS}><rect x="2.5" y="6" width="8" height="8" rx="1.2"/><rect x="13.5" y="10" width="8" height="8" rx="1.2"/><path d="M11.4 8.6h3.4"/><path d="M13.2 7.1l1.6 1.5-1.6 1.5"/></svg>`,
    inpaint: `<svg ${ICON_ATTRS}><path d="M4 20l.9-3.7L15.6 5.6a2.05 2.05 0 0 1 2.9 2.9L7.7 19.1z"/><path d="M13.9 7.3l2.8 2.8"/></svg>`,
    upscale: `<svg ${ICON_ATTRS}><path d="M4 9V4h5"/><path d="M20 15v5h-5"/><path d="M4 4l6 6"/><path d="M20 20l-6-6"/></svg>`,
    // Расширение кадра: рамка пошире вокруг рамки поуже — ровно то, что
    // происходит с картинкой.
    outpaint: `<svg ${ICON_ATTRS}><rect x="3" y="5" width="18" height="14" rx="1.5"/><rect x="8" y="9" width="8" height="6" rx="1"/></svg>`,
    // Пачка: стопка кадров одного размера, уходящая вглубь. Не список строк —
    // на выходе получаются картинки, и знак говорит именно об этом.
    batch: `<svg ${ICON_ATTRS}><rect x="3" y="8.5" width="12" height="12" rx="1.4"/><path d="M6.5 5.5h11a1.5 1.5 0 0 1 1.5 1.5v10"/><path d="M9.5 2.5h11A1.5 1.5 0 0 1 22 4v10"/></svg>`,
    settings: `<svg ${ICON_ATTRS}><path d="M4 7h7M15 7h5M4 12h11M19 12h1M4 17h3M11 17h9"/><circle cx="13" cy="7" r="2"/><circle cx="17" cy="12" r="2"/><circle cx="9" cy="17" r="2"/></svg>`,
    // Packs: a box seen head-on, with the seam a parcel has.
    packs: `<svg ${ICON_ATTRS}><path d="M4 8.5l8-4 8 4v7l-8 4-8-4z"/><path d="M4 8.5l8 4 8-4"/><path d="M12 12.5v7"/></svg>`,
};

const STRINGS = {
    en: {
        appLabel: "TS Image Studio",
        close: "Close (Esc)",
        collapse: "Collapse or expand the asset panel (Ctrl+B)",
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
        aspectCustom: "w:h",
        aspectCustomTip: "Your own ratio, as width:height — 21:9, 5:4, 2.35:1",
        sizeFromReference: "Size follows the reference image",
        resolution: "Resolution",
        resolutionTitle: "How many megapixels the result has — the aspect ratio stays",
        seed: "Seed",
        randomize: "randomize",
        randomizeTip: "Random seed on every run — click to pin the current one",
        seedFixedTip: "Pinned seed — click to randomise on every run",
        seedDice: "Roll a new seed now and pin it",
        seedFieldTip: "Type a seed to reproduce an image exactly",
        seedHintRandom: "New seed every run",
        seedHintFixed: "This seed is used every run",
        advanced: "Advanced",
        advancedTitle: "Show the settings that are rarely touched",
        modelTitle: "Which model does the work",
        stages: {
            load: "Loading the model",
            prompt: "Reading the prompt",
            encode: "Preparing the image",
            sample: "Drawing",
            decode: "Assembling the picture",
            crop: "Taking the crop",
            restore: "Putting it back",
            save: "Saving",
            other: "Working",
        },
        cmp: { before: "before", after: "after",
               shown: "Drag the divider — the original is on the left." },
        iter: {
            undo: "Previous version",
            redo: "Next version",
            keep: "Save",
            kept: "Saved",
            keptMsg: "Saved to the library.",
            keepFailed: (m) => `Could not save: ${m}`,
        },
        fitView: "Fit to the work area (double-click, or wheel to zoom)",
        run: "Run",
        runTitle: "Start the job with the current settings",
        sourceGone: "The image of this sitting is no longer on disk — drop it in again.",
        stop: "Stop",
        stopTip: "Stop the run in progress. Anything running from the canvas is left alone.",
        stopAll: "Stop all",
        stopAllTip: "Drop every studio run still queued.",
        stopping: "Stopping the current run…",
        stoppingAll: (n) => `Stopping ${n} run${n === 1 ? "" : "s"}…`,
        runHint: "Ctrl+Enter",
        queued: (n) => `${n} in queue`,
        cancel: "Cancel",
        generating: (p) => `Generating… ${p}%`,
        tabSession: "Session",
        tabSessionTitle: "What this session has produced, newest first",
        tabLibrary: "Library",
        tabLibraryTitle: "Everything saved to the output folder",
        libraryHint: "Recent server results. Drag into a reference slot; double-click to view.",
        libraryPickTip: "Drag into a slot · double-click to view on the stage",
        libraryEmpty: "No recent images on this server yet.",
        galleryEmpty: "Results of this session appear here.",
        stageEmpty: "Describe the image and press Run",
        stageEmptyHint: "The result appears here. Drop a picture onto this area to work from it.",
        menuSendTo: (mode) => `Send to ${mode}`,
        menuAsReference: "Goes into a reference slot",
        menuClear: "Clear the canvas",
        menuClearHint: "Only this tab",
        menuSent: (mode) => `Sent to ${mode}`,
        upscaleStarting: "Starting upscale",
        droppedAsReference: (slot) => `Added to reference ${slot}`,
        refsFull: "All reference slots are taken — free one first.",
        outpainting: "Outpainting",
        videoFirst: "Take the first frame",
        videoLast: "Take the last frame",
        videoHint: "A video was dropped — the studio works on one frame",
        videoTaking: "Taking the frame…",
        videoFailed: (m) => `Could not take a frame: ${m}`,
        warmupTitle: "Getting ready",
        warmupNote: "The model is loading. The picture starts appearing as soon as drawing begins.",
        backendBroken: "unavailable",
        noBackends: "No backend workflows are available for any installed model.",
        runFailed: (m) => `Run failed: ${m}`,
        upscaleNeedsImage: "Select an image in the Session gallery first, then Run.",
        requiresMissing: (p) => `Add the required image first (${p}).`,
        pngRestored: (f) => `Settings restored from the image (${f}).`,
        pngNotStudio: "This image carries no studio settings.",
        inPack: "in a pack",
        pngNoBackend: (id) => `The image was made by backend '${id}', which is not available here.`,
        modes: { generate: "Generate", t2i: "Generate", edit: "Edit",
                 img2img: "Image to Image",
                 inpaint: "Inpaint", outpaint: "Outpaint", upscale: "Upscale",
                 batch: "Batch" },
        batch: {
            title: "Batch",
            placeholder: "One prompt per line.",
            dropHint: "Drop images here — their prompts are read out and added.",
            enhance: "Enhance prompts first",
            enhanceTip: "All prompts go through the model in one pass, before any "
                + "image is drawn: loading it once costs a minute, loading it "
                + "twenty times costs twenty.",
            seeds: "Seeds",
            seedNames: {
                fixed: "same for all",
                series: "in order",
                random: "random",
            },
            baseSeed: "from",
            start: "Start", stop: "Stop",
            empty: "Nothing queued yet.",
            count: (n) => `${n} in the queue`,
            states: { waiting: "waiting", running: "working", done: "done",
                      failed: "failed", skipped: "skipped" },
            phasePrompts: (done, total) => `Prompts: ${done} of ${total}`,
            phaseImages: (done, total) => `Images: ${done} of ${total}`,
            reading: "Reading prompts out of the images…",
            readNone: "No prompts in those images.",
            readSome: (n) => `${n} prompt(s) added.`,
            finished: (done, failed) => failed
                ? `Done: ${done} in the library, ${failed} failed.`
                : `Done: ${done} in the library.`,
            needModel: "Pick a model first.",
        },
        outNeedsImage: "Bring in an image first — outpaint extends what is there.",
        i2iNeedsImage: "Bring in an image first — this mode redraws the one you have.",
        outFrame: (w, h) => `${w} × ${h}`,
        tabQueue: "Queue",
        tabQueueTitle: "Jobs waiting to run and the one running now",
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
        loraAddTitle: "Search the LoRAs installed here and add one",
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
            eraser: "Eraser — paint to take mask away (E, or hold Alt)",
            fit: "Fit to the work area (double-click, or wheel to zoom)",
            keep: "Save",
            keepTip: "Keep this version in the library. Repaints are drafts until you do.",
            kept: "Saved to the library.",
            keepFailed: (m) => `Could not save: ${m}`,
            brushMode: "Brush",
            brushModeTip: "Paint the mask (E switches, Alt erases while held)",
            eraserMode: "Eraser",
            clear: "Clear the mask",
            undo: "Undo (Ctrl+Z)", redo: "Redo (Ctrl+Y)",
            empty: "Drop an image here, pick a session result, or drag from the Library.",
            cleaning: "Cleaning…",
            cleaned: (s) => `Cleaned in ${s} s`,
            repainted: "Repainted — the result is on the canvas and in the gallery.",
            needImage: "Add an image to inpaint first.",
            needMask: "Paint a mask first.",
            // Cleanup consumes the mask the moment a stroke ends, so "paint a
            // mask" is true and useless — the person did paint one.
            needRepaint: "Cleanup runs on its own as you paint. To have a model "
                + "repaint the area, switch to Repaint first.",
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
            library: "Ready-made prompts for refining detail",
            librarySearch: "Search prompts…",
            libraryEmpty: "No prompts match.",
            opFailed: (m) => `Failed: ${m}`,
        },
    },
    ru: {
        appLabel: "TS Image Studio",
        close: "Закрыть (Esc)",
        collapse: "Свернуть или развернуть панель ассетов (Ctrl+B)",
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
        aspectCustom: "ш:в",
        aspectCustomTip: "Своё соотношение в виде ширина:высота — 21:9, 5:4, 2.35:1",
        sizeFromReference: "Размер берётся от референса",
        resolution: "Разрешение",
        resolutionTitle: "Сколько мегапикселей в результате — пропорция при этом сохраняется",
        seed: "Seed",
        randomize: "случайный",
        randomizeTip: "Случайный сид на каждый запуск — клик закрепит текущий",
        seedFixedTip: "Сид закреплён — клик включит случайный на каждый запуск",
        seedDice: "Сгенерировать новый сид и закрепить его",
        seedFieldTip: "Введите сид, чтобы точно повторить изображение",
        seedHintRandom: "Новый сид на каждый запуск",
        seedHintFixed: "Этот сид используется на каждом запуске",
        advanced: "Дополнительно",
        advancedTitle: "Показать настройки, которые трогают редко",
        modelTitle: "Какая модель будет работать",
        stages: {
            load: "Загружаю модель",
            prompt: "Читаю промпт",
            encode: "Готовлю изображение",
            sample: "Рисую",
            decode: "Собираю картинку",
            crop: "Беру вырез",
            restore: "Возвращаю на место",
            save: "Сохраняю",
            other: "Работаю",
        },
        cmp: { before: "до", after: "после",
               shown: "Тяните шторку — слева исходник." },
        iter: {
            undo: "Предыдущая версия",
            redo: "Следующая версия",
            keep: "Сохранить",
            kept: "Сохранено",
            keptMsg: "Сохранено в библиотеку.",
            keepFailed: (m) => `Не удалось сохранить: ${m}`,
        },
        fitView: "Вписать в рабочую область (двойной клик; колесо — масштаб)",
        run: "Run",
        runTitle: "Запустить с текущими настройками",
        sourceGone: "Картинки этой сессии больше нет на диске — перетащите её заново.",
        stop: "Стоп",
        stopTip: "Остановить текущий прогон. То, что считается с холста, не трогается.",
        stopAll: "Снять очередь",
        stopAllTip: "Убрать из очереди все прогоны студии.",
        stopping: "Останавливаю текущий прогон…",
        stoppingAll: (n) => `Останавливаю прогонов: ${n}…`,
        runHint: "Ctrl+Enter",
        queued: (n) => `в очереди: ${n}`,
        cancel: "Отмена",
        generating: (p) => `Генерация… ${p}%`,
        tabSession: "Сессия",
        tabSessionTitle: "Что получилось в этой сессии, новое сверху",
        tabLibrary: "Библиотека",
        tabLibraryTitle: "Всё, что сохранено в папку output",
        libraryHint: "Недавние результаты сервера. Тяните в слот референса; двойной клик — просмотр.",
        libraryPickTip: "Тяните в слот · двойной клик — показать на холсте",
        libraryEmpty: "На сервере пока нет недавних изображений.",
        galleryEmpty: "Результаты этой сессии появятся здесь.",
        stageEmpty: "Опишите изображение и нажмите Run",
        stageEmptyHint: "Здесь появится результат. Картинку можно бросить прямо в эту область, чтобы работать с ней.",
        menuSendTo: (mode) => `Отправить в ${mode}`,
        menuAsReference: "Ляжет в слот референса",
        menuClear: "Очистить канвас",
        menuClearHint: "Только эта вкладка",
        menuSent: (mode) => `Отправлено в ${mode}`,
        upscaleStarting: "Запускаю апскейл",
        droppedAsReference: (slot) => `В референс ${slot}`,
        refsFull: "Слоты референсов заняты — освободите один.",
        outpainting: "Outpainting",
        videoFirst: "Взять первый кадр",
        videoLast: "Взять последний кадр",
        videoHint: "Брошен ролик — студия работает по одному кадру",
        videoTaking: "Достаю кадр…",
        videoFailed: (m) => `Кадр не достать: ${m}`,
        warmupTitle: "Готовимся",
        warmupNote: "Загружается модель. Картинка начнёт появляться, как только пойдёт рисование.",
        backendBroken: "недоступен",
        noBackends: "Нет доступных workflow ни для одной установленной модели.",
        runFailed: (m) => `Ошибка запуска: ${m}`,
        upscaleNeedsImage: "Сначала выберите изображение в галерее сессии, затем Run.",
        requiresMissing: (p) => `Сначала добавьте обязательное изображение (${p}).`,
        pngRestored: (f) => `Настройки восстановлены из изображения (${f}).`,
        pngNotStudio: "В этом изображении нет настроек студии.",
        inPack: "в наборе",
        pngNoBackend: (id) => `Изображение сделано бэкендом '${id}', он здесь недоступен.`,
        modes: { generate: "Генерация", t2i: "Генерация", edit: "Редактирование",
                 img2img: "Image to Image",
                 inpaint: "Inpaint", outpaint: "Outpaint", upscale: "Upscale",
                 batch: "Batch" },
        batch: {
            title: "Batch",
            placeholder: "По одному промпту в строке.",
            dropHint: "Бросьте сюда картинки — промпты вытащим из них.",
            enhance: "Сначала улучшить промпты",
            enhanceTip: "Все промпты проходят через модель разом, до рисования: "
                + "загрузить её один раз — минута, двадцать раз — двадцать.",
            seeds: "Сиды",
            seedNames: {
                fixed: "один на все",
                series: "по порядку",
                random: "случайные",
            },
            baseSeed: "от",
            start: "Запустить", stop: "Остановить",
            empty: "Очередь пуста.",
            count: (n) => `в очереди: ${n}`,
            states: { waiting: "ждёт", running: "в работе", done: "готово",
                      failed: "не вышло", skipped: "пропущено" },
            phasePrompts: (done, total) => `Промпты: ${done} из ${total}`,
            phaseImages: (done, total) => `Картинки: ${done} из ${total}`,
            reading: "Читаю промпты из картинок…",
            readNone: "В этих картинках промптов нет.",
            readSome: (n) => `Добавлено промптов: ${n}.`,
            finished: (done, failed) => failed
                ? `Готово: ${done} в библиотеке, не вышло ${failed}.`
                : `Готово: ${done} в библиотеке.`,
            needModel: "Сначала выберите модель.",
        },
        outNeedsImage: "Сначала принесите картинку — расширять пока нечего.",
        i2iNeedsImage: "Сначала принесите картинку — этот режим перерисовывает её.",
        outFrame: (w, h) => `${w} × ${h}`,
        tabQueue: "Очередь",
        tabQueueTitle: "Задания в очереди и то, что считается сейчас",
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
        loraAddTitle: "Найти LoRA среди установленных и добавить её",
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
            eraser: "Ластик — стирает маску (E, или удерживая Alt)",
            fit: "Вписать в рабочую область (двойной клик; колесо — масштаб)",
            keep: "Сохранить",
            keepTip: "Оставить эту версию в библиотеке. До этого перерисовки — черновики.",
            kept: "Сохранено в библиотеку.",
            keepFailed: (m) => `Не удалось сохранить: ${m}`,
            brushMode: "Кисть",
            brushModeTip: "Рисовать маску (E переключает, Alt стирает пока зажат)",
            eraserMode: "Ластик",
            clear: "Очистить маску",
            undo: "Отменить (Ctrl+Z)", redo: "Вернуть (Ctrl+Y)",
            empty: "Перетащите изображение, выберите результат сессии или тяните из Библиотеки.",
            cleaning: "Очистка…",
            cleaned: (s) => `Очищено за ${s} с`,
            repainted: "Перерисовано — результат на холсте и в галерее.",
            needImage: "Сначала добавьте изображение.",
            needMask: "Сначала нарисуйте маску.",
            needRepaint: "Cleanup срабатывает сам, пока вы красите. Чтобы область "
                + "перерисовала модель, переключитесь на Repaint.",
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
            library: "Готовые промпты для доводки деталей",
            librarySearch: "Поиск промптов…",
            libraryEmpty: "Промпты не найдены.",
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
/* ⚠️ Отступ ВНУТРЕННИЙ (padding), а не внешний. Снаружи он оставлял по краю полосу
   в 12 px, которая рабочей областью не считалась: правый клик там проваливался
   мимо меню, и дроп тоже. Теперь область накрывает контейнер целиком, а воздух
   вокруг кадра даёт padding. */
.ts-istudio__stagefit{position:absolute;inset:0;padding:12px;display:flex;
    align-items:center;justify-content:center;overflow:hidden;touch-action:none;
    box-sizing:border-box}
/* Масштаб и сдвиг живут на этой коробке, а не на самой картинке: так их видно
   в одном месте и так же ведёт себя шторка сравнения. */
.ts-istudio__zoom{display:flex;align-items:center;justify-content:center;
    width:100%;height:100%;transform-origin:0 0;will-change:transform}
/* Две вложенные коробки — вся укладка сцены.

   fitbox — место, отведённое под показ. Обычно это сам кадр; в расширении
   пропорция берётся будущего кадра (--ts-fit-ratio), и тогда картинка внутри
   ужимается ровно настолько, чтобы новые области попали в поле зрения.

   frame — картинка по своей пропорции, вписанная в это место.

   Накладки лежат ВНУТРИ этих коробок (inset:0, доли в процентах), поэтому
   масштаб и сдвиг коробки зума двигают их вместе с кадром — разъехаться
   нечему. ⚠️ Раньше сетка и рамка ставились по getBoundingClientRect(), и на
   любом зуме, кроме единицы, получали двойной масштаб. */
.ts-istudio__fitbox{position:relative;display:flex;align-items:center;
    justify-content:center;flex:0 0 auto}
/* Подложка занимает ВЕСЬ будущий кадр и лежит под исходником: в расширении
   сквозь неё видны только дорисовываемые области. */
.ts-istudio__under{position:absolute;inset:0;width:100%;height:100%;display:none;
    object-fit:fill;border-radius:4px;opacity:0;transition:opacity .45s ease}
.ts-istudio__under.is-active{display:block;opacity:1}
.ts-istudio__frame{position:relative;border-radius:4px;overflow:hidden;flex:0 0 auto;
    z-index:1}
/* contain, а не растяжение: рамку считает JS по настоящему размеру картинки,
   но пока новая картинка грузится, размера ещё нет. Растяжение в этот миг
   показывает сплющенный кадр (на вертикальных это особенно заметно), а
   вписывание — просто поле по краю, которое тут же исчезает. */
.ts-istudio__frame img{display:block;width:100%;height:100%;object-fit:contain}
.ts-istudio__fit{position:absolute;right:10px;top:10px;z-index:6;width:26px;height:26px;
    display:none;align-items:center;justify-content:center;padding:0;cursor:pointer;
    border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);
    background:var(--ts-elevated);color:var(--ts-muted);font-size:13px}
.ts-istudio__fit.is-active{display:flex}
.ts-istudio__fit:hover{color:var(--ts-text)}
/* Заглушка — по центру ВСЕЙ области: она лежит абсолютно, поэтому коробка
   зума её больше не сдвигает. */
.ts-istudio__stageempty{position:absolute;inset:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;gap:8px;text-align:center;padding:24px;
    pointer-events:none}
.ts-istudio__emptytitle{font-size:20px;line-height:1.25;color:var(--ts-text);
    letter-spacing:.01em;max-width:34ch}
.ts-istudio__emptyhint{font-size:var(--ts-fs);color:var(--ts-muted);max-width:44ch}
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
.ts-istudio__stop{width:100%}
.ts-istudio__runhint{text-align:center;color:var(--ts-muted);font-size:var(--ts-fs-xs)}
.ts-istudio__progress{height:3px;border-radius:2px;background:var(--ts-border-soft);overflow:hidden;display:none}
.ts-istudio__progress.is-active{display:block}
.ts-istudio__progress div{height:100%;width:0%;background:var(--ts-accent);transition:width .2s ease}
${ITERATIONS_CSS}${BATCH_CSS}`;
    document.head.appendChild(style);
}

// The studio that is on screen right now, if any. Kept at module level so a
// caller arriving from outside — an asset browser asking to rebuild a session
// — can hand its state to the open studio instead of opening a second one.
let openInstance = null;

/**
 * The studio currently on screen, or null.
 *
 * @returns {?object} Shell handle, extended with `applyStudioState(state)`.
 */
export function openStudioInstance() {
    return openInstance?.isOpen?.() ? openInstance : null;
}

// ⚠️ Открытие идёт СЕКУНДАМИ: сначала /object_info, потом все графы
// бэкендов, потом состояние наборов. Всё это время на экране ничего нет, и
// человек жмёт кнопку снова — а раньше каждый клик честно начинал новую
// сборку. Две студии оказывались друг на друге: закрыл одну, под ней вторая.
// Тесты интерфейса от этого разваливались пачками, потому что запросы к DOM
// попадали в чужую, уже неактуальную копию.
//
// Поэтому «одна студия» — свойство самой функции, а не договорённость между
// вызывающими: их четыре (кнопка ноды, браузер ассетов, меню, восстановление
// сессии), и уследить обязан кто-то один.
let pendingOpen = null;

/**
 * Open the studio for one node instance. Returns the overlay handle.
 *
 * Повторный вызов, пока студия открыта или ещё открывается, НЕ создаёт вторую:
 * возвращается та же самая.
 */
export async function openStudio(node, persist) {
    const already = openStudioInstance();
    if (already) {
        already.parkFocus?.();
        return already;
    }
    if (pendingOpen) return pendingOpen;
    pendingOpen = buildStudio(node, persist).finally(() => { pendingOpen = null; });
    return pendingOpen;
}

async function buildStudio(node, persist) {
    ensureAppStyles();
    ensureControlStyles();
    const t = pickLocaleStrings(STRINGS);
    // Backend manifests carry their own {en, ru} labels, and the controls pick
    // one by this code. Ask the theme what language the UI is in — comparing
    // the merged strings object against STRINGS.ru never matched (the helper
    // returns a fresh object), so every manifest label read as English no
    // matter what ComfyUI was set to.
    const locale = getUiLanguage() === "ru" ? "ru" : "en";

    const objectInfo = await (await api.fetchApi("/object_info")).json();
    // Backend workflow files are WEB_DIRECTORY statics: /extensions/* lives
    // OUTSIDE the /api prefix that api.fetchApi prepends, so plain fetch.
    const readBackends = () =>
        loadBackends((url, options) => fetch(url, options), objectInfo,
                     (url, options) => api.fetchApi(url, options));
    // Not const: installing a pack adds backend files, and the studio rereads
    // them in place rather than asking to be reopened.
    let backends = await readBackends();
    // Что человек выключил и каким уровнем смотрит. Читается ДО первой сборки
    // списка моделей и отдельным роутом от каталога: каталог может ждать
    // недоступный хост, а то, что стоит на этой машине, ждать не должно.
    let packState = await readPackState();
    let hiddenFamilies = [];
    let families = takeFamilies(backends);
    // Была ли у узла своя сессия. Пустая означает либо новый узел, либо
    // обнулённые ComfyUI свойства — во втором случае рабочее место лежит под
    // прежним идентификатором, и строгий поиск его не найдёт.
    const hadSession = Boolean(persist.sessionId);
    const sessionId = persist.sessionId || newSessionId();
    persist.setSessionId(sessionId);

    const gate = createGate({ api, onChange: () => showcase?.refresh() });
    await gate.refresh();
    const optionalIndex = buildOptionalIndex(objectInfo);
    const runner = createRunner(api);
    const values = {};      // param -> value for the active backend
    let activeBackend = null;
    let showcase = null;            // built once the shell exists; see below
    // Models the catalogue offers that this machine does not have. They are
    // shown in the picker greyed out — a studio with nothing installed would
    // otherwise give no hint that Krea 2 or Ideogram exist at all.
    let offers = [];
    let catalogData = null;         // last read of the packs catalogue
    let queueCount = 0;
    // Идентификаторы своих прогонов: по ним работает остановка.
    const liveRuns = new Set();

    /**
     * Разделы, которым есть чем работать.
     *
     * Пустых вкладок в студии не бывает: модели приезжают паками, и раздел,
     * для которого не установлено ни одного семейства, — это обещание,
     * которое нечем выполнить. Считается заново после каждой перечитки
     * бэкендов, потому что пак можно включить и выключить не выходя из студии.
     */
    function availableModes(list) {
        const present = new Set([...list.values()].flatMap((f) => [...f.modes.keys()]));
        return UI_MODES.filter((m) => m.backendModes.some((b) => present.has(b)));
    }

    let uiModes = availableModes(families);
    let modeIds = uiModes.map((m) => m.id);
    /** Пересобрать пункты меню под текущий список разделов (см. ниже). */
    let registerStageCommands = () => {};
    const backendModesOf = (uiMode) =>
        UI_MODES.find((m) => m.id === uiMode)?.backendModes || [uiMode];
    const railModes = (list) => list.map((m) => ({
        id: m.id,
        title: t.modes[m.id] || m.id,
        icon: ICONS[m.id] || ICONS.generate,
    }));

    /**
     * Что на этой машине выключено и каким уровнем её показывать.
     *
     * Молчаливый откат к «всё видно» — не лень: у пользователя этого файла
     * нет вовсе, и упавший запрос не должен прятать половину студии.
     */
    async function readPackState() {
        try {
            const response = await api.fetchApi("/ts_studio/packs/state");
            if (response.ok) return await response.json();
        } catch (err) {
            console.warn("[TS Studio] pack state unavailable", err);
        }
        return { disabled: [], viewTier: null, packs: [] };
    }

    /**
     * Сколько клеток нарежет этот граф — и какой формы.
     *
     * Своя нода-резчик про свою сетку наружу не сообщает ничего, а показать
     * её надо ДО того, как она отработает: между нажатием и первым событием
     * проходят десятки секунд загрузки моделей. Поэтому геометрия считается
     * здесь, по параметрам самого графа и размеру исходника.
     *
     * @param {object} graph готовый к отправке граф
     * @param {object} runValues значения контролов этого прогона
     * @returns {{cols:number,rows:number}|null} null, если резчика в графе нет
     */
    function plannedTileGrid(graph, run) {
        const nodes = Object.values(graph || {});
        const splitter = nodes.find((node) => node?.class_type === "TS_ImageTileSplitter");
        if (!splitter) return null;
        const { width, height } = stage.naturalSize();
        if (!(width > 0 && height > 0)) return null;
        // Увеличение стоит перед резкой, поэтому сетка считается по конечному
        // размеру, а не по исходному.
        const scale = Number(run?.scale ?? run?.upscale ?? 1) || 1;
        return splitterGrid({
            width: Math.round(width * scale),
            height: Math.round(height * scale),
            tileWidth: Number(splitter.inputs?.tile_width) || 0,
            tileHeight: Number(splitter.inputs?.tile_height) || 0,
            overlap: Number(splitter.inputs?.overlap) || 0,
        });
    }

    /**
     * Показать, куда дорисуется кадр.
     *
     * Зовётся при каждом поводе, от которого рамка могла измениться: смена
     * пропорции или размера, новая картинка, переход в раздел. «21:9» само по
     * себе ничего не говорит — человек должен видеть, сколько допишется слева
     * и справа.
     */
    /** Геометрия сетки апскейла: посчитана при отправке, показана по первому
     *  шагу сэмплера. */
    let plannedGrid = null;
    /**
     * Ждёт ли рамка расширения своего прогона.
     *
     * ⚠️ Рамка обещает, КУДА вырастет картинка. Как только результат пришёл,
     * обещание выполнено: на экране уже расширенный кадр, и полосы поверх него
     * — ложь (замечено владельцем: заштрихованные области оставались поверх
     * готового результата). Взводится, когда на сцену кладут исходник, и
     * снимается по концу прогона.
     */
    let outframeArmed = true;

    // ⚠️ ПОКАЗ ОЖИДАНИЯ ПРИНАДЛЕЖИТ ПРОГОНУ, А НЕ ВКЛАДКЕ.
    //
    // Жалоба владельца: запустил перерисовку, ушёл на другую вкладку, вернулся
    // — и пусто, будто ничего не считается. Так и было: уход из раздела гасил
    // ожидание (`restore` зовёт `warmup.hide()`), а вернуть его было нечему —
    // следующее событие прогресса приходило в закрытое кольцо и молчало.
    //
    // Здесь лежит всё, что нужно, чтобы показать ожидание заново: для какого
    // раздела оно, с какими доводами открыто, на каком этапе и какая доля
    // пройдена. Обновляется там же, где рисуется ход, и стирается вместе с
    // концом прогона — в один заход для всех исходов.
    let liveWait = null;

    /** Показать ожидание заново — если для этого раздела оно ещё живо. */
    function restoreWaiting(modeId) {
        if (!liveWait || liveWait.mode !== modeId) return;
        stage.warmup.show(liveWait.args);
        if (liveWait.stage) stage.warmup.setStage(liveWait.stage);
        stage.warmup.setProgress(liveWait.fraction);
        if (deckWidgets) {
            deckWidgets.progress.classList.add("is-active");
            deckWidgets.progressFill.style.width =
                `${Math.max(2, Math.min(100, Math.round(liveWait.fraction * 100)))}%`;
        }
        // Сетка тайлов — часть того же показа: без неё апскейл возвращается
        // с кольцом, но без разметки, по которой видно, где сейчас считают.
        if (modeId === "upscale" && liveWait.grid) tiles.prepare(liveWait.grid);
    }

    function paintOutFrame() {
        const size = stage.naturalSize();
        if (!outframeArmed || activeModeId !== "outpaint"
            || !stage.hasImage() || !(size.height > 0)) {
            stage.reserveRatio(0);
            stage.outframe.hide();
            return;
        }
        const aspect = String(values.frame || "16:9");
        // Сначала место, потом рамка: сцена отводит под показ будущий кадр, и
        // картинка внутри ужимается сама. Иначе новые области оказывались за
        // краем области — их «выносило вбок».
        stage.reserveRatio(parseFrameAspect(aspect));
        stage.outframe.show({
            imageRatio: size.width / size.height,
            aspect,
            megapixels: Number(values.megapixels || 1.5),
        });
    }

    /** Семейства, которые студия показывает; скрытые уезжают в серый список. */
    function takeFamilies(list) {
        const split = applyPackState(groupByFamily(list), packState);
        hiddenFamilies = split.hidden;
        return split.families;
    }

    /**
     * Rebuild the offer list from the packs catalogue.
     *
     * A family counts as offered only while it is absent here: once a pack is
     * installed its models are real entries, and the ghost must not linger
     * beside them.
     */
    function setOffers(data) {
        // Kept so the list can be recomputed after backends reload: removing a
        // pack tells us about it while its families are still loaded, and an
        // offer worked out at that moment would look like it is already here.
        if (data) catalogData = data;
        offers = buildOffers({
            families, hidden: hiddenFamilies, catalog: catalogData, locale,
        });
    }

    /** Offered families that would serve this UI mode. */
    function offersForMode(uiMode) {
        return offersForModes(offers, backendModesOf(uiMode));
    }

    /** Families offering any backend of this UI mode, with their roles. */
    function familiesForMode(uiMode) {
        return rolesForModes(families, backendModesOf(uiMode));
    }

    const shell = createShell({
        label: t.appLabel,
        closeTitle: t.close,
        collapseTitle: t.collapse,
        modes: railModes(uiModes),
        onMode: (id) => selectMode(id),
        onClose: () => {
            // Last read before everything is torn down, then written now
            // rather than on the timer — the page may be leaving.
            captureValues();
            memory.flush();
            rememberWorkspaceNow();
            openInstance = null;
            stageDropTeardown?.();
            // ⚠️ Сцена тоже разбирается. Её `teardown` существовал, но здесь
            // его не звали: слушатели документа и окна (зум, панорама, шторка)
            // оставались жить, а вместе с ними в памяти держалась вся студия
            // целиком — со всеми кадрами истории.
            stage.teardown?.();
            stageMenu.teardown();
            gallery.teardown?.();
            queuePanel.teardown();
            helpPanel.teardown?.();
            settingsPanel.teardown?.();
            showcase.teardown?.();
            gate.teardown?.();
            inpaintMode?.teardown();
            for (const instance of controlInstances) instance.teardown?.();
            promptTools?.teardown();
            runner.destroy();
        },
        onKey: (event, { typing = false } = {}) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                event.preventDefault();
                run();
                return;
            }
            // Всё остальное — команды холста, и в поле ввода им делать нечего:
            // Ctrl+Z в промпте отменяет ТЕКСТ, а не мазок маски.
            if (typing) return;
            if (activeModeId === "inpaint" && inpaintMode
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
            } else if (activeModeId === "inpaint" && inpaintMode
                       && (event.key === "e" || event.key === "E")
                       && !event.ctrlKey && !event.metaKey && !event.altKey) {
                event.preventDefault();
                inpaintMode.toggleEraser();
            } else if (event.key === "F1") {
                event.preventDefault();
                helpPanel.toggle();
            } else if ((event.key === "b" || event.key === "B" || event.key === "и"
                        || event.key === "И")
                       && (event.ctrlKey || event.metaKey)) {
                // ⚠️ Раньше панель сворачивал ГОЛЫЙ Tab — и обойти студию с
                // клавиатуры было нельзя вовсе: обработчик гасил каждое
                // нажатие, фокус не двигался ни на одну кнопку. Tab вернулся
                // обходу, сворачивание переехало на Ctrl/Cmd+B — тот же
                // аккорд, что у боковых панелей редакторов. Кнопка в рельсе
                // никуда не делась.
                event.preventDefault();
                shell.setSideCollapsed(!shell.isSideCollapsed());
            }
        },
    });

    // ── сцена ───────────────────────────────────────────────────────────── //
    // Всё, что показывает картинку, живёт в одном модуле: кадр, зум, шторка,
    // сетка тайлов, подпись и память сцены по режимам (`_stage.js`).
    const stage = createStage({
        host: shell.stage,
        strings: {
            empty: t.stageEmpty,
            emptyHint: t.stageEmptyHint,
            // Этапы — те же слова, что в панели: человек читает одно и то же
            // в двух местах, а не два разных словаря.
            stages: t.stages,
            warmupTitle: t.warmupTitle,
            warmupNote: t.warmupNote,
            fitView: t.fitView,
            recreate: t.recreate,
            recreateTip: t.recreateTip,
            compare: { before: t.cmp.before, after: t.cmp.after },
        },
        onRecreate: (state) => {
            applyStudioState(state).catch((err) => setStatus(String(err?.message || err)));
        },
        onChange: () => rememberWorkspace(),
        // ⚠️ Рамку расширения считать можно только по ЗАГРУЖЕННОЙ картинке.
        // Раньше её рисовали сразу после приёма файла: у нового кадра ещё не
        // было натурального размера, и полосы считались по ПРОШЛОЙ пропорции —
        // между картинкой и новой областью появлялась широкая серая щель
        // (замечено владельцем при повторном закидывании).
        onLayout: () => paintOutFrame(),
    });
    const tiles = stage.tiles;

    let selectedResult = null;

    function showResult(result) {
        selectedResult = result;
        const params = result.params || {};
        const bits = [];
        if (params.width && params.height) bits.push(`${params.width} × ${params.height}`);
        if (params.seed !== undefined) bits.push(`seed ${params.seed}`);
        // ⚠️ Тип — у самого файла. Черновики итераций лежат во временной
        // папке, и по `type=output` они попросту не открываются.
        stage.show(resultViewUrl(result.image),
            { caption: bits.join(" · "), state: result.state || null });
    }

    function showLibraryAsset(asset) {
        outframeArmed = true;          // взяли картинку из библиотеки — как дроп
        // Картинка из библиотеки — тоже то, что на экране: если её взяли под
        // апскейл, она обязана вернуться вместе с рабочим местом.
        stage.show(asset.url, { caption: asset.name || "", keepSource: true });
        stage.setSource(asset.annotated || "");
        requestAnimationFrame(() => paintOutFrame());
    }

    function showPreviewBlob(blob) {
        stage.showBlob(blob);
    }

    const helpPanel = createHelpPanel({
        stage: shell.stage, t, locale,
        studioRoot: shell.root,
        pagesBase: "/extensions/comfyui-timesaver/image/studio/help",
    });
    // Preferences that belong to the person: applied as the studio opens, and
    // again the moment one is changed.
    const settingsPanel = createSettingsPanel({
        host: shell.stage,
        onChange: (key, value) => {
            if (key === "browserSide") shell.setSidePlacement(value);
        },
        // Resetting forgets, then rebuilds the deck so the person sees the
        // defaults immediately rather than after the next switch.
        memory: {
            stats: () => memory.stats(),
            forget: () => {
                suppressCapture = true;
                try {
                    memory.forgetAll();
                    sessionRefs.clear();
                    if (activeBackend) buildDeck(activeBackend);
                } finally {
                    suppressCapture = false;
                }
            },
        },
        pass: {
            state: () => gate.state(),
            clear: () => gate.forget().then((state) => { showcase?.refresh(); return state; }),
            prompt: () => gate.prompt(),
        },
        // Testing mode. The panel draws nothing unless the server reports the
        // marker file, so a user never sees this row (nodes/_studio_dev.py).
        dev: {
            state: async () => {
                const response = await api.fetchApi("/ts_studio/dev");
                return response.ok ? response.json() : {};
            },
            set: async (patch) => {
                const response = await api.fetchApi("/ts_studio/dev", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(patch),
                });
                return response.ok ? response.json() : {};
            },
            // Both switches change what the studio may fetch, so the pass and
            // the catalogue are reread and the deck redrawn from them.
            onChange: async () => {
                await gate.refresh();
                await showcase.refresh();
                await reloadBackends();
            },
        },
    });
    shell.setSidePlacement(readSetting("browserSide"));

    const settingsButton = document.createElement("button");
    settingsButton.type = "button";
    settingsButton.className = "ts-studio__railbtn";
    settingsButton.title = settingsStrings().open;
    settingsButton.setAttribute("aria-label", settingsStrings().open);
    settingsButton.innerHTML = ICONS.settings;
    settingsButton.addEventListener("click", () => {
        helpPanel.close?.();
        showcase?.close();
        settingsPanel.toggle();
    });
    shell.rail.appendChild(settingsButton);

    // The showcase is where a person sees what a subscription buys, and where
    // an installed pack turns into models in the picker.
    showcase = createShowcase({
        host: shell.stage,
        api,
        onCatalog: (data) => setOffers(data),
        onInstalled: () => reloadBackends(),
        onWantAccess: () => gate.prompt(),
        readiness: (pack) => packReadiness(pack, backends),
    });
    // Read the catalogue once at startup rather than on first open: the deck
    // needs the offers to draw them, and a studio that never opens this screen
    // should still show what it is missing. Failure is silent — the studio
    // works offline, it just cannot offer anything.
    showcase.refresh().then(() => {
        if (offers.length && activeModeId) selectMode(activeModeId);
    }).catch(() => {});

    const packsButton = document.createElement("button");
    packsButton.type = "button";
    packsButton.className = "ts-studio__railbtn";
    packsButton.title = showcase.strings.open;
    packsButton.setAttribute("aria-label", showcase.strings.open);
    packsButton.innerHTML = ICONS.packs;
    packsButton.addEventListener("click", () => {
        helpPanel.close?.();
        settingsPanel.close();
        showcase.toggle();
    });
    shell.rail.appendChild(packsButton);

    const helpButton = document.createElement("button");
    helpButton.type = "button";
    helpButton.className = "ts-studio__railbtn";
    helpButton.title = t.help.open;
    helpButton.setAttribute("aria-label", t.help.open);
    helpButton.textContent = "?";
    helpButton.addEventListener("click", () => {
        settingsPanel.close();
        showcase.close();
        helpPanel.toggle();
    });
    shell.rail.appendChild(helpButton);

    // ── asset panel: session results, library, queue ────────────────────── //
    const queuePanel = createQueuePanel({ api, t });
    const gallery = createGallery({
        t,
        extraTabs: [{
            id: "queue",
            label: t.tabQueue,
            title: t.tabQueueTitle,
            element: queuePanel.element,
            onVisible: (visible) => queuePanel.setVisible(visible),
        }],
        onSelect: (result) => {
            showResult(result);
            persist.setResultPath(resultRelPath(result.image));
        },
        // Какая вкладка панели открыта — такая же настройка рабочего места,
        // как ширина колонок: выбрал браузер, вернулся — он и открыт.
        onTab: (which) => shell.rememberPanelTab?.(which),
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
    // Установленный внешний браузер (Artius) — это то, ради чего панель и
    // открывают: когда он есть, вкладка браузера и становится начальной.
    // Явный выбор человека сильнее: если вкладку уже переключали, берётся она.
    pickAssetProvider().then((provider) => {
        const remembered = shell.panelTab?.();
        if (remembered) {
            gallery.showTab(remembered);
            return;
        }
        if (provider && provider.id !== "fallback") gallery.showTab("library");
    }).catch(() => { /* панель откроется на сессии */ });
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

    // Everything the person set is remembered — see _studio/_memory.js for
    // which values follow the work and which belong to one graph. References
    // are the exception: they point at uploaded files, so they are carried
    // within a sitting but not written to disk, where a stale name would come
    // back as a broken thumbnail.
    const sessionRefs = new Map();  // param -> value, this sitting only

    // Which graph the controls on screen belong to. Not `activeBackend`: that
    // already points at the NEW graph by the time a rebuild starts, and the
    // values still on screen belong to the old one — capturing under the wrong
    // key silently moved Inpaint's numbers onto Upscale.
    let deckGraphId = "";
    // Режим, в котором собрана нынешняя дека. Снимок её значений помечается
    // именно им: сборка новой деки происходит уже после смены режима.
    let deckScope = "";
    // Set while the deck is rebuilt on purpose to forget: without it the
    // capture that precedes every rebuild would write the values still on
    // screen straight back into the store we just cleared.
    let suppressCapture = false;

    function graphKey(backend) {
        return backend?.manifest?.id || "unknown";
    }

    /** Read every live control and remember it. Called before a rebuild. */
    function captureValues() {
        if (suppressCapture || !deckGraphId) return;
        const key = deckGraphId;
        for (const [param, instance] of controlsByParam) {
            try {
                const value = instance.get();
                if (instance.kind === "refs") sessionRefs.set(param, value);
                else memory.remember(key, param, instance.kind, value, deckScope);
            } catch (err) {
                console.warn(`[TS Studio] could not keep '${param}'`, err);
            }
        }
        if (promptTools) {
            memory.remember(key, "__styles", "styles",
                promptTools.getSelectedStyles(), deckScope);
        }
    }

    function buildDeck(backend) {
        captureValues();
        deckGraphId = graphKey(backend);
        deckScope = activeModeId || backend.manifest.mode;
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
        select.title = t.modelTitle || "";
        for (const role of roles.values()) {
            const option = document.createElement("option");
            option.value = role.family.family;
            const usable = role.primary.available || role.edit?.available;
            // A backend that is here is a backend that runs: what a pack
            // brought stays usable whether or not a pass is current. The pass
            // buys the delivery, not the running (nodes/_pass.py).
            option.textContent = usable ? role.label : `${role.label} — ${t.backendBroken}`;
            option.disabled = !usable;
            option.selected = role.family.family === backend.manifest.family;
            select.appendChild(option);
        }
        // Models the catalogue offers but this machine does not have. Showing
        // them greyed out is how anyone learns a subscription exists at all.
        const ghosts = offersForMode(activeModeId || backend.manifest.mode);
        for (const ghost of ghosts) {
            const option = document.createElement("option");
            option.value = `${GHOST_PREFIX}${ghost.family}`;
            option.textContent = `🔒 ${ghost.label} — ${t.inPack}`;
            select.appendChild(option);
        }
        select.addEventListener("change", () => {
            if (select.value.startsWith(GHOST_PREFIX)) {
                // Reaching for something not installed is how most people meet
                // the packs screen: show it there, and put the picker back.
                select.value = backend.manifest.family;
                settingsPanel.close();
                helpPanel.close?.();
                showcase.open();
                return;
            }
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
            // Read what was remembered BEFORE the control exists: a control
            // announces its default the moment it is built, and that write
            // would land on top of the value we are about to restore.
            const kept = control.kind === "refs"
                ? sessionRefs.get(control.param)
                : memory.recall(graphKey(backend), control.param, control.kind);
            const instance = renderer(control, {
                t, locale, loraOptions,
                uploadImage: (blob, name) => uploadImage(api, blob, name),
                // A designer editor opens seeded with what the deck shows.
                getPrompt: () => controlsByParam.get("prompt")?.get() || "",
                getSize: () => controlsByParam.get("size")?.get(),
                onChange: (param, value) => {
                    values[param] = value;
                    if (param === "frame" || param === "megapixels") paintOutFrame();
                    // Written as it changes, not only when the deck is rebuilt:
                    // a closed tab or a reload must not cost the last move.
                    if (control.kind === "refs") sessionRefs.set(param, value);
                    else memory.remember(graphKey(backend), param, control.kind, value);
                    rememberWorkspace();
                    // A filled reference sends the run to the edit graph,
                    // which takes its frame from that image.
                    if (param === "__refs") {
                        const used = Object.values(value || {}).some(Boolean);
                        controlsByParam.get("size")?.setDisabled?.(used);
                        // In Inpaint the reference is only read during a full
                        // redraw — a partial one has nothing to fill from. So
                        // dropping an object in turns Replace on rather than
                        // leaving the picture silently unchanged.
                        const replace = controlsByParam.get("replace");
                        if (used && replace && replace.get() === false) replace.set(true);
                    }
                },
            });
            instance.kind = control.kind;
            controlInstances.push(instance);
            if (control.param) controlsByParam.set(control.param, instance);
            if (kept !== undefined) {
                // What the person set outlives the file's default.
                try { instance.set(kept); }
                catch (err) { console.warn(`[TS Studio] could not restore '${control.param}'`, err); }
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
                        // Там, где работа идёт НАД картинкой, прикреплять
                        // отдельную незачем: улучшение промпта берёт ту, что на
                        // холсте. Кнопка и превью только сбивали бы с толку.
                        attach: !SOURCE_MODES.has(activeModeId)
                            && activeModeId !== "inpaint",
                        currentImage: () => currentImageUrl(),
                        initialStyles: memory.recall(graphKey(backend), "__styles", "styles") || [],
                        // Готовые промпты зависят от режима: доводка деталей
                        // осмысленна в инпэйнте и бессмысленна при генерации с
                        // нуля, где описывают ещё не существующий кадр.
                        presets: promptPresetsFor(backend.manifest?.mode),
                        // Улучшалку промпта называет сам бэкенд: у перерисовки
                        // свой диалект (инструкция правки или описание
                        // результата — зависит от того, что понимает модель),
                        // у Ideogram — свой пресет под его подписи. Поменять
                        // можно одной строкой манифеста.
                        enhancePreset: backend.manifest?.enhance_preset
                            || backend.manifest?.designer?.preset,
                    });
                }
            }
        }
        if (advanced.length) {
            const toggle = document.createElement("button");
            toggle.type = "button";
            toggle.className = "ts-studio__advanced";
            toggle.title = t.advancedTitle || "";
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
        runButton.title = t.runTitle || "";
        runButton.addEventListener("click", () => run());
        // Остановка стоит там же, где запуск, и появляется только когда есть
        // что останавливать. Снимает ТОЛЬКО свои прогоны: рендер, запущенный с
        // холста ComfyUI, не наш и трогать его нельзя.
        const stopButton = document.createElement("button");
        stopButton.type = "button";
        stopButton.className = "ts-ui-btn ts-istudio__stop";
        stopButton.textContent = t.stop;
        stopButton.title = t.stopTip;
        stopButton.style.display = "none";
        stopButton.addEventListener("click", () => stopRuns(false));

        const stopAllButton = document.createElement("button");
        stopAllButton.type = "button";
        stopAllButton.className = "ts-ui-btn ts-istudio__stop";
        stopAllButton.textContent = t.stopAll;
        stopAllButton.title = t.stopAllTip;
        stopAllButton.style.display = "none";
        stopAllButton.addEventListener("click", () => stopRuns(true));

        const hint = document.createElement("div");
        hint.className = "ts-istudio__runhint";
        hint.textContent = t.runHint;
        runWrap.append(runButton, stopButton, stopAllButton, progress, hint);
        foot.appendChild(runWrap);
        shell.deck.appendChild(foot);
        deckWidgets = { runButton, stopButton, stopAllButton, progress, progressFill, hint };
    }

    let deckWidgets = null;

    function updateHint() {
        if (!deckWidgets) return;
        deckWidgets.hint.textContent = queueCount > 0
            ? `${t.runHint} · ${t.queued(queueCount)}` : t.runHint;
        // Останавливать нечего — кнопок нет. Снять всю очередь предлагаем
        // только когда в ней действительно больше одного прогона.
        deckWidgets.stopButton.style.display = liveRuns.size > 0 ? "" : "none";
        deckWidgets.stopAllButton.style.display = liveRuns.size > 1 ? "" : "none";
    }

    /**
     * Остановить свои прогоны. `all` — снять всю очередь студии, иначе только
     * тот, что считается прямо сейчас. Чужое из очереди ComfyUI не трогаем:
     * рантаймер шлёт /interrupt лишь когда исполняется именно наш прогон.
     */
    async function stopRuns(all) {
        const ids = [...liveRuns];
        if (!ids.length) return;
        tiles.hide();   // работа прекращена — сетке над кадром делать нечего
        stage.warmup.hide();
        stage.outframe.setWorking(false);
        setStatus(all ? t.stoppingAll(ids.length) : t.stopping);
        const targets = all ? ids : ids.slice(0, 1);
        for (const id of targets) {
            try {
                await runner.cancel(id);
            } catch (err) {
                console.warn("[TS Studio] stop failed", err);
            }
        }
    }

    function selectBackend(backend) {
        activeBackend = backend;
        buildDeck(backend);
        rememberWorkspace();
    }

    // ── рабочее место: что было на экране, то и будет ───────────────────── //
    //
    // Значения контролов помнит `_memory.js`. Здесь запоминается остальное —
    // вкладка, модель и исходники, — потому что без них возвращение в студию
    // означает собирать рабочее место заново. Снимок того же формата, что
    // уезжает в PNG, и применяется тем же `applyStudioState`.
    let workspaceTimer = null;
    let restoringWorkspace = false;

    function currentStudioState() {
        if (!activeBackend) return null;
        // Исходник берётся у того режима, который сейчас на экране.
        //
        // Раньше спрашивали только инпэйнт, и апскейл после закрытия студии
        // возвращался с пустой сценой: его картинка живёт не в холсте маски, а
        // на самой сцене, и в снимок не попадала.
        let modeSources = {};
        if (activeModeId === "inpaint" && inpaintMode) {
            modeSources = inpaintMode.currentSources?.() || {};
        } else if (stage.source()) {
            modeSources = { source_image: stage.source() };
        } else if (selectedResult?.image) {
            // После прогона на сцене лежит результат, а не то, что положили
            // руками: сохраняем именно его — это и есть «что было на экране».
            modeSources = { source_image: resultAnnotated(selectedResult.image) };
        }
        const inpaintSources = modeSources;
        return buildStudioState({
            backendId: activeBackend.manifest.id,
            family: activeBackend.manifest.family,
            familyLabel: activeBackend.manifest.family_label,
            mode: activeBackend.manifest.mode,
            uiMode: activeModeId,
            values: { ...values },
            loras: Array.isArray(values.loras) ? values.loras : [],
            styles: promptTools?.getStyleNames() || [],
            size: controlsByParam.get("size")?.get(),
            sessionId,
            sources: { ...inpaintSources, ...(values.__refs || {}) },
        });
    }

    /** Отложенная запись: правка ползунка не должна писать в хранилище. */
    function rememberWorkspace() {
        if (restoringWorkspace) return;
        clearTimeout(workspaceTimer);
        workspaceTimer = setTimeout(rememberWorkspaceNow, 500);
    }

    /**
     * Записать рабочее место целиком, включая нарисованную маску.
     *
     * Маска снимается и здесь, а не только на закрытии: закрытие бывает
     * ненастоящим — перезагрузка страницы, пересборка графа, — и тогда
     * последней записью оказывается именно отложенная. Дорого это не выходит:
     * запись случается на правку деки и на смену картинки, а не на каждый мазок.
     */
    function rememberWorkspaceNow() {
        if (restoringWorkspace) return;
        clearTimeout(workspaceTimer);
        const state = currentStudioState();
        if (!state) return;
        const mask = activeModeId === "inpaint" && inpaintMode?.hasMask?.()
            ? inpaintMode.maskDataUrl?.() || "" : "";
        saveWorkspace(sessionId, state, mask);
    }

    /**
     * Reread the backend files after a pack is installed or removed.
     *
     * Rebuilding the deck for the current mode is enough for new models,
     * which is what packs carry. A pack introducing a whole new mode would
     * also need the rail rebuilt — it says so rather than half-showing it.
     */
    async function reloadBackends() {
        backends = await readBackends();
        // Состояние паков перечитывается вместе с графами: выключатель в
        // менеджере обязан действовать сразу, без переоткрытия студии.
        packState = await readPackState();
        families = takeFamilies(backends);
        setOffers(null);            // against the families that exist now
        // Набор разделов зависит от установленных моделей: пак принёс новый —
        // вкладка появляется, выключили последний поддерживающий — исчезает.
        uiModes = availableModes(families);
        modeIds = uiModes.map((m) => m.id);
        // Разделов стало больше или меньше — меню обязано это знать.
        registerStageCommands();
        shell.setModes(railModes(uiModes));
        activeBackend = null;               // the old object is from the old read
        // Раздел, из-под которого ушла последняя модель, оставлять нельзя:
        // человек оказался бы в пустой вкладке без единого способа выйти.
        const next = modeIds.includes(activeModeId) ? activeModeId : modeIds[0];
        if (next) selectMode(next);
    }

    let inpaintMode = null;
    let activeModeId = null;

    /**
     * Перенести черновик в библиотеку и показать его в галерее.
     *
     * Один путь на все разделы: инпэйнт, апскейл и перерисовка сохраняют
     * одинаково, и расходиться им незачем.
     *
     * @param {{filename: string, subfolder?: string, type?: string}} draft
     * @returns {Promise<object>} то, что ответил сервер
     */
    async function keepDraftInLibrary(draft) {
        const response = await api.fetchApi("/ts_studio/keep", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                ...draft,
                family: activeBackend?.manifest?.family || "studio",
            }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const image = await response.json();
        gallery.add({ image, params: {}, state: null });
        return image;
    }

    // ── итерации апскейла и перерисовки ─────────────────────────────────── //
    //
    // У каждого такого раздела своя линейка версий: исходник апскейла не имеет
    // отношения к перерисовке, и переключение вкладок ничего не теряет — ровно
    // как со сценой. Панель тоже своя, все висят на сцене и показывается та,
    // чей раздел открыт.
    /** @type {Map<string, {history: object, panel: object}>} */
    const iterations = new Map();

    function iterationsFor(modeId) {
        if (!ITERATIVE_MODES.has(modeId)) return null;
        let entry = iterations.get(modeId);
        if (entry) return entry;
        const history = createHistory();
        const panel = createIterations({
            history,
            strings: t.iter,
            /** Откат и возврат: версия становится и картинкой, и исходником. */
            onShow: (version) => {
                if (!version?.url) return;
                stage.hideCompare?.();
                stage.show(version.url, { caption: version.meta?.caption || "",
                                          keepSource: true, keepView: true });
                stage.setSource(version.annotated || "");
            },
            onKeep: async (draft) => {
                try {
                    const image = await keepDraftInLibrary(draft);
                    setStatus(t.iter.keptMsg);
                    return image;
                } catch (err) {
                    setStatus(t.iter.keepFailed(String(err?.message || err)));
                    return null;
                }
            },
        });
        panel.setActive(activeModeId === modeId);
        stage.element.appendChild(panel.element);
        entry = { history, panel };
        iterations.set(modeId, entry);
        return entry;
    }

    /** Панель показывает только тот раздел, который открыт. */
    function syncIterationPanels() {
        for (const [modeId, entry] of iterations) {
            entry.panel.setActive(modeId === activeModeId);
        }
    }

    /**
     * Отметить, с чего начинается новая линейка версий.
     *
     * Зовётся перед прогоном и сверяется с тем, что на экране: так любая
     * смена исходника — дроп, выбор из библиотеки, возврат вкладки — начинает
     * историю заново, и не нужно ловить каждую из них по отдельности.
     */
    function anchorIterations(modeId) {
        const entry = iterationsFor(modeId);
        if (!entry) return null;
        const shown = stage.url?.() || "";
        if (!shown) return entry;
        const current = entry.history.current();
        if (!current || current.url !== shown) {
            entry.history.reset({ url: shown, annotated: stage.source?.() || "" });
        }
        return entry;
    }

    function ensureInpaintMounted() {
        if (!inpaintMode) {
            inpaintMode = createInpaintMode({
                api, t, sessionId,
                onEngineChange: () => {},
                /** Перенести черновик в библиотеку — общей дорогой. */
                keepDraft: keepDraftInLibrary,
                // Смена картинки на холсте — не изменение контрола, и раньше
                // она в снимок рабочего места не попадала: он писался только на
                // правку деки и на закрытие. Если студию закрывали не кнопкой
                // (перезагрузка страницы, пересборка графа), исходник терялся —
                // вкладка возвращалась, а холст был пуст.
                onSourceChange: () => rememberWorkspace(),
            });
            shell.stage.appendChild(inpaintMode.element);
            const selectedUrl = stage.url();
            if (selectedUrl) {
                inpaintMode.setImageFromUrl(selectedUrl)
                    .catch((err) => setStatus(t.runFailed(String(err?.message || err))));
            }
        }
        inpaintMode.element.style.display = "";
        // Инпэйнт рисует на своём холсте — сцена вместе с подписью уходит.
        stage.setVisible(false);
    }

    function leaveInpaint() {
        if (inpaintMode) inpaintMode.element.style.display = "none";
        stage.setVisible(true);
    }

    // ── пачка ───────────────────────────────────────────────────────────── //
    //
    // Порядок фаз — главное, ради чего это не «прогнать N раз»: сначала одна
    // модель проходит по всем промптам, потом другая рисует все кадры. Смена
    // модели стоит десятки секунд и всю видеопамять, и чередовать их — значит
    // потратить час на переезды.
    let batchPanel = null;
    let batchQueue = null;

    /** Промпт через улучшайку — тем же роутом, что и кнопка «AI» в панели. */
    async function enhanceOnePrompt(text, index) {
        const response = await api.fetchApi("/ts_super_prompt/enhance", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, seed: index + 1 }),
        });
        const payload = await response.json();
        if (!response.ok || payload.error) {
            throw new Error(payload.error || `HTTP ${response.status}`);
        }
        return String(payload.text || "").trim() || text;
    }

    /**
     * Отложить готовые промпты во временный файл.
     *
     * Час работы модели не должен зависеть от того, переживёт ли вкладка
     * следующие двадцать минут. Не записалось — не беда: кадры важнее, и
     * пачка идёт дальше.
     */
    async function stashBatchPrompts(tasks) {
        const prompts = tasks.map((task) => task.result?.text || task.input?.prompt || "");
        const response = await api.fetchApi("/ts_studio/batch/prompts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session: sessionId, prompts }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    }

    function buildBatchQueue(settings) {
        const phases = [];
        if (settings.enhance) {
            phases.push({
                id: "prompts",
                run: async (task, ctx) => ({
                    text: await enhanceOnePrompt(task.input.prompt, ctx.index),
                }),
            });
        }
        phases.push({
            id: "images",
            run: async (task, ctx) => {
                const seed = seedFor(settings.seedMode, settings.baseSeed, ctx.index);
                const outcome = await run({
                    prompt: task.result?.text || task.input.prompt,
                    seed,
                });
                if (outcome.cancelled) throw new Error(t.batch.states.skipped);
                if (outcome.error) throw new Error(outcome.error);
                const image = outcome.images?.[0];
                if (!image) throw new Error("no image");
                return { seed, url: resultViewUrl(image), image };
            },
        });
        return createBatch({
            phases,
            onProgress: (state) => {
                batchPanel?.render(state);
                if (state.phase === "prompts") {
                    batchPanel?.setStatus(t.batch.phasePrompts(state.done, state.total));
                } else if (state.phase === "images") {
                    batchPanel?.setStatus(t.batch.phaseImages(state.done, state.total));
                }
            },
            onPhaseDone: async (phaseId, state) => {
                if (phaseId === "prompts") await stashBatchPrompts(state.tasks);
            },
        });
    }

    async function startBatch() {
        if (!activeBackend) { batchPanel?.setStatus(t.batch.needModel); return; }
        const prompts = batchPanel.prompts();
        if (!prompts.length) return;
        const settings = batchPanel.settings();
        batchQueue = buildBatchQueue(settings);
        batchQueue.load(prompts.map((prompt) => ({ prompt })));
        batchPanel.setRunning(true);
        try {
            const state = await batchQueue.run();
            batchPanel.setStatus(t.batch.finished(state.done, state.failed));
        } finally {
            batchPanel.setRunning(false);
            batchQueue = null;
        }
    }

    /**
     * Промпты из брошенных картинок. Пустые — не беда: их просто нет.
     *
     * Приходят разобранные `DropItem`, поэтому диск, библиотека и Artius
     * читаются одинаково: у каждого есть `getBlob()`.
     */
    async function readBatchPrompts(items) {
        batchPanel?.setStatus(t.batch.reading);
        const found = [];
        for (const item of items) {
            try {
                const prompt = await promptFromPng(await item.getBlob());
                if (prompt) found.push(prompt);
            } catch (err) {
                console.warn("[TS Studio] could not read a prompt", err);
            }
        }
        const added = batchPanel?.addPrompts(found) || 0;
        batchPanel?.setStatus(added ? t.batch.readSome(added) : t.batch.readNone);
    }

    function ensureBatchMounted() {
        if (!batchPanel) {
            batchPanel = createBatchPanel({
                t: t.batch,
                onStart: () => { startBatch().catch((err) => {
                    batchPanel?.setStatus(String(err?.message || err));
                }); },
                onStop: () => batchQueue?.stop(),
                onFiles: (items) => { readBatchPrompts(items).catch(() => {}); },
            });
            shell.stage.appendChild(batchPanel.element);
        }
        batchPanel.element.style.display = "";
        stage.setVisible(false);
    }

    function leaveBatch() {
        if (batchPanel) batchPanel.element.style.display = "none";
        // ⚠️ Сцену возвращает тот раздел, который её занимает. Инпэйнт делает
        // это сам, поэтому здесь показывать её нельзя — иначе, уходя из пачки
        // в инпэйнт, человек увидит обе поверхности разом.
    }

    function selectMode(modeId) {
        // ⚠️ Клик по вкладке, на которой уже стоишь, не должен ничего менять.
        // Раньше он всё равно доходил до `restore`, а тот возвращал снимок с
        // ПРОШЛОГО ухода из режима — то есть стирал со сцены картинку, которую
        // только что перетащили, вместе с путём исходника. Дальше «Run» честно
        // отвечал, что работать не над чем.
        // Вошли в раздел — рамка снова про будущее, а не про прошлый результат.
        if (modeId === "outpaint") outframeArmed = true;
        const changing = activeModeId !== modeId;
        // Сцена принадлежит режиму: исходник апскейла не имеет отношения к
        // генерации, и «переезжать» за человеком по вкладкам не должен.
        if (changing && activeModeId) stage.remember(activeModeId);
        shell.setMode(modeId);
        activeModeId = modeId;
        // Промпт живёт по режимам: у инпэйнта своя задача, у генерации своя.
        // Снимок уходящей деки помечен её собственной областью (`deckScope`),
        // поэтому переключить область можно прямо здесь.
        memory.setScope(modeId);
        // Инпэйнт рисует на собственном холсте и сцены не касается.
        if (changing && modeId !== "inpaint") stage.restore(modeId);
        tiles.hide();
        // Рамка расширения принадлежит своему разделу; в остальных её нет.
        stage.outframe.hide();
        requestAnimationFrame(() => paintOutFrame());
        rememberWorkspace();
        if (modeId === "inpaint") ensureInpaintMounted();
        else leaveInpaint();
        if (modeId === "batch") ensureBatchMounted();
        else leaveBatch();
        syncIterationPanels();
        // ⚠️ ПОСЛЕ поверхностей: `restore` и `leaveInpaint` гасят ожидание, и
        // вернуть его можно только последним делом.
        restoreWaiting(modeId);
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
        return backendForRun({
            mode: activeModeId,
            backend: activeBackend,
            roles: familiesForMode("generate"),
            refs: values.__refs,
        });
    }

    // ── run ─────────────────────────────────────────────────────────────── //

    /**
     * Прогон и ожидание его конца.
     *
     * Кнопка Run зовёт без аргументов и результата не ждёт. Пачка передаёт
     * свой промпт и свой сид и ждёт — ей нужно знать, когда браться за
     * следующее задание. Путь отправки при этом ОДИН: стили, LoRA, референсы,
     * метаданные и снимок собираются в одном месте, а не в двух похожих.
     *
     * @param {object} [override]
     * @param {string} [override.prompt] Промпт вместо того, что в поле.
     * @param {number} [override.seed] Сид вместо того, что в деке.
     * @returns {Promise<{images?: object[], error?: string, cancelled?: boolean}>}
     */
    function run(override = {}) {
        return new Promise((resolve) => {
            let settled = false;
            const settle = (outcome) => {
                if (settled) return;
                settled = true;
                resolve(outcome);
            };
            startRun(override, settle).then(
                // Прогон не начался (нет модели, нет исходника, отказ сервера)
                // — причину уже показал статус, а ждать больше нечего.
                (promptId) => { if (!promptId) settle({ error: "run did not start" }); },
                (err) => settle({ error: String(err?.message || err) }),
            );
        });
    }

    async function startRun(override, settle) {
        if (!activeBackend) return null;
        const target = runBackend();
        if (!target) return null;

        const seedState = values.seed || { value: 0, randomize: true };
        const seed = override.seed !== undefined
            ? Math.floor(Number(override.seed) || 0)
            : (seedState.randomize ? randomSeed() : Number(seedState.value || 0));
        seedControl?.showSeed(seed);

        // Правила сбора значений живут отдельно (`_runvalues.js`): в граф идёт
        // только объявленное им, сила определяет режим, стили дописываются так
        // же, как это делает нода-селектор.
        const runValues = collectDeckValues(values, target);
        runValues.seed = seed;
        applyStrengthRule(runValues, target);
        // ⚠️ Только когда промпт вообще есть. Раньше строка выполнялась всегда,
        // и в разделе без поля промпта (расширение кадра — текст зашит в граф)
        // в значения попадал `undefined`. Он уходил в узел, при сериализации
        // ключ `value` пропадал, и ComfyUI отвечал 400 «required input
        // missing»: Run молча ничего не делал, а на экране навсегда оставалась
        // «загрузка модели».
        const styleTail = promptTools?.getStylePrompts().join(", ") || "";
        // Промпт пачки заменяет напечатанный, но стили остаются: их выбирают
        // на всю серию, а не на каждый кадр.
        if (typeof override.prompt === "string" && override.prompt.trim()) {
            runValues.prompt = override.prompt.trim();
        }
        if (runValues.prompt !== undefined || styleTail) {
            const merged = withStyles(runValues.prompt ?? "", styleTail);
            if (merged) runValues.prompt = merged;
            else delete runValues.prompt;
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
            if (!prepared) return null;             // status already explains
        }
        // Filled reference slots become image params; empty ones become
        // dropParams so the patcher removes their optional branches.
        // Что лежало на сцене до прогона — для шторки сравнения в апскейле.
        // Что лежало на сцене до прогона — для шторки «до и после». Она нужна
        // везде, где работа идёт НАД картинкой и результат хочется сравнить с
        // исходником: апскейл и перерисовка.
        const sourceBeforeRun = (activeModeId === "upscale" || activeModeId === "img2img")
            ? stage.url() : "";

        // Весь ход прогона считает отдельный автомат (`_progress.js`): этап по
        // узлу, номер тайла по перезапускам сэмплера, доля шагов. Здесь
        // остаётся только показ.
        const progress = createRunProgress({
            classOf: (nodeId) => patched?.[nodeId]?.class_type || "",
        });

        /**
         * Нарисовать ход: заполнение полосы и словами — что происходит.
         *
         * Сэмплер занимает последнюю треть шкалы, всё остальное — первые две.
         * Так видно и что идёт подготовка, и сколько её осталось, а прыжок в
         * конце не съедает весь ход одним махом.
         */
        function paintProgress() {
            if (!deckWidgets) return;
            const now = progress.get();
            const nodePart = now.nodesTotal > 0 ? now.nodesDone / now.nodesTotal : 0;
            const pct = now.samplerFraction === null
                ? Math.round(nodePart * 66)
                : Math.round(66 + now.samplerFraction * 34);
            deckWidgets.progress.classList.add("is-active");
            deckWidgets.progressFill.style.width = `${Math.max(2, Math.min(100, pct))}%`;
            // Тот же ход помнится для возврата на вкладку.
            if (liveWait) {
                liveWait.stage = now.stage || liveWait.stage;
                liveWait.fraction = pct / 100;
            }
            // Ожидание живёт теми же числами, что и полоса под кнопкой: два
            // разных хода на одном экране читались бы как две разные работы.
            if (stage.warmup.isOpen()) {
                stage.warmup.setStage(now.stage);
                stage.warmup.setProgress(pct / 100);
                // ⚠️ Кольцо гаснет НЕ по первому шагу сэмплера, а когда его
                // сменяет настоящая картинка. Между шагом и первым показанным
                // превью проходит секунда-другая (первые кадры — шум, их
                // пропускаем), и в эту дырку выглядывал исходник — скачок,
                // которого человек не просил.
                //
                // В апскейле смена происходит здесь: куски ложатся на кадр
                // ровно на первом шаге, они и есть показ работы.
                if (now.samplerFraction !== null && activeModeId === "upscale"
                    && !tiles.isActive()) {
                    if (plannedGrid) tiles.prepare(plannedGrid);
                    else tiles.warm();
                    stage.warmup.hide();
                }
            }
            const label = t.stages[now.stage] || t.stages.other;
            const steps = now.samplerFraction === null
                ? "" : ` ${Math.round(now.samplerFraction * 100)}%`;
            deckWidgets.hint.textContent = queueCount > 1
                ? `${label}${steps} · ${t.queued(queueCount)}` : `${label}${steps}`;
        }
        const refs = collectRefs(values.__refs, target);
        Object.assign(runValues, refs.values);
        const dropParams = refs.drop;

        if (activeModeId === "inpaint" && inpaintMode) {
            try {
                const collected = await inpaintMode.collectRunValues();
                Object.assign(runValues, collected);
            } catch (err) {
                setStatus(String(err.message || err));
                return;
            }
        }
        if (SOURCE_MODES.has(activeModeId)) {
            // A dropped image wins over the gallery selection: it is the more
            // deliberate act of the two.
            if (stage.source()) {
                runValues.source_image = stage.source();
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
                setStatus(activeModeId === "outpaint" ? t.outNeedsImage
                    : (activeModeId === "img2img" ? t.i2iNeedsImage
                        : t.upscaleNeedsImage));
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
            sessionId,
            sources: {
                source_image: runValues.source_image,
                mask: runValues.mask,
                ...(values.__refs || {}),
            },
        });

        // Линейка версий начинается с того, что сейчас на экране: так любая
        // смена исходника (дроп, библиотека, возврат вкладки) начинает историю
        // заново, и ловить каждую из них по отдельности не нужно.
        const iterating = anchorIterations(activeModeId);
        iterating?.panel.setBusy(true);

        let patched;
        try {
            patched = patchBackend(target.graph, target.spec, {
                values: runValues,
                modelFiles: target.modelFiles,
                loras,
                dropParams,
                filenamePrefix: outputPrefix(target.manifest.family),
                // Итерации пишутся во временную папку: браузер ассетов её не
                // индексирует, и проба не лежит в библиотеке рядом с работой.
                // В библиотеку версию переносит кнопка Save.
                outputFolder: ITERATIVE_MODES.has(activeModeId) ? "temp" : undefined,
                isOptionalInput: optionalIndex,
                promptText: authoredPrompt || (typeof runValues.prompt === "string"
                    ? runValues.prompt : ""),
                studioState,
            });
        } catch (err) {
            setStatus(t.runFailed(err.message));
            return null;
        }
        // Сетка встаёт на кадр ДО отправки: дальше идут десятки секунд
        // загрузки моделей, и пустой экран в это время читается как «ничего не
        // происходит». Пока работа не началась, по клеткам бежит волна.
        if (activeModeId === "upscale") {
            // Ступеней проявления ровно столько, сколько шагов делает сэмплер
            // на кусок: проявление обязано совпадать с работой, а не изображать
            // её.
            tiles.setSteps(Number(runValues.steps) || 0);
            // ⚠️ Сетку СЕЙЧАС не показываем. До первого кадра превью на экране
            // кольцо ожидания; куски раскладываются, когда работа действительно
            // до них дошла. Геометрию запоминаем — считать её потом будет не по
            // чему.
            plannedGrid = plannedTileGrid(patched, runValues);
        } else if (activeModeId === "outpaint") {
            // Рамка уже показывает, куда прирастёт кадр; на время прогона она
            // оживает — это идиома именно этого раздела.
            paintOutFrame();
            stage.outframe.setWorking(true);
        }
        // Этапы прогона показываются ВЕЗДЕ — меняется только вид. Между
        // нажатием и первым превью проходят десятки секунд загрузки моделей, и
        // пустая полоса в панели — единственное, что об этом говорило.
        //
        // Полный вид (во всю область) — только когда показывать нечего.
        // Там, где на экране уже лежит картинка (исходник апскейла, холст
        // инпэйнта, кадр перед расширением), идёт карточка внизу области:
        // закрыть анимацией то, с чем идёт работа, значит спрятать её.
        {
            // Этапы ДО сэмплера занимают всю рабочую область — там, где потом
            // появится превью: смотреть в это время всё равно не на что, а
            // мелкая плашка внизу теряется. Как только сэмплер выдаёт первый
            // кадр (превью или клетку сетки), показ уходит и уступает место
            // картинке — этим занимается `paintProgress`.
            //
            // Исключение — инпэйнт: там рабочая поверхность это ХОЛСТ с маской,
            // которую человек только что нарисовал. Закрывать её целиком нельзя
            // (маску не видно, отменить нечего), поэтому там карточка.
            const size = stage.naturalSize();
            const wanted = Number(runValues.width) > 0 && Number(runValues.height) > 0
                ? Number(runValues.width) / Number(runValues.height)
                : 0;
            const waitArgs = {
                compact: activeModeId === "inpaint",
                ratio: wanted || (size.height > 0 ? size.width / size.height : 0),
                // У апскейла свой ход и своя надпись: человек ждёт не «загрузку
                // модели вообще», а начало апскейла.
                label: activeModeId === "upscale" ? t.upscaleStarting
                    : (activeModeId === "outpaint" ? t.outpainting : ""),
            };
            // Запоминаем ДО показа: если человек уйдёт с вкладки в ту же
            // секунду, вернуть будет чем.
            liveWait = { mode: activeModeId, args: waitArgs, stage: "", fraction: 0,
                         grid: plannedGrid };
            stage.warmup.show(waitArgs);
        }
        // ⚠️ Объявление ВЫШЕ try: номер прогона возвращается наружу, и внутри
        // блока он бы туда не дожил.
        let promptId = null;
        try {
            queueCount += 1;
            updateHint();
            promptId = await runner.submit(patched, {
                onQueued: (id) => {
                    // Помним свои прогоны поимённо: остановка обязана снимать
                    // только их и не трогать то, что человек запустил с холста.
                    liveRuns.add(id);
                    updateHint();
                },
                // The snapshot travels as extra_pnginfo so the saver writes
                // the ts_studio chunk even though the studio sends no
                // LiteGraph workflow of its own.
                pngInfo: { ts_studio: studioState },
                // Ход прогона складывается из двух источников. Узлы дают
                // грубую канву («шестой из девяти»), сэмплер — точный ход
                // внутри своего шага. Пока сэмплер не начал, полоса живёт по
                // узлам: раньше она просто стояла пустой всё время загрузки.
                onNode: (nodeId) => {
                    const { changed } = progress.node(nodeId);
                    // Сетка в апскейле живёт от нажатия до результата: она и
                    // есть индикатор этого режима. Гасить её на смене этапа
                    // значило бы мигать ею на каждом узле графа.
                    if (changed && activeModeId !== "upscale") tiles.hide();
                    paintProgress();
                },
                onNodeProgress: (done, total) => {
                    progress.nodes(done, total);
                    paintProgress();
                },
                onProgress: (value, max, nodeId) => {
                    // Всё, «что это было», решает автомат: тайлы у движка,
                    // который их считает, или номер куска по перезапуску
                    // сэмплера у того, который молчит.
                    const step = progress.progress(value, max, nodeId);
                    if (step.kind === "tiles" && (!tiles.isActive()
                        || tiles.size().total !== step.tileTotal)) {
                        const size = stage.naturalSize();
                        tiles.showByCount(step.tileTotal,
                            size.height > 0 ? size.width / size.height : 1);
                    }
                    if (tiles.isActive()) {
                        // Доля — это ход ВНУТРИ текущего куска: по ней он и
                        // проявляется, а соседние клетки ждут своей очереди.
                        tiles.advance(step.tileIndex, step.fraction);
                    }
                    paintProgress();
                },
                onPreview: (blob) => {
                    const shot = progress.preview(PREVIEW_SKIP_STEPS);
                    // Пока стоит сетка, превью тайла на экран НЕ ставим.
                    // Показать его «на своём месте» нельзя: резчик работает с
                    // перекрытием и собирает куски со смешиванием, так что в
                    // равномерную сетку они не ложатся — кусок оказывался
                    // смещённым. Ход работы показывает сама сетка: клетки
                    // проясняют кадр по мере готовности.
                    if (tiles.isActive()) return;
                    // Расширение: превью уходит ПОД исходник. Сверху остаётся
                    // оригинал, и на экране проявляются ровно те области,
                    // которые дорисовываются.
                    if (activeModeId === "outpaint") {
                        // Первый кадр — чистый шум: показывать его под
                        // исходником незачем, картинка начинается со второго.
                        if (shot.index <= OUTPAINT_SKIP_STEPS) return;
                        stage.showUnderlay(blob);
                        stage.outframe.setPreviewing(true);
                        return;
                    }
                    // Первые шаги — почти чистый шум: быстрый декодер латента
                    // показывает его честно, и поверх лица это выглядит
                    // пугающе, а полезного там ещё нет.
                    if (!shot.show) return;
                    // Пришёл настоящий кадр нового размера — рамка своё
                    // отслужила: дальше заполнение видно по самой картинке.
                    if (activeModeId === "outpaint") stage.outframe.hide();
                    if (activeModeId === "inpaint" && inpaintMode) inpaintMode.showPreview(blob);
                    else showPreviewBlob(blob);
                },
                onDone: (images) => {
                    settle({ images });
                    liveWait = null;
                    queueCount -= 1;
                    liveRuns.delete(promptId);
                    updateHint();
                    deckWidgets.progress.classList.remove("is-active");
                    tiles.hide();
                    stage.warmup.hide();
                    stage.outframe.setWorking(false);
                    stage.clearUnderlay();
                    // Результат пришёл — обещать больше нечего.
                    outframeArmed = false;
                    stage.outframe.hide();
                    iterating?.panel.setBusy(false);
                    for (const image of images) {
                        const result = { image, params: { ...runValues }, state: studioState };
                        gallery.add(result);   // лента сессии показывает и черновики
                        showResult(result);
                        // Ещё одна версия кадра. Черновик помнит, откуда его
                        // потом забирать в библиотеку.
                        iterating?.history.push({
                            url: resultViewUrl(image),
                            annotated: resultAnnotated(image),
                            draft: image.type === "temp" ? { ...image } : null,
                        });
                        persist.setResultPath(resultRelPath(image));
                        // Апскейл показывает пару: слева то, что было.
                        // Исходник запоминается до прогона — showResult его
                        // сбрасывает, потому что дальше сцена живёт результатом.
                        if (sourceBeforeRun) {
                            if (stage.showCompare(sourceBeforeRun, resultViewUrl(image))) {
                                setStatus(t.cmp.shown);
                            }
                        }
                        if (activeModeId === "inpaint" && inpaintMode) {
                            inpaintMode.hidePreview();
                            // Тип берётся у самого результата: перерисовки
                            // приходят из временной папки, а не из библиотеки.
                            inpaintMode.noteDraft?.(image.type === "temp" ? image : null);
                            inpaintMode.acceptRepaintResult(resultViewUrl(image), image.filename)
                                // Кадр посчитан и лежит в галерее, а на холст
                                // не встал: молчать об этом нельзя — человек
                                // увидит прежнюю картинку и решит, что прогон
                                // ничего не сделал.
                                .catch((err) => setStatus(
                                    t.runFailed(String(err?.message || err))));
                        }
                    }
                },
                onError: (message) => {
                    settle({ error: message });
                    liveWait = null;
                    iterating?.panel.setBusy(false);
                    queueCount -= 1;
                    liveRuns.delete(promptId);
                    updateHint();
                    stage.warmup.hide();
                    stage.outframe.setWorking(false);
                    stage.clearUnderlay();
                    deckWidgets.progress.classList.remove("is-active");
                    tiles.hide();
                    setStatus(t.runFailed(message));
                },
                onCancelled: () => {
                    settle({ cancelled: true });
                    liveWait = null;
                    iterating?.panel.setBusy(false);
                    queueCount -= 1;
                    liveRuns.delete(promptId);
                    updateHint();
                    deckWidgets.progress.classList.remove("is-active");
                },
            });
        } catch (err) {
            liveWait = null;
            iterating?.panel.setBusy(false);
            // ⚠️ Показ ожидания обязан погаснуть и здесь. Иначе отказ сервера
            // выглядит как вечная «загрузка модели»: останавливать нечего,
            // кнопки остановки нет, и человек ждёт того, чего не будет.
            stage.warmup.hide();
            stage.outframe.setWorking(false);
            tiles.hide();
            queueCount = Math.max(0, queueCount - 1);
            updateHint();
            deckWidgets?.progress.classList.remove("is-active");
            setStatus(t.runFailed(err.message));
            return null;
        }
        return promptId;
    }

    /**
     * Сказать что-то человеку.
     *
     * ⚠️ Адресат зависит от раздела. Инпэйнт и пачка прячут сцену
     * (`stage.setVisible(false)`), а вместе с ней и её подпись — сообщения об
     * отказе прогона там уходили в скрытый элемент, и «Run failed: …» видел
     * только тот, кто держал открытой консоль браузера. У каждого из этих
     * разделов есть своя строка состояния; ей и говорим.
     */
    function setStatus(message) {
        if (activeModeId === "inpaint" && inpaintMode?.setStatus) {
            inpaintMode.setStatus(message);
        } else if (activeModeId === "batch" && batchPanel) {
            batchPanel.setStatus(message);
        } else {
            stage.setCaption(message, null);
        }
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
        const { applyFrameToDesign } = await import("../../_studio/_editors.js");
        // The node reads width/height from design_json in BOTH modes, so the
        // deck's frame has to travel even when there is no design — otherwise
        // Auto runs fall back to the node's own default aspect (measured: the
        // deck said 1:1 and the render came out wide).
        runValues[designer.param] = JSON.stringify(
            applyFrameToDesign(design || {}, size?.aspect, size?.mp));
        if (design) {
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
        // Backends that were folded into another one still answer for the
        // images they made: the generic second pass became Z-Image's upscale,
        // which is the same recipe with its tile ControlNet added.
        const RETIRED = { "secondpass/upscale": "z-image/upscale" };
        const wanted = RETIRED[state.backend] || state.backend;
        const backend = backends.find((b) => b.manifest?.id === wanted && b.available)
            || backends.find((b) => b.manifest?.family === state.family
                && b.manifest?.mode === state.mode && b.available);
        if (!backend) {
            setStatus(t.pngNoBackend(state.backend));
            return true;
        }
        const uiMode = UI_MODES.find((m) => m.id === state.ui_mode)
            || UI_MODES.find((m) => m.backendModes.includes(backend.manifest.mode));
        // A snapshot replaces the deck wholesale, so nothing carried from the
        // last render may leak into it. What it sets is then remembered like
        // any other change — after a recreate, that IS the current state.
        sessionRefs.clear();
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
        if (sources.mask && (uiMode?.id || backend.manifest.mode) === "inpaint") {
            const maskUrl = annotatedImageUrl(sources.mask);
            if (maskUrl) await inpaintMode?.setMaskFromUrl?.(maskUrl).catch(() => {});
        }
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
            // Молча проглоченная ошибка здесь и выглядела как «вкладка
            // вернулась, а холст пустой»: файл исходника мог не пережить
            // сессию, и об этом никто не узнавал. Теперь это видно в статусе.
            try {
                await inpaintMode.setImageFromUrl(url);
            } catch (err) {
                console.warn("[TS Studio] source not restored", source, err);
                setStatus(t.sourceGone);
            }
        } else if (uiMode === "upscale") {
            stage.show(url, { keepSource: true });
            stage.setSource(source);
        }
    }

    /** A dropped image either recreates its session or becomes the source. */
    async function acceptDroppedImage(item) {
        const blob = await item.getBlob();
        // Inpaint and Upscale work ON an image, so a drop there is always the
        // image to work on — even a studio render, which used to hijack the
        // drop and restore its whole session instead. Rebuilding a session is
        // its own act: the Recreate button, or the browser's own command.
        // ⚠️ Брошенная картинка — это КАРТИНКА, и ничего больше. Студийные PNG
        // несут в себе снимок сессии, и раньше бросок такого файла в генерацию
        // ВОССТАНАВЛИВАЛ ту сессию — вместе с её вкладкой: человек ронял
        // картинку на холст генерации и оказывался в апскейле (жалоба
        // владельца). Восстановление осталось отдельным действием: кнопка
        // «Повторить» под картинкой и пункт меню в браузере ассетов.
        const annotated = await uploadImage(api, blob, item.name || "dropped.png");
        if (activeModeId === "inpaint") {
            ensureInpaintMounted();
            await inpaintMode.setImageFromBlob(blob, item.name || "dropped.png");
        } else if (SOURCE_MODES.has(activeModeId)) {
            // Новая картинка отменяет старое сравнение: шторка показывает
            // пару прошлого прогона и накрыла бы собой то, что человек только
            // что принёс. `show` снимает её сам.
            stage.show(URL.createObjectURL(blob), { keepSource: true });
            stage.setSource(annotated);
            outframeArmed = true;      // новый исходник — снова есть что обещать
            // Рамка считается по настоящему размеру кадра — ждём, пока
            // браузер его узнает.
            requestAnimationFrame(() => paintOutFrame());
        } else {
            // Генерация делает картинку из текста; принесённая картинка здесь
            // может быть только РЕФЕРЕНСОМ. На холст её не кладём: холст
            // показывает результат прогона, и чужая картинка там — обещание,
            // которого прогон не выполнит.
            const refs = controlsByParam.get("__refs");
            const current = refs?.get() || [];
            const slot = current.findIndex((v) => !v);
            // Модель без референсов брошенную картинку использовать не может —
            // тогда не происходит ничего.
            if (!refs) return false;
            if (slot < 0) {
                setStatus(t.refsFull);
                return false;
            }
            current[slot] = annotated;
            refs.set(current);
            setStatus(t.droppedAsReference(slot + 1));
            return true;
        }
        setStatus(t.sourceSet);
        return true;
    }

    // Reachable from outside: an asset browser can hand an image straight to
    // the mode on screen. Same path as a drop, so the two cannot diverge.
    shell.acceptImage = (url, name) => acceptDroppedImage({
        name: name || "image.png",
        getBlob: async () => {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.blob();
        },
    });
    shell.activeMode = () => activeModeId;

    // ── контекстное меню рабочей области ───────────────────────────────── //
    // Сами пункты объявлены данными в `_studio/_ctxmenu.js`; здесь студия даёт
    // им то, чего они знать не могут: какие разделы есть в этой сборке, что
    // сейчас на экране и как отправить картинку соседу.
    // ⚠️ Список разделов передаётся ФУНКЦИЕЙ, а регистрация повторяется при
    // каждой пересборке рельса. Снимок здесь был пустым: модели приезжают
    // асинхронно, а меню собиралось при открытии студии — и в нём оставался
    // один пункт «очистить холст».
    registerStageCommands = () => registerStudioStageCommands({
        modes: () => modeIds,
        modeLabel: (id) => t.modes[id] || id,
        strings: {
            sendTo: t.menuSendTo,
            clear: t.menuClear,
            clearHint: t.menuClearHint,
        },
    });
    registerStageCommands();

    const stageMenu = createContextMenu({
        onError: (err) => setStatus(String(err?.message || err)),
        // Внутрь студии, а не в body: студия — полноэкранный оверлей, и меню
        // из нижнего слоя оказалось бы за ним.
        host: shell.root,
    });

    /**
     * Что сейчас на экране — одинаково для сцены и для холста инпэйнта.
     *
     * Промежуточное превью прогона картинкой не считается: отправлять в другой
     * раздел незаконченный кадр нечестно, и человек об этом узнал бы только по
     * результату.
     */
    function currentImageUrl() {
        if (activeModeId === "inpaint" && inpaintMode) return inpaintMode.imageUrl?.() || "";
        return stage.isFinal() ? stage.url() : "";
    }

    async function sendImageTo(targetMode) {
        const url = currentImageUrl();
        if (!url) return;
        // Переключаемся ПЕРЕД приёмом: `acceptDroppedImage` кладёт картинку
        // туда, где человек находится, — тем же путём, что и перетаскивание,
        // поэтому два способа отдать картинку не могут разойтись.
        selectMode(targetMode);
        await acceptDroppedImage({
            name: "sent.png",
            getBlob: async () => {
                const response = await fetch(url);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return response.blob();
            },
        });
        setStatus(t.menuSent(t.modes[targetMode] || targetMode));
    }

    function clearCanvas() {
        if (activeModeId === "inpaint" && inpaintMode) {
            inpaintMode.clear?.();
        } else {
            stage.clear();
            stage.setSource("");
        }
        tiles.hide();
        stage.outframe.hide();
        rememberWorkspace();
    }

    shell.stage.addEventListener("contextmenu", (event) => {
        // Слушатель висит на всей рабочей области — значит «где показывать»
        // уже решено. Здесь остаётся назвать места, где меню НЕ нужно: это
        // всплывающие панели поверх области.
        //
        // ⚠️ Два способа промахнуться, оба уже случились:
        //   1. «не показывать внутри .ts-ui-modal» — студия САМА живёт в
        //      полноэкранном оверлее с этим классом, и меню молчало везде;
        //   2. «показывать только над .ts-istudio__stagefit» — мимо кадра
        //      (пустая область, край, подпись) клик снова проваливался.
        // Поэтому список — точный и проверяется тестом.
        if (event.target.closest(
            ".ts-ctxmenu, .ts-help, .ts-settings, .ts-showcase, .ts-gate, .ts-queue")) {
            return;
        }
        // ⚠️ НИКАКИХ немых веток. Здесь стояла проверка «картинка ещё не
        // готова — глушим клик», и она съедала правый клик целиком: своего меню
        // нет, браузерное подавлено, нажатие проваливается в пустоту. Хуже
        // отсутствия меню только меню, которое отменяет и чужое.
        //
        // Незавершённость теперь влияет ровно на одно: промежуточный кадр не
        // предлагают отправить соседу (`currentImageUrl`). Само меню
        // открывается всегда, когда в нём есть хоть один пункт.
        const items = commandsFor({
            modeId: activeModeId,
            image: { url: currentImageUrl() },
            canClear: activeModeId === "inpaint" && Boolean(inpaintMode),
            actions: { sendTo: sendImageTo, clear: clearCanvas },
        });
        if (stageMenu.open(event.clientX, event.clientY, items)) event.preventDefault();
    });

    /**
     * Ролик вместо картинки: спрашиваем, какой кадр взять.
     *
     * Спрашиваем ТЕМ ЖЕ меню, что и правый клик: второе окно того же смысла
     * рядом с первым читалось бы как чужая деталь. Пока человек не выбрал,
     * ничего не происходит — угадать за него нельзя.
     *
     * @returns {Promise<object|null>} элемент с кадром или null, если ролик
     *   и выбор ещё впереди (тогда работу продолжит сам выбор)
     */
    async function askForVideoFrame(item, event) {
        const point = {
            x: event?.clientX ?? window.innerWidth / 2,
            y: event?.clientY ?? window.innerHeight / 2,
        };
        stageMenu.open(point.x, point.y, frameChoiceItems({
            strings: { first: t.videoFirst, last: t.videoLast, hint: t.videoHint },
            onPick: async (which) => {
                try {
                    setStatus(t.videoTaking);
                    const blob = await extractFrame(await item.getBlob(), which);
                    await acceptDroppedImage({
                        name: (item.name || "video").replace(/\.[^.]+$/, "")
                            + (which === "last" ? "_last.png" : "_first.png"),
                        getBlob: async () => blob,
                    });
                } catch (err) {
                    setStatus(t.videoFailed(err.message || String(err)));
                }
            },
        }));
        return null;
    }

    const stageDropTeardown = makeDropZone(stage.element, {
        max: 1,
        onDrop: async ([item], event) => {
            if (isVideoItem(item)) {
                await askForVideoFrame(item, event);
                return;
            }
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
        // Сначала — рабочее место прошлого раза. Студия, открытая из ноды,
        // возвращается к своей сессии; открытая без ноды (из браузера
        // ассетов) получает новую сессию каждый раз, поэтому ей отдаётся
        // последнее рабочее место вообще — иначе она открывалась бы пустой
        // всегда. Не получилось применить — открываемся как раньше, с первой
        // вкладки: рабочее место не должно быть причиной не открыться.
        const saved = loadWorkspace(sessionId, !node || !hadSession);
        let restored = false;
        if (saved?.state) {
            restoringWorkspace = true;
            try {
                restored = await applyStudioState(saved.state);
                if (restored && saved.mask && activeModeId === "inpaint") {
                    await inpaintMode?.setMaskFromUrl?.(saved.mask);
                }
            } catch (err) {
                console.warn("[TS Studio] workspace not restored", err);
                restored = false;
            } finally {
                restoringWorkspace = false;
            }
        }
        if (!restored) selectMode(modeIds[0]);
    } else {
        const note = document.createElement("div");
        note.className = "ts-studio__galleryempty";
        note.textContent = t.noBackends;
        shell.deck.appendChild(note);
        for (const backend of backends) {
            console.warn(`[TS Studio] backend ${backend.id}:`, backend.problems.join("; "));
        }
    }
    // Reachable from outside: an asset browser can hand a snapshot straight to
    // the open studio (see js/_studio/_asset_actions.js).
    shell.applyStudioState = (state) => applyStudioState(state);
    // ⚠️ Подстраховка на случай, когда предыдущая студия ушла не через свой
    // `close` (перезагрузили расширение, оборвалась сборка): её корень мог
    // остаться в документе. Живая — ровно одна, остальные убираются молча, но
    // со следом в консоли: это не норма, это чинят.
    for (const stray of document.querySelectorAll(".ts-studio")) {
        if (stray !== shell.root && stray.isConnected) {
            console.warn("[TS Studio] removing a stray studio root");
            stray.remove();
        }
    }
    openInstance = shell;
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
