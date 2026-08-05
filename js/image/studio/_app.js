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
import { createCompare } from "../../_studio/_compare.js";
import { attachZoomPan, clampScale } from "../../_studio/_zoompan.js";
import { createTileGrid } from "../../_studio/_tilegrid.js";
import { createRunner } from "../../_studio/_runner.js";
import { loadBackends, groupByFamily } from "../../_studio/_backends.js";
import { patchBackend } from "../../_studio/_markers.js";
import { newSessionId, outputPrefix, resultRelPath, restoreResults } from "../../_studio/_session.js";
import { mountPromptTools } from "../../_studio/_prompt_tools.js";
import { pickAssetProvider } from "../../_studio/_assets.js";
import { createInpaintMode } from "./_modes_inpaint.js";
import { createDownloadPanel } from "../../_studio/_downloads.js";
import { createHelpPanel } from "../../_studio/_help.js";
import { createSettingsPanel, readSetting, settingsStrings }
    from "../../_studio/_settings.js";
import { uploadImage, makeDropZone, annotatedImageUrl } from "../../_studio/_dnd.js";
import { buildStudioState, studioStateFromPng } from "../../_studio/_pnginfo.js";
import { loadWorkspace, saveWorkspace } from "../../_studio/_workspace.js";
import { REPLACE_DENOISE } from "../../_studio/_crop_geometry.js";
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

// Какой этап идёт прямо сейчас — по классу считающегося узла.
//
// Полоса шагов сэмплера появляется поздно: до неё грузятся веса (десятки
// секунд), считается текстовый энкодер, кодируется картинка. Раньше всё это
// время окно молчало, и отличить «работает» от «зависло» было нельзя.
// Порядок проверок важен: имена классов пересекаются («VAELoader» содержит и
// «VAE», и «Loader»), поэтому список идёт от частного к общему.
const STAGE_BY_CLASS = [
    [/InpaintCrop/i, "crop"],
    [/InpaintRestore/i, "restore"],
    // Собственные ноды пака сами делают всю работу целиком — у Klein это
    // TSSmartInpaint, и он же занимает почти всё время прогона.
    [/SmartInpaint|LamaCleanup/i, "sample"],
    [/Loader$|Loader[A-Z]|LoraStack/i, "load"],
    [/TextEncode|Conditioning|Designer|Guider|Scheduler|Sigmas/i, "prompt"],
    [/VAEEncode|NoiseMask|DifferentialDiffusion|ModelSampling|CFG/i, "encode"],
    [/VAEDecode/i, "decode"],
    [/Sampler|KSampler|Noise$/i, "sample"],
    [/SaveImage|StudioOutput|PreviewImage/i, "save"],
    // Маркеры студии исполняются мгновенно; называть их этапом — только мигать
    // подписью, поэтому они остаются в общем «работаю».
];

function stageOf(classType) {
    const name = String(classType || "");
    for (const [pattern, stage] of STAGE_BY_CLASS) {
        if (pattern.test(name)) return stage;
    }
    return "other";
}

const UI_MODES = [
    { id: "generate", backendModes: ["t2i", "edit"] },
    { id: "inpaint", backendModes: ["inpaint"] },
    { id: "upscale", backendModes: ["upscale"] },
];

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
    inpaint: `<svg ${ICON_ATTRS}><path d="M4 20l.9-3.7L15.6 5.6a2.05 2.05 0 0 1 2.9 2.9L7.7 19.1z"/><path d="M13.9 7.3l2.8 2.8"/></svg>`,
    upscale: `<svg ${ICON_ATTRS}><path d="M4 9V4h5"/><path d="M20 15v5h-5"/><path d="M4 4l6 6"/><path d="M20 20l-6-6"/></svg>`,
    settings: `<svg ${ICON_ATTRS}><path d="M4 7h7M15 7h5M4 12h11M19 12h1M4 17h3M11 17h9"/><circle cx="13" cy="7" r="2"/><circle cx="17" cy="12" r="2"/><circle cx="9" cy="17" r="2"/></svg>`,
    // Packs: a box seen head-on, with the seam a parcel has.
    packs: `<svg ${ICON_ATTRS}><path d="M4 8.5l8-4 8 4v7l-8 4-8-4z"/><path d="M4 8.5l8 4 8-4"/><path d="M12 12.5v7"/></svg>`,
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
        aspectCustom: "w:h",
        aspectCustomTip: "Your own ratio, as width:height — 21:9, 5:4, 2.35:1",
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
        fitView: "Fit to the work area (double-click, or wheel to zoom)",
        run: "Run",
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
        packsNewMode: "The pack brings a new section — reopen the studio to see it.",
        inPack: "in a pack",
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
        aspectCustom: "ш:в",
        aspectCustomTip: "Своё соотношение в виде ширина:высота — 21:9, 5:4, 2.35:1",
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
        fitView: "Вписать в рабочую область (двойной клик; колесо — масштаб)",
        run: "Run",
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
        packsNewMode: "Набор добавил новый раздел — откройте студию заново, чтобы он появился.",
        inPack: "в наборе",
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
.ts-istudio__stagefit{position:absolute;inset:12px;display:flex;align-items:center;
    justify-content:center;overflow:hidden;touch-action:none}
/* Масштаб и сдвиг живут на этой коробке, а не на самой картинке: так их видно
   в одном месте и так же ведёт себя шторка сравнения. */
.ts-istudio__zoom{display:flex;align-items:center;justify-content:center;
    width:100%;height:100%;transform-origin:0 0;will-change:transform}
.ts-istudio__stagefit img{max-width:100%;max-height:100%;object-fit:contain;border-radius:4px}
.ts-istudio__fit{position:absolute;right:10px;top:10px;z-index:6;width:26px;height:26px;
    display:none;align-items:center;justify-content:center;padding:0;cursor:pointer;
    border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);
    background:var(--ts-elevated);color:var(--ts-muted);font-size:13px}
.ts-istudio__fit.is-active{display:flex}
.ts-istudio__fit:hover{color:var(--ts-text)}
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
.ts-istudio__stop{width:100%}
.ts-istudio__runhint{text-align:center;color:var(--ts-muted);font-size:var(--ts-fs-xs)}
.ts-istudio__progress{height:3px;border-radius:2px;background:var(--ts-border-soft);overflow:hidden;display:none}
.ts-istudio__progress.is-active{display:block}
.ts-istudio__progress div{height:100%;width:0%;background:var(--ts-accent);transition:width .2s ease}
`;
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

/** Open the studio for one node instance. Returns the overlay handle. */
export async function openStudio(node, persist) {
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
        loadBackends((url) => fetch(url), objectInfo, (url) => api.fetchApi(url));
    // Not const: installing a pack adds backend files, and the studio rereads
    // them in place rather than asking to be reopened.
    let backends = await readBackends();
    let families = groupByFamily(backends);
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

    const present = new Set([...families.values()].flatMap((f) => [...f.modes.keys()]));
    const uiModes = UI_MODES.filter((m) => m.backendModes.some((b) => present.has(b)));
    const modeIds = uiModes.map((m) => m.id);
    const backendModesOf = (uiMode) =>
        UI_MODES.find((m) => m.id === uiMode)?.backendModes || [uiMode];

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
        const seen = new Set();
        offers = [];
        for (const pack of catalogData?.packs || []) {
            if (pack.installed) continue;
            for (const family of pack.families || []) {
                if (!family?.family || families.has(family.family)) continue;
                if (seen.has(family.family)) continue;
                seen.add(family.family);
                offers.push({ ...family, packId: pack.id });
            }
        }
    }

    /** Offered families that would serve this UI mode. */
    function offersForMode(uiMode) {
        const modes = backendModesOf(uiMode);
        return offers.filter((offer) =>
            (offer.modes || []).some((mode) => modes.includes(mode)));
    }

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
            // Last read before everything is torn down, then written now
            // rather than on the timer — the page may be leaving.
            captureValues();
            memory.flush();
            rememberWorkspaceNow();
            openInstance = null;
            stageDropTeardown?.();
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
            } else if (activeModeId === "inpaint" && inpaintMode
                       && (event.key === "e" || event.key === "E")
                       && !event.ctrlKey && !event.metaKey && !event.altKey) {
                event.preventDefault();
                inpaintMode.toggleEraser();
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
    // Всё, что показывает сцена, лежит внутри коробки зума: колесо приближает,
    // средняя кнопка таскает, кнопка в углу возвращает вписанный вид.
    const stageZoom = document.createElement("div");
    stageZoom.className = "ts-istudio__zoom";
    stageZoom.append(stageImg);
    // Шторка «до и после» — для апскейла: результат нельзя оценить, глядя
    // только на результат.
    const compare = createCompare({ before: t.cmp.before, after: t.cmp.after });
    stageZoom.appendChild(compare.element);
    stageFit.append(stageEmpty, stageZoom);

    // Сетка тайлов ложится поверх картинки на время тайлового прохода.
    const tiles = createTileGrid();
    stageZoom.appendChild(tiles.element);

    const stageFitBtn = document.createElement("button");
    stageFitBtn.type = "button";
    stageFitBtn.className = "ts-istudio__fit";
    stageFitBtn.textContent = "⤢";
    stageFitBtn.title = t.fitView;
    stageFit.appendChild(stageFitBtn);

    const stageView = { scale: 1, x: 0, y: 0 };
    function paintStageView() {
        stageZoom.style.transform =
            `translate(${stageView.x}px, ${stageView.y}px) scale(${stageView.scale})`;
        stageFitBtn.classList.toggle("is-active", stageView.scale !== 1
            || stageView.x !== 0 || stageView.y !== 0);
    }
    function fitStage() {
        stageView.scale = 1;
        stageView.x = 0;
        stageView.y = 0;
        paintStageView();
    }
    attachZoomPan(stageFit, {
        zoomAt(clientX, clientY, factor) {
            const rect = stageFit.getBoundingClientRect();
            const x = clientX - rect.left;
            const y = clientY - rect.top;
            // Вписанный вид — это масштаб 1: картинка уже подогнана правилами
            // CSS. Поэтому нижняя граница здесь единица, а не доля от неё.
            const next = clampScale(stageView.scale * factor, 1);
            if (next === stageView.scale) return;
            stageView.x = x - ((x - stageView.x) / stageView.scale) * next;
            stageView.y = y - ((y - stageView.y) / stageView.scale) * next;
            stageView.scale = next;
            paintStageView();
        },
        panBy(dx, dy) {
            stageView.x += dx;
            stageView.y += dy;
            paintStageView();
        },
        reset: fitStage,
    });
    stageFitBtn.addEventListener("click", fitStage);
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
        compare.hide();
        fitStage();
        upscaleSource = "";
        stageImg.src = `/view?filename=${encodeURIComponent(result.image.filename)}` +
            `&subfolder=${encodeURIComponent(result.image.subfolder || "")}&type=output`;
        stageImg.style.display = "";
        stageEmpty.style.display = "none";
        rememberWorkspace();
        const params = result.params || {};
        const bits = [];
        if (params.width && params.height) bits.push(`${params.width} × ${params.height}`);
        if (params.seed !== undefined) bits.push(`seed ${params.seed}`);
        setCaption(bits.join(" · "), result.state || null);
    }

    function showLibraryAsset(asset) {
        compare.hide();
        fitStage();
        stageImg.src = asset.url;
        // Картинка из библиотеки — тоже то, что на экране: если её взяли под
        // апскейл, она обязана вернуться вместе с рабочим местом.
        if (asset.annotated) upscaleSource = asset.annotated;
        rememberWorkspace();
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
                        initialStyles: memory.recall(graphKey(backend), "__styles", "styles") || [],
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
        } else if (upscaleSource) {
            modeSources = { source_image: upscaleSource };
        } else if (selectedResult?.image) {
            // После прогона на сцене лежит результат, а не то, что положили
            // руками: сохраняем именно его — это и есть «что было на экране».
            const image = selectedResult.image;
            const sub = String(image.subfolder || "").replace(/\\/g, "/");
            const path = sub ? `${sub}/${image.filename}` : image.filename;
            modeSources = { source_image: `${path} [${image.type || "output"}]` };
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
        families = groupByFamily(backends);
        setOffers(null);            // against the families that exist now
        const nowPresent = new Set([...families.values()]
            .flatMap((family) => [...family.modes.keys()]));
        const unseen = UI_MODES.some((mode) => !modeIds.includes(mode.id)
            && mode.backendModes.some((backendMode) => nowPresent.has(backendMode)));
        if (unseen) setStatus(t.packsNewMode);
        if (activeModeId) {
            activeBackend = null;           // the old object is from the old read
            selectMode(activeModeId);
        }
    }

    let inpaintMode = null;
    let activeModeId = null;

    function ensureInpaintMounted() {
        if (!inpaintMode) {
            inpaintMode = createInpaintMode({
                api, t, sessionId,
                onEngineChange: () => {},
                /** Перенести черновик в библиотеку и показать его в галерее. */
                keepDraft: async (draft) => {
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
                },
                // Смена картинки на холсте — не изменение контрола, и раньше
                // она в снимок рабочего места не попадала: он писался только на
                // правку деки и на закрытие. Если студию закрывали не кнопкой
                // (перезагрузка страницы, пересборка графа), исходник терялся —
                // вкладка возвращалась, а холст был пуст.
                onSourceChange: () => rememberWorkspace(),
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
        // Промпт живёт по режимам: у инпэйнта своя задача, у генерации своя.
        // Снимок уходящей деки помечен её собственной областью (`deckScope`),
        // поэтому переключить область можно прямо здесь.
        memory.setScope(modeId);
        // Шторка сравнения принадлежит апскейлу: это его результат рядом с его
        // исходником. В генерации она показывала бы чужую пару.
        if (modeId !== "upscale" && compare.isActive()) {
            compare.hide();
            stageImg.style.display = stageImg.src ? "" : "none";
        }
        rememberWorkspace();
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
        // Режим выводится из силы, а не из отдельного тумблера.
        //
        // Раньше их было два — ползунок и переключатель Replace, — и при
        // включённом переключателе ползунок ничего не значил. Теперь шкала
        // одна: её последняя ступень И ЕСТЬ замена. Всё, что ниже порога, —
        // доработка.
        const strength = Number(runValues.denoise);
        if (Number.isFinite(strength)) {
            const replacing = strength >= REPLACE_DENOISE;
            if (replacing) runValues.denoise = 1.0;
            // Klein принимает режим прямо в ноду; у остальных семейств такого
            // входа в графе нет, и передавать его туда нельзя — патчер
            // справедливо ругается на параметр, которого граф не объявлял.
            if (target.spec.params.has("replace") || target.spec.literals?.has("replace")) {
                runValues.replace = replacing;
            }
            // Доработка идёт БЕЗ размышлений LanPaint.
            //
            // Его внутренний цикл согласует перерисованное с окружением, и на
            // полной замене это то, что нужно. На доработке он же уводит
            // результат далеко за запрошенную силу: замерено на одном сиде —
            // средний сдвиг пикселя внутри маски 69.5 против 32.8 у обычного
            // прохода, и молодое лицо при силе 0.45 возвращалось пожилым.
            //
            // Ноль размышлений вырождает LanPaint в обычный сэмплер: сверено
            // с настоящим KSampler на том же сиде — среднее расхождение 0.057
            // из 255. То есть это и есть классический инпэйнт, только без
            // второй ветки в графе. Заодно доработка ускоряется в девять раз
            // (24 с против 209 с).
            if (!replacing && target.spec.params.has("think_steps")) {
                runValues.think_steps = 0;
            }
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
        // Что лежало на сцене до прогона — для шторки сравнения в апскейле.
        const sourceBeforeRun = activeModeId === "upscale" && stageImg.src
            && stageImg.style.display !== "none" ? stageImg.src : "";
        // Считаем превью этого прогона: пропуск первых шагов должен работать
        // на каждом запуске, а не один раз за сессию.
        let previewsSeen = 0;
        let runStage = "";
        let samplerFraction = null;
        let nodesDone = 0;
        let nodesTotal = 0;

        /**
         * Нарисовать ход: заполнение полосы и словами — что происходит.
         *
         * Сэмплер занимает последнюю треть шкалы, всё остальное — первые две.
         * Так видно и что идёт подготовка, и сколько её осталось, а прыжок в
         * конце не съедает весь ход одним махом.
         */
        function paintProgress() {
            if (!deckWidgets) return;
            const nodePart = nodesTotal > 0 ? nodesDone / nodesTotal : 0;
            const pct = samplerFraction === null
                ? Math.round(nodePart * 66)
                : Math.round(66 + samplerFraction * 34);
            deckWidgets.progress.classList.add("is-active");
            deckWidgets.progressFill.style.width = `${Math.max(2, Math.min(100, pct))}%`;
            const label = t.stages[runStage] || t.stages.other;
            const steps = samplerFraction === null ? "" : ` ${Math.round(samplerFraction * 100)}%`;
            deckWidgets.hint.textContent = queueCount > 1
                ? `${label}${steps} · ${t.queued(queueCount)}` : `${label}${steps}`;
        }
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
            sessionId,
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
                filenamePrefix: outputPrefix(target.manifest.family),
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
            const promptId = await runner.submit(patched, {
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
                    const stage = stageOf(patched?.[nodeId]?.class_type);
                    if (stage !== runStage) tiles.hide();
                    runStage = stage;
                    if (stage !== "sample") samplerFraction = null;
                    paintProgress();
                },
                onNodeProgress: (done, total) => {
                    nodesDone = done;
                    nodesTotal = total;
                    paintProgress();
                },
                onProgress: (value, max, nodeId) => {
                    // Этап определяем по узлу из самого события: `executing`
                    // на этой сборке приходит пустым, и без этого весь прогон
                    // выглядел безликим «работаю».
                    if (nodeId !== undefined && patched?.[nodeId]) {
                        const stage = stageOf(patched[nodeId].class_type);
                        if (stage !== runStage) tiles.hide();
                        runStage = stage;
                    }
                    // Тайловый VAE отчитывается за каждый тайл — по этому же
                    // событию, только `max` там равен их числу, а не шагам
                    // сэмплера. Отличаем по этапу: на кодировании и сборке
                    // картинки прогресс — это тайлы, и показывать его надо
                    // сеткой поверх кадра, а не полосой внизу.
                    const tiled = (runStage === "decode" || runStage === "encode") && max > 1;
                    if (tiled) {
                        const area = stageImg.getBoundingClientRect();
                        const host = stageZoom.getBoundingClientRect();
                        if (!tiles.isActive()) tiles.show(max, area, host);
                        else tiles.place(area, host);
                        tiles.advance(value);
                        nodesDone = Math.max(nodesDone, 0);
                        paintProgress();
                        return;
                    }
                    samplerFraction = max ? value / max : null;
                    runStage = "sample";
                    paintProgress();
                },
                onPreview: (blob) => {
                    // Пока идёт тайловый проход, превью — это НЕ кадр целиком,
                    // а отдельные куски по 512 пикселей. Каждый такой кусок
                    // подменял бы картинку на сцене целиком, и вместо работы
                    // человек видел мельтешение тайлов, растянутых во весь
                    // экран. Ход этого прохода показывает сетка поверх
                    // неподвижного кадра — куски здесь просто выбрасываются.
                    if (tiles.isActive()) return;
                    // Первые шаги — почти чистый шум: быстрый декодер латента
                    // показывает его честно, и выглядит это пугающе, особенно
                    // когда оно лежит поверх лица. Ничего полезного там ещё
                    // нет, поэтому показ начинается с третьего шага.
                    previewsSeen += 1;
                    if (previewsSeen <= PREVIEW_SKIP_STEPS) return;
                    if (activeModeId === "inpaint" && inpaintMode) inpaintMode.showPreview(blob);
                    else showPreviewBlob(blob);
                },
                onDone: (images) => {
                    queueCount -= 1;
                    liveRuns.delete(promptId);
                    updateHint();
                    deckWidgets.progress.classList.remove("is-active");
                    tiles.hide();
                    for (const image of images) {
                        const result = { image, params: { ...runValues }, state: studioState };
                        gallery.add(result);   // лента сессии показывает и черновики
                        showResult(result);
                        persist.setResultPath(resultRelPath(image));
                        // Апскейл показывает пару: слева то, что было.
                        // Исходник запоминается до прогона — showResult его
                        // сбрасывает, потому что дальше сцена живёт результатом.
                        if (activeModeId === "upscale" && sourceBeforeRun) {
                            const resultUrl = "/view?" + new URLSearchParams({
                                filename: image.filename,
                                subfolder: image.subfolder || "",
                                type: image.type || "output",
                            });
                            if (compare.show(sourceBeforeRun, resultUrl)) {
                                stageImg.style.display = "none";
                                setStatus(t.cmp.shown);
                            }
                        }
                        if (activeModeId === "inpaint" && inpaintMode) {
                            inpaintMode.hidePreview();
                            // Тип берётся у самого результата: перерисовки
                            // приходят из временной папки, а не из библиотеки.
                            const url = "/view?" + new URLSearchParams({
                                filename: image.filename,
                                subfolder: image.subfolder || "",
                                type: image.type || "output",
                            });
                            inpaintMode.noteDraft?.(image.type === "temp" ? image : null);
                            inpaintMode.acceptRepaintResult(url, image.filename)
                                .catch(() => {});
                        }
                    }
                },
                onError: (message) => {
                    queueCount -= 1;
                    liveRuns.delete(promptId);
                    updateHint();
                    deckWidgets.progress.classList.remove("is-active");
                    setStatus(t.runFailed(message));
                },
                onCancelled: () => {
                    queueCount -= 1;
                    liveRuns.delete(promptId);
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
            upscaleSource = source;
            stageImg.src = url;
            stageImg.style.display = "";
            stageEmpty.style.display = "none";
        }
    }

    /** A dropped image either recreates its session or becomes the source. */
    async function acceptDroppedImage(item) {
        const blob = await item.getBlob();
        // Inpaint and Upscale work ON an image, so a drop there is always the
        // image to work on — even a studio render, which used to hijack the
        // drop and restore its whole session instead. Rebuilding a session is
        // its own act: the Recreate button, or the browser's own command.
        const worksOnSource = activeModeId === "inpaint" || activeModeId === "upscale";
        const found = worksOnSource ? null : await studioStateFromPng(blob);
        if (found) return applyStudioState(found.state);
        const annotated = await uploadImage(api, blob, item.name || "dropped.png");
        if (activeModeId === "inpaint") {
            ensureInpaintMounted();
            await inpaintMode.setImageFromBlob(blob, item.name || "dropped.png");
        } else if (activeModeId === "upscale") {
            upscaleSource = annotated;
            // Новая картинка отменяет старое сравнение: шторка показывает пару
            // прошлого прогона и накрыла бы собой то, что человек только что
            // принёс.
            compare.hide();
            fitStage();
            stageImg.src = URL.createObjectURL(blob);
            stageImg.style.display = "";
            stageEmpty.style.display = "none";
            rememberWorkspace();
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
