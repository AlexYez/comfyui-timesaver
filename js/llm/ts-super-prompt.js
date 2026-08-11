// TS_SuperPrompt frontend — full DOM-rendered UI built around a single
// flex-grow textarea. The schema still exposes ``text``, ``high_quality``,
// ``system_preset`` and ``attached_image`` so workflows serialise correctly,
// but the standard ComfyUI widgets are hidden — every control lives inside
// the DOM widget so the visual stack stays consistent.
//
// Layout (top → bottom inside the DOM widget):
//   • compact toolbar (~30px): attach button (shows mini thumbnail when set),
//     high-quality toggle, preset select, record button, AI prompt button;
//   • prompt textarea (flex: 1, the main UI surface);
//   • thin status / progress strip (~16px).

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import { TS_UI_CLASS, ensureThemeStyles, pickLocaleStrings } from "../_theme.js";
import { addResizableDomWidget, hideWidget } from "../_dom_widget.js";
// The pack already has one drag-and-drop service, and it is the one that
// knows how an Artius card, an OS file and a ComfyUI preview each arrive —
// including the fallback for a drag that starts inside a shadow root and
// reaches us with its MIME types stripped. A second copy of that knowledge
// would go stale the day Artius changes anything.
import { makeDropZone } from "../_studio/_dnd.js";

const EXTENSION_ID = "ts.superPrompt";
const NODE_NAME = "TS_SuperPrompt";
const DOM_WIDGET_NAME = "ts_super_prompt_ui";

const VOICE_ROUTE_BASE = "/ts_voice_recognition";
const AI_ROUTE_BASE = "/ts_super_prompt";
const UPLOAD_ROUTE = "/upload/image";
const VOICE_EVENT_PREFIX = "ts_voice_recognition";
const AI_EVENT_PREFIX = "ts_super_prompt";

const TEXT_WIDGET = "text";
const HIGH_QUALITY_WIDGET = "high_quality";
const SYSTEM_PRESET_WIDGET = "system_preset";
const ATTACHED_IMAGE_WIDGET = "attached_image";
const ATTACHED_IMAGE_2_WIDGET = "attached_image_2";
const BIGGER_MODEL_WIDGET = "bigger_model";
// Two slots, and the order is the meaning: one image is a reference, two
// are the first and the last frame of the shot being described.
const IMAGE_SLOTS = [ATTACHED_IMAGE_WIDGET, ATTACHED_IMAGE_2_WIDGET];

const DEFAULT_MODEL = "base";
const HIGH_QUALITY_MODEL = "turbo";
const AUDIO_BITS_PER_SECOND = 128_000;
const PROGRESS_CLEAR_DELAY_MS = 900;
const STATUS_RESET_DELAY_MS = 2400;

const IMAGE_ACCEPT = "image/*";
const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"]);
const MIME_CANDIDATES = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
];

const STYLE_ID = "ts-super-prompt-style";
const STYLE_TEXT = `
/* In-flow flex fill inside the DOM-widget slot — NOT position:absolute. An
   absolute+inset:0 container escapes to the nearest positioned ancestor (none
   of the Vue node wrappers are positioned), so it floated over the node title
   and the whole canvas. Transparent background lets the node's own surface and
   title bar show; the inner controls carry their own colours. */
.ts-sp{position:relative;width:100%;height:100%;min-height:0;display:flex;flex-direction:column;gap:4px;padding:4px;
    color:var(--ts-text);font-family:var(--ts-font);font-size:var(--ts-fs-sm);line-height:1.3;box-sizing:border-box;}
.ts-sp.is-drag-over{outline:2px dashed var(--ts-accent-line);outline-offset:-3px;border-radius:6px}
.ts-sp__bar{display:flex;align-items:center;gap:6px;height:26px;flex:0 0 auto}
/* Микрофон и HQ — одно управление распознаванием речи. Общая рамка вместо
   зазора: порознь «HQ» читалось как отдельная функция неизвестно от чего.
   Правка чисто внешняя — ни имён, ни значений виджетов она не трогает, и на
   сохранённые workflow не влияет. */
.ts-sp__group{display:inline-flex;align-items:center;gap:0;flex:0 0 auto;
    border:1px solid var(--ts-border);border-radius:7px;overflow:hidden}
.ts-sp__group:hover{border-color:var(--ts-border-strong)}
.ts-sp__group > *{border-radius:0 !important;border:none !important;margin:0}
.ts-sp__group > * + *{border-left:1px solid var(--ts-border) !important}
.ts-sp__textarea{flex:1 1 auto;min-height:0;width:100%;resize:none;box-sizing:border-box;
    padding:6px 8px;border-radius:6px;border:1px solid var(--ts-border-soft);
    background:var(--ts-sunken);color:var(--ts-text);font-family:inherit;font-size:var(--ts-fs);line-height:1.4;
    outline:none;transition:border-color .15s,background .15s}
.ts-sp__textarea:focus{border-color:var(--ts-accent-line);background:var(--ts-sunken)}
.ts-sp__textarea::placeholder{color:var(--ts-faint)}
.ts-sp__btn{display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto;
    width:26px;height:26px;padding:0;border-radius:6px;border:1px solid var(--ts-border);
    background:var(--ts-surface);color:var(--ts-text);cursor:pointer;
    transition:background .15s,border-color .15s,color .15s,transform .08s;user-select:none}
.ts-sp__btn:hover:not(:disabled){background:var(--ts-accent-soft);border-color:var(--ts-accent-line)}
.ts-sp__btn:active:not(:disabled){transform:translateY(1px)}
.ts-sp__btn:disabled{opacity:.5;cursor:not-allowed}
.ts-sp__btn svg{width:14px;height:14px;display:block;fill:none;stroke:currentColor;
    stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
/* Red here is semantic (mic is live), so it stays red — but routed through the
   pack's muted --ts-danger instead of a raw neon red. */
.ts-sp__btn--record.is-recording{background:var(--ts-danger);border-color:var(--ts-danger);color:var(--ts-bg);
    box-shadow:0 0 0 3px color-mix(in srgb,var(--ts-danger) 22%,transparent)}
.ts-sp__btn--record.is-recording:hover{background:color-mix(in srgb,var(--ts-danger) 84%,#000)}
.ts-sp__pill{display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto;
    height:26px;padding:0 9px;border-radius:6px;border:1px solid var(--ts-border);
    background:var(--ts-surface);color:var(--ts-muted);font-size:var(--ts-fs-xs);font-weight:700;
    letter-spacing:.4px;cursor:pointer;transition:background .15s,border-color .15s,color .15s,transform .08s;
    user-select:none;font-family:inherit}
.ts-sp__pill:hover:not(:disabled){background:var(--ts-accent-soft);border-color:var(--ts-accent-line);color:var(--ts-text)}
.ts-sp__pill:active:not(:disabled){transform:translateY(1px)}
.ts-sp__pill:disabled{opacity:.5;cursor:not-allowed}
.ts-sp__pill--toggle.is-on{background:var(--ts-accent);border-color:var(--ts-accent-strong);color:var(--ts-accent-contrast)}
.ts-sp__pill--ai{letter-spacing:.6px}
.ts-sp__attach{position:relative;flex:0 0 auto;width:26px;height:26px;border-radius:6px;
    overflow:hidden;border:1px solid var(--ts-border);background:var(--ts-surface);
    color:var(--ts-text);cursor:pointer;transition:background .15s,border-color .15s;padding:0;
    display:inline-flex;align-items:center;justify-content:center}
.ts-sp__attach:hover{background:var(--ts-accent-soft);border-color:var(--ts-accent-line)}
.ts-sp__attach svg{width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:2;
    stroke-linecap:round;stroke-linejoin:round}
.ts-sp__attach.has-image{border-color:var(--ts-accent)}
.ts-sp__attach-thumb{position:absolute;inset:0;background-position:center;background-size:cover;
    background-repeat:no-repeat;display:none}
.ts-sp__attach.has-image .ts-sp__attach-thumb{display:block}
.ts-sp__attach.has-image .ts-sp__attach-icon{display:none}
/* Deliberate exception: these two badges float ON TOP of the attached
   thumbnail, so they need a fixed dark chip + white glyph to stay legible over
   any image.
   The clear badge sits INSIDE the button. It used to be 14px at -3px, which on
   a 26px button meant a badge more than half the width of the thing it closed,
   with its outer half sliced off by the overflow:hidden above. */
.ts-sp__attach-clear{position:absolute;top:1px;right:1px;width:11px;height:11px;border-radius:50%;
    border:1px solid rgba(255,255,255,.18);background:#0a0d12;color:#fff;font-size:9px;line-height:1;
    cursor:pointer;display:none;align-items:center;justify-content:center;padding:0;
    box-shadow:0 1px 2px rgba(0,0,0,.5)}
.ts-sp__attach.has-image .ts-sp__attach-clear{display:flex}
/* Which frame this is. Shown only when BOTH slots are filled: a lone reference
   image is not the first frame of anything. */
.ts-sp__attach-badge{position:absolute;left:1px;bottom:1px;min-width:11px;height:11px;padding:0 2px;
    border-radius:3px;border:1px solid rgba(255,255,255,.18);background:#0a0d12;color:#fff;
    font-size:9px;line-height:9px;font-weight:700;letter-spacing:.02em;
    display:none;align-items:center;justify-content:center;pointer-events:none}
.ts-sp__attach.has-frame .ts-sp__attach-badge{display:flex}
.ts-sp__attach.is-dragging{opacity:.45}
.ts-sp__attach.is-drop-target{border-color:var(--ts-accent);box-shadow:0 0 0 2px var(--ts-accent-soft)}
.ts-sp__attach-clear:hover{background:var(--ts-danger);border-color:var(--ts-danger)}
.ts-sp__select{flex:1 1 auto;min-width:0;height:26px;padding:0 6px;border-radius:6px;
    border:1px solid var(--ts-border);background:var(--ts-sunken);color:var(--ts-text);
    font-size:var(--ts-fs-sm);font-family:inherit;cursor:pointer;outline:none;
    -webkit-appearance:none;-moz-appearance:none;appearance:none;
    /* Deliberate exception: a data: URI cannot read CSS custom properties, so
       the chevron fill stays a literal. */
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23e6e9ef' d='M0 0l5 6 5-6z'/%3E%3C/svg%3E");
    background-repeat:no-repeat;background-position:right 6px center;padding-right:18px}
.ts-sp__select:focus{border-color:var(--ts-accent-line)}
.ts-sp__select option{background:var(--ts-surface);color:var(--ts-text)}
.ts-sp__status{display:flex;align-items:center;gap:6px;min-height:14px;flex:0 0 auto;
    font-size:var(--ts-fs-sm);color:var(--ts-muted)}
.ts-sp__status-text{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ts-sp__status.is-error .ts-sp__status-text{color:var(--ts-danger)}
.ts-sp__progress{flex:0 0 70px;height:3px;border-radius:2px;background:var(--ts-border-soft);
    overflow:hidden;display:none;position:relative}
.ts-sp__progress.is-active{display:block}
.ts-sp__progress-fill{height:100%;background:linear-gradient(90deg,var(--ts-accent-strong),var(--ts-accent));width:0%;
    transition:width .2s ease-out}
.ts-sp__progress.is-indeterminate .ts-sp__progress-fill{
    width:35%;animation:ts-sp-indeterminate 1.1s linear infinite}
@keyframes ts-sp-indeterminate{
    0%{transform:translateX(-100%)}100%{transform:translateX(285%)}
}
.ts-sp__file{position:fixed;left:-9999px;top:-9999px}
/* Оверлей работы: накрывает ноду целиком, пока модель занята.
   ⚠️ Он именно НАКРЫВАЕТ, а не блокирует по одному: пока идёт генерация,
   нажимать в этой ноде нечего, кроме отмены, и пусть это будет видно. */
.ts-sp__busy{position:absolute;inset:0;z-index:40;display:none;
    flex-direction:column;align-items:center;justify-content:center;gap:14px;
    padding:18px;box-sizing:border-box;text-align:center;
    background:color-mix(in srgb, var(--ts-bg) 88%, transparent);
    backdrop-filter:blur(3px);border-radius:var(--ts-radius)}
.ts-sp__busy.is-on{display:flex}
.ts-sp__busy-stage{font-size:15px;font-weight:600;color:var(--ts-text);
    line-height:1.25;max-width:92%}
.ts-sp__busy-detail{font-size:12px;color:var(--ts-text-dim);min-height:1.2em;
    max-width:92%;word-break:break-word}
.ts-sp__busy-track{position:relative;width:min(340px,92%);height:8px;
    border-radius:99px;overflow:hidden;
    background:color-mix(in srgb, var(--ts-text) 12%, transparent)}
.ts-sp__busy-fill{position:absolute;inset:0 auto 0 0;width:0%;border-radius:99px;
    background:linear-gradient(90deg,
        color-mix(in srgb, var(--ts-accent) 60%, transparent), var(--ts-accent));
    transition:width .35s ease}
/* Пока процент неизвестен (скачивание без общего размера) — бегущая полоса,
   а не замерший ноль: замерший ноль читается как «зависло». */
.ts-sp__busy.is-waiting .ts-sp__busy-fill{width:35%;
    animation:ts-sp-slide 1.4s ease-in-out infinite}
@keyframes ts-sp-slide{0%{transform:translateX(-110%)}100%{transform:translateX(320%)}}
.ts-sp__busy-steps{display:flex;gap:6px;flex-wrap:wrap;justify-content:center}
.ts-sp__busy-step{font-size:10px;letter-spacing:.04em;text-transform:uppercase;
    padding:3px 8px;border-radius:99px;color:var(--ts-text-dim);
    border:1px solid var(--ts-border)}
.ts-sp__busy-step.is-done{color:var(--ts-text);
    border-color:color-mix(in srgb, var(--ts-accent) 45%, var(--ts-border))}
.ts-sp__busy-step.is-now{color:var(--ts-bg);background:var(--ts-accent);
    border-color:var(--ts-accent)}
.ts-sp__busy-cancel{min-width:150px;font-size:13px;padding:9px 18px}
`;

const SVG_ICON_MIC = `<svg viewBox="0 0 24 24"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 19v3"/></svg>`;
const SVG_ICON_STOP = `<svg viewBox="0 0 24 24"><rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" stroke="none"/></svg>`;
const SVG_ICON_IMAGE = `<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="11" r="1.6" fill="currentColor" stroke="none"/><path d="M3 17l5-5 4 4 3-3 6 6"/></svg>`;

// User-visible strings (en is the base; ru overrides per key). Backend-sent
// messages (WS event ``text`` payloads, server ``error`` fields) are shown
// as received and are NOT in this table.
const STRINGS = {
    en: {
        aiHqTitle: "Higher-quality prompt model (4B). Off: the fast 2B one. The 4B model is downloaded on first use.",
        busyPrepare: "Preparing",
        busyDownload: "Downloading the model",
        busyLoad: "Loading the model into memory",
        busyGenerate: "Writing the prompt",
        busyCancel: "Cancel",
        busyCancelling: "Cancelling…",
        busyCancelled: "Cancelled",
        busyStepPrepare: "prepare",
        busyStepDownload: "download",
        busyStepLoad: "load",
        busyStepGenerate: "write",
        attachTitle: "Attach image (drop from Artius / paste / click)",
        attachSecondTitle: "Attach a second image — the two become the first and last frame",
        firstFrame: "first frame",
        lastFrame: "last frame",
        attachedFrameTitle: (role, name) => `${role}: ${name} (click × to remove)`,
        bothSlotsFull: "Both image slots are taken — remove one first",
        dragToSwap: "Drag onto the other image to swap the frames",
        readingInput: "Running the connected branch…",
        usingInputFrames: (count) => `Using ${count} frame${count === 1 ? "" : "s"} from the input`,
        inputNotReadable: "The connected branch produced no image — check what is wired into it, or disconnect it and attach an image here",
        framesSwapped: "Frames swapped",
        attachClearTitle: "Remove image",
        attachedTitle: (name) => `Attached: ${name} (click × to remove)`,
        hqTitle: "High Quality voice: Whisper turbo (large-v3 turbo). Off: fast base model.",
        recordTitle: "Record from the microphone. Click again while recording to stop and transcribe.",
        aiLabel: "Enhance",
        aiTitle: "Enhances the text via Huihui-Qwen3.5-2B-abliterated. If an image is attached, it is used as a reference.",
        presetTitle: "System preset for prompt enhancement.",
        placeholder: "Prompt. Use the microphone, attach an image and press AI prompt to enhance.",
        ready: "Ready",
        stopAndTranscribe: "Stop recording and transcribe",
        stopModelLoading: "Stop recording (model still loading)",
        working: "Working...",
        missingDependency: (dep) => `Missing dependency: ${dep}`,
        recordIdle: "Record from the microphone",
        recordModelLoading: "Record (model still loading)",
        recordModelWillLoad: "Record (model loads on stop)",
        aiBusyTitle: "AI working...",
        aiIdleTitle: "Enhance the prompt via AI",
        voiceModelProgress: (percent) => `Voice model ${percent}%`,
        workingStatus: "Working",
        voiceModelReady: "Voice model ready",
        voiceError: (text) => `Voice error: ${text}`,
        failed: "failed",
        aiProgressFallback: "AI Prompt",
        aiReady: "AI prompt ready",
        aiError: (text) => `AI error: ${text}`,
        voiceUnavailable: "Voice unavailable",
        missingShort: (dep) => `Missing ${dep}`,
        downloadingModel: (model) => `Downloading ${model}...`,
        preloadFailed: "preload failed",
        recognizing: "Recognizing speech...",
        noSpeech: "No speech detected",
        speechInserted: "Speech inserted",
        micUnsupported: "Microphone unsupported",
        openingMic: "Opening microphone...",
        recording: "Recording...",
        micError: (message) => `Mic error: ${message}`,
        noAudio: "No audio captured",
        waitingModel: "Waiting for voice model...",
        modelNotReady: "Voice model not ready",
        preparingAudio: "Preparing audio...",
        noPromptOrImage: "No prompt text or image",
        startingAi: "Starting AI prompt...",
        enhanceFailed: "enhance failed",
        imageSuffix: " (image)",
        emptyAiResult: "Empty AI result",
        notImageFile: "Not an image file",
        uploadingImage: "Uploading image...",
        imageAttached: "Image attached",
        uploadError: (message) => `Upload error: ${message}`,
        imageRemoved: "Image removed",
        finishRecordingHq: "Finish recording before switching HQ",
    },
    ru: {
        aiHqTitle: "Модель промпта покрупнее (4B). Выключено — быстрая 2B. Четырёхмиллиардная скачивается при первом включении.",
        busyPrepare: "Подготовка",
        busyDownload: "Скачивание модели",
        busyLoad: "Загрузка модели в память",
        busyGenerate: "Пишем промпт",
        busyCancel: "Отмена",
        busyCancelling: "Отменяем…",
        busyCancelled: "Отменено",
        busyStepPrepare: "подготовка",
        busyStepDownload: "скачивание",
        busyStepLoad: "загрузка",
        busyStepGenerate: "написание",
        attachTitle: "Прикрепить изображение (перетащить из Artius / вставить / клик)",
        attachSecondTitle: "Прикрепить вторую — вместе они станут первым и последним кадром",
        firstFrame: "первый кадр",
        lastFrame: "последний кадр",
        attachedFrameTitle: (role, name) => `${role}: ${name} (нажмите ×, чтобы убрать)`,
        bothSlotsFull: "Оба места заняты — сначала уберите одно",
        dragToSwap: "Перетащите на вторую картинку, чтобы поменять кадры местами",
        readingInput: "Прогоняю подключённую ветку…",
        usingInputFrames: (count) => `Беру ${count} кадр(а) со входа`,
        inputNotReadable: "Подключённая ветка не дала картинки — проверьте, что в неё воткнуто, или отключите вход и приложите картинку здесь",
        framesSwapped: "Кадры поменялись местами",
        attachClearTitle: "Убрать изображение",
        attachedTitle: (name) => `Прикреплено: ${name} (нажмите ×, чтобы убрать)`,
        hqTitle: "Качество распознавания: вкл — Whisper turbo (large-v3), выкл — быстрая базовая модель.",
        recordTitle: "Запись с микрофона. Нажмите ещё раз во время записи, чтобы остановить и распознать.",
        aiLabel: "Улучшить",
        aiTitle: "Улучшает текст через Huihui-Qwen3.5-2B-abliterated. Если прикреплена картинка — она используется как референс.",
        presetTitle: "Системный пресет для улучшения промпта.",
        placeholder: "Промпт. Используйте микрофон, прикрепите картинку и нажмите AI prompt для улучшения.",
        ready: "Готово",
        stopAndTranscribe: "Остановить запись и распознать",
        stopModelLoading: "Остановить запись (модель догружается)",
        working: "Работаю...",
        missingDependency: (dep) => `Не хватает зависимости: ${dep}`,
        recordIdle: "Запись с микрофона",
        recordModelLoading: "Запись (модель ещё догружается)",
        recordModelWillLoad: "Запись (модель загрузится при остановке)",
        aiBusyTitle: "AI работает...",
        aiIdleTitle: "Улучшить промпт через AI",
        voiceModelProgress: (percent) => `Голосовая модель ${percent}%`,
        workingStatus: "Работаю",
        voiceModelReady: "Голосовая модель готова",
        voiceError: (text) => `Ошибка голоса: ${text}`,
        failed: "сбой",
        aiProgressFallback: "AI промпт",
        aiReady: "AI промпт готов",
        aiError: (text) => `Ошибка AI: ${text}`,
        voiceUnavailable: "Голос недоступен",
        missingShort: (dep) => `Не хватает ${dep}`,
        downloadingModel: (model) => `Загрузка ${model}...`,
        preloadFailed: "сбой загрузки модели",
        recognizing: "Распознавание речи...",
        noSpeech: "Речь не распознана",
        speechInserted: "Текст добавлен",
        micUnsupported: "Микрофон не поддерживается",
        openingMic: "Открытие микрофона...",
        recording: "Запись...",
        micError: (message) => `Ошибка микрофона: ${message}`,
        noAudio: "Звук не записан",
        waitingModel: "Ожидание голосовой модели...",
        modelNotReady: "Голосовая модель не готова",
        preparingAudio: "Подготовка аудио...",
        noPromptOrImage: "Нет текста промпта или изображения",
        startingAi: "Запуск AI промпта...",
        enhanceFailed: "сбой улучшения",
        imageSuffix: " (изображение)",
        emptyAiResult: "Пустой результат AI",
        notImageFile: "Файл не является изображением",
        uploadingImage: "Загрузка изображения...",
        imageAttached: "Изображение прикреплено",
        uploadError: (message) => `Ошибка загрузки: ${message}`,
        imageRemoved: "Изображение убрано",
        finishRecordingHq: "Завершите запись перед сменой HQ",
    },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Стадии работы, в порядке прохождения. Сопоставление идёт по тексту, который
// шлёт сервер: он один и тот же в обоих местах, и держать здесь копию процентов
// значило бы обещать точность, которой нет.
const BUSY_STEPS = [
    { label: "busyStepPrepare", title: "busyPrepare",
      match: /prepar|waiting|checking/i },
    { label: "busyStepDownload", title: "busyDownload",
      match: /download/i },
    { label: "busyStepLoad", title: "busyLoad",
      match: /loading|loaded|memory|offline|found locally|model files/i },
    { label: "busyStepGenerate", title: "busyGenerate",
      match: /generat|writing|prompt|token|unloading/i },
];


function busyStepIndexFor(text) {
    const line = String(text || "");
    // Идём с конца: «Loading Qwen model into memory» содержит и «model», и
    // «loading», а стадия у неё поздняя.
    for (let index = BUSY_STEPS.length - 1; index >= 0; index -= 1) {
        if (BUSY_STEPS[index].match.test(line)) return index;
    }
    return 0;
}


function ensureStylesInjected(doc) {
    // Colours come from the shared --ts-* tokens in js/_theme.js; the
    // stylesheet below is layout only.
    ensureThemeStyles();
    if (!doc || doc.getElementById(STYLE_ID)) return;
    const styleEl = doc.createElement("style");
    styleEl.id = STYLE_ID;
    styleEl.textContent = STYLE_TEXT;
    doc.head.appendChild(styleEl);
}

function getWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || node?._tsHiddenWidgets?.[name] || null;
}

function getWidgetValue(node, name, fallback = null) {
    const widget = getWidget(node, name);
    return widget?.value ?? fallback;
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return false;
    if (widget.value === value) return true;
    widget.value = value;
    if (typeof widget.callback === "function") {
        try {
            widget.callback(value);
        } catch {
            // Some widgets are picky about callback signatures — ignore.
        }
    }
    return true;
}

function toBoolean(value) {
    if (typeof value === "boolean") return value;
    if (typeof value === "string") {
        return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
    }
    return Boolean(value);
}

function setDirty(node) {
    node?.setDirtyCanvas?.(true, true);
    app?.graph?.setDirtyCanvas?.(true, true);
}

async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(data.error || response.statusText || `HTTP ${response.status}`);
    }
    return data;
}

function createAudioRecorder(stream, mimeType) {
    const options = { audioBitsPerSecond: AUDIO_BITS_PER_SECOND };
    if (mimeType) options.mimeType = mimeType;
    try {
        return new MediaRecorder(stream, options);
    } catch {
        return mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    }
}

function buildAnnotatedPath(uploadPayload) {
    const filename = String(uploadPayload?.name || "").trim();
    if (!filename) return "";
    const uploadType = String(uploadPayload?.type || "input").trim() || "input";
    const subfolder = String(uploadPayload?.subfolder || "")
        .trim()
        .replace(/\\/g, "/")
        .replace(/^\/+|\/+$/g, "");
    return subfolder ? `${subfolder}/${filename} [${uploadType}]` : `${filename} [${uploadType}]`;
}

function resolveAnnotatedThumbUrl(annotatedPath) {
    if (!annotatedPath) return "";
    const match = annotatedPath.match(/^(.+?)\s*\[([^\]]+)\]\s*$/);
    if (!match) return "";
    const rawPath = match[1].trim();
    const type = match[2].trim() || "input";
    const segments = rawPath.split("/").filter(Boolean);
    if (segments.length === 0) return "";
    const filename = segments.pop();
    const subfolder = segments.join("/");
    const params = new URLSearchParams({ filename, type });
    if (subfolder) params.set("subfolder", subfolder);
    params.set("t", String(Date.now()));
    return `/view?${params.toString()}`;
}

function fileExtensionOk(name) {
    const idx = String(name || "").lastIndexOf(".");
    if (idx < 0) return false;
    return IMAGE_EXTS.has(name.slice(idx).toLowerCase());
}

function clampPercent(value) {
    if (!Number.isFinite(Number(value))) return null;
    return Math.max(0, Math.min(100, Number(value)));
}

function getPresetOptions(node) {
    const widget = getWidget(node, SYSTEM_PRESET_WIDGET);
    const values = widget?.options?.values;
    if (Array.isArray(values) && values.length) return values;
    const current = widget?.value;
    return current ? [String(current)] : ["Prompts enhance"];
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

function setupSuperPrompt(node) {
    if (!node) return;
    if (typeof node._tsSuperPromptCleanup === "function") {
        node._tsSuperPromptCleanup();
    }

    const L = pickLocaleStrings(STRINGS);

    // Hide every native widget — the DOM widget renders all controls. The shared
    // hideWidget also drops each converted-input row from the Nodes 2.0 grid and
    // guards the value against the hidden-type serialization skip.
    for (const widgetName of [TEXT_WIDGET, HIGH_QUALITY_WIDGET, SYSTEM_PRESET_WIDGET,
        ATTACHED_IMAGE_WIDGET, ATTACHED_IMAGE_2_WIDGET]) {
        hideWidget(node, widgetName);
    }

    const doc = node?.graph?.canvas?.canvas?.ownerDocument || document;
    ensureStylesInjected(doc);

    const container = doc.createElement("div");
    container.className = `${TS_UI_CLASS} ts-sp`;
    container.setAttribute("data-ts-super-prompt", "1");

    // -------- Toolbar --------
    const bar = doc.createElement("div");
    bar.className = "ts-sp__bar";

    // One attach button per slot. The second only appears once the first is
    // taken: an empty second square next to an empty first one is a question
    // nobody asked, and most prompts want no image at all.
    function buildAttachButton(slot) {
        const button = doc.createElement("button");
        button.type = "button";
        button.className = "ts-sp__attach";
        button.dataset.slot = String(slot);
        const icon = doc.createElement("span");
        icon.className = "ts-sp__attach-icon";
        icon.innerHTML = SVG_ICON_IMAGE;
        const thumb = doc.createElement("span");
        thumb.className = "ts-sp__attach-thumb";
        const clear = doc.createElement("button");
        clear.type = "button";
        clear.className = "ts-sp__attach-clear";
        clear.textContent = "×";
        clear.title = L.attachClearTitle;
        const badge = doc.createElement("span");
        badge.className = "ts-sp__attach-badge";
        button.append(icon, thumb, badge, clear);
        return { button, thumb, clear, badge, slot };
    }

    const attachSlots = [buildAttachButton(0), buildAttachButton(1)];
    const attachBtn = attachSlots[0].button;

    // ---- Voice group: HQ toggle + record button (both speak to Whisper). ----
    const voiceGroup = doc.createElement("div");
    voiceGroup.className = "ts-sp__group";

    const hqToggle = doc.createElement("button");
    hqToggle.type = "button";
    hqToggle.className = "ts-sp__pill ts-sp__pill--toggle";
    hqToggle.title = L.hqTitle;
    hqToggle.textContent = "HQ";

    const recordBtn = doc.createElement("button");
    recordBtn.type = "button";
    recordBtn.className = "ts-sp__btn ts-sp__btn--record";
    recordBtn.title = L.recordTitle;
    recordBtn.innerHTML = SVG_ICON_MIC;

    // Mic first (primary action), HQ flag right next to it.
    voiceGroup.append(recordBtn, hqToggle);

    // ---- AI group: the enhance pill + its own HQ flag. ----
    // Ровно та же пара, что у микрофона: действие и флажок качества рядом. У
    // голоса HQ означает крупную модель распознавания, здесь — крупную модель
    // промпта. Держать это виджетом в теле ноды было неверно: галочка
    // относится к кнопке, а не к параметрам прогона.
    const aiGroup = doc.createElement("div");
    aiGroup.className = "ts-sp__group";

    const aiBtn = doc.createElement("button");
    aiBtn.type = "button";
    aiBtn.className = "ts-sp__pill ts-sp__pill--ai";
    aiBtn.title = L.aiTitle;
    // The primary action names itself. "AI" sat between two icon buttons as
    // an abbreviation nobody can decode without hovering, and it is the one
    // control in this toolbar that does the node's actual work.
    aiBtn.textContent = L.aiLabel;

    const aiHqToggle = doc.createElement("button");
    aiHqToggle.type = "button";
    aiHqToggle.className = "ts-sp__pill ts-sp__pill--toggle";
    aiHqToggle.title = L.aiHqTitle;
    aiHqToggle.textContent = "HQ";

    aiGroup.append(aiBtn, aiHqToggle);

    // ---- Preset select (fills remaining toolbar space). ----
    const presetSelect = doc.createElement("select");
    presetSelect.className = "ts-sp__select";
    presetSelect.title = L.presetTitle;
    for (const opt of getPresetOptions(node)) {
        const option = doc.createElement("option");
        option.value = String(opt);
        option.textContent = String(opt);
        presetSelect.appendChild(option);
    }

    // Order: [🎤 + HQ] voice · [🖼 attach] · [AI] · [preset ▼]
    // Mic is the primary input action so it leads; HQ sits with it as the
    // voice-quality flag. The image button visually groups with AI because
    // both feed the prompt-enhance pipeline. Preset stretches to fill.
    bar.append(voiceGroup, attachSlots[0].button, attachSlots[1].button,
        aiGroup, presetSelect);

    // -------- Textarea (main surface) --------
    const textarea = doc.createElement("textarea");
    textarea.className = "ts-sp__textarea";
    textarea.placeholder = L.placeholder;
    textarea.spellcheck = false;

    // -------- Status row --------
    const statusRow = doc.createElement("div");
    statusRow.className = "ts-sp__status";
    const statusText = doc.createElement("span");
    statusText.className = "ts-sp__status-text";
    statusText.textContent = L.ready;
    const progress = doc.createElement("div");
    progress.className = "ts-sp__progress";
    const progressFill = doc.createElement("div");
    progressFill.className = "ts-sp__progress-fill";
    progress.appendChild(progressFill);
    statusRow.append(statusText, progress);

    // Hidden file input for the attach picker.
    const fileInput = doc.createElement("input");
    fileInput.type = "file";
    fileInput.accept = IMAGE_ACCEPT;
    fileInput.className = "ts-sp__file";

    // -------- Экран работы --------
    // Пока модель занята, в ноде нечего нажимать, кроме отмены. Оверлей это и
    // показывает: крупная стадия, дорожка стадий, полоса и одна кнопка.
    const busy = doc.createElement("div");
    busy.className = "ts-sp__busy";
    const busyStage = doc.createElement("div");
    busyStage.className = "ts-sp__busy-stage";
    const busySteps = doc.createElement("div");
    busySteps.className = "ts-sp__busy-steps";
    const busyStepEls = BUSY_STEPS.map((step) => {
        const el = doc.createElement("span");
        el.className = "ts-sp__busy-step";
        el.textContent = L[step.label];
        busySteps.appendChild(el);
        return el;
    });
    const busyTrack = doc.createElement("div");
    busyTrack.className = "ts-sp__busy-track";
    const busyFill = doc.createElement("div");
    busyFill.className = "ts-sp__busy-fill";
    busyTrack.appendChild(busyFill);
    const busyDetail = doc.createElement("div");
    busyDetail.className = "ts-sp__busy-detail";
    const busyCancel = doc.createElement("button");
    busyCancel.type = "button";
    busyCancel.className = "ts-ui-btn ts-sp__busy-cancel";
    busyCancel.textContent = L.busyCancel;
    busy.append(busyStage, busySteps, busyTrack, busyDetail, busyCancel);

    // Виджет остаётся в схеме (значение обязано доехать до `execute` и до
    // сохранённого workflow), но из тела ноды убирается: им управляет кнопка HQ.
    hideWidget(node, BIGGER_MODEL_WIDGET);
    refreshAiHqToggle();

    container.append(bar, textarea, statusRow, fileInput, busy);

    // -----------------------------------------------------------------
    // State
    // -----------------------------------------------------------------
    let disposed = false;
    let mediaRecorder = null;
    let mediaStream = null;
    let chunks = [];
    let statusResetTimer = 0;
    let progressClearTimer = 0;

    const state = {
        activeModelName: DEFAULT_MODEL,
        isRecording: false,
        isVoiceBusy: false,        // transcription step (after stop)
        isModelLoading: false,     // background preload triggered by record click
        modelReadyPromise: null,   // awaited in onstop so transcribe waits for the model
        isAiBusy: false,
        modelReady: false,
        missingDependencies: [],
        activeAiOperationId: "",
        attachedImages: IMAGE_SLOTS.map(
            (widget) => String(getWidgetValue(node, widget, "") || "")),
    };

    // Pull the latest values from the (hidden) native widgets into the DOM
    // UI. Run on first paint, after copy/paste, and after workflow load —
    // LiteGraph restores widget values after onNodeCreated, so we need an
    // explicit pull-through point.
    function syncUiFromWidgets() {
        if (disposed) return;
        textarea.value = String(getWidgetValue(node, TEXT_WIDGET, "") || "");
        hqToggle.classList.toggle(
            "is-on",
            toBoolean(getWidgetValue(node, HIGH_QUALITY_WIDGET, false)),
        );
        const preset = String(getWidgetValue(node, SYSTEM_PRESET_WIDGET, "") || "");
        if (preset && Array.from(presetSelect.options).some((o) => o.value === preset)) {
            presetSelect.value = preset;
        }
        state.attachedImages = IMAGE_SLOTS.map(
            (widget) => String(getWidgetValue(node, widget, "") || ""));
        renderAttached();
    }

    // Initial values from hidden widgets.
    syncUiFromWidgets();

    // -----------------------------------------------------------------
    // Sync helpers
    // -----------------------------------------------------------------
    function syncTextFromUi() {
        setWidgetValue(node, TEXT_WIDGET, textarea.value);
    }
    function syncHighQualityFromUi() {
        setWidgetValue(node, HIGH_QUALITY_WIDGET, hqToggle.classList.contains("is-on"));
    }
    function syncPresetFromUi() {
        setWidgetValue(node, SYSTEM_PRESET_WIDGET, presetSelect.value);
    }

    function isHighQualityEnabled() {
        return hqToggle.classList.contains("is-on");
    }
    function getActiveVoiceModel() {
        return isHighQualityEnabled() ? HIGH_QUALITY_MODEL : DEFAULT_MODEL;
    }
    function syncActiveVoiceModel() {
        const modelName = getActiveVoiceModel();
        if (state.activeModelName !== modelName) {
            state.activeModelName = modelName;
            state.modelReady = false;
        }
        return modelName;
    }

    let busyStepShown = -1;

    function showBusy(on) {
        busy.classList.toggle("is-on", Boolean(on));
        if (on) {
            busyCancel.disabled = false;
            busyCancel.textContent = L.busyCancel;
            busyStepShown = -1;
            setBusyStage(L.busyPrepare, null, L.busyPrepare);
        }
    }

    function setBusyStage(serverText, percent, forcedTitle = "") {
        const index = forcedTitle ? 0 : busyStepIndexFor(serverText);
        if (index !== busyStepShown) {
            busyStepShown = index;
            busyStepEls.forEach((el, i) => {
                el.classList.toggle("is-done", i < index);
                el.classList.toggle("is-now", i === index);
            });
        }
        busyStage.textContent = forcedTitle || L[BUSY_STEPS[index].title];
        // Строка от сервера — подпись под стадией: она конкретнее (какой файл,
        // сколько мегабайт) и меняется чаще, чем сама стадия.
        busyDetail.textContent = forcedTitle ? "" : String(serverText || "");
        const known = Number.isFinite(percent);
        busy.classList.toggle("is-waiting", !known);
        if (known) busyFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    }

    busyCancel.addEventListener("click", async () => {
        if (busyCancel.disabled) return;
        busyCancel.disabled = true;
        busyCancel.textContent = L.busyCancelling;
        busyDetail.textContent = "";
        try {
            await api.fetchApi(`${AI_ROUTE_BASE}/cancel`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ operation_id: state.activeAiOperationId }),
            });
        } catch (error) {
            console.warn("[TS Super Prompt] cancel failed", error);
        }
    });

    function setProgress({ percent, active, error, indeterminate }) {
        window.clearTimeout(progressClearTimer);
        progress.classList.toggle("is-indeterminate", Boolean(indeterminate));
        if (error) {
            progress.classList.remove("is-active");
            progressFill.style.width = "0%";
            return;
        }
        if (active) {
            progress.classList.add("is-active");
            if (Number.isFinite(percent)) {
                progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
            }
            return;
        }
        if (Number.isFinite(percent) && percent >= 100) {
            progress.classList.add("is-active");
            progressFill.style.width = "100%";
            progressClearTimer = window.setTimeout(() => {
                progress.classList.remove("is-active");
                progressFill.style.width = "0%";
            }, PROGRESS_CLEAR_DELAY_MS);
            return;
        }
        progress.classList.remove("is-active");
        progressFill.style.width = "0%";
    }

    function setStatus(text, kind = "info", resetMs = 0) {
        window.clearTimeout(statusResetTimer);
        statusText.textContent = String(text || "");
        statusRow.classList.toggle("is-error", kind === "error");
        if (resetMs > 0) {
            statusResetTimer = window.setTimeout(() => {
                if (disposed) return;
                statusText.textContent = L.ready;
                statusRow.classList.remove("is-error");
            }, resetMs);
        }
    }

    function refreshRecordButton() {
        syncActiveVoiceModel();
        if (state.isRecording) {
            recordBtn.classList.add("is-recording");
            recordBtn.innerHTML = SVG_ICON_STOP;
            recordBtn.disabled = false;
            recordBtn.title = state.modelReady ? L.stopAndTranscribe : L.stopModelLoading;
            return;
        }
        recordBtn.classList.remove("is-recording");
        recordBtn.innerHTML = SVG_ICON_MIC;
        if (state.isVoiceBusy) {
            recordBtn.disabled = true;
            recordBtn.title = L.working;
            return;
        }
        if (state.missingDependencies.length > 0) {
            recordBtn.disabled = true;
            recordBtn.title = L.missingDependency(state.missingDependencies[0]);
            return;
        }
        // Model loading in the background does NOT disable the button —
        // the user can start recording immediately and onstop will wait
        // for the download to finish before transcription.
        recordBtn.disabled = state.isAiBusy;
        if (state.modelReady) {
            recordBtn.title = L.recordIdle;
        } else if (state.isModelLoading) {
            recordBtn.title = L.recordModelLoading;
        } else {
            recordBtn.title = L.recordModelWillLoad;
        }
    }
    function refreshAiButton() {
        if (state.isAiBusy) {
            aiBtn.disabled = true;
            aiBtn.title = L.aiBusyTitle;
            return;
        }
        aiBtn.disabled = state.isRecording || state.isVoiceBusy;
        aiBtn.title = L.aiIdleTitle;
    }

    function shortName(annotated) {
        const match = annotated.match(/^(.+?)\s*\[/);
        const path = match ? match[1] : annotated;
        const segments = path.split("/");
        return segments[segments.length - 1];
    }

    function renderAttached() {
        const [first, second] = state.attachedImages;
        // Two images mean a shot with a start and an end, so the buttons say
        // which is which. One image is just a reference and gets the plain
        // wording — calling it "first frame" would promise a video nobody asked
        // for.
        const paired = Boolean(first && second);
        attachSlots.forEach(({ button, thumb, badge, slot }) => {
            const annotated = state.attachedImages[slot] || "";
            // The second square appears when the first is taken, and stays
            // while it holds something of its own.
            const visible = slot === 0 || Boolean(first) || Boolean(annotated);
            button.style.display = visible ? "" : "none";
            button.classList.toggle("has-image", Boolean(annotated));
            // The digit says which frame this is. Only with two images: on a
            // lone reference it would label the first frame of a shot that
            // does not exist.
            button.classList.toggle("has-frame", paired);
            badge.textContent = slot === 0 ? "1" : "2";
            // Only a filled slot can be picked up, and only a pair is worth
            // reordering.
            button.draggable = paired;
            const url = annotated ? resolveAnnotatedThumbUrl(annotated) : "";
            thumb.style.backgroundImage = url ? `url("${url}")` : "";
            if (annotated) {
                button.title = paired
                    ? `${L.attachedFrameTitle(slot === 0 ? L.firstFrame : L.lastFrame,
                        shortName(annotated))}\n${L.dragToSwap}`
                    : L.attachedTitle(shortName(annotated));
            } else {
                button.title = slot === 0 ? L.attachTitle : L.attachSecondTitle;
            }
        });
    }

    function setAttachedImage(annotated, slot = 0) {
        const value = String(annotated || "");
        state.attachedImages[slot] = value;
        setWidgetValue(node, IMAGE_SLOTS[slot], value);
        // Clearing the first of two leaves a hole where an ordered pair used
        // to be, and "last frame with no first frame" is not a thing. The
        // second image slides down into the empty slot.
        if (!value && slot === 0 && state.attachedImages[1]) {
            state.attachedImages[0] = state.attachedImages[1];
            state.attachedImages[1] = "";
            setWidgetValue(node, IMAGE_SLOTS[0], state.attachedImages[0]);
            setWidgetValue(node, IMAGE_SLOTS[1], "");
        }
        renderAttached();
        setDirty(node);
    }

    /** The slot a newly attached image should land in, or -1 when both are full. */
    function nextFreeSlot() {
        const index = state.attachedImages.findIndex((value) => !value);
        return index;
    }

    // -----------------------------------------------------------------
    // WebSocket events
    // -----------------------------------------------------------------
    function matchesActiveModel(detail) {
        return !detail?.model || !state.activeModelName || detail.model === state.activeModelName;
    }
    function matchesActiveAiOperation(detail) {
        // An event carrying somebody else's operation id is somebody else's
        // business: the Ideogram node now drives the same engine, and an idle
        // SuperPrompt used to display ITS progress ("Generating AI prompt 78%")
        // because "no operation of my own" was treated as "match anything".
        // Events with no id at all stay accepted — that is how this node's own
        // queue-time runs report themselves.
        if (!detail?.operation_id) return true;
        return detail.operation_id === state.activeAiOperationId;
    }

    function onVoiceProgress(event) {
        if (!matchesActiveModel(event.detail)) return;
        const percent = Number(event.detail?.percent || 0);
        setStatus(L.voiceModelProgress(Math.round(percent)));
        setProgress({ percent, active: true });
    }
    function onVoiceStatus(event) {
        if (!matchesActiveModel(event.detail)) return;
        // The "Recording..." status is owned by startRecording — never let
        // a server-side progress event overwrite it while the mic is open.
        if (state.isRecording) return;
        // Only show server progress while a voice action is in flight.
        if (!state.isVoiceBusy && !state.isModelLoading) return;
        const text = String(event.detail?.text || L.workingStatus);
        const percent = clampPercent(event.detail?.percent);
        setStatus(text);
        setProgress({ percent, active: true, indeterminate: percent === null });
    }
    function onVoiceDone(event) {
        if (!matchesActiveModel(event.detail)) return;
        state.modelReady = true;
        // isVoiceBusy / isModelLoading are owned by HTTP promises (download
        // and transcribe) — leave them alone here. Just update the UI if
        // nothing else is using the status bar.
        if (!state.isRecording && !state.isVoiceBusy) {
            setStatus(L.voiceModelReady, "info", STATUS_RESET_DELAY_MS);
            setProgress({ percent: 100, active: false });
        }
        refreshRecordButton();
        refreshAiButton();
    }
    function onVoiceError(event) {
        if (!matchesActiveModel(event.detail)) return;
        // Same ownership rule: don't flip flags from a WS event, the
        // owning HTTP promise's catch branch handles that. Show the
        // error only if the user isn't actively recording (the message
        // would just disappear once the local "Recording..." status
        // refreshes).
        if (!state.isRecording) {
            setStatus(L.voiceError(event.detail?.text || L.failed), "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
        }
        refreshRecordButton();
        refreshAiButton();
    }
    function onAiProgress(event) {
        if (!matchesActiveAiOperation(event.detail)) return;
        const text = String(event.detail?.text || L.aiProgressFallback);
        const percent = clampPercent(event.detail?.percent);
        setStatus(text);
        setBusyStage(text, percent);
        setProgress({ percent, active: true, indeterminate: percent === null });
    }
    function onAiDone(event) {
        if (!matchesActiveAiOperation(event.detail)) return;
        setStatus(String(event.detail?.text || L.aiReady), "info", STATUS_RESET_DELAY_MS);
        setProgress({ percent: 100, active: false });
    }
    function onAiError(event) {
        if (!matchesActiveAiOperation(event.detail)) return;
        setStatus(L.aiError(event.detail?.text || L.failed), "error", STATUS_RESET_DELAY_MS);
        setProgress({ active: false, error: true });
    }

    api.addEventListener(`${VOICE_EVENT_PREFIX}.progress`, onVoiceProgress);
    api.addEventListener(`${VOICE_EVENT_PREFIX}.status`, onVoiceStatus);
    api.addEventListener(`${VOICE_EVENT_PREFIX}.done`, onVoiceDone);
    api.addEventListener(`${VOICE_EVENT_PREFIX}.error`, onVoiceError);
    api.addEventListener(`${AI_EVENT_PREFIX}.progress`, onAiProgress);
    api.addEventListener(`${AI_EVENT_PREFIX}.done`, onAiDone);
    api.addEventListener(`${AI_EVENT_PREFIX}.error`, onAiError);

    // -----------------------------------------------------------------
    // Text manipulation (cursor-preserving insert + full replace)
    // -----------------------------------------------------------------
    let savedSelection = null;
    function rememberCursor() {
        if (doc.activeElement === textarea) {
            savedSelection = {
                start: textarea.selectionStart ?? textarea.value.length,
                end: textarea.selectionEnd ?? textarea.value.length,
            };
        } else {
            savedSelection = null;
        }
    }
    function insertRecognizedText(newText) {
        const text = String(newText || "").trim();
        if (!text) return false;
        const currentValue = textarea.value;
        let combined;
        let cursorPosition = null;
        if (savedSelection) {
            const start = Math.max(0, savedSelection.start);
            const end = Math.max(start, savedSelection.end);
            const before = currentValue.slice(0, start);
            const after = currentValue.slice(end);
            const prefix = before.length > 0 && !/\s$/.test(before) ? " " : "";
            const suffix = after.length > 0 && !/^\s/.test(after) ? " " : "";
            const inserted = `${prefix}${text}${suffix}`;
            combined = `${before}${inserted}${after}`;
            cursorPosition = start + inserted.length;
        } else {
            const separator = currentValue.length > 0 && !/\s$/.test(currentValue) ? " " : "";
            combined = `${currentValue}${separator}${text}`;
        }
        textarea.value = combined;
        syncTextFromUi();
        if (cursorPosition !== null) {
            textarea.selectionStart = cursorPosition;
            textarea.selectionEnd = cursorPosition;
            textarea.focus();
        }
        savedSelection = null;
        setDirty(node);
        return true;
    }
    function replaceText(newText) {
        const text = String(newText || "").trim();
        if (!text) return false;
        textarea.value = text;
        syncTextFromUi();
        setDirty(node);
        return true;
    }

    // -----------------------------------------------------------------
    // Voice
    // -----------------------------------------------------------------
    async function refreshStatus() {
        const modelName = syncActiveVoiceModel();
        const params = new URLSearchParams({
            model: modelName,
            high_quality: isHighQualityEnabled() ? "true" : "false",
        });
        try {
            const data = await fetchJson(`${VOICE_ROUTE_BASE}/status?${params.toString()}`);
            const info = data[state.activeModelName] || {};
            state.modelReady = Boolean(info.downloaded);
            state.missingDependencies = Array.isArray(info.missing_dependencies)
                ? info.missing_dependencies
                : [];
            setStatus(L.ready);
            setProgress({ active: false });
        } catch {
            state.modelReady = false;
            state.missingDependencies = [];
            setStatus(L.voiceUnavailable, "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
        }
        refreshRecordButton();
        refreshAiButton();
    }

    function downloadVoiceModel(force = false) {
        syncActiveVoiceModel();
        if (state.missingDependencies.length > 0) {
            setStatus(L.missingShort(state.missingDependencies[0]), "error");
            refreshRecordButton();
            return Promise.reject(new Error(`Missing ${state.missingDependencies[0]}`));
        }
        // Reuse the in-flight promise if a previous click already started
        // a download — clicking record twice in a row must not start two
        // parallel /preload requests.
        if (state.isModelLoading && state.modelReadyPromise) {
            return state.modelReadyPromise;
        }
        state.isModelLoading = true;
        // Don't clobber the "Recording..." status when the download was
        // triggered as a side-effect of starting a recording. Only show
        // the download status if the user is not actively recording.
        if (!state.isRecording) {
            setStatus(L.downloadingModel(state.activeModelName));
            setProgress({ active: true, indeterminate: true });
        }
        refreshRecordButton();

        const promise = (async () => {
            try {
                const data = await fetchJson(`${VOICE_ROUTE_BASE}/preload`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        model: state.activeModelName,
                        high_quality: isHighQualityEnabled(),
                        force,
                    }),
                });
                if (!data.ok) throw new Error(data.error || L.preloadFailed);
                state.modelReady = true;
                // Only surface "ready" UI if nothing else is using the
                // status bar — recording/transcribe own the bar otherwise.
                if (!state.isRecording && !state.isVoiceBusy) {
                    setStatus(L.voiceModelReady, "info", STATUS_RESET_DELAY_MS);
                    setProgress({ percent: 100, active: false });
                }
            } catch (error) {
                if (!state.isRecording && !state.isVoiceBusy) {
                    setStatus(L.voiceError(error.message), "error", STATUS_RESET_DELAY_MS);
                    setProgress({ active: false, error: true });
                }
                throw error;
            } finally {
                state.isModelLoading = false;
                refreshRecordButton();
                refreshAiButton();
            }
        })();
        state.modelReadyPromise = promise;
        // Swallow rejection on the stored handle so the unhandled-rejection
        // tracker stays quiet — actual error handling happens at the await
        // sites (onRecordClick caller and onstop awaiter).
        promise.catch(() => {});
        return promise;
    }

    async function sendAudioToServer(blob) {
        syncActiveVoiceModel();
        const form = new FormData();
        form.append("model", state.activeModelName);
        form.append("high_quality", isHighQualityEnabled() ? "true" : "false");
        form.append("audio", blob, "recording.webm");
        setStatus(L.recognizing);
        setProgress({ active: true, indeterminate: true });
        try {
            const data = await fetchJson(`${VOICE_ROUTE_BASE}/transcribe`, {
                method: "POST",
                body: form,
            });
            state.isVoiceBusy = false;
            if (!insertRecognizedText(data.text)) {
                setStatus(L.noSpeech, "info", STATUS_RESET_DELAY_MS);
                setProgress({ active: false });
            } else {
                setStatus(L.speechInserted, "info", STATUS_RESET_DELAY_MS);
                setProgress({ percent: 100, active: false });
            }
        } catch (error) {
            state.isVoiceBusy = false;
            setStatus(L.voiceError(error.message), "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
        }
        refreshRecordButton();
        refreshAiButton();
    }

    async function startRecording() {
        if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
            setStatus(L.micUnsupported, "error", STATUS_RESET_DELAY_MS);
            return;
        }
        rememberCursor();
        setStatus(L.openingMic);
        try {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
            });
            const mimeType = MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m)) || "";
            mediaRecorder = createAudioRecorder(mediaStream, mimeType);
            chunks = [];
            mediaRecorder.ondataavailable = (event) => {
                if (event.data?.size > 0) chunks.push(event.data);
            };
            mediaRecorder.onstop = async () => {
                if (mediaStream) {
                    mediaStream.getTracks().forEach((track) => track.stop());
                    mediaStream = null;
                }
                if (disposed) return;
                const blob = new Blob(chunks, { type: mimeType || "audio/webm" });
                if (blob.size <= 0) {
                    state.isVoiceBusy = false;
                    setStatus(L.noAudio, "info", STATUS_RESET_DELAY_MS);
                    setProgress({ active: false });
                    refreshRecordButton();
                    refreshAiButton();
                    return;
                }
                // If the model is still downloading (started in the
                // background by onRecordClick), block transcription until
                // it's ready. This is the cost the user pays for being
                // able to record on the very first click.
                if (!state.modelReady && state.modelReadyPromise) {
                    setStatus(L.waitingModel);
                    setProgress({ active: true, indeterminate: true });
                    refreshRecordButton();
                    try {
                        await state.modelReadyPromise;
                    } catch (error) {
                        state.isVoiceBusy = false;
                        setStatus(L.voiceError(error.message), "error", STATUS_RESET_DELAY_MS);
                        setProgress({ active: false, error: true });
                        refreshRecordButton();
                        refreshAiButton();
                        return;
                    }
                }
                if (!state.modelReady) {
                    state.isVoiceBusy = false;
                    setStatus(L.modelNotReady, "error", STATUS_RESET_DELAY_MS);
                    setProgress({ active: false, error: true });
                    refreshRecordButton();
                    refreshAiButton();
                    return;
                }
                await sendAudioToServer(blob);
            };
            mediaRecorder.start();
            state.isRecording = true;
            setStatus(L.recording);
            setProgress({ active: true, indeterminate: true });
            refreshRecordButton();
            refreshAiButton();
        } catch (error) {
            state.isRecording = false;
            state.isVoiceBusy = false;
            setStatus(L.micError(error.message), "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
            refreshRecordButton();
        }
    }

    function stopRecording() {
        if (!mediaRecorder || !state.isRecording) return;
        state.isRecording = false;
        state.isVoiceBusy = true;
        setStatus(L.preparingAudio);
        setProgress({ active: true, indeterminate: true });
        refreshRecordButton();
        try {
            mediaRecorder.stop();
        } catch (error) {
            state.isVoiceBusy = false;
            setStatus(L.voiceError(error.message), "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
            refreshRecordButton();
        }
    }

    function onRecordClick(event) {
        if (state.isAiBusy) return;
        syncActiveVoiceModel();
        if (state.isRecording) {
            stopRecording();
            return;
        }
        if (state.isVoiceBusy) return;
        // Trigger model download in the background if it isn't ready yet,
        // but DO NOT wait for it — recording must start on the very first
        // click. The transcribe step in mediaRecorder.onstop awaits
        // state.modelReadyPromise before posting the audio, so a long
        // download just delays transcription instead of blocking capture.
        if (!state.modelReady && !state.isModelLoading) {
            downloadVoiceModel(Boolean(event?.shiftKey));
        }
        startRecording();
    }

    // -----------------------------------------------------------------
    // AI enhance
    // -----------------------------------------------------------------
    function aiHqEnabled() {
        return Boolean(getWidgetValue(node, BIGGER_MODEL_WIDGET, false));
    }

    function refreshAiHqToggle() {
        aiHqToggle.classList.toggle("is-on", aiHqEnabled());
        aiHqToggle.setAttribute("aria-pressed", aiHqEnabled() ? "true" : "false");
    }

    aiHqToggle.addEventListener("click", () => {
        const next = !aiHqEnabled();
        // Значение живёт в скрытом виджете: так оно уезжает в сохранённый
        // workflow и доезжает до `execute`, а зеркало в properties страхует
        // Vue-режим (§12.5.13).
        const widget = getWidget(node, BIGGER_MODEL_WIDGET);
        if (widget) {
            widget.value = next;
            widget.callback?.(next);
        }
        node.properties ||= {};
        node.properties[BIGGER_MODEL_WIDGET] = next;
        refreshAiHqToggle();
    });

    function buildAiPayload(frames) {
        const list = (frames && frames.length)
            ? frames
            : state.attachedImages.filter(Boolean);
        return {
            text: String(textarea.value || ""),
            system_preset: String(presetSelect.value || "Prompts enhance"),
            attached_images: list,
            // Kept for a server that has not been restarted since the update.
            attached_image: String(list[0] || ""),
            attached_image_2: String(list[1] || ""),
            // Галочка «крупнее модель» — такое же значение виджета, как пресет.
            // Сервер, не знающий о ней, поле просто игнорирует.
            bigger_model: Boolean(getWidgetValue(node, BIGGER_MODEL_WIDGET, false)),
            operation_id: state.activeAiOperationId,
        };
    }

    async function enhancePrompt() {
        if (state.isVoiceBusy || state.isAiBusy || state.isRecording || socketBusy) return;
        syncTextFromUi();
        syncPresetFromUi();

        // A wired input wins, exactly as it does on a real run.
        let frames = [];
        const wired = socketIsWired();
        if (wired) {
            setStatus(L.readingInput);
            frames = await framesFromSocket();
            if (frames.length) {
                setStatus(L.usingInputFrames(frames.length), "info", STATUS_RESET_DELAY_MS);
            } else {
                // Say it and stop. Enhancing from the text alone while the
                // person is looking at a connected picture would hand back a
                // prompt about nothing they can see — and it would look like
                // the button worked.
                setStatus(L.inputNotReadable, "error", STATUS_RESET_DELAY_MS);
                return;
            }
        }

        const payload = buildAiPayload(frames);
        if (!payload.text.trim() && !payload.attached_images.length) {
            setStatus(L.noPromptOrImage, "info", STATUS_RESET_DELAY_MS);
            return;
        }
        state.isAiBusy = true;
        state.activeAiOperationId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
        showBusy(true);
        payload.operation_id = state.activeAiOperationId;
        setStatus(L.startingAi);
        setProgress({ active: true, indeterminate: true });
        refreshAiButton();
        refreshRecordButton();
        try {
            const data = await fetchJson(`${AI_ROUTE_BASE}/enhance`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!data.ok) throw new Error(data.error || L.enhanceFailed);
            if (replaceText(data.text)) {
                const ref = data.used_image ? L.imageSuffix : "";
                setStatus(`${L.aiReady}${ref}`, "info", STATUS_RESET_DELAY_MS);
            } else {
                setStatus(L.emptyAiResult, "info", STATUS_RESET_DELAY_MS);
            }
            setProgress({ percent: 100, active: false });
        } catch (error) {
            setStatus(L.aiError(error.message), "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
        } finally {
            state.isAiBusy = false;
            showBusy(false);
            if (!disposed) {
                refreshAiButton();
                refreshRecordButton();
            }
        }
    }

    // -----------------------------------------------------------------
    // Attach (file picker / drag-drop / paste)
    // -----------------------------------------------------------------
    async function uploadImageFile(file, slot = null) {
        if (!file) return "";
        if (!file.type.startsWith("image/") && !fileExtensionOk(file.name)) {
            setStatus(L.notImageFile, "error", STATUS_RESET_DELAY_MS);
            return "";
        }
        const target = slot === null ? nextFreeSlot() : slot;
        if (target < 0) {
            setStatus(L.bothSlotsFull, "info", STATUS_RESET_DELAY_MS);
            return "";
        }
        setStatus(L.uploadingImage);
        setProgress({ active: true, indeterminate: true });
        try {
            const form = new FormData();
            form.append("image", file, file.name);
            const response = await api.fetchApi(UPLOAD_ROUTE, { method: "POST", body: form });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload?.error || payload?.message || `HTTP ${response.status}`);
            }
            const annotated = buildAnnotatedPath(payload);
            if (!annotated) throw new Error("Upload returned no filename");
            setAttachedImage(annotated, target);
            setStatus(L.imageAttached, "info", STATUS_RESET_DELAY_MS);
            setProgress({ percent: 100, active: false });
            return annotated;
        } catch (error) {
            setStatus(L.uploadError(error.message), "error", STATUS_RESET_DELAY_MS);
            setProgress({ active: false, error: true });
            return "";
        }
    }

    function pointerOverContainer(event) {
        const path = typeof event.composedPath === "function" ? event.composedPath() : [];
        return path.includes(container);
    }

    /**
     * Put the two images the other way round.
     *
     * Which is the first frame and which is the last is the whole meaning of
     * the pair, and getting it backwards is easy — so fixing it must not mean
     * removing both and attaching them again in the other order.
     */
    function swapSlots() {
        const [first, second] = state.attachedImages;
        if (!first || !second) return;
        state.attachedImages = [second, first];
        setWidgetValue(node, IMAGE_SLOTS[0], second);
        setWidgetValue(node, IMAGE_SLOTS[1], first);
        renderAttached();
        setDirty(node);
        setStatus(L.framesSwapped, "info", STATUS_RESET_DELAY_MS);
    }

    // Dragging one thumbnail onto the other swaps them. The payload is a MIME
    // of our own, which is also what keeps the container's upload drop zone
    // out of it: that zone only reacts to sources it knows, and this is not
    // one of them.
    const SLOT_DRAG_MIME = "application/x-ts-super-prompt-slot";
    for (const { button, slot } of attachSlots) {
        button.addEventListener("dragstart", (event) => {
            if (!button.draggable) return;
            event.stopPropagation();
            event.dataTransfer?.setData(SLOT_DRAG_MIME, String(slot));
            if (event.dataTransfer) event.dataTransfer.effectAllowed = "move";
            button.classList.add("is-dragging");
        });
        button.addEventListener("dragend", () => {
            button.classList.remove("is-dragging");
            for (const other of attachSlots) other.button.classList.remove("is-drop-target");
        });
        const carriesSlot = (event) =>
            [...(event.dataTransfer?.types || [])].includes(SLOT_DRAG_MIME);
        button.addEventListener("dragover", (event) => {
            if (!carriesSlot(event)) return;
            event.preventDefault();
            event.stopPropagation();
            if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
            button.classList.add("is-drop-target");
        });
        button.addEventListener("dragleave", () => button.classList.remove("is-drop-target"));
        button.addEventListener("drop", (event) => {
            if (!carriesSlot(event)) return;
            event.preventDefault();
            event.stopPropagation();
            button.classList.remove("is-drop-target");
            const from = Number(event.dataTransfer?.getData(SLOT_DRAG_MIME));
            if (Number.isFinite(from) && from !== slot) swapSlots();
        });
    }

    // -----------------------------------------------------------------
    // Картинки со входа: считаем только подключённую ветку
    // -----------------------------------------------------------------
    //
    // A wired `images` input holds no value until something computes it, and
    // the whole point of this button is a prompt without running the workflow.
    //
    // The first attempt read what was knowable for free — previews the upstream
    // node kept from the last run, or the filename sitting in a loader widget.
    // That was wrong. `node.imgs` exists only on preview and save nodes, so
    // anything assembled on the way — two loaders joined into a batch, a
    // resize, a crop — offered nothing at all, which is precisely what people
    // wire up. The button answered "no image" while an image was plainly
    // connected.
    //
    // So ask the server for the picture, but ask it for ONLY the branch that
    // feeds this input. `graphToPrompt` hands over the graph in API form with
    // the virtual nodes — reroutes, bypasses — already resolved; walking back
    // from our own input keeps just the nodes that branch needs, and a
    // PreviewImage pinned to its end is what gives the run something to
    // produce. Nothing else is in the prompt, so nothing else runs: no
    // sampler, no save, no side effects on the rest of the workflow.
    //
    // NOT cached here. The first version kept the frames against the pruned
    // prompt as a signature, on the theory that an unchanged prompt means
    // unchanged pictures. It does not: swap the file behind a loader and the
    // graph reads the same, while the picture is a different one — and the
    // button then enhanced the OLD image without a word. ComfyUI already
    // caches this properly one level down, by what the nodes actually read
    // (core LoadImage hashes the file's contents), so a fresh queue costs
    // nothing when nothing changed and is correct when something did. A cache
    // that is right most of the time is worse than no cache at all when being
    // wrong is silent.
    const IMAGES_INPUT = "images";
    const MAX_SOCKET_FRAMES = 4;
    const SOCKET_SINK_CLASS = "PreviewImage";
    const SOCKET_POLL_MS = 400;
    const SOCKET_TIMEOUT_MS = 300_000;

    let socketBusy = false;

    /** True when something is wired into `images`, whatever it may hold. */
    function socketIsWired() {
        const input = (node.inputs || []).find((slot) => slot?.name === IMAGES_INPUT);
        return Boolean(input && input.link !== null && input.link !== undefined);
    }

    /**
     * The smallest prompt that produces what is wired into `images`.
     *
     * Null when nothing is connected or the branch cannot be traced.
     */
    async function socketRunPlan() {
        let graph;
        try {
            graph = await app.graphToPrompt();
        } catch (error) {
            console.warn("[TS SuperPrompt] could not serialise the graph", error);
            return null;
        }
        const output = graph?.output || {};
        const source = output[String(node.id)]?.inputs?.[IMAGES_INPUT];
        if (!Array.isArray(source) || source.length < 2) return null;

        // Everything that branch needs, and nothing else.
        const needed = new Set();
        const pending = [String(source[0])];
        while (pending.length) {
            const id = pending.shift();
            if (!id || needed.has(id) || !output[id]) continue;
            needed.add(id);
            for (const value of Object.values(output[id].inputs || {})) {
                if (Array.isArray(value) && value.length === 2) pending.push(String(value[0]));
            }
        }
        if (!needed.size) return null;

        const prompt = {};
        for (const id of needed) prompt[id] = output[id];
        // Without an output node there is nothing for ComfyUI to execute for.
        const sinkId = `ts_sp_${node.id}`;
        prompt[sinkId] = {
            class_type: SOCKET_SINK_CLASS,
            inputs: { images: [String(source[0]), source[1]] },
            _meta: { title: "TS Super Prompt input" },
        };
        return { prompt, sinkId };
    }

    async function historyEntry(promptId) {
        try {
            const response = await api.fetchApi(`/history/${encodeURIComponent(promptId)}`);
            if (!response.ok) return null;
            const data = await response.json();
            return data?.[promptId] || null;
        } catch (error) {
            return null;
        }
    }

    /**
     * Queue the branch and answer with the frames it produced.
     *
     * The finished run is read from history rather than from the websocket:
     * history holds the outputs whether the branch really executed or came
     * straight back from ComfyUI's own cache, and a cached branch emits no
     * `executed` event at all.
     */
    async function runInputBranch(plan) {
        let promptId = "";
        try {
            const response = await api.fetchApi("/prompt", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    client_id: api.clientId || api.initialClientId || undefined,
                    prompt: plan.prompt,
                }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(payload?.error?.message || `HTTP ${response.status}`);
            promptId = String(payload?.prompt_id || "");
            if (!promptId) throw new Error("the server returned no prompt id");
        } catch (error) {
            console.warn("[TS SuperPrompt] could not queue the connected branch", error);
            return [];
        }
        // Polling, not events: the run may sit behind whatever else is queued,
        // and history is the one place that tells the truth in every case.
        const deadline = Date.now() + SOCKET_TIMEOUT_MS;
        while (Date.now() < deadline) {
            const entry = await historyEntry(promptId);
            if (entry) {
                if (String(entry.status?.status_str || "") === "error") {
                    console.warn("[TS SuperPrompt] the connected branch failed to run");
                    return [];
                }
                return (entry.outputs?.[plan.sinkId]?.images || [])
                    .slice(0, MAX_SOCKET_FRAMES)
                    .map((image) => buildAnnotatedPath({
                        name: image?.filename,
                        subfolder: image?.subfolder,
                        type: image?.type || "temp",
                    }))
                    .filter(Boolean);
            }
            await new Promise((done) => setTimeout(done, SOCKET_POLL_MS));
        }
        console.warn("[TS SuperPrompt] the connected branch did not finish in time");
        return [];
    }

    /** Frames for the wired input, as annotated paths the backend can load. */
    async function framesFromSocket() {
        const plan = await socketRunPlan();
        if (!plan) return [];
        socketBusy = true;
        try {
            return await runInputBranch(plan);
        } finally {
            socketBusy = false;
        }
    }


    let pickingForSlot = 0;
    fileInput.addEventListener("change", async () => {
        const file = fileInput.files?.[0];
        fileInput.value = "";
        if (file) await uploadImageFile(file, pickingForSlot);
    });
    for (const { button, clear, slot } of attachSlots) {
        button.addEventListener("click", (event) => {
            if (event.target === clear) return;
            pickingForSlot = slot;
            fileInput.click();
        });
        clear.addEventListener("click", (event) => {
            event.stopPropagation();
            setAttachedImage("", slot);
            setStatus(L.imageRemoved, "info", STATUS_RESET_DELAY_MS);
        });
    }

    // Anything the shared service can turn into an image is accepted: a card
    // dragged out of the Artius browser, a file from the desktop, a ComfyUI
    // preview. It only checks that this used to be limited to OS files, which
    // is why dragging from Artius did nothing at all.
    const teardownDropZone = makeDropZone(container, {
        max: IMAGE_SLOTS.length,
        onDrop: async (items) => {
            for (const item of items) {
                const slot = nextFreeSlot();
                if (slot < 0) {
                    setStatus(L.bothSlotsFull, "info", STATUS_RESET_DELAY_MS);
                    return;
                }
                let blob;
                try {
                    blob = await item.getBlob();
                } catch (error) {
                    setStatus(L.uploadError(error.message), "error", STATUS_RESET_DELAY_MS);
                    return;
                }
                // A Blob is not a File, and the upload wants a name; the drop
                // item carries the one the source knew it by.
                const file = new File([blob], item.name || "dropped.png",
                    { type: blob.type || "image/png" });
                // Sequential on purpose: two images racing to call
                // nextFreeSlot() would both read the same free slot.
                await uploadImageFile(file, slot);
            }
        },
    });

    async function onDocumentPaste(event) {
        if (disposed) return;
        if (!pointerOverContainer(event)) return;
        const items = event.clipboardData?.items || [];
        for (const item of items) {
            if (item.kind === "file" && item.type.startsWith("image/")) {
                const file = item.getAsFile();
                if (file) {
                    event.preventDefault();
                    await uploadImageFile(file);
                    return;
                }
            }
        }
    }
    doc.addEventListener("paste", onDocumentPaste);

    // -----------------------------------------------------------------
    // Toolbar event wiring
    // -----------------------------------------------------------------
    textarea.addEventListener("input", () => syncTextFromUi());
    textarea.addEventListener("blur", () => syncTextFromUi());
    hqToggle.addEventListener("click", () => {
        // Switching models mid-recording/transcription resets modelReady and
        // desyncs the model the recording will be transcribed with.
        if (state.isRecording || state.isVoiceBusy) {
            setStatus(L.finishRecordingHq, "info", STATUS_RESET_DELAY_MS);
            return;
        }
        hqToggle.classList.toggle("is-on");
        syncHighQualityFromUi();
        syncActiveVoiceModel();
        refreshStatus();
    });
    presetSelect.addEventListener("change", () => syncPresetFromUi());
    recordBtn.addEventListener("click", onRecordClick);
    aiBtn.addEventListener("click", () => enhancePrompt());

    // -----------------------------------------------------------------
    // Mount DOM widget
    // -----------------------------------------------------------------
    // Mount through the shared resizer so sizing is correct in BOTH renderers
    // (computeSize for Nodes 1.0, getMinHeight/getMaxHeight for Nodes 2.0) and
    // the node stays resizable — the same plumbing every other TS GUI node uses.
    // The node has no visible input rows (all four widgets are hidden), so the
    // chrome above the widget is just the title bar.
    const { domWidget } = addResizableDomWidget(node, container, {
        name: DOM_WIDGET_NAME,
        minWidth: 260,
        minHeight: 150,
        defaultWidth: 400,
        defaultHeight: 230,
        chromeHeight: 34,
        minWidgetHeight: 110,
    });
    domWidget.__tsSuperPromptUi = true;

    // -----------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------
    function cleanup() {
        if (disposed) return;
        disposed = true;
        window.clearTimeout(statusResetTimer);
        window.clearTimeout(progressClearTimer);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.progress`, onVoiceProgress);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.status`, onVoiceStatus);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.done`, onVoiceDone);
        api.removeEventListener(`${VOICE_EVENT_PREFIX}.error`, onVoiceError);
        api.removeEventListener(`${AI_EVENT_PREFIX}.progress`, onAiProgress);
        api.removeEventListener(`${AI_EVENT_PREFIX}.done`, onAiDone);
        api.removeEventListener(`${AI_EVENT_PREFIX}.error`, onAiError);
        doc.removeEventListener("paste", onDocumentPaste);
        if (mediaRecorder && state.isRecording) {
            try {
                mediaRecorder.stop();
            } catch {
                // recorder may already be stopped — safe to ignore.
            }
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach((track) => track.stop());
            mediaStream = null;
        }
        if (Array.isArray(node.widgets)) {
            const idx = node.widgets.indexOf(domWidget);
            if (idx >= 0) node.widgets.splice(idx, 1);
        }
        container.remove();
        // Detach onConfigure wrapper + sync-handle so a fresh setupSuperPrompt
        // can install its own without stacking wrappers.
        if (Object.prototype.hasOwnProperty.call(node, "_tsSuperPromptOriginalOnConfigure")) {
            node.onConfigure = node._tsSuperPromptOriginalOnConfigure;
            delete node._tsSuperPromptOriginalOnConfigure;
        }
        delete node._tsSuperPromptSyncUi;
    }

    node._tsSuperPromptCleanup = cleanup;
    node._tsSuperPromptSyncUi = syncUiFromWidgets;
    if (!node._tsSuperPromptOriginalOnRemoved) {
        node._tsSuperPromptOriginalOnRemoved = node.onRemoved;
    }
    node.onRemoved = function onRemovedWrapper() {
        cleanup();
        return node._tsSuperPromptOriginalOnRemoved?.apply(this, arguments);
    };

    // Pull-through after LiteGraph restores widget values. ``onConfigure``
    // fires for both ``LGraph.configure`` (workflow load + tab switch) and
    // ``LGraphNode.clone`` (copy / paste) — in both cases widget values
    // land *after* onNodeCreated, so we re-sync the DOM UI here.
    if (!Object.prototype.hasOwnProperty.call(node, "_tsSuperPromptOriginalOnConfigure")) {
        node._tsSuperPromptOriginalOnConfigure = node.onConfigure;
    }
    node.onConfigure = function onConfigureWrapper() {
        const result = node._tsSuperPromptOriginalOnConfigure?.apply(this, arguments);
        if (!disposed) syncUiFromWidgets();
        return result;
    };

    // Initial paint.
    renderAttached();
    refreshRecordButton();
    refreshAiButton();
    refreshStatus();
}

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedWrapper() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            if (!getWidget(this, DOM_WIDGET_NAME)) {
                setupSuperPrompt(this);
            }
            return result;
        };
    },
    loadedGraphNode(node) {
        if (![node?.type, node?.comfyClass].includes(NODE_NAME)) return;
        if (!getWidget(node, DOM_WIDGET_NAME)) {
            setupSuperPrompt(node);
            return;
        }
        // DOM widget already exists (e.g. node was created by onNodeCreated
        // and is now being configured by LGraph). Re-pull widget values so
        // the textarea / preset / HQ toggle reflect the restored workflow
        // state — onConfigure also catches this, but ComfyUI versions vary
        // on the order of events.
        if (typeof node._tsSuperPromptSyncUi === "function") {
            node._tsSuperPromptSyncUi();
        }
    },
});
