// TS Film Emulation — ONE choice, "the look": a thumbnail of it in the node and
// a gallery of every look (built-in stocks and the .cube files in the LUT
// folder) rendered on one reference portrait.
//
// ⚠️ Why the two combos are hidden. The schema has film_preset, whose
// "External LUT" entry is really a mode switch, and lut_choice, which is
// ignored unless that switch is set. Picking a LUT while a stock was selected
// silently did nothing. Now a card IS the choice: a stock card sets
// film_preset, a LUT card sets film_preset="External LUT" + lut_choice — the
// mode follows from what was picked, never set by hand.
//
// Likewise there were two strengths (film_strength for stocks, lut_strength
// for LUTs) and only one ever applies. One visible "strength" drives both; the
// inactive one is unused by the backend, so mirroring never changes a render.
//
// Nothing about the node's inputs changes: the same widgets carry the same
// values (hidden through the pack's hideWidget, which keeps saved positional
// widgets_values readable), so old workflows and API calls behave as before.
// The DOM widget is appended LAST and does not serialize.
//
// Compact layout (12.13): the node is the preview and nothing else. EVERY
// stock widget is hidden; the settings live in a drawer that slides over the
// preview from the gear button. A wire already plugged into one of those
// inputs keeps working — hideWidget leaves a connected socket in place and the
// prompt keeps the link.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import { TS_UI_CLASS, ensureThemeStyles, getUiLanguage, pickLocaleStrings } from "../../_theme.js";
import { addResizableDomWidget, getWidget, hideWidget } from "../../_dom_widget.js";
import { openFullscreenOverlay } from "../../_fullscreen.js";

const EXTENSION_ID = "ts.filmEmulation";
const NODE_NAME = "TS_Film_Emulation";
const DOM_WIDGET_NAME = "ts_film_look";
const STYLE_ID = "ts-film-emulation-styles";
const INPUT_PRESET = "film_preset";
const INPUT_LUT = "lut_choice";
const INPUT_FILM_STRENGTH = "film_strength";
const INPUT_LUT_STRENGTH = "lut_strength";
const EXTERNAL_LUT = "External LUT";
const NO_LUT = "None";
const ALL = "__all__";
const MIN_NODE_WIDTH = 220;
const THUMB_HEIGHT = 180;
const MIN_THUMB_HEIGHT = 120;

// The settings drawer. Order, defaults and ranges mirror the Python schema
// (tests/test_film_emulation_ui.py keeps them in step); the widget's own
// options win when present, so a schema change cannot be clamped here.
const SETTINGS = [
    { name: "film_strength", kind: "slider", def: 1.0, min: 0, max: 1, step: 0.01, group: "tone", preview: "strength" },
    { name: "contrast_curve", kind: "slider", def: 1.0, min: 0, max: 3, step: 0.01, group: "tone", preview: "contrast" },
    { name: "warmth", kind: "slider", def: 0.0, min: -1, max: 1, step: 0.01, group: "tone", preview: "warmth" },
    { name: "fade", kind: "slider", def: 0.0, min: 0, max: 0.5, step: 0.01, group: "tone", preview: "fade" },
    { name: "shadow_saturation", kind: "slider", def: 1.0, min: 0, max: 2, step: 0.01, group: "colour", preview: "shadow_saturation" },
    { name: "highlight_saturation", kind: "slider", def: 1.0, min: 0, max: 2, step: 0.01, group: "colour", preview: "highlight_saturation" },
    { name: "grain_intensity", kind: "slider", def: 0.02, min: 0, max: 0.5, step: 0.01, group: "grain" },
    { name: "grain_size", kind: "slider", def: 0.5, min: 0.5, max: 5, step: 0.1, group: "grain" },
    { name: "grain_speed", kind: "slider", def: 1.0, min: 0.05, max: 1, step: 0.05, group: "grain" },
    { name: "grain_seed", kind: "int", def: 0, min: 0, max: 0xFFFFFFFF, group: "grain" },
];
// Hidden but not in the drawer: the look picks these (lut_strength mirrors strength).
const PICKED_BY_LOOK = [INPUT_PRESET, INPUT_LUT, INPUT_LUT_STRENGTH];
// Hidden and left alone: ComfyUI's Bypass replaces `enable` (a graph saved with
// it off shows an "Off" badge that turns it back on), and the backend ignores
// `gamma_correction` since 12.13.
const INPUT_ENABLE = "enable";
const RETIRED = [INPUT_ENABLE, "gamma_correction"];

const GEAR_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19.4 13a7.6 7.6 0 0 0 0-2l2.1-1.6-2-3.5-2.5 1a7.4 7.4 0 0 0-1.7-1L15 3.3h-4l-.4 2.6a7.4 7.4 0 0 0-1.7 1l-2.5-1-2 3.5L6.5 11a7.6 7.6 0 0 0 0 2l-2.1 1.6 2 3.5 2.5-1a7.4 7.4 0 0 0 1.7 1l.4 2.6h4l.4-2.6a7.4 7.4 0 0 0 1.7-1l2.5 1 2-3.5zM13 15.5a3.5 3.5 0 1 1 0-7 3.5 3.5 0 0 1 0 7z" transform="translate(-1 0)"/></svg>';
const CLOSE_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18.3 5.7 12 12l6.3 6.3-1.4 1.4L10.6 13.4 4.3 19.7 2.9 18.3 9.2 12 2.9 5.7 4.3 4.3l6.3 6.3 6.3-6.3z" transform="translate(1 0)"/></svg>';

const STRINGS = {
    en: {
        galleryTitle: "Film looks",
        description: "Browse every film look on a reference portrait",
        search: "Search looks...",
        all: "All",
        original: "Original",
        originalFamily: "No look",
        lutFamily: "LUT file",
        empty: "Nothing matches the search.",
        loadFailed: "Could not load the looks. Is the server running?",
        current: "Selected",
        thumbTitle: "Click to choose a look",
        settings: "Settings",
        close: "Close settings",
        reset: "Reset",
        resetTip: "Back to the defaults. Double-click a slider to reset just that one.",
        off: "Off — click to turn on",
        groupTone: "Tone",
        groupColour: "Colour",
        groupGrain: "Grain",
        fields: {
            film_strength: "Look strength",
            contrast_curve: "Contrast",
            warmth: "Warmth",
            fade: "Fade",
            shadow_saturation: "Shadow saturation",
            highlight_saturation: "Highlight saturation",
            grain_intensity: "Amount",
            grain_size: "Size",
            grain_speed: "Speed (video)",
            grain_seed: "Seed",
        },
        tips: {
            film_strength: "How much of the chosen look: 0 = the original, 1 = the full look.",
            contrast_curve: "S-curve around mid-grey; black and white are never clipped.",
            warmth: "White balance: right warms, left cools. Black stays black.",
            fade: "Matte look: lifts the blacks, whites stay white.",
            shadow_saturation: "Colour in the dark tones. 1 = unchanged.",
            highlight_saturation: "Colour in the bright tones. 1 = unchanged.",
            grain_intensity: "Film grain. 0 = none. Not shown on the preview.",
            grain_size: "Grain particle size: higher is coarser.",
            grain_speed: "Clips only: 1 = a new grain pattern every frame, 0.5 = every two frames.",
            grain_seed: "The same seed gives the same grain on a re-render.",
        },
    },
    ru: {
        galleryTitle: "Плёночные образы",
        description: "Все плёночные образы на эталонном портрете",
        search: "Поиск образа...",
        all: "Все",
        original: "Оригинал",
        originalFamily: "Без обработки",
        lutFamily: "LUT-файл",
        empty: "Ничего не найдено.",
        loadFailed: "Не удалось загрузить образы. Сервер запущен?",
        current: "Выбрано",
        thumbTitle: "Щёлкните, чтобы выбрать образ",
        settings: "Настройки",
        close: "Закрыть настройки",
        reset: "Сброс",
        resetTip: "Вернуть значения по умолчанию. Двойной щелчок по слайдеру сбрасывает только его.",
        off: "Выключено — щёлкните, чтобы включить",
        groupTone: "Тон",
        groupColour: "Цвет",
        groupGrain: "Зерно",
        fields: {
            film_strength: "Сила образа",
            contrast_curve: "Контраст",
            warmth: "Теплота",
            fade: "Выцветание",
            shadow_saturation: "Насыщенность теней",
            highlight_saturation: "Насыщенность светов",
            grain_intensity: "Количество",
            grain_size: "Размер",
            grain_speed: "Скорость (видео)",
            grain_seed: "Seed",
        },
        tips: {
            film_strength: "Сколько выбранного образа: 0 — оригинал, 1 — образ полностью.",
            contrast_curve: "S-кривая вокруг среднего серого; чёрное и белое не обрезаются.",
            warmth: "Баланс белого: вправо теплее, влево холоднее. Чёрное остаётся чёрным.",
            fade: "Матовый вид: приподнимает чёрное, белое остаётся белым.",
            shadow_saturation: "Цвет в тёмных тонах. 1 — без изменений.",
            highlight_saturation: "Цвет в светлых тонах. 1 — без изменений.",
            grain_intensity: "Плёночное зерно. 0 — без зерна. На превью не показывается.",
            grain_size: "Размер зёрен: больше — крупнее.",
            grain_speed: "Только для видео: 1 — новый узор зерна каждый кадр, 0.5 — раз в два кадра.",
            grain_seed: "Один и тот же seed даёт то же зерно при повторном рендере.",
        },
    },
};

function ensureStyles() {
    // Colours come from the shared --ts-* tokens (js/_theme.js); this sheet is layout.
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
/* Nodes 2.0 hands a DOM widget the height of its min-content; the thumbnail
   may shrink, so without this floor the node drew it as a 45 px sliver. */
.ts-film {
    position: relative; display: flex; flex-direction: column; padding: 4px;
    box-sizing: border-box; height: 100%; min-height: ${MIN_THUMB_HEIGHT}px; overflow: hidden;
    color: var(--ts-text); font-family: var(--ts-font); pointer-events: auto;
}
/* The preview IS the node: it takes all the height the node has. */
.ts-film__thumb {
    position: relative; flex: 1 1 0; min-height: 60px;
    border-radius: var(--ts-radius); overflow: hidden; cursor: pointer;
    background: var(--ts-sunken); border: 1px solid var(--ts-border-soft);
}
.ts-film__thumb:hover { border-color: var(--ts-accent-line); }
.ts-film__thumb img {
    position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: cover; display: block; transition: opacity .15s ease, filter .15s ease;
}
.ts-film.is-off .ts-film__thumb img { opacity: .45; filter: grayscale(1); }
/* Caption and buttons sit ON the preview, so they must read on any picture:
   the scrim and its white text are fixed colours on purpose (theme exception
   for chrome over user content, see doc/THEME.md). */
.ts-film__caption {
    position: absolute; left: 0; right: 0; bottom: 0; padding: 18px 8px 6px;
    display: flex; flex-direction: column; pointer-events: none;
    background: linear-gradient(to top, rgba(0,0,0,.72), rgba(0,0,0,0));
    color: #fff;
}
.ts-film__name {
    font-size: var(--ts-fs-sm); font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,.6);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ts-film__family {
    font-size: var(--ts-fs-xs); opacity: .8;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ts-film__gear { position: absolute; top: 6px; right: 6px; z-index: 2; }
.ts-film__gear.ts-ui-btn { background: rgba(0,0,0,.45); border-color: rgba(255,255,255,.18); color: #fff; }
.ts-film__gear.ts-ui-btn:hover:not([disabled]) { background: rgba(0,0,0,.65); border-color: rgba(255,255,255,.35); }
/* A dot on the gear: something in the drawer differs from the defaults. */
.ts-film__gear::after {
    content: ""; position: absolute; top: 3px; right: 3px; width: 6px; height: 6px;
    border-radius: 999px; background: var(--ts-accent); display: none;
}
.ts-film__gear.is-modified::after { display: block; }
.ts-film__off {
    position: absolute; top: 8px; left: 8px; right: 44px; padding: 3px 9px; border-radius: 999px;
    border: none; cursor: pointer; text-align: left; font: inherit;
    font-size: var(--ts-fs-xs); font-weight: 700; background: var(--ts-danger); color: var(--ts-accent-contrast);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; width: fit-content; max-width: calc(100% - 52px);
}
.ts-film__off[hidden] { display: none; }

/* Settings drawer: slides over the preview, scrolls inside, never resizes the node. */
.ts-film__panel {
    position: absolute; inset: 4px; z-index: 3; display: flex; flex-direction: column;
    background: var(--ts-elevated); border: 1px solid var(--ts-border-soft);
    border-radius: var(--ts-radius); box-shadow: var(--ts-shadow);
    transform: translateY(8px); opacity: 0; visibility: hidden;
    transition: transform .14s ease, opacity .14s ease, visibility 0s linear .14s;
}
.ts-film.is-open .ts-film__panel {
    transform: none; opacity: 1; visibility: visible;
    transition: transform .14s ease, opacity .14s ease, visibility 0s;
}
.ts-film__panel-head {
    display: flex; align-items: center; gap: 6px; padding: 6px 6px 6px 10px; flex: 0 0 auto;
    border-bottom: 1px solid var(--ts-border-soft);
}
.ts-film__panel-title { flex: 1 1 auto; font-weight: 600; font-size: var(--ts-fs-sm); }
.ts-film__panel-head .ts-ui-btn:not(.ts-ui-btn--icon) { padding: 4px 9px; }
.ts-film__panel-body {
    flex: 1 1 auto; min-height: 0; overflow-y: auto; overscroll-behavior: contain;
    padding: 6px 10px 10px; display: flex; flex-direction: column; gap: 9px;
}
.ts-film__group { padding-top: 4px; }
.ts-film__group + .ts-film__field { margin-top: -3px; }
.ts-film__field { display: flex; flex-direction: column; gap: 5px; }
.ts-film__field-row { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
.ts-film__field-name { font-size: var(--ts-fs-sm); color: var(--ts-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ts-film__field-value {
    font-variant-numeric: tabular-nums; font-size: var(--ts-fs-sm); font-weight: 600;
    color: var(--ts-text); min-width: 3.5em; text-align: right;
}
.ts-film__field.is-changed .ts-film__field-name { color: var(--ts-text); }
.ts-film__field.is-changed .ts-film__field-value { color: var(--ts-accent); }
/* Filled track: --fill is the thumb position (0-100%), --origin where the fill
   starts (the 0 of a bipolar control such as warmth). */
.ts-film__slider.ts-ui-slider {
    height: 5px; margin: 4px 0;
    background: linear-gradient(to right,
        var(--ts-border-soft) 0 min(var(--origin), var(--fill)),
        var(--ts-accent) min(var(--origin), var(--fill)) max(var(--origin), var(--fill)),
        var(--ts-border-soft) max(var(--origin), var(--fill)) 100%);
}
.ts-film__slider.ts-ui-slider::-webkit-slider-thumb { width: 14px; height: 14px; transition: transform .1s ease; }
.ts-film__slider.ts-ui-slider:hover::-webkit-slider-thumb { transform: scale(1.15); }
.ts-film__seed.ts-ui-input { width: 9em; text-align: right; font-variant-numeric: tabular-nums; padding: 3px 6px; }
.ts-film-gallery {
    display: flex; flex-direction: column; gap: 10px; width: 100%; height: 100%;
    box-sizing: border-box; padding: 12px 16px 16px; min-height: 0;
}
.ts-film-gallery__head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; flex: 0 0 auto; }
.ts-film-gallery__search {
    flex: 0 1 280px; min-width: 160px; padding: 6px 10px; border-radius: var(--ts-radius-sm);
    border: 1px solid var(--ts-border); background: var(--ts-sunken); color: var(--ts-text);
    font: inherit; font-size: var(--ts-fs-sm); outline: none;
}
.ts-film-gallery__search:focus { border-color: var(--ts-accent-line); }
.ts-film-gallery__chips { display: flex; flex-wrap: wrap; gap: 6px; }
.ts-film-gallery__scroll { flex: 1 1 auto; min-height: 0; overflow-y: auto; padding-right: 4px; }
.ts-film-gallery__grid {
    display: grid; gap: 12px;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
}
.ts-film-card {
    display: flex; flex-direction: column; padding: 0; text-align: left; cursor: pointer;
    border-radius: var(--ts-radius); overflow: hidden; color: var(--ts-text);
    background: var(--ts-surface); border: 1px solid var(--ts-border-soft); font: inherit;
    transition: border-color .12s ease, transform .12s ease;
}
.ts-film-card:hover { border-color: var(--ts-accent-line); transform: translateY(-2px); }
.ts-film-card:focus-visible { outline: 2px solid var(--ts-accent); outline-offset: 1px; }
.ts-film-card.is-selected { border-color: var(--ts-accent); box-shadow: 0 0 0 1px var(--ts-accent); }
.ts-film-card[hidden] { display: none; }
.ts-film-card__img { position: relative; aspect-ratio: 4 / 3; background: var(--ts-sunken); }
.ts-film-card__img img { width: 100%; height: 100%; object-fit: cover; display: block; }
.ts-film-card__badge {
    position: absolute; top: 6px; left: 6px; padding: 2px 7px; border-radius: 999px;
    font-size: var(--ts-fs-xs); background: var(--ts-accent); color: var(--ts-accent-contrast);
}
.ts-film-card__badge[hidden] { display: none; }
.ts-film-card__meta { padding: 8px 10px 10px; display: flex; flex-direction: column; gap: 3px; }
.ts-film-card__name { font-size: var(--ts-fs-sm); font-weight: 600; }
.ts-film-card__family { font-size: var(--ts-fs-xs); color: var(--ts-accent); }
.ts-film-card__note { font-size: var(--ts-fs-xs); color: var(--ts-muted); line-height: 1.35; }
.ts-film-gallery__empty { padding: 32px; text-align: center; color: var(--ts-muted); }
.ts-film-gallery__empty[hidden] { display: none; }
`;
    document.head.appendChild(style);
}

// ── catalogue ───────────────────────────────────────────────────────────── //

let catalogueRequest = null;

/** The looks, fetched once and shared; `fresh` re-reads the LUT folder. */
function loadCatalogue(fresh = false) {
    if (fresh || !catalogueRequest) {
        catalogueRequest = api.fetchApi("/ts_film/looks")
            .then((response) => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return response.json();
            })
            .catch((error) => {
                catalogueRequest = null;
                throw error;
            });
    }
    return catalogueRequest;
}

function previewUrl(preset, lut, size = "small", adjust = null) {
    const query = new URLSearchParams({ preset, lut: lut || NO_LUT, size });
    for (const [name, value] of Object.entries(adjust || {})) query.set(name, String(value));
    return api.apiURL(`/ts_film/preview?${query.toString()}`);
}

/** Which card the two widgets currently describe. */
function lookIdOf(preset, lut) {
    if (preset === EXTERNAL_LUT) return !lut || lut === NO_LUT ? "original" : `lut:${lut}`;
    return `stock:${preset}`;
}

function familyLabel(look, families, strings, lang) {
    if (look.kind === "original") return strings.originalFamily;
    if (look.kind === "lut") return strings.lutFamily;
    const family = families.find((item) => item.id === look.family);
    return family ? (family[lang] || family.en) : "";
}

function stop(element, events) {
    for (const name of events) element.addEventListener(name, (event) => event.stopPropagation());
}

// ── node ────────────────────────────────────────────────────────────────── //

function setupFilmNode(node) {
    if (node._tsFilmReady || typeof node.addDOMWidget !== "function") return;
    node._tsFilmReady = true;
    ensureStyles();
    const strings = pickLocaleStrings(STRINGS);
    const lang = getUiLanguage() === "ru" ? "ru" : "en";

    const presetWidget = getWidget(node, INPUT_PRESET);
    const lutWidget = getWidget(node, INPUT_LUT);
    const strengthWidget = getWidget(node, INPUT_FILM_STRENGTH);
    const lutStrengthWidget = getWidget(node, INPUT_LUT_STRENGTH);
    if (!presetWidget || !lutWidget || !strengthWidget || !lutStrengthWidget) {
        console.warn("[TS Film Emulation] expected widgets not found; gallery disabled");
        return;
    }
    // Every stock widget goes: the look replaces the mode switch and the LUT
    // list, the drawer carries the rest. Widgets are looked up BEFORE hiding —
    // getWidget keeps finding them in the stash afterwards.
    const settingWidgets = new Map();
    for (const spec of SETTINGS) {
        const widget = getWidget(node, spec.name);
        if (widget) settingWidgets.set(spec.name, widget);
    }
    const enableWidget = getWidget(node, INPUT_ENABLE);
    for (const name of [...PICKED_BY_LOOK, ...RETIRED, ...settingWidgets.keys()]) hideWidget(node, name);

    const isLutLook = () => String(presetWidget.value) === EXTERNAL_LUT;

    const writeValue = (widget, value) => {
        widget.value = value;
        // Hidden widgets live on node.properties between save and load.
        if (node._tsHiddenWidgets?.[widget.name]) {
            node.properties ||= {};
            node.properties[widget.name] = value;
        }
    };
    // One knob for both strengths: the one that is not in use is ignored by
    // the backend, so keeping them equal never changes what a graph renders.
    const mirrorStrength = () => {
        if (Number(lutStrengthWidget.value) !== Number(strengthWidget.value)) {
            writeValue(lutStrengthWidget, strengthWidget.value);
        }
    };
    const originalStrengthCallback = strengthWidget.callback;
    strengthWidget.callback = function tsFilmStrength(...args) {
        const result = originalStrengthCallback?.apply(this, args);
        mirrorStrength();
        return result;
    };
    // Belt and braces: whatever path moved the slider, the queued prompt agrees.
    strengthWidget.beforeQueued = mirrorStrength;
    // A graph saved before 12.13 may hold different values; show the one that
    // actually applies to the selected look (the other was never used).
    const showActiveStrength = () => {
        if (isLutLook() && Number(strengthWidget.value) !== Number(lutStrengthWidget.value)) {
            writeValue(strengthWidget, lutStrengthWidget.value);
        }
    };

    // A first guess at what the stock widgets take, for the initial size only.
    // The live layout reads the DOM widget's measured top instead (below): a
    // fixed guess measured 24 px short (the caption row was clipped) and, once
    // padded, 40 px long (an empty band under the UI).
    const chrome = Math.ceil(node.computeSize?.()[1] || 420);
    let lookWidget = null;
    const liveChrome = () => {
        const top = Number(lookWidget?.y);
        return Number.isFinite(top) && top > 0 ? top + 6 : chrome;
    };

    const container = document.createElement("div");
    container.className = `${TS_UI_CLASS} ts-film`;

    const thumb = document.createElement("div");
    thumb.className = "ts-film__thumb";
    thumb.title = strings.thumbTitle;
    const thumbImg = document.createElement("img");
    thumbImg.alt = "";
    thumbImg.decoding = "async";
    const caption = document.createElement("div");
    caption.className = "ts-film__caption";
    const nameEl = document.createElement("span");
    nameEl.className = "ts-film__name";
    const familyEl = document.createElement("span");
    familyEl.className = "ts-film__family";
    caption.append(nameEl, familyEl);
    // Only a graph saved with `enable` off ever shows this; one click undoes it
    // (the switch itself is gone — Bypass does that job).
    const offBadge = document.createElement("button");
    offBadge.type = "button";
    offBadge.className = "ts-film__off";
    offBadge.textContent = strings.off;
    offBadge.hidden = true;
    offBadge.addEventListener("click", (event) => {
        event.stopPropagation();
        if (enableWidget) setWidget(enableWidget, true);
        settings.sync();
        node.setDirtyCanvas?.(true, true);
    });
    thumb.append(thumbImg, caption, offBadge);

    const gear = document.createElement("button");
    gear.type = "button";
    gear.className = "ts-ui-btn ts-ui-btn--icon ts-film__gear";
    gear.title = strings.settings;
    gear.setAttribute("aria-label", strings.settings);
    gear.innerHTML = GEAR_SVG;

    const settings = buildSettingsPanel();
    container.append(thumb, gear, settings.panel);

    thumb.addEventListener("click", (event) => {
        event.stopPropagation();
        openGallery();
    });
    gear.addEventListener("click", (event) => {
        event.stopPropagation();
        setOpen(!container.classList.contains("is-open"));
    });
    // keydown too: typing a seed must not reach the canvas shortcuts.
    stop(container, ["pointerdown", "pointerup", "mousedown", "mouseup", "wheel", "dblclick", "contextmenu", "keydown"]);

    function setOpen(open) {
        container.classList.toggle("is-open", open);
        gear.classList.toggle("is-active", open);
        if (open) {
            settings.sync();
            settings.focus();
        }
    }

    // ⚠️ No padding on this floor: the guess is sometimes exact (322 on a graph
    // load) and sometimes 70 short (252 on a fresh node), and a padded floor
    // grew every loaded node by 70. The layout itself never gives the widget
    // less than minWidgetHeight, and the thumbnail is what shrinks first.
    lookWidget = addResizableDomWidget(node, container, {
        name: DOM_WIDGET_NAME,
        minWidth: MIN_NODE_WIDTH,
        minHeight: chrome + MIN_THUMB_HEIGHT,
        defaultWidth: Math.max(MIN_NODE_WIDTH, node.size?.[0] || 0),
        defaultHeight: chrome + THUMB_HEIGHT,
        chromeHeight: liveChrome,
        minWidgetHeight: MIN_THUMB_HEIGHT,
    }).domWidget;

    // ── settings drawer ─────────────────────────────────────────────────── //

    function buildSettingsPanel() {
        const panel = document.createElement("div");
        panel.className = "ts-film__panel";
        const head = document.createElement("div");
        head.className = "ts-film__panel-head";
        const title = document.createElement("span");
        title.className = "ts-film__panel-title";
        title.textContent = strings.settings;
        const reset = document.createElement("button");
        reset.type = "button";
        reset.className = "ts-ui-btn ts-ui-btn--ghost";
        reset.textContent = strings.reset;
        reset.title = strings.resetTip;
        const close = document.createElement("button");
        close.type = "button";
        close.className = "ts-ui-btn ts-ui-btn--icon ts-ui-btn--ghost";
        close.title = strings.close;
        close.setAttribute("aria-label", strings.close);
        close.innerHTML = CLOSE_SVG;
        head.append(title, reset, close);
        const body = document.createElement("div");
        body.className = "ts-film__panel-body";
        // ⚠️ Nodes 2.0 zooms the canvas on a wheel over a node even when the
        // event's propagation is stopped. The frontend leaves the wheel alone
        // only inside an element marked with this attribute AND only while the
        // focus is in it (useCanvasInteractions, measured on frontend 1.4x) —
        // so the body is focusable and takes the focus when the drawer opens;
        // a slider that was touched holds it just as well.
        body.dataset.captureWheel = "true";
        body.tabIndex = -1;
        body.style.outline = "none";
        panel.append(head, body);

        const fieldNames = strings.fields || {};
        const controls = [];
        let group = null;
        for (const spec of SETTINGS) {
            const widget = settingWidgets.get(spec.name);
            if (!widget) continue;
            if (spec.group !== group) {
                group = spec.group;
                const heading = document.createElement("div");
                heading.className = "ts-ui-title ts-film__group";
                heading.textContent = { tone: strings.groupTone, colour: strings.groupColour, grain: strings.groupGrain }[group];
                body.appendChild(heading);
            }
            const label = fieldNames[spec.name] || STRINGS.en.fields[spec.name] || spec.name;
            const tip = (strings.tips || {})[spec.name] || STRINGS.en.tips[spec.name] || "";
            controls.push(buildControl(spec, widget, label, tip, body));
        }

        const sync = () => {
            let modified = false;
            for (const control of controls) modified = control.sync() || modified;
            gear.classList.toggle("is-modified", modified);
            const enabled = enableWidget?.value !== false;
            container.classList.toggle("is-off", !enabled);
            offBadge.hidden = enabled;
        };
        reset.addEventListener("click", (event) => {
            event.stopPropagation();
            for (const control of controls) control.reset();
            sync();
            scheduleThumb();
        });
        close.addEventListener("click", (event) => {
            event.stopPropagation();
            setOpen(false);
        });
        for (const control of controls) control.onChange = sync;
        body.addEventListener("pointerdown", () => {
            if (!body.contains(document.activeElement)) body.focus({ preventScroll: true });
        });
        return { panel, sync, focus: () => body.focus({ preventScroll: true }) };
    }

    function buildControl(spec, widget, label, tip, parent) {
        const opts = widget.options || {};
        const min = Number.isFinite(Number(opts.min)) ? Number(opts.min) : spec.min;
        const max = Number.isFinite(Number(opts.max)) ? Number(opts.max) : spec.max;
        const step = spec.step || 1;
        const decimals = Math.max(0, (String(step).split(".")[1] || "").length);
        const control = { onChange: null };

        const commit = (value) => {
            setWidget(widget, value);
            if (spec.name === INPUT_FILM_STRENGTH) mirrorStrength();
            control.onChange?.();
            // The thumbnail follows the tone and colour sliders (not grain).
            if (spec.preview) scheduleThumb();
            node.setDirtyCanvas?.(true, true);
        };

        const field = document.createElement("div");
        field.className = "ts-film__field";
        if (tip) field.title = tip;
        const row = document.createElement("div");
        row.className = "ts-film__field-row";
        const name = document.createElement("span");
        name.className = "ts-film__field-name";
        name.textContent = label;
        row.appendChild(name);
        field.appendChild(row);
        parent.appendChild(field);

        if (spec.kind === "int") {
            const input = document.createElement("input");
            input.type = "number";
            input.className = "ts-ui-input ts-film__seed";
            input.min = String(min);
            input.max = String(max);
            input.step = "1";
            row.appendChild(input);
            input.addEventListener("change", () => {
                const parsed = Math.round(Number(input.value));
                commit(Number.isFinite(parsed) ? Math.min(max, Math.max(min, parsed)) : spec.def);
                control.sync();
            });
            control.sync = () => {
                const value = Number(widget.value ?? spec.def);
                if (document.activeElement !== input) input.value = String(value);
                const changed = value !== spec.def;
                field.classList.toggle("is-changed", changed);
                return changed;
            };
            control.reset = () => setWidget(widget, spec.def);
            return control;
        }

        const valueEl = document.createElement("span");
        valueEl.className = "ts-film__field-value";
        row.appendChild(valueEl);
        const slider = document.createElement("input");
        slider.type = "range";
        slider.className = "ts-ui-slider ts-film__slider";
        slider.min = String(min);
        slider.max = String(max);
        slider.step = String(step);
        field.appendChild(slider);
        // The fill starts at 0 for a control that crosses it (warmth), else at the left end.
        const origin = min < 0 && max > 0 ? (0 - min) / (max - min) : 0;
        slider.style.setProperty("--origin", `${origin * 100}%`);

        const paint = (value) => {
            valueEl.textContent = value.toFixed(decimals);
            slider.style.setProperty("--fill", `${((value - min) / (max - min || 1)) * 100}%`);
        };
        slider.addEventListener("input", () => {
            const value = Number(slider.value);
            paint(value);
            commit(value);
        });
        slider.addEventListener("dblclick", (event) => {
            event.stopPropagation();
            commit(spec.def);
            control.sync();
        });
        control.sync = () => {
            const value = Number(widget.value ?? spec.def);
            slider.value = String(value);
            paint(value);
            const changed = Math.abs(value - spec.def) > step / 2;
            field.classList.toggle("is-changed", changed);
            return changed;
        };
        control.reset = () => {
            setWidget(widget, spec.def);
            if (spec.name === INPUT_FILM_STRENGTH) mirrorStrength();
        };
        return control;
    }

    let shownUrl = "";
    // The node's thumbnail is the look WITH the tone and colour sliders, so a
    // slider is judged by eye; the gallery cards show each look alone.
    const thumbAdjustments = () => {
        const adjust = {};
        for (const spec of SETTINGS) {
            if (!spec.preview) continue;
            const value = Number(settingWidgets.get(spec.name)?.value ?? spec.def);
            if (Number.isFinite(value) && Math.abs(value - spec.def) > 1e-6) adjust[spec.preview] = value;
        }
        return adjust;
    };
    const updateThumb = () => {
        const preset = String(presetWidget.value ?? EXTERNAL_LUT);
        const lut = String(lutWidget.value ?? NO_LUT);
        const url = previewUrl(preset, preset === EXTERNAL_LUT ? lut : NO_LUT, "small", thumbAdjustments());
        if (url !== shownUrl) {
            shownUrl = url;
            thumbImg.src = url;
        }
    };
    // A dragged slider fires dozens of events; one request per pause is enough
    // (the server renders a 480 px frame in tens of milliseconds).
    let thumbTimer = 0;
    function scheduleThumb() {
        clearTimeout(thumbTimer);
        thumbTimer = setTimeout(updateThumb, 120);
    }
    const refresh = () => {
        showActiveStrength();
        settings.sync();
        const preset = String(presetWidget.value ?? EXTERNAL_LUT);
        const lut = String(lutWidget.value ?? NO_LUT);
        const id = lookIdOf(preset, lut);
        updateThumb();
        if (id === "original") {
            nameEl.textContent = strings.original;
            familyEl.textContent = strings.originalFamily;
        } else if (id.startsWith("lut:")) {
            nameEl.textContent = lut.replace(/\.cube$/i, "");
            familyEl.textContent = strings.lutFamily;
        } else {
            nameEl.textContent = preset;
            familyEl.textContent = "";
            loadCatalogue().then((data) => {
                const look = data.looks.find((item) => item.id === id);
                if (look && lookIdOf(String(presetWidget.value), String(lutWidget.value)) === id) {
                    familyEl.textContent = familyLabel(look, data.families, strings, lang);
                }
            }).catch(() => {});
        }
    };

    // The combos are hidden, but anything that still writes them through their
    // callback (another extension, a script) keeps the thumbnail honest.
    for (const widget of [presetWidget, lutWidget]) {
        const original = widget.callback;
        widget.callback = function tsFilmFollow(...args) {
            const result = original?.apply(this, args);
            refresh();
            return result;
        };
    }

    const setWidget = (widget, value) => {
        const values = widget.options?.values;
        // A LUT dropped into the folder after the page loaded is in the
        // gallery (it reads the folder) but not yet in the combo.
        if (Array.isArray(values) && !values.includes(value)) values.push(value);
        writeValue(widget, value);
        widget.callback?.(value);
    };

    const choose = (look) => {
        setWidget(presetWidget, look.preset);
        // A stock ignores lut_choice; clearing it keeps the saved graph honest
        // about what is applied instead of carrying a dead LUT name around.
        setWidget(lutWidget, look.kind === "stock" ? NO_LUT : look.lut);
        mirrorStrength();
        refresh();
        node.setDirtyCanvas?.(true, true);
    };

    let overlay = null;
    async function openGallery() {
        if (overlay) return;
        const content = document.createElement("div");
        content.className = "ts-film-gallery";
        const head = document.createElement("div");
        head.className = "ts-film-gallery__head";
        const search = document.createElement("input");
        search.type = "search";
        search.className = "ts-film-gallery__search";
        search.placeholder = strings.search;
        const chips = document.createElement("div");
        chips.className = "ts-film-gallery__chips";
        head.append(search, chips);
        const scroll = document.createElement("div");
        scroll.className = "ts-film-gallery__scroll";
        const grid = document.createElement("div");
        grid.className = "ts-film-gallery__grid";
        const empty = document.createElement("div");
        empty.className = "ts-film-gallery__empty";
        empty.hidden = true;
        empty.textContent = strings.empty;
        scroll.append(grid, empty);
        content.append(head, scroll);

        overlay = openFullscreenOverlay(content, {
            label: strings.galleryTitle,
            closeOnBackdrop: true,
            onClose: () => { overlay = null; },
        });

        let data;
        try {
            data = await loadCatalogue(true);
        } catch (error) {
            console.error("[TS Film Emulation] could not load looks", error);
            empty.textContent = strings.loadFailed;
            empty.hidden = false;
            return;
        }

        const selectedId = lookIdOf(String(presetWidget.value), String(lutWidget.value));
        let family = ALL;
        const cards = [];
        for (const look of data.looks) {
            const card = document.createElement("button");
            card.type = "button";
            card.className = "ts-film-card";
            if (look.id === selectedId) card.classList.add("is-selected");
            const imgBox = document.createElement("div");
            imgBox.className = "ts-film-card__img";
            const img = document.createElement("img");
            img.loading = "lazy";
            img.decoding = "async";
            img.alt = "";
            img.src = previewUrl(look.preset, look.lut);
            const badge = document.createElement("span");
            badge.className = "ts-film-card__badge";
            badge.textContent = strings.current;
            badge.hidden = look.id !== selectedId;
            imgBox.append(img, badge);
            const meta = document.createElement("div");
            meta.className = "ts-film-card__meta";
            const name = document.createElement("span");
            name.className = "ts-film-card__name";
            name.textContent = look.kind === "original" ? strings.original : look.name;
            const fam = document.createElement("span");
            fam.className = "ts-film-card__family";
            fam.textContent = familyLabel(look, data.families, strings, lang);
            const note = document.createElement("span");
            note.className = "ts-film-card__note";
            note.textContent = (lang === "ru" ? look.note_ru : look.note_en) || "";
            meta.append(name, fam, note);
            card.append(imgBox, meta);
            card.addEventListener("click", () => {
                choose(look);
                overlay?.close?.();
            });
            grid.appendChild(card);
            const haystack = `${name.textContent} ${fam.textContent} ${note.textContent}`.toLowerCase();
            cards.push({ card, look, haystack });
        }

        const filter = () => {
            const query = search.value.trim().toLowerCase();
            let shown = 0;
            for (const item of cards) {
                const inFamily = family === ALL || item.look.family === family || item.look.kind === "original";
                const visible = inFamily && (!query || item.haystack.includes(query));
                item.card.hidden = !visible;
                if (visible) shown += 1;
            }
            empty.hidden = shown > 0;
        };

        const familyIds = [ALL, ...data.families.map((item) => item.id)
            .filter((id) => data.looks.some((look) => look.family === id))];
        const chipButtons = [];
        for (const id of familyIds) {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "ts-ui-btn";
            const known = data.families.find((item) => item.id === id);
            chip.textContent = id === ALL ? strings.all : (known?.[lang] || known?.en || id);
            chip.classList.toggle("is-active", id === family);
            chip.addEventListener("click", () => {
                family = id;
                for (const other of chipButtons) other.classList.toggle("is-active", other === chip);
                filter();
            });
            chipButtons.push(chip);
            chips.appendChild(chip);
        }
        search.addEventListener("input", filter);
        cards.find((item) => item.look.id === selectedId)?.card.scrollIntoView({ block: "nearest" });
    }

    node._tsFilmRefresh = refresh;
    refresh();

    // A NEW node gets its height from the measured layout once it exists; a
    // node coming from a saved graph keeps the size it was saved with.
    const previousConfigure = node.onConfigure;
    node.onConfigure = function tsFilmFromGraph(...args) {
        node._tsFilmFromGraph = true;
        return previousConfigure?.apply(this, args);
    };
    const fitFreshNode = (attempt = 0) => {
        if (node._tsFilmFromGraph || !node.graph) return;
        const top = Number(lookWidget?.y);
        // ⚠️ During loadGraphData the node exists BEFORE its configure: a timer
        // fired in that gap and grew a saved 280x400 node to 582 (measured).
        // Wait for the load to finish; by then onConfigure has flagged it.
        if (app.configuringGraph || !(Number.isFinite(top) && top > 0)) {
            // A canvas that is not being drawn (hidden tab) never measures the
            // widget; give it ~10 s, then keep the default — it works, just smaller.
            if (attempt < 100) setTimeout(() => fitFreshNode(attempt + 1), 100);
            return;
        }
        const wanted = Math.ceil(top + 6 + THUMB_HEIGHT);
        if ((node.size?.[1] || 0) < wanted) {
            node.setSize?.([node.size[0], wanted]);
            node.setDirtyCanvas?.(true, true);
        }
    };
    setTimeout(fitFreshNode, 0);
}

app.registerExtension({
    name: EXTENSION_ID,
    nodeCreated(node) {
        if (node?.comfyClass === NODE_NAME) setupFilmNode(node);
    },
    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_NAME) return;
        node._tsFilmFromGraph = true;  // keeps its saved size (see fitFreshNode)
        // Widget values are restored by now (they were defaults in nodeCreated).
        if (!node._tsFilmReady) setupFilmNode(node);
        node._tsFilmRefresh?.();
    },
    refreshComboInNodes() {
        // "R" re-reads model folders; the gallery re-reads the LUT folder too.
        catalogueRequest = null;
    },
});
