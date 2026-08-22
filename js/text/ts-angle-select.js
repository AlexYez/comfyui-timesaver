/**
 * TS Angle Select — визуальный выбор ракурса камеры.
 *
 * Сверху — 3D-превью: объект, орбита вокруг него, камера на орбите и рельс
 * приближения. Под ним три отдельных регулятора: поворот, высота, крупность.
 *
 * ⚠️ Высота превью и ширины колонок ЗАФИКСИРОВАНЫ. Внизу была строка с готовым
 * промптом; она переносилась на разное число строк, из-за этого менялась высота
 * превью, нода дёргалась — и ползунок уезжал из-под курсора прямо во время
 * перетаскивания. Ничто в этом виджете не имеет права менять раскладку от
 * значения.
 *
 * ⚠️ Превью НИЧЕГО не принимает от мыши. Настраивать три величины одним
 * движением по одному холсту оказалось неудобно (замечание владельца пака):
 * у каждой величины свой регулятор, а сцена показывает результат.
 *
 * ⚠️ Сцена и Three.js — в `_angle_scene.js`, и библиотека подтягивается ЛЕНИВО
 * из `nodes/text/_vendor/` через свой маршрут. Класть её в `js/` нельзя:
 * ComfyUI импортирует каждый `.js` веб-папки при загрузке страницы, и 675 КБ
 * платили бы все, даже никогда не поставив эту ноду.
 *
 * Размер виджета — через общий `addResizableDomWidget`: он один раз кодирует
 * правильную работу и в Nodes 1.0, и в Nodes 2.0 (Vue).
 */

import { app } from "/scripts/app.js";

import {
    TS_UI_CLASS,
    ensureThemeStyles,
    getThemeColors,
    pickLocaleStrings,
} from "../_theme.js";
import { addResizableDomWidget, hideWidget, getWidget } from "../_dom_widget.js";
import { createAngleScene, loadThree } from "./_angle_scene.js";

const EXTENSION_ID = "ts.angleSelect";
const NODE_ID = "TS_AngleSelect";
const STYLE_ID = "ts-angle-select-style";
const WIDGET_NAME = "ts_angle_select";

const INPUT_PRESET = "preset";
const INPUT_AZIMUTH = "azimuth";
const INPUT_HEIGHT = "height";
const INPUT_FRAMING = "framing";

const AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315];
const HEIGHTS = ["low", "eye-level", "elevated", "high"];
const FRAMINGS = ["wide", "medium", "close-up"];

const MIN_NODE_WIDTH = 340;
const MIN_NODE_HEIGHT = 400;
const DEFAULT_NODE_WIDTH = 420;
const DEFAULT_NODE_HEIGHT = 520;
const CHROME_HEIGHT = 30;
const MIN_WIDGET_HEIGHT = 300;

const STRINGS = {
    en: {
        wide: "wide",
        medium: "medium",
        "close-up": "close-up",
        low: "low",
        "eye-level": "eye level",
        elevated: "elevated",
        high: "high",
        rotation: "Rotation",
        height: "Height",
        zoom: "Zoom",
        rotationTip: "Where the camera stands around the subject. Eight positions, "
            + "45 degrees apart — the ones the model was trained on.",
        heightTip: "How high the camera sits: below the subject, level with it, a little "
            + "above, or well above.",
        zoomTip: "How much of the subject is in frame: the whole of it and its "
            + "surroundings, roughly waist up, or head and shoulders.",
        noThree: "3D preview unavailable",
    },
    ru: {
        wide: "общий",
        medium: "средний",
        "close-up": "крупный",
        low: "снизу",
        "eye-level": "на уровне глаз",
        elevated: "приподнято",
        high: "сверху",
        rotation: "Поворот",
        height: "Высота",
        zoom: "Приближение",
        rotationTip: "Где камера стоит вокруг объекта. Восемь положений через 45° — "
            + "те, на которых обучалась модель.",
        heightTip: "На какой высоте камера: ниже объекта, на уровне глаз, чуть выше "
            + "или заметно выше.",
        zoomTip: "Сколько объекта в кадре: целиком с окружением, примерно по пояс "
            + "или голова и плечи.",
        noThree: "3D-превью недоступно",
    },
};

const STYLE_TEXT = `
.ts-angle{position:relative;width:100%;height:100%;min-height:0;display:flex;flex-direction:column;
    gap:6px;padding:6px;box-sizing:border-box}
.ts-angle__stage{position:relative;flex:1 1 auto;min-height:0;border-radius:var(--ts-radius);
    border:1px solid var(--ts-border);background:var(--ts-surface);overflow:hidden;outline:none}
.ts-angle__badges{position:absolute;left:8px;top:8px;display:flex;gap:5px;pointer-events:none;
    z-index:2}
.ts-angle__badge{font-size:11px;line-height:1;padding:4px 7px;border-radius:999px;
    background:var(--ts-sunken);border:1px solid var(--ts-border);color:var(--ts-text);
    white-space:nowrap}
.ts-angle__rows{flex:0 0 auto;display:flex;flex-direction:column;gap:4px}
.ts-angle__row{display:flex;align-items:center;gap:8px}
.ts-angle__name{flex:0 0 74px;width:74px;font-size:11px;letter-spacing:.03em;
    text-transform:uppercase;color:var(--ts-muted);overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap}
/* ⚠️ Дорожка задаётся ЯВНО, через псевдоэлементы обоих движков. Общий
   '.ts-ui-slider' красит фон самого элемента цветом '--ts-border-soft'; на
   тёмной поверхности ноды при высоте 4 px её попросту не видно — оставался
   один бегунок, висящий в пустоте. */
.ts-angle__slider{flex:1 1 auto;min-width:0;height:16px;background:transparent}
.ts-angle__slider::-webkit-slider-runnable-track{height:4px;border-radius:999px;
    background:var(--ts-border);border:0}
.ts-angle__slider::-moz-range-track{height:4px;border-radius:999px;
    background:var(--ts-border);border:0}
.ts-angle__slider::-webkit-slider-thumb{width:14px;height:14px;margin-top:-5px}
.ts-angle__slider::-moz-range-thumb{width:14px;height:14px}
/* ⚠️ Ширина ФИКСИРОВАНА, а не минимальна: «eye level» и «high» разной длины,
   и от плавающей колонки дорожка ползунка меняла бы длину прямо под курсором. */
.ts-angle__value{flex:0 0 92px;width:92px;text-align:right;font-size:12px;
    color:var(--ts-text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ts-angle__note{position:absolute;right:8px;top:8px;z-index:2;font-size:11px;
    padding:4px 7px;border-radius:999px;background:var(--ts-sunken);
    border:1px solid var(--ts-border);color:var(--ts-muted)}
`;

function ensureStyles(doc) {
    ensureThemeStyles();
    if (!doc || doc.getElementById(STYLE_ID)) return;
    const style = doc.createElement("style");
    style.id = STYLE_ID;
    style.textContent = STYLE_TEXT;
    doc.head.appendChild(style);
}

function stopPropagation(element, events) {
    for (const name of events) {
        element.addEventListener(name, (event) => event.stopPropagation());
    }
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (widget) {
        widget.value = value;
        if (typeof widget.callback === "function") widget.callback(value);
    }
    // Скрытый виджет в Nodes 2.0 может не донести значение до workflow JSON —
    // дублируем в properties (§12.5.13 в CLAUDE.md).
    node.properties ||= {};
    node.properties[name] = value;
}

function readValue(node, name, fallback) {
    const widget = getWidget(node, name);
    const raw = widget?.value;
    if (raw !== undefined && raw !== null && raw !== "") return raw;
    const stored = node?.properties?.[name];
    if (stored !== undefined && stored !== null && stored !== "") return stored;
    return fallback;
}

function snapAzimuthValue(value) {
    return ((Math.round(Number(value) / 45) * 45) % 360 + 360) % 360;
}

function setupAngleSelect(node) {
    if (node._tsAngleSelectInitialized) return;
    if (typeof node.addDOMWidget !== "function") return;
    node._tsAngleSelectInitialized = true;

    // ⚠️ LiteGraph успевает посчитать размер по виджетам ДО нас, и хелпер
    // уважает уже выставленный `node.size` — иначе нода открывалась бы в
    // минимальном размере. Сохранённый размер это не задевает: он приезжает
    // позже, в onConfigure.
    node.size = [DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT];

    const doc = node?.graph?.canvas?.canvas?.ownerDocument || document;
    ensureStyles(doc);
    const L = pickLocaleStrings(STRINGS);

    for (const name of [INPUT_AZIMUTH, INPUT_HEIGHT, INPUT_FRAMING]) {
        hideWidget(node, name);
    }

    const container = doc.createElement("div");
    container.className = `${TS_UI_CLASS} ts-angle`;

    const stage = doc.createElement("div");
    stage.className = "ts-angle__stage";
    stage.tabIndex = 0;

    const badges = doc.createElement("div");
    badges.className = "ts-angle__badges";
    const azimuthBadge = doc.createElement("div");
    azimuthBadge.className = "ts-angle__badge";
    const heightBadge = doc.createElement("div");
    heightBadge.className = "ts-angle__badge";
    const framingBadge = doc.createElement("div");
    framingBadge.className = "ts-angle__badge";
    badges.append(azimuthBadge, heightBadge, framingBadge);

    stage.append(badges);

    // Три регулятора — по одному на величину. Каждый шагает по своим детентам:
    // между ними у модели просто нет слов, поэтому промежуточных положений и
    // не предлагаем.
    const rows = doc.createElement("div");
    rows.className = "ts-angle__rows";

    function makeRow(name, count, tip) {
        const row = doc.createElement("div");
        row.className = "ts-angle__row";
        const label = doc.createElement("div");
        label.className = "ts-angle__name";
        label.textContent = name;
        const slider = doc.createElement("input");
        slider.type = "range";
        slider.className = "ts-ui-slider ts-angle__slider";
        slider.min = "0";
        slider.max = String(count - 1);
        slider.step = "1";
        // Сторож `test_tooltips` требует подсказку у каждого управляющего
        // элемента пака, и требует по делу: ползунок без подписи не объясняет,
        // что именно он двигает.
        slider.title = tip;
        label.title = tip;
        const value = doc.createElement("div");
        value.className = "ts-angle__value";
        // ⚠️ Свои события мыши и клавиш: общий stopPropagation на контейнере
        // ловит всплытие, но LiteGraph слушает холст, и без этого ползунок
        // «отклеивается» от курсора, стоит выйти за пределы дорожки.
        stopPropagation(slider, ["pointerdown", "pointermove", "pointerup", "mousedown",
                                 "mouseup", "dblclick", "contextmenu", "wheel", "keydown"]);
        row.append(label, slider, value);
        rows.appendChild(row);
        return { slider, value };
    }

    const rotationRow = makeRow(L.rotation, AZIMUTHS.length, L.rotationTip);
    const heightRow = makeRow(L.height, HEIGHTS.length, L.heightTip);
    const zoomRow = makeRow(L.zoom, FRAMINGS.length, L.zoomTip);

    container.append(stage, rows);

    const state = {
        azimuth: snapAzimuthValue(readValue(node, INPUT_AZIMUTH, 0)),
        height: String(readValue(node, INPUT_HEIGHT, "eye-level")),
        framing: String(readValue(node, INPUT_FRAMING, "medium")),
        scene: null,
    };
    if (!HEIGHTS.includes(state.height)) state.height = "eye-level";
    if (!FRAMINGS.includes(state.framing)) state.framing = "medium";

    function refresh() {
        azimuthBadge.textContent = `${state.azimuth}°`;
        heightBadge.textContent = L[state.height] || state.height;
        framingBadge.textContent = L[state.framing] || state.framing;
        rotationRow.slider.value = String(Math.max(0, AZIMUTHS.indexOf(state.azimuth)));
        rotationRow.value.textContent = `${state.azimuth}°`;
        heightRow.slider.value = String(Math.max(0, HEIGHTS.indexOf(state.height)));
        heightRow.value.textContent = L[state.height] || state.height;
        zoomRow.slider.value = String(Math.max(0, FRAMINGS.indexOf(state.framing)));
        zoomRow.value.textContent = L[state.framing] || state.framing;
    }

    function commit() {
        setWidgetValue(node, INPUT_AZIMUTH, state.azimuth);
        setWidgetValue(node, INPUT_HEIGHT, state.height);
        setWidgetValue(node, INPUT_FRAMING, state.framing);
        refresh();
        state.scene?.setState(state);
        node.setDirtyCanvas?.(true, true);
    }

    const bind = (row, values, field) => {
        row.slider.addEventListener("input", () => {
            const index = Math.min(values.length - 1, Math.max(0, Number(row.slider.value) | 0));
            if (values[index] === state[field]) return;
            state[field] = values[index];
            commit();
        });
    };
    bind(rotationRow, AZIMUTHS, "azimuth");
    bind(heightRow, HEIGHTS, "height");
    bind(zoomRow, FRAMINGS, "framing");

    stopPropagation(container, [
        "pointerdown", "pointerup", "mousedown", "mouseup", "wheel", "dblclick", "contextmenu",
    ]);

    addResizableDomWidget(node, container, {
        name: WIDGET_NAME,
        minWidth: MIN_NODE_WIDTH,
        minHeight: MIN_NODE_HEIGHT,
        defaultWidth: DEFAULT_NODE_WIDTH,
        defaultHeight: DEFAULT_NODE_HEIGHT,
        chromeHeight: CHROME_HEIGHT,
        minWidgetHeight: MIN_WIDGET_HEIGHT,
        onResize: () => state.scene?.resize(),
    });

    if (typeof ResizeObserver === "function") {
        const observer = new ResizeObserver(() => state.scene?.resize());
        observer.observe(stage);
        const previousRemoved = node.onRemoved;
        node.onRemoved = function onRemoved(...args) {
            observer.disconnect();
            node._tsAngleSelectDisposed = true;
            state.scene?.dispose();
            state.scene = null;
            return previousRemoved?.apply(this, args);
        };
    }

    node._tsAngleSelectRehydrate = () => {
        state.azimuth = snapAzimuthValue(readValue(node, INPUT_AZIMUTH, state.azimuth));
        const height = String(readValue(node, INPUT_HEIGHT, state.height));
        const framing = String(readValue(node, INPUT_FRAMING, state.framing));
        state.height = HEIGHTS.includes(height) ? height : state.height;
        state.framing = FRAMINGS.includes(framing) ? framing : state.framing;
        state.scene?.setState(state);
        refresh();
    };

    refresh();


    loadThree()
        .then((THREE) => {
            if (node._tsAngleSelectDisposed) return;
            state.scene = createAngleScene({
                container: stage,
                THREE,
                colors: getThemeColors(),
                state,
            });
            state.scene.resize();
            refresh();
        })
        .catch((error) => {
            console.error("[TS Angle Select] 3D preview failed to load:", error);
            const note = doc.createElement("div");
            note.className = "ts-angle__note";
            note.textContent = L.noThree;
            stage.appendChild(note);
        });
}

app.registerExtension({
    name: EXTENSION_ID,
    nodeCreated(node) {
        if (node?.comfyClass !== NODE_ID) return;
        setupAngleSelect(node);
    },
    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_ID) return;
        if (!node._tsAngleSelectInitialized) {
            setupAngleSelect(node);
            return;
        }
        node._tsAngleSelectRehydrate?.();
    },
});
