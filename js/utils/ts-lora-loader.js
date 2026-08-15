// TS LoRA Loader — стопка LoRA в одной ноде.
//
// Что человек видит: список строк и плюс под ним. Плюс открывает поиск по
// установленным LoRA, строка показывает имя и силу, за ручку слева строку можно
// перетащить выше или ниже. Всё.
//
// Порядок в списке — не оформление: LoRA накладываются последовательно, и от
// перестановки результат меняется. Поэтому перетаскивание двигает именно ту
// строку, за которую взялись, и видно, куда она встанет.
//
// РАЗДЕЛЕНИЕ. Список и правила — в `_lora_stack.js` (без DOM, проверяется без
// браузера). Здесь только интерфейс и связь с нодой. Загрузку LoRA не делает ни
// тот, ни другой: нода на бэкенде разворачивается в цепочку родных
// `LoraLoaderModelOnly`.
//
// ХРАНЕНИЕ. Одна скрытая строка JSON (`loras_json`) плюс зеркало в
// `node.properties`: в Nodes 2.0 скрытые виджеты не всегда доносят значение до
// сохранённого workflow (§12.5.13 CLAUDE.md), а список терять нельзя.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import {
    TS_UI_CLASS,
    ensureThemeStyles,
    pickLocaleStrings,
} from "../_theme.js";
import { addResizableDomWidget, getWidget, hideWidget } from "../_dom_widget.js";
import {
    addLora,
    clampStrength,
    filterNames,
    formatStrength,
    moveItem,
    parseStack,
    removeAt,
    serialiseStack,
    setEnabled,
    setStrength,
    setStrengthSpec,
    shortName,
    stepStrength,
    strengthSpec,
} from "./_lora_stack.js";

const NODE_TYPE = "TS_LoraLoader";
const STORE_WIDGET = "loras_json";
const DOM_WIDGET = "ts_lora_stack";
const STYLE_ID = "ts-lora-loader-styles";

const MIN_NODE_WIDTH = 300;
const MIN_NODE_HEIGHT = 140;
const DEFAULT_NODE_WIDTH = 340;
const DEFAULT_NODE_HEIGHT = 200;
const ROW_HEIGHT = 30;
// Чувствительность протаскивания. Шаг стрелок берётся у родной ноды (он там
// мелкий, 0.01), а тянуть им было бы мучением: 0.01 за пиксель — сто пикселей
// на единицу силы. Это число про руку, а не про формат значения.
const SCRUB_PER_PIXEL = 0.05;

function ensureStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-lora{display:flex;flex-direction:column;gap:6px;height:100%;box-sizing:border-box;
    padding:2px;overflow:hidden}
.ts-lora__list{flex:1 1 0;min-height:0;overflow-y:auto;display:flex;flex-direction:column;gap:4px}
/* flex:1 1 0 — основа НОЛЬ. С «auto» список просит высоту под все свои строки,
   и в Nodes 2.0 нода растягивается вниз по всему холсту (§12.5.1). */
/* ⚠️ Любое своё display в этом файле обязано иметь пару под [hidden]: браузерное
   [hidden]{display:none} слабее правила по классу и молча отменяется им. Ниже у
   всплывающего списка такая пара есть — не потерять её при правках. */
.ts-lora__empty{margin:auto;padding:10px 6px;text-align:center;color:var(--ts-muted);
    font-size:var(--ts-fs-sm);line-height:1.4}
.ts-lora__row{display:flex;align-items:center;gap:6px;padding:3px 4px;
    border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);
    background:var(--ts-surface);height:${ROW_HEIGHT - 4}px;box-sizing:border-box}
/* ⚠️ Бледнеет СОДЕРЖИМОЕ, а не строка. Прозрачность на строке делает группу, из
   которой ребёнку не выйти: выключатель — то самое, чем строку возвращают, —
   гас бы вместе с ней и оказывался самым незаметным местом строки. */
.ts-lora__row.is-off .ts-lora__name,
.ts-lora__row.is-off .ts-lora__spin{opacity:.45}
/* Взятая строка прилипает к курсору: сама она едет отдельной карточкой в корне
   документа, а на её месте в списке остаётся пустая рамка — видно и что несёшь,
   и куда оно встанет. */
.ts-lora__row.is-placeholder{border-style:dashed;border-color:var(--ts-accent);
    background:transparent}
.ts-lora__row.is-placeholder>*{visibility:hidden}
.ts-lora__ghost{position:fixed;z-index:11001;pointer-events:none;margin:0;
    border-color:var(--ts-accent);
    /* Тень — единственное, чем карточка отличается от строки: она летит над
       списком, и это должно читаться сразу. Никаких наклонов: строка едет
       ровно, как и лежала. */
    box-shadow:0 10px 26px rgba(0,0,0,.5)}
.ts-lora__list.is-sorting{cursor:grabbing}
.ts-lora__list.is-sorting .ts-lora__grip{cursor:grabbing}
/* Пока тащат — соседи не должны перехватывать наведение и подмигивать. */
.ts-lora__list.is-sorting .ts-lora__row{pointer-events:none}
.ts-lora__grip{flex:0 0 auto;width:14px;text-align:center;cursor:grab;color:var(--ts-muted);
    font-size:12px;line-height:1;user-select:none;touch-action:none}
.ts-lora__grip:active{cursor:grabbing}
/* Выключатель строки. Тот же квадрат, что у переключателей групп в
   TS Group Bypasser (js/utils/group_bypasser/_groups_view.js): пак должен
   читаться как одна система, а не как набор разных галочек. */
.ts-lora__on{flex:0 0 auto;position:relative;width:15px;height:15px;padding:0;
    border:1px solid var(--ts-border-strong);border-radius:4px;
    background:var(--ts-sunken);cursor:pointer}
.ts-lora__on::after{content:"";position:absolute;inset:3px;border-radius:2px;
    background:transparent}
.ts-lora__on[aria-checked="true"]{border-color:var(--ts-accent)}
.ts-lora__on[aria-checked="true"]::after{background:var(--ts-accent)}
.ts-lora__on:hover{border-color:var(--ts-accent)}
.ts-lora__name{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap;font-size:var(--ts-fs-sm);color:var(--ts-text);cursor:pointer}
/* Сила и стрелки — одним блоком, как числовой виджет самого ComfyUI. */
.ts-lora__spin{flex:0 0 auto;display:flex;align-items:center;
    background:var(--ts-sunken);border:1px solid var(--ts-border);
    border-radius:var(--ts-radius-sm);overflow:hidden}
.ts-lora__step{flex:0 0 auto;width:16px;height:100%;padding:0;border:none;
    background:transparent;color:var(--ts-muted);cursor:pointer;
    font-size:9px;line-height:1;font-family:inherit;user-select:none;touch-action:none}
.ts-lora__step:hover{color:var(--ts-accent);background:var(--ts-surface-hover)}
.ts-lora__strength{flex:0 0 auto;width:48px;text-align:center;font-size:var(--ts-fs-sm);
    font-variant-numeric:tabular-nums;cursor:ew-resize;
    background:transparent;border:none;color:var(--ts-text);padding:2px 0}
.ts-lora__strength:focus{outline:1px solid var(--ts-accent);cursor:text}
.ts-lora__drop{flex:0 0 auto;width:20px;height:20px;padding:0;line-height:1;
    display:inline-flex;align-items:center;justify-content:center;cursor:pointer;
    border:none;background:transparent;color:var(--ts-muted);font-size:14px}
.ts-lora__drop:hover{color:var(--ts-danger)}
.ts-lora__add{flex:0 0 auto;height:26px;border:1px dashed var(--ts-border-strong);
    border-radius:var(--ts-radius-sm);background:transparent;color:var(--ts-muted);
    cursor:pointer;font-size:var(--ts-fs-sm);font-family:inherit}
.ts-lora__add:hover{color:var(--ts-accent);border-color:var(--ts-accent)}
/* ⚠️ Поиск живёт В КОРНЕ ДОКУМЕНТА, а не внутри ноды, и позиционируется у
   кнопки. Две причины, обе выяснены дорогой ценой:
   1) Внутри ноды ему негде развернуться: нода по умолчанию около 300×146, и на
      список находок оставалось 20 пикселей — человек видел пустоту.
   2) Панель, absolute-позиционированная ВНУТРИ виджета, вылезает за его
      прямоугольник, и по её кнопкам не попасть: сверху лежит холст графа и
      забирает нажатия себе («canvas intercepts pointer events» в Playwright).
   Элемент в корне документа со своим z-index свободен и от того, и от другого —
   ровно так же ведут себя выпадающие списки самого ComfyUI. */
.ts-lora__pick{position:fixed;z-index:11000;box-sizing:border-box;
    display:flex;flex-direction:column;gap:4px;padding:4px;
    border:1px solid var(--ts-border-strong);border-radius:var(--ts-radius);
    background:var(--ts-surface);
    /* Тень — единственный способ отделить всплывающий список от того, что под
       ним: под ним чужой холст любого цвета. */
    box-shadow:0 8px 24px rgba(0,0,0,.45)}
.ts-lora__pick[hidden]{display:none}
.ts-lora__found{flex:1 1 0;min-height:0;overflow-y:auto;
    display:flex;flex-direction:column;gap:2px}
/* ⚠️ flex:0 0 auto — иначе строки СЖИМАЮТСЯ. Колонка-флекс ужимает своих детей
   по умолчанию, и полторы сотни находок в ней превращались в полоски по 8
   пикселей: список выглядел пустым, хотя данные были на месте. */
.ts-lora__found button{flex:0 0 auto;text-align:left;padding:4px 6px;border:none;background:transparent;
    color:var(--ts-text);font-size:var(--ts-fs-sm);border-radius:var(--ts-radius-sm);
    cursor:pointer;font-family:inherit;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ts-lora__found button:hover,.ts-lora__found button.is-active{background:var(--ts-surface-hover)}
.ts-lora__none{padding:6px;color:var(--ts-muted);font-size:var(--ts-fs-xs)}
`;
    document.head.appendChild(style);
}

const STRINGS = {
    en: {
        add: "+  Add LoRA",
        addHint: "Search the LoRAs installed here and add one to the stack",
        cancel: "Cancel",
        empty: "No LoRA yet.\nPress the button below to add one.",
        search: "Type to search…",
        searchHint: "Fragments in any order: \"det xl\" finds add_detail_XL",
        none: "Nothing matches.",
        remove: "Remove",
        strength: "Strength — drag sideways, type, or step with the arrows. "
            + "Negative values are allowed.",
        stepDown: "Step down",
        stepUp: "Step up",
        grip: "Drag to reorder. Order matters: LoRAs apply one after another.",
        toggle: "Click the name to set the row aside without removing it",
        turnOff: "Turn this LoRA off — it stays in the list with its strength",
        turnOn: "Turn this LoRA back on",
        missing: "This LoRA is not in the loras folder on this machine",
    },
    ru: {
        add: "+  Добавить LoRA",
        addHint: "Найти LoRA среди установленных и добавить её в стопку",
        cancel: "Отмена",
        empty: "Пока пусто.\nНажмите кнопку ниже, чтобы добавить LoRA.",
        search: "Начните печатать…",
        searchHint: "Куски в любом порядке: «det xl» найдёт add_detail_XL",
        none: "Ничего не нашлось.",
        remove: "Убрать",
        strength: "Сила — тяните вбок, впишите или шагайте стрелками. "
            + "Отрицательная тоже можно.",
        stepDown: "Шаг вниз",
        stepUp: "Шаг вверх",
        grip: "Перетащите, чтобы изменить порядок. Порядок важен: LoRA "
            + "накладываются одна за другой.",
        toggle: "Нажмите на имя, чтобы отложить строку, не удаляя её",
        turnOff: "Выключить эту LoRA — строка останется в списке вместе с силой",
        turnOn: "Включить эту LoRA обратно",
        missing: "Такой LoRA нет в папке loras на этой машине",
    },
};

// Родную ноду спрашиваем ОДИН раз на страницу: ответ одинаков для всех наших
// нод, а /object_info — пара сотен килобайт.
//
// ⚠️ Спрашиваем не только имена LoRA, но и параметры силы (границы, шаг,
// умолчание). Свои копии этих чисел разошлись бы с ComfyUI при первом же его
// обновлении, а нода разворачивается именно в родную — значит и принимать
// должна ровно то же, что примет она.
const NATIVE_LOADER = "LoraLoaderModelOnly";
let nativePromise = null;

/**
 * Варианты выпадающего списка из описания входа.
 *
 * ⚠️ Форматов ДВА, и они сосуществуют в одной сборке: у ноды на V1 вход выглядит
 * как `[[...варианты], {...}]`, у ноды на V3 — как `["COMBO", {options: [...]}]`.
 * В этой установке 256 виджетов описаны первым способом и 576 вторым (замерено).
 * Родной загрузчик LoRA пока на V1 — но читать только его форму значит остаться
 * с пустым списком в тот день, когда ComfyUI переведёт ноду на V3.
 */
function comboOptions(definition) {
    if (!Array.isArray(definition) || !definition.length) return null;
    const [head, spec] = definition;
    if (Array.isArray(head)) return head.map(String);
    if (Array.isArray(spec?.options)) return spec.options.map(String);
    return null;
}

/** Разобрать описание родного загрузчика: имена LoRA + параметры силы. */
function readNativeDef(def) {
    const required = def?.input?.required ?? {};
    const names = comboOptions(required.lora_name);
    if (!names) return null;
    // Второй элемент — параметры виджета; у старых сборок их может не быть
    // вовсе, тогда останется запасной набор из _lora_stack.js.
    if (required.strength_model?.[1]) setStrengthSpec(required.strength_model[1]);
    return names;
}

async function nativeSpec() {
    if (!nativePromise) {
        nativePromise = (async () => {
            try {
                const response = await api.fetchApi(`/object_info/${NATIVE_LOADER}`);
                const info = await response.json();
                const names = readNativeDef(info?.[NATIVE_LOADER]);
                if (names) return names;
                console.warn(`[TS LoRA Loader] ${NATIVE_LOADER} has no lora_name options`);
                nativePromise = null;
                return [];
            } catch (error) {
                console.warn(`[TS LoRA Loader] could not read ${NATIVE_LOADER}`, error);
                // ⚠️ Неудачу НЕ запоминаем. Раньше промис оставался в
                // `nativePromise` вместе с пустым списком, и одна временная
                // осечка `/object_info` (сервер ещё поднимался, сеть моргнула)
                // означала пустой список LoRA до перезагрузки страницы —
                // выбрать было нечего, а причина ничем не показывалась.
                nativePromise = null;
                return [];
            }
        })();
    }
    return nativePromise;
}

/**
 * Забыть список установленных LoRA.
 *
 * Он спрашивается один раз на страницу — иначе каждая нода тянула бы сотни
 * килобайт `/object_info`. Но человек, положивший файл в `models/loras`, жмёт
 * «R» и ждёт, что новая LoRA появится: с этого момента запомненный список
 * УСТАРЕЛ и его надо выбросить, а не ждать перезагрузки страницы.
 *
 * @param {object} [defs] ответ обновления; если родной загрузчик в нём есть,
 *   свежий список берётся прямо оттуда и второй запрос не нужен.
 * @returns {string[] | null} новые имена, когда они пришли вместе с `defs`.
 */
function forgetNativeSpec(defs) {
    const fresh = readNativeDef(defs?.[NATIVE_LOADER]);
    nativePromise = fresh ? Promise.resolve(fresh) : null;
    return fresh;
}

/** Прочитать список из ноды: сперва виджет, затем зеркало в свойствах. */
function readStack(node) {
    const widget = getWidget(node, STORE_WIDGET);
    const fromWidget = parseStack(widget?.value);
    if (fromWidget.length) return fromWidget;
    return parseStack(node?.properties?.[STORE_WIDGET]);
}

/** Записать список в ноду — в оба места сразу. */
function writeStack(node, stack) {
    const text = serialiseStack(stack);
    const widget = getWidget(node, STORE_WIDGET);
    if (widget) {
        widget.value = text;
        if (typeof widget.callback === "function") widget.callback(text);
    }
    node.properties ||= {};
    node.properties[STORE_WIDGET] = text;
}

function setupLoraLoader(node) {
    ensureStyles();
    const t = pickLocaleStrings(STRINGS);

    const container = document.createElement("div");
    container.className = `${TS_UI_CLASS} ts-lora`;
    container.style.position = "relative";

    const list = document.createElement("div");
    list.className = "ts-lora__list";
    const empty = document.createElement("div");
    empty.className = "ts-lora__empty";
    empty.textContent = t.empty;

    const addButton = document.createElement("button");
    addButton.type = "button";
    addButton.className = "ts-lora__add";
    addButton.textContent = t.add;
    addButton.title = t.addHint;

    // ── поиск ────────────────────────────────────────────────────────────── #
    const picker = document.createElement("div");
    // Токены темы едут вместе с элементом: он лежит вне ноды, и наследовать их
    // от неё уже не может.
    picker.className = `${TS_UI_CLASS} ts-lora__pick`;
    picker.hidden = true;
    const search = document.createElement("input");
    search.type = "text";
    search.className = "ts-ui-input";
    search.placeholder = t.search;
    search.title = t.searchHint;
    search.spellcheck = false;
    const found = document.createElement("div");
    found.className = "ts-lora__found";
    picker.append(search, found);

    container.append(list, addButton);
    document.body.appendChild(picker);

    let stack = [];
    let names = [];
    let dragFrom = -1;

    const commit = () => {
        writeStack(node, stack);
        render();
    };

    // ── строки ───────────────────────────────────────────────────────────── #
    function buildRow(entry, index) {
        const row = document.createElement("div");
        row.className = "ts-lora__row";
        if (entry.on === false) row.classList.add("is-off");
        row.dataset.index = String(index);

        const grip = document.createElement("span");
        grip.className = "ts-lora__grip";
        grip.textContent = "⠿";
        grip.title = t.grip;
        grip.addEventListener("pointerdown", (event) => startDrag(event, index));

        // Выключатель строки. Отложить LoRA можно было и раньше — нажатием по
        // имени, — но об этом знал только тот, кто прочитал подсказку: строка
        // просто бледнела, и человек удалял LoRA вместо того, чтобы выключить.
        // Выключатель говорит о себе сам и держит то же состояние `on`.
        const on = entry.on !== false;
        const power = document.createElement("button");
        power.type = "button";
        power.className = "ts-lora__on";
        power.setAttribute("role", "switch");
        power.setAttribute("aria-checked", String(on));
        power.title = on ? t.turnOff : t.turnOn;
        power.addEventListener("click", () => {
            stack = setEnabled(stack, index, !on);
            commit();
        });

        const name = document.createElement("span");
        name.className = "ts-lora__name";
        name.textContent = shortName(entry.name);
        name.title = names.length && !names.includes(entry.name)
            ? `${entry.name}\n${t.missing}` : `${entry.name}\n${t.toggle}`;
        if (names.length && !names.includes(entry.name)) name.style.color = "var(--ts-warning)";
        name.addEventListener("click", () => {
            stack = setEnabled(stack, index, entry.on === false);
            commit();
        });

        // Сила показана и управляется как у родной ноды: две цифры после
        // точки и стрелки по краям, шагающие на её же 0.01.
        const spin = document.createElement("div");
        spin.className = "ts-lora__spin";

        const strength = document.createElement("input");
        strength.className = "ts-lora__strength";
        strength.value = formatStrength(entry.strength);
        strength.title = t.strength;
        strength.inputMode = "decimal";
        strength.addEventListener("change", () => {
            stack = setStrength(stack, index, strength.value);
            commit();
        });
        attachScrub(strength, index);

        const arrow = (direction, label, hint) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "ts-lora__step";
            button.textContent = label;
            button.title = hint;
            // Зажатая стрелка должна идти сама: набрать 0.01-шагами заметную
            // разницу отдельными щелчками — работа, а не настройка.
            let timer = null;
            let repeat = null;
            const bump = () => {
                stack = setStrength(stack, index, stepStrength(readStrength(index), direction));
                strength.value = formatStrength(readStrength(index));
                writeStack(node, stack);
            };
            const stop = () => {
                clearTimeout(timer);
                clearInterval(repeat);
                timer = null;
                repeat = null;
                render();
            };
            button.addEventListener("pointerdown", (event) => {
                button.setPointerCapture?.(event.pointerId);
                bump();
                timer = setTimeout(() => { repeat = setInterval(bump, 60); }, 400);
                event.preventDefault();
                event.stopPropagation();
            });
            button.addEventListener("pointerup", stop);
            button.addEventListener("pointercancel", stop);
            return button;
        };

        spin.append(arrow(-1, "◀", t.stepDown), strength, arrow(1, "▶", t.stepUp));

        const drop = document.createElement("button");
        drop.type = "button";
        drop.className = "ts-lora__drop";
        drop.textContent = "×";
        drop.title = t.remove;
        drop.addEventListener("click", () => {
            stack = removeAt(stack, index);
            commit();
        });

        row.append(grip, power, name, spin, drop);
        return row;
    }

    /** Текущая сила строки — из списка, а не из поля: поле лишь показывает. */
    function readStrength(index) {
        return stack[index]?.strength ?? strengthSpec().default;
    }

    /**
     * Тянуть силу вбок — как у обычного числового виджета ComfyUI.
     *
     * Мышь не отпускается на границе ноды: захват указателя держит жест до
     * конца, куда бы курсор ни ушёл.
     *
     * ⚠️ Жест и щелчок делятся ПОРОГОМ в несколько пикселей. Нажатие само по
     * себе ещё ничего не значит: `preventDefault` в нём нужен, чтобы холст не
     * начал тащить ноду, но он же отбирает у поля фокус. Поэтому фокус
     * возвращается руками — на отпускании, если указатель так и не сдвинулся.
     * Без этого поле нельзя было заполнить с клавиатуры вовсе, хотя подсказка
     * это обещает.
     */
    function attachScrub(input, index) {
        const DRAG_THRESHOLD = 3;
        let dragging = false;
        let moved = false;
        let startX = 0;
        let startValue = 0;
        input.addEventListener("pointerdown", (event) => {
            if (document.activeElement === input) return;   // печатают — не мешаем
            dragging = true;
            moved = false;
            startX = event.clientX;
            startValue = Number(input.value) || 0;
            input.setPointerCapture?.(event.pointerId);
            event.preventDefault();
        });
        input.addEventListener("pointermove", (event) => {
            if (!dragging) return;
            const shift = event.clientX - startX;
            if (!moved && Math.abs(shift) < DRAG_THRESHOLD) return;
            moved = true;
            const value = clampStrength(startValue + shift * SCRUB_PER_PIXEL);
            input.value = String(value);
            stack = setStrength(stack, index, value);
            writeStack(node, stack);
        });
        const stop = (event) => {
            if (!dragging) return;
            dragging = false;
            input.releasePointerCapture?.(event.pointerId);
            if (!moved) {
                // Обычный щелчок: дать напечатать. Перерисовка тут запрещена —
                // она заменила бы строку вместе с полем, в которое целились.
                input.focus();
                input.select();
                return;
            }
            render();
        };
        input.addEventListener("pointerup", stop);
        input.addEventListener("pointercancel", stop);
    }

    // ── перетаскивание строк ─────────────────────────────────────────────── #
    /**
     * Тащить строку — с перестановкой на ходу.
     *
     * Список переписывается сразу, как только курсор переходит на соседа: строка
     * едет вместе с ним, и порядок виден по дороге, а не открывается в конце.
     * Колебаний это не даёт — после перестановки строка оказывается ровно под
     * курсором, и следующий шаг требует нового перехода.
     *
     * ⚠️ Указатель захватывается САМИМ СПИСКОМ, а не ручкой, за которую взялись.
     * Перестановка перерисовывает строки, ручка вместе с ними исчезает — и
     * захват на ней оборвал бы жест на первом же шаге. Список переживает
     * перерисовку: меняются только его дети.
     */
    function startDrag(event, index) {
        const source = [...list.querySelectorAll(".ts-lora__row")][index];
        if (!source) return;
        const rect = source.getBoundingClientRect();
        // За какое место строки взялись — карточка обязана держаться под тем же
        // местом курсора, иначе она «прыгает» в руку при первом же движении.
        const grabX = event.clientX - rect.left;
        const grabY = event.clientY - rect.top;

        // Карточка живёт В КОРНЕ ДОКУМЕНТА: внутри списка её обрезало бы его
        // прокруткой и рамками ноды. Токены темы там не наследуются — класс
        // приходится нести с собой.
        const ghost = source.cloneNode(true);
        ghost.classList.add(TS_UI_CLASS, "ts-lora__ghost");
        ghost.style.width = `${rect.width}px`;
        ghost.style.height = `${rect.height}px`;
        const follow = (clientX, clientY) => {
            ghost.style.left = `${clientX - grabX}px`;
            ghost.style.top = `${clientY - grabY}px`;
        };
        follow(event.clientX, event.clientY);
        document.body.appendChild(ghost);

        dragFrom = index;
        list.classList.add("is-sorting");
        list.setPointerCapture?.(event.pointerId);
        render();

        const onMove = (move) => {
            if (dragFrom < 0) return;
            follow(move.clientX, move.clientY);
            const to = rowIndexAt(move.clientY);
            if (to < 0 || to === dragFrom) return;
            stack = moveItem(stack, dragFrom, to);
            dragFrom = to;
            writeStack(node, stack);      // запись без перерисовки — она ниже
            render();
        };
        const finish = (up) => {
            list.releasePointerCapture?.(up.pointerId);
            list.removeEventListener("pointermove", onMove);
            list.removeEventListener("pointerup", finish);
            list.removeEventListener("pointercancel", finish);
            list.classList.remove("is-sorting");
            ghost.remove();
            dragFrom = -1;
            commit();
        };
        list.addEventListener("pointermove", onMove);
        list.addEventListener("pointerup", finish);
        list.addEventListener("pointercancel", finish);
        event.preventDefault();
        event.stopPropagation();
    }

    /**
     * Под какой строкой сейчас курсор.
     *
     * ⚠️ Считается по ЭКРАННЫМ прямоугольникам самих строк, а не делением
     * высоты на число строк: список прокручивается, а нода бывает в масштабе,
     * и арифметика по высоте промахивается ровно тогда, когда список длинный.
     */
    function rowIndexAt(clientY) {
        const rows = [...list.querySelectorAll(".ts-lora__row")];
        for (let index = 0; index < rows.length; index += 1) {
            const rect = rows[index].getBoundingClientRect();
            if (clientY >= rect.top && clientY <= rect.bottom) return index;
        }
        if (!rows.length) return -1;
        // Выше первой или ниже последней — кладём с краю: жест, доведённый за
        // список, обычно значит «в самое начало» или «в самый конец».
        const first = rows[0].getBoundingClientRect();
        return clientY < first.top ? 0 : rows.length - 1;
    }

    function render() {
        list.textContent = "";
        if (!stack.length) {
            list.appendChild(empty);
            return;
        }
        stack.forEach((entry, index) => {
            const row = buildRow(entry, index);
            // Место, куда строка встанет, едет вместе с ней: сама строка
            // сейчас летит карточкой за курсором, а здесь — пустая рамка.
            if (index === dragFrom) row.classList.add("is-placeholder");
            list.appendChild(row);
        });
    }

    // ── добавление ───────────────────────────────────────────────────────── #
    function renderFound() {
        const matches = filterNames(names, search.value).slice(0, 200);
        found.textContent = "";
        if (!matches.length) {
            const none = document.createElement("div");
            none.className = "ts-lora__none";
            none.textContent = t.none;
            found.appendChild(none);
            return;
        }
        for (const name of matches) {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = name;
            button.title = name;
            button.addEventListener("click", () => {
                stack = addLora(stack, name);
                closePicker();
                commit();
            });
            found.appendChild(button);
        }
    }

    /**
     * Поставить список у кнопки.
     *
     * Ширина — по ноде, но не уже читаемого; высота — сколько есть до края
     * экрана, в разумных пределах. Если снизу тесно, список раскрывается вверх:
     * нода часто стоит у нижнего края окна.
     */
    function positionPicker() {
        const anchor = addButton.getBoundingClientRect();
        const margin = 8;
        const width = Math.max(240, Math.min(420, anchor.width));
        const below = window.innerHeight - anchor.bottom - margin;
        const above = anchor.top - margin;
        const up = below < 180 && above > below;
        const height = Math.max(120, Math.min(360, up ? above : below));
        picker.style.width = `${width}px`;
        picker.style.height = `${height}px`;
        picker.style.left =
            `${Math.max(margin, Math.min(window.innerWidth - width - margin, anchor.left))}px`;
        picker.style.top = up
            ? `${Math.max(margin, anchor.top - height - 4)}px`
            : `${anchor.bottom + 4}px`;
    }

    // Список лежит вне ноды и потому обязан закрываться сам: по нажатию мимо, по
    // прокрутке холста (нода уезжает из-под него) и по смене размера окна.
    const onOutside = (event) => {
        if (picker.hidden) return;
        if (picker.contains(event.target) || addButton.contains(event.target)) return;
        closePicker();
    };
    const onScrollAway = () => { if (!picker.hidden) closePicker(); };

    function openPicker() {
        picker.hidden = false;
        addButton.textContent = t.cancel;
        search.value = "";
        renderFound();
        positionPicker();
        search.focus();
        document.addEventListener("pointerdown", onOutside, true);
        window.addEventListener("wheel", onScrollAway, true);
        window.addEventListener("resize", onScrollAway);
    }

    function closePicker() {
        picker.hidden = true;
        addButton.textContent = t.add;
        document.removeEventListener("pointerdown", onOutside, true);
        window.removeEventListener("wheel", onScrollAway, true);
        window.removeEventListener("resize", onScrollAway);
    }

    addButton.addEventListener("click", async () => {
        if (!names.length) names = await nativeSpec();
        if (picker.hidden) openPicker();
        else closePicker();
    });

    // ⚠️ Список живёт в корне документа и сам по себе ноду не переживёт: без
    // уборки он останется висеть в документе вместе со слушателями.
    const previousRemoved = node.onRemoved;
    node.onRemoved = function tsLoraRemoved(...args) {
        closePicker();
        picker.remove();
        // Ноду могли удалить прямо в жесте — карточка иначе останется висеть.
        document.querySelectorAll(".ts-lora__ghost").forEach((item) => item.remove());
        return previousRemoved?.apply(this, args);
    };
    search.addEventListener("input", renderFound);
    search.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            event.stopPropagation();       // Escape закрывает поиск, а не ноду
            closePicker();
        }
        if (event.key === "Enter") {
            found.querySelector("button")?.click();
        }
    });
    addResizableDomWidget(node, container, {
        name: DOM_WIDGET,
        minWidth: MIN_NODE_WIDTH,
        minHeight: MIN_NODE_HEIGHT,
        defaultWidth: DEFAULT_NODE_WIDTH,
        defaultHeight: DEFAULT_NODE_HEIGHT,
    });
    // Строка JSON — внутреннее хозяйство, человеку её показывать незачем.
    hideWidget(node, STORE_WIDGET);

    /**
     * Перечитать состояние из ноды.
     *
     * ⚠️ Отдельно от сборки интерфейса: при загрузке workflow значения виджетов
     * приезжают ПОСЛЕ `onNodeCreated` (§12.5.12), и список, собранный один раз
     * на старте, навсегда остался бы пустым.
     */
    node._tsLoraRehydrate = async () => {
        stack = readStack(node);
        render();
        if (!names.length) {
            names = await nativeSpec();
            render();                       // имена нужны, чтобы отметить пропавшие
        }
    };
    node._tsLoraRehydrate();

    /**
     * Обновление списков по «R» (Refresh Node Definitions).
     *
     * ⚠️ Родные combo-виджеты ComfyUI обновляет сам, перебирая `node.widgets`.
     * У нашей ноды виджета-списка нет — весь выбор нарисован своими руками, — и
     * до сих пор это означало, что положенная в `models/loras` LoRA не
     * появлялась в поиске до перезагрузки страницы: «R» её не доносил.
     *
     * `refreshComboInNode` — штатный крючок ровно для этого случая: ComfyUI
     * зовёт его у КАЖДОЙ ноды графа, включая лежащие внутри подграфов, и
     * передаёт свежие описания. Список имён приезжает вместе с ними, так что
     * второй поход на сервер не нужен.
     */
    const previousRefresh = node.refreshComboInNode;
    node.refreshComboInNode = function tsLoraRefreshCombo(defs) {
        const result = previousRefresh?.apply(this, arguments);
        const fresh = forgetNativeSpec(defs);
        if (fresh) {
            names = fresh;
            render();                       // пропавшие отмечаются заново
            if (!picker.hidden) renderFound();
        } else {
            // Описания родного загрузчика в ответе не оказалось (чужая сборка,
            // урезанный ответ) — спрашиваем сами, кэш уже сброшен.
            nativeSpec().then((loaded) => {
                names = loaded;
                render();
                if (!picker.hidden) renderFound();
            });
        }
        return result;
    };
}

app.registerExtension({
    name: "ts.loraLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function tsLoraCreated(...args) {
            const result = onCreated?.apply(this, args);
            setupLoraLoader(this);
            return result;
        };
    },
    // Пар к `refreshComboInNode` выше: тот обходит ноды, ЛЕЖАЩИЕ в графе, а
    // этот сбрасывает общий кэш. Без него «R» в пустом графе не менял ничего, и
    // нода, поставленная сразу после него, получала список до обновления.
    refreshComboInNodes(defs) {
        forgetNativeSpec(defs);
    },
    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_TYPE && node?.type !== NODE_TYPE) return;
        // ⚠️ Не пересобирать виджет: в Nodes 2.0 повторная регистрация DOM-виджета
        // двоит верхнюю часть ноды (§12.5.12). Только перечитать состояние.
        if (!getWidget(node, DOM_WIDGET)) setupLoraLoader(node);
        else node._tsLoraRehydrate?.();
    },
});
