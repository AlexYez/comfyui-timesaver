// Значок байпаса на самой группе — как в rgthree.
//
// Кнопка в правом верхнем углу заголовка группы: нажал — вся группа ушла в
// байпас, нажал ещё раз — вернулась. Это не нода: значок принадлежит КАЖДОЙ
// группе на холсте, а нода, без которой не работает оформление холста, была бы
// странным требованием. Панель со списком групп (`TS_GroupBypasser`) остаётся
// на месте — она про обзор и правила, а этот значок про одно движение рукой.
//
// Выключается в настройках ComfyUI: чужой холст — не место для того, что
// человек не может убрать.
//
// ЧТО ЗДЕСЬ ЕСТЬ И ЧЕГО НЕТ. Здесь только геометрия значка, рисование и
// попадание курсора. Кто входит в группу и как пишется режим — в
// `_groups_watch.js`, что означает набор режимов — в `_groups_model.js`. Второй
// записи байпаса в паке нет и быть не должно: разойдутся.

import { app } from "/scripts/app.js";

import {
    STATE_EMPTY,
    STATE_OFF,
    STATE_ON,
    groupState,
} from "./_groups_model.js";
import { managedNodes, setGroupBypassed } from "./_groups_watch.js";

const SETTING_ID = "TS.GroupBypassBadge";

// Размеры в координатах ГРАФА, а не экрана: значок живёт на группе и должен
// ездить и масштабироваться вместе с ней.
const BADGE_SIZE = 20;
const BADGE_INSET = 8;
// Ниже этого масштаба значок мельче пальца и только сорит на схеме, где человек
// смотрит на всю картину целиком.
const MIN_SCALE = 0.35;

/** Включён ли значок. Настройка читается на каждый кадр — она дешёвая. */
function badgeEnabled() {
    try {
        const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
        if (value !== undefined && value !== null) return Boolean(value);
    } catch { /* старый фронтенд — ниже запасной путь */ }
    try {
        const value = app?.ui?.settings?.getSettingValue?.(SETTING_ID, true);
        return value === undefined || value === null ? true : Boolean(value);
    } catch {
        return true;
    }
}

/** Заголовок группы: его высота — единственное, что задаёт место значку. */
function titleHeight(group) {
    const own = Number(group?.titleHeight);
    if (Number.isFinite(own) && own > 0) return own;
    return Number(window.LiteGraph?.NODE_TITLE_HEIGHT) || 24;
}

/**
 * Где лежит значок этой группы, в координатах графа.
 *
 * @returns {{x: number, y: number, size: number} | null}
 */
export function badgeRect(group) {
    const bounds = group?._bounding || group?.bounding;
    if (!bounds || bounds.length < 4) return null;
    const [x, y, width] = bounds;
    const head = titleHeight(group);
    // Не больше половины заголовка: у мелкой группы значок не должен вылезать
    // за свою полосу.
    const size = Math.min(BADGE_SIZE, head * 0.72);
    if (!(size > 0) || !(width > size + BADGE_INSET * 2)) return null;
    return {
        x: x + width - size - BADGE_INSET,
        y: y + (head - size) / 2,
        size,
    };
}

/** Состояние группы: включена, выключена, вперемешку или пустая. */
function stateOf(group) {
    return groupState(managedNodes(group).map((node) => Number(node.mode)));
}

/**
 * Нарисовать значок.
 *
 * Рисуется своими цветами, а не токенами темы: значок лежит поверх
 * пользовательского цвета группы и обязан читаться на любом — от почти белого
 * до почти чёрного. Отсюда тёмная подложка и светлый глиф.
 */
function drawBadge(ctx, rect, state) {
    const { x, y, size } = rect;
    const radius = size * 0.28;
    const on = state === STATE_ON;
    const off = state === STATE_OFF;

    ctx.save();
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, size, size, radius);
    else ctx.rect(x, y, size, size);
    ctx.fillStyle = off ? "rgba(20,20,24,.78)" : "rgba(20,20,24,.55)";
    ctx.fill();
    ctx.lineWidth = Math.max(1, size * 0.07);
    ctx.strokeStyle = off ? "rgba(255,255,255,.30)" : "rgba(255,255,255,.55)";
    ctx.stroke();

    // Глиф питания: дуга с разрывом сверху и вертикальная черта. Понятен без
    // подписи и не зависит от языка.
    const cx = x + size / 2;
    const cy = y + size / 2;
    const r = size * 0.26;
    ctx.strokeStyle = on ? "rgba(255,255,255,.92)"
        : (off ? "rgba(255,255,255,.42)" : "rgba(255,210,120,.92)");
    ctx.lineWidth = Math.max(1.2, size * 0.11);
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.arc(cx, cy + size * 0.03, r, -Math.PI * 0.72, Math.PI * 1.72);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 1.25);
    ctx.lineTo(cx, cy + size * 0.02);
    ctx.stroke();
    ctx.restore();
}

/** Группы графа — под всеми именами, какие встречались у фронтенда. */
function graphGroups(graph) {
    const groups = graph?._groups || graph?.groups || [];
    return Array.isArray(groups) ? groups : [];
}

let installed = false;

/**
 * Повесить рисование и попадание курсора на холст.
 *
 * ⚠️ Оборачиваем ОДИН раз и навсегда: перезаход в `setup` при переключении
 * рабочих процессов накрутил бы обёртку на обёртку, и значок рисовался бы
 * дважды, а нажатие срабатывало бы дважды.
 */
function install() {
    const Canvas = window.LGraphCanvas;
    if (installed || !Canvas?.prototype?.drawGroups) return;
    installed = true;

    const originalDraw = Canvas.prototype.drawGroups;
    Canvas.prototype.drawGroups = function tsDrawGroups(canvas, ctx, ...rest) {
        const result = originalDraw.call(this, canvas, ctx, ...rest);
        try {
            if (!badgeEnabled()) return result;
            const scale = Number(this.ds?.scale) || 1;
            if (scale < MIN_SCALE) return result;
            for (const group of graphGroups(this.graph)) {
                const state = stateOf(group);
                // Пустой группе выключать нечего — значок был бы обманом.
                if (state === STATE_EMPTY) continue;
                const rect = badgeRect(group);
                if (rect) drawBadge(ctx, rect, state);
            }
        } catch (err) {
            console.warn("[TS GroupBadge] could not draw the badges", err);
        }
        return result;
    };

    // ⚠️ Нажатие ловится слушателем в ФАЗЕ ПЕРЕХВАТА, а не обёрткой над
    // `processMouseDown`. Обёртка на прототипе не срабатывает: LiteGraph
    // привязывает свой обработчик к элементу холста при создании канваса —
    // то есть ДО того, как расширение успевает что-то обернуть, и в DOM висит
    // ссылка на исходный метод. Замерено: обёртка стояла (`tsProcessMouseDown`),
    // а нажатие по значку до неё не доходило.
    //
    // Слушатель на документе, а не на самом холсте: фронтенд пересоздаёт
    // элемент холста при некоторых переходах, и подписка на конкретный узел
    // однажды повисла бы в пустоте. Чужие нажатия отсекаются проверкой цели.
    document.addEventListener("pointerdown", onPointerDown, true);
}

function onPointerDown(event) {
    const canvas = app?.canvas;
    const element = canvas?.canvas;
    if (!canvas || !element || event.target !== element) return;
    try {
        if (!hitBadge(canvas, event)) return;
        // Гасим целиком: иначе LiteGraph следом начнёт тащить группу.
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
    } catch (err) {
        console.warn("[TS GroupBadge] could not handle the click", err);
    }
}

/**
 * Нажали ли по значку — и если да, переключить группу.
 *
 * Порядок перебора обратный: группы рисуются снизу вверх, поэтому верхняя
 * (последняя) должна получать нажатие первой — иначе на вложенных группах
 * сработает не та.
 */
function hitBadge(canvas, event) {
    if (!badgeEnabled()) return false;
    if (event?.button !== undefined && event.button !== 0) return false;
    const scale = Number(canvas.ds?.scale) || 1;
    if (scale < MIN_SCALE) return false;

    const point = canvas.convertEventToCanvasOffset?.(event);
    if (!point) return false;
    const [px, py] = point;

    const groups = graphGroups(canvas.graph);
    for (let index = groups.length - 1; index >= 0; index -= 1) {
        const group = groups[index];
        const rect = badgeRect(group);
        if (!rect) continue;
        if (px < rect.x || px > rect.x + rect.size) continue;
        if (py < rect.y || py > rect.y + rect.size) continue;
        const state = stateOf(group);
        if (state === STATE_EMPTY) return false;
        // Вперемешку — включаем: человек, ткнувший в наполовину выключенную
        // группу, хочет её вернуть, а не добить.
        setGroupBypassed(group, state !== STATE_ON);
        canvas.setDirty?.(true, true);
        return true;
    }
    return false;
}

app.registerExtension({
    name: "ts.groupBypassBadge",
    settings: [
        {
            id: SETTING_ID,
            category: ["TS Timesaver", "Canvas", "Group bypass badge"],
            name: "Bypass button on group headers",
            tooltip: "A small power button in the top-right corner of every "
                + "group: one click bypasses the whole group, another brings "
                + "it back.",
            type: "boolean",
            defaultValue: true,
            onChange: () => { app?.canvas?.setDirty?.(true, true); },
        },
    ],
    async setup() {
        install();
    },
});
