// «Разложить аккуратно» — команды наведения порядка в меню правой кнопки.
//
// Выделил ноды, нажал правой кнопкой, выбрал команду — схема встала по
// колонкам слева направо, каждая нода получила размер под своё содержимое, всё
// выровнялось по сетке. Ничего настраивать не нужно.
//
// Это НЕ нода: наводить порядок на холсте — работа холста, и требовать ради
// этого поставить ноду в граф было бы странно. Так же сделан значок байпаса на
// группах (`js/utils/group_bypasser/ts-group-badge.js`).
//
// ГДЕ ЖИВУТ КОМАНДЫ. Правый клик по ноде и правый клик по пустому холсту —
// крючки штатные (`getNodeMenuItems`, `getCanvasMenuItems`), ими пользуется сам
// ComfyUI. Плюс палитра команд, где на них можно повесить горячую клавишу.
//
// Пункт один, но с подпунктами: полная раскладка и отдельно — только точки на
// проводах, когда ноды трогать не надо.
//
// Расстановка — в `_tidy_layout.js` (чистая математика, проверяется без
// браузера), связь с графом — в `_tidy_graph.js`, точки на линиях — в
// `_tidy_reroutes.js`.

import { app } from "/scripts/app.js";

import { pickLocaleStrings } from "../../_theme.js";
import {
    adoptRenderedSizes,
    applyPositions,
    applySizes,
    collectTargets,
    equaliseColumnWidths,
    planPositions,
} from "./_tidy_graph.js";
import { PACKED } from "./_tidy_layout.js";
import { applyPipes, nodeBoxes, planPipes, segmentHitsBox } from "./_tidy_pipes.js";
import { applyReroutes, planReroutes } from "./_tidy_reroutes.js";

const COMMAND_TIDY = "TS.TidyLayout";
const COMMAND_REROUTES = "TS.TidyReroutes";
const COMMAND_PIPES = "TS.TidyPipes";
const COMMAND_PACKED = "TS.TidyPacked";

const STRINGS = {
    en: {
        menu: "Tidy up",
        title: "Tidy layout",
        piped: "Tidy layout + route the wires",
        packed: "Pack as tiles (wires untouched)",
        reroutes: "Align link dots only",
        packedDone: (n) => `Packed ${n} node(s) into columns.`,
        nothing: "Select at least two nodes, or open a graph that has them.",
        done: (n) => `Tidy layout: ${n} node(s) arranged.`,
        routed: (n, made) => (made
            ? `Tidy layout: ${n} node(s) arranged, ${made} link dot(s) added.`
            : `Tidy layout: ${n} node(s) arranged — no wire needed rerouting.`),
        dots: (n) => (n ? `Aligned ${n} link dot(s).` : "No link dots to align."),
        failed: "Tidy layout failed — see the browser console.",
    },
    ru: {
        menu: "Навести порядок",
        title: "Разложить аккуратно",
        piped: "Разложить и развести провода",
        packed: "Уложить плитками (провода не трогать)",
        reroutes: "Выровнять только точки на связях",
        packedDone: (n) => `Уложено плитками: ${n}.`,
        nothing: "Выделите хотя бы две ноды — или откройте граф, где они есть.",
        done: (n) => `Разложено нод: ${n}.`,
        routed: (n, made) => (made
            ? `Разложено нод: ${n}, добавлено точек: ${made}.`
            : `Разложено нод: ${n} — разводить было нечего.`),
        dots: (n) => (n ? `Выровнено точек на связях: ${n}.` : "Точек на связях нет."),
        failed: "Разложить не удалось — подробности в консоли браузера.",
    },
};

function toast(severity, detail) {
    try {
        app.extensionManager?.toast?.add?.({ severity, detail, life: 2500 });
    } catch {
        /* без всплывашек тоже живём */
    }
}

/**
 * Дать рендереру доделать своё.
 *
 * ⚠️ Между «поставили размер» и «прочитали размер» обязана пройти отрисовка:
 * Nodes 2.0 правит высоту ноды по своей раскладке уже после нашего `setSize`
 * (замерено: SaveImage 58 → 70). Два кадра — потому что первый только запускает
 * пересчёт, а поправленное значение видно на втором.
 */
function settled() {
    return new Promise((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(resolve, 0)));
    });
}

/**
 * Поставить точки на связях на прямую между концами провода.
 *
 * ⚠️ Кроме тех, чей обход НЕ ЗРЯ. Если прямая между гнёздами режет чужую ноду,
 * точки этой связи остаются как есть: их поставили, чтобы обойти препятствие.
 * Без этой оговорки обычная раскладка, запущенная после разводки, распрямляла
 * провода обратно и возвращала пересечения (замерено: 0 → 5).
 *
 * @param {object} board холст
 * @param {object[]} [nodes] ноды, которые считаются препятствиями
 */
function alignReroutes(board, nodes = null) {
    const graph = board?.graph || app.graph;
    const boxes = nodeBoxes(nodes || graph?._nodes || []);
    const blocked = (link, start, end) => boxes.some((box) => (
        box.node !== graph.getNodeById?.(link.origin_id)
        && box.node !== graph.getNodeById?.(link.target_id)
        && segmentHitsBox(start[0], start[1], end[0], end[1], box)
    ));
    const moved = applyReroutes(planReroutes(graph, { blocked }));
    if (moved) {
        app.graph?.change?.();
        board?.setDirty?.(true, true);
    }
    return moved;
}

/**
 * Разложить то, что сейчас выделено (или весь текущий граф).
 *
 * @param {object} canvas холст
 * @param {boolean} [route] ставить ли новые точки там, где провод режет ноды
 */
async function tidy(canvas, route = false) {
    const t = pickLocaleStrings(STRINGS);
    const board = canvas || app.canvas;
    try {
        const targets = collectTargets(board);
        if (targets.length < 2) {
            toast("info", t.nothing);
            return;
        }
        const before = applySizes(targets);
        await settled();
        // Что нарисовано, то и правда: Nodes 2.0 отказывается ужиматься ниже
        // своего минимума и молчит об этом (см. `adoptRenderedSizes`).
        adoptRenderedSizes(board, before);
        await settled();
        const plan = planPositions(board);
        if (!plan.count) {
            toast("info", t.nothing);
            return;
        }
        applyPositions(app, plan);
        // Точки на проводах считаются ПОСЛЕ переезда нод: их место выводится из
        // положения гнёзд, а оно только что изменилось.
        await settled();
        // ⚠️ ВЫРАВНИВАНИЕ И РАЗВОДКА — ПРОТИВОПОЛОЖНОСТИ, и вместе их звать
        // нельзя. Выравнивание тянет точки на прямую между гнёздами, разводка
        // нарочно уводит провод в обход. Первая версия делала и то, и другое:
        // выравнивание шло последним и растаскивало только что разведённое —
        // замерено, пересечений стало БОЛЬШЕ, чем без разводки вовсе. А если
        // просто поменять порядок, то второй запуск команды распрямил бы то,
        // что развёл первый.
        //
        // Поэтому: обычная раскладка выравнивает (точки человека должны
        // переехать вслед за нодами), раскладка с разводкой — не трогает
        // готовые точки и добавляет свои только тем связям, у которых их нет.
        // Оба режима от этого повторяемы: второй запуск ничего не ломает.
        let made = 0;
        if (route) {
            made = applyPipes(board.graph, planPipes(board.graph, targets));
        } else {
            alignReroutes(board, targets);
        }
        toast("success", route ? t.routed(plan.count, made) : t.done(plan.count));
    } catch (error) {
        console.error("[TS TidyLayout] failed", error);
        toast("error", t.failed);
    }
}

/**
 * Уложить плитками: плотно, колонка к колонке, ширины внутри колонки равны.
 *
 * Провода НЕ трогаются вовсе — ни выравнивания, ни разводки. Это другой запрос:
 * не «покажи поток», а «убери пустоту». Раскладка по колонкам та же (иначе
 * получилась бы просто сетка без смысла), но всё сдвинуто вплотную.
 */
async function tidyPacked(canvas) {
    const t = pickLocaleStrings(STRINGS);
    const board = canvas || app.canvas;
    try {
        const targets = collectTargets(board);
        if (targets.length < 2) {
            toast("info", t.nothing);
            return;
        }
        const before = applySizes(targets);
        await settled();
        adoptRenderedSizes(board, before);
        await settled();

        // ⚠️ ШИРИНЫ РАВНЯЮТСЯ ПО ТЕМ КОЛОНКАМ, В КОТОРЫХ НОДЫ И ОКАЖУТСЯ.
        //
        // Было: черновая раскладка → выравнивание по ней → ВТОРАЯ раскладка. А
        // вторая давала другие колонки (ширины-то изменились), и нода уносила с
        // собой ширину чужой колонки: нижняя оказывалась шире всей колонки, а
        // ManualSigmas — уже соседей (живое замечание, видно на снимке).
        //
        // Теперь считаем и равняем по кругу, пока равнять не станет нечего.
        // Сходится быстро: выравнивание поднимает узких до самого широкого, а
        // сам он не меняется — значит и ширина колонки не плывёт.
        let plan = planPositions(board, { ...PACKED, packed: true });
        if (!plan.count) {
            toast("info", t.nothing);
            return;
        }
        for (let pass = 0; pass < 3; pass += 1) {
            const resizes = equaliseColumnWidths(plan.moves);
            if (!resizes.length) break;
            for (const { node, width, height } of resizes) {
                if (typeof node.setSize === "function") node.setSize([width, height]);
                else node.size = [width, height];
            }
            await settled();
            adoptRenderedSizes(board, new Map(
                resizes.map(({ node }) => [node, [node.size[0], node.size[1]]])));
            await settled();
            plan = planPositions(board, { ...PACKED, packed: true });
        }
        applyPositions(app, plan);
        toast("success", t.packedDone(plan.count));
    } catch (error) {
        console.error("[TS TidyLayout] packing failed", error);
        toast("error", t.failed);
    }
}

/** Только точки на связях: ноды остаются где стоят. */
function tidyReroutesOnly(canvas) {
    const t = pickLocaleStrings(STRINGS);
    try {
        toast("success", t.dots(alignReroutes(canvas || app.canvas)));
    } catch (error) {
        console.error("[TS TidyLayout] reroutes failed", error);
        toast("error", t.failed);
    }
}

/**
 * Пункт меню с подпунктами.
 *
 * ⚠️ Подменю в LiteGraph — это не поле `submenu`, а обработчик, который сам
 * открывает второе меню и передаёт ему `parentMenu`: без этого первое меню
 * закроется вместе со вторым, и выбрать в нём будет нечего.
 */
function menuEntry(canvas) {
    const t = pickLocaleStrings(STRINGS);
    return {
        content: t.menu,
        has_submenu: true,
        callback: (_value, _options, event, parentMenu) => {
            const board = canvas || app.canvas;
            const entries = [
                { content: t.title, callback: () => tidy(board) },
                { content: t.piped, callback: () => tidy(board, true) },
                { content: t.packed, callback: () => tidyPacked(board) },
                { content: t.reroutes, callback: () => tidyReroutesOnly(board) },
            ];
            new LiteGraph.ContextMenu(entries, {
                event,
                parentMenu,
                callback: (entry) => entry?.callback?.(),
            });
        },
    };
}

app.registerExtension({
    name: "ts.tidyLayout",

    commands: [
        {
            id: COMMAND_TIDY,
            // Подпись читается один раз при регистрации: смена языка в ComfyUI
            // перезагружает страницу, живой перерисовки не нужно.
            label: pickLocaleStrings(STRINGS).title,
            icon: "pi pi-th-large",
            function: () => tidy(app.canvas),
        },
        {
            id: COMMAND_PIPES,
            label: pickLocaleStrings(STRINGS).piped,
            icon: "pi pi-sitemap",
            function: () => tidy(app.canvas, true),
        },
        {
            id: COMMAND_PACKED,
            label: pickLocaleStrings(STRINGS).packed,
            icon: "pi pi-table",
            function: () => tidyPacked(app.canvas),
        },
        {
            id: COMMAND_REROUTES,
            label: pickLocaleStrings(STRINGS).reroutes,
            icon: "pi pi-share-alt",
            function: () => tidyReroutesOnly(app.canvas),
        },
    ],

    // Правый клик по ноде.
    getNodeMenuItems() {
        return [menuEntry(app.canvas)];
    },

    // Правый клик по пустому месту холста.
    getCanvasMenuItems(canvas) {
        return [menuEntry(canvas)];
    },
});
