// Мост между графом ComfyUI и раскладкой.
//
// Здесь всё, что знает про LiteGraph: кого раскладывать, какого размера должна
// быть нода, кто в какой группе и как всё это применить. Сама расстановка — в
// `_tidy_layout.js`, и про ноды она не знает ничего.
//
// ⚠️ ДВА ШАГА, А НЕ ОДИН. Сперва ноды получают размер, и только потом, когда
// рендерер его поправил по-своему, считается расстановка. Причина замерена:
// `computeSize()` в обоих режимах отдаёт ОДНО И ТО ЖЕ, а вот итоговый размер
// ноды разный — SaveImage 58 в Nodes 1.0 и 70 в Nodes 2.0, чекпойнт 98 против
// 102. Посчитать раскладку по своим числам значило бы промахнуться на десяток
// пикселей в каждой строке — как раз столько, чтобы рамка группы обрезала
// нижнюю ноду. Перечитать настоящие размеры дешевле, чем держать таблицу
// поправок на каждый режим.
//
// ГРУППЫ. Ноды группы раскладываются ВНУТРИ неё, а сама группа участвует во
// внешней раскладке одним блоком. Иначе оформленный человеком workflow после
// команды рассыпался бы: рамки остались бы на месте, а содержимое разъехалось.
//
// ⚠️ СОСТАВ ГРУПП СНИМАЕТСЯ ДО ПЕРЕЕЗДА. Группа в LiteGraph состав не хранит —
// он вычисляется геометрией (центр ноды внутри рамки). Сдвинул ноды — состав
// уже другой. Поэтому он снимается один раз, в начале.

import { layoutBlocks, snap } from "./_tidy_layout.js";

// Отступ от содержимого до рамки группы.
export const GROUP_PADDING = 16;
// Заголовок ноды рисуется НАД `pos`, и в `size` его нет. Для раскладки он часть
// габарита — иначе соседний ряд наезжает на заголовок.
const NODE_TITLE = 30;

/** Ссылка по номеру: в новых сборках это Map, в старых — обычный объект. */
function linkById(graph, id) {
    const links = graph?.links;
    if (!links) return null;
    return typeof links.get === "function" ? links.get(id) : links[id];
}

/**
 * Пара чисел: точка или размер.
 *
 * ⚠️ НЕ `Array.isArray`. У ноды `pos` и `size` — не обычные массивы, а
 * типизированные (`Float32Array`), и проверка на массив выбрасывала АБСОЛЮТНО
 * ВСЕ ноды: команда честно отрабатывала и ничего не двигала.
 */
function isPair(value) {
    return Boolean(value) && typeof value === "object"
        && Number.isFinite(Number(value[0])) && Number.isFinite(Number(value[1]));
}

/** Похоже ли на ноду (а не на группу или точку перегиба в выделении). */
function isNode(item) {
    return Boolean(item && isPair(item.pos) && isPair(item.size)
        && typeof item.id !== "undefined" && item.inputs !== undefined);
}

/**
 * Ноды, которые команда имеет право двигать.
 *
 * Выделение — если в нём есть ноды; иначе весь текущий граф (в подграфе — его
 * содержимое, а не корень: человек раскладывает то, что видит). Закреплённые
 * ноды не трогаем: это единственный способ сказать «эта пусть стоит где стоит».
 */
export function collectTargets(canvas) {
    const graph = canvas?.graph;
    if (!graph) return [];
    const selected = [...(canvas.selectedItems || [])].filter(isNode);
    const pool = selected.length ? selected : (graph._nodes || []);
    return pool.filter((node) => isNode(node) && !node.flags?.pinned);
}

/**
 * Размер, который нода просит под своё содержимое.
 *
 * ⚠️ Не меньше собственного минимума. У нод со своим интерфейсом (плеер, стопка
 * LoRA, студия) минимум объявлен не для красоты: под ним интерфейс не
 * помещается. Свёрнутую ноду не трогаем — у неё размер ни при чём.
 */
export function idealSize(node) {
    if (node?.flags?.collapsed) return [node.size[0], node.size[1]];
    let computed = null;
    try {
        computed = node.computeSize?.();
    } catch {
        computed = null;
    }
    const min = node.min_size || [0, 0];
    const width = Math.max(Number(computed?.[0]) || node.size[0], Number(min[0]) || 0);
    const height = Math.max(Number(computed?.[1]) || node.size[1], Number(min[1]) || 0);
    return [Math.round(width), Math.round(height)];
}

/** Габарит ноды на холсте: то, что видно, вместе с заголовком. */
export function boxOf(node) {
    if (node.flags?.collapsed) {
        // У свёрнутой ноды `size` остаётся прежним, а видно только заголовок.
        return [node._collapsed_width || 80, NODE_TITLE];
    }
    return [node.size[0], node.size[1] + NODE_TITLE];
}

/** Рисует ли ноды Vue (Nodes 2.0): у него каждая нода — элемент документа. */
export function vueNodes() {
    return typeof document !== "undefined" && Boolean(document.querySelector(".lg-node"));
}

/**
 * Сколько нода занимает НА ЭКРАНЕ, в единицах графа.
 *
 * ⚠️ Единственное место в паке, где размер берётся из прямоугольника документа
 * — и по необходимости. Замерено: в Nodes 2.0 `setSize([140,46])` ложится в
 * `node.size` как есть, а на экране нода остаётся 225×84 и обратно ничего не
 * пишет — ни через кадр, ни через 750 мс. То есть `node.size` там перестаёт
 * быть правдой, и расстановка по нему сажает соседей друг на друга (VAE Decode
 * наезжал на Save Image — видно на снимке).
 *
 * Деление на масштаб холста обязательно: прямоугольник документа — в пикселях
 * экрана, а раскладка живёт в единицах графа (CLAUDE.md §12.5.3).
 */
export function renderedSize(canvas, node) {
    if (typeof document === "undefined") return null;
    const element = document.querySelector(`.lg-node[data-node-id="${node.id}"]`);
    if (!element) return null;
    const scale = Number(canvas?.ds?.scale) || 1;
    const rect = element.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return [
        Math.round(rect.width / scale),
        Math.round(rect.height / scale) - NODE_TITLE,
    ];
}

/**
 * Выровнять ширины внутри колонок: все плитки одной колонки одинаковы.
 *
 * Колонки берутся по уже посчитанной раскладке — по левому краю нод. Ширина
 * колонки — самая широкая её нода: ужимать соседей до узкой значило бы прятать
 * их содержимое.
 *
 * @param {Array<{node: object, x: number}>} moves план расстановки
 * @returns {Array<{node: object, width: number, height: number}>}
 */
export function equaliseColumnWidths(moves) {
    const byColumn = new Map();
    for (const move of moves) {
        const key = Math.round(move.x);
        if (!byColumn.has(key)) byColumn.set(key, []);
        byColumn.get(key).push(move.node);
    }
    const resizes = [];
    for (const nodes of byColumn.values()) {
        const widest = Math.max(...nodes.map((node) => node.size[0]));
        for (const node of nodes) {
            if (node.flags?.collapsed || node.size[0] === widest) continue;
            resizes.push({ node, width: widest, height: node.size[1] });
        }
    }
    return resizes;
}

/**
 * Шаг 1: привести размеры. Возвращает прежние — они нужны шагу 1.5.
 *
 * Отдельно от расстановки: между шагами нужно дать рендереру сказать своё
 * слово (см. предупреждение вверху файла).
 */
export function applySizes(targets) {
    const previous = new Map();
    for (const node of targets) {
        if (node.flags?.collapsed) continue;
        previous.set(node, [node.size[0], node.size[1]]);
        const [width, height] = idealSize(node);
        if (width === node.size[0] && height === node.size[1]) continue;
        if (typeof node.setSize === "function") node.setSize([width, height]);
        else node.size = [width, height];
    }
    return previous;
}

/**
 * Шаг 1.5: согласовать размер с тем, что нарисовано.
 *
 * В классическом режиме делать нечего: там нарисовано ровно `node.size`. В
 * Nodes 2.0 нода не ужимается ниже своего минимума — забираем измеренное.
 *
 * ⚠️ Ноду, которой на экране НЕТ (уехала за край, Vue её не отрисовал), не
 * ужимаем вовсе: мерить нечего, а сжать вслепую — значит получить наложение
 * там, где его никто не увидит до следующей прокрутки.
 */
export function adoptRenderedSizes(canvas, previous) {
    if (!vueNodes()) return;
    for (const [node, before] of previous) {
        const measured = renderedSize(canvas, node);
        const target = measured
            ? [Math.max(measured[0], node.size[0]), Math.max(measured[1], node.size[1])]
            : [Math.max(before[0], node.size[0]), Math.max(before[1], node.size[1])];
        if (target[0] === node.size[0] && target[1] === node.size[1]) continue;
        if (typeof node.setSize === "function") node.setSize(target);
        else node.size = target;
    }
}

/**
 * Состав групп ДО переезда, от самой тесной к самой просторной.
 *
 * Порядок важен для подгонки рамок: внешняя должна считаться после того, как
 * вложенная уже встала на новое место.
 */
export function snapshotGroups(graph) {
    const groups = [];
    for (const group of graph?.groups || []) {
        try {
            group.recomputeInsideNodes();
        } catch {
            continue;
        }
        groups.push({ group, nodes: new Set(group._nodes || []) });
    }
    return groups.sort((a, b) => (
        (a.group.size[0] * a.group.size[1]) - (b.group.size[0] * b.group.size[1])
    ));
}

/** Связи между ключами блоков; повторы и петли выброшены. */
export function edgesBetween(graph, nodes, keyOf) {
    const inside = new Set(nodes);
    const seen = new Set();
    const edges = [];
    for (const node of nodes) {
        for (const input of node.inputs || []) {
            if (input?.link === null || input?.link === undefined) continue;
            const link = linkById(graph, input.link);
            if (!link) continue;
            const origin = graph.getNodeById?.(link.origin_id);
            if (!origin || !inside.has(origin)) continue;
            const from = keyOf(origin);
            const to = keyOf(node);
            if (!from || !to || from === to) continue;
            const mark = `${from} ${to}`;
            if (seen.has(mark)) continue;
            seen.add(mark);
            edges.push({ from, to });
        }
    }
    return edges;
}

/**
 * Шаг 2: посчитать расстановку по НАСТОЯЩИМ размерам. Ничего не двигает.
 *
 * @returns {{moves: Array<{node:object,x:number,y:number}>,
 *            frames: Array<{group:object,x:number,y:number,width:number,height:number}>,
 *            count: number}}
 */
export function planPositions(canvas, options = {}) {
    const graph = canvas?.graph;
    const targets = collectTargets(canvas);
    if (!graph || targets.length < 2) return { moves: [], frames: [], count: 0 };

    const { columnGap, rowGap, grid = 10, packed = false, aspect = 0 } = options;
    const layoutOptions = { columnGap, rowGap, grid, packed, aspect };
    const size = new Map(targets.map((node) => [node, boxOf(node)]));

    // Кто в какой группе. Нода может попасть в несколько (вложенность) — держит
    // её самая тесная.
    const homeGroup = new Map();
    for (const { group, nodes } of snapshotGroups(graph)) {
        if (group.pinned) continue;
        for (const node of nodes) if (!homeGroup.has(node)) homeGroup.set(node, group);
    }

    const inGroup = new Map();
    const free = [];
    for (const node of targets) {
        const group = homeGroup.get(node);
        if (!group) {
            free.push(node);
            continue;
        }
        if (!inGroup.has(group)) inGroup.set(group, []);
        inGroup.get(group).push(node);
    }

    const groupKey = new Map([...inGroup.keys()].map((group, index) => [group, `g${index}`]));
    const keyOf = (node) => {
        const group = homeGroup.get(node);
        return group && groupKey.has(group) ? groupKey.get(group) : `n${node.id}`;
    };

    // Внутренняя раскладка каждой группы.
    const innerPlaces = new Map();
    const blockSize = new Map();
    for (const [group, nodes] of inGroup) {
        const blocks = nodes.map((node) => ({
            key: `n${node.id}`,
            kind: String(node.type || ""),
            width: size.get(node)[0],
            height: size.get(node)[1],
            y: node.pos[1],
        }));
        const inner = layoutBlocks(
            blocks, edgesBetween(graph, nodes, (n) => `n${n.id}`), layoutOptions);
        const byKey = new Map(nodes.map((node) => [`n${node.id}`, node]));
        innerPlaces.set(group, inner.places.map((place) => ({
            node: byKey.get(place.key), x: place.x, y: place.y,
        })));
        const title = group.titleHeight || 24;
        blockSize.set(groupKey.get(group), [
            inner.width + GROUP_PADDING * 2,
            inner.height + GROUP_PADDING * 2 + title,
        ]);
    }

    // Внешняя раскладка: группы одним блоком, одиночки — сами собой.
    const outerBlocks = [];
    for (const [group, nodes] of inGroup) {
        const [width, height] = blockSize.get(groupKey.get(group));
        outerBlocks.push({ key: groupKey.get(group), width, height, y: group.pos[1] });
        void nodes;
    }
    for (const node of free) {
        outerBlocks.push({
            key: `n${node.id}`,
            kind: String(node.type || ""),
            width: size.get(node)[0],
            height: size.get(node)[1],
            y: node.pos[1],
        });
    }
    const outer = layoutBlocks(outerBlocks, edgesBetween(graph, targets, keyOf), layoutOptions);

    // Точка отсчёта — левый верхний угол того, что было: команда наводит порядок
    // НА МЕСТЕ, а не увозит схему в начало координат, где её потом искать.
    const originX = Math.min(...targets.map((node) => node.pos[0]));
    const originY = Math.min(...targets.map((node) => node.pos[1] - NODE_TITLE));
    const baseX = snap(originX, grid);
    const baseY = snap(originY, grid);

    const moves = [];
    const frames = [];
    const placeOf = new Map(outer.places.map((place) => [place.key, place]));

    for (const [group, nodes] of inGroup) {
        const place = placeOf.get(groupKey.get(group));
        if (!place) continue;
        const [width, height] = blockSize.get(groupKey.get(group));
        const title = group.titleHeight || 24;
        const left = baseX + place.x;
        const top = baseY + place.y;
        frames.push({ group, x: left, y: top, width, height });
        for (const item of innerPlaces.get(group)) {
            moves.push({
                node: item.node,
                // ⚠️ На сетку — ЗДЕСЬ, а не только в раскладке. Внутри группы к
                // координате прибавляются отступ рамки и высота заголовка, и
                // сумма легко перестаёт делиться на шаг сетки: ноды в группах
                // вставали на 6 пикселей мимо, а по холсту это видно сразу.
                x: snap(left + GROUP_PADDING + item.x, grid),
                // `y` блока — верх заголовка ноды, а `pos` — верх её тела.
                y: snap(top + title + GROUP_PADDING + item.y, grid) + NODE_TITLE,
            });
        }
        void nodes;
    }
    for (const node of free) {
        const place = placeOf.get(`n${node.id}`);
        if (!place) continue;
        moves.push({
            node,
            x: snap(baseX + place.x, grid),
            y: snap(baseY + place.y, grid) + NODE_TITLE,
        });
    }

    return { moves, frames, count: moves.length };
}

/** Применить расстановку. Возвращает число переставленных нод. */
export function applyPositions(app, plan) {
    for (const { node, x, y } of plan.moves) {
        // ⚠️ Именно setPos: так двигает ноды и родное выравнивание ComfyUI, и
        // только через него позиция доезжает до рендерера Nodes 2.0.
        if (typeof node.setPos === "function") node.setPos(x, y);
        else node.pos = [x, y];
    }
    for (const { group, x, y, width, height } of plan.frames) {
        if (group.pinned) continue;
        group.pos[0] = x;
        group.pos[1] = y;
        group.size[0] = width;
        group.size[1] = height;
    }
    for (const { group } of plan.frames) {
        try {
            group.recomputeInsideNodes();
        } catch {
            /* группа без графа — не наша забота */
        }
    }
    app?.graph?.change?.();
    app?.canvas?.setDirty?.(true, true);
    return plan.count;
}
