// Точки-роуты на линиях: поставить их на прямую между концами провода.
//
// Речь о НОВОМ механизме ComfyUI — маленьких круглых точках прямо на связи
// (`graph.reroutes`), а не о старой ноде `Reroute`. Нода — это нода: её
// раскладывает общая раскладка, наравне со всеми.
//
// Что делает выравнивание: провод, шедший ломаной через случайно брошенную
// точку, становится прямым. Точки распределяются по отрезку между гнездом-
// источником и гнездом-приёмником, по своему месту в цепочке, и садятся на
// сетку.
//
// Чего оно НЕ делает: не создаёт и не удаляет точки. Граф после команды тот же.
//
// ⚠️ ЦЕПОЧКА СЧИТАЕТСЯ ПО СВЯЗИ, а не по самой точке. `reroute.getReroutes()`
// отдаёт цепочку ОТ ИСТОЧНИКА ДО СЕБЯ, и каждая точка считала себя последней:
// две точки на одном проводе вставали на 1/2 и 2/3 вместо честных 1/3 и 2/3
// (замерено). Правильный путь — от `link.parentId` вверх по родителям.

/** Округлить к сетке. Шага нет — просто целое. */
function snap(value, grid) {
    if (!(grid > 0)) return Math.round(value);
    return Math.round(value / grid) * grid;
}

/** Координаты гнезда: `getConnectionPos(вход?, номер)`. */
function slotPos(node, isInput, index) {
    if (!node || !(index >= 0)) return null;
    try {
        const point = node.getConnectionPos(isInput, index);
        const x = Number(point?.[0]);
        const y = Number(point?.[1]);
        return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
    } catch {
        return null;
    }
}

/** Все связи графа списком: в новых сборках это Map, в старых — объект. */
function allLinks(graph) {
    const links = graph?.links;
    if (!links) return [];
    if (typeof links.values === "function") return [...links.values()];
    return Object.values(links);
}

/** Цепочка точек этой связи, по порядку от источника к приёмнику. */
function chainOf(graph, link) {
    const reroutes = graph?.reroutes;
    const chain = [];
    const seen = new Set();
    let id = link?.parentId;
    while (id !== null && id !== undefined && !seen.has(id)) {
        seen.add(id);
        const reroute = reroutes?.get?.(id);
        if (!reroute) break;
        chain.unshift(reroute);
        id = reroute.parentId;
    }
    return chain;
}

/**
 * Посчитать, где должны стоять точки. Ничего не двигает.
 *
 * Точку могут делить несколько связей (один выход раздаёт её нескольким
 * входам) — тогда она встаёт в среднее из того, что просит каждая: провод
 * одинаково честен ко всем.
 *
 * ⚠️ Провод, который в обход идёт НЕ ЗРЯ, выпрямлять нельзя. Если прямая между
 * гнёздами режет чужую ноду, точки этой связи остаются как есть: их поставили,
 * чтобы обойти препятствие, — своё или чужой командой разводки. Без этой
 * оговорки обычная раскладка, запущенная после разводки, распрямляла провода
 * обратно и возвращала все пересечения (замерено: 0 → 7).
 *
 * @param {object} graph граф LiteGraph
 * @param {object} [options] `grid` — шаг сетки; `blocked(link, start, end)` —
 *   вернуть true, если прямая этой связи кого-то режет
 * @returns {Array<{reroute: object, x: number, y: number}>}
 */
export function planReroutes(graph, options = {}) {
    const grid = options.grid ?? 10;
    const blocked = typeof options.blocked === "function" ? options.blocked : null;
    if (!graph?.reroutes) return [];

    // точка -> [сумма X, сумма Y, сколько связей попросили]
    const wanted = new Map();

    for (const link of allLinks(graph)) {
        const chain = chainOf(graph, link);
        if (!chain.length) continue;

        const origin = graph.getNodeById?.(link.origin_id);
        const target = graph.getNodeById?.(link.target_id);
        const start = slotPos(origin, false, link.origin_slot);
        const end = slotPos(target, true, link.target_slot);
        if (!start || !end) continue;
        if (blocked && blocked(link, start, end)) continue;   // обход не зря

        chain.forEach((reroute, index) => {
            const share = (index + 1) / (chain.length + 1);
            const x = start[0] + (end[0] - start[0]) * share;
            const y = start[1] + (end[1] - start[1]) * share;
            const sum = wanted.get(reroute) || [0, 0, 0];
            wanted.set(reroute, [sum[0] + x, sum[1] + y, sum[2] + 1]);
        });
    }

    const plan = [];
    for (const [reroute, [sumX, sumY, count]] of wanted) {
        if (!count) continue;
        plan.push({
            reroute,
            x: snap(sumX / count, grid),
            y: snap(sumY / count, grid),
        });
    }
    return plan;
}

/** Применить план. Возвращает число сдвинутых точек. */
export function applyReroutes(plan) {
    let moved = 0;
    for (const { reroute, x, y } of plan) {
        const [wasX, wasY] = reroute.pos || [];
        if (Math.round(wasX) === x && Math.round(wasY) === y) continue;
        reroute.pos = [x, y];
        moved += 1;
    }
    return moved;
}
