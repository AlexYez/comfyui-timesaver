// Разводка проводов: расставить точки так, чтобы связь не резала чужие ноды.
//
// Раскладка по колонкам делает поток читаемым, но провод от первой колонки к
// четвёртой всё равно идёт по прямой — сквозь всё, что стоит между ними. Здесь
// такие провода уводятся в КОРИДОРЫ: вертикальные промежутки между колонками
// свободны по построению, и связь ведётся от коридора к коридору, а поперёк
// колонки — по свободной полосе между её нодами.
//
// Что делается:
//   1. Колонки восстанавливаются по итоговым позициям нод (левый край).
//   2. Для каждой связи проверяется, режет ли её прямая чужую ноду.
//   3. Если режет — в каждом пройденном коридоре ставится точка, а высота
//      выбирается по свободной полосе той колонки, которую предстоит пересечь.
//
// Чего НЕ делается:
//   * связи, у которых точки уже есть, не трогаются — это хозяйство человека,
//     его только выравнивают (`_tidy_reroutes.js`);
//   * провода, идущие ПРОТИВ потока (приёмник левее источника), оставляются как
//     есть: их правильный обход — отдельная задача, и делать её наполовину
//     хуже, чем не делать.
//
// ⚠️ Точки создаются В ПОРЯДКЕ ВЫЗОВА, а не по своим координатам (замерено:
// созданная первой дальняя точка встала в цепочке первой, и провод завязался
// узлом). Поэтому список точек обязан идти от источника к приёмнику.

// Насколько провод обходит ноду и края свободной полосы.
const MARGIN = 24;
// Через сколько идут соседние полосы. Меньше — линии сливаются в одну, больше —
// схема раздувается по высоте.
const LANE_STEP = 20;
// Колонки восстанавливаются по левому краю; после раскладки он общий, но пара
// пикселей расхождения не должна разбивать колонку надвое.
const COLUMN_TOLERANCE = 4;
// Во что обходится вариант разводки. Подобрано замером на настоящем workflow
// владельца (32 ноды, 47 связей), а не на глаз.
//
// ⚠️ НАЛОЖЕНИЕ — ПОЧТИ ЗАПРЕТ. При цене 150 разводка соглашалась положить
// провод на провод, лишь бы не пересечь пяток чужих: замерено 67 пересечений и
// 3 наложения. При 500 наложений НОЛЬ ценой 81 пересечения — и это правильный
// размен: пересечение видно и понятно, а два провода в одной линии не
// различить вовсе («сколько их там» — жалоба владельца).
export const WEIGHTS = { node: 1000, stack: 500, cross: 35, detour: 30, laneStep: 20 };

function snap(value, grid) {
    if (!(grid > 0)) return Math.round(value);
    return Math.round(value / grid) * grid;
}

/** Прямоугольники нод вместе с заголовками. */
export function nodeBoxes(nodes, titleHeight = 30) {
    return (nodes || []).map((node) => boxOf(node, titleHeight));
}

/** Прямоугольник ноды вместе с заголовком. */
function boxOf(node, titleHeight) {
    const [x, y] = node.pos;
    const width = node.flags?.collapsed ? (node._collapsed_width || 80) : node.size[0];
    const height = node.flags?.collapsed ? 0 : node.size[1];
    return { node, left: x, top: y - titleHeight, right: x + width, bottom: y + height };
}

/** Пересекает ли отрезок прямоугольник (алгоритм Лианга—Барски). */
export function segmentHitsBox(x1, y1, x2, y2, box) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    let enter = 0;
    let leave = 1;
    const edges = [
        [-dx, x1 - box.left], [dx, box.right - x1],
        [-dy, y1 - box.top], [dy, box.bottom - y1],
    ];
    for (const [p, q] of edges) {
        if (p === 0) {
            if (q < 0) return false;                 // параллельно и снаружи
            continue;
        }
        const t = q / p;
        if (p < 0) {
            if (t > leave) return false;
            if (t > enter) enter = t;
        } else {
            if (t < enter) return false;
            if (t < leave) leave = t;
        }
    }
    return enter < leave;
}

/** Колонки по левому краю нод, слева направо. */
export function columnsOf(boxes) {
    const sorted = boxes.slice().sort((a, b) => a.left - b.left);
    const columns = [];
    for (const box of sorted) {
        const last = columns[columns.length - 1];
        if (last && Math.abs(box.left - last.left) <= COLUMN_TOLERANCE) {
            last.boxes.push(box);
            last.right = Math.max(last.right, box.right);
            continue;
        }
        columns.push({ left: box.left, right: box.right, boxes: [box] });
    }
    return columns;
}

/**
 * Свободные высоты рядом с желаемой — НЕСКОЛЬКО вариантов, ближние первыми.
 *
 * Одной полосы мало: по одному и тому же коридору идут несколько проводов, и
 * все они выбирают одну и ту же «ближайшую свободную» высоту — на схеме это
 * видно как одна линия вместо трёх (живое замечание). Варианты дают разводке
 * из чего выбирать, а выбирает она по цене — см. `scorePath`.
 */
export function laneCandidates(columns, wantY, skip = [], laneStep = LANE_STEP) {
    const ignore = new Set(skip);
    const list = Array.isArray(columns) ? columns : [columns];
    const taken = list
        .flatMap((column) => column.boxes)
        .filter((box) => !ignore.has(box.node))
        .map((box) => [box.top - MARGIN, box.bottom + MARGIN])
        .sort((a, b) => a[0] - b[0]);
    if (!taken.length) return [wantY];

    const spans = [];
    for (let index = 0; index < taken.length - 1; index += 1) {
        const top = taken[index][1];
        const bottom = taken[index + 1][0];
        if (bottom - top >= MARGIN) spans.push([top, bottom]);
    }
    // Над всеми и под всеми — там места сколько угодно.
    spans.push([taken[0][0] - laneStep * 8, taken[0][0]]);
    spans.push([taken[taken.length - 1][1], taken[taken.length - 1][1] + laneStep * 8]);

    const options = [];
    for (const [top, bottom] of spans) {
        const middle = (top + bottom) / 2;
        for (let step = 0; step <= 6; step += 1) {
            for (const y of step ? [middle - step * laneStep, middle + step * laneStep] : [middle]) {
                if (y >= top && y <= bottom) options.push(y);
            }
        }
    }
    if (!taken.some(([top, bottom]) => wantY > top && wantY < bottom)) options.unshift(wantY);
    return [...new Set(options)].sort((a, b) => Math.abs(a - wantY) - Math.abs(b - wantY));
}

/** Пересекаются ли отрезки (строго, без касаний). */
function segmentsCross(a1, a2, b1, b2) {
    const side = (p, q, r) => (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0]);
    const d1 = side(a1, a2, b1);
    const d2 = side(a1, a2, b2);
    const d3 = side(b1, b2, a1);
    const d4 = side(b1, b2, a2);
    return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0))
        && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
}

/**
 * Лежат ли два отрезка друг на друге.
 *
 * ⚠️ ЛЮБОГО НАКЛОНА, а не только по осям. Первая версия сравнивала лишь строго
 * горизонтальные и строго вертикальные отрезки — и не видела трёх косых
 * проводов, идущих вплотную из одной точки (живая жалоба: «наложены три
 * пайпа»). Здесь считается честно: направления почти совпадают, расстояние
 * между линиями меньше шага полос, и есть общий кусок длины.
 */
function segmentsStack(a1, a2, b1, b2, laneStep = LANE_STEP) {
    const ax = a2[0] - a1[0];
    const ay = a2[1] - a1[1];
    const bx = b2[0] - b1[0];
    const by = b2[1] - b1[1];
    const aLength = Math.hypot(ax, ay);
    const bLength = Math.hypot(bx, by);
    if (aLength < 1 || bLength < 1) return false;

    const ux = ax / aLength;
    const uy = ay / aLength;
    const vx = bx / bLength;
    const vy = by / bLength;
    // Синус угла между направлениями: 0.2 — это примерно 11 градусов. Провода,
    // расходящиеся сильнее, читаются как разные и без нашей помощи.
    if (Math.abs(ux * vy - uy * vx) > 0.2) return false;

    // Расстояние от концов второго отрезка до линии первого.
    const away = (point) => Math.abs((point[0] - a1[0]) * uy - (point[1] - a1[1]) * ux);
    if (away(b1) > laneStep || away(b2) > laneStep) return false;

    // Общий кусок вдоль первого отрезка.
    const along = (point) => (point[0] - a1[0]) * ux + (point[1] - a1[1]) * uy;
    const start = Math.max(0, Math.min(along(b1), along(b2)));
    const end = Math.min(aLength, Math.max(along(b1), along(b2)));
    return end - start > MARGIN;
}

/**
 * Сколько точек чужих проводов стоят вплотную к нашим.
 *
 * ⚠️ Отдельно от отрезков. Две точки могут совпасть, а отрезки вокруг них —
 * разойтись веером: формально наложения нет, а на схеме из одного кружка растут
 * три провода и не понять, куда какой идёт.
 */
function crowdedDots(points, placed, laneStep) {
    let crowd = 0;
    for (const point of points) {
        for (const other of placed) {
            // Концы чужого пути — это гнёзда нод, они и должны быть общими.
            for (let index = 1; index < other.points.length - 1; index += 1) {
                const dot = other.points[index];
                if (Math.hypot(dot[0] - point[0], dot[1] - point[1]) < laneStep) crowd += 1;
            }
        }
    }
    return crowd;
}

/**
 * Цена варианта разводки: чем меньше, тем лучше.
 *
 * Складывается из того, на что жалуется глаз: провод лёг ПОВЕРХ другого (хуже
 * всего — две линии читаются как одна), провод пересёк другой, и длина крюка.
 * Пересечение с нодой стоит запретительно дорого: такой вариант не берём вовсе.
 */
export function scorePath(points, from, to, placed, boxes, origin, target,
                          source = null, weights = WEIGHTS) {
    const path = [from, ...points, to];
    // Крюк считается от ОБОИХ концов: полоса по ходу провода дешевле той,
    // что уводит его вбок. Заодно параллельные провода расходятся в том же
    // порядке, в каком стоят их приёмники, — и меньше пересекаются.
    const lane = points.length ? points[Math.floor(points.length / 2)][1] : from[1];
    let cost = (Math.abs(lane - from[1]) + Math.abs(lane - to[1])) / weights.detour;
    cost += crowdedDots(points, placed, weights.laneStep ?? LANE_STEP) * weights.stack;
    for (let index = 0; index < path.length - 1; index += 1) {
        const a1 = path[index];
        const a2 = path[index + 1];
        for (const box of boxes) {
            if (box.node === origin || box.node === target) continue;
            if (segmentHitsBox(a1[0], a1[1], a2[0], a2[1], box)) cost += weights.node;
        }
        for (const other of placed) {
            // ⚠️ НИКАКИХ ИСКЛЮЧЕНИЙ. Была поблажка проводам из одного гнезда:
            // мол, пучок читается как одна линия. Он и читается — как ОДНА, и
            // сколько их там на самом деле, по схеме не понять. Владелец назвал
            // это недопустимым, и он прав: наложение штрафуется всегда.
            for (let step = 0; step < other.points.length - 1; step += 1) {
                const b1 = other.points[step];
                const b2 = other.points[step + 1];
                if (segmentsStack(a1, a2, b1, b2, weights.laneStep ?? LANE_STEP)) {
                    cost += weights.stack;
                }
                else if (segmentsCross(a1, a2, b1, b2)) cost += weights.cross;
            }
        }
    }
    return cost;
}

/** Режет ли путь (ломаная) чужие ноды. Свои концы не в счёт. */
export function pathBlocked(points, boxes, origin, target) {
    for (let index = 0; index < points.length - 1; index += 1) {
        const [x1, y1] = points[index];
        const [x2, y2] = points[index + 1];
        for (const box of boxes) {
            if (box.node === origin || box.node === target) continue;
            if (segmentHitsBox(x1, y1, x2, y2, box)) return true;
        }
    }
    return false;
}

/**
 * Выбросить точки, без которых провод всё равно никого не режет.
 *
 * ⚠️ Это правило заменяет собой ворох частных случаев. Живые замечания были
 * ровно про лишние точки: ступенька там, где хватало одной прямой, и угол
 * прямо перед входом, куда провод и так заходил свободно. Проверять каждую
 * точку на «а нужна ли она» дешевле и честнее, чем угадывать, где её не
 * ставить.
 */
export function prunePath(candidate, from, to, boxes, origin, target, costOf = null) {
    const path = [];
    for (const point of candidate) {
        const last = path[path.length - 1];
        if (last && Math.abs(last[0] - point[0]) < 1 && Math.abs(last[1] - point[1]) < 1) continue;
        path.push(point);
    }
    let result = path;
    let index = 0;
    while (index < result.length) {
        const trial = result.slice(0, index).concat(result.slice(index + 1));
        // ⚠️ Точка лишняя, только если БЕЗ НЕЁ НЕ ХУЖЕ. Первая версия смотрела
        // лишь на ноды — и сносила весь обход у провода, который уводили не от
        // ноды, а от другого провода: он тут же ложился обратно.
        const free = !pathBlocked([from, ...trial, to], boxes, origin, target);
        const notWorse = !costOf || costOf(trial) <= costOf(result);
        if (free && notWorse) result = trial;
        else index += 1;
    }
    return result;
}

/** Ссылки графа списком: Map в новых сборках, объект в старых. */
function allLinks(graph) {
    const links = graph?.links;
    if (!links) return [];
    return typeof links.values === "function" ? [...links.values()] : Object.values(links);
}

/** Координаты гнезда. */
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

/**
 * Посчитать, где нужны новые точки. Ничего не создаёт.
 *
 * @param {object} graph граф LiteGraph
 * @param {object[]} nodes ноды, которые считаются «своими» (раскладка их знает)
 * @param {object} [options] `grid`, `titleHeight`
 * @returns {Array<{link: object, points: Array<[number, number]>}>}
 */
export function planPipes(graph, nodes, options = {}) {
    const grid = options.grid ?? 10;
    const titleHeight = options.titleHeight ?? 30;
    const laneStep = options.laneStep ?? LANE_STEP;
    const weights = { ...WEIGHTS, laneStep, ...(options.weights || {}) };
    if (!graph || !nodes?.length) return [];

    const boxes = nodes.map((node) => boxOf(node, titleHeight));
    const boxOfNode = new Map(boxes.map((box) => [box.node, box]));
    const columns = columnsOf(boxes);
    const columnOf = new Map();
    columns.forEach((column, index) => {
        for (const box of column.boxes) columnOf.set(box.node, index);
    });

    // Провода, о которых разводка обязана помнить. Сначала — все, что идут
    // прямо: они уже нарисованы, и класть новый поверх них нельзя.
    const placed = [];
    const pending = [];
    for (const link of allLinks(graph)) {
        const origin = graph.getNodeById?.(link.origin_id);
        const target = graph.getNodeById?.(link.target_id);
        if (!boxOfNode.has(origin) || !boxOfNode.has(target)) continue;
        const from = slotPos(origin, false, link.origin_slot);
        const to = slotPos(target, true, link.target_slot);
        if (!from || !to) continue;

        const hasDots = link.parentId !== null && link.parentId !== undefined;
        const fromColumn = columnOf.get(origin);
        const toColumn = columnOf.get(target);
        const blocked = boxes.some((box) => (
            box.node !== origin && box.node !== target
            && segmentHitsBox(from[0], from[1], to[0], to[1], box)
        ));
        // Кандидат — связь БЕЗ точек, идущая по потоку. Разводить ли её,
        // решается ниже: там уже видно, кто на ком лежит.
        const source = `${link.origin_id}:${link.origin_slot}`;
        if (hasDots || !(toColumn > fromColumn)) {
            placed.push({ points: [from, to], source });
            continue;
        }
        pending.push({ link, origin, target, from, to, fromColumn, toColumn, source, blocked });
    }

    // ⚠️ ДЛИННЫЕ ПЕРВЫМИ. Разводка жадная: кто раньше, тот и выбирает полосу.
    // Длинному проводу выбор нужнее — коротким проще найти щель.
    pending.sort((a, b) => (b.toColumn - b.fromColumn) - (a.toColumn - a.fromColumn));

    // ⚠️ ПОВОД РАЗВЕСТИ — НЕ ТОЛЬКО ПОРЕЗАННАЯ НОДА. Провод, легший на другой
    // провод, тоже надо уводить: на схеме их не отличить, и сколько их там —
    // непонятно. Раньше такие связи оставались как есть, потому что «ноду ведь
    // не режут», и в итоге под одним проводом лежал второй (жалоба владельца).
    const lyingOnSomebody = (start, end) => placed.some((other) => {
        for (let step = 0; step < other.points.length - 1; step += 1) {
            if (segmentsStack(start, end, other.points[step], other.points[step + 1])) return true;
        }
        return false;
    });

    // Выбрать лучший путь для одной связи. `skip` — её собственный прежний путь:
    // при переразводке он не должен мешать сам себе.
    const bestFor = ({ origin, target, from, to, fromColumn, toColumn, source }, skip = null) => {
        const rivals = skip ? placed.filter((entry) => entry !== skip) : placed;

        // ⚠️ ВЫСОТА МЕНЯЕТСЯ ТОЛЬКО ВНУТРИ КОРИДОРА, углом из двух точек.
        //
        // Первая попытка ставила по одной точке на коридор, и провод шёл в неё
        // наискось — прямо через соседей по своей же колонке. Замерено: было 3
        // пересечения, стало 6, то есть разводка делала ХУЖЕ. Труба идёт так:
        // горизонталь на высоте гнезда до коридора, вертикаль внутри коридора
        // (там пусто по построению), длинная горизонталь по свободной полосе
        // через ВСЕ мешавшие колонки — и такой же угол перед приёмником.
        const leftX = snap((columns[fromColumn].right + columns[fromColumn + 1].left) / 2, grid);
        const rightX = snap((columns[toColumn - 1].right + columns[toColumn].left) / 2, grid);
        const crossed = columns.slice(fromColumn + 1, toColumn);
        const middleY = (from[1] + to[1]) / 2;

        // ⚠️ ВЫБОР, А НЕ ЕДИНСТВЕННЫЙ ОТВЕТ. Раньше бралась «ближайшая
        // свободная» полоса — и все провода одного коридора выбирали её же,
        // ложась друг на друга: три линии читались как одна (живое замечание).
        // Теперь перебираются варианты, и берётся самый дешёвый: цена растёт от
        // наложений, пересечений с чужими проводами и длины крюка.
        // ⚠️ У СОСЕДНИХ КОЛОНОК тоже должен быть выбор. Когда пересекать
        // нечего, вариант был ровно один — и провод, легший на другой, не мог
        // никуда деться: разводка возвращала пустой путь и оставляла всё как
        // есть. В коридоре пусто по построению, так что любая высота там
        // безопасна.
        const lanes = crossed.length
            ? laneCandidates(crossed, middleY, [origin, target], laneStep)
            : [middleY, ...Array.from({ length: 8 }, (_unused, step) => (
                step % 2 ? middleY - laneStep * Math.ceil((step + 1) / 2)
                    : middleY + laneStep * Math.ceil((step + 1) / 2)))];
        let best = null;
        for (const rawLane of lanes.slice(0, 32)) {
            const lane = snap(rawLane, grid);
            const costOf = (candidate) => scorePath(candidate, from, to, rivals, boxes,
                                                    origin, target, source, weights);
            const points = prunePath([
                [leftX, snap(from[1], grid)],
                [leftX, lane],
                [rightX, lane],
                [rightX, snap(to[1], grid)],
            ], from, to, boxes, origin, target, costOf);
            if (!points.length) continue;
            const cost = costOf(points);
            if (!best || cost < best.cost) best = { points, cost };
            if (best.cost === 0) break;                  // дешевле не бывает
        }
        return best;
    };

    const plan = [];
    const routed = [];
    for (const wire of pending) {
        if (!wire.blocked && !lyingOnSomebody(wire.from, wire.to)) {
            placed.push({ points: [wire.from, wire.to], source: wire.source });  // и так читается
            continue;
        }
        const best = bestFor(wire);
        if (!best) continue;
        const entry = { points: [wire.from, ...best.points, wire.to], source: wire.source };
        placed.push(entry);                              // следующий знает о нём
        routed.push({ wire, entry });
        plan.push({ link: wire.link, points: best.points });
    }

    // ⚠️ ВТОРОЙ ПРОХОД — «снять и проложить заново».
    //
    // Разводка жадная: первый провод выбирал полосу, когда остальных ещё не
    // было, и мог занять чужое место. Во втором проходе каждый снимается со
    // схемы и прокладывается снова — теперь уже зная обо ВСЕХ. Дешевле искать
    // так, чем перебирать порядок связей: проход один, а картина считается по
    // тем же ценам.
    for (const item of routed) {
        const best = bestFor(item.wire, item.entry);
        if (!best) continue;
        const before = scorePath(item.entry.points.slice(1, -1), item.wire.from, item.wire.to,
                                 placed.filter((entry) => entry !== item.entry),
                                 boxes, item.wire.origin, item.wire.target,
                                 item.wire.source, weights);
        if (best.cost >= before) continue;                // лучше не стало — не трогаем
        item.entry.points = [item.wire.from, ...best.points, item.wire.to];
        const line = plan.find((row) => row.link === item.wire.link);
        if (line) line.points = best.points;
    }

    // ⚠️ ТРЕТИЙ ПРОХОД — РАСПРЯМИТЬ ЛИШНЕЕ. Провод могли увести из-под другого,
    // а тот потом сам ушёл в сторону: крюк остался ни за чем, а лишний крюк —
    // это лишние пересечения. Проверяем каждый разведённый: если прямая уже
    // никого не режет и ни на ком не лежит, точки убираются.
    const straightened = new Set();
    for (const item of routed) {
        const { from, to, origin, target } = item.wire;
        const rivals = placed.filter((entry) => entry !== item.entry);
        const cutsNode = boxes.some((box) => (
            box.node !== origin && box.node !== target
            && segmentHitsBox(from[0], from[1], to[0], to[1], box)
        ));
        if (cutsNode) continue;
        const lies = rivals.some((other) => {
            for (let step = 0; step < other.points.length - 1; step += 1) {
                if (segmentsStack(from, to, other.points[step], other.points[step + 1], laneStep)) {
                    return true;
                }
            }
            return false;
        });
        if (lies) continue;
        item.entry.points = [from, to];
        straightened.add(item.wire.link);
    }
    return plan.filter((row) => !straightened.has(row.link));
}

/** Создать точки. Возвращает, сколько их получилось. */
export function applyPipes(graph, plan) {
    let made = 0;
    for (const { link, points } of plan) {
        for (const [x, y] of points) {
            try {
                // ⚠️ По порядку от источника: цепочка строится по порядку
                // вызовов, а не по координатам.
                graph.createReroute([x, y], link);
                made += 1;
            } catch (error) {
                console.warn("[TS TidyLayout] could not add a link dot", error);
            }
        }
    }
    return made;
}
