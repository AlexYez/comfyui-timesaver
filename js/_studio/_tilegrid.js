// TS Studio kit — как выглядит апскейл, пока он идёт (ui-kit layer).
//
// Увеличение считается кусками. Быстрый декодер латента честно отдаёт эти
// куски по одному, и на экране мелькали обрывки в случайных местах: понять по
// ним нельзя было ни сколько осталось, ни что вообще происходит.
//
// Здесь то же самое показано так, как оно устроено: поверх картинки лежит
// сетка ровно из тех клеток, на которые кадр реально разрезан, и клетки гаснут
// по мере готовности.
//
// ОТКУДА БЕРЁТСЯ ЧИСЛО КЛЕТОК — главный вопрос, и ответ разный:
//
//   свой резчик   `TS_ImageTileSplitter` режет кадр САМ, до сэмплера, и наружу
//                 про тайлы не сообщает ничего: в событиях виден только ход
//                 сэмплера. Зато его арифметика известна (tile_width, overlap),
//                 поэтому сетка считается заранее и точно — той же формулой,
//                 что в `nodes/image/ts_image_tile_splitter.py`.
//   тайловый VAE  `VAEEncodeTiled` / `VAEDecodeTiled` отчитываются за каждый
//                 тайл через ProgressBar, и число приходит в событии как `max`.
//
// Первый случай — это большинство апскейлов пака. Пока сетка умела только
// второй, у владельца она не появлялась вовсе: измерение событий на прогоне
// Klein показало три события `progress` от одного узла `sample` и ни одного
// про тайлы.
//
// ПОДГОТОВКА. Между нажатием и первым событием проходит долгая тишина: грузятся
// модель, текстовый энкодер, VAE. Поэтому сетка появляется сразу по нажатию и
// в это время «расчерчивает» кадр — волна пробегает по клеткам. Это не
// украшение: пустой экран на минуту читается как «ничего не работает».
//
// Сетка — DOM, а не канвас: клеток десятки, анимация достаётся от браузера
// даром, и ни один кадр не стоит перерисовки.

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";

const STYLE_ID = "ts-studio-tilegrid-styles";

export function ensureTileGridStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-tiles{position:absolute;inset:0;display:none;pointer-events:none;z-index:4}
.ts-tiles.is-active{display:grid}
.ts-tiles__cell{position:relative;border:1px solid var(--ts-accent-line);
    background:var(--ts-scrim);opacity:1;
    transition:opacity .45s ease,background .45s ease}
/* Готовый тайл гаснет и открывает картинку под собой — заполнение читается как
   проявление, а не как мигание. */
.ts-tiles__cell.is-done{opacity:0;background:transparent;border-color:transparent}
/* Тайл, который считается прямо сейчас: тонкая пульсация, чтобы взгляд знал,
   куда смотреть. */
.ts-tiles__cell.is-live{background:var(--ts-accent-soft);
    animation:ts-tiles-pulse 1.1s ease-in-out infinite}
@keyframes ts-tiles-pulse{0%,100%{opacity:.85}50%{opacity:.45}}
/* Подготовка: кадр расчерчивается волной по диагонали, и волна не
   останавливается, пока грузятся модели. Задержка у каждой клетки своя —
   отсюда ощущение, что сетку кладут на картинку, а не показывают разом. */
.ts-tiles.is-preparing .ts-tiles__cell{
    animation:ts-tiles-draw .55s ease-out backwards,
              ts-tiles-sweep 2.4s ease-in-out infinite;
    animation-delay:var(--ts-wave),calc(var(--ts-wave) + .55s)}
@keyframes ts-tiles-draw{
    from{opacity:0;transform:scale(.82)}
    to{opacity:1;transform:none}}
@keyframes ts-tiles-sweep{
    0%,70%,100%{background:var(--ts-scrim)}
    18%{background:var(--ts-accent-soft)}}
/* Ожидание, когда число тайлов ещё неизвестно. Рисовать в это время сетку
   «примерно» нельзя: клетки — это обещание, где пойдёт работа, и обещание
   должно быть верным. Поэтому просто мягкая волна по кадру. */
.ts-tiles.is-warming{display:block;background:var(--ts-scrim);
    animation:ts-tiles-warm 1.8s ease-in-out infinite}
@keyframes ts-tiles-warm{0%,100%{opacity:.55}50%{opacity:.28}}
/* Превью тайла живёт в СВОЕЙ клетке и вписано в неё: пока он считается —
   это и есть его место на картинке. Раньше тот же кусок растягивался во весь
   экран, и кадр мельтешил обрывками. */
.ts-tiles__cell img{position:absolute;inset:0;width:100%;height:100%;
    object-fit:cover;display:block;opacity:0;transition:opacity .25s ease}
.ts-tiles__cell.has-image img{opacity:1}
/* Клетка с картинкой не гаснет: посчитанные тайлы остаются на экране, и кадр
   собирается на глазах — ровно в том порядке, в каком его считают. */
.ts-tiles__cell.has-image{background:transparent;border-color:var(--ts-accent-line)}
.ts-tiles__cell.has-image.is-done{opacity:1;border-color:transparent}
.ts-tiles__cell.has-image.is-live{animation:none;background:transparent}
`;
    document.head.appendChild(style);
}

/**
 * Разложить N тайлов на строки и столбцы под форму картинки.
 *
 * Нужно только там, где наружу приходит одно число (тайловый VAE). Свой резчик
 * даёт настоящие строки и столбцы — их и надо брать, а не угадывать.
 *
 * @param {number} count сколько тайлов
 * @param {number} aspect ширина/высота картинки
 * @returns {{cols: number, rows: number}}
 */
export function tileLayout(count, aspect) {
    const total = Math.max(1, Math.round(count));
    const ratio = aspect > 0 ? aspect : 1;
    let best = { cols: total, rows: 1 };
    let bestError = Infinity;
    for (let cols = 1; cols <= total; cols += 1) {
        const rows = Math.ceil(total / cols);
        // Насколько решётка похожа на кадр — и насколько мало в ней пустых мест.
        const error = Math.abs(Math.log((cols / rows) / ratio)) + (cols * rows - total) * 0.05;
        if (error < bestError) {
            bestError = error;
            best = { cols, rows };
        }
    }
    return best;
}

/**
 * Сколько клеток нарежет `TS_ImageTileSplitter` — та же арифметика, что в ноде.
 *
 * Повторение формулы здесь осознанное: наружу нода про свою сетку не сообщает
 * ничего, а показать её надо ДО того, как она отработает. Формула короткая и
 * меняется примерно никогда; расхождение ловит тест, который считает обе.
 *
 * @param {{width:number,height:number,tileWidth:number,tileHeight:number,overlap:number}} plan
 * @returns {{cols:number,rows:number}}
 */
export function splitterGrid(plan) {
    const width = Math.max(1, Math.round(plan?.width || 0));
    const height = Math.max(1, Math.round(plan?.height || 0));
    const tileWidth = Math.max(1, Math.min(Math.round(plan?.tileWidth || width), width));
    const tileHeight = Math.max(1, Math.min(Math.round(plan?.tileHeight || height), height));
    const overlap = Math.max(0, Math.min(Math.round(plan?.overlap || 0),
        Math.max(0, Math.min(tileWidth - 1, tileHeight - 1))));
    const strideW = Math.max(1, tileWidth - overlap);
    const strideH = Math.max(1, tileHeight - overlap);
    return {
        cols: Math.max(1, Math.ceil((width - overlap) / strideW)),
        rows: Math.max(1, Math.ceil((height - overlap) / strideH)),
    };
}

/**
 * @returns {{element, prepare, showByCount, showByGrid, place, advance,
 *            advanceFraction, hide, isActive, isPreparing, size}}
 */
export function createTileGrid() {
    ensureTileGridStyles();
    const element = document.createElement("div");
    element.className = `${TS_UI_CLASS} ts-tiles`;

    let cells = [];
    let total = 0;
    let shape = { cols: 0, rows: 0 };

    function build(cols, rows) {
        if (shape.cols === cols && shape.rows === rows && cells.length) return;
        shape = { cols, rows };
        element.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        element.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
        element.replaceChildren();
        cells = [];
        for (let row = 0; row < rows; row += 1) {
            for (let col = 0; col < cols; col += 1) {
                const cell = document.createElement("div");
                cell.className = "ts-tiles__cell";
                // Волна идёт по диагонали: задержка растёт от левого верхнего
                // угла, как если бы сетку рисовали от руки.
                cell.style.setProperty("--ts-wave", `${(row + col) * 90}ms`);
                element.appendChild(cell);
                cells.push(cell);
            }
        }
        total = cols * rows;
    }

    function place(area, host) {
        if (!host || !area || !(area.height > 0)) return;
        // Сетка ложится РОВНО на картинку, а не на всю сцену: тайлы считаются
        // по кадру, и клетки на пустом фоне врали бы о том, где идёт работа.
        element.style.left = `${area.left - host.left}px`;
        element.style.top = `${area.top - host.top}px`;
        element.style.width = `${area.width}px`;
        element.style.height = `${area.height}px`;
        element.style.right = "auto";
        element.style.bottom = "auto";
    }

    // Адреса превью держим сами: без revoke каждый шаг каждого тайла оставлял
    // бы в памяти по картинке, а их за прогон сотни.
    let cellUrls = [];

    function dropUrls() {
        for (const url of cellUrls) {
            if (url) URL.revokeObjectURL(url);
        }
        cellUrls = [];
    }

    function reset() {
        dropUrls();
        for (const cell of cells) {
            cell.classList.remove("is-done", "is-live", "has-image");
            cell.replaceChildren();
        }
    }

    return {
        element,
        /**
         * Показать сетку ДО начала работы: кадр расчерчивается и ждёт.
         *
         * @param {{cols:number,rows:number}} grid
         */
        prepare(grid, area, host) {
            if (!grid || !(grid.cols > 0) || !(grid.rows > 0)) return false;
            if (grid.cols * grid.rows < 2) return false;   // одна клетка — не сетка
            build(grid.cols, grid.rows);
            reset();
            place(area, host);
            element.classList.add("is-active", "is-preparing");
            element.classList.remove("is-warming");
            return true;
        },
        /**
         * Ожидание без сетки: движок ещё не сказал, на сколько кусков режет.
         *
         * Честнее, чем нарисовать сетку наугад: клетка — это обещание места,
         * где пойдёт работа, и промахнувшееся обещание хуже его отсутствия.
         */
        warm(area, host) {
            if (!area || !(area.height > 0)) return false;
            element.replaceChildren();
            cells = [];
            total = 0;
            shape = { cols: 0, rows: 0 };
            place(area, host);
            element.classList.add("is-active", "is-warming");
            element.classList.remove("is-preparing");
            return true;
        },
        /** Сетка по числу тайлов — когда движок сообщает только количество. */
        showByCount(count, area, host) {
            if (!(count > 1) || !area || !(area.height > 0)) return false;
            const grid = tileLayout(count, area.width / area.height);
            build(grid.cols, grid.rows);
            place(area, host);
            element.classList.add("is-active");
            element.classList.remove("is-preparing", "is-warming");
            return true;
        },
        /** Сетка по известной геометрии — когда её посчитал резчик. */
        showByGrid(grid, area, host) {
            if (!grid || grid.cols * grid.rows < 2) return false;
            build(grid.cols, grid.rows);
            place(area, host);
            element.classList.add("is-active");
            element.classList.remove("is-preparing", "is-warming");
            return true;
        },
        place,
        /**
         * Показать превью тайла в его собственной клетке.
         *
         * Это и есть «настоящая отрисовка»: кусок, который движок считает
         * прямо сейчас, стоит там, где он окажется в готовом кадре, а не во
         * весь экран. Предыдущий адрес освобождается — за прогон их сотни.
         *
         * @param {number} index номер тайла в порядке обхода (по строкам)
         * @param {Blob} blob превью от движка
         */
        setCellImage(index, blob) {
            const cell = cells[index];
            if (!cell || !blob) return false;
            const url = URL.createObjectURL(blob);
            let img = cell.querySelector("img");
            if (!img) {
                img = document.createElement("img");
                img.alt = "";
                cell.appendChild(img);
            }
            if (cellUrls[index]) URL.revokeObjectURL(cellUrls[index]);
            cellUrls[index] = url;
            img.src = url;
            cell.classList.add("has-image");
            element.classList.remove("is-preparing");
            return true;
        },
        /**
         * Сколько тайлов готово. Первый же вызов снимает подготовку: работа
         * началась, ждать больше нечего.
         *
         * @param {number} done абсолютное число готовых клеток
         */
        advance(done) {
            element.classList.remove("is-preparing");
            const ready = Math.max(0, Math.min(total, Math.round(done)));
            cells.forEach((cell, index) => {
                cell.classList.toggle("is-done", index < ready);
                cell.classList.toggle("is-live", index === ready && ready < total);
            });
        },
        /**
         * Доля выполнения (0..1) — когда движок считает не тайлы, а шаги.
         *
         * Честнее, чем кажется: клетки гаснут не «по тайлу», а по общему ходу,
         * зато их число и расположение настоящие, и ожидание перестаёт быть
         * пустым экраном.
         */
        advanceFraction(fraction) {
            const value = Math.max(0, Math.min(1, Number(fraction) || 0));
            element.classList.remove("is-preparing");
            const ready = Math.floor(value * total);
            cells.forEach((cell, index) => {
                cell.classList.toggle("is-done", index < ready);
                cell.classList.toggle("is-live", index === ready && ready < total);
            });
        },
        hide() {
            element.classList.remove("is-active", "is-preparing", "is-warming");
            element.style.cssText = "";
            dropUrls();
            element.replaceChildren();
            cells = [];
            total = 0;
            shape = { cols: 0, rows: 0 };
        },
        isActive: () => element.classList.contains("is-active"),
        isPreparing: () => element.classList.contains("is-preparing"),
        size: () => ({ ...shape, total }),
    };
}
