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
// ЧТО ИМЕННО ПОКАЗЫВАЕТ КЛЕТКА. Кадр остаётся на месте; клетка накрывает свой
// кусок размытием и притемнением — это «сюда работа ещё не дошла». Кусок,
// который считают прямо сейчас, ПРОЯВЛЯЕТСЯ из размытия в резкость СТУПЕНЯМИ
// по числу шагов сэмплера: три шага — три ступени, восемь шагов — восемь.
// Досчитанные куски остаются резкими.
//
// Так на экране видно три вещи разом: что готово, где идёт работа и сколько
// шагов внутри этого куска осталось.
//
// ⚠️ Превью самих кусков в клетки не кладём. Резчик режет С ПЕРЕКРЫТИЕМ и
// собирает куски со смешиванием краёв — в равномерную сетку они не ложатся, и
// кусок оказывается смещённым (замечено владельцем).
//
// ⚠️ Ничего циклического. Анимация «туда-сюда по кругу» показывает не работу, а
// сам факт ожидания; здесь у каждого движения есть причина — пришёл шаг.

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
/* Сетка растянута по кадру, внутри которого лежит. Пикселей от замеров экрана
   здесь нет и быть не должно: раньше клетки ставились по
   getBoundingClientRect(), и на зуме сетка получала двойной масштаб. */
.ts-tiles{position:absolute;inset:0;display:none;pointer-events:none;z-index:4}
.ts-tiles.is-active{display:grid}
/* Клетка размывает СВОЙ кусок кадра. Ступень размытия ставит прогон: одна
   ступень — один шаг сэмплера. Переход короткий, чтобы шаг читался как шаг, а
   не как плавное «что-то происходит». */
.ts-tiles__cell{position:relative;border:1px solid var(--ts-accent-line);
    backdrop-filter:blur(var(--ts-tile-blur,9px));
    -webkit-backdrop-filter:blur(var(--ts-tile-blur,9px));
    transition:backdrop-filter .3s ease,border-color .35s ease}
/* Притемнение — отдельным слоем, чтобы уходить ВМЕСТЕ с размытием и по тем же
   ступеням: кусок одновременно светлеет и становится резким, а не сначала одно,
   потом другое. */
.ts-tiles__cell::after{content:"";position:absolute;inset:0;
    background:var(--ts-scrim);opacity:var(--ts-tile-dark,1);
    transition:opacity .3s ease;pointer-events:none}
/* Готовый кусок — резкий и светлый: он и есть результат. */
.ts-tiles__cell.is-done{border-color:transparent;
    backdrop-filter:none;-webkit-backdrop-filter:none}
.ts-tiles__cell.is-done::after{opacity:0}
/* Кусок в работе: рамка ярче — взгляд знает, куда смотреть. */
.ts-tiles__cell.is-live{border-color:var(--ts-accent);
    box-shadow:inset 0 0 0 1px var(--ts-accent-line)}
/* Раскладка сетки — ОДИН раз, волной по диагонали: клетки ложатся на кадр по
   очереди, а не появляются разом. Дальше движения нет. */
.ts-tiles.is-preparing .ts-tiles__cell{
    animation:ts-tiles-lay .45s cubic-bezier(.2,.8,.3,1) backwards;
    animation-delay:var(--ts-wave)}
@keyframes ts-tiles-lay{
    from{opacity:0;transform:scale(.86)}
    to{opacity:1;transform:none}}
/* Ожидание, когда число кусков ещё неизвестно: сетку «примерно» рисовать
   нельзя — клетка обещает место работы, и обещание должно быть верным. */
.ts-tiles.is-warming{display:block;background:var(--ts-scrim);
    backdrop-filter:blur(9px);-webkit-backdrop-filter:blur(9px)}
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
 * @returns {{element, setSteps, prepare, warm, showByCount, showByGrid, advance,
 *            advanceFraction, hide, isActive, isPreparing, size}}
 */
/** Насколько размыт кусок, к которому работа ещё не подошла. */
const MAX_BLUR = 9;

export function createTileGrid() {
    ensureTileGridStyles();
    const element = document.createElement("div");
    element.className = `${TS_UI_CLASS} ts-tiles`;

    let cells = [];
    let total = 0;
    let shape = { cols: 0, rows: 0 };
    /** Сколько шагов сэмплера приходится на один кусок. */
    let steps = 0;
    /** Номер куска, который считают прямо сейчас. */
    let liveIndex = -1;

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
                cell.style.setProperty("--ts-wave", `${(row + col) * 80}ms`);
                element.appendChild(cell);
                cells.push(cell);
            }
        }
        total = cols * rows;
    }

    function reset() {
        liveIndex = -1;
        for (const cell of cells) {
            cell.classList.remove("is-done", "is-live");
            clearStep(cell);
        }
    }

    /**
     * Насколько размыт кусок при такой доле выполнения.
     *
     * Ступеней ровно столько, сколько шагов делает сэмплер на кусок: три шага —
     * три ступени, восемь — восемь. Так проявление СОВПАДАЕТ с работой, а не
     * изображает её. Числа шагов нет — считаем по доле, но всё равно ступенями,
     * иначе плавное размытие читается как «висит».
     *
     * @param {number} fraction доля выполнения куска, 0..1
     * @returns {number} радиус размытия в пикселях
     */
    function blurFor(fraction) {
        const value = Math.max(0, Math.min(1, Number(fraction) || 0));
        const ladder = steps > 0 ? steps : 6;
        const done = Math.min(ladder, Math.ceil(value * ladder));
        return (MAX_BLUR * (ladder - done)) / ladder;
    }

    /**
     * Поставить куску его ступень: и резкость, и свет разом.
     *
     * Одно число ведёт оба свойства — иначе картинка сначала светлеет, а потом
     * резчеет (или наоборот), и переход рвётся на два.
     */
    function applyStep(cell, fraction) {
        const ladder = steps > 0 ? steps : 6;
        const value = Math.max(0, Math.min(1, Number(fraction) || 0));
        const left = (ladder - Math.min(ladder, Math.ceil(value * ladder))) / ladder;
        cell.style.setProperty("--ts-tile-blur", `${MAX_BLUR * left}px`);
        cell.style.setProperty("--ts-tile-dark", String(left));
    }

    function clearStep(cell) {
        cell.style.removeProperty("--ts-tile-blur");
        cell.style.removeProperty("--ts-tile-dark");
    }

    /** Насколько размыт кусок сейчас — для тестов и для отладки. */
    function blurOf(index) {
        const cell = cells[index];
        return cell ? cell.style.getPropertyValue("--ts-tile-blur") : "";
    }

    /** Насколько притемнён кусок сейчас — для тестов и для отладки. */
    function darkOf(index) {
        const cell = cells[index];
        return cell ? cell.style.getPropertyValue("--ts-tile-dark") : "";
    }

    return {
        element,
        /**
         * Сколько шагов сэмплера приходится на кусок.
         *
         * По этому числу считаются ступени проявления: сколько шагов — столько
         * и ступеней. Ноль означает «неизвестно», и тогда лестница берётся
         * стандартная.
         *
         * @param {number} count шаги сэмплера
         */
        setSteps(count) {
            steps = Math.max(0, Math.round(Number(count) || 0));
        },
        /**
         * Показать сетку ДО начала работы: кадр расчерчивается и ждёт.
         *
         * Куда её положить, спрашивать не нужно: сетка лежит внутри кадра и
         * растянута по нему.
         *
         * @param {{cols:number,rows:number}} grid
         */
        prepare(grid) {
            if (!grid || !(grid.cols > 0) || !(grid.rows > 0)) return false;
            if (grid.cols * grid.rows < 2) return false;   // одна клетка — не сетка
            build(grid.cols, grid.rows);
            reset();
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
        warm() {
            element.replaceChildren();
            cells = [];
            total = 0;
            shape = { cols: 0, rows: 0 };
            element.classList.add("is-active", "is-warming");
            element.classList.remove("is-preparing");
            return true;
        },
        /**
         * Сетка по числу тайлов — когда движок сообщает только количество.
         *
         * @param {number} count сколько тайлов
         * @param {number} aspect ширина/высота КАРТИНКИ (её собственные данные,
         *   а не замер экрана) — по ней раскладываются строки и столбцы
         */
        showByCount(count, aspect) {
            if (!(count > 1)) return false;
            const grid = tileLayout(count, aspect > 0 ? aspect : 1);
            build(grid.cols, grid.rows);
            element.classList.add("is-active");
            element.classList.remove("is-preparing", "is-warming");
            return true;
        },
        /** Сетка по известной геометрии — когда её посчитал резчик. */
        showByGrid(grid) {
            if (!grid || grid.cols * grid.rows < 2) return false;
            build(grid.cols, grid.rows);
            element.classList.add("is-active");
            element.classList.remove("is-preparing", "is-warming");
            return true;
        },
        /**
         * Сколько тайлов готово. Первый же вызов снимает подготовку: работа
         * началась, ждать больше нечего.
         *
         * @param {number} done абсолютное число готовых клеток
         */
        /**
         * Сколько кусков готово и насколько продвинулся текущий.
         *
         * @param {number} done абсолютное число готовых клеток
         * @param {number|null} [fraction] ход ВНУТРИ текущего куска, 0..1
         */
        advance(done, fraction = null) {
            element.classList.remove("is-preparing");
            const ready = Math.max(0, Math.min(total, Math.round(done)));
            liveIndex = ready < total ? ready : -1;
            const inside = typeof fraction === "number" && Number.isFinite(fraction)
                ? Math.max(0, Math.min(1, fraction)) : 0;
            cells.forEach((cell, index) => {
                cell.classList.toggle("is-done", index < ready);
                cell.classList.toggle("is-live", index === liveIndex);
                // Проявляется ТОЛЬКО текущий кусок: сосед, который начнут
                // считать через минуту, не должен резчеть заранее.
                if (index === liveIndex) applyStep(cell, inside);
                else clearStep(cell);
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
            const exact = value * total;
            const ready = Math.floor(exact);
            liveIndex = ready < total ? ready : -1;
            cells.forEach((cell, index) => {
                cell.classList.toggle("is-done", index < ready);
                cell.classList.toggle("is-live", index === liveIndex);
                if (index === liveIndex) applyStep(cell, exact - ready);
                else clearStep(cell);
            });
        },
        hide() {
            element.classList.remove("is-active", "is-preparing", "is-warming");
            steps = 0;
            element.replaceChildren();
            cells = [];
            total = 0;
            shape = { cols: 0, rows: 0 };
        },
        blurOf,
        darkOf,
        isActive: () => element.classList.contains("is-active"),
        isPreparing: () => element.classList.contains("is-preparing"),
        size: () => ({ ...shape, total }),
    };
}
