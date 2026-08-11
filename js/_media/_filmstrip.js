// Полоса кадров под таймлайном: тайлы, кэш и отмена запросов.
//
// ⚠️ КАДРЫ ЗАПРАШИВАЮТСЯ НЕ ПООДИНОЧКЕ, А ЛЕНТАМИ. Часовой ролик на первом
// уровне зума — это сотни миниатюр; по запросу на каждую сервер ляжет, а
// браузер захлебнётся соединениями. Один спрайт из шестнадцати штук = один
// запрос, дальше он режется при отрисовке.
//
// Уровни приближения взяты лестницей, а не «сколько получилось»: иначе каждое
// движение колеса меняло бы шаг на доли процента и обнуляло весь кэш.

/** Секунд на одну миниатюру — фиксированные ступени. */
export const STEPS = [0.04, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];

/** Сколько миниатюр в одном спрайте. */
export const TILE_COLS = 16;

/** Ширина миниатюры по умолчанию, если форма кадра ещё неизвестна. */
const TARGET_PX = 96;

/** Сколько лент держим в памяти. Больше — незачем, меньше — начинает мигать. */
const CACHE_LIMIT = 96;

/**
 * Ступень приближения для такого окна.
 *
 * ⚠️ Шаг подбирается под ШИРИНУ КАРТОЧКИ, а не под круглое число пикселей.
 * Ширину карточки задаёт форма кадра (высота дорожки × пропорции ролика), и если
 * шаг с ней не согласован, ячейка выходит уже кадра — из широкого ролика 16:9
 * получаются вертикальные обрезки.
 *
 * @param {number} viewSeconds сколько секунд видно на панели
 * @param {number} width ширина панели в пикселях
 * @param {number} [cardPx] желаемая ширина одной миниатюры
 */
export function pickStep(viewSeconds, width, cardPx = TARGET_PX) {
    if (!(viewSeconds > 0) || !(width > 0)) return STEPS[STEPS.length - 1];
    const target = cardPx > 4 ? cardPx : TARGET_PX;
    const wanted = (viewSeconds * target) / width;
    // Ближайшая ступень к нужному шагу: этот шаг задаёт только ЗАГРУЗКУ и ключи
    // кэша, а расстановкой карточек занимается отрисовка — ей нужен непрерывный
    // шаг, иначе карточки налезают друг на друга или расходятся щелями.
    let best = STEPS[0];
    for (const step of STEPS) {
        if (Math.abs(step - wanted) < Math.abs(best - wanted)) best = step;
    }
    return best;
}

/**
 * Источник миниатюр.
 *
 * @param {object} options
 * @param {object} options.api api ComfyUI (нужен `fetchApi` и `apiURL`)
 * @param {string} options.route базовый путь роутов
 * @param {() => string} options.getPath текущий файл
 * @param {() => number} options.getHeight высота миниатюры в пикселях
 * @param {() => void} options.onReady дёргается, когда пришли новые данные
 */
export function createStripSource({ api, route, getPath, getHeight, onReady }) {
    const cache = new Map();          // ключ -> {bitmap, cellWidth}
    const inFlight = new Map();       // ключ -> AbortController
    let overview = null;              // {bitmap, cellWidth, step, count}
    let overviewToken = 0;
    let timer = null;

    const key = (step, index, height) => `${step}:${index}:${height}`;

    const evict = () => {
        while (cache.size > CACHE_LIMIT) {
            const oldest = cache.keys().next().value;
            const entry = cache.get(oldest);
            cache.delete(oldest);
            // ⚠️ close() обязателен: сотня незакрытых ImageBitmap держит
            // десятки мегабайт видеопамяти и сборщик до них не доберётся.
            entry?.bitmap?.close?.();
        }
    };

    async function fetchSprite(step, index, height, signal) {
        const params = new URLSearchParams({
            filepath: getPath(),
            step: String(step),
            index: String(index),
            count: String(TILE_COLS),
            height: String(height),
        });
        const response = await api.fetchApi(`${route}/strip?${params}`, { signal });
        if (!response.ok) throw new Error(`strip ${response.status}`);
        const blob = await response.blob();
        const bitmap = await createImageBitmap(blob);
        return { bitmap, cellWidth: bitmap.width / TILE_COLS };
    }

    async function load(step, index, height) {
        const id = key(step, index, height);
        if (cache.has(id) || inFlight.has(id)) return;
        const controller = new AbortController();
        inFlight.set(id, controller);
        try {
            const entry = await fetchSprite(step, index, height, controller.signal);
            cache.set(id, entry);
            evict();
            onReady?.();
        } catch (error) {
            // Отмена — это норма, а не сбой: так гасятся ленты, уехавшие за
            // край экрана. Ругаться в консоль на них нельзя.
            if (error?.name !== "AbortError") {
                console.warn("[TS Video] filmstrip tile failed", error);
            }
        } finally {
            inFlight.delete(id);
        }
    }

    return {
        /**
         * Обзорная лента на весь ролик.
         *
         * Никогда не вытесняется и служит подложкой на любом приближении —
         * поэтому при зуме не бывает пустых дыр, пока грузится точный уровень.
         */
        async ensureOverview(duration) {
            const token = ++overviewToken;
            const height = getHeight();
            if (!(duration > 0)) return;
            const step = duration / TILE_COLS;
            try {
                const entry = await fetchSprite(step, 0, height, undefined);
                if (token !== overviewToken) { entry.bitmap.close?.(); return; }
                overview?.bitmap?.close?.();
                overview = { ...entry, step, count: TILE_COLS };
                onReady?.();
            } catch (error) {
                if (error?.name !== "AbortError") {
                    console.warn("[TS Video] overview strip failed", error);
                }
            }
        },

        /** Готовая ячейка точного уровня, если она уже есть. */
        lookup(step, cellIndex, height) {
            const tile = Math.floor(cellIndex / TILE_COLS);
            const entry = cache.get(key(step, tile, height));
            if (!entry) return null;
            const column = cellIndex - tile * TILE_COLS;
            return {
                bitmap: entry.bitmap,
                sx: column * entry.cellWidth,
                sw: entry.cellWidth,
                sh: entry.bitmap.height,
            };
        },

        /** Грубая подложка из обзорной ленты. */
        lookupCoarse(seconds) {
            if (!overview || !(overview.step > 0)) return null;
            const column = Math.max(0, Math.min(overview.count - 1,
                Math.floor(seconds / overview.step)));
            return {
                bitmap: overview.bitmap,
                sx: column * overview.cellWidth,
                sw: overview.cellWidth,
                sh: overview.bitmap.height,
            };
        },

        /**
         * Заказать ленты, покрывающие видимые ячейки.
         *
         * С дебаунсом: пока человек тащит скроллбар, запросы не уходят вовсе —
         * рисуется подложка. Иначе протаскивание по часу даёт сотни запросов.
         */
        request(step, firstCell, lastCell) {
            clearTimeout(timer);
            const height = getHeight();
            timer = setTimeout(() => {
                const firstTile = Math.floor(firstCell / TILE_COLS);
                const lastTile = Math.floor(lastCell / TILE_COLS);
                const tiles = [];
                for (let tile = firstTile; tile <= lastTile; tile += 1) tiles.push(tile);
                // От центра наружу: то, на что человек смотрит, приезжает первым.
                const centre = (firstTile + lastTile) / 2;
                tiles.sort((a, b) => Math.abs(a - centre) - Math.abs(b - centre));
                for (const tile of tiles.slice(0, 4)) load(step, tile, height);
            }, 120);
        },

        /** Снять всё, что уже неактуально (сменился уровень или файл). */
        abortAll() {
            clearTimeout(timer);
            for (const controller of inFlight.values()) controller.abort();
            inFlight.clear();
        },

        /** Забыть всё — при смене файла или высоты дорожки. */
        clear() {
            this.abortAll();
            for (const entry of cache.values()) entry.bitmap?.close?.();
            cache.clear();
            overview?.bitmap?.close?.();
            overview = null;
            overviewToken += 1;
        },

        dispose() { this.clear(); },
    };
}
