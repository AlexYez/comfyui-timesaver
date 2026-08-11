// Вьюпорт таймлайна: какой кусок времени сейчас на экране.
//
// Ни одной ссылки на документ — поэтому проверяется обычным Node, без браузера.
// Модель ровно из двух чисел: во сколько раз приближено (`zoom`) и с какой
// секунды начинается видимое окно (`viewStart`). Всё остальное выводится.

/** Зажать значение в границы. */
export function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
}

/**
 * Потолок приближения для ролика такой длины.
 *
 * ⚠️ НЕ КОНСТАНТА. Аудио-лоадер живёт с потолком 200×, и для песни этого хватает.
 * Для часового ролика 200× — окно в 18 секунд: выбрать секунду нельзя в принципе.
 * Считаем от того, что должно быть видно на всю панель — примерно восемь кадров.
 */
export function maxZoomFor(duration, fps) {
    const rate = fps > 0 ? fps : 25;
    const minView = Math.max(0.5, 8 / rate);
    if (!(duration > 0)) return 1;
    return clamp(duration / minView, 1, 200000);
}

/**
 * Создать вьюпорт.
 *
 * @param {object} options
 * @param {() => number} options.getDuration длительность ролика в секундах
 * @param {() => number} [options.getFps] частота — нужна только для потолка зума
 * @param {() => void} [options.onChange] дёргается после любого изменения
 */
export function createTimeViewport({ getDuration, getFps = () => 0, onChange } = {}) {
    const state = { zoom: 1, viewStart: 0 };

    const duration = () => Math.max(0, Number(getDuration?.() ?? 0));
    const maxZoom = () => maxZoomFor(duration(), Number(getFps?.() ?? 0));
    const viewSeconds = () => (duration() > 0 ? duration() / state.zoom : 0);

    const changed = () => { onChange?.(); };

    const clampViewStart = () => {
        const total = duration();
        // ⚠️ Пока длительность неизвестна, позицию НЕ трогаем: иначе
        // восстановленное из workflow значение обнулится и вьюпорт прыгнет в
        // начало ещё до того, как приедут метаданные (грабли аудио-лоадера).
        if (total <= 0) return;
        const maxStart = Math.max(0, total - viewSeconds());
        state.viewStart = clamp(state.viewStart, 0, maxStart);
    };

    return {
        get zoom() { return state.zoom; },
        get viewStart() { return state.viewStart; },
        set viewStart(value) {
            state.viewStart = Number.isFinite(value) ? Math.max(0, value) : 0;
            clampViewStart();
        },
        set zoom(value) {
            state.zoom = clamp(Number(value) || 1, 1, maxZoom());
            clampViewStart();
        },

        getViewSeconds: viewSeconds,
        getViewEnd: () => state.viewStart + viewSeconds(),
        getMaxZoom: maxZoom,
        clampViewStart,

        /** Секунда → пиксель по горизонтали. */
        secondsToX(seconds, width) {
            const view = viewSeconds();
            if (view <= 0) return 0;
            return ((seconds - state.viewStart) / view) * width;
        },

        /** Пиксель → секунда. */
        xToSeconds(x, width) {
            const view = viewSeconds();
            if (view <= 0 || width <= 0) return 0;
            return state.viewStart + (x / width) * view;
        },

        /**
         * Приблизить, оставив указанную секунду на месте.
         *
         * Без якоря зум колесом уводит картинку из-под курсора, и попасть в
         * нужное место становится невозможно.
         */
        setZoom(next, anchorSeconds) {
            const before = viewSeconds();
            const anchor = Number.isFinite(anchorSeconds)
                ? anchorSeconds
                : state.viewStart + before / 2;
            const fraction = before > 0 ? (anchor - state.viewStart) / before : 0.5;
            state.zoom = clamp(Number(next) || 1, 1, maxZoom());
            state.viewStart = anchor - fraction * viewSeconds();
            clampViewStart();
            changed();
        },

        /** Сдвинуть окно на столько секунд. */
        panBy(seconds) {
            state.viewStart += seconds;
            clampViewStart();
            changed();
        },

        /** Показать ролик целиком. */
        fit() {
            state.zoom = 1;
            state.viewStart = 0;
            changed();
        },

        /** Приблизить так, чтобы отрезок занял почти всю панель. */
        zoomToRange(from, to, pad = 0.08) {
            const total = duration();
            const span = Math.abs(to - from);
            if (total <= 0 || span <= 0) return;
            const padded = span * (1 + pad * 2);
            state.zoom = clamp(total / padded, 1, maxZoom());
            state.viewStart = Math.min(from, to) - span * pad;
            clampViewStart();
            changed();
        },

        toJSON: () => ({ zoom: state.zoom, viewStart: state.viewStart }),

        fromJSON(data) {
            const zoom = Number(data?.zoom);
            const start = Number(data?.viewStart);
            if (Number.isFinite(zoom)) state.zoom = clamp(zoom, 1, maxZoom());
            if (Number.isFinite(start)) state.viewStart = Math.max(0, start);
            clampViewStart();
        },
    };
}
