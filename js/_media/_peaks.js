// Огибающая звука для таймлайна: обзорная и детальная.
//
// Обзорная (около тысячи столбиков на весь ролик) приезжает вместе с
// метаданными и годится, пока виден весь клип. Стоит приблизиться — она
// растягивается в лесенку, и тогда запрашивается окно ровно по экрану.
//
// ⚠️ Столбиков запрашивается не больше, чем пикселей на панели: рисовать вторую
// тысячу значений в те же двести пикселей — работа впустую.

const DEBOUNCE_MS = 140;

/**
 * @param {object} options
 * @param {object} options.api api ComfyUI
 * @param {string} options.route базовый путь роутов
 * @param {() => string} options.getPath текущий файл
 * @param {() => void} options.onReady дёргается, когда окно приехало
 */
export function createPeakSource({ api, route, getPath, onReady }) {
    let overview = null;          // Array<number> на весь ролик
    let window = null;            // {start, end, data}
    let controller = null;
    let timer = null;

    const cancel = () => {
        clearTimeout(timer);
        timer = null;
        controller?.abort();
        controller = null;
    };

    return {
        setOverview(values) {
            overview = Array.isArray(values) && values.length ? values : null;
            window = null;
        },

        get hasData() { return Boolean(overview || window); },

        /**
         * Уровень звука в этот момент — из окна, если оно накрывает, иначе из
         * обзора. Дыр не бывает: пока детальное едет, рисуется грубое.
         */
        sample(seconds, duration) {
            if (window && seconds >= window.start && seconds < window.end) {
                const span = window.end - window.start;
                const index = Math.floor(((seconds - window.start) / span) * window.data.length);
                return window.data[Math.max(0, Math.min(window.data.length - 1, index))] || 0;
            }
            if (!overview || !(duration > 0)) return 0;
            const index = Math.floor((seconds / duration) * overview.length);
            return overview[Math.max(0, Math.min(overview.length - 1, index))] || 0;
        },

        /**
         * Заказать детальное окно.
         *
         * Пока человек тащит вьюпорт, запросы не уходят вовсе — только после
         * того, как движение остановилось.
         */
        request(start, end, bins) {
            if (!(end > start) || !getPath()) return;
            if (window && Math.abs(window.start - start) < 1e-3
                && Math.abs(window.end - end) < 1e-3) return;
            cancel();
            timer = setTimeout(async () => {
                controller = new AbortController();
                const params = new URLSearchParams({
                    filepath: getPath(),
                    start: String(start),
                    end: String(end),
                    bins: String(Math.max(8, Math.min(2048, Math.round(bins)))),
                });
                try {
                    const response = await api.fetchApi(`${route}/peaks?${params}`,
                                                        { signal: controller.signal });
                    if (!response.ok) return;
                    const payload = await response.json();
                    if (!Array.isArray(payload?.data) || !payload.data.length) return;
                    window = { start: payload.start, end: payload.end, data: payload.data };
                    onReady?.();
                } catch (error) {
                    // Отмена — норма: так гасится запрос, уехавший за край экрана.
                    if (error?.name !== "AbortError") {
                        console.warn("[TS Video] peaks window failed", error);
                    }
                } finally {
                    controller = null;
                }
            }, DEBOUNCE_MS);
        },

        clear() {
            cancel();
            overview = null;
            window = null;
        },

        dispose() { this.clear(); },
    };
}
