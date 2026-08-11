// Шкала времени: где ставить деления и как писать таймкод.
//
// Без DOM — проверяется обычным Node.

// Лестница «красивых» шагов. Произвольный шаг вида 0.37 с человек читать не
// умеет: деления должны попадать на кадры, секунды, минуты и часы.
const NICE_STEPS = [
    0.04, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30,
    60, 120, 300, 600, 900, 1800, 3600, 7200, 10800,
];

/**
 * Шаг делений для видимого окна.
 *
 * @param {number} viewSeconds сколько секунд на панели
 * @param {number} width ширина панели в пикселях
 * @param {number} minPx минимальное расстояние между подписями
 */
export function pickTickStep(viewSeconds, width, minPx = 64) {
    if (!(viewSeconds > 0) || !(width > 0)) return 1;
    const wanted = (viewSeconds * minPx) / width;
    return NICE_STEPS.find((step) => step >= wanted) ?? NICE_STEPS[NICE_STEPS.length - 1];
}

function pad(value, size = 2) {
    return String(Math.floor(Math.abs(value))).padStart(size, "0");
}

/**
 * Секунды → таймкод.
 *
 * Формат подстраивается под масштаб: секунда с сотыми на коротком ролике,
 * часы:минуты:секунды на длинном. Читать «3673.42» человек не должен.
 *
 * @param {number} seconds
 * @param {object} [options]
 * @param {boolean} [options.frames] показывать кадры вместо сотых
 * @param {number} [options.fps] частота — нужна для кадров
 * @param {boolean} [options.compact] опускать часы, когда их нет
 */
export function formatTimecode(seconds, { frames = false, fps = 0, compact = true } = {}) {
    const total = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = Math.floor(total % 60);

    const tail = frames && fps > 0
        ? `:${pad(Math.round((total % 1) * fps))}`
        : `.${pad(Math.round((total % 1) * 100))}`;

    if (hours > 0 || !compact) return `${pad(hours)}:${pad(minutes)}:${pad(secs)}${tail}`;
    return `${pad(minutes)}:${pad(secs)}${tail}`;
}

/**
 * Таймкод → секунды.
 *
 * Понимает всё, что человек может напечатать: «12.5», «1:02.25», «1:02:03.5» и
 * «01:02:03:12» (последняя группа — кадры, если известна частота).
 *
 * @returns {number} секунды или NaN, если разобрать не вышло
 */
export function parseTimecode(text, fps = 0) {
    const clean = String(text ?? "").trim().replace(",", ".");
    if (!clean) return NaN;
    if (/^\d*\.?\d+$/.test(clean)) return Number(clean);

    const parts = clean.split(":");
    if (parts.some((part) => part !== "" && !/^\d*\.?\d+$/.test(part))) return NaN;

    // Четыре группы — это кадры в последней (ЧЧ:ММ:СС:КК).
    if (parts.length === 4) {
        const [h, m, s, f] = parts.map(Number);
        const frames = fps > 0 ? f / fps : 0;
        return h * 3600 + m * 60 + s + frames;
    }
    if (parts.length === 3) {
        const [h, m, s] = parts.map(Number);
        return h * 3600 + m * 60 + s;
    }
    if (parts.length === 2) {
        const [m, s] = parts.map(Number);
        return m * 60 + s;
    }
    return NaN;
}

/** Длительность одной строкой: «1:03:22» или «42 с». */
export function formatDuration(seconds) {
    const total = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
    if (total < 60) return `${total.toFixed(total < 10 ? 2 : 1)} s`;
    return formatTimecode(total, { compact: true }).replace(/\.\d+$/, "");
}

/** Размер файла человеческими словами. */
export function formatBytes(value) {
    const bytes = Number(value) || 0;
    if (bytes <= 0) return "";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
    return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}
