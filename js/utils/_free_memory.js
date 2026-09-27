// Чистая часть кнопки «Освободить память»: как назвать освобождённое.
//
// Без `app` и `api` — её проверяет настоящий Node в тестах, без браузера.

/** Байты -> «12,3 ГБ» по правилам языка интерфейса. */
export function formatBytes(value, locale = undefined) {
    const size = Math.max(0, Number(value) || 0);
    const units = String(locale || "").toLowerCase().startsWith("ru")
        ? ["Б", "КБ", "МБ", "ГБ", "ТБ"]
        : ["B", "KB", "MB", "GB", "TB"];
    let index = 0;
    let left = size;
    while (left >= 1024 && index < units.length - 1) {
        left /= 1024;
        index += 1;
    }
    const digits = left >= 10 || index === 0 ? 0 : 1;
    return `${left.toLocaleString(locale || undefined, { maximumFractionDigits: digits })} ${units[index]}`;
}

// Меньше этого — шум измерения, а не освобождённая память: драйвер и
// аллокатор двигают свободный объём на мегабайты и без нас.
const NOISE_BYTES = 16 * 1024 * 1024;

/**
 * Что сказать человеку по ответу `/ts_memory/free`.
 *
 * `null` в ответе значит «не измерить» (нет CUDA, нет psutil) — это не ноль,
 * и сказать «освобождать было нечего» по нему было бы неправдой.
 *
 * @param {object} answer   ответ сервера.
 * @param {object} strings  словарь: `deferred`, `nothing`, `doneShort`,
 *   `done(list)`, `vram(size)`, `ram(size)`, `busyNote`.
 * @param {string} [locale] язык чисел.
 * @returns {{severity: string, text: string}}
 */
export function describeResult(answer, strings, locale = undefined) {
    if (answer?.deferred) return { severity: "info", text: strings.deferred };

    const parts = [];
    let measured = false;
    for (const [key, label] of [["vram_freed", strings.vram], ["ram_freed", strings.ram]]) {
        const value = answer?.[key];
        if (value === null || value === undefined) continue;
        measured = true;
        const size = Math.max(0, Number(value) || 0);
        if (size >= NOISE_BYTES) parts.push(label(formatBytes(size, locale)));
    }

    let severity = "success";
    let text;
    if (!measured) {
        text = strings.doneShort;
    } else if (!parts.length) {
        severity = "info";
        text = strings.nothing;
    } else {
        text = strings.done(parts.join(", "));
    }
    // Кэш, занятый генерацией, не тронут — промолчать значило бы соврать.
    if (Array.isArray(answer?.busy) && answer.busy.length) {
        severity = "warn";
        text = `${text} ${strings.busyNote}`;
    }
    return { severity, text };
}
