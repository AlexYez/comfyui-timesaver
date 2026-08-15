// Огибающая звука — одна отрисовка на весь пак.
//
// Вынесена из TS Audio Loader слово в слово: столбики от середины в обе
// стороны, выделенное — акцентом в полную силу, остальное — приглушённым серым.
// Раньше это жило только внутри загрузчика, и вторая вейвформа (в Дизайнере
// промпта) получилась тонкой линией — похожей на другую программу.
//
// ⚠️ Значения (`peaks`) — обзорные, на весь файл. Окно рисуется выборкой из
// них: панель шириной в двести пикселей не станет лучше от второй тысячи чисел.

/**
 * Нарисовать столбики огибающей в прямоугольник холста.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} opts
 * @param {number[]} opts.peaks      Обзорные значения 0..1 на весь файл.
 * @param {number} opts.duration     Длительность файла, секунды.
 * @param {number} opts.x            Левый край области.
 * @param {number} opts.y            Верхний край области.
 * @param {number} opts.width
 * @param {number} opts.height
 * @param {number} [opts.viewStart]  Начало видимого окна, секунды.
 * @param {number} [opts.viewEnd]    Конец видимого окна, секунды.
 * @param {{left: number, right: number}} [opts.selection] Выделенный кусок.
 * @param {object} opts.colors       Результат getThemeColors().
 * @param {boolean} [opts.baseline]  Тонкая линия по центру, когда тихо.
 */
export function drawPeakBars(ctx, {
    peaks, duration, x = 0, y = 0, width, height,
    viewStart = 0, viewEnd = null, selection = null, colors, baseline = true,
}) {
    if (!(width > 0) || !(height > 0)) return;
    const end = viewEnd === null || viewEnd <= viewStart ? duration : viewEnd;
    const span = Math.max(1e-6, end - viewStart);
    const middle = y + height / 2;

    if (baseline) {
        ctx.strokeStyle = colors.faint;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, middle + 0.5);
        ctx.lineTo(x + width, middle + 0.5);
        ctx.stroke();
    }
    if (!Array.isArray(peaks) || !peaks.length || !(duration > 0)) return;

    const secondsToX = (seconds) => x + ((seconds - viewStart) / span) * width;
    const count = peaks.length;
    const perPeak = duration / count;
    const first = clamp(Math.floor(viewStart / perPeak), 0, count - 1);
    const last = clamp(Math.ceil(end / perPeak), first, count - 1);
    // Зазор между столбиками только пока их меньше, чем пикселей: иначе
    // единичный отступ съедает половину картинки.
    const inset = count > width ? 0 : 1;

    for (let index = first; index <= last; index += 1) {
        const value = clamp(Number(peaks[index]) || 0, 0, 1);
        const from = index * perPeak;
        const to = from + perPeak;
        const x0 = secondsToX(from);
        const x1 = secondsToX(to);
        const barWidth = Math.max(1, (x1 - x0) - inset);
        const barHeight = Math.max(2, value * (height * 0.46));
        const inside = !selection
            || (to >= selection.left && from <= selection.right);
        // Невыделенное не красится в другой цвет, а гасится: контраст между
        // «звучит» и «не войдёт» должен читаться, а не спорить с акцентом.
        ctx.globalAlpha = inside ? 1 : 0.42;
        ctx.fillStyle = inside ? colors.accent : colors.muted;
        ctx.fillRect(x0, middle - barHeight, barWidth, barHeight * 2);
    }
    ctx.globalAlpha = 1;
}

function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
}
