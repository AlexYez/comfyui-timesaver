// Выделенный отрезок: ручки, попадание по ним и жест перетаскивания.
//
// Без DOM: сюда приходят уже пересчитанные секунды, наружу уходит новое
// состояние отрезка. Поэтому конечный автомат жеста проверяется без браузера.

/** Насколько далеко от ручки ещё считается попаданием, в пикселях. */
export const HANDLE_HITBOX_PX = 10;

/** Минимальный сдвиг, после которого нажатие превращается в перетаскивание. */
export const DRAG_THRESHOLD_PX = 3;

/**
 * За что схватились в этой точке.
 *
 * @returns {"left"|"right"|null}
 */
export function hitTestHandle(pointerSeconds, { left, right, secondsPerPixel, hitboxPx = HANDLE_HITBOX_PX }) {
    const tolerance = Math.abs(secondsPerPixel) * hitboxPx;
    const toLeft = Math.abs(pointerSeconds - left);
    const toRight = Math.abs(pointerSeconds - right);
    if (toLeft <= tolerance && toLeft <= toRight) return "left";
    if (toRight <= tolerance) return "right";
    return null;
}

/**
 * Привести границы к порядку и к пределам ролика.
 *
 * ⚠️ Значение `-1` у конца означает «до конца файла» и обязано таким остаться:
 * это общий язык пака (та же конвенция у аудио-лоадера), и «починка» его в
 * число ломает workflow, сохранённые до того, как длительность стала известна.
 */
export function normaliseRange(start, end, duration) {
    const total = Math.max(0, Number(duration) || 0);
    const left = Math.max(0, Math.min(Number(start) || 0, total || Infinity));
    const rawEnd = Number(end);
    const right = (!Number.isFinite(rawEnd) || rawEnd < 0 || (total > 0 && rawEnd > total))
        ? total
        : rawEnd;
    return { left, right: Math.max(left, right) };
}

/**
 * Создать жест выделения.
 *
 * Состояния: `left`/`right` — тянут ручку; `pending` — нажали, но ещё не
 * сдвинулись (может оказаться простым щелчком); `range` — тянут новый отрезок.
 *
 * @param {object} options
 * @param {() => {left:number,right:number}} options.getBounds
 * @param {(left:number, right:number, moved:"left"|"right"|"range") => void}
 *   options.setBounds — третьим доводом идёт то, ЧТО именно двигали: плееру
 *   нужно следовать за той границей, за которую тянут
 * @param {(seconds:number) => void} [options.onSeek] щелчок без сдвига
 * @param {() => number} options.getSecondsPerPixel
 * @param {() => number} options.getDuration
 */
export function createRangeDrag({
    getBounds,
    setBounds,
    onSeek,
    getSecondsPerPixel,
    getDuration,
}) {
    let mode = null;
    let anchorSeconds = 0;
    let anchorX = 0;

    const clampToClip = (seconds) => {
        const total = getDuration();
        return Math.max(0, total > 0 ? Math.min(seconds, total) : seconds);
    };

    return {
        get mode() { return mode; },
        get active() { return mode !== null; },

        /** Нажатие. Возвращает то, за что взялись, — для смены курсора. */
        begin(pointerSeconds, clientX) {
            const bounds = getBounds();
            const handle = hitTestHandle(pointerSeconds, {
                left: bounds.left,
                right: bounds.right,
                secondsPerPixel: getSecondsPerPixel(),
            });
            mode = handle ?? "pending";
            anchorSeconds = clampToClip(pointerSeconds);
            anchorX = clientX;
            return mode;
        },

        /** Движение. Возвращает true, если состояние отрезка изменилось. */
        move(pointerSeconds, clientX) {
            if (mode === null) return false;
            const seconds = clampToClip(pointerSeconds);
            const bounds = getBounds();

            if (mode === "pending") {
                if (Math.abs(clientX - anchorX) < DRAG_THRESHOLD_PX) return false;
                mode = "range";
            }

            if (mode === "left") {
                setBounds(Math.min(seconds, bounds.right), bounds.right, "left");
                return true;
            }
            if (mode === "right") {
                setBounds(bounds.left, Math.max(seconds, bounds.left), "right");
                return true;
            }
            if (mode === "range") {
                setBounds(Math.min(anchorSeconds, seconds),
                          Math.max(anchorSeconds, seconds), "range");
                return true;
            }
            return false;
        },

        /** Отпускание. Нажатие без сдвига — это перемотка, а не пустое выделение. */
        end(pointerSeconds) {
            const wasPending = mode === "pending";
            mode = null;
            if (wasPending) {
                onSeek?.(clampToClip(pointerSeconds));
                return "seek";
            }
            return "range";
        },

        cancel() { mode = null; },
    };
}

// Курсоры для ручек. ⚠️ Ахроматические намеренно: они лежат ПОВЕРХ кадра
// пользователя и обязаны читаться на любой картинке — это законное исключение
// из запрета хардкод-цвета (§12.6 CLAUDE.md).
function makeCursor(body) {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">${body}</svg>`;
    return `url("data:image/svg+xml;utf8,${encodeURIComponent(svg)}") 10 10, ew-resize`;
}

export const HANDLE_CURSOR = makeCursor(
    '<path d="M10 2v16" stroke="#d8d8dc" stroke-width="2"/>'
    + '<path d="M6 7l-3 3 3 3M14 7l3 3-3 3" fill="none" stroke="#d8d8dc" stroke-width="2"/>');

export const HANDLE_ACTIVE_CURSOR = makeCursor(
    '<path d="M10 2v16" stroke="#ffffff" stroke-width="2"/>'
    + '<path d="M6 7l-3 3 3 3M14 7l3 3-3 3" fill="none" stroke="#ffffff" stroke-width="2"/>');
