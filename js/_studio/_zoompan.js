// TS Studio kit — жесты масштаба и перемещения (core layer, no DOM of its own).
//
// Приближать колесом и таскать средней кнопкой умеет каждая программа, где
// смотрят на картинку, и делают это везде одинаково. Здесь жесты собраны в
// одном месте, а КАК именно двигать содержимое, решает поверхность: у холста
// маски свой масштаб внутри состояния (по нему считаются мазки), у сцены —
// обычный CSS-трансформ. Общее у них только поведение под рукой.
//
// Почему middle-drag, а не левая: левая на холсте рисует. Пробел+левая — тоже
// привычка, но он занят прокруткой страницы под оверлеем, и ловить его в
// текстовом поле промпта пришлось бы отдельно.

/** Насколько один щелчок колеса меняет масштаб. */
const WHEEL_STEP = 1.12;

/**
 * Повесить жесты на элемент.
 *
 * @param {HTMLElement} target Элемент, над которым работают жесты.
 * @param {object} handlers
 * @param {(clientX: number, clientY: number, factor: number) => void} handlers.zoomAt
 *   Приблизить/отдалить, сохраняя точку под курсором на месте.
 * @param {(dx: number, dy: number) => void} handlers.panBy Сдвинуть содержимое.
 * @param {() => void} [handlers.reset] Вернуть вписанный масштаб.
 * @returns {() => void} Снять слушатели.
 */
export function attachZoomPan(target, handlers) {
    const onWheel = (event) => {
        // Колесо над картинкой — это масштаб, а не прокрутка страницы под
        // оверлеем. Отменяем явно: иначе браузер прокручивает то, что за нами.
        event.preventDefault();
        const factor = event.deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP;
        handlers.zoomAt(event.clientX, event.clientY, factor);
    };
    target.addEventListener("wheel", onWheel, { passive: false });

    let panning = false;
    let lastX = 0;
    let lastY = 0;
    const onDown = (event) => {
        if (event.button !== 1) return;            // только средняя
        event.preventDefault();
        panning = true;
        lastX = event.clientX;
        lastY = event.clientY;
        try { target.setPointerCapture(event.pointerId); } catch { /* ничего */ }
    };
    const onMove = (event) => {
        if (!panning) return;
        // Кнопка — источник правды, как и у кисти: потерянный pointerup не
        // должен оставить картинку приклеенной к курсору.
        if ((event.buttons & 4) === 0) { panning = false; return; }
        handlers.panBy(event.clientX - lastX, event.clientY - lastY);
        lastX = event.clientX;
        lastY = event.clientY;
    };
    const stop = () => { panning = false; };
    target.addEventListener("pointerdown", onDown);
    target.addEventListener("pointermove", onMove);
    target.addEventListener("pointerup", stop);
    target.addEventListener("pointercancel", stop);
    target.addEventListener("lostpointercapture", stop);
    window.addEventListener("blur", stop);
    // Средняя кнопка в браузере открывает автопрокрутку — она здесь ни к чему.
    const onAux = (event) => { if (event.button === 1) event.preventDefault(); };
    target.addEventListener("auxclick", onAux);

    const onDouble = () => handlers.reset?.();
    target.addEventListener("dblclick", onDouble);

    return () => {
        target.removeEventListener("wheel", onWheel);
        target.removeEventListener("pointerdown", onDown);
        target.removeEventListener("pointermove", onMove);
        target.removeEventListener("pointerup", stop);
        target.removeEventListener("pointercancel", stop);
        target.removeEventListener("lostpointercapture", stop);
        target.removeEventListener("auxclick", onAux);
        target.removeEventListener("dblclick", onDouble);
        window.removeEventListener("blur", stop);
    };
}

/**
 * Зажать масштаб в разумные пределы.
 *
 * Нижняя граница — доля вписанного: дальше картинка превращается в марку и
 * искать на ней нечего. Верхняя абсолютная: на 32x уже видно устройство
 * пикселя, дальше незачем.
 *
 * @param {number} scale желаемый масштаб
 * @param {number} fitScale масштаб, при котором картинка вписана целиком
 */
export function clampScale(scale, fitScale) {
    const min = Math.min(fitScale, fitScale * 0.5);
    return Math.min(32, Math.max(min, scale));
}
