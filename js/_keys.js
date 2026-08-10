// Один ответ на вопрос «человек сейчас печатает?».
//
// Горячая клавиша, срабатывающая в поле ввода, — это не мелкая неточность, а
// сломанный ввод: в студии буква «e» в промпте переключала ластик вместо того,
// чтобы напечататься, `[` и `]` меняли размер кисти, Tab складывал панель. Всё
// это ловилось на уровне оверлея, который раздавал события, никого не спрашивая.
//
// Правило простое и одно на весь пак:
//
//   пока фокус в поле ввода, простые клавиши принадлежат ПОЛЮ;
//   горячей может быть только комбинация с Ctrl/Cmd — и то не всякая
//   (Ctrl+Z в текстовом поле — это отмена ТЕКСТА, а не маски).
//
// ⚠️ Проверять надо цель события, а не только `document.activeElement`: событие
// может прийти из другого документа (оверлей студии живёт в своём) или из
// shadow root, и активный элемент там свой.

/** Поля, в которых любая буква — это ввод, а не команда. */
const TYPING_TAGS = new Set(["INPUT", "TEXTAREA", "SELECT"]);

/**
 * Печатает ли человек прямо сейчас в этом элементе.
 *
 * @param {EventTarget|null} target Цель события клавиатуры.
 * @returns {boolean}
 */
export function isTypingTarget(target) {
    const element = target && target.nodeType === 1 ? target : null;
    if (!element) return false;
    if (TYPING_TAGS.has(element.tagName)) {
        // Кнопка и чекбокс — тоже <input>, но в них не печатают: пробел и
        // стрелки там ничего не вводят, и горячие клавиши им не мешают.
        if (element.tagName === "INPUT") {
            const type = String(element.type || "text").toLowerCase();
            return !["button", "submit", "reset", "checkbox", "radio", "range",
                     "color", "file", "image"].includes(type);
        }
        return true;
    }
    if (element.isContentEditable) return true;
    // Ближайший редактируемый предок: внутри contenteditable целью бывает
    // вложенный span.
    return Boolean(element.closest?.("[contenteditable='']," +
                                     "[contenteditable='true'],input,textarea,select")
        && isTypingTarget(element.closest("[contenteditable=''],"
            + "[contenteditable='true'],input,textarea,select")));
}

/**
 * Можно ли отдавать это событие горячим клавишам.
 *
 * @param {KeyboardEvent} event
 * @param {object} [options]
 * @param {boolean} [options.allowModifier=true] Пропускать ли комбинации с
 *   Ctrl/Cmd, пока человек печатает. Ctrl+Enter «запустить» — да; Ctrl+Z в поле
 *   принадлежит тексту, и такие случаи обработчик отсеивает сам.
 * @returns {boolean}
 */
export function hotkeysAllowed(event, { allowModifier = true } = {}) {
    if (!isTypingTarget(event?.target)) return true;
    if (!allowModifier) return false;
    return Boolean(event.ctrlKey || event.metaKey);
}
