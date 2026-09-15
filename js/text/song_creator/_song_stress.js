/**
 * Ударение в тексте песни: чистые функции без DOM.
 *
 * ⚠️ Ударение — это КОМБИНИРУЮЩИЙ знак U+0301 ПОСЛЕ гласной, а не отдельная
 * буква. Готовых букв с акутом в кириллице нет ни одной (в Unicode есть только
 * `ѐ` и `ѝ` с грависом), поэтому «заменить гласную на гласную с ударением» —
 * это дописать к ней один невидимый символ. Так пишут словари, так это читают
 * модели, и так оно переживает копирование в любое поле.
 *
 * ⚠️ Из-за этого длина строки растёт, а видимых букв — нет. Поэтому функции
 * возвращают не только текст, но и новые границы выделения: без этого курсор
 * после каждой постановки уезжал бы на символ влево от того, что человек видит.
 *
 * Модуль без импортов и без DOM намеренно — его проверяет `tests/test_song_creator.py`
 * настоящим Node, а не браузером.
 */

/** Комбинирующий акут: то, что превращает «а» в «а́». */
export const STRESS_MARK = "́";

/**
 * Гласные обоих языков, с которыми работает нода.
 *
 * ⚠️ `ё` здесь тоже есть, хотя она и так всегда ударная: человек вправе
 * поставить знак явно, а модель — прочитать его. Запрещать это не за что.
 */
const VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯaeiouyAEIOUY";

/**
 * Гласная ли это (без учёта уже стоящего ударения).
 *
 * @param {string} char один символ.
 * @returns {boolean}
 */
export function isVowel(char) {
    return typeof char === "string" && char.length === 1 && VOWELS.includes(char);
}

/**
 * Найти гласную, с которой будет работать команда.
 *
 * Правила подобраны по тому, как человек на самом деле целится мышью:
 * • выделил ровно гласную (возможно, уже с ударением) — работаем с ней;
 * • выделил кусок, где гласная одна — работаем с ней, не заставляя целиться точнее;
 * • ничего не выделил — берём букву слева от курсора, затем справа: так работает
 *   правый клик, поставивший каретку внутрь слова.
 *
 * @param {string} text содержимое поля.
 * @param {number} selectionStart начало выделения.
 * @param {number} selectionEnd конец выделения.
 * @returns {{index:number, stressed:boolean}|null} позиция гласной, либо null.
 */
export function findStressTarget(text, selectionStart, selectionEnd) {
    const value = String(text ?? "");
    let start = Math.max(0, Math.min(value.length, Number(selectionStart) || 0));
    let end = Math.max(0, Math.min(value.length, Number(selectionEnd) || 0));
    if (start > end) [start, end] = [end, start];

    const stressedAt = (index) => value[index + 1] === STRESS_MARK;

    if (start === end) {
        for (const index of [start - 1, start]) {
            if (index >= 0 && isVowel(value[index])) {
                return { index, stressed: stressedAt(index) };
            }
        }
        // Каретка могла встать МЕЖДУ гласной и её знаком ударения — снаружи
        // это выглядит как «курсор на букве с ударением», и команда обязана
        // сработать, а не промолчать.
        if (value[start] === STRESS_MARK && isVowel(value[start - 1])) {
            return { index: start - 1, stressed: true };
        }
        return null;
    }

    const found = [];
    for (let index = start; index < end; index += 1) {
        if (isVowel(value[index])) found.push(index);
        if (found.length > 1) return null;      // две гласные — целиться надо точнее
    }
    if (found.length !== 1) return null;
    return { index: found[0], stressed: stressedAt(found[0]) };
}

/**
 * Поставить или снять ударение — одной командой, как переключатель.
 *
 * @param {string} text содержимое поля.
 * @param {number} selectionStart начало выделения.
 * @param {number} selectionEnd конец выделения.
 * @returns {{text:string, selectionStart:number, selectionEnd:number, stressed:boolean}|null}
 *   новое состояние поля, либо null, если целиться не во что.
 */
export function toggleStress(text, selectionStart, selectionEnd) {
    const value = String(text ?? "");
    const target = findStressTarget(value, selectionStart, selectionEnd);
    if (!target) return null;

    const { index, stressed } = target;
    if (stressed) {
        const next = value.slice(0, index + 1) + value.slice(index + 2);
        return { text: next, selectionStart: index, selectionEnd: index + 1, stressed: false };
    }
    const next = value.slice(0, index + 1) + STRESS_MARK + value.slice(index + 1);
    // Выделение накрывает букву ВМЕСТЕ со знаком: иначе следующий же ввод
    // заменил бы букву, оставив знак сиротой посреди слова.
    return { text: next, selectionStart: index, selectionEnd: index + 2, stressed: true };
}

/**
 * Снять все ударения в тексте — для случая «вставил из словаря, не надо».
 *
 * @param {string} text содержимое поля.
 * @returns {string} текст без комбинирующих акутов.
 */
export function stripAllStress(text) {
    return String(text ?? "").split(STRESS_MARK).join("");
}

/**
 * Сколько ударений уже стоит — для строки состояния.
 *
 * @param {string} text содержимое поля.
 * @returns {number}
 */
export function countStress(text) {
    return String(text ?? "").split(STRESS_MARK).length - 1;
}
