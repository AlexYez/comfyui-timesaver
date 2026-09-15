/**
 * Сборка поля стиля из двух частей: жанр и вокал.
 *
 * ⚠️ Поле остаётся ОБЫЧНЫМ ТЕКСТОМ, который человек правит руками. Поэтому
 * пересобирать его целиком из выбранных карточек нельзя: правка пропала бы при
 * первом же клике. Вместо этого заменяется ровно тот кусок, который мы сами
 * туда положили, — а если человек его переписал и куска больше нет, новый
 * дописывается в конец, ничего не затирая.
 *
 * Без этого выбор трёх голосов подряд оставлял в поле три вокала сразу, и
 * модель получала противоречивый заказ.
 *
 * Модуль без DOM — его проверяет `tests/test_song_creator.py` настоящим Node.
 */

/** Разделитель дескрипторов: у YuE и ACE-Step это запятая. */
const SEP = ", ";

/**
 * Прибрать список дескрипторов: лишние запятые, пробелы, пустые куски.
 *
 * @param {string} text содержимое поля.
 * @returns {string}
 */
export function tidy(text) {
    return String(text ?? "")
        .split(",")
        .map((part) => part.trim())
        .filter(Boolean)
        .join(SEP);
}

/**
 * Заменить ранее вставленный фрагмент новым (или дописать, если его нет).
 *
 * @param {string} text текущее содержимое поля.
 * @param {string} previous что мы клали сюда в прошлый раз ("" — ничего).
 * @param {string} next что кладём теперь ("" — просто убрать прежнее).
 * @returns {string} новое содержимое.
 */
export function replaceSegment(text, previous, next) {
    const current = tidy(text);
    const was = tidy(previous);
    const now = tidy(next);

    if (!was) return current ? (now ? `${current}${SEP}${now}` : current) : now;

    // Сравниваем ПО КУСКАМ, а не поиском подстроки: человек мог переставить
    // дескрипторы местами или поправить пробелы, и подстрока перестала бы
    // находиться там, где смысл никуда не делся.
    const wasParts = was.split(SEP);
    const parts = current.split(SEP).filter(Boolean);
    const start = findRun(parts, wasParts);
    if (start < 0) {
        return current ? (now ? `${current}${SEP}${now}` : current) : now;
    }
    const nowParts = now ? now.split(SEP) : [];
    parts.splice(start, wasParts.length, ...nowParts);
    return parts.join(SEP);
}

/**
 * Где в списке начинается подряд идущая последовательность.
 *
 * @param {string[]} parts куски поля.
 * @param {string[]} needle куски искомого фрагмента.
 * @returns {number} индекс начала, либо -1.
 */
function findRun(parts, needle) {
    if (!needle.length || needle.length > parts.length) return -1;
    for (let index = 0; index + needle.length <= parts.length; index += 1) {
        let hit = true;
        for (let offset = 0; offset < needle.length; offset += 1) {
            if (parts[index + offset].toLowerCase() !== needle[offset].toLowerCase()) {
                hit = false;
                break;
            }
        }
        if (hit) return index;
    }
    return -1;
}

/**
 * Собрать поле заново из жанра и вокала — для первой подстановки.
 *
 * @param {string} style промпт жанра.
 * @param {string} vocal промпт вокала.
 * @returns {string}
 */
export function compose(style, vocal) {
    return tidy([style, vocal].filter(Boolean).join(SEP));
}

/**
 * Есть ли фрагмент в поле — для подсветки активной карточки.
 *
 * @param {string} text содержимое поля.
 * @param {string} fragment промпт карточки.
 * @returns {boolean}
 */
export function contains(text, fragment) {
    const needle = tidy(fragment);
    if (!needle) return false;
    return findRun(tidy(text).split(SEP).filter(Boolean), needle.split(SEP)) >= 0;
}
