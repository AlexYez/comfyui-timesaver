// TS Studio kit — то немногое, что нужно всем видам контролов.
//
// Сюда попадает только по-настоящему общее. Соблазн сложить в такой файл
// «всякое» велик, и через полгода он становится вторым god-объектом; поэтому
// правило простое: если помощник нужен одному виду — он живёт в его файле.

/**
 * Подпись из манифеста на языке интерфейса.
 *
 * Манифест пишет подписи либо строкой, либо словарём по языкам. Живут обе
 * формы: у части контролов подпись одинакова во всех языках (CFG, LoRA), и
 * заводить для неё словарь — лишний шум.
 *
 * @param {string|object|undefined} labelSpec подпись из манифеста
 * @param {string} locale "ru" | "en"
 * @param {string} fallback что показать, если подписи нет
 */
export function localized(labelSpec, locale, fallback) {
    if (!labelSpec) return fallback;
    if (typeof labelSpec === "string") return labelSpec;
    return labelSpec[locale] || labelSpec.en || fallback;
}
