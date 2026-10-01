// Чистая часть TS Prompt Library: подстановка полей, разбор каталога.
//
// Без `app` и `api` — её проверяет настоящий Node в тестах. ⚠️ Подстановка
// обязана совпадать с серверной (`nodes/text/_prompt_library.py: fill`):
// кнопка «Копировать» и выход ноды должны давать ОДИН И ТОТ ЖЕ текст.

/** Поле подстановки: {TARGET}, {TIME_OF_DAY}. Только заглавные. */
const FIELD = /\{([A-Z][A-Z0-9_]*)\}/g;

/** Поля промпта по порядку первого появления, без повторов. */
export function placeholdersIn(prompt) {
    const seen = [];
    for (const match of String(prompt || "").matchAll(FIELD)) {
        if (!seen.includes(match[1])) seen.push(match[1]);
    }
    return seen;
}

function valueOf(values, name) {
    const value = values?.[name];
    return typeof value === "string" || typeof value === "number" ? String(value).trim() : "";
}

/**
 * Подставить значения. Пустое поле остаётся в скобках.
 *
 * Один проход по исходному тексту: значение, в котором само есть «{OBJECT}»,
 * повторно не раскрывается.
 */
export function fillPrompt(prompt, values) {
    return String(prompt || "").replace(FIELD, (whole, name) => valueOf(values, name) || whole);
}

/**
 * Промпт кусками для показа: обычный текст и поля (заполненные — со
 * значением, пустые — с именем в скобках).
 *
 * @returns {Array<{text: string, field?: string, filled?: boolean}>}
 */
export function promptSegments(prompt, values) {
    const text = String(prompt || "");
    const out = [];
    let last = 0;
    for (const match of text.matchAll(FIELD)) {
        if (match.index > last) out.push({ text: text.slice(last, match.index) });
        const value = valueOf(values, match[1]);
        out.push({ text: value || match[0], field: match[1], filled: Boolean(value) });
        last = match.index + match[0].length;
    }
    if (last < text.length) out.push({ text: text.slice(last) });
    return out;
}

/** Незаполненные поля пресета — для предупреждения перед копированием. */
export function emptyFields(prompt, values) {
    return placeholdersIn(prompt).filter((name) => !valueOf(values, name));
}

/** Текст на языке интерфейса; запасные — английский, русский, что есть. */
export function pickText(value, lang = "en") {
    if (typeof value === "string") return value;
    if (!value || typeof value !== "object") return "";
    return value[lang] || value.en || value.ru || Object.values(value).find(Boolean) || "";
}

/** JSON значений полей из виджета — всегда объект. */
export function parseFields(text) {
    try {
        const data = JSON.parse(String(text || "") || "{}");
        return data && typeof data === "object" && !Array.isArray(data) ? data : {};
    } catch {
        return {};
    }
}

/** Все пресеты каталога по порядку, с их местом. */
export function flattenCatalog(catalog) {
    const out = [];
    for (const section of catalog?.sections || []) {
        for (const collection of section.collections || []) {
            for (const group of collection.groups || []) {
                for (const preset of group.presets || []) {
                    out.push({ section, collection, group, preset });
                }
            }
        }
    }
    return out;
}

/** Место пресета по ключу `section/collection/ID`, либо null. */
export function findPreset(catalog, key) {
    return flattenCatalog(catalog).find((entry) => entry.preset.key === key) || null;
}

/**
 * Ключ при смене раздела или модели.
 *
 * Коды задач совпадают между коллекциями одного раздела (R01 у Qwen и у
 * Klein — одна и та же задача), поэтому при смене модели остаёмся на той же
 * задаче, если она там есть; иначе — первый пресет.
 */
export function switchKey(catalog, currentKey, sectionId, collectionId = null) {
    const section = (catalog?.sections || []).find((s) => s.id === sectionId);
    if (!section) return currentKey;
    const collection = (section.collections || []).find((c) => c.id === collectionId)
        || section.collections?.[0];
    if (!collection) return currentKey;
    const code = String(currentKey || "").split("/").pop();
    const presets = (collection.groups || []).flatMap((g) => g.presets || []);
    const same = presets.find((p) => p.id === code);
    return (same || presets[0])?.key || currentKey;
}

/** Строчные, «ё» = «е»: человек пишет как привык, а не как в каталоге. */
function normalise(text) {
    return String(text || "").toLowerCase().replace(/ё/g, "е");
}

/** Слова запроса; пустой запрос — пустой список. */
export function queryWords(query) {
    return normalise(query).split(/\s+/).filter(Boolean);
}

/**
 * Подходит ли пресет под строку поиска (и в ноде, и во весь экран).
 *
 * Ищем по коду, названию и описанию на ВСЕХ языках, по самому промпту и по
 * названию группы: человек помнит то «R02», то «раскрашивание», то «colorize»,
 * то «удаление». Несколько слов — все должны найтись (порядок любой).
 * Примечания не участвуют: в них упоминаются ДРУГИЕ коды («возьмите O12»), и
 * поиск «o12» находил бы чужой пресет.
 */
export function presetMatches(preset, query, groupTitle = null) {
    const words = queryWords(query);
    if (!words.length) return true;
    const texts = (value) => (typeof value === "string" ? [value]
        : value && typeof value === "object" ? Object.values(value) : []);
    const haystack = normalise([
        preset?.id, ...texts(preset?.title), ...texts(preset?.summary), preset?.prompt,
        ...texts(groupTitle),
    ].filter(Boolean).join("\n"));
    return words.every((word) => haystack.includes(word));
}

/** Группы коллекции, в которых остались только подходящие пресеты. */
export function filterGroups(collection, query) {
    return (collection?.groups || [])
        .map((group) => ({
            ...group,
            presets: (group.presets || []).filter((p) => presetMatches(p, query, group.title)),
        }))
        .filter((group) => group.presets.length);
}

/** Найденные пресеты коллекции одним списком, в порядке каталога. */
export function searchPresets(collection, query) {
    return filterGroups(collection, query).flatMap((group) => group.presets);
}

/**
 * Соседний пресет в пределах коллекции (стрелки ‹ › и ↑ ↓), по кругу.
 *
 * С запросом листаются только найденные: иначе стрелка уводила бы на пресет,
 * которого в списке нет. Текущий в найденное не входит — шаг вперёд ведёт на
 * первый найденный, назад — на последний.
 */
export function neighbourKey(catalog, currentKey, step, query = "") {
    const found = findPreset(catalog, currentKey);
    if (!found) return currentKey;
    const presets = searchPresets(found.collection, query);
    if (!presets.length) return currentKey;
    const index = presets.findIndex((p) => p.key === currentKey);
    if (index < 0) return (step > 0 ? presets[0] : presets[presets.length - 1]).key;
    const next = presets[(index + step + presets.length) % presets.length];
    return next?.key || currentKey;
}
