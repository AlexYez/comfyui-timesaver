// Что студия предлагает выбрать — и почему именно это.
//
// Три вопроса, на которые отвечает список моделей, раньше решались вперемешку
// внутри двухтысячной функции, и каждый ответ приходилось проверять живым
// прогоном:
//
//   какие семейства показывать    те, что здесь и не спрятаны;
//   что показывать серым          то, чего здесь нет, но оно существует —
//                                 иначе человек не узнает, что Krea 2 бывает;
//   готов ли пак к работе         все ли файлы моделей на месте.
//
// Здесь это чистые функции над данными: ни DOM, ни сети. Их можно прогнать
// целиком без браузера, и именно они определяют, что человек видит в списке.
//
// ФОРМЫ ЗАПИСИ. Семейства пака записываются двумя способами: каталог сборки
// перечисляет их именами (`"krea2"`), собранный каталог доставки —
// описаниями (`{family, label, modes}`). Живут оба, и код, знающий одну
// форму, тихо не находит ничего — так устаревший набор однажды остался висеть
// в менеджере. Поэтому нормализация здесь одна на всех: `familyNames()`.

/**
 * Имена семейств пака, в какой бы форме их ни записали.
 *
 * @param {object} pack запись каталога
 * @returns {string[]}
 */
export function familyNames(pack) {
    return (pack?.families || [])
        .map((item) => (typeof item === "string" ? item : item?.family))
        .filter(Boolean)
        .map(String);
}

/**
 * Готов ли пак к работе на этой машине.
 *
 * Считается по живым бэкендам, а не по каталогу: какие файлы моделей нужны,
 * знает манифест каждого графа, и студия уже сверила их с тем, что стоит.
 * Отвечать на этот вопрос ДО запуска, а не красной ошибкой через десять минут
 * ожидания, — половина смысла менеджера паков.
 *
 * @param {object} pack запись каталога
 * @param {object[]} backends загруженные бэкенды
 * @returns {{ready:number,total:number,missing:string[]}|null} null, если пака
 *          здесь нет и проверять нечего
 */
export function packReadiness(pack, backends) {
    const names = new Set(familyNames(pack));
    const mine = (backends || []).filter((backend) => names.has(backend?.manifest?.family));
    if (!mine.length) return null;
    const missing = [...new Set(mine.flatMap((backend) => (backend.problems || [])
        .map((problem) => `${backend.manifest?.mode || backend.id}: ${problem}`)))];
    return {
        total: mine.length,
        ready: mine.filter((backend) => backend.available).length,
        missing,
    };
}

/**
 * Что показать серым: модели, которых здесь нет, и те, что человек спрятал.
 *
 * Предложение живёт ровно до тех пор, пока семейства нет на машине: как только
 * пак установлен, его модели становятся настоящими записями, и призрак рядом с
 * ними — враньё.
 *
 * @param {object} options
 * @param {Map} options.families семейства, которые есть здесь
 * @param {object[]} options.hidden скрытые семейства (выключены или выше уровня)
 * @param {object} options.catalog ответ роута паков
 * @param {string} options.locale "ru" | "en" — для имени пака из каталога сборки
 * @returns {object[]} записи вида {family, label, modes, packId, why?}
 */
export function buildOffers({ families, hidden, catalog, locale }) {
    const seen = new Set();
    const offers = [];
    // Скрытое семейство остаётся в списке серым: иначе выключение пака выглядит
    // как пропажа модели, и вернуть её человеку неоткуда.
    for (const item of hidden || []) {
        seen.add(item.family);
        offers.push({ ...item });
    }
    for (const pack of catalog?.packs || []) {
        if (pack.installed || pack.builtin) continue;
        for (const item of pack.families || []) {
            const name = typeof item === "string" ? item : item?.family;
            if (!name || families?.has?.(name) || seen.has(name)) continue;
            seen.add(name);
            const described = typeof item === "string"
                ? {
                    family: name,
                    label: pack.name?.[locale] || pack.name?.en || name,
                    modes: pack.modes || [],
                }
                : item;
            offers.push({ ...described, packId: pack.id });
        }
    }
    return offers;
}

/**
 * Семейства, у которых есть бэкенд под этот режим, — с ролями.
 *
 * Роль важна только в генерации: без референсов работает основной граф, с
 * референсом — граф редактирования. В перерисовке и апскейле такой пары нет, и
 * слоты референсов там появляться не должны только потому, что семейство их
 * где-то умеет.
 *
 * @param {Map} families семейства студии
 * @param {string[]} modes режимы бэкендов, годные для этого режима интерфейса
 * @returns {Map} family -> {family, primary, edit, label}
 */
export function rolesForModes(families, modes) {
    const out = new Map();
    for (const family of families.values()) {
        const found = modes.map((mode) => family.modes.get(mode)).filter(Boolean);
        if (!found.length) continue;
        const primary = family.modes.get(modes[0]) || found[0];
        const edit = modes.includes("edit") ? (family.modes.get("edit") || null) : null;
        out.set(family.family, { family, primary, edit, label: family.label });
    }
    return out;
}

/**
 * Предложенные семейства, которые обслужили бы этот режим.
 *
 * @param {object[]} offers результат buildOffers
 * @param {string[]} modes режимы бэкендов
 */
export function offersForModes(offers, modes) {
    return (offers || []).filter((offer) =>
        (offer.modes || []).some((mode) => modes.includes(mode)));
}

/**
 * Какой бэкенд реально уйдёт в очередь.
 *
 * В генерации ответ зависит от референсов: пусто — текст в картинку, заполнено
 * — граф редактирования этого же семейства. Одна вкладка, два бэкенда.
 *
 * @param {object} options
 * @param {string} options.mode режим интерфейса
 * @param {object} options.backend выбранный бэкенд
 * @param {Map} options.roles роли семейств этого режима
 * @param {object} options.refs значения слотов референсов
 */
export function backendForRun({ mode, backend, roles, refs }) {
    if (mode !== "generate" || !backend) return backend;
    const role = roles?.get?.(backend.manifest.family);
    const hasRef = Object.values(refs || {}).some(Boolean);
    if (hasRef && role?.edit?.available) return role.edit;
    if (!hasRef && role?.primary?.available) return role.primary;
    return backend;
}
