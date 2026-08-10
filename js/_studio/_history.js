// TS Studio kit — история версий кадра (core layer, no DOM of its own).
//
// Одна и та же работа повторялась в студии дважды и вот-вот повторилась бы в
// третий раз: инпэйнт умеет откатывать свои проходы и сохранять результат по
// кнопке, а апскейл и перерисовка — нет, хотя итерируют ровно так же: прогнал,
// посмотрел, не понравилось — вернулся и прогнал иначе.
//
// ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. Правило «в библиотеку попадает только то, что человек
// оставил» — это не деталь интерфейса, а поведение всей студии. Оно должно
// звучать в одном месте, иначе следующий раздел заведёт свою версию и разойдётся
// с остальными: где-то Save, где-то автосохранение, где-то откат теряет
// черновик. Здесь — чистое состояние без DOM, чтобы его можно было проверить
// без браузера и подключить к любой поверхности.
//
// МОДЕЛЬ. Линейка версий и указатель на текущую — как в любом редакторе:
//
//     v0 ──► v1 ──► v2            cursor = 2
//            ▲ undo   ▼ redo
//
// Новая версия поверх отката отрезает «будущее»: человек выбрал другую ветку, и
// хранить прошлую было бы обещанием вернуться, которого никто не давал.
//
// ЧЕРНОВИК И СОХРАНЁННОЕ — РАЗНЫЕ ВЕЩИ. Прогон кладёт файл во временную папку
// (её не индексирует браузер ассетов), и версия помнит, откуда она взялась.
// `keep()` переносит текущую версию в библиотеку и помечает её сохранённой:
// второй раз то же самое не сохранится, а откат к уже сохранённой версии не
// предложит сохранить её снова.

/**
 * Версия кадра.
 *
 * @typedef {object} HistoryVersion
 * @property {string} url        Что показывать (адрес картинки).
 * @property {string} [annotated] Имя файла для повторного прогона («a.png [temp]»).
 * @property {object} [draft]    Что переносить в библиотеку: {filename, subfolder, type}.
 * @property {boolean} [kept]    Уже сохранена — переносить нечего.
 * @property {object} [meta]     Что угодно от раздела: подпись, параметры прогона.
 */

/**
 * Линейка версий кадра.
 *
 * @param {object} [options]
 * @param {number} [options.limit=40] Сколько версий держать. Старые уходят с
 *   начала: держать сотню кадров сессии в памяти незачем, а полсотни правок
 *   подряд никто не делает.
 * @param {(state: object) => void} [options.onChange] Состояние изменилось —
 *   поверхности пора перерисовать кнопки.
 * @returns {object} контракт истории
 */
export function createHistory({ limit = 40, onChange } = {}) {
    /** @type {HistoryVersion[]} */
    let versions = [];
    let cursor = -1;

    const state = () => ({
        current: versions[cursor] || null,
        index: cursor,
        total: versions.length,
        canUndo: cursor > 0,
        canRedo: cursor >= 0 && cursor < versions.length - 1,
        // Сохранять есть что, только пока текущая версия — черновик.
        canKeep: Boolean(versions[cursor]?.draft) && !versions[cursor]?.kept,
    });

    // Слушателей может быть несколько: панель кнопок подписывается сама, а
    // раздел — своим `onChange`. Подписка на стороне поверхности важнее, чем
    // кажется: иначе каждый новый экран обязан не забыть позвать `sync()`, и
    // однажды не позовёт — кнопки останутся от прошлой версии.
    const listeners = new Set();
    const announce = () => {
        onChange?.(state());
        for (const listener of listeners) listener(state());
    };

    return {
        /**
         * Положить версию поверх текущей.
         *
         * @param {HistoryVersion} version
         * @returns {object} состояние после записи
         */
        push(version) {
            if (!version?.url) throw new Error("[TS Studio] history needs a url");
            // Всё, что было «впереди», отрезается: человек выбрал другую ветку.
            versions = versions.slice(0, cursor + 1);
            versions.push({ ...version });
            if (versions.length > limit) versions = versions.slice(versions.length - limit);
            cursor = versions.length - 1;
            announce();
            return state();
        },

        /** Шаг назад. Возвращает версию, которую теперь показывать. */
        undo() {
            if (cursor > 0) cursor -= 1;
            announce();
            return versions[cursor] || null;
        },

        /** Шаг вперёд — пока впереди что-то есть. */
        redo() {
            if (cursor < versions.length - 1) cursor += 1;
            announce();
            return versions[cursor] || null;
        },

        /**
         * Пометить текущую версию сохранённой.
         *
         * Саму передачу в библиотеку делает вызывающий: у него есть роут и
         * галерея, а у истории — только знание, что сохранять уже нечего.
         *
         * @param {object} [image] Что вернул сервер: {filename, subfolder, type}.
         */
        markKept(image) {
            const version = versions[cursor];
            if (!version) return state();
            version.kept = true;
            if (image) {
                version.saved = { ...image };
                // Дальше версия живёт по сохранённому адресу: временный файл
                // переезжает, и старая ссылка однажды перестанет открываться.
                version.draft = { ...image };
            }
            announce();
            return state();
        },

        /**
         * Подписаться на изменения. Возвращает отписку.
         *
         * @param {(state: object) => void} listener
         * @returns {() => void}
         */
        subscribe(listener) {
            if (typeof listener !== "function") return () => {};
            listeners.add(listener);
            return () => listeners.delete(listener);
        },

        /** Черновик текущей версии — то, что предстоит сохранить. */
        draft: () => (versions[cursor]?.kept ? null : versions[cursor]?.draft || null),
        current: () => versions[cursor] || null,
        /** Первая версия: с чего всё началось — для шторки «до и после». */
        origin: () => versions[0] || null,
        state,

        /** Новая работа — новая история. */
        reset(version = null) {
            versions = version?.url ? [{ ...version }] : [];
            cursor = versions.length - 1;
            announce();
            return state();
        },

        /** Снимок для памяти рабочего места. */
        serialise: () => ({ versions: versions.map((v) => ({ ...v })), cursor }),
        restore(snapshot) {
            const list = Array.isArray(snapshot?.versions) ? snapshot.versions : [];
            versions = list.filter((v) => v?.url).map((v) => ({ ...v }));
            const at = Number(snapshot?.cursor);
            cursor = Number.isInteger(at) && at >= 0 && at < versions.length
                ? at : versions.length - 1;
            announce();
            return state();
        },
    };
}
