// Студию закрыли — студия открылась там же (core layer, no DOM).
//
// `_memory.js` помнит ЗНАЧЕНИЯ контролов: сколько шагов, какая сила, какой
// промпт. Этого мало. Когда окно закрывают и открывают снова, теряется всё
// остальное: какая вкладка была выбрана, на какой модели работали, какая
// картинка лежала на холсте и что на ней нарисовано. Человек возвращается к
// пустому экрану и собирает рабочее место заново.
//
// Здесь хранится ровно один объект — снимок рабочего места, того же формата,
// что студия и так пишет в PNG (`buildStudioState` в `_pnginfo.js`). Формат
// общий не для экономии: у студии уже есть проверенный путь «применить
// снимок» (`applyStudioState`), который умеет выставить вкладку, модель, все
// контролы и вернуть исходники на холст. Открытие после закрытия — это тот же
// путь, что и «Recreate» с готовой картинки.
//
// Снимок привязан к сессии: у ноды на графе она своя и живёт в самой ноде, так
// что два разных узла не перетирают друг другу рабочее место. Студия, открытая
// без ноды (из браузера ассетов), получает новую сессию каждый раз — ей
// отдаётся последний снимок вообще, иначе она бы всегда открывалась пустой.

const STORAGE_KEY = "ts.studio.workspace";
const VERSION = 1;

// Сколько рабочих мест держать. Больше — только мусор: к сессии недельной
// давности не возвращаются, а место в localStorage общее на весь ComfyUI.
const MAX_SITTINGS = 8;

// Нарисованная маска едет в снимке картинкой в data-URL, и она может быть
// большой. Выше этого порога маска не сохраняется: рабочее место важнее одного
// мазка, а переполненный localStorage не сохранит ни того, ни другого.
const MAX_MASK_BYTES = 1_500_000;

function read() {
    try {
        const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
        if (!raw || raw.v !== VERSION || typeof raw.sittings !== "object") return null;
        return raw;
    } catch (err) {
        console.warn("[TS Studio] workspace unreadable", err);
        return null;
    }
}

function write(data) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
        return true;
    } catch (err) {
        // Место кончилось — сбрасываем всё, кроме текущего рабочего места.
        // Потерять старые сессии не жалко, потерять открытую — жалко.
        try {
            const newest = (data.order || []).slice(-1)[0];
            const slim = { v: VERSION, order: newest ? [newest] : [],
                           sittings: newest ? { [newest]: data.sittings[newest] } : {} };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(slim));
            return true;
        } catch (inner) {
            console.warn("[TS Studio] workspace not saved", inner);
            return false;
        }
    }
}

/**
 * Запомнить рабочее место сессии.
 *
 * @param {string} sessionId сессия студии; пустая строка не сохраняется
 * @param {object} state снимок формата `buildStudioState`
 * @param {string} [maskDataUrl] нарисованная маска, если она есть
 */
export function saveWorkspace(sessionId, state, maskDataUrl = "") {
    if (!sessionId || !state || typeof state !== "object") return false;
    const data = read() || { v: VERSION, order: [], sittings: {} };
    const entry = { state, at: Date.now() };
    if (maskDataUrl && maskDataUrl.length <= MAX_MASK_BYTES) entry.mask = maskDataUrl;
    data.sittings[sessionId] = entry;
    data.order = [...(data.order || []).filter((id) => id !== sessionId), sessionId];
    while (data.order.length > MAX_SITTINGS) {
        delete data.sittings[data.order.shift()];
    }
    return write(data);
}

/**
 * Рабочее место сессии. `fallbackToLast` — для студии без ноды: у неё сессия
 * каждый раз новая, и без этого она открывалась бы пустой всегда.
 *
 * @returns {{state: object, mask?: string, at: number}|null}
 */
export function loadWorkspace(sessionId, fallbackToLast = false) {
    const data = read();
    if (!data) return null;
    const own = sessionId ? data.sittings[sessionId] : null;
    if (own) return own;
    if (!fallbackToLast) return null;
    const last = (data.order || []).slice(-1)[0];
    return last ? data.sittings[last] || null : null;
}

/** Забыть одно рабочее место — например, когда ноду удалили. */
export function forgetWorkspace(sessionId) {
    const data = read();
    if (!data || !data.sittings[sessionId]) return;
    delete data.sittings[sessionId];
    data.order = (data.order || []).filter((id) => id !== sessionId);
    write(data);
}

/** Забыть все — для настроек и тестов. */
export function forgetAllWorkspaces() {
    try { localStorage.removeItem(STORAGE_KEY); }
    catch (err) { console.warn("[TS Studio] workspace not cleared", err); }
}

export const WORKSPACE_KEY_FOR_TEST = STORAGE_KEY;
export const MAX_SITTINGS_FOR_TEST = MAX_SITTINGS;
