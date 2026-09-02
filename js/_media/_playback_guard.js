// Один сторож на всё воспроизведение пака: пауза на время прогона и пауза,
// когда ноду не видно.
//
// ⚠️ Зачем это вообще. Ноды пака показывают результат прямо на холсте, и видео
// в браузере декодирует ТА ЖЕ видеокарта, на которой считает ComfyUI. Пока
// ролик крутится, декодер и композитор отнимают карту у генерации — отсюда
// жалоба «после одной-двух генераций следующая идёт заметно медленнее, а если
// смотреть с другого компьютера, всё быстро». Смотреть с другого компьютера
// быстрее ровно потому, что декодирование уходит на чужой GPU.
//
// Сторож общий НАМЕРЕННО. У загрузчика наблюдатель видимости был написан
// давно, у сейвера и превью — нет; расходиться этим трём реализациям нельзя,
// а третья копия появилась бы при следующей медиа-ноде.
//
// Что он делает и чего НЕ делает:
//   • на старте прогона ставит на паузу всё, что играет;
//   • ставит на паузу то, что ушло за край экрана;
//   • НИЧЕГО не возобновляет сам. Решение «играть» принимает человек (или
//     нода, показывая свежий результат) — сторож не возвращает то, что снял.

import { api } from "/scripts/api.js";

/** @type {Set<{media:HTMLMediaElement, onHidden:(()=>void)|null, observer:IntersectionObserver|null}>} */
const guarded = new Set();
let listening = false;

function pauseEntry(entry) {
    try {
        entry.media?.pause?.();
    } catch (error) {
        console.warn("[TS PlaybackGuard] pause failed", error);
    }
}

function pauseEverything() {
    for (const entry of guarded) pauseEntry(entry);
}

function listenOnce() {
    if (listening) return;
    listening = true;
    // `execution_start` приходит, когда очередь взяла промпт в работу, — это
    // самый ранний момент, когда карта уже нужна вычислению.
    api.addEventListener("execution_start", pauseEverything);
    // Вкладку свернули или ушли на другую — играть незачем и здесь.
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) pauseEverything();
    });
}

/**
 * Взять медиа-элемент под присмотр.
 *
 * @param {object} options
 * @param {HTMLMediaElement} options.media  что ставить на паузу.
 * @param {HTMLElement} [options.watch]     за видимостью чего следить; по
 *   умолчанию сам медиа-элемент.
 * @param {() => void} [options.onHidden]   что ещё сделать, когда пропало из
 *   виду (отменить подгрузку миниатюр, например).
 * @returns {() => void} снять с присмотра — обязательно вызвать при удалении
 *   ноды, иначе наблюдатель переживёт её вместе со всем замыканием.
 */
export function guardPlayback({ media, watch, onHidden } = {}) {
    if (!media) return () => {};
    listenOnce();

    const entry = { media, onHidden: onHidden || null, observer: null };
    const target = watch || media;

    if (typeof IntersectionObserver === "function" && target) {
        entry.observer = new IntersectionObserver(([seen]) => {
            if (seen && !seen.isIntersecting) {
                pauseEntry(entry);
                try {
                    entry.onHidden?.();
                } catch (error) {
                    console.warn("[TS PlaybackGuard] onHidden failed", error);
                }
            }
        });
        entry.observer.observe(target);
    }

    guarded.add(entry);
    return () => {
        entry.observer?.disconnect();
        guarded.delete(entry);
    };
}

/** Сколько элементов под присмотром — для проверок. */
export function guardedCount() {
    return guarded.size;
}
