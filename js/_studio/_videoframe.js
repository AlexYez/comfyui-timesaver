// Видео, брошенное в студию: какой кадр взять — первый или последний.
//
// Студия работает с картинками, но приносят ей и ролики: из библиотеки Artius,
// из папки, из превью ноды. Отказывать в этом случае глупо — почти всегда
// человеку нужен ровно один кадр: первый (с чего начинается) или последний
// (чем закончилось). Спросить об этом — одно движение, угадать — нельзя.
//
// ПОЧЕМУ КАДР БЕРЁТСЯ В БРАУЗЕРЕ. Ролик и так уже доступен по HTTP, а <video> и
// холст умеют отдать кадр без единого запроса на сервер. Путь через ffmpeg на
// бэкенде означал бы новый роут, перезапуск ComfyUI ради него и ожидание в
// секундах вместо мгновенного результата. Если браузер формат не понимает — об
// этом честно сообщается, и человек конвертирует ролик сам.
//
// ⚠️ Последний кадр берётся не по `duration`: у многих контейнеров последний
// кадр по этому времени уже пуст, и на холст попадает чернота. Отступаем назад
// на кадр (0.04 с) и, если холст оказался пустым, отступаем ещё.

const LAST_FRAME_BACKOFF = [0.04, 0.12, 0.35];

/** До этой длины ролик проще досмотреть, чем доверять его заголовку. */
const LONG_CLIP_SECONDS = 12;

/** Расширения, по которым дроп считается видео, когда MIME не пришёл. */
const VIDEO_EXTENSIONS = /\.(mp4|webm|mov|m4v|mkv|avi|gif)$/i;

/**
 * Видео ли это.
 *
 * @param {{type?: string, name?: string, mime?: string}} item элемент дропа
 * @returns {boolean}
 */
export function isVideoItem(item) {
    if (!item) return false;
    if (item.type === "video") return true;
    if (String(item.mime || "").startsWith("video/")) return true;
    return VIDEO_EXTENSIONS.test(String(item.name || ""));
}

/**
 * Достать кадр из ролика.
 *
 * @param {Blob|string} source блоб ролика или его адрес
 * @param {"first"|"last"} which какой кадр
 * @returns {Promise<Blob>} PNG с кадром
 */
export async function extractFrame(source, which = "first") {
    const url = typeof source === "string" ? source : URL.createObjectURL(source);
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.crossOrigin = "anonymous";
    try {
        await new Promise((resolve, reject) => {
            video.addEventListener("loadeddata", resolve, { once: true });
            video.addEventListener("error", () => reject(
                new Error("browser cannot decode this video")), { once: true });
            video.src = url;
        });
        if (which !== "last") return (await grab(video, 0)).blob;

        const duration = await realDuration(video);
        // ⚠️ Короткий ролик ДОСМАТРИВАЕМ, а не перематываем, даже когда
        // длительность вроде бы известна. У записи из браузера она врёт: в
        // заголовке стоит длина первого куска, перемотка по ней приводит в
        // начало, и «последний кадр» оказывался первым (поймано живым тестом).
        // Просмотр — единственный источник правды, и для секундного ролика он
        // стоит доли секунды.
        if (duration <= 0 || duration <= LONG_CLIP_SECONDS) {
            await playToEnd(video);
            const shot = await snapshot(video);
            if (shot.filled || duration <= 0) return shot.blob;
        }
        if (duration > 0) {
            let last = null;
            for (const back of LAST_FRAME_BACKOFF) {
                const frame = await grab(video, Math.max(0, duration - back));
                if (frame.filled) return frame.blob;
                last = frame.blob;              // пустой кадр — отступим ещё
            }
            if (last) return last;
        }
        // Длительность неизвестна — досматриваем ролик до конца и берём то, на
        // чём он остановился. Так ведут себя записи из браузера (webm из
        // MediaRecorder) и часть потоковых файлов: у них нет ни `duration`, ни
        // области перемотки, и любой seek возвращает начало. Поймано живым
        // тестом: «последний кадр» приносил первый.
        await playToEnd(video);
        return (await snapshot(video)).blob;
    } finally {
        video.src = "";
        if (typeof source !== "string") URL.revokeObjectURL(url);
    }
}

/**
 * Настоящая длительность ролика.
 *
 * ⚠️ У потоковой записи (webm из MediaRecorder, многие ролики из сети)
 * `duration` равна Infinity или нулю, пока браузер не досмотрит файл до конца.
 * Заставляем его это сделать: перемотка «в бесконечность» упирается в конец, и
 * `currentTime` показывает, где он. Без этого «последний кадр» брался из
 * начала — поймано живым тестом.
 */
async function realDuration(video) {
    const known = Number(video.duration);
    if (Number.isFinite(known) && known > 0) return known;
    await new Promise((resolve) => {
        const done = () => resolve();
        video.addEventListener("seeked", done, { once: true });
        video.addEventListener("timeupdate", done, { once: true });
        video.currentTime = 1e7;
        setTimeout(done, 1500);          // повреждённый файл не должен подвесить
    });
    const reached = Number(video.currentTime);
    return Number.isFinite(reached) && reached > 0 ? reached : 0;
}

/** Досмотреть ролик до конца — когда перемотать его нельзя. */
async function playToEnd(video) {
    await new Promise((resolve) => {
        const done = () => resolve();
        video.addEventListener("ended", done, { once: true });
        video.playbackRate = 4;               // короткий ролик — доли секунды
        const started = video.play();
        if (started && typeof started.catch === "function") started.catch(done);
        // Ролик без конца не должен держать человека: шесть секунд — потолок.
        setTimeout(done, 6000);
    });
    video.pause();
}

/** Снять то, что сейчас на кадре. */
async function snapshot(video) {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 0;
    canvas.height = video.videoHeight || 0;
    if (!canvas.width || !canvas.height) throw new Error("video has no picture");
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    return { blob, filled: notBlank(context, canvas) };
}

/** Перемотать и снять кадр; заодно сказать, есть ли на нём хоть что-то. */
async function grab(video, time) {
    await new Promise((resolve) => {
        const done = () => resolve();
        video.addEventListener("seeked", done, { once: true });
        // Перемотка в уже текущую позицию события не даёт — подстраховываемся.
        if (Math.abs(video.currentTime - time) < 0.001) {
            video.removeEventListener("seeked", done);
            resolve();
            return;
        }
        video.currentTime = time;
    });
    return snapshot(video);
}

/**
 * Есть ли на кадре хоть что-то, кроме одного сплошного цвета.
 *
 * Считаем по редкой выборке: полный обход кадра 4K ради такой проверки — это
 * десятки миллисекунд на каждый кадр, а нужен только ответ «не чернота ли».
 */
function notBlank(context, canvas) {
    const step = Math.max(1, Math.floor(canvas.width / 32));
    const row = Math.max(1, Math.floor(canvas.height / 2));
    const data = context.getImageData(0, row, canvas.width, 1).data;
    let first = null;
    for (let x = 0; x < canvas.width; x += step) {
        const at = x * 4;
        const value = data[at] + data[at + 1] + data[at + 2];
        if (first === null) first = value;
        else if (Math.abs(value - first) > 12) return true;
    }
    return first !== null && first > 24;      // не чёрное поле
}

/**
 * Пункты меню «какой кадр взять» — для общего меню студии (`_ctxmenu.js`).
 *
 * Возвращаем данные, а не рисуем своё окно: у студии уже есть меню, и второй
 * его вид рядом с первым выглядел бы как чужая деталь.
 *
 * @param {object} options
 * @param {{en?: string, ru?: string}|object} options.strings подписи
 *   {first, last, hint}
 * @param {(which: "first"|"last") => void|Promise} options.onPick
 * @returns {object[]} пункты для `contextMenu.open()`
 */
export function frameChoiceItems({ strings, onPick }) {
    return [
        {
            id: "video.frame.first",
            group: "video",
            groupOrder: 0,
            order: 0,
            label: strings.first,
            hint: strings.hint || "",
            disabled: false,
            run: () => onPick("first"),
        },
        {
            id: "video.frame.last",
            group: "video",
            groupOrder: 0,
            order: 1,
            label: strings.last,
            hint: "",
            disabled: false,
            run: () => onPick("last"),
        },
    ];
}
