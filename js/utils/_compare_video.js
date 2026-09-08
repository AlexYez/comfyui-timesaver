// Шторка «до и после» для видео: ОДИН файл, в котором A лежит над B.
//
// ⚠️ Почему не два плеера. Два `<video>` расходятся на кадр-два на быстром
// движении — сравнение начинает врать, и заметить это нельзя. И второй декодер
// бьёт туда же, куда била жалоба, ради которой в 12.5.0 появился сторож: видео
// в браузере декодирует ТА ЖЕ карта, на которой считает ComfyUI.
//
// Здесь один элемент `<video>` высотой 2H: верхняя половина — A, нижняя — B.
// Рассинхрона не бывает физически, это один кадр. Обе половины рисуются на
// канвас: слева от шторки верхняя, справа нижняя.
//
// Родных controls у канваса нет, поэтому полоса воспроизведения своя.

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";
import { guardPlayback } from "../_media/_playback_guard.js";

const STYLE_ID = "ts-compare-video-styles";

export function ensureCompareVideoStyles() {
    // ⚠️ Тема — первой строкой, до раннего возврата (§12.6).
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-cmpv{display:flex;flex-direction:column;gap:6px;height:100%;min-height:0}
.ts-cmpv__stage{
  /* ⚠️ min-width:0 ОБЯЗАТЕЛЕН, и без обратных кавычек в комментарии:
     весь блок стилей — шаблонная строка, и кавычка внутри рвёт её целиком. Без него у flex-элемента ширина не может стать
     меньше содержимого, и растянутый после полного экрана канвас НЕ ДАВАЛ сцене
     сжаться: пересчёт мерил его же размер и оставлял всё как есть. Замерено —
     кадр возвращался в ноду размером 1450×816 вместо 300×150. */
  flex:1 1 auto;min-width:0;min-height:0;position:relative;display:flex;
  align-items:center;justify-content:center;overflow:hidden;
  border:1px solid var(--ts-border);border-radius:var(--ts-radius);
  /* Поля вокруг кадра НАМЕРЕННО чёрные: цвет темы по краям видео читается
     как часть картинки и мешает судить о ней. */
  background:#000;
  cursor:ew-resize;touch-action:none}
/* Канвас ВЫНУТ ИЗ ПОТОКА намеренно: иначе он участвует в измерении своей же
   сцены — растянувшись однажды, не даёт ей сжаться обратно, и пересчёт
   возвращает тот же размер. Центрируется полями. */
.ts-cmpv__canvas{
  display:block;position:absolute;inset:0;margin:auto;
  max-width:100%;max-height:100%}
.ts-cmpv__handle{
  position:absolute;top:0;width:2px;background:var(--ts-accent);
  pointer-events:none;transform:translateX(-1px)}
.ts-cmpv__tag{
  position:absolute;top:10px;padding:4px 12px;border-radius:999px;
  /* Подписи читают с расстояния, глядя на картинку, а не на текст: мелкий
     служебный кегль здесь не годится. */
  font-size:15px;font-weight:600;letter-spacing:.01em;pointer-events:none;
  /* Подписи лежат ПОВЕРХ кадра и обязаны читаться на любом: тёмная плашка со
     светлым текстом — единственное, что работает и на белом, и на чёрном. */
  background:rgba(0,0,0,.55);color:#fff}
.ts-cmpv__tag--a{left:10px}
.ts-cmpv__tag--b{right:10px}
.ts-cmpv__bar{display:flex;align-items:center;gap:8px;flex:0 0 auto}
.ts-cmpv__time{
  font-size:var(--ts-fs-xs);color:var(--ts-muted);white-space:nowrap;
  font-variant-numeric:tabular-nums}
.ts-cmpv__seek{flex:1 1 auto;min-width:0}
`;
    document.head.appendChild(style);
}

function clock(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
    const whole = Math.floor(seconds);
    return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
}

/**
 * Шторка по сложенному видео.
 *
 * @param {object} strings подписи сторон и кнопки.
 * @returns {{element:HTMLElement, show:Function, teardown:Function}}
 */
export function createVideoCompare(strings = {}) {
    ensureCompareVideoStyles();

    const element = document.createElement("div");
    element.className = `${TS_UI_CLASS} ts-cmpv`;

    const stage = document.createElement("div");
    stage.className = "ts-cmpv__stage";

    const canvas = document.createElement("canvas");
    canvas.className = "ts-cmpv__canvas";
    const ctx = canvas.getContext("2d");

    const handle = document.createElement("div");
    handle.className = "ts-cmpv__handle";

    const tagA = document.createElement("div");
    tagA.className = "ts-cmpv__tag ts-cmpv__tag--a";
    const tagB = document.createElement("div");
    tagB.className = "ts-cmpv__tag ts-cmpv__tag--b";

    stage.append(canvas, handle, tagA, tagB);

    // Сам источник на экран не попадает: его половины рисует канвас.
    const video = document.createElement("video");
    video.style.display = "none";
    // ⚠️ `auto`, а не `metadata`. С метаданными известны размеры, но НИ ОДНОГО
    // декодированного кадра ещё нет, и канвас до первого нажатия Play оставался
    // чёрным — ровно та жалоба. Нужен `loadeddata`, а он приходит только когда
    // кадр действительно есть.
    video.preload = "auto";
    video.playsInline = true;
    video.loop = false;
    stage.appendChild(video);

    const bar = document.createElement("div");
    bar.className = "ts-cmpv__bar ts-ui-toolbar";

    const play = document.createElement("button");
    play.type = "button";
    play.className = "ts-ui-btn";
    play.textContent = strings.play || "Play";
    play.title = strings.playHint || strings.play || "Play";

    const seek = document.createElement("input");
    seek.type = "range";
    seek.className = "ts-ui-slider ts-cmpv__seek";
    seek.min = "0";
    seek.max = "1000";
    seek.value = "0";
    seek.title = strings.seekHint || "";

    const time = document.createElement("span");
    time.className = "ts-cmpv__time";
    time.textContent = "00:00";

    const expand = document.createElement("button");
    expand.type = "button";
    expand.className = "ts-ui-btn ts-ui-btn--icon";
    expand.textContent = "⛶";
    expand.title = strings.fullscreen || "Fullscreen";

    bar.append(play, seek, time, expand);
    element.append(stage, bar);

    let split = 0.5;
    let frameW = 0;
    let frameH = 0;
    let looping = false;

    function layout() {
        if (!(frameW > 0 && frameH > 0)) return;
        const boxW = stage.clientWidth;
        const boxH = stage.clientHeight;
        if (!(boxW > 0 && boxH > 0)) return;
        // ⚠️ clientWidth/clientHeight, а не getBoundingClientRect: они не зависят
        // от масштаба холста ComfyUI (§12.5.3).
        const scale = Math.min(boxW / frameW, boxH / frameH);
        canvas.width = Math.max(2, Math.round(frameW * scale));
        canvas.height = Math.max(2, Math.round(frameH * scale));
        paint();
    }

    function paint() {
        if (!ctx || !(frameW > 0 && frameH > 0)) return;
        if (!video.videoWidth) return;
        const w = canvas.width;
        const h = canvas.height;
        const cut = Math.round(w * split);

        // Верхняя половина источника — сторона A, во всю ширину.
        ctx.drawImage(video, 0, 0, frameW, frameH, 0, 0, w, h);

        // Нижняя половина — сторона B, только справа от шторки.
        if (cut < w) {
            ctx.save();
            ctx.beginPath();
            ctx.rect(cut, 0, w - cut, h);
            ctx.clip();
            ctx.drawImage(video, 0, frameH, frameW, frameH, 0, 0, w, h);
            ctx.restore();
        }

        // Ручка идёт по краю кадра, а не сцены: канвас центрирован полями.
        handle.style.left = `${canvas.offsetLeft + cut}px`;
        handle.style.top = `${canvas.offsetTop}px`;
        handle.style.height = `${canvas.offsetHeight}px`;
    }

    /** Пока играет — рисуем каждый кадр источника, а не каждый кадр экрана. */
    function pump() {
        paint();
        if (video.paused || video.ended) {
            looping = false;
            return;
        }
        if (typeof video.requestVideoFrameCallback === "function") {
            video.requestVideoFrameCallback(pump);
        } else {
            requestAnimationFrame(pump);
        }
    }

    function startPump() {
        if (looping) return;
        looping = true;
        pump();
    }

    function setSplitFromPointer(event) {
        const rect = canvas.getBoundingClientRect();
        if (!(rect.width > 0)) return;
        const ratio = (event.clientX - rect.left) / rect.width;
        split = Math.min(1, Math.max(0, ratio));
        paint();
    }

    let dragging = false;
    stage.addEventListener("pointerdown", (event) => {
        dragging = true;
        stage.setPointerCapture?.(event.pointerId);
        setSplitFromPointer(event);
    });
    stage.addEventListener("pointermove", (event) => {
        if (dragging) setSplitFromPointer(event);
    });
    const endDrag = (event) => {
        dragging = false;
        stage.releasePointerCapture?.(event.pointerId);
    };
    stage.addEventListener("pointerup", endDrag);
    stage.addEventListener("pointercancel", endDrag);

    play.addEventListener("click", () => {
        if (video.paused) video.play().catch(() => {});
        else video.pause();
    });

    seek.addEventListener("input", () => {
        if (!Number.isFinite(video.duration) || video.duration <= 0) return;
        video.currentTime = (Number(seek.value) / 1000) * video.duration;
    });

    video.addEventListener("play", () => {
        play.textContent = strings.pause || "Pause";
        startPump();
    });
    video.addEventListener("pause", () => {
        play.textContent = strings.play || "Play";
        paint();
    });
    video.addEventListener("seeked", paint);
    video.addEventListener("timeupdate", () => {
        const duration = video.duration;
        if (Number.isFinite(duration) && duration > 0) {
            seek.value = String(Math.round((video.currentTime / duration) * 1000));
            time.textContent = `${clock(video.currentTime)} / ${clock(duration)}`;
        }
    });
    video.addEventListener("loadedmetadata", () => {
        // Источник вдвое выше кадра: сверху A, снизу B.
        frameW = video.videoWidth;
        frameH = Math.floor(video.videoHeight / 2);
        layout();
        time.textContent = `00:00 / ${clock(video.duration)}`;
    });
    // ⚠️ Вот здесь появляется первый кадр. До этого события рисовать нечего:
    // `drawImage` с пустого видео даёт пустоту, и человек видит чёрный экран.
    video.addEventListener("loadeddata", () => {
        layout();
        paint();
    });

    const observer = typeof ResizeObserver === "function"
        ? new ResizeObserver(() => layout())
        : null;
    observer?.observe(stage);
    // ⚠️ Следим И за корнем: во весь экран элемент ПЕРЕЕЗЖАЕТ в оверлей, и
    // размер меняется у него, а не у сцены — без этого кадр остаётся размером
    // с ноду посреди пустого экрана.
    observer?.observe(element);

    // Пауза на старте прогона и когда ноду не видно; сам сторож не возобновляет.
    const unguard = guardPlayback({ media: video, watch: element });

    return {
        element,
        /** Кнопка «во весь экран»: обработчик вешает нода, у неё есть оверлей. */
        expandButton: expand,
        /** Показать сложенный файл. */
        show(url, { labelA, labelB } = {}) {
            tagA.textContent = labelA || strings.before || "A";
            tagB.textContent = labelB || strings.after || "B";
            split = 0.5;
            if (video.src !== url) {
                video.pause();
                video.src = url;
                video.load();
            }
        },
        /** Перерисовать по месту — например, когда ноду изменили в размере. */
        relayout: layout,
        teardown() {
            unguard();
            observer?.disconnect();
            try {
                video.pause();
                video.removeAttribute("src");
                video.load();
            } catch (error) {
                console.warn("[TS Compare] could not release the video", error);
            }
        },
    };
}
