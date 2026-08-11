// Редактор подрезки: таймлайн, плеер и жесты.
//
// Ничего не знает о ноде — только о своём элементе и о том, что ему сказали
// снаружи. Поэтому его же можно перевесить в полноэкранный оверлей, не
// пересоздавая: состояние живёт в этом замыкании, вместе с буфером видео.
//
// ⚠️ ДВА СЛОЯ ХОЛСТА. Нижний (плёнка, звук, шкала) перерисовывается только при
// смене окна или приходе данных; верхний (выделение, ручки, плейхед) — каждый
// кадр воспроизведения. Иначе каждое движение плейхеда перерисовывало бы сотню
// миниатюр.

import { getThemeColors } from "../../_theme.js";
import { createStripSource, pickStep } from "../../_media/_filmstrip.js";
import { createPeakSource } from "../../_media/_peaks.js";
import { createPlayback } from "../../_media/_playback.js";
import {
    HANDLE_ACTIVE_CURSOR,
    HANDLE_CURSOR,
    createRangeDrag,
    hitTestHandle,
    normaliseRange,
} from "../../_media/_range.js";
import { formatTimecode, pickTickStep } from "../../_media/_ruler.js";
import { clamp, createTimeViewport } from "../../_media/_viewport.js";

const NEED_BASE = 1;
const NEED_OVERLAY = 2;
const NEED_ALL = NEED_BASE | NEED_OVERLAY;

const LANE_RULER = 16;
// Звуковая дорожка — это инструмент попадания в такт и в реплику, а не
// украшение: слишком тонкая читается хуже, чем не рисуется вовсе.
const LANE_WAVE = 40;

export function createVideoEditor({ api, route, strings, onRangeChange, onViewportChange }) {
    const L = strings;

    // ── разметка ─────────────────────────────────────────────────────────── #
    const root = document.createElement("div");
    root.className = "ts-vid";

    const stage = document.createElement("div");
    stage.className = "ts-vid__stage";
    stage.title = L.stageHint;
    const video = document.createElement("video");
    video.className = "ts-vid__video";
    video.playsInline = true;
    video.muted = true;
    // ⚠️ "none" до тех пор, пока метаданные не подтвердят faststart: у файла без
    // него браузер ради индекса утягивает весь ролик целиком.
    video.preload = "none";
    const empty = document.createElement("div");
    empty.className = "ts-vid__empty";
    empty.textContent = L.dropHint;
    const badge = document.createElement("div");
    badge.className = "ts-vid__badge";
    stage.append(video, empty, badge);

    const timeline = document.createElement("div");
    timeline.className = "ts-vid__timeline";
    // ⚠️ Подсказки у таймлайна НЕТ намеренно: она всплывает ровно там, где идёт
    // работа, и закрывает строку состояния — а в ней как раз написано, сколько
    // кадров выйдет. Как тянуть ручки, рассказано во встроенной справке.
    const base = document.createElement("canvas");
    base.className = "ts-vid__canvas";
    const overlay = document.createElement("canvas");
    overlay.className = "ts-vid__canvas ts-vid__canvas--overlay";
    timeline.append(base, overlay);

    const scroll = document.createElement("div");
    scroll.className = "ts-vid__scroll";
    const thumb = document.createElement("div");
    thumb.className = "ts-vid__thumb";
    scroll.appendChild(thumb);

    // ── состояние ────────────────────────────────────────────────────────── #
    const state = {
        path: "",
        duration: 0,
        fps: 0,
        width: 0,
        height: 0,
        hasAudio: false,
        playable: true,
        cropStart: 0,
        cropEnd: -1,
        looping: true,
        showWave: true,
        ready: false,
    };

    const viewport = createTimeViewport({
        getDuration: () => state.duration,
        getFps: () => state.fps,
        onChange: () => { onViewportChange?.(viewport.toJSON()); scheduleDraw(NEED_ALL); },
    });

    const strip = createStripSource({
        api,
        route,
        getPath: () => state.path,
        getHeight: () => Math.max(24, laneGeometry().film),
        onReady: () => scheduleDraw(NEED_BASE),
    });

    const peaks = createPeakSource({
        api,
        route,
        getPath: () => state.path,
        onReady: () => scheduleDraw(NEED_BASE),
    });

    const playback = createPlayback(video, {
        getRange: () => bounds(),
        isLooping: () => state.looping,
        getFps: () => state.fps,
        onTime: () => scheduleDraw(NEED_OVERLAY),
        onState: () => scheduleDraw(NEED_OVERLAY),
    });

    // ── геометрия ────────────────────────────────────────────────────────── #
    function laneGeometry() {
        const height = timeline.clientHeight || 104;
        const wave = state.showWave && state.hasAudio ? LANE_WAVE : 0;
        return { ruler: LANE_RULER, film: Math.max(20, height - LANE_RULER - wave), wave };
    }

    function bounds() {
        return normaliseRange(state.cropStart, state.cropEnd, state.duration);
    }

    /**
     * Подготовить холст под текущий размер.
     *
     * ⚠️ Плотность считается с поправкой на масштаб родителя (зум графа в
     * Nodes 1.0, transform ноды в Vue), иначе на приближённом холсте картинка
     * мылится. Потолок в 3 не даёт выделить полотно в двенадцать раз больше
     * нужного при зуме 400 %.
     */
    function syncCanvas(canvas) {
        const cssWidth = canvas.clientWidth || 1;
        const cssHeight = canvas.clientHeight || 1;
        const rect = canvas.getBoundingClientRect();
        const parentScale = cssWidth > 0 ? rect.width / cssWidth : 1;
        const ratio = clamp((window.devicePixelRatio || 1) * parentScale, 1, 3);
        const width = Math.round(cssWidth * ratio);
        const height = Math.round(cssHeight * ratio);
        if (canvas.width !== width) canvas.width = width;
        if (canvas.height !== height) canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, cssWidth, cssHeight);
        return { ctx, width: cssWidth, height: cssHeight };
    }

    /** Секунда под указателем. Считается долями — обе величины из одной системы. */
    function pointerSeconds(event) {
        const rect = timeline.getBoundingClientRect();
        const fraction = clamp((event.clientX - rect.left) / (rect.width || 1), 0, 1);
        return viewport.viewStart + fraction * viewport.getViewSeconds();
    }

    function secondsPerPixel() {
        const width = timeline.clientWidth || 1;
        return viewport.getViewSeconds() / width;
    }

    // ── отрисовка ────────────────────────────────────────────────────────── #
    let pending = 0;
    let rafId = 0;

    function scheduleDraw(what) {
        pending |= what;
        if (rafId) return;
        rafId = requestAnimationFrame(() => {
            rafId = 0;
            const todo = pending;
            pending = 0;
            if (todo & NEED_BASE) drawBase();
            if (todo & NEED_OVERLAY) drawOverlay();
        });
    }

    function drawBase() {
        const { ctx, width, height } = syncCanvas(base);
        const colors = getThemeColors();
        const lanes = laneGeometry();

        ctx.fillStyle = colors.sunken;
        ctx.fillRect(0, 0, width, height);

        drawRuler(ctx, width, lanes, colors);
        drawFilm(ctx, width, lanes, colors);
        if (lanes.wave > 0) drawWave(ctx, width, lanes, colors);

        if (!state.ready && state.duration > 0) {
            state.ready = true;
            root.dataset.tsReady = "1";
        }
    }

    function drawRuler(ctx, width, lanes, colors) {
        const view = viewport.getViewSeconds();
        if (!(view > 0)) return;
        const step = pickTickStep(view, width, 72);
        const first = Math.floor(viewport.viewStart / step) * step;

        ctx.fillStyle = colors.bg;
        ctx.fillRect(0, 0, width, lanes.ruler);
        // ⚠️ Шрифт на холсте задаётся конкретным списком: CSS-переменные тут не
        // работают, и `var(--ts-font)` молча откатился бы к системному.
        ctx.font = "9px system-ui, -apple-system, Segoe UI, sans-serif";
        ctx.textBaseline = "middle";

        for (let time = first; time <= viewport.getViewEnd() + step; time += step) {
            const x = viewport.secondsToX(time, width);
            if (x < -40 || x > width + 40) continue;
            ctx.fillStyle = colors.faint;
            ctx.fillRect(Math.round(x), 0, 1, lanes.ruler);
            ctx.fillStyle = colors.muted;
            ctx.fillText(formatTimecode(time, { compact: true }).replace(/\.\d+$/, ""),
                         Math.round(x) + 3, lanes.ruler / 2);
        }
    }

    // Пропорции нарисованной карточки — наружу, в разметку. Перекос на ленте
    // видно только глазом и только на подходящем ролике, поэтому число, по
    // которому его можно поймать проверкой, выносится в атрибут. Пишем лишь при
    // изменении: отрисовка идёт каждый кадр воспроизведения.
    let lastCardGeometry = "";

    function reportCardGeometry(cell, cardWidth, laneHeight) {
        if (!cell || !(laneHeight > 0) || !(cell.sh > 0)) return;
        const drawn = cardWidth / laneHeight;
        const source = cell.sw / cell.sh;
        const value = `${drawn.toFixed(4)}/${source.toFixed(4)}`;
        if (value === lastCardGeometry) return;
        lastCardGeometry = value;
        timeline.dataset.tsCard = value;
    }

    function drawFilm(ctx, width, lanes, colors) {
        const view = viewport.getViewSeconds();
        if (!(view > 0) || lanes.film <= 0) return;

        // ⚠️ ДВА РАЗНЫХ ШАГА, и путать их нельзя.
        //
        // Шаг ОТРИСОВКИ непрерывен и равен ровно ширине карточки: карточка имеет
        // форму кадра (высота дорожки × пропорции ролика), стоит вплотную к
        // соседней и покрывает ровно свой отрезок времени. Поэтому нет ни щелей,
        // ни налезаний — а налезание как раз и превращало широкий ролик в
        // «квадраты»: от каждой карточки наружу торчала только узкая полоска,
        // остальное закрывала следующая.
        //
        // Шаг ЗАГРУЗКИ берётся с фиксированной лестницы: по нему устроен кэш, и
        // непрерывный шаг обнулял бы его при каждом движении колеса. Картинка
        // берётся с ближайшего к нужному моменту кадра этой лестницы — так и
        // делают ленты в монтажках.
        const aspect = (state.width > 0 && state.height > 0)
            ? state.width / state.height : 16 / 9;
        const cardWidth = Math.max(8, Math.round(lanes.film * aspect));
        const drawStep = (view * cardWidth) / Math.max(1, width);
        const fetchStep = pickStep(view, width, cardWidth);
        const height = Math.max(24, lanes.film);

        const firstCard = Math.floor(viewport.viewStart / drawStep);
        const lastCard = Math.ceil(viewport.getViewEnd() / drawStep);
        let drawnCell = null;

        ctx.save();
        ctx.beginPath();
        ctx.rect(0, lanes.ruler, width, lanes.film);
        ctx.clip();
        ctx.fillStyle = colors.bg;
        ctx.fillRect(0, lanes.ruler, width, lanes.film);

        for (let card = firstCard; card <= lastCard; card += 1) {
            const time = card * drawStep;
            const x = viewport.secondsToX(time, width);
            const cell = Math.round(time / fetchStep);
            // Точный уровень → грубая подложка. Дыр при зуме не бывает:
            // обзорная лента всегда в памяти.
            const tile = strip.lookup(fetchStep, cell, height) || strip.lookupCoarse(time);
            if (!tile) continue;
            // Кадр целиком, своей шириной — ни растяжения, ни обрезки.
            ctx.drawImage(tile.bitmap, tile.sx, 0, tile.sw, tile.sh,
                          x, lanes.ruler, cardWidth + 1, lanes.film);
            drawnCell = tile;
        }
        ctx.restore();
        reportCardGeometry(drawnCell, cardWidth, lanes.film);
        strip.request(fetchStep, Math.floor(viewport.viewStart / fetchStep),
                      Math.ceil(viewport.getViewEnd() / fetchStep));
    }

    function drawWave(ctx, width, lanes, colors) {
        if (!peaks.hasData || !(state.duration > 0)) return;
        const top = lanes.ruler + lanes.film;
        const middle = top + lanes.wave / 2;
        ctx.fillStyle = colors.bg;
        ctx.fillRect(0, top, width, lanes.wave);
        ctx.fillStyle = colors.muted;

        for (let x = 0; x < width; x += 1) {
            const level = peaks.sample(viewport.xToSeconds(x, width), state.duration);
            const half = Math.max(0.5, (level * lanes.wave) / 2);
            ctx.fillRect(x, middle - half, 1, half * 2);
        }

        // Приблизились — обзорная тысяча столбиков превращается в лесенку;
        // просим окно ровно по экрану.
        if (viewport.zoom > 2) {
            peaks.request(viewport.viewStart, viewport.getViewEnd(), width);
        }
    }

    function drawOverlay() {
        const { ctx, width, height } = syncCanvas(overlay);
        if (!(state.duration > 0)) return;
        const colors = getThemeColors();
        const { left, right } = bounds();

        const leftX = viewport.secondsToX(left, width);
        const rightX = viewport.secondsToX(right, width);

        // Затемнение вне выделения — намеренный литерал: оно лежит поверх кадров
        // пользователя и должно читаться на любой картинке.
        ctx.fillStyle = "rgba(0,0,0,0.45)";
        if (leftX > 0) ctx.fillRect(0, 0, leftX, height);
        if (rightX < width) ctx.fillRect(rightX, 0, width - rightX, height);

        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2;
        for (const x of [leftX, rightX]) {
            if (x < -2 || x > width + 2) continue;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
            ctx.fillStyle = colors.accentStrong || colors.accent;
            ctx.fillRect(x - 2, 6, 4, height - 12);
        }

        const playX = viewport.secondsToX(video.currentTime || 0, width);
        if (playX >= 0 && playX <= width) {
            ctx.strokeStyle = colors.text;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(playX, 0);
            ctx.lineTo(playX, height);
            ctx.stroke();
        }

        updateThumb();
    }

    function updateThumb() {
        const total = state.duration;
        if (!(total > 0) || viewport.zoom <= 1) {
            thumb.style.left = "0%";
            thumb.style.width = "100%";
            return;
        }
        const fraction = viewport.getViewSeconds() / total;
        thumb.style.width = `${Math.max(4, fraction * 100)}%`;
        thumb.style.left = `${(viewport.viewStart / total) * 100}%`;
    }

    // ── жесты ────────────────────────────────────────────────────────────── #
    const drag = createRangeDrag({
        getBounds: bounds,
        setBounds: (left, right, moved) => {
            setRange(left, right);
            // ⚠️ Плеер идёт ЗА ручкой, которую тянут. Без этого границу
            // выставляют вслепую: цифры меняются, а какой кадр окажется первым —
            // видно только после того, как отпустишь и нажмёшь воспроизведение.
            if (moved === "left") playback.seek(left, { scrub: true });
            else if (moved === "right") playback.seek(right, { scrub: true });
        },
        onSeek: (seconds) => playback.seek(seconds),
        getSecondsPerPixel: secondsPerPixel,
        getDuration: () => state.duration,
    });

    let panning = null;

    timeline.addEventListener("pointerdown", (event) => {
        if (!(state.duration > 0)) return;
        timeline.setPointerCapture?.(event.pointerId);
        const seconds = pointerSeconds(event);

        if (event.button === 1 || event.shiftKey) {
            panning = { x: event.clientX, start: viewport.viewStart };
            return;
        }
        const rect = timeline.getBoundingClientRect();
        if (event.clientY - rect.top <= LANE_RULER) {
            playback.seek(seconds);         // по шкале — скраб, не выделение
            panning = { scrub: true };
            return;
        }
        drag.begin(seconds, event.clientX);
        event.preventDefault();
    });

    timeline.addEventListener("pointermove", (event) => {
        if (panning?.scrub) {
            playback.seek(pointerSeconds(event));
            return;
        }
        if (panning) {
            const perPixel = secondsPerPixel();
            viewport.viewStart = panning.start - (event.clientX - panning.x) * perPixel;
            scheduleDraw(NEED_ALL);
            return;
        }
        if (drag.active) {
            if (drag.move(pointerSeconds(event), event.clientX)) scheduleDraw(NEED_OVERLAY);
            return;
        }
        updateCursor(pointerSeconds(event));
    });

    const endGesture = (event) => {
        timeline.releasePointerCapture?.(event.pointerId);
        panning = null;
        if (drag.active) drag.end(pointerSeconds(event));
        scheduleDraw(NEED_OVERLAY);
    };
    timeline.addEventListener("pointerup", endGesture);
    timeline.addEventListener("pointercancel", endGesture);
    timeline.addEventListener("dblclick", () => setRange(0, -1));

    function updateCursor(seconds) {
        const { left, right } = bounds();
        const handle = hitTestHandle(seconds, {
            left, right, secondsPerPixel: secondsPerPixel(),
        });
        timeline.style.cursor = handle ? HANDLE_CURSOR : "crosshair";
    }

    /**
     * Колесо над таймлайном: зум с якорем или панорама.
     *
     * ⚠️ Слушатель висит НА ДОКУМЕНТЕ В ФАЗЕ ПЕРЕХВАТА, а не на самом
     * таймлайне. В Nodes 2.0 колесо перехватывает и гасит Vue-обёртка ноды
     * раньше, чем событие доходит до нашего элемента (замерено: в классическом
     * режиме зум работал, в Vue — нет). Перехват на документе — единственное
     * место, куда событие приходит гарантированно.
     */
    const onWheel = (event) => {
        if (!(state.duration > 0)) return;
        if (!timeline.contains(event.target)) return;
        const zoomGesture = event.ctrlKey || event.metaKey;
        // Простое колесо панорамирует только когда есть куда: на единичном
        // масштабе событие честно достаётся холсту графа.
        if (!zoomGesture && viewport.zoom <= 1) return;
        event.preventDefault();
        event.stopPropagation();
        if (zoomGesture) {
            const factor = event.deltaY < 0 ? 1.25 : 1 / 1.25;
            viewport.setZoom(viewport.zoom * factor, pointerSeconds(event));
        } else {
            viewport.panBy(Math.sign(event.deltaY) * viewport.getViewSeconds() * 0.15);
        }
    };
    document.addEventListener("wheel", onWheel, { capture: true, passive: false });

    // Скроллбар
    let thumbDrag = null;
    thumb.addEventListener("pointerdown", (event) => {
        thumb.setPointerCapture?.(event.pointerId);
        thumbDrag = { x: event.clientX, start: viewport.viewStart };
        event.stopPropagation();
    });
    thumb.addEventListener("pointermove", (event) => {
        if (!thumbDrag) return;
        const width = scroll.clientWidth || 1;
        const perPixel = state.duration / width;
        viewport.viewStart = thumbDrag.start + (event.clientX - thumbDrag.x) * perPixel;
        scheduleDraw(NEED_ALL);
    });
    const stopThumb = () => { thumbDrag = null; };
    thumb.addEventListener("pointerup", stopThumb);
    thumb.addEventListener("pointercancel", stopThumb);
    scroll.addEventListener("pointerdown", (event) => {
        if (event.target === thumb || !(state.duration > 0)) return;
        const rect = scroll.getBoundingClientRect();
        const fraction = clamp((event.clientX - rect.left) / (rect.width || 1), 0, 1);
        viewport.viewStart = fraction * state.duration - viewport.getViewSeconds() / 2;
        scheduleDraw(NEED_ALL);
    });

    // ── наружный интерфейс ───────────────────────────────────────────────── #
    function setRange(left, right) {
        const total = state.duration;
        state.cropStart = Math.max(0, left);
        // Сентинел «до конца» сохраняется: правая ручка у самого края означает
        // именно «до конца файла», а не конкретную секунду.
        state.cropEnd = (total > 0 && right >= total - 0.01) ? -1 : right;
        onRangeChange?.(state.cropStart, state.cropEnd);
        scheduleDraw(NEED_OVERLAY);
    }

    return {
        element: root,
        stage,
        video,
        timeline,
        scroll,
        state,
        viewport,
        playback,
        strip,
        badge,
        empty,
        scheduleDraw,
        bounds,
        setRange,
        pointerSeconds,
        laneGeometry,
        NEED_BASE,
        NEED_OVERLAY,
        NEED_ALL,

        mount(bar, transport, status) {
            root.append(bar, stage, transport, timeline, scroll, status);
        },

        applyMetadata(meta) {
            state.duration = Number(meta?.duration) || 0;
            state.fps = Number(meta?.fps) || 0;
            state.width = Number(meta?.width) || 0;
            state.height = Number(meta?.height) || 0;
            state.hasAudio = Boolean(meta?.has_audio);
            peaks.setOverview(meta?.peaks);
            state.playable = meta?.browser_playable !== false;
            // Индекс в начале файла — можно смело просить метаданные; иначе
            // браузер ради перемотки утянет весь ролик.
            video.preload = meta?.faststart ? "metadata" : "none";
            viewport.clampViewStart();
            strip.clear();
            if (state.duration > 0) strip.ensureOverview(state.duration);
            scheduleDraw(NEED_ALL);
        },

        setSource(url) {
            if (!url) {
                video.removeAttribute("src");
                video.load();
                return;
            }
            video.src = url;
            video.load();
        },

        dispose() {
            document.removeEventListener("wheel", onWheel, { capture: true });
            playback.dispose();
            strip.dispose();
            peaks.dispose();
            cancelAnimationFrame(rafId);
            video.pause();
            video.removeAttribute("src");
            video.load();
        },
    };
}
