// TS Studio kit — mask painting canvas (ui-kit layer).
//
// A deliberately small, self-contained paint surface for inpaint modes:
// image underneath, tinted mask on top, brush/eraser with an HTML cursor,
// stroke-level undo/redo. Follows the pack's hard-won canvas rules
// (CLAUDE.md §12.5): full-bleed canvas, scale in state (not CSS), incremental
// tinted overlay, cursor as a DOM element so moves never trigger redraws.
// LamaCleanup's engine stays untouched — this is the kit's own surface.

import { TS_UI_CLASS, ensureThemeStyles, getThemeColors } from "../_theme.js";
import { attachZoomPan, clampScale } from "./_zoompan.js";

const STYLE_ID = "ts-studio-mask-styles";

export function ensureMaskStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-mask{position:absolute;inset:0}
.ts-mask__canvas{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none}
.ts-mask__canvas.has-image{cursor:none}
.ts-mask__cursor.is-preview{border-width:2px;opacity:1}
.ts-mask__cursor{position:absolute;pointer-events:none;border-radius:50%;display:none;
    /* Over user content: must read on any image, so not a theme token. */
    border:1.5px solid rgba(255,255,255,.9);box-shadow:0 0 0 1px rgba(0,0,0,.55)}
`;
    document.head.appendChild(style);
}

/**
 * @param {object} options
 * @param {() => void} [options.onStrokeEnd] Fired after a completed stroke.
 * @param {() => void} [options.onMaskChanged]
 * @returns Mask surface handle.
 */
export function createMaskCanvas(options = {}) {
    ensureMaskStyles();
    const root = document.createElement("div");
    root.className = `${TS_UI_CLASS} ts-mask`;
    const canvas = document.createElement("canvas");
    canvas.className = "ts-mask__canvas";
    const cursor = document.createElement("div");
    cursor.className = "ts-mask__cursor";
    root.append(canvas, cursor);
    const ctx = canvas.getContext("2d");

    const state = {
        image: null, imageW: 0, imageH: 0,
        scale: 1, offsetX: 0, offsetY: 0,
        brush: 48, eraser: false, painting: false,
        fitted: true,          // вид не трогали — подгоняем под область
        undo: [], redo: [],
    };
    let maskCanvas = null;   // full-resolution mask (white = repaint)
    let maskCtx = null;

    const PAD = 10;

    /** Масштаб, при котором картинка целиком видна в рабочей области. */
    function fitScale() {
        const rect = root.getBoundingClientRect();
        if (!(state.imageW > 0) || !(rect.width > 0)) return 1;
        const usableW = Math.max(1, rect.width - PAD * 2);
        const usableH = Math.max(1, rect.height - PAD * 2);
        return Math.min(usableW / state.imageW, usableH / state.imageH);
    }

    /** Вписать картинку целиком и поставить её по центру. */
    function fit() {
        const rect = root.getBoundingClientRect();
        state.scale = fitScale();
        state.offsetX = (rect.width - state.imageW * state.scale) / 2;
        state.offsetY = (rect.height - state.imageH * state.scale) / 2;
        state.fitted = true;
        redraw();
    }

    function resize() {
        const rect = root.getBoundingClientRect();
        canvas.width = Math.max(1, Math.round(rect.width));
        canvas.height = Math.max(1, Math.round(rect.height));
        // Если человек приблизил и увёл картинку, менять её при изменении
        // размера окна — значит терять место, на которое он смотрел. Подгоняем
        // только пока вид не трогали.
        if (state.imageW > 0 && rect.width > 0 && state.fitted) fit();
        else redraw();
    }
    const observer = new ResizeObserver(resize);
    observer.observe(root);

    function redraw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!state.image) return;
        const w = state.imageW * state.scale;
        const h = state.imageH * state.scale;
        ctx.drawImage(state.image, state.offsetX, state.offsetY, w, h);
        if (maskCanvas) {
            ctx.save();
            ctx.globalAlpha = 0.55;
            const colors = getThemeColors();
            // Tint through an offscreen pass: draw mask, then colorize.
            ctx.drawImage(tinted(colors.accent), state.offsetX, state.offsetY, w, h);
            ctx.restore();
        }
    }

    let tintCache = null;
    let tintDirty = true;
    // Цвет, которым собрана копия: сменилась тема — пересобрать.
    let tintColor = "";
    function tinted(color) {
        const sizeChanged = tintCache
            && (tintCache.width !== maskCanvas.width || tintCache.height !== maskCanvas.height);
        if (!tintDirty && tintCache && tintColor === color && !sizeChanged) return tintCache;
        tintCache = tintCache || document.createElement("canvas");
        tintCache.width = maskCanvas.width;
        tintCache.height = maskCanvas.height;
        const tctx = tintCache.getContext("2d");
        tctx.clearRect(0, 0, tintCache.width, tintCache.height);
        tctx.drawImage(maskCanvas, 0, 0);
        tctx.globalCompositeOperation = "source-in";
        tctx.fillStyle = color;
        tctx.fillRect(0, 0, tintCache.width, tintCache.height);
        tctx.globalCompositeOperation = "source-over";
        tintDirty = false;
        tintColor = color;
        return tintCache;
    }

    function toLocal(event) {
        const rect = canvas.getBoundingClientRect();
        const layoutW = canvas.offsetWidth || rect.width;
        const parentScale = layoutW > 0 ? rect.width / layoutW : 1;
        const inv = parentScale > 0.001 ? 1 / parentScale : 1;
        const x = (event.clientX - rect.left) * inv;
        const y = (event.clientY - rect.top) * inv;
        return {
            x: (x - state.offsetX) / state.scale,
            y: (y - state.offsetY) / state.scale,
            cssX: x, cssY: y,
        };
    }

    // ⚠️ Мазок кладётся СРАЗУ В ОБА холста — в маску и в её подкрашенную копию.
    //
    // Раньше каждая точка помечала тонировку грязной, а `redraw` пересобирал её
    // целиком: очистка, копия всей маски, заливка `source-in` — в полном
    // разрешении картинки, на КАЖДОЕ движение мыши. На 4K кисть заметно
    // отставала от курсора. Тот же приём уже применён в TS_LamaCleanup
    // (CLAUDE.md §12.5.6), здесь он просто не был перенесён.
    //
    // Полная пересборка осталась там, где она и нужна: отмена, очистка,
    // смена размера холста и смена цвета темы.
    function drawDot(x, y) {
        const radius = state.brush / 2 / state.scale;
        const mode = state.eraser ? "destination-out" : "source-over";

        maskCtx.globalCompositeOperation = mode;
        maskCtx.fillStyle = "#ffffff";
        maskCtx.beginPath();
        maskCtx.arc(x, y, radius, 0, Math.PI * 2);
        maskCtx.fill();

        const tctx = tintContextForPainting();
        if (tctx) {
            tctx.globalCompositeOperation = mode;
            tctx.fillStyle = tintColor || "#ffffff";
            tctx.beginPath();
            tctx.arc(x, y, radius, 0, Math.PI * 2);
            tctx.fill();
            tctx.globalCompositeOperation = "source-over";
        }
    }

    /** Контекст подкрашенной копии, если она уже собрана и совпадает по размеру. */
    function tintContextForPainting() {
        if (tintDirty || !tintCache) return null;
        if (tintCache.width !== maskCanvas.width || tintCache.height !== maskCanvas.height) {
            tintDirty = true;
            return null;
        }
        return tintCache.getContext("2d");
    }

    let last = null;
    let painterId = null;        // указатель, которым начали мазок
    let strokeButtons = 1;       // маска кнопки, которой ведут мазок
    let eraserBeforeStroke = null;  // инструмент до временного ластика
    function drawSegment(a, b) {
        const dist = Math.hypot(b.x - a.x, b.y - a.y);
        const steps = Math.max(1, Math.ceil(dist / Math.max(2, state.brush / 6 / state.scale)));
        for (let i = 0; i <= steps; i += 1) {
            drawDot(a.x + ((b.x - a.x) * i) / steps, a.y + ((b.y - a.y) * i) / steps);
        }
    }

    function snapshot() {
        const copy = document.createElement("canvas");
        copy.width = maskCanvas.width;
        copy.height = maskCanvas.height;
        copy.getContext("2d").drawImage(maskCanvas, 0, 0);
        return copy;
    }

    function restore(copy) {
        maskCtx.globalCompositeOperation = "source-over";
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        if (copy) maskCtx.drawImage(copy, 0, 0);
        tintDirty = true;
        redraw();
        options.onMaskChanged?.();
    }

    const detachZoomPan = attachZoomPan(canvas, {
        zoomAt(clientX, clientY, factor) {
            if (!state.image) return;
            const rect = canvas.getBoundingClientRect();
            const x = clientX - rect.left;
            const y = clientY - rect.top;
            const next = clampScale(state.scale * factor, fitScale());
            if (next === state.scale) return;
            // Точка под курсором остаётся на месте: сдвиг считается от неё, а
            // не от угла. Иначе приближение уводит взгляд с того, что смотрят.
            state.offsetX = x - ((x - state.offsetX) / state.scale) * next;
            state.offsetY = y - ((y - state.offsetY) / state.scale) * next;
            state.scale = next;
            state.fitted = false;
            redraw();
        },
        panBy(dx, dy) {
            if (!state.image) return;
            state.offsetX += dx;
            state.offsetY += dy;
            state.fitted = false;
            redraw();
        },
        reset: () => { if (state.image) fit(); },
    });

    // Мазок начинается только ПЕРВИЧНЫМ указателем: `isPrimary` отсекает
    // второй палец на тачскрине, из-за которого мазок начинался дважды и вёл
    // линию между пальцами.
    //
    // Левая кнопка рисует текущим инструментом, правая всегда стирает — на
    // время своего мазка, не трогая выбранный в панели. Так устроен любой
    // редактор, и это избавляет от беготни к переключателю ради одной правки.
    canvas.addEventListener("pointerdown", (event) => {
        if (!state.image || event.isPrimary === false) return;
        if (event.button !== 0 && event.button !== 2) return;
        event.preventDefault();
        try { canvas.setPointerCapture(event.pointerId); } catch { /* уже отпущен */ }
        painterId = event.pointerId;
        strokeButtons = event.button === 2 ? 2 : 1;
        eraserBeforeStroke = state.eraser;
        if (event.button === 2) state.eraser = true;
        state.undo.push(snapshot());
        state.redo.length = 0;
        state.painting = true;
        const p = toLocal(event);
        last = p;
        drawDot(p.x, p.y);
        redraw();
    });
    // Правая кнопка стирает — своё меню тут только мешает.
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    // Where the ring sits. Kept across events so a size change can redraw it
    // without waiting for the pointer to move — the size of a brush is only
    // meaningful as a circle you can see.
    let cursorAt = null;
    let previewTimer = null;
    let hovering = false;

    function placeCursor(cssX, cssY) {
        cursorAt = { cssX, cssY };
        cursor.style.width = `${state.brush}px`;
        cursor.style.height = `${state.brush}px`;
        cursor.style.left = `${cssX - state.brush / 2}px`;
        cursor.style.top = `${cssY - state.brush / 2}px`;
        cursor.style.display = state.image ? "block" : "none";
    }

    /**
     * Show the ring at its current size after a change made elsewhere — while
     * dragging the size slider, the pointer is on the slider, not the picture.
     * It fades on its own so it does not sit over the work afterwards.
     */
    function previewBrush() {
        if (!state.image) return;
        // Кольцо показываем В ЦЕНТРЕ КАРТИНКИ, а не там, где курсор был в
        // прошлый раз. Размер кисти меняют слайдером — курсор в этот момент на
        // слайдере, и оставшаяся от него точка обычно где-то с краю: круг
        // всплывал в стороне и ничего не показывал. Под курсором кольцо и так
        // живёт постоянно, поэтому там, где мышь над холстом, ничего не трогаем.
        let at = cursorAt;
        if (!hovering || !at) {
            const rect = canvas.getBoundingClientRect();
            at = state.imageW > 0
                ? {
                    cssX: state.offsetX + (state.imageW * state.scale) / 2,
                    cssY: state.offsetY + (state.imageH * state.scale) / 2,
                }
                : { cssX: rect.width / 2, cssY: rect.height / 2 };
        }
        placeCursor(at.cssX, at.cssY);
        cursor.classList.add("is-preview");
        clearTimeout(previewTimer);
        previewTimer = setTimeout(() => {
            cursor.classList.remove("is-preview");
            if (!hovering) cursor.style.display = "none";
        }, 900);
    }

    canvas.addEventListener("pointerenter", () => { hovering = true; });
    canvas.addEventListener("pointermove", (event) => {
        const p = toLocal(event);
        hovering = true;
        placeCursor(p.cssX, p.cssY);
        if (!state.painting) return;
        // Кнопка — единственный источник правды.
        //
        // Флага `painting` недостаточно: `pointerup` теряется буднично —
        // отпустили за пределами окна, сорвался перехват указателя, ушли по
        // Alt+Tab посреди мазка. Флаг оставался поднятым, и кисть начинала
        // рисовать по простому движению мыши. Поэтому на каждом движении
        // спрашиваем у события, нажата ли левая кнопка ПРЯМО СЕЙЧАС.
        if ((event.buttons & strokeButtons) === 0 || event.pointerId !== painterId) {
            endStroke();
            return;
        }
        drawSegment(last, p);
        last = p;
        redraw();
    });
    const endStroke = () => {
        if (!state.painting) return;
        state.painting = false;
        // Правая кнопка стирала только на время своего мазка.
        if (eraserBeforeStroke !== null) {
            state.eraser = eraserBeforeStroke;
            eraserBeforeStroke = null;
        }
        strokeButtons = 1;
        if (painterId !== null) {
            try { canvas.releasePointerCapture(painterId); } catch { /* уже отпущен */ }
            painterId = null;
        }
        last = null;
        options.onMaskChanged?.();
        options.onStrokeEnd?.();
    };
    canvas.addEventListener("pointerup", endStroke);
    canvas.addEventListener("pointercancel", endStroke);
    // Перехват может сорваться сам — например, когда элемент переверстали.
    canvas.addEventListener("lostpointercapture", endStroke);
    // Последняя линия обороны: отпустили кнопку где угодно, ушли из окна,
    // переключили вкладку — мазок обязан закончиться там же, где закончился
    // жест, а не висеть до следующего движения над холстом.
    const stopAnywhere = () => endStroke();
    window.addEventListener("pointerup", stopAnywhere, true);
    window.addEventListener("pointercancel", stopAnywhere, true);
    window.addEventListener("blur", stopAnywhere);
    document.addEventListener("visibilitychange", stopAnywhere);
    canvas.addEventListener("pointerleave", () => {
        hovering = false;
        // A ring shown for a size change outlives the pointer leaving: that is
        // the one moment when it is the thing being looked at.
        if (!cursor.classList.contains("is-preview")) cursor.style.display = "none";
    });

    /**
     * Показать картинку на холсте.
     *
     * @param {string} url адрес
     * @param {object} [options]
     * @param {boolean} [options.keepView] Не возвращать вписанный вид. Нужно
     *   при листании версий одного кадра: человек приблизил кусок ровно
     *   затем, чтобы сравнить его до и после, и сброс масштаба это отменяет.
     */
    async function loadImage(url, { keepView = false } = {}) {
        const image = new Image();
        image.crossOrigin = "anonymous";
        await new Promise((resolve, reject) => {
            image.onload = resolve;
            image.onerror = () => reject(new Error("image load failed"));
            image.src = url;
        });
        state.image = image;
        state.imageW = image.naturalWidth;
        state.imageH = image.naturalHeight;
        maskCanvas = document.createElement("canvas");
        maskCanvas.width = state.imageW;
        maskCanvas.height = state.imageH;
        maskCtx = maskCanvas.getContext("2d");
        tintCache = null;
        tintDirty = true;
        state.undo.length = 0;
        state.redo.length = 0;
        canvas.classList.add("has-image");
        // Новая картинка приходит вписанной целиком, каким бы ни был вид до
        // неё. Версии одного кадра — не «новая картинка»: там вид сохраняется,
        // а `resize()` сам решит, вписывать ли (по флагу `fitted`).
        if (!keepView) state.fitted = true;
        resize();
    }

    function hasMask() {
        if (!maskCanvas) return false;
        const data = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
        for (let i = 3; i < data.length; i += 4) {
            if (data[i] > 8) return true;
        }
        return false;
    }

    function maskDataUrl() {
        // White strokes on a FULLY TRANSPARENT background. Both consumers
        // agree on this shape: LaMa's decoder reads the alpha channel of an
        // RGBA PNG (an opaque black background would select the whole frame
        // — measured), and the studio's mask marker reads luminance, where
        // transparent decodes to black.
        return maskCanvas.toDataURL("image/png");
    }

    function maskBBox() {
        if (!maskCanvas) return null;
        const step = 4;
        const data = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
        let minX = Infinity, minY = Infinity, maxX = -1, maxY = -1;
        for (let y = 0; y < maskCanvas.height; y += step) {
            for (let x = 0; x < maskCanvas.width; x += step) {
                if (data[(y * maskCanvas.width + x) * 4 + 3] > 8) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }
        if (maxX < 0) return null;
        return { x: minX, y: minY, w: maxX - minX + step, h: maxY - minY + step };
    }

    function imageRectToCss(rect) {
        return {
            left: state.offsetX + rect.x * state.scale,
            top: state.offsetY + rect.y * state.scale,
            width: rect.w * state.scale,
            height: rect.h * state.scale,
        };
    }

    return {
        element: root,
        loadImage,
        maskBBox,
        imageRectToCss,
        imageSize: () => ({ w: state.imageW, h: state.imageH }),
        setBrush: (px) => {
            state.brush = Math.max(4, px);
            previewBrush();
        },
        getBrush: () => state.brush,
        setEraser: (on) => { state.eraser = Boolean(on); },
        clearMask: () => { state.undo.push(snapshot()); state.redo.length = 0; restore(null); },
        /**
         * Картинка холста как data-URL.
         *
         * Именно картинка, без маски: её отправляют в соседнюю вкладку, а маска
         * принадлежит этой задаче и туда не едет. Данные берутся у самого
         * изображения, поэтому годится и то, что пришло после перерисовки.
         */
        imageDataUrl: () => {
            if (!state.image) return "";
            const canvas = document.createElement("canvas");
            canvas.width = state.imageW;
            canvas.height = state.imageH;
            canvas.getContext("2d").drawImage(state.image, 0, 0);
            return canvas.toDataURL("image/png");
        },
        /** Забыть картинку и маску — холст становится пустым. */
        clearImage: () => {
            state.image = null;
            state.imageW = 0;
            state.imageH = 0;
            state.undo.length = 0;
            state.redo.length = 0;
            restore(null);
        },
        undo: () => {
            const prev = state.undo.pop();
            if (prev !== undefined) { state.redo.push(snapshot()); restore(prev); }
        },
        redo: () => {
            const next = state.redo.pop();
            if (next !== undefined) { state.undo.push(snapshot()); restore(next); }
        },
        canUndo: () => state.undo.length > 0,
        canRedo: () => state.redo.length > 0,
        hasMask,
        maskDataUrl,
        /**
         * Положить на холст готовую маску — тем же снимком, что отдаёт
         * `maskDataUrl`. Нужно, чтобы закрытая и снова открытая студия
         * возвращала нарисованное, а не чистый холст: картинка без своей маски
         * — это половина работы.
         *
         * Размер приводится к холсту маски, потому что источник мог прийти в
         * другом разрешении (например, после перерисовки). Шаг ложится в
         * историю отмены как обычный мазок.
         */
        setMaskFromUrl: (url) => new Promise((resolve) => {
            if (!url || !maskCanvas) { resolve(false); return; }
            const img = new Image();
            img.onload = () => {
                state.undo.push(snapshot());
                state.redo.length = 0;
                maskCtx.globalCompositeOperation = "source-over";
                maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                maskCtx.drawImage(img, 0, 0, maskCanvas.width, maskCanvas.height);
                tintDirty = true;
                redraw();
                options.onMaskChanged?.();
                resolve(true);
            };
            img.onerror = () => resolve(false);
            img.src = url;
        }),
        hasImage: () => Boolean(state.image),
        /** Вписать картинку в рабочую область целиком. */
        fit,
        /** Текущий масштаб — для подписи в интерфейсе. */
        zoom: () => state.scale / (fitScale() || 1),
        teardown: () => {
            observer.disconnect();
            detachZoomPan();
            window.removeEventListener("pointerup", stopAnywhere, true);
            window.removeEventListener("pointercancel", stopAnywhere, true);
            window.removeEventListener("blur", stopAnywhere);
            document.removeEventListener("visibilitychange", stopAnywhere);
        },
    };
}
