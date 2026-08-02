// TS Studio kit — mask painting canvas (ui-kit layer).
//
// A deliberately small, self-contained paint surface for inpaint modes:
// image underneath, tinted mask on top, brush/eraser with an HTML cursor,
// stroke-level undo/redo. Follows the pack's hard-won canvas rules
// (CLAUDE.md §12.5): full-bleed canvas, scale in state (not CSS), incremental
// tinted overlay, cursor as a DOM element so moves never trigger redraws.
// LamaCleanup's engine stays untouched — this is the kit's own surface.

import { TS_UI_CLASS, ensureThemeStyles, getThemeColors } from "../_theme.js";

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
        undo: [], redo: [],
    };
    let maskCanvas = null;   // full-resolution mask (white = repaint)
    let maskCtx = null;

    const PAD = 10;

    function resize() {
        const rect = root.getBoundingClientRect();
        canvas.width = Math.max(1, Math.round(rect.width));
        canvas.height = Math.max(1, Math.round(rect.height));
        if (state.imageW > 0 && rect.width > 0) {
            const usableW = Math.max(1, rect.width - PAD * 2);
            const usableH = Math.max(1, rect.height - PAD * 2);
            state.scale = Math.min(usableW / state.imageW, usableH / state.imageH);
            state.offsetX = PAD + (usableW - state.imageW * state.scale) / 2;
            state.offsetY = PAD + (usableH - state.imageH * state.scale) / 2;
        }
        redraw();
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
    function tinted(color) {
        if (!tintDirty && tintCache) return tintCache;
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

    function drawDot(x, y) {
        maskCtx.globalCompositeOperation = state.eraser ? "destination-out" : "source-over";
        maskCtx.fillStyle = "#ffffff";
        maskCtx.beginPath();
        maskCtx.arc(x, y, state.brush / 2 / state.scale, 0, Math.PI * 2);
        maskCtx.fill();
        tintDirty = true;
    }

    let last = null;
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

    canvas.addEventListener("pointerdown", (event) => {
        if (!state.image || event.button !== 0) return;
        canvas.setPointerCapture(event.pointerId);
        state.undo.push(snapshot());
        state.redo.length = 0;
        state.painting = true;
        const p = toLocal(event);
        last = p;
        drawDot(p.x, p.y);
        redraw();
    });
    canvas.addEventListener("pointermove", (event) => {
        const p = toLocal(event);
        cursor.style.display = state.image ? "block" : "none";
        cursor.style.width = `${state.brush}px`;
        cursor.style.height = `${state.brush}px`;
        cursor.style.left = `${p.cssX - state.brush / 2}px`;
        cursor.style.top = `${p.cssY - state.brush / 2}px`;
        if (!state.painting) return;
        drawSegment(last, p);
        last = p;
        redraw();
    });
    const endStroke = () => {
        if (!state.painting) return;
        state.painting = false;
        options.onMaskChanged?.();
        options.onStrokeEnd?.();
    };
    canvas.addEventListener("pointerup", endStroke);
    canvas.addEventListener("pointercancel", endStroke);
    canvas.addEventListener("pointerleave", () => { cursor.style.display = "none"; });

    async function loadImage(url) {
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
        setBrush: (px) => { state.brush = Math.max(4, px); },
        getBrush: () => state.brush,
        setEraser: (on) => { state.eraser = Boolean(on); },
        clearMask: () => { state.undo.push(snapshot()); state.redo.length = 0; restore(null); },
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
        hasImage: () => Boolean(state.image),
        teardown: () => observer.disconnect(),
    };
}
