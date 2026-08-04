// TS Image Studio — the inpaint mode (app layer).
//
// One paint surface, two engines behind a segmented switch:
//   Cleanup — LaMa through the LamaCleanup node's own live routes: paint,
//             release, the region is gone in about a second. No prompt.
//   Repaint — the family's diffusion inpaint backend (TSSmartInpaint for
//             Klein, the pack's Universal Inpaint Sampler elsewhere): the
//             mask + prompt go through the standard studio run path.
// The mask survives switching engines — paint once, try both.

import { TS_UI_CLASS, ensureThemeStyles } from "../../_theme.js";
import { cropBox } from "../../_studio/_crop_geometry.js";
import { createMaskCanvas } from "../../_studio/_mask.js";
import { makeDropZone, uploadImage } from "../../_studio/_dnd.js";

const STYLE_ID = "ts-istudio-inpaint-styles";

function ensureInpaintStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-inp{position:absolute;inset:0}
/* Centred on the canvas, but never under the fullscreen close button: the
   usable strip stops short of that corner, and the bar centres inside it. */
.ts-inp__bar{position:absolute;top:8px;left:0;right:var(--ts-fs-safe-right);z-index:5;
    width:max-content;max-width:calc(100% - var(--ts-fs-safe-right) - 8px);
    margin:0 auto;overflow-x:auto;
    display:flex;align-items:center;gap:8px;padding:4px 8px;background:var(--ts-elevated);
    border:1px solid var(--ts-border);border-radius:var(--ts-radius)}
.ts-inp__seg{display:flex;border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);
    overflow:hidden}
.ts-inp__segbtn{border:none;background:none;color:var(--ts-muted);cursor:pointer;
    padding:3px 10px;font-size:var(--ts-fs-sm)}
.ts-inp__segbtn.is-active{background:var(--ts-accent-soft);color:var(--ts-accent)}
.ts-inp__tool{width:24px;height:24px;display:flex;align-items:center;justify-content:center;
    border:none;background:none;color:var(--ts-muted);cursor:pointer;border-radius:var(--ts-radius-sm);
    padding:0;font-size:13px}
.ts-inp__tool:hover{color:var(--ts-text);background:var(--ts-border-soft)}
.ts-inp__tool.is-active{color:var(--ts-accent);background:var(--ts-accent-soft)}
.ts-inp__tool:disabled{opacity:.35;cursor:default}
.ts-inp__sep{width:1px;height:16px;background:var(--ts-border)}
.ts-inp__bar input[type=range]{width:70px}
.ts-inp__empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
    color:var(--ts-muted);font-size:var(--ts-fs-lg);text-align:center;padding:24px}
.ts-inp__empty.is-drag-over{color:var(--ts-accent)}
.ts-inp__status{position:absolute;left:10px;bottom:10px;z-index:5;padding:3px 8px;
    font-size:var(--ts-fs-sm);color:var(--ts-muted);background:var(--ts-elevated);
    border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm)}
.ts-inp__pip{position:absolute;right:10px;bottom:10px;z-index:5;width:172px;max-height:172px;
    object-fit:contain;border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);
    background:var(--ts-sunken);display:none}
.ts-inp__pip.is-active{display:block}
/* Превью НА МЕСТЕ правки — не карточка в углу, а сама правка на своём месте.
   Рамка, скругление и непрозрачный фон здесь рисуют прямоугольник поверх
   картинки, даже когда содержимое обрезано по форме маски: именно это и
   выглядело «квадратом». Потолок высоты сплющивал вытянутые правки. */
.ts-inp__pip.is-inplace{border:0;border-radius:0;background:transparent;
    max-height:none;max-width:none;object-fit:fill}
`;
    document.head.appendChild(style);
}

/**
 * @param {object} ctx {api, t, sessionId, getSelectedResultUrl, onResult}
 * @returns inpaint mode handle for the app.
 */
export function createInpaintMode(ctx) {
    ensureInpaintStyles();
    const root = document.createElement("div");
    root.className = `${TS_UI_CLASS} ts-inp`;

    const mask = createMaskCanvas({
        onStrokeEnd: () => { if (state.engine === "cleanup") runCleanup(); },
    });
    mask.element.style.display = "none";

    const empty = document.createElement("div");
    empty.className = "ts-inp__empty";
    empty.textContent = ctx.t.inp.empty;

    // ── toolbar ─────────────────────────────────────────────────────────── //
    const bar = document.createElement("div");
    bar.className = "ts-inp__bar";
    const seg = document.createElement("div");
    seg.className = "ts-inp__seg";
    const segCleanup = segButton(ctx.t.inp.cleanup, ctx.t.inp.cleanupTip);
    const segRepaint = segButton(ctx.t.inp.repaint, ctx.t.inp.repaintTip);
    seg.append(segCleanup, segRepaint);

    const brush = document.createElement("input");
    brush.type = "range";
    brush.className = "ts-ui-slider";
    brush.min = "6";
    brush.max = "200";
    brush.value = "48";
    brush.title = ctx.t.inp.brush;
    brush.addEventListener("input", () => mask.setBrush(Number(brush.value)));

    // Кисть и ластик — парный переключатель, а не безымянный значок.
    //
    // Ластик тут был и раньше — одинокой кнопкой «◐» с подсказкой, и его
    // просто не находили. Инструмент, который нельзя увидеть, всё равно что
    // отсутствует, поэтому теперь это сегмент с подписями, как везде: видно,
    // что режима два, и видно, какой включён. Alt зажимает ластик на время —
    // привычка из любого редактора; клавиша E переключает.
    const paintSeg = document.createElement("div");
    paintSeg.className = "ts-inp__seg";
    const brushBtn = document.createElement("button");
    brushBtn.type = "button";
    brushBtn.className = "ts-inp__segbtn is-active";
    brushBtn.textContent = ctx.t.inp.brushMode;
    brushBtn.title = ctx.t.inp.brushModeTip;
    const eraser = document.createElement("button");
    eraser.type = "button";
    eraser.className = "ts-inp__segbtn";
    eraser.textContent = ctx.t.inp.eraserMode;
    eraser.title = ctx.t.inp.eraser;
    paintSeg.append(brushBtn, eraser);

    /** @param {boolean} on ластик включён */
    function setEraser(on) {
        eraser.classList.toggle("is-active", on);
        brushBtn.classList.toggle("is-active", !on);
        mask.setEraser(on);
    }
    brushBtn.addEventListener("click", () => setEraser(false));
    eraser.addEventListener("click", () => setEraser(true));

    // Alt — временный ластик: отпустил, и снова кисть. Слушатели на документе,
    // потому что клавишу могут нажать, когда фокус на ползунке или промпте;
    // снимаются в teardown вместе со всем остальным.
    let eraserBeforeAlt = null;
    const onAltDown = (event) => {
        if (event.key !== "Alt" || eraserBeforeAlt !== null) return;
        eraserBeforeAlt = eraser.classList.contains("is-active");
        if (!eraserBeforeAlt) setEraser(true);
    };
    const onAltUp = (event) => {
        if (event.key !== "Alt" || eraserBeforeAlt === null) return;
        setEraser(eraserBeforeAlt);
        eraserBeforeAlt = null;
    };
    document.addEventListener("keydown", onAltDown);
    document.addEventListener("keyup", onAltUp);
    const clear = tool("✕", ctx.t.inp.clear);
    clear.addEventListener("click", () => mask.clearMask());
    const undoBtn = tool("↶", ctx.t.inp.undo);
    const redoBtn = tool("↷", ctx.t.inp.redo);
    undoBtn.addEventListener("click", () => history.undo());
    redoBtn.addEventListener("click", () => history.redo());

    const sep1 = separator();
    const sep2 = separator();
    bar.append(seg, sep1, brush, paintSeg, clear, sep2, undoBtn, redoBtn);

    const status = document.createElement("div");
    status.className = "ts-inp__status";
    status.style.display = "none";

    const pip = document.createElement("canvas");
    pip.className = "ts-inp__pip";
    root.append(mask.element, empty, bar, status, pip);

    let previewBox = null;   // frozen at run start: where the mask was painted

    function capturePreviewBox() {
        // Замораживаем на момент запуска: маску можно продолжать править, а
        // превью обязано остаться там, откуда его взяли.
        const bbox = mask.maskBBox?.();
        if (!bbox) { previewBox = null; return; }
        const image = mask.imageSize();
        const box = cropBox({
            imageW: image.w, imageH: image.h,
            mask: { x: bbox.x, y: bbox.y, w: bbox.w, h: bbox.h },
            contextPct: ctx.getContextPct?.() ?? 25,
            denoise: ctx.getDenoise?.() ?? 1,
        });
        if (!box) { previewBox = null; return; }
        const crop = { x: box.x0, y: box.y0, w: box.x1 - box.x0, h: box.y1 - box.y0 };
        previewBox = {
            crop,
            css: mask.imageRectToCss(crop),
            alpha: mask.maskDataUrl(),
            feather: Math.max(2, Math.round(Math.min(bbox.w, bbox.h) * 0.05)),
        };
    }

    /**
     * Показать то, что модель рисует прямо сейчас, ровно по нарисованной маске.
     *
     * Приходит латентный предпросмотр ВЫРЕЗА — не всего кадра и не рамки маски.
     * Поэтому он кладётся в рамку выреза (её считает та же геометрия, что и
     * нода), а затем обрезается по форме маски с мягким краем: человек видит
     * правку на её месте и в её масштабе, а не квадрат поверх картинки.
     *
     * Предпросмотр приходит маленьким по своей природе (это быстрый декодер
     * латента), поэтому включено сглаживание — иначе он лезет квадратами.
     */
    async function showPreview(blob) {
        if (!previewBox) capturePreviewBox();
        const bitmap = await createImageBitmap(blob);
        if (!previewBox) {
            // Маски нет — показываем как есть, в углу.
            pip.width = bitmap.width; pip.height = bitmap.height;
            const plain = pip.getContext("2d");
            plain.imageSmoothingEnabled = true;
            plain.imageSmoothingQuality = "high";
            plain.clearRect(0, 0, pip.width, pip.height);
            plain.drawImage(bitmap, 0, 0);
            pip.style.cssText = "";
            pip.classList.remove("is-inplace");
            pip.classList.add("is-active");
            bitmap.close?.();
            return;
        }
        const { crop, css, alpha, feather } = previewBox;
        const w = Math.max(1, Math.round(css.width));
        const h = Math.max(1, Math.round(css.height));
        pip.width = w;
        pip.height = h;
        const c = pip.getContext("2d");
        c.imageSmoothingEnabled = true;
        c.imageSmoothingQuality = "high";
        c.clearRect(0, 0, w, h);
        c.drawImage(bitmap, 0, 0, w, h);
        bitmap.close?.();

        if (alpha) {
            // Маска приходит белыми штрихами на прозрачном — это готовая альфа.
            // Оставляем от превью только её форму, слегка размывая край.
            const shape = await loadImage(alpha);
            c.globalCompositeOperation = "destination-in";
            const blur = feather * (w / Math.max(1, crop.w));
            c.filter = blur > 0.5 ? `blur(${blur.toFixed(1)}px)` : "none";
            c.drawImage(shape, crop.x, crop.y, crop.w, crop.h, 0, 0, w, h);
            c.filter = "none";
            c.globalCompositeOperation = "source-over";
        }
        pip.style.cssText = `left:${css.left}px;top:${css.top}px;` +
            `width:${css.width}px;height:${css.height}px;right:auto;bottom:auto;`;
        pip.classList.add("is-inplace", "is-active");
    }

    function loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = src;
        });
    }

    function hidePreview() {
        pip.classList.remove("is-active", "is-inplace");
        pip.style.cssText = "";
        previewBox = null;
    }

    const state = {
        engine: "cleanup",
        sourceAnnotated: "",   // upload name of the CURRENT canvas image
        cleanupWorking: "",    // LaMa working_path chain
        versions: [],          // unified command stack: {kind, url, annotated, working}
        cursor: -1,
    };

    // ── unified history (plan §6): strokes live inside mask; pixel states here ─ //
    const history = {
        push(entry) {
            state.versions.splice(state.cursor + 1);
            state.versions.push(entry);
            state.cursor = state.versions.length - 1;
            syncButtons();
        },
        async undo() {
            if (mask.canUndo()) { mask.undo(); return; }
            if (state.cursor > 0) {
                state.cursor -= 1;
                await applyVersion(state.versions[state.cursor]);
            }
            syncButtons();
        },
        async redo() {
            if (mask.canRedo()) { mask.redo(); return; }
            if (state.cursor < state.versions.length - 1) {
                state.cursor += 1;
                await applyVersion(state.versions[state.cursor]);
            }
            syncButtons();
        },
    };

    function syncButtons() {
        undoBtn.disabled = !mask.canUndo() && state.cursor <= 0;
        redoBtn.disabled = !mask.canRedo() && state.cursor >= state.versions.length - 1;
    }

    async function applyVersion(version) {
        await mask.loadImage(version.url);
        state.sourceAnnotated = version.annotated || "";
        state.cleanupWorking = version.working || "";
    }

    function segButton(label, title) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-inp__segbtn";
        button.textContent = label;
        button.title = title;
        button.addEventListener("click", () => setEngine(button === segCleanup ? "cleanup" : "repaint"));
        return button;
    }

    function tool(glyph, title) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-inp__tool";
        button.textContent = glyph;
        button.title = title;
        button.setAttribute("aria-label", title);
        return button;
    }

    function separator() {
        const el = document.createElement("span");
        el.className = "ts-inp__sep";
        return el;
    }

    function setEngine(engine) {
        state.engine = engine;
        segCleanup.classList.toggle("is-active", engine === "cleanup");
        segRepaint.classList.toggle("is-active", engine === "repaint");
        ctx.onEngineChange?.(engine);
    }

    function setStatus(text) {
        status.textContent = text || "";
        status.style.display = text ? "" : "none";
    }

    // ── image intake: gallery selection, drop, paste ────────────────────── //
    async function setImageFromBlob(blob, name) {
        const annotated = await uploadImage(ctx.api, blob, name || "inpaint_src.png");
        const url = URL.createObjectURL(blob);
        await mask.loadImage(url);
        state.sourceAnnotated = annotated;
        state.cleanupWorking = "";
        state.versions = [{ kind: "source", url, annotated, working: "" }];
        state.cursor = 0;
        empty.style.display = "none";
        mask.element.style.display = "";
        syncButtons();
        ctx.onSourceChange?.();
    }

    async function setImageFromUrl(url, name) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        await setImageFromBlob(await response.blob(), name);
    }

    // Three zones for one intention. The empty state and the mask canvas are
    // the obvious targets, but a drop that lands on the toolbar or the margin
    // around them used to fall through to the page and be lost — so the mode's
    // whole surface accepts as well.
    const acceptDrop = { max: 1,
        onDrop: async ([item]) => setImageFromBlob(await item.getBlob(), item.name) };
    const dropTeardown = makeDropZone(empty, acceptDrop);
    const dropTeardown2 = makeDropZone(mask.element, acceptDrop);
    const dropTeardown3 = makeDropZone(root, acceptDrop);

    // ── Cleanup engine: LaMa live routes ────────────────────────────────── //
    let cleaning = false;
    async function runCleanup() {
        if (cleaning || !mask.hasImage() || !mask.hasMask()) return;
        cleaning = true;
        setStatus(ctx.t.inp.cleaning);
        const started = performance.now();
        try {
            const response = await ctx.api.fetchApi("/ts_lama_cleanup/inpaint", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: `studio_${ctx.sessionId}`,
                    source_path: state.cleanupWorking ? "" : state.sourceAnnotated,
                    working_path: state.cleanupWorking,
                    mask: mask.maskDataUrl(),
                    max_resolution: 1024,
                    mask_padding: 64,
                    feather: 4,
                }),
            });
            const payload = await response.json();
            if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
            state.cleanupWorking = payload.working_path;
            const url = `/ts_lama_cleanup/view?filepath=${encodeURIComponent(payload.working_path)}`
                + `&v=${Date.now()}`;
            await mask.loadImage(url);
            history.push({ kind: "cleanup", url,
                           annotated: state.sourceAnnotated, working: payload.working_path });
            const seconds = ((performance.now() - started) / 1000).toFixed(1);
            setStatus(ctx.t.inp.cleaned(seconds));
        } catch (err) {
            setStatus(ctx.t.inp.paintFailed(err.message));
        } finally {
            cleaning = false;
        }
    }

    // ── Repaint: values for the standard run path ───────────────────────── //
    async function collectRunValues() {
        if (!mask.hasImage()) throw new Error(ctx.t.inp.needImage);
        if (!mask.hasMask()) {
            // In Cleanup a stroke is spent the moment it ends — LaMa runs and
            // the cleaned image comes back with an empty mask. Telling someone
            // who just painted to "paint a mask" is technically true and
            // completely unhelpful; the real answer is which engine is armed.
            throw new Error(state.engine === "cleanup"
                ? (ctx.t.inp.needRepaint || ctx.t.inp.needMask)
                : ctx.t.inp.needMask);
        }
        capturePreviewBox();
        const maskBlob = await (await fetch(mask.maskDataUrl())).blob();
        const maskAnnotated = await uploadImage(ctx.api, maskBlob, "inpaint_mask.png");
        // The CURRENT canvas (after any cleanups) is the repaint source.
        let source = state.sourceAnnotated;
        if (state.cleanupWorking) {
            const current = await fetch(`/ts_lama_cleanup/view?filepath=${encodeURIComponent(state.cleanupWorking)}`);
            source = await uploadImage(ctx.api, await current.blob(), "inpaint_current.png");
        }
        return { source_image: source, mask: maskAnnotated };
    }

    async function acceptRepaintResult(url, name) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const blob = await response.blob();
        const annotated = await uploadImage(ctx.api, blob, name || "repaint.png");
        const objectUrl = URL.createObjectURL(blob);
        await mask.loadImage(objectUrl);
        state.sourceAnnotated = annotated;
        state.cleanupWorking = "";
        history.push({ kind: "repaint", url: objectUrl, annotated, working: "" });
        setStatus(ctx.t.inp.repainted);
        ctx.onSourceChange?.();
    }

    setEngine("cleanup");
    syncButtons();

    return {
        element: root,
        engine: () => state.engine,
        setImageFromUrl,
        setImageFromBlob,
        collectRunValues,
        acceptRepaintResult,
        hasImage: () => mask.hasImage(),
        hasMask: () => mask.hasMask(),
        maskDataUrl: () => (mask.hasMask() ? mask.maskDataUrl() : ""),
        setMaskFromUrl: (url) => mask.setMaskFromUrl(url),
        /**
         * Что сейчас лежит на холсте — в терминах снимка сессии.
         *
         * Отличается от `collectRunValues` тем, что ничего никуда не грузит:
         * снимок рабочего места пишется на каждое движение, и загрузка файла
         * на сервер при каждом мазке была бы неуместной. Отдаётся то, что уже
         * лежит на сервере — исходник и, если он есть, рабочий файл очистки.
         */
        currentSources: () => (state.sourceAnnotated
            ? { source_image: state.sourceAnnotated } : {}),
        undo: () => history.undo(),
        redo: () => history.redo(),
        brushDelta: (delta) => {
            const next = Math.max(6, Math.min(200, Number(brush.value) + delta));
            brush.value = String(next);
            mask.setBrush(next);
        },
        showPreview,
        hidePreview,
        toggleEraser: () => setEraser(!eraser.classList.contains("is-active")),
        teardown: () => {
            document.removeEventListener("keydown", onAltDown);
            document.removeEventListener("keyup", onAltUp);
            hidePreview();
            dropTeardown();
            dropTeardown2();
            dropTeardown3();
            mask.teardown();
        },
    };
}
