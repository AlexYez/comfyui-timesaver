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
.ts-inp__bar{position:absolute;top:8px;left:50%;transform:translateX(-50%);z-index:5;
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

    const eraser = tool("◐", ctx.t.inp.eraser);
    eraser.addEventListener("click", () => {
        const on = !eraser.classList.contains("is-active");
        eraser.classList.toggle("is-active", on);
        mask.setEraser(on);
    });
    const clear = tool("✕", ctx.t.inp.clear);
    clear.addEventListener("click", () => mask.clearMask());
    const undoBtn = tool("↶", ctx.t.inp.undo);
    const redoBtn = tool("↷", ctx.t.inp.redo);
    undoBtn.addEventListener("click", () => history.undo());
    redoBtn.addEventListener("click", () => history.redo());

    const sep1 = separator();
    const sep2 = separator();
    bar.append(seg, sep1, brush, eraser, clear, sep2, undoBtn, redoBtn);

    const status = document.createElement("div");
    status.className = "ts-inp__status";
    status.style.display = "none";

    const pip = document.createElement("img");
    pip.className = "ts-inp__pip";
    pip.alt = "";
    root.append(mask.element, empty, bar, status, pip);

    let pipUrl = "";
    let previewBox = null;   // frozen at run start: where the mask was painted

    function capturePreviewBox() {
        const bbox = mask.maskBBox?.();
        previewBox = bbox ? { bbox, css: mask.imageRectToCss(bbox) } : null;
    }

    async function showPreview(blob) {
        // The preview belongs WHERE THE MASK WAS PAINTED. Two shapes arrive:
        // a full-frame latent preview (LanPaint recipes) — crop our bbox out
        // of it; a crop preview (TSSmartInpaint) — its aspect matches the
        // mask region, place it whole. Distinguish by aspect ratio.
        if (!previewBox) capturePreviewBox();
        if (pipUrl) URL.revokeObjectURL(pipUrl);
        if (!previewBox) {
            pipUrl = URL.createObjectURL(blob);
            pip.src = pipUrl;
            pip.style.cssText = "";
            pip.classList.add("is-active");
            return;
        }
        const bitmap = await createImageBitmap(blob);
        const image = mask.imageSize();
        const frameAspect = image.w / image.h;
        const previewAspect = bitmap.width / bitmap.height;
        const { bbox, css } = previewBox;
        let source = bitmap;
        if (Math.abs(previewAspect - frameAspect) / frameAspect < 0.12) {
            const sx = bitmap.width / image.w;
            const sy = bitmap.height / image.h;
            const cut = document.createElement("canvas");
            cut.width = Math.max(1, Math.round(bbox.w * sx));
            cut.height = Math.max(1, Math.round(bbox.h * sy));
            cut.getContext("2d").drawImage(bitmap,
                bbox.x * sx, bbox.y * sy, bbox.w * sx, bbox.h * sy,
                0, 0, cut.width, cut.height);
            source = cut;
        }
        pipUrl = source instanceof HTMLCanvasElement
            ? source.toDataURL("image/png") : URL.createObjectURL(blob);
        pip.src = pipUrl;
        pip.style.cssText = `left:${css.left}px;top:${css.top}px;` +
            `width:${css.width}px;height:${css.height}px;right:auto;bottom:auto;` +
            `object-fit:cover;`;
        pip.classList.add("is-active");
        if (source !== bitmap) bitmap.close?.();
    }
    function hidePreview() {
        pip.classList.remove("is-active");
        pip.style.cssText = "";
        previewBox = null;
        if (pipUrl && pipUrl.startsWith("blob:")) URL.revokeObjectURL(pipUrl);
        pipUrl = "";
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
    }

    async function setImageFromUrl(url, name) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        await setImageFromBlob(await response.blob(), name);
    }

    const dropTeardown = makeDropZone(empty, {
        max: 1,
        onDrop: async ([item]) => setImageFromBlob(await item.getBlob(), item.name),
    });
    const dropTeardown2 = makeDropZone(mask.element, {
        max: 1,
        onDrop: async ([item]) => setImageFromBlob(await item.getBlob(), item.name),
    });

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
        if (!mask.hasMask()) throw new Error(ctx.t.inp.needMask);
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
        undo: () => history.undo(),
        redo: () => history.redo(),
        brushDelta: (delta) => {
            const next = Math.max(6, Math.min(200, Number(brush.value) + delta));
            brush.value = String(next);
            mask.setBrush(next);
        },
        showPreview,
        hidePreview,
        teardown: () => {
            hidePreview();
            dropTeardown();
            dropTeardown2();
            mask.teardown();
        },
    };
}
