// Fullscreen-only spatial crop editing; one editor instance survives tab changes.
import { pickLocaleStrings } from "../../_theme.js";
import { formatTimecode } from "../../_media/_ruler.js";
import { alignedGeometry, clamp, cropPosition, hitCropHandle, normaliseCrop, resizeCrop } from "./_spatial_crop_model.js";

const STRINGS = {
    en: {
        crop: "Crop", draw: "New region", key: "+ Keyframe", remove: "Delete keyframe",
        off: "Crop off", on: "Crop on", multiple: "Multiple", smooth: "Smooth", linear: "Linear",
        hint: "Draw a region (Shift: square). Drag an edge or corner to resize, or drag inside to move. Size applies to the whole clip; only position is animated.",
        newHint: "Draw a new region and reset its motion keyframes.",
        keyHint: "Save position at the playhead. After the first keyframe, dragging creates or updates a keyframe here.",
        removeHint: "Remove the keyframe at the playhead.",
        pad: "edge padding", full: "Enable or disable spatial cropping for this clip.",
    },
    ru: {
        crop: "Кроп", draw: "Новая область", key: "+ Ключ", remove: "Удалить ключ",
        off: "Кроп выключен", on: "Кроп включён", multiple: "Кратность", smooth: "Плавно", linear: "Линейно",
        hint: "Нарисуйте область (Shift — квадрат). Тяните за край или угол для размера, внутри — для сдвига. Размер общий для ролика; анимируется положение.",
        newHint: "Нарисовать новую область и сбросить ключи её движения.",
        keyHint: "Запомнить положение на текущем кадре. После первого ключа перетаскивание создаёт или меняет ключ на текущем кадре.",
        removeHint: "Удалить ключ на текущем кадре.",
        pad: "дополнение у края", full: "Включить или выключить пространственный кроп этого ролика.",
    },
};

export function createSpatialCropEditor({ editor, getSize, onChange, isAvailable = () => true }) {
    const L = pickLocaleStrings(STRINGS);
    const canvas = document.createElement("canvas");
    canvas.className = "ts-vid__crop-canvas";
    canvas.hidden = true;
    const panel = document.createElement("div");
    panel.className = "ts-vid__crop-panel";
    panel.hidden = true;
    const toolbar = document.createElement("div");
    toolbar.className = "ts-ui-toolbar ts-vid__crop-toolbar";
    const chips = document.createElement("div");
    chips.className = "ts-vid__crop-keys";
    const hint = document.createElement("div");
    hint.className = "ts-vid__crop-hint";
    hint.textContent = L.hint;
    const stats = document.createElement("span");
    stats.className = "ts-vid__crop-stats";
    let config = normaliseCrop({}), fullscreen = false, drawMode = false, drag = null;
    let disposed = false, raf = 0, hover = null;
    const cursors = { n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize",
        ne: "nesw-resize", sw: "nesw-resize", nw: "nwse-resize", se: "nwse-resize", move: "grab" };
    const undo = [], redo = [];
    const time = () => Number((editor.video.currentTime || 0).toFixed(6));
    const clone = (value = config) => JSON.parse(JSON.stringify(value));
    const tc = (seconds) => formatTimecode(seconds, { fps: editor.state.fps || 24 });
    const tolerance = () => .5 / Math.max(1, editor.state.fps || 24);
    const currentKey = () => config.keyframes.findIndex((key) => Math.abs(key.time - time()) <= tolerance());

    function action(label, title, callback) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-ui-btn";
        button.textContent = label;
        button.title = title;
        button.addEventListener("click", callback);
        toolbar.append(button);
        return button;
    }
    const toggle = action(L.off, L.full, () => {
        remember(); config.enabled = !config.enabled; drawMode = config.enabled && !config.keyframes.length;
        commit();
    });
    const draw = action(L.draw, L.newHint, () => { drawMode = true; schedule(); });
    const add = action(L.key, L.keyHint, () => {
        remember();
        const p = cropPosition(config, time());
        if (!config.keyframes.length && time() > editor.bounds().left + tolerance()) {
            config.keyframes.push({ time: editor.bounds().left, x: config.x, y: config.y });
        }
        upsert(p.x, p.y);
        drawMode = false;
        commit();
    });
    const remove = action(L.remove, L.removeHint, () => {
        const index = currentKey();
        if (index < 0) return;
        remember();
        const p = cropPosition(config, time());
        config.keyframes.splice(index, 1);
        if (!config.keyframes.length) { config.x = p.x; config.y = p.y; }
        commit();
    });
    const multiple = document.createElement("select");
    multiple.className = "ts-ui-input ts-vid__crop-select";
    multiple.title = L.multiple;
    for (const value of [1, 8, 16, 32, 64, 128, 256]) {
        multiple.add(new Option(`${L.multiple}: ${value}`, String(value)));
    }
    multiple.addEventListener("change", () => { remember(); config.multiple = Number(multiple.value); commit(); });
    const interpolation = document.createElement("select");
    interpolation.className = "ts-ui-input ts-vid__crop-select";
    interpolation.add(new Option(L.smooth, "smooth"));
    interpolation.add(new Option(L.linear, "linear"));
    interpolation.addEventListener("change", () => { remember(); config.interpolation = interpolation.value; commit(); });
    toolbar.append(multiple, interpolation, stats);
    panel.append(toolbar, chips, hint);
    editor.stage.append(canvas);
    editor.stage.after(panel);

    function remember(before = clone()) {
        undo.push(before); if (undo.length > 50) undo.shift(); redo.length = 0;
    }
    function upsert(x, y) {
        const index = currentKey();
        const point = { time: time(), x: clamp(x, 0, 1 - config.width), y: clamp(y, 0, 1 - config.height) };
        if (index >= 0) config.keyframes[index] = point; else config.keyframes.push(point);
        config.keyframes.sort((a, b) => a.time - b.time);
    }
    function commit() {
        onChange(JSON.stringify(config));
        editor.setSpatialEditing(fullscreen && config.enabled && isAvailable());
        rebuildChips(); schedule();
    }
    function rebuildChips() {
        chips.replaceChildren();
        for (const key of config.keyframes) {
            const button = document.createElement("button");
            button.type = "button"; button.className = "ts-ui-btn";
            button.textContent = `◆ ${tc(key.time)}`;
            button.addEventListener("click", () => editor.seekPlayhead(key.time, { immediate: true }));
            chips.append(button);
        }
    }
    function imageRect() {
        const box = canvas.getBoundingClientRect();
        const vw = editor.state.width || editor.video.videoWidth;
        const vh = editor.state.height || editor.video.videoHeight;
        if (!vw || !vh || !box.width || !box.height) return null;
        const scale = Math.min(box.width / vw, box.height / vh);
        return { left: (box.width - vw * scale) / 2, top: (box.height - vh * scale) / 2,
            width: vw * scale, height: vh * scale, box };
    }
    function pointer(event) {
        const r = imageRect();
        if (!r) return null;
        return { x: clamp((event.clientX - r.box.left - r.left) / r.width, 0, 1),
            y: clamp((event.clientY - r.box.top - r.top) / r.height, 0, 1) };
    }
    function hit(event) {
        const r = imageRect(); if (!r) return null;
        const size = getSize(), g = alignedGeometry(config, size.width, size.height, time());
        return hitCropHandle({ x: event.clientX - r.box.left, y: event.clientY - r.box.top },
            { x: r.left + g.x / size.width * r.width, y: r.top + g.y / size.height * r.height,
                width: g.width / size.width * r.width, height: g.height / size.height * r.height });
    }
    function setCursor() {
        canvas.style.cursor = drawMode ? "crosshair"
            : drag?.kind === "move" ? "grabbing" : cursors[drag?.kind || hover] || "default";
    }
    function render() {
        raf = 0;
        if (disposed || !fullscreen) return;
        panel.hidden = false;
        toggle.disabled = !isAvailable() || !editor.state.path || !editor.state.width || !editor.state.playable;
        canvas.hidden = !config.enabled || toggle.disabled;
        editor.setSpatialEditing(config.enabled && !toggle.disabled);
        toggle.textContent = config.enabled ? L.on : L.off;
        toggle.classList.toggle("is-active", config.enabled);
        draw.classList.toggle("is-active", drawMode);
        for (const element of [draw, add, multiple, interpolation]) element.disabled = !config.enabled || toggle.disabled;
        remove.disabled = !config.enabled || toggle.disabled || currentKey() < 0;
        multiple.value = String(config.multiple); interpolation.value = config.interpolation;
        if (!config.enabled || toggle.disabled) { stats.textContent = ""; return; }
        const r = imageRect();
        if (!r) return;
        const size = getSize();
        const g = alignedGeometry(config, size.width, size.height, time());
        stats.textContent = `${g.output_width} × ${g.output_height}`
            + (g.output_width > size.width || g.output_height > size.height ? ` · ${L.pad}` : "");
        setCursor();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(r.box.width * dpr); canvas.height = Math.round(r.box.height * dpr);
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const x = r.left + g.x / size.width * r.width, y = r.top + g.y / size.height * r.height;
        const w = g.width / size.width * r.width, h = g.height / size.height * r.height;
        ctx.fillStyle = "rgba(0,0,0,.5)";
        ctx.beginPath(); ctx.rect(r.left, r.top, r.width, r.height); ctx.rect(x, y, w, h); ctx.fill("evenodd");
        if (config.keyframes.length > 1) {
            ctx.strokeStyle = "rgba(255,255,255,.6)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]); ctx.beginPath();
            config.keyframes.forEach((key, index) => {
                const px = r.left + (key.x + config.width / 2) * r.width;
                const py = r.top + (key.y + config.height / 2) * r.height;
                if (!index) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            });
            ctx.stroke(); ctx.setLineDash([]);
        }
        ctx.strokeStyle = "white"; ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = "white";
        for (const [px, py] of [[x, y], [x + w / 2, y], [x + w, y], [x, y + h / 2],
            [x + w, y + h / 2], [x, y + h], [x + w / 2, y + h], [x + w, y + h]]) {
            ctx.fillRect(px - 4, py - 4, 8, 8);
        }
        for (const [index, button] of [...chips.children].entries()) {
            const key = config.keyframes[index];
            button.classList.toggle("is-active", Boolean(key && Math.abs(key.time - time()) <= tolerance()));
        }
        if (!editor.video.paused) schedule();
    }
    function schedule() { if (!disposed && !raf) raf = requestAnimationFrame(render); }
    canvas.addEventListener("pointerdown", (event) => {
        if (drag || !fullscreen || !config.enabled || !isAvailable() || event.button !== 0) return;
        const p = pointer(event); if (!p) return;
        const kind = drawMode ? "draw" : hit(event);
        if (!kind) return;
        event.preventDefault(); event.stopPropagation(); editor.playback.pause();
        editor.seekPlayhead(time(), { immediate: true });
        drag = { kind, start: p, before: clone(), position: cropPosition(config, time()), pointer: event.pointerId, moved: false };
        canvas.setPointerCapture(event.pointerId);
        setCursor();
    });
    canvas.addEventListener("pointermove", (event) => {
        if (!drag) { hover = hit(event); setCursor(); return; }
        if (drag.pointer !== event.pointerId) return;
        const p = pointer(event); if (!p) return;
        const dx = p.x - drag.start.x, dy = p.y - drag.start.y;
        if (Math.abs(dx) + Math.abs(dy) < .00001) {
            if (drag.moved) {
                config = clone(drag.before);
                drag.moved = false;
                schedule();
            }
            return;
        }
        drag.moved = true;
        if (drag.kind === "draw") {
            const size = getSize();
            let width = Math.abs(dx), height = Math.abs(dy);
            if (event.shiftKey) {
                const pixels = Math.min(Math.max(width * size.width, height * size.height),
                    (dx < 0 ? drag.start.x : 1 - drag.start.x) * size.width,
                    (dy < 0 ? drag.start.y : 1 - drag.start.y) * size.height);
                width = pixels / size.width; height = pixels / size.height;
            }
            config.width = Math.max(1 / size.width, width); config.height = Math.max(1 / size.height, height);
            config.x = clamp(dx < 0 ? drag.start.x - config.width : drag.start.x, 0, 1 - config.width);
            config.y = clamp(dy < 0 ? drag.start.y - config.height : drag.start.y, 0, 1 - config.height);
            config.keyframes = [];
        } else if (drag.kind === "move") {
            config = clone(drag.before);
            const x = clamp(drag.position.x + dx, 0, 1 - config.width);
            const y = clamp(drag.position.y + dy, 0, 1 - config.height);
            if (config.keyframes.length) upsert(x, y); else { config.x = x; config.y = y; }
        } else {
            const size = getSize();
            config = resizeCrop(drag.before, time(), drag.kind, dx, dy, size.width, size.height);
        }
        schedule();
    });
    function finish(accept) {
        if (!drag) return;
        const saved = drag; drag = null;
        if (accept && saved.moved && JSON.stringify(config) !== JSON.stringify(saved.before)) {
            remember(saved.before); drawMode = false; commit();
        }
        else { config = saved.before; schedule(); }
        if (canvas.hasPointerCapture(saved.pointer)) canvas.releasePointerCapture(saved.pointer);
    }
    canvas.addEventListener("pointerup", (event) => { if (drag?.pointer === event.pointerId) finish(true); });
    canvas.addEventListener("pointerleave", () => { if (!drag) { hover = null; setCursor(); } });
    const cancelPointer = (event) => { if (drag?.pointer === event.pointerId) finish(false); };
    canvas.addEventListener("pointercancel", cancelPointer);
    canvas.addEventListener("lostpointercapture", cancelPointer);
    const events = ["timeupdate", "seeked", "play", "pause", "loadedmetadata"];
    for (const name of events) editor.video.addEventListener(name, schedule);
    const observer = new ResizeObserver(schedule); observer.observe(editor.stage);
    return {
        setFullscreen(flag) {
            finish(false); fullscreen = flag; panel.hidden = !flag; canvas.hidden = !flag || !config.enabled;
            editor.setSpatialEditing(flag && config.enabled && isAvailable()); schedule();
        },
        setValue(value) { finish(false); config = normaliseCrop(value); drawMode = false; hover = null;
            rebuildChips(); editor.setSpatialEditing(fullscreen && config.enabled && isAvailable()); schedule(); },
        reset() { finish(false); config = normaliseCrop({}); drawMode = false; undo.length = 0; redo.length = 0; commit(); },
        refresh: schedule,
        onKey(event) {
            if (!fullscreen || !(event.ctrlKey || event.metaKey)) return false;
            const key = event.key.toLowerCase();
            if (!["z", "y"].includes(key)) return false;
            finish(false);
            const back = key === "z" && !event.shiftKey;
            const from = back ? undo : redo, to = back ? redo : undo;
            if (from.length) { to.push(clone()); config = from.pop(); commit(); }
            return true;
        },
        dispose() {
            disposed = true; finish(false); observer.disconnect(); cancelAnimationFrame(raf);
            for (const name of events) editor.video.removeEventListener(name, schedule);
            canvas.remove(); panel.remove();
        },
    };
}
