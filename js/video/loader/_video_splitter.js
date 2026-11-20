// Fullscreen-only preview/timeline splitter. Node layout stays unchanged.
import { pickLocaleStrings } from "../../_theme.js";

const DEFAULT_HEIGHT = 96;
const STRINGS = {
    en: { label: "Preview / timeline", hint: "Drag to resize the timeline. Double-click to reset. Arrow keys adjust the height." },
    ru: { label: "Превью / таймлайн", hint: "Потяните, чтобы изменить высоту таймлайна. Двойной щелчок — сброс. Стрелки меняют высоту." },
};

export function createVideoSplitter({ root, stage, timeline, onChange, onResize }) {
    const L = pickLocaleStrings(STRINGS);
    const element = document.createElement("div");
    element.className = "ts-vid__splitter";
    element.hidden = true;
    element.tabIndex = 0;
    element.setAttribute("role", "separator");
    element.setAttribute("aria-orientation", "horizontal");
    element.setAttribute("aria-label", L.label);
    element.title = L.hint;
    let fullscreen = false, preferred = DEFAULT_HEIGHT, drag = null, disposed = false;

    function limits() {
        const available = stage.clientHeight + timeline.clientHeight;
        return { min: DEFAULT_HEIGHT, max: Math.max(DEFAULT_HEIGHT,
            Math.min(available - 140, Math.floor(available * .6))) };
    }
    function apply() {
        if (!fullscreen || disposed) return;
        const { min, max } = limits();
        const height = Math.round(Math.max(min, Math.min(max, preferred)));
        root.style.setProperty("--ts-video-timeline-height", `${height}px`);
        element.setAttribute("aria-valuemin", String(min));
        element.setAttribute("aria-valuemax", String(Math.round(max)));
        element.setAttribute("aria-valuenow", String(height));
        onResize?.();
    }
    function change(value, persist = true) {
        const { min, max } = limits();
        preferred = Math.max(min, Math.min(max, value));
        apply();
        if (persist) onChange?.(Math.round(preferred));
    }
    function finish(accept) {
        if (!drag) return;
        const saved = drag; drag = null;
        if (!accept) preferred = saved.before;
        apply();
        if (accept) onChange?.(Math.round(preferred));
        if (element.hasPointerCapture(saved.pointer)) element.releasePointerCapture(saved.pointer);
    }
    element.addEventListener("pointerdown", event => {
        if (!fullscreen || event.button !== 0) return;
        event.preventDefault(); event.stopPropagation();
        drag = { pointer: event.pointerId, y: event.clientY, height: timeline.clientHeight, before: preferred };
        element.setPointerCapture(event.pointerId);
    });
    element.addEventListener("pointermove", event => {
        if (!drag || drag.pointer !== event.pointerId) return;
        change(drag.height - (event.clientY - drag.y), false);
    });
    element.addEventListener("pointerup", () => finish(true));
    element.addEventListener("pointercancel", () => finish(false));
    element.addEventListener("lostpointercapture", () => finish(false));
    element.addEventListener("dblclick", event => {
        event.preventDefault(); event.stopPropagation(); finish(false); change(DEFAULT_HEIGHT);
    });
    const observer = new ResizeObserver(apply);
    observer.observe(root); observer.observe(stage);
    return {
        element,
        setHeight(value) {
            finish(false);
            preferred = Number.isFinite(Number(value)) && Number(value) > 0 ? Number(value) : DEFAULT_HEIGHT;
            apply();
        },
        setFullscreen(flag) {
            finish(false); fullscreen = Boolean(flag);
            root.classList.toggle("ts-vid--fullscreen", fullscreen);
            element.hidden = !fullscreen;
            if (fullscreen) apply(); else root.style.removeProperty("--ts-video-timeline-height");
        },
        onKey(event) {
            if (!fullscreen || event.target !== element || !["ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return false;
            const { min, max } = limits();
            change(event.key === "Home" ? min : event.key === "End" ? max
                : timeline.clientHeight + (event.key === "ArrowUp" ? 16 : -16));
            return true;
        },
        dispose() {
            finish(false); disposed = true; observer.disconnect();
            root.classList.remove("ts-vid--fullscreen");
            root.style.removeProperty("--ts-video-timeline-height"); element.remove();
        },
    };
}
