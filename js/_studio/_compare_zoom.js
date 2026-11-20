const MAX_SCALE = 8;

/** A viewport into the common A/B frame. Its origin is in frame fractions. */
export function bindCompareZoom(surface, frame, isReady, redraw) {
    const view = { scale: 1, x: 0, y: 0 };

    function onWheel(event) {
        if (!event.deltaY || !isReady() || !surface.contains(event.target)
            || event.target.closest("button")) return;
        const rect = frame.getBoundingClientRect();
        if (!(rect.width > 0 && rect.height > 0)) return;
        // Fractions are independent of both the ComfyUI canvas scale and
        // the size of the fullscreen viewport.
        const u = (event.clientX - rect.left) / rect.width;
        const v = (event.clientY - rect.top) / rect.height;
        if (u < 0 || u > 1 || v < 0 || v > 1) return;
        event.preventDefault();
        event.stopPropagation();

        const unit = event.deltaMode === 1 ? 33 : event.deltaMode === 2 ? 400 : 1;
        const delta = Math.max(-200, Math.min(200, event.deltaY * unit));
        const scale = Math.min(MAX_SCALE, Math.max(1, view.scale * 1.25 ** (-delta / 100)));
        if (Math.abs(scale - view.scale) < 1e-9) return;
        // Keep the source point under the cursor unless the visible crop
        // reaches a frame edge, where it must stop to avoid blank padding.
        const change = 1 / view.scale - 1 / scale;
        const limit = 1 - 1 / scale;
        view.x = Math.min(limit, Math.max(0, view.x + u * change));
        view.y = Math.min(limit, Math.max(0, view.y + v * change));
        view.scale = scale;
        redraw();
    }

    // Nodes 2.0 intercepts wheel before it reaches the DOM widget. Capture
    // before that wrapper, and only while the pointer is over this viewer.
    const bind = () => document.addEventListener("wheel", onWheel, { capture: true, passive: false });
    const unbind = () => document.removeEventListener("wheel", onWheel, true);
    surface.addEventListener("pointerenter", bind);
    surface.addEventListener("pointerleave", unbind);

    return {
        view,
        reset() {
            view.scale = 1;
            view.x = 0;
            view.y = 0;
        },
        teardown() {
            unbind();
            surface.removeEventListener("pointerenter", bind);
            surface.removeEventListener("pointerleave", unbind);
        },
    };
}
