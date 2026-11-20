// Pure geometry shared by the crop editor and its headless parity checks.
export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export function normaliseCrop(raw) {
    let value;
    try { value = typeof raw === "string" ? JSON.parse(raw) : raw; } catch { value = {}; }
    const width = clamp(Number(value?.width) || .4, .000001, 1);
    const height = clamp(Number(value?.height) || .4, .000001, 1);
    const x = clamp(Number(value?.x) || 0, 0, 1 - width);
    const y = clamp(Number(value?.y) || 0, 0, 1 - height);
    const keys = new Map();
    for (const key of Array.isArray(value?.keyframes) ? value.keyframes : []) {
        if ([key?.time, key?.x, key?.y].every(Number.isFinite) && key.time >= 0) {
            keys.set(key.time, { time: key.time, x: clamp(key.x, 0, 1 - width), y: clamp(key.y, 0, 1 - height) });
        }
    }
    return { version: 1, enabled: value?.enabled === true, width, height, x, y,
        multiple: clamp(Math.round(Number(value?.multiple) || 32), 1, 256),
        interpolation: value?.interpolation === "linear" ? "linear" : "smooth",
        keyframes: [...keys.values()].sort((a, b) => a.time - b.time) };
}

export function cropPosition(config, seconds) {
    const keys = config.keyframes;
    if (!keys.length) return { x: config.x, y: config.y };
    if (seconds <= keys[0].time) return keys[0];
    for (let i = 1; i < keys.length; i++) {
        const b = keys[i];
        if (seconds > b.time) continue;
        const a = keys[i - 1];
        let t = (seconds - a.time) / (b.time - a.time);
        if (config.interpolation === "smooth") t = t * t * (3 - 2 * t);
        return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
    }
    return keys.at(-1);
}

export function alignedGeometry(config, width, height, seconds) {
    const m = config.multiple;
    const cw = Math.ceil(Math.max(1, Math.round(config.width * width)) / m) * m;
    const ch = Math.ceil(Math.max(1, Math.round(config.height * height)) / m) * m;
    const p = cropPosition(config, seconds);
    return {
        x: clamp(Math.floor((p.x + config.width / 2) * width - cw / 2 + .5), 0, Math.max(0, width - cw)),
        y: clamp(Math.floor((p.y + config.height / 2) * height - ch / 2 + .5), 0, Math.max(0, height - ch)),
        width: Math.min(cw, width), height: Math.min(ch, height), output_width: cw, output_height: ch,
    };
}

/** Hit targets use CSS pixels, independently of video resolution and DPR. */
export function hitCropHandle(point, rect, tolerance = 10) {
    const right = rect.x + rect.width, bottom = rect.y + rect.height;
    if (point.x < rect.x - tolerance || point.x > right + tolerance
        || point.y < rect.y - tolerance || point.y > bottom + tolerance) return null;
    const horizontal = Math.abs(point.x - rect.x) <= tolerance && point.x <= rect.x + rect.width / 2 ? "w"
        : Math.abs(point.x - right) <= tolerance ? "e" : "";
    const vertical = Math.abs(point.y - rect.y) <= tolerance && point.y <= rect.y + rect.height / 2 ? "n"
        : Math.abs(point.y - bottom) <= tolerance ? "s" : "";
    return vertical + horizontal || "move";
}

/** Resize once for the entire clip; motion keys keep their times and trajectory. */
export function resizeCrop(config, seconds, handle, dx, dy, width, height) {
    const next = normaliseCrop(config), p = cropPosition(config, seconds);
    const g = alignedGeometry(config, width, height, seconds);
    const axis = (low, length, total, delta, near, far) => {
        if (!near && !far) return null;
        const anchor = near ? low + length : low;
        const available = near ? anchor : total - anchor;
        const desired = clamp(length + (near ? -delta : delta), 1, available);
        // Snap dimensions, then derive the moving edge from the opposite edge.
        // A whole-frame selection also supports a padded, non-multiple source.
        const maxAligned = Math.floor(available / config.multiple) * config.multiple;
        const size = available === total && desired >= total - config.multiple / 2 ? total
            : maxAligned > 0 ? clamp(Math.round(desired / config.multiple) * config.multiple,
                config.multiple, maxAligned) : available;
        return { start: near ? anchor - size : anchor, size };
    };
    const x = axis(g.x, g.width, width, dx * width, handle.includes("w"), handle.includes("e"));
    const y = axis(g.y, g.height, height, dy * height, handle.includes("n"), handle.includes("s"));
    if (x) next.width = x.size / width;
    if (y) next.height = y.size / height;
    const shiftX = x ? x.start / width - p.x : 0, shiftY = y ? y.start / height - p.y : 0;
    next.x = clamp(config.x + shiftX, 0, 1 - next.width);
    next.y = clamp(config.y + shiftY, 0, 1 - next.height);
    next.keyframes = config.keyframes.map(key => ({ time: key.time,
        x: clamp(key.x + shiftX, 0, 1 - next.width), y: clamp(key.y + shiftY, 0, 1 - next.height) }));
    return next;
}

function roundEven(value) {
    const n = Math.floor(value);
    return value - n === .5 ? n + (n % 2) : Math.round(value);
}

export function processingSize(width, height, longer = 0, shorter = 0, multiple = 1) {
    let w = width >= height ? longer : shorter;
    let h = width >= height ? shorter : longer;
    if (!w && !h) { w = width; h = height; }
    else if (!w) w = roundEven(h * width / height);
    else if (!h) h = roundEven(w * height / width);
    const step = Math.max(1, multiple);
    return { width: Math.max(2, Math.floor(w / step) * step), height: Math.max(2, Math.floor(h / step) * step) };
}
