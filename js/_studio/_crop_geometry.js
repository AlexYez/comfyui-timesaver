// TS Studio kit — где именно окажется вырез под маской (core layer, no DOM).
//
// Ту же геометрию считает нода `TS_StudioInpaintCrop`
// (`nodes/image/_inpaint_crop.py`). Здесь она повторена по одной причине:
// интерфейсу нужно показать превью ровно там, где модель на самом деле рисует,
// а спрашивать об этом сервер посреди генерации — дороже и медленнее, чем
// посчитать четыре числа.
//
// Раздвоение опасно тем, что расходится молча, поэтому обе реализации сверяет
// один тест (`tests/test_crop_geometry_parity.py`): он гоняет этот файл под
// Node, ту же функцию под Python и сравнивает рамки на одинаковых случаях.
// Меняя числа здесь, поменяйте и там — тест не даст забыть.

export const CONTEXT_CEIL_PX = 1024;
export const CONTEXT_FLOOR_PX = 64;
export const CONTEXT_FLOOR_IMAGE_PCT = 3.0;
export const CC_ANALYSIS_MARGIN_PX = 32;
export const KNOWN_AREA_MIN = 0.5;
export const KNOWN_AREA_MIN_REPLACE = 0.75;
export const REPLACE_DENOISE = 0.6;

/** Проценты от размера маски — в пиксели, зажатые в [floor, ceil]. */
export function pctToPx(pct, basePx, floorPx, ceilPx) {
    return Math.min(ceilPx, Math.max(floorPx, (pct / 100) * basePx));
}

/**
 * Насколько отступить от рамки маски, чтобы она заняла меньше `knownMin`
 * площади выреза. Решается точно: (w+2p)(h+2p) >= w*h / (1-knownMin).
 */
export function padForKnownArea(maskW, maskH, knownMin = KNOWN_AREA_MIN) {
    if (maskW <= 0 || maskH <= 0) return 0;
    const target = (maskW * maskH) / Math.max(1e-6, 1 - knownMin);
    const b = 2 * (maskW + maskH);
    const c = maskW * maskH - target;
    const disc = b * b - 16 * c;
    if (disc <= 0) return 0;
    return Math.max(0, (-b + Math.sqrt(disc)) / 8);
}

/**
 * Рамка выреза в пикселях исходного кадра.
 *
 * @param {object} o
 * @param {number} o.imageW ширина кадра
 * @param {number} o.imageH высота кадра
 * @param {{x:number,y:number,w:number,h:number}} o.mask плотная рамка маски
 * @param {number} [o.contextPct] запас вокруг маски, % от её размера
 * @param {number} [o.denoise] сила перерисовки: при полной замене контекста нужно больше
 * @returns {{x0:number,y0:number,x1:number,y1:number}|null}
 */
export function cropBox({ imageW, imageH, mask, contextPct = 25, denoise = 1 }) {
    if (!mask || mask.w <= 0 || mask.h <= 0) return null;
    const maskMinSide = Math.min(mask.w, mask.h);

    let pad = pctToPx(contextPct, maskMinSide, 0, CONTEXT_CEIL_PX);
    pad = Math.max(pad, CONTEXT_FLOOR_PX, (CONTEXT_FLOOR_IMAGE_PCT / 100) * Math.min(imageH, imageW));
    pad = Math.max(pad, CC_ANALYSIS_MARGIN_PX);
    const knownMin = denoise >= REPLACE_DENOISE ? KNOWN_AREA_MIN_REPLACE : KNOWN_AREA_MIN;
    pad = Math.max(pad, padForKnownArea(mask.w, mask.h, knownMin));

    const grow = Math.round(pad);
    const x0 = Math.max(0, mask.x - grow);
    const y0 = Math.max(0, mask.y - grow);
    const x1 = Math.min(imageW, mask.x + mask.w + grow);
    const y1 = Math.min(imageH, mask.y + mask.h + grow);
    if (x1 - x0 <= 0 || y1 - y0 <= 0) return null;
    return { x0, y0, x1, y1 };
}
