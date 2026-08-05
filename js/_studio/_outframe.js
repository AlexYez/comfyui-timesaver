// TS Studio kit — куда именно дорисуется кадр (ui-kit layer).
//
// Расширение кадра — единственная операция студии, у которой результат больше
// исходника. Без показа человек выбирает пропорцию вслепую: «21:9» ничего не
// говорит о том, сколько именно допишется слева и справа и не срежется ли верх.
//
// Поэтому поверх сцены лежит рамка будущего кадра: сам снимок остаётся на
// месте, а новые области видно — они заштрихованы и медленно дышат, пока идёт
// работа. Когда приходят первые превью, они рисуются уже в этих границах, и
// заполнение читается само.
//
// Геометрия здесь чистая (`outpaintFrame`) и проверяется без браузера: она
// повторяет то, что делает `TS_ResolutionSelector` — вписывает исходник в
// холст выбранной пропорции по центру.

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";

const STYLE_ID = "ts-studio-outframe-styles";

export function ensureOutFrameStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-outframe{position:absolute;display:none;pointer-events:none;z-index:4}
.ts-outframe.is-active{display:block}
/* Рамка будущего кадра: тонкая линия акцента, чтобы границу было видно на
   любой картинке, и подпись с итоговым размером. */
.ts-outframe__box{position:absolute;inset:0;border:1px solid var(--ts-accent-line);
    border-radius:2px}
/* Новые области. Штриховка — не украшение: сплошная заливка читалась бы как
   «здесь уже что-то есть», а полоски сразу говорят «пока пусто». */
.ts-outframe__band{position:absolute;background:var(--ts-scrim);
    background-image:repeating-linear-gradient(45deg,
        transparent 0 6px, var(--ts-accent-soft) 6px 12px);
    animation:ts-outframe-breathe 2.6s ease-in-out infinite}
@keyframes ts-outframe-breathe{0%,100%{opacity:.75}50%{opacity:.45}}
.ts-outframe__size{position:absolute;left:50%;bottom:-11px;transform:translateX(-50%);
    padding:2px 8px;border-radius:999px;font-size:var(--ts-fs-xs);font-weight:600;
    letter-spacing:.02em;white-space:nowrap;
    background:var(--ts-scrim-strong);color:var(--ts-on-media)}
`;
    document.head.appendChild(style);
}

/**
 * Разобрать пропорцию «21:9» в число. Мусор считается квадратом — так же, как
 * это делает нода: лучше показать квадрат, чем ничего.
 *
 * @param {string} text
 * @returns {number}
 */
export function parseFrameAspect(text) {
    const [w, h] = String(text || "").split(":").map(Number);
    return w > 0 && h > 0 ? w / h : 1;
}

/**
 * Где окажется исходник в новом кадре и какие полосы предстоит дорисовать.
 *
 * Повторяет `_fit_image_to_canvas` из `TS_ResolutionSelector`: картинка
 * вписывается в холст целиком и по центру, всё остальное — новое.
 *
 * @param {{width:number,height:number}} image размер исходника (пиксели экрана)
 * @param {number} aspect ширина/высота нового кадра
 * @returns {{frame:{width:number,height:number},
 *            inner:{left:number,top:number,width:number,height:number},
 *            bands:Array<{left:number,top:number,width:number,height:number}>}}
 */
export function outpaintFrame(image, aspect) {
    const width = Math.max(1, image?.width || 0);
    const height = Math.max(1, image?.height || 0);
    const ratio = aspect > 0 ? aspect : 1;
    const own = width / height;

    // Кадр охватывает исходник целиком: растёт та сторона, которой не хватает.
    const frame = own > ratio
        ? { width, height: width / ratio }
        : { width: height * ratio, height };

    const inner = {
        left: (frame.width - width) / 2,
        top: (frame.height - height) / 2,
        width,
        height,
    };

    const bands = [];
    const push = (left, top, w, h) => {
        if (w > 0.5 && h > 0.5) bands.push({ left, top, width: w, height: h });
    };
    push(0, 0, frame.width, inner.top);                                   // сверху
    push(0, inner.top + inner.height, frame.width,
         frame.height - inner.top - inner.height);                        // снизу
    push(0, inner.top, inner.left, inner.height);                         // слева
    push(inner.left + inner.width, inner.top,
         frame.width - inner.left - inner.width, inner.height);           // справа
    return { frame, inner, bands };
}

/**
 * Итоговый размер в пикселях — тот же расчёт, что в ноде: бюджет мегапикселей
 * раскладывается по выбранной пропорции и округляется вниз до кратности 32.
 *
 * @param {number} aspect ширина/высота
 * @param {number} megapixels бюджет
 * @returns {{width:number,height:number}}
 */
export function frameResolution(aspect, megapixels) {
    const mp = Math.max(0.05, Number(megapixels) || 1);
    const ratio = aspect > 0 ? aspect : 1;
    const height = Math.sqrt((mp * 1_000_000) / ratio);
    const snap = (value) => Math.max(32, Math.round(value / 32) * 32);
    return { width: snap(height * ratio), height: snap(height) };
}

/**
 * @returns {{element, show, hide, isActive}}
 */
export function createOutpaintFrame() {
    ensureOutFrameStyles();
    const element = document.createElement("div");
    element.className = `${TS_UI_CLASS} ts-outframe`;
    const box = document.createElement("div");
    box.className = "ts-outframe__box";
    const size = document.createElement("div");
    size.className = "ts-outframe__size";
    element.append(box, size);

    return {
        element,
        /**
         * Показать рамку вокруг картинки на сцене.
         *
         * @param {object} options
         * @param {DOMRect} options.imageRect прямоугольник картинки
         * @param {DOMRect} options.hostRect коробка, в которой лежит рамка
         * @param {string} options.aspect выбранная пропорция, «21:9»
         * @param {number} options.megapixels бюджет размера
         * @param {string} [options.label] подпись; по умолчанию — размер
         */
        show({ imageRect, hostRect, aspect, megapixels, label }) {
            if (!imageRect || !hostRect || !(imageRect.height > 0)) return false;
            const ratio = parseFrameAspect(aspect);
            const plan = outpaintFrame(imageRect, ratio);
            // Рамка может вылезти за сцену — она и должна: человеку важно
            // видеть, что кадр станет шире, даже если целиком он не помещается.
            element.style.left = `${imageRect.left - hostRect.left - plan.inner.left}px`;
            element.style.top = `${imageRect.top - hostRect.top - plan.inner.top}px`;
            element.style.width = `${plan.frame.width}px`;
            element.style.height = `${plan.frame.height}px`;

            for (const old of [...element.querySelectorAll(".ts-outframe__band")]) {
                old.remove();
            }
            for (const band of plan.bands) {
                const strip = document.createElement("div");
                strip.className = "ts-outframe__band";
                strip.style.left = `${band.left}px`;
                strip.style.top = `${band.top}px`;
                strip.style.width = `${band.width}px`;
                strip.style.height = `${band.height}px`;
                element.insertBefore(strip, box);
            }
            const target = frameResolution(ratio, megapixels);
            size.textContent = label || `${target.width} × ${target.height}`;
            element.classList.add("is-active");
            return plan.bands.length > 0;
        },
        hide() {
            element.classList.remove("is-active");
            for (const band of [...element.querySelectorAll(".ts-outframe__band")]) {
                band.remove();
            }
        },
        isActive: () => element.classList.contains("is-active"),
    };
}
