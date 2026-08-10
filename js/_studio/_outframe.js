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
/* Рамка растянута по коробке отведённого места — сцена уже отвела его под
   БУДУЩИЙ кадр. Поэтому здесь нет ни одного пикселя из замеров экрана: зум и
   панорама двигают коробку вместе с рамкой, и разъехаться нечему. */
.ts-outframe{position:absolute;inset:0;display:none;pointer-events:none;z-index:4}
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
/* Стрелка в каждой новой области — куда именно поедет край кадра. Полоска
   говорит «здесь пусто», стрелка добавляет «и вот в эту сторону оно вырастет»:
   при выборе пропорции это единственное, что читается мгновенно. */
.ts-outframe__arrow{position:absolute;left:50%;top:50%;width:34px;height:34px;
    margin:-17px 0 0 -17px;display:flex;align-items:center;justify-content:center;
    border-radius:999px;background:var(--ts-scrim-strong);color:var(--ts-on-media);
    font-size:15px;line-height:1;font-weight:700;
    animation:ts-outframe-nudge 1.8s ease-in-out infinite}
/* Каждая сторона толкает свою стрелку в свою сторону: направление и есть
   сообщение, поэтому у полос оно разное. */
.ts-outframe__band[data-side="top"] .ts-outframe__arrow{--ts-nudge-x:0px;--ts-nudge-y:-7px}
.ts-outframe__band[data-side="bottom"] .ts-outframe__arrow{--ts-nudge-x:0px;--ts-nudge-y:7px}
.ts-outframe__band[data-side="left"] .ts-outframe__arrow{--ts-nudge-x:-7px;--ts-nudge-y:0px}
.ts-outframe__band[data-side="right"] .ts-outframe__arrow{--ts-nudge-x:7px;--ts-nudge-y:0px}
@keyframes ts-outframe-nudge{
    0%,100%{transform:translate(0,0);opacity:.85}
    50%{transform:translate(var(--ts-nudge-x,0),var(--ts-nudge-y,0));opacity:1}}
/* Узкая полоса не вместит кружок — там честнее ничего не рисовать, чем
   выпускать стрелку за пределы области. */
.ts-outframe__band.is-narrow .ts-outframe__arrow{display:none}
/* ⚠️ Бегущего света по новым областям здесь больше НЕТ. Он появился, когда
   рамка была единственным признаком работы; теперь работу показывает кольцо
   поверх кадра, а полосы, едущие слева направо, стали лишним слоем поверх
   картинки. Убрано по просьбе владельца — не возвращать без повода. */
/* Пришло превью — штриховка и стрелки не нужны: под ними уже видно, что
   именно дорисовалось. Контур кадра остаётся. */
.ts-outframe.is-previewing .ts-outframe__band{background:none;
    background-image:none;animation:none}
.ts-outframe.is-previewing .ts-outframe__arrow{opacity:0}
.ts-outframe.is-working .ts-outframe__box{border-color:var(--ts-accent);
    box-shadow:0 0 0 1px var(--ts-accent-line)}
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
 *            bands:Array<{left:number,top:number,width:number,height:number,
 *                         side:"top"|"bottom"|"left"|"right"}>}}
 */
/**
 * Ниже этой доли кадра прирост не показываем.
 *
 * Два процента на глаз — это несколько пикселей на любом разумном экране:
 * рисовать там полосу значит показывать шов, которого человек не заказывал.
 */
const MIN_GROWTH = 0.02;

export function outpaintFrame(image, aspect) {
    // ⚠️ Защита от нуля — это ПРОВЕРКА, а не `Math.max(1, ...)`. Так здесь и
    // было, пока размер приходил в пикселях. Потом расчёт перешёл на доли, и
    // вертикальный кадр (ширина 0.5625 при высоте 1) молча превращался в
    // квадрат: полосы считались уже, чем надо, и между ними и картинкой
    // оставались серые прямоугольники. На горизонтальных кадрах ширина всегда
    // больше единицы, поэтому годами не всплывало.
    const width = image?.width > 0 ? image.width : 1;
    const height = image?.height > 0 ? image.height : 1;
    const ratio = aspect > 0 ? aspect : 1;
    const own = width / height;

    // Кадр охватывает исходник целиком: растёт та сторона, которой не хватает.
    const frame = own > ratio
        ? { width, height: width / ratio }
        : { width: height * ratio, height };

    // ⚠️ Почти совпавшая пропорция — это НЕ повод рисовать полоску.
    // Результат прошлого прогона нода округляет до кратности 32, поэтому
    // «21:9» на деле 2.320 вместо 2.333: по краям остаётся несколько пикселей,
    // и на экране они читались серой ниткой между кадром и новой областью
    // (замечено владельцем при повторном расширении). Если прирост меньше
    // порога — считаем, что по этой оси расширять нечего, и кадр занимает её
    // целиком.
    const growX = (frame.width - width) / frame.width;
    const growY = (frame.height - height) / frame.height;
    const inner = {
        left: growX < MIN_GROWTH ? 0 : (frame.width - width) / 2,
        top: growY < MIN_GROWTH ? 0 : (frame.height - height) / 2,
        width: growX < MIN_GROWTH ? frame.width : width,
        height: growY < MIN_GROWTH ? frame.height : height,
    };

    const bands = [];
    // Сторона едет вместе с полосой: по ней рисуется стрелка, и вычислять её
    // заново в отрисовке значило бы считать одну геометрию дважды.
    // ⚠️ Порог — ДОЛЯ кадра, а не пиксели. Когда расчёт перешёл на доли,
    // прежнее «больше половины пикселя» стало «больше половины кадра», и все
    // полосы молча отбрасывались: рамка не появлялась вовсе.
    const minW = frame.width * 0.002;
    const minH = frame.height * 0.002;
    const push = (left, top, w, h, side) => {
        if (w > minW && h > minH) bands.push({ left, top, width: w, height: h, side });
    };
    push(0, 0, frame.width, inner.top, "top");
    push(0, inner.top + inner.height, frame.width,
         frame.height - inner.top - inner.height, "bottom");
    push(0, inner.top, inner.left, inner.height, "left");
    push(inner.left + inner.width, inner.top,
         frame.width - inner.left - inner.width, inner.height, "right");
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
/** Куда показывает стрелка в каждой из новых областей. */
const ARROW_BY_SIDE = { top: "\u2191", bottom: "\u2193",
                        left: "\u2190", right: "\u2192" };

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
         * Единицы здесь — доли самой рамки, а не пиксели экрана: сцена уже
         * отвела место под будущий кадр, рамка растянута по нему, а полосы
         * ставятся в процентах. Так показ не зависит ни от зума, ни от размера
         * окна, и пересчитывать его не нужно вовсе.
         *
         * @param {object} options
         * @param {number} options.imageRatio ширина/высота картинки на сцене
         * @param {string} options.aspect выбранная пропорция, «21:9»
         * @param {number} options.megapixels бюджет размера
         * @param {string} [options.label] подпись; по умолчанию — размер
         */
        show({ imageRatio, aspect, megapixels, label }) {
            if (!(imageRatio > 0)) return false;
            const ratio = parseFrameAspect(aspect);
            const plan = outpaintFrame({ width: imageRatio, height: 1 }, ratio);
            const pctX = (value) => `${(value / plan.frame.width) * 100}%`;
            const pctY = (value) => `${(value / plan.frame.height) * 100}%`;

            for (const old of [...element.querySelectorAll(".ts-outframe__band")]) {
                old.remove();
            }
            // Дорисовывать нечего — показывать рамку не о чем. Так же и после
            // прогона: результат уже нужной формы, полос вокруг него нет.
            if (!plan.bands.length) {
                element.classList.remove("is-active");
                const target0 = frameResolution(ratio, megapixels);
                size.textContent = label || `${target0.width} × ${target0.height}`;
                return false;
            }
            for (const band of plan.bands) {
                const strip = document.createElement("div");
                strip.className = "ts-outframe__band";
                strip.dataset.side = band.side || "";
                strip.style.left = pctX(band.left);
                strip.style.top = pctY(band.top);
                strip.style.width = pctX(band.width);
                strip.style.height = pctY(band.height);
                // Кружку со стрелкой нужно место. Мерить его в пикселях экрана
                // здесь нечем, да и незачем: узкая полоса узка в любых
                // единицах — считаем по доле от рамки.
                strip.classList.toggle("is-narrow",
                    Math.min(band.width / plan.frame.width,
                             band.height / plan.frame.height) < 0.08);
                const arrow = document.createElement("div");
                arrow.className = "ts-outframe__arrow";
                arrow.textContent = ARROW_BY_SIDE[band.side] || "";
                strip.appendChild(arrow);
                element.insertBefore(strip, box);
            }
            const target = frameResolution(ratio, megapixels);
            size.textContent = label || `${target.width} × ${target.height}`;
            element.classList.add("is-active");
            return plan.bands.length > 0;
        },
        /**
         * Идёт ли сейчас прогон.
         *
         * Рамка — единственное место, где расширение может показать работу до
         * первого превью: закрывать исходник полноэкранной анимацией нельзя,
         * человек смотрит именно на него и на то, куда он вырастет.
         */
        setWorking(on) {
            element.classList.toggle("is-working", Boolean(on));
        },
        /**
         * Пришло первое превью.
         *
         * Штриховка говорила «здесь пусто»; когда сквозь исходник проступает
         * дорисовка, это уже неправда — остаётся только контур будущего кадра.
         */
        setPreviewing(on) {
            element.classList.toggle("is-previewing", Boolean(on));
        },
        hide() {
            element.classList.remove("is-active");
            element.classList.remove("is-working");
            element.classList.remove("is-previewing");
            for (const band of [...element.querySelectorAll(".ts-outframe__band")]) {
                band.remove();
            }
        },
        isActive: () => element.classList.contains("is-active"),
    };
}
