// TS Studio kit — шторка «до и после» (ui-kit layer).
//
// Апскейл — единственная операция, результат которой нельзя оценить, глядя
// только на результат: увеличенная картинка всегда выглядит нормально, пока не
// с чем сравнить. Переключаться между двумя изображениями бесполезно — глаз
// теряет место, на которое смотрел.
//
// Поэтому здесь одна картинка поверх другой и вертикальная шторка: слева то,
// что было, справа то, что стало. Обе подогнаны под один прямоугольник, так что
// точка под курсором — одна и та же точка кадра в обоих вариантах.
//
// Реализация нарочито скучная: два `<img>` в общей коробке, у верхнего
// `clip-path: inset(...)`. Никакого канваса — браузер сам масштабирует и сам
// перерисовывает, а перетаскивание шторки не стоит ни одного кадра пересборки.

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";

const STYLE_ID = "ts-studio-compare-styles";

export function ensureCompareStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-cmp{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
    overflow:hidden;touch-action:none;user-select:none}
.ts-cmp.is-active{display:flex}
.ts-cmp__box{position:relative;max-width:100%;max-height:100%;line-height:0}
.ts-cmp__img{display:block;max-width:100%;max-height:100%;width:auto;height:auto}
/* Верхний слой лежит ровно поверх нижнего и обрезается шторкой. */
.ts-cmp__img--after{position:absolute;inset:0;width:100%;height:100%}
.ts-cmp__handle{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;cursor:ew-resize;
    /* Поверх пользовательской картинки: обязана читаться на любом кадре,
       поэтому не токен темы. */
    background:rgba(255,255,255,.9);box-shadow:0 0 0 1px rgba(0,0,0,.45)}
.ts-cmp__grip{position:absolute;top:50%;left:50%;width:30px;height:30px;transform:translate(-50%,-50%);
    border-radius:50%;background:rgba(255,255,255,.92);box-shadow:0 1px 4px rgba(0,0,0,.45);
    display:flex;align-items:center;justify-content:center;color:#111;font-size:13px}
.ts-cmp__tag{position:absolute;top:8px;padding:2px 7px;border-radius:var(--ts-radius-sm);
    font-size:var(--ts-fs-xs);color:#fff;background:rgba(0,0,0,.55);pointer-events:none}
.ts-cmp__tag--before{left:8px}
.ts-cmp__tag--after{right:8px}
`;
    document.head.appendChild(style);
}

/**
 * @param {object} strings {before, after, hint}
 * @returns {{element: HTMLElement, show: Function, hide: Function, isActive: Function}}
 */
export function createCompare(strings = {}) {
    ensureCompareStyles();
    const element = document.createElement("div");
    element.className = `${TS_UI_CLASS} ts-cmp`;

    const box = document.createElement("div");
    box.className = "ts-cmp__box";
    const before = document.createElement("img");
    before.className = "ts-cmp__img ts-cmp__img--before";
    before.alt = "";
    const after = document.createElement("img");
    after.className = "ts-cmp__img ts-cmp__img--after";
    after.alt = "";
    const handle = document.createElement("div");
    handle.className = "ts-cmp__handle";
    const grip = document.createElement("div");
    grip.className = "ts-cmp__grip";
    grip.textContent = "⇔";
    handle.appendChild(grip);
    const tagBefore = document.createElement("div");
    tagBefore.className = "ts-cmp__tag ts-cmp__tag--before";
    tagBefore.textContent = strings.before || "before";
    const tagAfter = document.createElement("div");
    tagAfter.className = "ts-cmp__tag ts-cmp__tag--after";
    tagAfter.textContent = strings.after || "after";
    box.append(before, after, handle, tagBefore, tagAfter);
    element.appendChild(box);

    let split = 0.5;

    function paint() {
        // Правая часть — результат: обрезаем его слева по шторке.
        after.style.clipPath = `inset(0 0 0 ${(split * 100).toFixed(2)}%)`;
        handle.style.left = `${(split * 100).toFixed(2)}%`;
    }

    function setFromClientX(clientX) {
        const rect = box.getBoundingClientRect();
        if (rect.width <= 0) return;
        split = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
        paint();
    }

    // Тянуть можно за всю картинку, а не только за саму полоску: попадать
    // мышью в два пикселя — работа, которой человек не подписывался.
    let dragging = false;
    element.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        dragging = true;
        try { element.setPointerCapture(event.pointerId); } catch { /* ничего */ }
        setFromClientX(event.clientX);
    });
    element.addEventListener("pointermove", (event) => {
        if (!dragging) return;
        // Кнопка — источник правды, как и у кисти: потерянный pointerup не
        // должен оставлять шторку приклеенной к курсору.
        if ((event.buttons & 1) === 0) { dragging = false; return; }
        setFromClientX(event.clientX);
    });
    const stop = () => { dragging = false; };
    element.addEventListener("pointerup", stop);
    element.addEventListener("pointercancel", stop);
    element.addEventListener("lostpointercapture", stop);
    window.addEventListener("blur", stop);

    return {
        element,
        /**
         * Показать пару. Оба адреса обязаны указывать на одну и ту же сцену —
         * иначе сравнение бессмысленно.
         *
         * @param {string} beforeUrl исходник
         * @param {string} afterUrl результат
         */
        show(beforeUrl, afterUrl) {
            if (!beforeUrl || !afterUrl) return false;
            before.src = beforeUrl;
            after.src = afterUrl;
            split = 0.5;
            paint();
            element.classList.add("is-active");
            return true;
        },
        hide() {
            element.classList.remove("is-active");
            before.removeAttribute("src");
            after.removeAttribute("src");
        },
        isActive: () => element.classList.contains("is-active"),
        teardown: () => window.removeEventListener("blur", stop),
    };
}
