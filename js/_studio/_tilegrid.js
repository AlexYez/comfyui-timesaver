// TS Studio kit — как выглядит апскейл, пока он идёт (ui-kit layer).
//
// Увеличение идёт тайлами: VAE кодирует и декодирует картинку кусками по 512
// пикселей с перехлёстом. Быстрый декодер латента честно отдавал эти куски по
// одному, и на экране мелькали обрывки в случайных местах — по ним нельзя было
// понять ни сколько осталось, ни что вообще происходит.
//
// Здесь то же самое показано так, как оно устроено: поверх картинки лежит сетка
// из ровно того числа клеток, сколько тайлов реально считает ComfyUI (число
// приходит в событии прогресса — тайловый VAE отчитывается за каждый), и клетки
// гаснут по мере готовности. Видно и общий ход, и то, что работа идёт кусками.
//
// Сетка — DOM, а не канвас: клеток десятки, анимация переходов достаётся от
// браузера даром, и ни один кадр не стоит перерисовки.

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";

const STYLE_ID = "ts-studio-tilegrid-styles";

export function ensureTileGridStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-tiles{position:absolute;inset:0;display:none;pointer-events:none;z-index:4}
.ts-tiles.is-active{display:grid}
.ts-tiles__cell{position:relative;border:1px solid var(--ts-accent-line);
    background:var(--ts-scrim);opacity:1;transition:opacity .45s ease,background .45s ease}
/* Готовый тайл гаснет и открывает картинку под собой — заполнение читается как
   проявление, а не как мигание. */
.ts-tiles__cell.is-done{opacity:0;background:transparent;border-color:transparent}
/* Тайл, который считается прямо сейчас: тонкая пульсация, чтобы взгляд знал,
   куда смотреть. */
.ts-tiles__cell.is-live{background:var(--ts-accent-soft);
    animation:ts-tiles-pulse 1.1s ease-in-out infinite}
@keyframes ts-tiles-pulse{0%,100%{opacity:.85}50%{opacity:.45}}
`;
    document.head.appendChild(style);
}

/**
 * Разложить N тайлов на строки и столбцы под форму картинки.
 *
 * Настоящая раскладка ComfyUI зависит от размера тайла и перехлёста, а наружу
 * приходит только их количество. Поэтому подбирается решётка, чья пропорция
 * ближе всего к пропорции кадра: при 12 тайлах и широкой картинке это 4x3, а не
 * 6x2 — так клетки ложатся туда же, где считаются на самом деле.
 *
 * @param {number} count сколько тайлов
 * @param {number} aspect ширина/высота картинки
 * @returns {{cols: number, rows: number}}
 */
export function tileLayout(count, aspect) {
    const total = Math.max(1, Math.round(count));
    const ratio = aspect > 0 ? aspect : 1;
    let best = { cols: total, rows: 1 };
    let bestError = Infinity;
    for (let cols = 1; cols <= total; cols += 1) {
        const rows = Math.ceil(total / cols);
        // Насколько решётка похожа на кадр — и насколько мало в ней пустых мест.
        const error = Math.abs(Math.log((cols / rows) / ratio)) + (cols * rows - total) * 0.05;
        if (error < bestError) {
            bestError = error;
            best = { cols, rows };
        }
    }
    return best;
}

/**
 * @returns {{element, show, advance, hide, isActive}}
 */
export function createTileGrid() {
    ensureTileGridStyles();
    const element = document.createElement("div");
    element.className = `${TS_UI_CLASS} ts-tiles`;

    let cells = [];
    let total = 0;

    function build(count, aspect) {
        const { cols, rows } = tileLayout(count, aspect);
        element.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        element.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
        element.replaceChildren();
        cells = [];
        for (let i = 0; i < cols * rows; i += 1) {
            const cell = document.createElement("div");
            cell.className = "ts-tiles__cell";
            element.appendChild(cell);
            cells.push(cell);
        }
        total = count;
    }

    return {
        element,
        /**
         * Показать сетку под данное число тайлов.
         *
         * @param {number} count сколько тайлов считает движок
         * @param {DOMRect|{width:number,height:number}} area прямоугольник картинки
         */
        show(count, area, host) {
            if (!(count > 1) || !area || !(area.height > 0)) return false;
            build(count, area.width / area.height);
            // Сетка ложится РОВНО на картинку, а не на всю сцену: тайлы
            // считаются по кадру, и клетки, висящие на пустом фоне, врали бы о
            // том, где идёт работа.
            if (host) {
                element.style.left = `${area.left - host.left}px`;
                element.style.top = `${area.top - host.top}px`;
                element.style.width = `${area.width}px`;
                element.style.height = `${area.height}px`;
                element.style.right = "auto";
                element.style.bottom = "auto";
            }
            element.classList.add("is-active");
            return true;
        },
        /** Сколько тайлов готово. */
        advance(done) {
            const ready = Math.max(0, Math.min(total, Math.round(done)));
            cells.forEach((cell, index) => {
                cell.classList.toggle("is-done", index < ready);
                cell.classList.toggle("is-live", index === ready && ready < total);
            });
        },
        hide() {
            element.classList.remove("is-active");
            element.style.cssText = "";
            element.replaceChildren();
            cells = [];
            total = 0;
        },
        isActive: () => element.classList.contains("is-active"),
    };
}
