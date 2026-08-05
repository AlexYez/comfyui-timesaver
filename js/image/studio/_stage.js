// Рабочая область студии: то, на что человек смотрит.
//
// Здесь живёт всё, что показывает картинку, и ничего кроме: сам кадр, зум и
// панорама, шторка «до и после», сетка тайлов, подпись под кадром и память о
// том, что было открыто в каждом режиме. Наружу — контракт из десятка методов;
// внутренности (какой элемент чем накрыт, кто кого прячет) не видны никому.
//
// ПОЧЕМУ ОТДЕЛЬНО. Раньше это было россыпью переменных внутри функции на две
// тысячи строк, и каждая новая мелочь требовала помнить весь порядок: показать
// картинку — не забудь снять шторку и сбросить зум; сменить режим — не забудь
// сохранить кадр, иначе он «переедет» в соседнюю вкладку. Обе ошибки в этом
// файле уже случались и обе чинились задним числом. Теперь порядок знает один
// модуль, и забыть его негде.
//
// ПАМЯТЬ ПО РЕЖИМАМ. Картинка принадлежит задаче: исходник апскейла не имеет
// отношения к генерации. `remember(mode)` снимает состояние уходящего режима,
// `restore(mode)` возвращает состояние входящего — вместе со шторкой и с
// путём исходника.

import { TS_UI_CLASS } from "../../_theme.js";
import { createCompare } from "../../_studio/_compare.js";
import { createTileGrid } from "../../_studio/_tilegrid.js";
import { attachZoomPan, clampScale } from "../../_studio/_zoompan.js";

/**
 * @param {object} options
 * @param {HTMLElement} options.host куда встроиться (сцена оболочки)
 * @param {object} options.strings подписи: {empty, fitView, recreate, recreateTip,
 *        compare: {before, after}}
 * @param {(state: object) => void} [options.onRecreate] нажали «Повторить»
 * @param {() => void} [options.onChange] что-то на сцене поменялось — повод
 *        запомнить рабочее место
 * @returns {object} контракт сцены
 */
export function createStage(options) {
    const { host, strings } = options;

    const element = document.createElement("div");
    // Свой скоуп токенов (вложенные безвредны): хром сцены сохраняет цвета,
    // даже если режим когда-нибудь вынесет её из оболочки.
    element.className = `${TS_UI_CLASS} ts-istudio__stagefit`;

    const empty = document.createElement("div");
    empty.className = "ts-istudio__stageempty";
    empty.textContent = strings.empty;

    const image = document.createElement("img");
    image.style.display = "none";
    image.alt = "";

    const caption = document.createElement("div");
    caption.className = "ts-istudio__caption";
    caption.style.display = "none";
    const captionText = document.createElement("span");
    // Кнопка повтора живёт рядом с картинкой, которую описывает, и появляется
    // только когда студия действительно знает, как эта картинка сделана.
    const recreate = document.createElement("button");
    recreate.type = "button";
    recreate.className = "ts-istudio__recreate";
    recreate.textContent = strings.recreate;
    recreate.title = strings.recreateTip;
    recreate.style.display = "none";
    caption.append(captionText, recreate);

    // Всё, что показывает сцена, лежит внутри коробки зума: колесо приближает,
    // средняя кнопка таскает, кнопка в углу возвращает вписанный вид.
    const zoom = document.createElement("div");
    zoom.className = "ts-istudio__zoom";
    zoom.append(image);

    const compare = createCompare({ before: strings.compare.before,
                                    after: strings.compare.after });
    zoom.appendChild(compare.element);

    const tiles = createTileGrid();
    zoom.appendChild(tiles.element);

    element.append(empty, zoom);

    const fitButton = document.createElement("button");
    fitButton.type = "button";
    fitButton.className = "ts-istudio__fit";
    fitButton.textContent = "⤢";
    fitButton.title = strings.fitView;
    element.appendChild(fitButton);

    host.append(element, caption);

    const view = { scale: 1, x: 0, y: 0 };
    /** Что сравнивает шторка сейчас — чтобы вернуть её вместе со сценой. */
    let comparePair = null;
    /** Путь исходника для апскейла: перетащенная картинка старше выбора в ленте. */
    let source = "";
    /** Снимки сцены по режимам: кадр, исходник, подпись, пара шторки. */
    const byMode = new Map();

    function paintView() {
        zoom.style.transform =
            `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
        fitButton.classList.toggle("is-active",
            view.scale !== 1 || view.x !== 0 || view.y !== 0);
    }

    function fit() {
        view.scale = 1;
        view.x = 0;
        view.y = 0;
        paintView();
    }

    attachZoomPan(element, {
        zoomAt(clientX, clientY, factor) {
            const rect = element.getBoundingClientRect();
            const x = clientX - rect.left;
            const y = clientY - rect.top;
            // Вписанный вид — это масштаб 1: картинка уже подогнана правилами
            // CSS. Поэтому нижняя граница здесь единица, а не доля от неё.
            const next = clampScale(view.scale * factor, 1);
            if (next === view.scale) return;
            view.x = x - ((x - view.x) / view.scale) * next;
            view.y = y - ((y - view.y) / view.scale) * next;
            view.scale = next;
            paintView();
        },
        panBy(dx, dy) {
            view.x += dx;
            view.y += dy;
            paintView();
        },
        reset: fit,
    });
    fitButton.addEventListener("click", fit);

    function setCaption(text, state) {
        captionText.textContent = text || "";
        recreate.style.display = state ? "" : "none";
        recreate.onclick = state ? () => options.onRecreate?.(state) : null;
        caption.style.display = (text || state) ? "" : "none";
    }

    /** Показать картинку. Снимает шторку и возвращает вписанный вид. */
    function show(url, { caption: text = "", state = null, keepSource = false } = {}) {
        hideCompare();
        fit();
        if (!keepSource) source = "";
        image.src = url;
        image.style.display = "";
        empty.style.display = "none";
        setCaption(text, state);
        options.onChange?.();
    }

    function hideCompare() {
        compare.hide();
        comparePair = null;
    }

    return {
        element,
        /** Сетка тайлов и шторка отданы наружу целиком — у них свои контракты. */
        tiles,

        show,
        /** Промежуточный кадр: без сброса зума и подписи — это не результат. */
        showBlob(blob) {
            image.src = URL.createObjectURL(blob);
            image.style.display = "";
            empty.style.display = "none";
        },
        /** Пустая сцена: ни картинки, ни шторки, ни подписи. */
        clear() {
            hideCompare();
            image.removeAttribute("src");
            image.style.display = "none";
            empty.style.display = "";
            setCaption("", null);
            fit();
        },
        setCaption,
        fit,

        /** Показать пару «до и после». Картинка уступает ей место. */
        showCompare(before, after) {
            if (!compare.show(before, after)) return false;
            comparePair = { before, after };
            image.style.display = "none";
            return true;
        },
        hideCompare,
        isComparing: () => compare.isActive(),

        /** Есть ли что показывать прямо сейчас. */
        hasImage: () => image.style.display !== "none" && Boolean(image.src),
        /** Адрес того, что на экране, или пустая строка. */
        url: () => (image.style.display !== "none" ? image.src || "" : ""),
        /** Настоящий размер кадра — по нему считается сетка тайлов. */
        naturalSize: () => ({ width: image.naturalWidth || 0,
                              height: image.naturalHeight || 0 }),
        /** Прямоугольник картинки и коробки — для наложения сетки. */
        imageRect: () => image.getBoundingClientRect(),
        hostRect: () => zoom.getBoundingClientRect(),

        /** Путь исходника, с которым пойдёт апскейл. */
        source: () => source,
        setSource(annotated) {
            source = annotated || "";
        },

        /**
         * Снять состояние уходящего режима.
         *
         * Без этого картинка «переезжает» за человеком по вкладкам: исходник
         * апскейла оказывается в генерации, где ему делать нечего.
         */
        remember(mode) {
            if (!mode) return;
            byMode.set(mode, {
                src: image.style.display === "none" ? "" : image.src || "",
                source,
                caption: captionText.textContent || "",
                compare: compare.isActive() ? comparePair : null,
            });
        },

        /** Вернуть состояние входящего режима. */
        restore(mode) {
            const kept = byMode.get(mode) || null;
            source = kept?.source || "";
            hideCompare();
            if (kept?.compare?.before && kept?.compare?.after) {
                compare.show(kept.compare.before, kept.compare.after);
                comparePair = kept.compare;
                image.style.display = "none";
                empty.style.display = "none";
                return;
            }
            if (kept?.src) {
                image.src = kept.src;
                image.style.display = "";
                empty.style.display = "none";
            } else {
                image.removeAttribute("src");
                image.style.display = "none";
                empty.style.display = "";
            }
            setCaption(kept?.caption || "", null);
            fit();
        },

        /** Спрятать сцену целиком — инпэйнт рисует на своём холсте. */
        setVisible(visible) {
            element.style.display = visible ? "" : "none";
            caption.style.display = visible && captionText.textContent ? "" : "none";
        },

        teardown() {
            compare.teardown?.();
            tiles.hide();
            element.remove();
            caption.remove();
        },
    };
}
