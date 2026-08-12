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
import { createOutpaintFrame } from "../../_studio/_outframe.js";
import { attachZoomPan, clampScale } from "../../_studio/_zoompan.js";
import { createWarmup } from "../../_studio/_warmup.js";

/**
 * @param {object} options
 * @param {HTMLElement} options.host куда встроиться (сцена оболочки)
 * @param {object} options.strings подписи: {empty, fitView, recreate, recreateTip,
 *        compare: {before, after}}
 * @param {(state: object) => void} [options.onRecreate] нажали «Повторить»
 * @param {() => void} [options.onChange] что-то на сцене поменялось — повод
 *        запомнить рабочее место
 * @param {() => void} [options.onLayout] кадр вписан заново (картинка
 *        загрузилась, область изменила размер) — накладки пора пересчитать
 * @returns {object} контракт сцены
 */
export function createStage(options) {
    const { host, strings } = options;

    const element = document.createElement("div");
    // Свой скоуп токенов (вложенные безвредны): хром сцены сохраняет цвета,
    // даже если режим когда-нибудь вынесет её из оболочки.
    element.className = `${TS_UI_CLASS} ts-istudio__stagefit`;

    // ⚠️ Заглушка лежит АБСОЛЮТНО, а не соседом коробки зума. Соседом она и
    // была: коробка занимает всю ширину и высоту, поэтому flex прижимал текст
    // к левому краю — «по центру» получалось по центру пустого остатка.
    const empty = document.createElement("div");
    empty.className = "ts-istudio__stageempty";
    const emptyTitle = document.createElement("div");
    emptyTitle.className = "ts-istudio__emptytitle";
    emptyTitle.textContent = strings.empty;
    const emptyHint = document.createElement("div");
    emptyHint.className = "ts-istudio__emptyhint";
    emptyHint.textContent = strings.emptyHint || "";
    empty.append(emptyTitle, emptyHint);

    // ── укладка кадра и накладок ─────────────────────────────────────────── //
    //
    // Две вложенные коробки, и это главное решение файла:
    //
    //   fitbox — то, подо что отводится место. Обычно это сам кадр; в
    //            расширении — БУДУЩИЙ кадр целиком, поэтому картинка внутри
    //            ужимается и новые области видно, а не «где-то сбоку».
    //   frame  — сама картинка по своей пропорции, вписанная в fitbox.
    //
    // Накладки (сетка тайлов, рамка расширения) лежат ВНУТРИ этих коробок и
    // растягиваются по ним (`inset:0`, доли в процентах). Ни одна из них не
    // считает экранных прямоугольников — и поэтому не может разъехаться:
    // зум и панорама двигают всю коробку целиком, вместе с накладками.
    //
    // ⚠️ Так было не всегда. Сетка и рамка позиционировались по
    // `getBoundingClientRect()`, а результат клался внутрь того же
    // трансформированного слоя — на любом зуме, кроме единицы, накладка
    // получала двойной масштаб. Возвращать тот способ нельзя.
    const fitbox = document.createElement("div");
    fitbox.className = "ts-istudio__fitbox";
    fitbox.style.display = "none";

    // Подложка — то, что рисуется ПОД исходником. В расширении кадра превью
    // приходит целиком, вместе с уже готовой серединой; накрыв её исходником,
    // мы показываем ровно то, что человек ждёт: как заполняются новые области.
    const underlay = document.createElement("img");
    underlay.className = "ts-istudio__under";
    underlay.alt = "";
    fitbox.appendChild(underlay);

    const frame = document.createElement("div");
    frame.className = "ts-istudio__frame";
    fitbox.appendChild(frame);

    const image = document.createElement("img");
    image.alt = "";
    frame.appendChild(image);

    // ⚠️ ОДНА точка смены картинки — она же отзывает предыдущий blob-URL.
    // Раньше кадры латентного превью и загруженные исходники держали по blob
    // на каждый показ: за прогон их десятки, и жили они до закрытия вкладки.
    // Адрес приходит и снаружи (`show` получает готовый URL из _app.js),
    // поэтому владение забирает область: она последняя, кто его показывает.
    let ownedBlobUrl = "";

    function setImageSrc(next) {
        if (ownedBlobUrl && ownedBlobUrl !== next) {
            URL.revokeObjectURL(ownedBlobUrl);
            ownedBlobUrl = "";
        }
        if (typeof next === "string" && next.startsWith("blob:")) ownedBlobUrl = next;
        image.src = next;
    }

    function clearImageSrc() {
        if (ownedBlobUrl) {
            URL.revokeObjectURL(ownedBlobUrl);
            ownedBlobUrl = "";
        }
        image.removeAttribute("src");
    }
    // Пропорция приходит из самой картинки — это её собственные данные, а не
    // замер экрана, поэтому правило «никакой JS-геометрии» не нарушено.
    image.addEventListener("load", () => layout());

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
    zoom.append(fitbox);

    const compare = createCompare({ before: strings.compare.before,
                                    after: strings.compare.after });
    zoom.appendChild(compare.element);

    // Сетка тайлов лежит НА КАДРЕ — внутри него, растянутая по нему. Клетка
    // обещает место, где пойдёт работа, и обещание обязано быть верным на
    // любом зуме.
    const tiles = createTileGrid();
    frame.appendChild(tiles.element);

    // Рамка будущего кадра — для расширения: показывает, куда именно
    // дорисуется картинка, ещё до запуска. Её место — коробка отведённого
    // места: в расширении та больше картинки ровно на то, что дорисуется.
    const outframe = createOutpaintFrame();
    fitbox.appendChild(outframe.element);

    // Ожидание — поверх области и под кнопкой «вписать»: пока считается модель,
    // показывать нечего, и пустой экран читается как «ничего не происходит».
    const warmup = createWarmup({
        labels: strings.stages || {},
        title: strings.warmupTitle || "",
        note: strings.warmupNote || "",
    });

    element.append(empty, zoom);

    const fitButton = document.createElement("button");
    fitButton.type = "button";
    fitButton.className = "ts-istudio__fit";
    fitButton.textContent = "⤢";
    fitButton.title = strings.fitView;
    element.appendChild(fitButton);

    host.append(element, caption, warmup.element);

    /**
     * Куда положить показ ожидания.
     *
     * ⚠️ На САМ КАДР, когда он есть: анимация обязана занимать картинку
     * целиком, а не висеть коробкой посреди области. Кадра нет — показ берёт
     * всю область. Сцена спрятана (инпэйнт рисует на своём холсте) — показ
     * уходит на контейнер, иначе исчез бы вместе со сценой.
     */
    function mountWarmup() {
        const hidden = element.style.display === "none";
        const target = hidden ? host
            : (fitbox.style.display === "none" ? element : frame);
        if (warmup.element.parentElement !== target) target.appendChild(warmup.element);
    }

    // Область меняет размер вместе с окном и панелью ассетов — вписывание
    // обязано пересчитаться, иначе кадр останется от прошлой ширины.
    const resizeWatch = typeof ResizeObserver === "function"
        ? new ResizeObserver(() => layout()) : null;
    resizeWatch?.observe(element);

    const view = { scale: 1, x: 0, y: 0 };
    /** Что сравнивает шторка сейчас — чтобы вернуть её вместе со сценой. */
    let comparePair = null;
    /** Путь исходника для апскейла: перетащенная картинка старше выбора в ленте. */
    let source = "";
    /** Снимки сцены по режимам: кадр, исходник, подпись, пара шторки. */
    const byMode = new Map();
    /**
     * Готовый ли кадр сейчас на сцене.
     *
     * Промежуточное превью — это не картинка, а состояние работы: сохранять,
     * отправлять в другой раздел и вообще предлагать что-либо с ним нельзя.
     * Отсюда отдельный признак, а не «на сцене что-то есть».
     */
    let final = false;

    /** Пропорция, под которую отведено место (0 — под саму картинку). */
    let reserved = 0;
    // Пропорция последней показанной картинки — догадка на время загрузки.
    let lastRatio = 0;
    /** Идёт ли пересчёт прямо сейчас — защита от захода в самого себя. */
    let laying = false;
    /** Пришла просьба пересчитать, пока считали: повторим, закончив. */
    let pending = false;
    /** Что было применено в прошлый раз: по нему видно, изменилось ли что-то. */
    let applied = "";

    /**
     * Вписать обе коробки: место — в область, картинку — в место.
     *
     * ⚠️ Считаем САМИ, потому что CSS этого не умеет. `aspect-ratio` вместе с
     * `max-height` не пережимает ширину: браузер оставляет ширину и обрезает
     * высоту, то есть РАСТЯГИВАЕТ содержимое. Замерено на живой студии:
     * картинка 1344×736 в кадре 21:9 показывалась как 1232×528.
     *
     * Меряем `clientWidth/clientHeight` — это раскладочные пиксели, они НЕ
     * зависят от трансформа. Поэтому зум и панорама на расчёт не влияют, и
     * запрет из §12.5.3 (никакой геометрии по экранным прямоугольникам)
     * соблюдён: `getBoundingClientRect` здесь не используется.
     */
    function layout() {
        // ⚠️ Заход в самого себя. Слушатель раскладки пересчитывает рамку
        // расширения, та отводит место (`reserveRatio`), а это снова раскладка:
        // получалась бесконечная рекурсия, и студия падала на открытии
        // (`Maximum call stack size exceeded`). Здесь — единственная точка, где
        // цикл можно разорвать.
        //
        // ⚠️⚠️ Но просто ВЫБРОСИТЬ вложенный вызов нельзя, а именно так тут и
        // было. Расширение кадра приходит по этой самой дороге: раскладка →
        // слушатель → `reserveRatio(16/9)` → раскладка. Вложенная терялась, и
        // отведённое место оставалось от старой пропорции: на кадре 9:16
        // полосы дорисовки считались по коробке размером с саму картинку и
        // ложились ПОВЕРХ неё — те самые серые прямоугольники.
        //
        // Поэтому вложенный вызов не выбрасывается, а откладывается: внешний
        // проход, закончив, повторяет счёт. Круг ограничен — пропорция
        // сходится за один-два шага, а больше и не бывает: `applyLayout`
        // молчит, когда геометрия не изменилась.
        if (laying) { pending = true; return; }
        const boxW = element.clientWidth;
        const boxH = element.clientHeight;
        if (!(boxW > 0) || !(boxH > 0)) return;
        laying = true;
        try {
            applyLayout(boxW, boxH);
            for (let round = 0; pending && round < 3; round += 1) {
                pending = false;
                applyLayout(element.clientWidth, element.clientHeight);
            }
        } finally {
            pending = false;
            laying = false;
        }
    }

    function applyLayout(boxW, boxH) {
        // ⚠️ Пока картинка грузится, её размера ещё нет. Брать в этот миг
        // 16:9 нельзя: вертикальный кадр на секунду становится лежачим и
        // сплющенным. Прошлая пропорция — куда более честная догадка: версии
        // одного кадра и результат апскейла её сохраняют.
        if (image.naturalWidth > 0 && image.naturalHeight > 0) {
            lastRatio = image.naturalWidth / image.naturalHeight;
        }
        const imageRatio = (image.naturalWidth > 0 && image.naturalHeight > 0)
            ? image.naturalWidth / image.naturalHeight
            : (lastRatio || 16 / 9);
        const fitRatio = reserved > 0 ? reserved : imageRatio;

        let placeW = boxW;
        let placeH = boxW / fitRatio;
        if (placeH > boxH) { placeH = boxH; placeW = boxH * fitRatio; }
        // Целые пиксели: дробная ширина оставляет между кадром и накладкой
        // полупрозрачный шов в полпикселя — он и читается серой ниткой.
        placeW = Math.round(placeW);
        placeH = Math.round(placeH);
        fitbox.style.width = `${placeW}px`;
        fitbox.style.height = `${placeH}px`;

        let frameW = placeW;
        let frameH = placeW / imageRatio;
        if (frameH > placeH) { frameH = placeH; frameW = placeH * imageRatio; }
        frame.style.width = `${Math.round(frameW)}px`;
        frame.style.height = `${Math.round(frameH)}px`;
        // Сообщаем только о НАСТОЯЩЕМ изменении: одинаковая геометрия дважды
        // подряд — это не событие, а повод для лишнего круга работы.
        const now = `${placeW}x${placeH}/${Math.round(frameW)}x${Math.round(frameH)}`;
        if (now === applied) return;
        applied = now;
        options.onLayout?.();
    }

    /**
     * Под какую пропорцию отвести место на сцене.
     *
     * Обычно это сама картинка. В расширении — будущий кадр: тогда картинка
     * ужимается, и новые области попадают в поле зрения целиком.
     *
     * @param {number} ratio ширина/высота, или 0 — вернуть пропорцию картинки
     */
    function reserveRatio(ratio) {
        const next = ratio > 0 ? ratio : 0;
        if (next === reserved) return;   // ничего не изменилось — и считать нечего
        reserved = next;
        layout();
    }

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

    /**
     * Показать картинку. Снимает шторку и по умолчанию возвращает вписанный вид.
     *
     * ⚠️ `keepView` — для листания версий одного кадра. Человек приблизил
     * колесом кусок, чтобы разглядеть детали, и жмёт «предыдущая версия»
     * ровно затем, чтобы сравнить ЭТОТ кусок. Сброс масштаба на каждом шаге
     * означает, что сравнить ничего нельзя.
     */
    function show(url, { caption: text = "", state = null, keepSource = false,
                          keepView = false } = {}) {
        hideCompare();
        outframe.hide();
        reserveRatio(0);
        // Результат пришёл — подложке под ним делать нечего.
        underlay.classList.remove("is-active");
        if (underlay.dataset.url) URL.revokeObjectURL(underlay.dataset.url);
        delete underlay.dataset.url;
        underlay.removeAttribute("src");
        warmup.hide();
        if (!keepView) fit();
        if (!keepSource) source = "";
        final = true;
        setImageSrc(url);
        fitbox.style.display = "";
        empty.style.display = "none";
        layout();
        setCaption(text, state);
        options.onChange?.();
    }

    function hideCompare() {
        compare.hide();
        comparePair = null;
    }

    return {
        element,
        /** Сетка тайлов и рамка кадра отданы наружу — у них свои контракты. */
        tiles,
        outframe,

        show,
        /**
         * Превью ПОД исходником: видно только то, что дорисовывается.
         *
         * @param {Blob} blob кадр от движка
         */
        showUnderlay(blob) {
            warmup.hide();
            if (underlay.dataset.url) URL.revokeObjectURL(underlay.dataset.url);
            const url = URL.createObjectURL(blob);
            underlay.dataset.url = url;
            underlay.src = url;
            underlay.classList.add("is-active");
        },
        /** Убрать подложку — результат пришёл или прогон кончился. */
        clearUnderlay() {
            underlay.classList.remove("is-active");
            if (underlay.dataset.url) URL.revokeObjectURL(underlay.dataset.url);
            delete underlay.dataset.url;
            underlay.removeAttribute("src");
        },
        /** Промежуточный кадр: без сброса зума и подписи — это не результат. */
        showBlob(blob) {
            // Первое превью и есть повод убрать ожидание: дальше человек
            // смотрит на картинку, а не на анимацию.
            warmup.hide();
            final = false;
            setImageSrc(URL.createObjectURL(blob));
            fitbox.style.display = "";
            empty.style.display = "none";
            layout();
        },

        /**
         * Анимация ожидания — её включает и гасит прогон (см. _app.js).
         *
         * Перед показом она переезжает туда, где сейчас работа: на кадр, на всю
         * область или на контейнер.
         */
        warmup: {
            ...warmup,
            show(options) {
                mountWarmup();
                // ⚠️ Заставляем браузер посчитать состояние ДО показа. Элемент
                // только что переехал на другого родителя, и если включить его
                // в том же кадре, перехода не будет вовсе: начинать не с чего.
                // Замерено — прозрачность прыгала в единицу за 30 мс.
                void warmup.element.offsetWidth;
                return warmup.show(options);
            },
        },
        /** Пустая сцена: ни картинки, ни шторки, ни подписи. */
        clear() {
            hideCompare();
            warmup.hide();
            final = false;
            reserveRatio(0);
            underlay.classList.remove("is-active");
            underlay.removeAttribute("src");
            clearImageSrc();
            fitbox.style.display = "none";
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
            warmup.hide();
            fitbox.style.display = "none";
            return true;
        },
        hideCompare,
        isComparing: () => compare.isActive(),

        /** Есть ли что показывать прямо сейчас. */
        hasImage: () => compare.isActive()
            || (fitbox.style.display !== "none" && Boolean(image.src)),
        /**
         * Адрес того, что на экране, или пустая строка.
         *
         * ⚠️ Шторка «до и после» ПРЯЧЕТ кадр и показывает пару. Пока сцена
         * отвечала по спрятанному кадру, после апскейла выходило «картинки
         * нет»: в меню оставался один выключенный пункт. На экране в этот
         * момент результат — его и отдаём.
         */
        url: () => {
            if (compare.isActive()) return comparePair?.after || "";
            return fitbox.style.display !== "none" ? image.src || "" : "";
        },
        /** Готовый результат, а не промежуточное превью прогона. */
        isFinal: () => (compare.isActive()
            ? Boolean(comparePair?.after)
            : final && fitbox.style.display !== "none"),
        /** Настоящий размер кадра — по нему считается сетка тайлов. */
        naturalSize: () => ({ width: image.naturalWidth || 0,
                              height: image.naturalHeight || 0 }),
        /**
         * Отвести место под пропорцию будущего кадра (расширение) или вернуть
         * его картинке (0).
         *
         * ⚠️ Замеров экрана сцена наружу не отдаёт СОЗНАТЕЛЬНО. Накладки,
         * посчитанные по `getBoundingClientRect()`, разъезжались на любом
         * зуме: прямоугольник уже с трансформом, а результат кладётся внутрь
         * того же трансформированного слоя.
         */
        reserveRatio,

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
                src: fitbox.style.display === "none" ? "" : image.src || "",
                final,
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
            warmup.hide();
            if (kept?.compare?.before && kept?.compare?.after) {
                compare.show(kept.compare.before, kept.compare.after);
                comparePair = kept.compare;
                fitbox.style.display = "none";
                empty.style.display = "none";
                return;
            }
            // Снимок без пометки — из старой сессии: там лежала готовая
            // картинка, и считать её «незавершённой» нельзя. Незавершённым
            // кадр бывает только между превью и концом прогона.
            final = kept?.final === undefined ? Boolean(kept?.src) : Boolean(kept.final);
            reserveRatio(0);
            if (kept?.src) {
                setImageSrc(kept.src);
                fitbox.style.display = "";
                empty.style.display = "none";
                layout();
            } else {
                clearImageSrc();
                fitbox.style.display = "none";
                empty.style.display = "";
            }
            setCaption(kept?.caption || "", null);
            fit();
        },

        /** Видна ли сцена сейчас — по ней решают, показывать ли ожидание. */
        isVisible: () => element.style.display !== "none",

        /** Спрятать сцену целиком — инпэйнт рисует на своём холсте. */
        setVisible(visible) {
            element.style.display = visible ? "" : "none";
            caption.style.display = visible && captionText.textContent ? "" : "none";
        },

        teardown() {
            resizeWatch?.disconnect();
            compare.teardown?.();
            tiles.hide();
            outframe.hide();
            // Последний показанный кадр держал blob — отпускаем вместе со сценой.
            clearImageSrc();
            if (underlay.dataset.url) {
                URL.revokeObjectURL(underlay.dataset.url);
                delete underlay.dataset.url;
            }
            element.remove();
            caption.remove();
        },
    };
}
