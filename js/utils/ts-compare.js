// TS Compare — шторка «до и после» прямо в ноде.
//
// Две ветки, потому что материал разный: пара одиночных кадров приходит двумя
// PNG (сравнивают детали, и сжатие уничтожило бы ровно их), а пачка — одним
// файлом, где A лежит над B. Почему именно так, а не два плеера, — в шапке
// `_compare_video.js` и в самой ноде.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import { TS_UI_CLASS, ensureThemeStyles, pickLocaleStrings } from "../_theme.js";
import { addResizableDomWidget } from "../_dom_widget.js";
import { openFullscreenOverlay } from "../_fullscreen.js";
import { createCompare } from "../_studio/_compare.js";
import { createVideoCompare } from "./_compare_video.js";

const EXTENSION_ID = "ts.compare";
const NODE_TYPE = "TS_Compare";
const UI_KEY = "ts_compare";
const DOM_WIDGET = "ts_compare_view";
const PROP_PAYLOAD = "ts_compare_payload";
const STYLE_ID = "ts-compare-styles";

const STRINGS = {
    en: {
        idle: "Run the graph to compare",
        play: "Play",
        pause: "Pause",
        playHint: "Play the assembled clip — it never starts on its own",
        seekHint: "Position in the clip",
        before: "A",
        after: "B",
        fullscreen: "Fullscreen (Esc to leave)",
        working: "Assembling the comparison…",
    },
    ru: {
        idle: "Запустите граф, чтобы сравнить",
        play: "Играть",
        pause: "Пауза",
        playHint: "Проиграть собранный ролик — сам он не запускается",
        seekHint: "Место в ролике",
        before: "A",
        after: "B",
        fullscreen: "Во весь экран (выход — Esc)",
        working: "Собираю сравнение…",
    },
};

function ensureStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-compare{display:flex;flex-direction:column;height:100%;min-height:0}
.ts-compare__slot{flex:1 1 auto;min-height:0;display:flex}
.ts-compare__slot > *{flex:1 1 auto;min-width:0;min-height:0}
.ts-compare__idle{
  flex:1 1 auto;display:flex;align-items:center;justify-content:center;
  text-align:center;padding:16px;font-size:var(--ts-fs-sm);color:var(--ts-muted);
  border:1px dashed var(--ts-border);border-radius:var(--ts-radius)}

/* Полоса сборки. Живёт на уровне НОДЫ, а не внутри плеера: на первом прогоне
   плеера ещё нет, а знать, что работа идёт, надо именно тогда. */
.ts-compare__progress{
  flex:0 0 auto;height:3px;margin-top:6px;border-radius:2px;
  background:var(--ts-border-soft);overflow:hidden;
  opacity:0;transition:opacity .15s ease}
.ts-compare__progress.is-busy{opacity:1}
.ts-compare__fill{
  height:100%;width:0%;border-radius:2px;background:var(--ts-accent);
  transition:width .2s linear}

/* ⚠️ Неопределённый режим. ComfyUI сообщает про эту ноду только «работает» и
   «готово» (value 0 / max 1), без долей — замерено на сокете. Показывать в
   таком случае ноль процентов честнее всего движением: человеку нужно знать,
   что работа идёт, а не сколько именно осталось. */
.ts-compare__progress.is-endless .ts-compare__fill{
  width:35%;transition:none;animation:ts-compare-sweep 1.1s ease-in-out infinite}
@keyframes ts-compare-sweep{
  0%{margin-left:-35%}
  100%{margin-left:100%}
}
@media (prefers-reduced-motion:reduce){
  .ts-compare__progress.is-endless .ts-compare__fill{animation:none;width:100%;opacity:.5}
}
`;
    document.head.appendChild(style);
}

function viewUrl(payload, filename) {
    const params = new URLSearchParams({
        filename: String(filename || ""),
        subfolder: String(payload.subfolder || ""),
        type: String(payload.type || "temp"),
        // ⚠️ Кэш браузера иначе показывает прошлый прогон: имя временного файла
        // новое, но при повторе с тем же именем сравнение бы «не обновилось».
        rand: String(Math.random()),
    });
    return api.apiURL(`/view?${params}`);
}

function setupCompare(node) {
    if (node.__tsCompare) return;
    ensureStyles();
    const t = pickLocaleStrings(STRINGS);

    const root = document.createElement("div");
    root.className = `${TS_UI_CLASS} ts-compare`;

    const slot = document.createElement("div");
    slot.className = "ts-compare__slot";

    const idle = document.createElement("div");
    idle.className = "ts-compare__idle";
    idle.textContent = t.idle;
    slot.appendChild(idle);

    const progress = document.createElement("div");
    progress.className = "ts-compare__progress";
    const fill = document.createElement("div");
    fill.className = "ts-compare__fill";
    progress.appendChild(fill);

    root.append(slot, progress);

    /**
     * Ход сборки. `null` — убрать полосу, число 0..1 — доля, `true` — работа
     * идёт, но долей нам не сообщили.
     */
    const setProgress = (value) => {
        if (value === null) {
            progress.classList.remove("is-busy", "is-endless");
            return;
        }
        progress.classList.add("is-busy");
        if (value === true) {
            progress.classList.add("is-endless");
            fill.style.width = "";
            return;
        }
        progress.classList.remove("is-endless");
        fill.style.width = `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
    };

    // Обе шторки создаются лениво: за прогон нужна ровно одна из них, а вторая
    // так и осталась бы висеть с наблюдателями и, для видео, с декодером.
    let stills = null;
    let clip = null;

    const clear = () => {
        slot.textContent = "";
    };

    const apply = (payload) => {
        if (!payload) return;
        node.properties ||= {};
        node.properties[PROP_PAYLOAD] = payload;

        if (payload.mode === "image") {
            clip?.teardown();
            clip = null;
            if (!stills) stills = createCompare({ before: t.before, after: t.after });
            clear();
            slot.appendChild(stills.element);
            stills.show(
                viewUrl(payload, payload.filename_a),
                viewUrl(payload, payload.filename_b),
            );
            return;
        }

        if (!clip) {
            clip = createVideoCompare(t);
            clip.expandButton.addEventListener("click", () => openFullscreen());
        }
        clear();
        slot.appendChild(clip.element);
        clip.show(viewUrl(payload, payload.filename), {
            labelA: payload.label_a,
            labelB: payload.label_b,
        });
        clip.relayout();
    };

    /**
     * Во весь экран.
     *
     * ⚠️ Элемент ПЕРЕЕЗЖАЕТ в оверлей и возвращается обратно, а не копируется:
     * копия означала бы второй `<video>`, то есть второй декодер и ту самую
     * борьбу за видеокарту, ради которой всё сравнение и сложено в один файл.
     */
    function openFullscreen() {
        const current = clip;
        if (!current) return;
        const host = document.createElement("div");
        // Хост занимает весь оверлей, а шторка внутри — весь хост: иначе кадр
        // остаётся размером с ноду и висит в углу пустого экрана.
        host.style.cssText = "flex:1 1 auto;display:flex;width:100%;height:100%;"
            + "min-width:0;min-height:0;padding:24px;box-sizing:border-box";
        // ⚠️ Без этого шторка занимает ширину СОДЕРЖИМОГО, а не экрана: замерено
        // — сцена выходила 305 px при экране 1600. В ноде её растягивает
        // правило слота, а в оверлее такого правила нет.
        current.element.style.flex = "1 1 auto";
        current.element.style.width = "100%";
        current.element.style.minWidth = "0";
        host.appendChild(current.element);
        openFullscreenOverlay(host, {
            label: t.fullscreen,
            closeTitle: t.fullscreen,
            onOpen: () => current.relayout(),
            onClose: () => {
                // Возвращаем как было: в ноде размер задаёт слот.
                current.element.style.flex = "";
                current.element.style.width = "";
                current.element.style.minWidth = "";
                slot.appendChild(current.element);
                // ⚠️ Пересчитать НЕ один раз. Замерено: сразу после возврата
                // сцена ещё меряется по оверлею, и кадр оставался растянутым
                // (1450×816 вместо 300×150). Та же механика, что при открытии.
                current.relayout();
                requestAnimationFrame(() => current.relayout());
                setTimeout(() => current.relayout(), 120);
                setTimeout(() => current.relayout(), 400);
            },
        });
        // ⚠️ Место у оверлея появляется не сразу и не за один кадр: пересчитываем
        // несколько раз, пока раскладка не улеглась. Тот же приём, что у
        // holdSize() в js/_dom_widget.js, и по той же причине.
        requestAnimationFrame(() => current.relayout());
        setTimeout(() => current.relayout(), 120);
        setTimeout(() => current.relayout(), 400);
    }

    addResizableDomWidget(node, root, {
        name: DOM_WIDGET,
        minWidth: 360,
        minHeight: 340,
        defaultWidth: 560,
        defaultHeight: 470,
        // ⚠️ Над виджетом стоят ТРИ обычных (fps и две подписи) плюс заголовок.
        // Замерено на живом сервере: с заниженным запасом сцена сжималась
        // до 53 пикселей высоты, и сравнивать было нечего.
        chromeHeight: 118,
        onResize: () => clip?.relayout(),
    });

    // ⚠️ Ход сборки приходит событием `progress_state`, а НЕ `progress`.
    // Замерено на сокете этой сборки ComfyUI: за прогон приходит несколько
    // `progress_state` и НИ ОДНОГО `progress`. Форма:
    //   {prompt_id, nodes: {"<id>": {value, max, state: "running"|"finished"}}}
    // Старое событие оставлено для сборок, где оно ещё живо.
    const onProgressState = (event) => {
        const entry = event?.detail?.nodes?.[String(node.id)];
        if (!entry) return;
        const total = Number(entry.max || 0);
        if (entry.state === "finished" || total <= 0) {
            setProgress(null);
            return;
        }
        // ⚠️ max === 1 означает «работает / готово», а не «один процент из ста».
        // Замерено: для этой ноды ComfyUI шлёт ровно такое состояние.
        setProgress(total > 1 ? Number(entry.value || 0) / total : true);
    };
    const onProgress = (event) => {
        const detail = event?.detail || {};
        if (String(detail.node ?? "") !== String(node.id)) return;
        const total = Number(detail.max || 0);
        if (total <= 0) return;
        setProgress(Number(detail.value || 0) / total);
    };
    const onDone = (event) => {
        if (String(event?.detail?.node ?? "") !== String(node.id)) return;
        setProgress(null);
    };
    api.addEventListener("progress_state", onProgressState);
    api.addEventListener("progress", onProgress);
    api.addEventListener("executed", onDone);

    // `element` наружу НАМЕРЕННО: запросы по всему документу цепляют элементы
    // удалённых нод, и проверки начинают врать (обжигались уже дважды).
    node.__tsCompare = { apply, setProgress, element: root };

    const previousRemoved = node.onRemoved;
    node.onRemoved = function tsCompareRemoved(...args) {
        api.removeEventListener("progress_state", onProgressState);
        api.removeEventListener("progress", onProgress);
        api.removeEventListener("executed", onDone);
        clip?.teardown();
        stills?.teardown?.();
        return previousRemoved?.apply(this, args);
    };

    // Результат прошлого прогона переживает перезагрузку страницы: файл лежит
    // в temp, а адрес — в properties ноды.
    const saved = node.properties?.[PROP_PAYLOAD];
    if (saved) apply(saved);
}

app.registerExtension({
    name: EXTENSION_ID,

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                setupCompare(this);
            } catch (error) {
                console.error("[TS Compare] setup failed", error);
            }
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function tsCompareExecuted(message, ...rest) {
            const payload = message?.[UI_KEY]?.[0];
            if (payload) this.__tsCompare?.apply(payload);
            return onExecuted?.apply(this, [message, ...rest]);
        };
    },

    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_TYPE && node?.type !== NODE_TYPE) return;
        // Виджет не пересоздаём — в Nodes 2.0 это двоит шапку ноды (§12.5.12).
        if (!node.__tsCompare) setupCompare(node);
    },
});
