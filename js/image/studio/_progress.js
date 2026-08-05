// Что происходит прямо сейчас — по событиям ComfyUI и ни по чему больше.
//
// Модуль без DOM: на вход события прогона, на выход — состояние («грузим
// модель», «третий тайл из шести», «шаг 12 из 20»). Так его можно проверить до
// последней ветки без браузера, а раньше эта логика жила внутри обработчиков
// на две тысячи строк, и единственным способом её проверить был живой прогон.
//
// ЧЕГО СТОИЛО НЕЗНАНИЕ. Сетка тайлов дважды «чинилась» вслепую, потому что
// апскейлы устроены по-разному, а проверялся один. Замеры на живых прогонах:
//
//   свой резчик   `TS_ImageTileSplitter` режет кадр до сэмплера. Наружу про
//                 тайлы не приходит НИЧЕГО — только ход сэмплера. Но после
//                 `TS_ImageBatchToImageList` ComfyUI гоняет остаток графа по
//                 одному куску, и сэмплер стартует заново на каждом: картинка
//                 в два тайла дала события 1,2,3 и снова 1,2,3. Значит номер
//                 текущего тайла — это число перезапусков.
//   тайловый VAE  `VAEEncodeTiled` / `VAEDecodeTiled` отчитываются за каждый
//                 тайл, и `max` в событии равен их числу.
//
// Оба случая здесь названы явно (`perTile` против `byPass`), потому что путать
// их — значит показывать человеку ход, которого нет.

/** Этап по классу узла: от частного к общему — имена классов пересекаются. */
const STAGE_BY_CLASS = [
    [/InpaintCrop/i, "crop"],
    [/InpaintRestore/i, "restore"],
    // Собственные ноды пака делают всю работу целиком — у Klein это
    // TSSmartInpaint, и он же занимает почти всё время прогона.
    [/SmartInpaint|LamaCleanup/i, "sample"],
    [/Loader$|Loader[A-Z]|LoraStack/i, "load"],
    [/TextEncode|Conditioning|Designer|Guider|Scheduler|Sigmas/i, "prompt"],
    [/VAEEncode|NoiseMask|DifferentialDiffusion|ModelSampling|CFG/i, "encode"],
    [/VAEDecode/i, "decode"],
    [/Sampler|KSampler|Noise$/i, "sample"],
    [/SaveImage|StudioOutput|PreviewImage/i, "save"],
    // Маркеры студии исполняются мгновенно; называть их этапом — только мигать
    // подписью, поэтому они остаются в общем «работаю».
];

/**
 * Этап прогона по классу узла.
 *
 * @param {string} classType class_type узла из графа
 * @returns {"crop"|"restore"|"load"|"prompt"|"encode"|"decode"|"sample"|"save"|"other"}
 */
export function stageOf(classType) {
    const name = String(classType || "");
    for (const [pattern, stage] of STAGE_BY_CLASS) {
        if (pattern.test(name)) return stage;
    }
    return "other";
}

/** Этапы, на которых прогресс считает тайлы, а не шаги. */
const TILED_STAGES = new Set(["encode", "decode"]);

/**
 * Следит за ходом одного прогона.
 *
 * Ничего не рисует и ничего не знает про экран: возвращает решения, а показывать
 * их — дело вызывающего.
 *
 * @param {object} [options]
 * @param {(nodeId: string) => string} [options.classOf] class_type узла по его id
 * @returns {object} автомат с методами node/progress/preview/reset
 */
export function createRunProgress(options = {}) {
    const classOf = options.classOf || (() => "");

    const state = {
        stage: "other",
        /** Доля хода сэмплера внутри его прогона (0..1) или null. */
        samplerFraction: null,
        /** Сколько узлов графа уже отработало и сколько всего. */
        nodesDone: 0,
        nodesTotal: 0,
        /** Номер тайла, который считается сейчас (с нуля). */
        tileIndex: 0,
        /** Сколько тайлов всего, когда движок это сообщает. */
        tileTotal: 0,
        /** Как считаются тайлы: "perTile" | "byPass" | "" (не тайловый прогон). */
        tileMode: "",
        lastSamplerValue: 0,
    };

    return {
        /** Текущее состояние — копия, чтобы его нельзя было испортить снаружи. */
        get: () => ({ ...state }),

        /** Начался узел графа. */
        node(nodeId) {
            const stage = stageOf(classOf(nodeId));
            const changed = stage !== state.stage;
            state.stage = stage;
            if (stage !== "sample") state.samplerFraction = null;
            return { stage, changed };
        },

        /** Сколько узлов графа отработало (грубая канва хода). */
        nodes(done, total) {
            state.nodesDone = Number(done) || 0;
            state.nodesTotal = Number(total) || 0;
        },

        /**
         * Событие прогресса.
         *
         * @param {number} value текущее значение
         * @param {number} max предел
         * @param {string} [nodeId] узел, который его прислал
         * @returns {{kind:"tiles"|"steps", tileIndex:number, tileTotal:number,
         *           fraction:number|null, stage:string, tileChanged:boolean}}
         */
        progress(value, max, nodeId) {
            if (nodeId !== undefined && nodeId !== null) {
                const stage = stageOf(classOf(nodeId));
                if (stage !== "other") state.stage = stage;
            }
            // Тайловый VAE: `max` — это число тайлов, и оно точное.
            if (TILED_STAGES.has(state.stage) && max > 1) {
                const changed = state.tileMode !== "perTile" || state.tileTotal !== max;
                state.tileMode = "perTile";
                state.tileTotal = max;
                state.tileIndex = Math.max(0, Math.min(max, value));
                state.samplerFraction = null;
                return {
                    kind: "tiles", tileIndex: state.tileIndex, tileTotal: max,
                    fraction: max ? value / max : null, stage: state.stage,
                    tileChanged: changed,
                };
            }
            // Свой резчик: тайлы наружу не видны, но сэмплер стартует заново на
            // каждом. Значение пошло назад — начался следующий кусок.
            const restarted = value <= state.lastSamplerValue && state.lastSamplerValue > 0;
            if (restarted && state.tileMode !== "perTile") {
                state.tileIndex += 1;
                state.tileMode = "byPass";
            }
            state.lastSamplerValue = value;
            state.stage = "sample";
            state.samplerFraction = max ? value / max : null;
            return {
                kind: "steps", tileIndex: state.tileIndex, tileTotal: state.tileTotal,
                fraction: state.samplerFraction, stage: "sample",
                tileChanged: restarted,
            };
        },

        /**
         * Пришло превью. Отвечает, кому оно принадлежит.
         *
         * Первые шаги — почти чистый шум: быстрый декодер показывает его честно,
         * и поверх лица это выглядит пугающе. Поэтому первые кадры пропускаются.
         *
         * @param {number} skip сколько первых превью не показывать
         * @returns {{show:boolean, tileIndex:number}}
         */
        preview(skip = 0) {
            state.previewsSeen = (state.previewsSeen || 0) + 1;
            return {
                show: state.previewsSeen > skip,
                tileIndex: state.tileIndex,
            };
        },

        /** Новый прогон — счётчики с нуля. */
        reset() {
            state.stage = "other";
            state.samplerFraction = null;
            state.nodesDone = 0;
            state.nodesTotal = 0;
            state.tileIndex = 0;
            state.tileTotal = 0;
            state.tileMode = "";
            state.lastSamplerValue = 0;
            state.previewsSeen = 0;
        },
    };
}
