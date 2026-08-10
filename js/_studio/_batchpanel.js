// TS Studio kit — поверхность пачки: список промптов, ход и результаты.
//
// Очередь и порядок фаз живут в `_batch.js` и ничего не знают об экране; здесь
// — только экран. Разделение то же, что у истории и панели итераций, и по той
// же причине: порядок фаз проверяется без браузера, а вёрстка меняется, не
// трогая логику.
//
// ЧТО ЧЕЛОВЕК ВИДИТ. Слева — промпты, по строке на кадр, и туда же можно
// бросить пачку картинок: из них вытащат промпты. Справа — задания с их
// состоянием и готовыми кадрами. Ход виден построчно, а не одной полосой на
// всё: в пачке из двадцати кадров «47%» не отвечает ни на один вопрос.

import { TS_UI_CLASS } from "../_theme.js";
import { SEED_MODES } from "./_batch.js";
import { makeDropZone } from "./_dnd.js";

/**
 * Поверхность пачки.
 *
 * @param {object} options
 * @param {object} options.t Подписи раздела (уже на нужном языке).
 * @param {() => void} [options.onStart]
 * @param {() => void} [options.onStop]
 * @param {(items: object[]) => void} [options.onFiles] Бросили картинки —
 *   промпты из них вытаскивает раздел: это его дело, а не поверхности.
 *   Приходят уже разобранные `DropItem` (у каждого `getBlob()`), поэтому
 *   диск и карточки Artius попадают сюда одинаково.
 * @returns {object} контракт поверхности
 */
export function createBatchPanel({ t, onStart, onStop, onFiles } = {}) {
    const element = document.createElement("div");
    element.className = `${TS_UI_CLASS} ts-batch`;

    const head = document.createElement("div");
    head.className = "ts-batch__head";
    const title = document.createElement("div");
    title.className = "ts-batch__title";
    title.textContent = t.title;
    const summary = document.createElement("div");
    summary.className = "ts-batch__summary";
    head.append(title, summary);

    const body = document.createElement("div");
    body.className = "ts-batch__body";

    // ── слева: промпты ───────────────────────────────────────────────────── #
    const left = document.createElement("div");
    left.className = "ts-batch__left";
    const editor = document.createElement("textarea");
    editor.className = "ts-ui-textarea ts-batch__editor";
    editor.placeholder = t.placeholder;
    editor.spellcheck = false;
    const hint = document.createElement("div");
    hint.className = "ts-batch__hint";
    hint.textContent = t.dropHint;

    const options = document.createElement("div");
    options.className = "ts-batch__options";

    const enhanceLabel = document.createElement("label");
    enhanceLabel.className = "ts-batch__check";
    const enhance = document.createElement("input");
    enhance.type = "checkbox";
    enhance.checked = true;
    enhanceLabel.append(enhance, document.createTextNode(` ${t.enhance}`));
    enhanceLabel.title = t.enhanceTip;

    const seedLabel = document.createElement("label");
    seedLabel.className = "ts-batch__field";
    seedLabel.append(document.createTextNode(t.seeds));
    const seedMode = document.createElement("select");
    seedMode.className = "ts-ui-select";
    for (const mode of SEED_MODES) {
        const option = document.createElement("option");
        option.value = mode;
        option.textContent = t.seedNames[mode] || mode;
        seedMode.appendChild(option);
    }
    seedMode.value = "random";
    seedLabel.appendChild(seedMode);

    const baseLabel = document.createElement("label");
    baseLabel.className = "ts-batch__field";
    baseLabel.append(document.createTextNode(t.baseSeed));
    const baseSeed = document.createElement("input");
    baseSeed.type = "number";
    baseSeed.className = "ts-ui-input ts-batch__seed";
    baseSeed.value = "0";
    baseSeed.min = "0";
    baseLabel.appendChild(baseSeed);
    // База нужна только там, где от неё считают: у случайного сида её нет.
    const syncSeedFields = () => {
        baseLabel.style.display = seedMode.value === "random" ? "none" : "";
    };
    seedMode.addEventListener("change", syncSeedFields);
    syncSeedFields();

    options.append(enhanceLabel, seedLabel, baseLabel);
    left.append(editor, hint, options);

    // ── справа: задания ──────────────────────────────────────────────────── #
    const list = document.createElement("div");
    list.className = "ts-batch__list";
    const empty = document.createElement("div");
    empty.className = "ts-batch__empty";
    empty.textContent = t.empty;
    list.appendChild(empty);

    body.append(left, list);

    // ── низ: пуск и остановка ────────────────────────────────────────────── #
    const foot = document.createElement("div");
    foot.className = "ts-batch__foot";
    const status = document.createElement("div");
    status.className = "ts-batch__status";
    const start = document.createElement("button");
    start.type = "button";
    start.className = "ts-ui-btn ts-ui-btn--primary";
    start.textContent = t.start;
    const stop = document.createElement("button");
    stop.type = "button";
    stop.className = "ts-ui-btn";
    stop.textContent = t.stop;
    stop.disabled = true;
    foot.append(status, stop, start);

    element.append(head, body, foot);

    /** Строки промптов: пустые не считаются, порядок сохраняется. */
    function prompts() {
        return editor.value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    }

    function updateSummary() {
        const total = prompts().length;
        summary.textContent = total ? t.count(total) : "";
        start.disabled = running || total === 0;
    }

    let running = false;
    // ⚠️ Сразу, а не только по вводу: иначе кнопка пуска открывается доступной
    // над пустой очередью, и первое же нажатие ничего не делает.
    updateSummary();
    editor.addEventListener("input", updateSummary);
    start.addEventListener("click", () => onStart?.());
    stop.addEventListener("click", () => onStop?.());

    // ⚠️ Общая зона, а не свой обработчик `drop`. Только она знает про все
    // источники студии разом: файлы с диска, карточки библиотеки и карточки
    // Artius (у тех перетаскивание порой доходит без своих типов, и снимок
    // подхватывается с window). Свой обработчик умел бы ровно один из трёх.
    const releaseZone = makeDropZone(element, {
        onDrop: (items) => {
            const images = items.filter((item) => item.type === "image");
            if (images.length) onFiles?.(images);
        },
    });

    return {
        element,
        prompts,

        /** Что выбрано в настройках прогона. */
        settings: () => ({
            enhance: enhance.checked,
            seedMode: seedMode.value,
            baseSeed: Math.max(0, Math.floor(Number(baseSeed.value) || 0)),
        }),

        /** Дописать промпты (из брошенных картинок или из браузера ассетов). */
        addPrompts(lines) {
            const clean = (Array.isArray(lines) ? lines : [])
                .map((line) => String(line || "").replace(/\s+/g, " ").trim())
                .filter(Boolean);
            if (!clean.length) return 0;
            const existing = editor.value.trim();
            editor.value = existing ? `${existing}\n${clean.join("\n")}` : clean.join("\n");
            updateSummary();
            return clean.length;
        },

        setStatus(message) { status.textContent = message || ""; },

        setRunning(value) {
            running = Boolean(value);
            editor.disabled = running;
            enhance.disabled = running;
            seedMode.disabled = running;
            baseSeed.disabled = running;
            stop.disabled = !running;
            updateSummary();
        },

        /**
         * Показать ход. Строка на задание: номер, промпт, состояние, кадр.
         *
         * @param {object} state то, что вернул `createBatch().state()`
         */
        render(state) {
            const tasks = state?.tasks || [];
            list.textContent = "";
            if (!tasks.length) {
                list.appendChild(empty);
                return;
            }
            for (const [index, task] of tasks.entries()) {
                const row = document.createElement("div");
                row.className = `ts-batch__row is-${task.status}`;
                const number = document.createElement("span");
                number.className = "ts-batch__num";
                number.textContent = String(index + 1);
                const text = document.createElement("span");
                text.className = "ts-batch__text";
                text.textContent = task.result?.text || task.input?.prompt || "";
                text.title = text.textContent;
                const chip = document.createElement("span");
                chip.className = "ts-batch__chip";
                chip.textContent = task.error
                    ? t.states.failed
                    : (t.states[task.status] || task.status);
                if (task.error) chip.title = task.error;
                row.append(number, text, chip);
                if (task.result?.url) {
                    const thumb = document.createElement("img");
                    thumb.className = "ts-batch__thumb";
                    thumb.src = task.result.url;
                    thumb.alt = "";
                    row.appendChild(thumb);
                }
                list.appendChild(row);
            }
            // Что идёт сейчас — наверх взгляда: в длинном списке строка
            // «выполняется» иначе уезжает за экран.
            const active = tasks.findIndex((task) => task.status === "running");
            if (active >= 0) list.children[active]?.scrollIntoView?.({ block: "nearest" });
        },

        destroy() { releaseZone?.(); element.remove(); },
    };
}

/** Стиль поверхности. Раскладка своя, цвета — только из токенов темы. */
export const BATCH_CSS = `
.ts-batch{position:absolute;inset:0;display:flex;flex-direction:column;gap:10px;
    padding:14px;box-sizing:border-box;background:var(--ts-bg);z-index:4}
/* Класс ставит общая зона дропа (makeDropZone) — свой заводить нельзя,
   иначе подсветка живёт отдельно от того, что зона на самом деле примет. */
.ts-batch.is-drag-over{outline:2px dashed var(--ts-accent);outline-offset:-6px}
.ts-batch__head{display:flex;align-items:baseline;gap:10px}
.ts-batch__title{font-size:var(--ts-fs-lg);color:var(--ts-text)}
.ts-batch__summary{font-size:var(--ts-fs-xs);color:var(--ts-muted)}
.ts-batch__body{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);
    gap:12px}
.ts-batch__left{display:flex;flex-direction:column;gap:8px;min-height:0}
.ts-batch__editor{flex:1;min-height:120px;resize:none;font-size:var(--ts-fs);line-height:1.45}
.ts-batch__hint{font-size:var(--ts-fs-xs);color:var(--ts-muted)}
.ts-batch__options{display:flex;flex-wrap:wrap;align-items:center;gap:12px;
    font-size:var(--ts-fs-sm);color:var(--ts-muted)}
.ts-batch__check,.ts-batch__field{display:inline-flex;align-items:center;gap:6px}
.ts-batch__seed{width:110px}
.ts-batch__list{min-height:0;overflow-y:auto;display:flex;flex-direction:column;gap:4px;
    border:1px solid var(--ts-border);border-radius:var(--ts-radius);
    background:var(--ts-sunken);padding:6px}
.ts-batch__empty{margin:auto;padding:20px;text-align:center;
    font-size:var(--ts-fs-sm);color:var(--ts-muted)}
.ts-batch__row{display:flex;align-items:center;gap:8px;padding:5px 7px;
    border-radius:var(--ts-radius-sm);background:var(--ts-surface)}
.ts-batch__row.is-running{background:var(--ts-accent-soft)}
.ts-batch__row.is-failed .ts-batch__chip{color:var(--ts-danger)}
.ts-batch__row.is-done .ts-batch__chip{color:var(--ts-success)}
.ts-batch__num{min-width:20px;text-align:right;font-size:var(--ts-fs-xs);
    color:var(--ts-muted);font-variant-numeric:tabular-nums}
.ts-batch__text{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
    font-size:var(--ts-fs-sm);color:var(--ts-text)}
.ts-batch__chip{font-size:var(--ts-fs-xs);color:var(--ts-muted);white-space:nowrap}
.ts-batch__thumb{width:34px;height:34px;object-fit:cover;border-radius:var(--ts-radius-sm);
    background:var(--ts-sunken)}
.ts-batch__foot{display:flex;align-items:center;gap:10px}
.ts-batch__status{flex:1;min-width:0;font-size:var(--ts-fs-xs);color:var(--ts-muted);
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
`;
