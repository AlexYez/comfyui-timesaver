// Список моделей TS Files Downloader: статусы, разделение адреса и папки.
//
// ⚠️ Почему не оставили обычное текстовое поле. Строка списка — это ДВЕ разные
// вещи, адрес и папка, а выглядела одной: длинный URL переносится, и папка,
// стоящая через пробел, читается как его хвост. И ещё: скачана модель или нет,
// было видно только по журналу сервера.
//
// ⚠️ Значение при этом хранится ТЕМ ЖЕ текстом в том же виджете `file_list`.
// Здесь только вид: разобрали строки, показали, собрали обратно. Воркфлоу,
// сохранённый год назад, открывается как открывался, а список, набранный
// руками в текстовом режиме, остаётся законным.

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";

const STATUS_ORDER = { partial: 0, missing: 1, unknown: 2, ready: 3, skip: 4 };

/** Разделитель, который видно. Пробел тоже понимается — см. splitLine. */
export const ARROW = " → ";

/**
 * Строка списка -> {url, target, comment}.
 *
 * Понимает три формы: стрелку, ASCII-стрелку и старый пробел. Последняя обязана
 * работать всегда: с ней приходят чужие воркфлоу.
 */
export function parseLine(raw) {
    const text = String(raw ?? "").trim();
    if (!text) return { blank: true };
    if (text.startsWith("#")) return { comment: text };

    for (const marker of [" → ", " -> "]) {
        const at = text.indexOf(marker);
        if (at > 0) {
            return { url: text.slice(0, at).trim(), target: text.slice(at + marker.length).trim() };
        }
    }
    const parts = text.split(/\s+/);
    return { url: parts[0] || "", target: parts.slice(1).join(" ") };
}

/** {url, target} -> строка в том виде, в каком она ложится в виджет. */
export function formatLine(entry) {
    if (entry?.comment) return entry.comment;
    if (entry?.blank) return "";
    const target = String(entry?.target || "").trim();
    return target ? `${entry.url}${ARROW}${target}` : String(entry?.url || "");
}

/** Имя файла из адреса — то же, что покажет сервер в статусе. */
export function fileNameOf(url) {
    const raw = String(url || "");
    let path = raw;
    try {
        path = new URL(raw).pathname;
    } catch {
        path = raw.split(/[?#]/)[0];
    }
    return decodeURIComponent(path.split("/").filter(Boolean).pop() || raw);
}

function bytes(value) {
    const size = Number(value || 0);
    if (!(size > 0)) return "";
    const units = ["Б", "КБ", "МБ", "ГБ", "ТБ"];
    let index = 0;
    let left = size;
    while (left >= 1024 && index < units.length - 1) {
        left /= 1024;
        index += 1;
    }
    return `${left >= 10 || index === 0 ? Math.round(left) : left.toFixed(1)} ${units[index]}`;
}

export function ensureListStyles() {
    // ⚠️ Тема — ПЕРВОЙ строкой, до раннего возврата (§12.6). Без неё токены
    // `--ts-*` не объявлены, и цветные точки статуса выходят невидимыми:
    // background берётся из пустой переменной.
    ensureThemeStyles();
    if (document.getElementById("ts-fdl-list-styles")) return;
    const style = document.createElement("style");
    style.id = "ts-fdl-list-styles";
    style.textContent = `
.ts-fdl-list{display:flex;flex-direction:column;gap:6px;height:100%;min-height:0;
  position:relative}
.ts-fdl-list__bar{display:flex;align-items:center;gap:8px;flex:0 0 auto}
.ts-fdl-list__count{font-size:var(--ts-fs-sm);color:var(--ts-muted);white-space:nowrap}
.ts-fdl-list__count b{color:var(--ts-text);font-weight:600}
.ts-fdl-list__spacer{flex:1 1 auto}

.ts-fdl-list__rows{
  flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;
  border:1px solid var(--ts-border);border-radius:var(--ts-radius);
  background:var(--ts-sunken);padding:3px}
.ts-fdl-list__empty{
  padding:18px 12px;text-align:center;color:var(--ts-muted);
  font-size:var(--ts-fs-sm);line-height:1.5}

.ts-fdl-row{
  display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:8px;
  padding:5px 7px;border-radius:var(--ts-radius-sm);
  border:1px solid transparent}
.ts-fdl-row:hover{background:var(--ts-surface);border-color:var(--ts-border-soft)}
.ts-fdl-row--comment{opacity:.6}

/* Точка статуса. Цвет — семантический из темы, форма одна на все состояния,
   чтобы взгляд ловил ряд, а не разнобой. */
.ts-fdl-dot{
  width:9px;height:9px;border-radius:50%;flex:0 0 auto;
  box-shadow:0 0 0 3px var(--ts-bg)}
.ts-fdl-dot--ready{background:var(--ts-success)}
.ts-fdl-dot--partial{background:var(--ts-warning)}
.ts-fdl-dot--missing{background:var(--ts-danger)}
.ts-fdl-dot--unknown{background:var(--ts-muted)}
.ts-fdl-dot--busy{background:var(--ts-accent);animation:ts-fdl-pulse 1.1s ease-in-out infinite}
@keyframes ts-fdl-pulse{0%,100%{opacity:1}50%{opacity:.35}}
@media (prefers-reduced-motion:reduce){.ts-fdl-dot--busy{animation:none}}

.ts-fdl-row__main{min-width:0;display:flex;flex-direction:column;gap:1px}
.ts-fdl-row__name{
  font-size:var(--ts-fs);color:var(--ts-text);white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.ts-fdl-row__where{display:flex;align-items:center;gap:6px;min-width:0}
.ts-fdl-row__folder{
  font-size:var(--ts-fs-xs);color:var(--ts-accent);
  background:var(--ts-accent-soft);border:1px solid var(--ts-accent-line);
  border-radius:999px;padding:0 6px;white-space:nowrap;flex:0 0 auto}
.ts-fdl-row__host{
  font-size:var(--ts-fs-xs);color:var(--ts-faint);white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.ts-fdl-row__size{
  font-size:var(--ts-fs-xs);color:var(--ts-muted);white-space:nowrap;
  font-variant-numeric:tabular-nums}
.ts-fdl-row__drop{
  border:0;background:none;color:var(--ts-faint);cursor:pointer;
  font-size:15px;line-height:1;padding:2px 4px;border-radius:var(--ts-radius-sm);
  opacity:0}
.ts-fdl-row:hover .ts-fdl-row__drop{opacity:1}
.ts-fdl-row__drop:hover{color:var(--ts-danger);background:var(--ts-surface-hover)}
.ts-fdl-row__drop:focus-visible{opacity:1;outline:2px solid var(--ts-accent);outline-offset:1px}

.ts-fdl-list__text{
  flex:1 1 auto;min-height:0;width:100%;resize:none;
  font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
  font-size:var(--ts-fs-sm);line-height:1.5}

/* Полоса загрузки живёт В СТРОКЕ модели: общий процент на кнопке не отвечает
   на вопрос «а эта скачалась?», когда моделей десяток. Место под неё
   резервируется всегда, иначе строки прыгают в момент старта загрузки. */
.ts-fdl-row__bar{
  height:3px;border-radius:2px;background:var(--ts-border-soft);
  overflow:hidden;margin-top:3px;opacity:0;transition:opacity .15s ease}
.ts-fdl-row--busy .ts-fdl-row__bar,
.ts-fdl-row--done .ts-fdl-row__bar{opacity:1}
.ts-fdl-row__fill{
  height:100%;width:0%;border-radius:2px;background:var(--ts-accent);
  transition:width .2s linear}
.ts-fdl-row--done .ts-fdl-row__fill{background:var(--ts-success)}

/* Действия — ПОД списком: сначала то, что качают, потом чем настраивают. */
.ts-fdl-list__actions{display:flex;gap:6px;flex:0 0 auto;flex-wrap:wrap}
.ts-fdl-list__actions .ts-ui-btn{flex:1 1 auto;min-width:0}
.ts-fdl-list__actions .ts-fdl-settings{flex:0 0 auto}
`;
    document.head.appendChild(style);
}

/**
 * Редактор списка: список со статусами и текстовый режим для массовой правки.
 *
 * @param {object} options
 * @param {() => string} options.getValue   текущее значение виджета.
 * @param {(text:string)=>void} options.setValue записать значение обратно.
 * @param {object} options.strings          локализованные строки.
 * @param {(text:string)=>Promise<object>} options.fetchStatus запрос статусов.
 */
export function createListEditor({ getValue, setValue, strings: L, fetchStatus }) {
    ensureListStyles();

    const root = document.createElement("div");
    root.className = `${TS_UI_CLASS} ts-fdl-list`;

    const bar = document.createElement("div");
    bar.className = "ts-fdl-list__bar ts-ui-toolbar";

    const count = document.createElement("span");
    count.className = "ts-fdl-list__count";

    const spacer = document.createElement("div");
    spacer.className = "ts-fdl-list__spacer";

    const refreshButton = document.createElement("button");
    refreshButton.type = "button";
    refreshButton.className = "ts-ui-btn";
    refreshButton.textContent = L.listRefresh;
    refreshButton.title = L.listRefreshHint;

    const modeButton = document.createElement("button");
    modeButton.type = "button";
    modeButton.className = "ts-ui-btn";
    modeButton.title = L.listModeHint;

    bar.append(count, spacer, refreshButton, modeButton);

    const rows = document.createElement("div");
    rows.className = "ts-fdl-list__rows";

    const textarea = document.createElement("textarea");
    textarea.className = "ts-ui-textarea ts-fdl-list__text";
    textarea.spellcheck = false;
    textarea.placeholder = L.listPlaceholder;
    textarea.hidden = true;

    // ⚠️ Порядок сборки и есть та компоновка, о которой просили: список
    // занимает верх ноды, действия стоят под ним, настройки прячутся ещё ниже.
    const actions = document.createElement("div");
    actions.className = "ts-fdl-list__actions";

    root.append(bar, rows, textarea, actions);

    /** Состояния строк, ключ — номер строки. */
    let statuses = new Map();
    /** Ход загрузки по строкам: {share, done} — живёт, пока идёт скачивание. */
    const progress = new Map();
    /** Имя файла, который качается прямо сейчас, и его строки. */
    let current = null;
    let currentRows = [];
    let asText = false;
    let pending = 0;

    function entries() {
        return String(getValue() || "").replace(/\r\n/g, "\n").split("\n").map(parseLine);
    }

    function writeBack(list) {
        setValue(list.map(formatLine).join("\n"));
    }

    function statusOf(index) {
        return statuses.get(index) || null;
    }

    /** Нарисовать долю на уже существующей строке. */
    function applyProgress(row, state) {
        const fill = row.querySelector(".ts-fdl-row__fill");
        if (!fill) return;
        fill.style.width = `${Math.round(Math.max(0, Math.min(1, state.share)) * 100)}%`;
        row.classList.toggle("ts-fdl-row--busy", !state.done);
        row.classList.toggle("ts-fdl-row--done", Boolean(state.done));
        const dot = row.querySelector(".ts-fdl-dot");
        if (dot && !state.done) {
            dot.className = "ts-fdl-dot ts-fdl-dot--busy";
        }
    }

    /** Найти строку по номеру — без перерисовки всего списка. */
    function rowAt(index) {
        return rows.querySelector(`.ts-fdl-row[data-ts-line="${index}"]`);
    }

    function renderRows() {
        rows.textContent = "";
        const list = entries();
        const real = list.filter((entry) => entry.url);

        if (!real.length) {
            const empty = document.createElement("div");
            empty.className = "ts-fdl-list__empty";
            empty.textContent = L.listEmpty;
            rows.appendChild(empty);
            updateCount([]);
            return;
        }

        list.forEach((entry, index) => {
            if (entry.blank) return;
            const row = document.createElement("div");
            row.className = "ts-fdl-row";

            const state = statusOf(index);
            const kind = entry.comment ? "skip" : (state?.status || "unknown");

            const dot = document.createElement("span");
            dot.className = `ts-fdl-dot ts-fdl-dot--${pending && !entry.comment ? "busy" : kind}`;
            dot.title = entry.comment ? L.listComment : (L.listStatus[kind] || kind);
            row.appendChild(dot);

            const main = document.createElement("div");
            main.className = "ts-fdl-row__main";

            const name = document.createElement("div");
            name.className = "ts-fdl-row__name";
            name.textContent = entry.comment
                ? entry.comment
                : (state?.filename || fileNameOf(entry.url));
            // Полный адрес — в подсказке: в строке ему места нет, а знать его
            // иногда нужно.
            name.title = entry.comment ? "" : entry.url;
            main.appendChild(name);

            if (!entry.comment) {
                const where = document.createElement("div");
                where.className = "ts-fdl-row__where";

                const folder = document.createElement("span");
                folder.className = "ts-fdl-row__folder";
                folder.textContent = entry.target || L.listNoFolder;
                folder.title = state?.directory || entry.target || "";
                where.appendChild(folder);

                const host = document.createElement("span");
                host.className = "ts-fdl-row__host";
                try {
                    host.textContent = new URL(entry.url).hostname;
                } catch {
                    host.textContent = "";
                }
                where.appendChild(host);
                main.appendChild(where);
            } else {
                row.classList.add("ts-fdl-row--comment");
            }

            if (!entry.comment) {
                const bar = document.createElement("div");
                bar.className = "ts-fdl-row__bar";
                const fill = document.createElement("div");
                fill.className = "ts-fdl-row__fill";
                bar.appendChild(fill);
                main.appendChild(bar);
                // Строку находим по индексу: имя файла может повториться в
                // разных папках, а номер строки — нет.
                row.dataset.tsLine = String(index);
            }

            row.appendChild(main);
            // ⚠️ Только ПОСЛЕ вставки: до неё полоска лежит вне строки,
            // и поиск по потомкам её не находит.
            const live = progress.get(index);
            if (live) applyProgress(row, live);

            const tail = document.createElement("div");
            tail.style.display = "flex";
            tail.style.alignItems = "center";
            tail.style.gap = "4px";

            if (state?.bytes) {
                const size = document.createElement("span");
                size.className = "ts-fdl-row__size";
                size.textContent = bytes(state.bytes);
                tail.appendChild(size);
            }

            const drop = document.createElement("button");
            drop.type = "button";
            drop.className = "ts-fdl-row__drop";
            drop.textContent = "✕";
            drop.title = L.listRemove;
            drop.addEventListener("click", () => {
                const current = entries();
                current.splice(index, 1);
                writeBack(current);
                statuses = new Map();
                render();
                refresh();
            });
            tail.appendChild(drop);
            row.appendChild(tail);

            rows.appendChild(row);
        });

        updateCount(list);
    }

    function updateCount(list) {
        const real = list.filter((entry) => entry.url);
        if (!real.length) {
            count.textContent = L.listCountEmpty;
            return;
        }
        let ready = 0;
        list.forEach((entry, index) => {
            if (!entry.url) return;
            if (statusOf(index)?.status === "ready") ready += 1;
        });
        count.textContent = L.listCount(real.length, ready);
    }

    function render() {
        modeButton.textContent = asText ? L.listAsRows : L.listAsText;
        rows.hidden = asText;
        textarea.hidden = !asText;
        refreshButton.disabled = asText;
        if (asText) {
            textarea.value = String(getValue() || "");
            updateCount(entries());
        } else {
            renderRows();
        }
    }

    /** Спросить сервер, что уже лежит на диске. */
    async function refresh() {
        const text = String(getValue() || "");
        if (!text.trim()) {
            statuses = new Map();
            render();
            return;
        }
        pending += 1;
        if (!asText) renderRows();
        try {
            const payload = await fetchStatus(text);
            const next = new Map();
            for (const entry of payload?.entries || []) {
                if (typeof entry?.line === "number") next.set(entry.line, entry);
            }
            statuses = next;
        } catch (error) {
            console.warn("[TS FilesDownloader] status check failed", error);
        } finally {
            pending = Math.max(0, pending - 1);
            render();
        }
    }

    modeButton.addEventListener("click", () => {
        if (asText) {
            // Уходим из текста — забираем то, что человек написал.
            setValue(textarea.value);
            statuses = new Map();
            asText = false;
            render();
            refresh();
            return;
        }
        asText = true;
        render();
        textarea.focus();
    });

    refreshButton.addEventListener("click", () => refresh());

    textarea.addEventListener("change", () => {
        setValue(textarea.value);
        statuses = new Map();
    });

    render();

    return {
        element: root,
        /** Куда нода вешает свои кнопки: «взять из воркфлоу», «скачать», «настройки». */
        actions,
        refresh,
        /** Перечитать значение виджета (после «взять модели из воркфлоу»). */
        reload() {
            statuses = new Map();
            render();
            refresh();
        },
        /**
         * Ход загрузки конкретной модели.
         *
         * ⚠️ Обновляется ТОЧЕЧНО, без перерисовки списка: событий прилетает по
         * несколько в секунду, и полная пересборка строк на каждом дёргала бы
         * прокрутку под курсором.
         */
        showProgress({ filename, doneBytes = 0, totalBytes = 0 } = {}) {
            if (!filename || asText) return;

            // ⚠️ Не всякое событие несёт байты: начало файла и его завершение
            // приходят с нулями. Считать их за «0 %» — значит сбрасывать полосу
            // в ноль на самом интересном месте.
            const known = totalBytes > 0;
            const share = known ? Math.max(0, Math.min(1, doneBytes / totalBytes)) : 0;

            // Смена имени = предыдущий файл закончен: загрузчик идёт
            // по списку строго по одному.
            if (current !== null && current !== filename) {
                for (const line of currentRows) {
                    const state = { share: 1, done: true };
                    progress.set(line, state);
                    const previous = rowAt(line);
                    if (previous) applyProgress(previous, state);
                }
                currentRows = [];
            }
            current = filename;

            const touched = [];
            entries().forEach((entry, index) => {
                if (!entry.url || fileNameOf(entry.url) !== filename) return;
                touched.push(index);
                const state = { share: known ? share : (progress.get(index)?.share || 0),
                                done: false };
                progress.set(index, state);
                const row = rowAt(index);
                if (row) applyProgress(row, state);
            });
            currentRows = touched;
        },

        /** Загрузка кончилась: полосы убрать, статусы перечитать с диска. */
        finishProgress() {
            progress.clear();
            current = null;
            currentRows = [];
            statuses = new Map();
            render();
            refresh();
        },
        isTextMode: () => asText,
    };
}

export { STATUS_ORDER };
