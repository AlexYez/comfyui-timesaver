// TS Prompt Library — выбор промпта из библиотеки, показ и копирование.
//
// Сверху раздел и модель, ниже поиск и пресет со стрелками; в карточке — превью (если
// оно есть у пресета), описание, сам промпт с подсвеченными полями, поля для
// подстановки, примечание и кнопка «Копировать». Выход ноды — тот же текст,
// что копирует кнопка: подстановка общая (`_prompt_library.js`) и совпадает с
// серверной.
//
// Значения живут в двух штатных виджетах, которые здесь только спрятаны:
// `preset` (ключ `section/collection/ID`) и `fields` (JSON значений полей).
// Поэтому граф сохраняется и работает и без этого файла.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import {
    TS_UI_CLASS,
    createOpenInterfaceButton,
    ensureThemeStyles,
    getUiLanguage,
    pickLocaleStrings,
} from "../_theme.js";
import { addResizableDomWidget, getWidget, hideWidget } from "../_dom_widget.js";
import { openFullscreenOverlay } from "../_fullscreen.js";
import {
    emptyFields,
    fillPrompt,
    filterGroups,
    findPreset,
    neighbourKey,
    parseFields,
    pickText,
    promptSegments,
    queryWords,
    searchPresets,
    switchKey,
} from "./_prompt_library.js";

const NODE_TYPE = "TS_PromptLibrary";
const DOM_WIDGET_NAME = "ts_prompt_library";
const STYLE_ID = "ts-prompt-library-styles";
const DEFAULT_NODE_WIDTH = 540;
const DEFAULT_NODE_HEIGHT = 620;

const STRINGS = {
    en: {
        section: "Section",
        model: "Model",
        preset: "Prompt",
        previous: "Previous prompt",
        next: "Next prompt",
        loading: "Loading the library…",
        failed: "The library could not be loaded — see the console.",
        empty: "The library is empty. Presets live in nodes/text/prompt_library.",
        missing: (key) => `Preset «${key}» is not in the library. Pick another one.`,
        inputs: (n) => (n === 0 ? "no input image" : n === 1 ? "1 image" : `${n} images`),
        inputsHint: (refs) => `Reference order: ${refs}`,
        fields: "Fields",
        useExample: "Example",
        useExampleHint: "Put the example into the field",
        note: "Note",
        about: "About this collection",
        source: "Source",
        copy: "Copy prompt",
        copyHint: "Copy the prompt with the fields filled in — the same text the node outputs",
        copied: "Copied",
        copyFailed: "Could not copy — select the text and press Ctrl+C.",
        unfilled: (names) => `Empty fields stay in braces: ${names}`,
        openHint: "Open the library full screen: every prompt in a list with search, the chosen one next to it",
        fullscreenTitle: "Prompt library",
        close: "Close (Esc)",
        search: "Search: code, title or prompt text",
        searchHint: "Every word must match: code, title, group, description or the prompt itself. ↑ / ↓ move through the results, Enter picks, Esc clears",
        searchKeys: "↑ / ↓ — move, Enter — pick, Esc — clear the search",
        found: (n, total) => (n === total ? `${total} prompts` : `${n} of ${total}`),
        nothingFound: "Nothing matches the search.",
        wired: "The preset comes from the wire. The card shows the node's own choice, which is not used while the wire is connected; the fields below are still applied.",
        copyWired: "The preset comes from the wire — the node outputs that preset, not this card",
    },
    ru: {
        section: "Раздел",
        model: "Модель",
        preset: "Промпт",
        previous: "Предыдущий промпт",
        next: "Следующий промпт",
        loading: "Загружаю библиотеку…",
        failed: "Библиотеку не удалось загрузить — подробности в консоли.",
        empty: "Библиотека пуста. Пресеты лежат в nodes/text/prompt_library.",
        missing: (key) => `Пресета «${key}» нет в библиотеке. Выберите другой.`,
        inputs: (n) => (n === 0 ? "без входного изображения"
            : n === 1 ? "1 изображение" : n < 5 ? `${n} изображения` : `${n} изображений`),
        inputsHint: (refs) => `Порядок референсов: ${refs}`,
        fields: "Поля",
        useExample: "Пример",
        useExampleHint: "Подставить пример в поле",
        note: "Примечание",
        about: "О коллекции",
        source: "Источник",
        copy: "Копировать промпт",
        copyHint: "Скопировать промпт с заполненными полями — тот же текст, что уходит с выхода ноды",
        copied: "Скопировано",
        copyFailed: "Не удалось скопировать — выделите текст и нажмите Ctrl+C.",
        unfilled: (names) => `Пустые поля останутся в скобках: ${names}`,
        openHint: "Открыть библиотеку во весь экран: все промпты списком с поиском, выбранный — рядом",
        fullscreenTitle: "Библиотека промптов",
        close: "Закрыть (Esc)",
        search: "Поиск: код, название или текст промпта",
        searchHint: "Совпасть должно каждое слово: код, название, группа, описание или сам промпт. ↑ / ↓ — по найденному, Enter — выбрать, Esc — сбросить",
        searchKeys: "↑ / ↓ — листать, Enter — выбрать, Esc — сбросить поиск",
        found: (n, total) => (n === total ? `Промптов: ${total}` : `${n} из ${total}`),
        nothingFound: "Ничего не нашлось.",
        wired: "Пресет приходит по проводу. В карточке — собственный выбор ноды: пока провод подключён, он не используется; поля ниже по-прежнему применяются.",
        copyWired: "Пресет приходит по проводу — с выхода уходит он, а не эта карточка",
    },
};

function ensureStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-plib{display:flex;flex-direction:column;gap:6px;height:100%;min-height:0;box-sizing:border-box;
  font-size:var(--ts-fs-sm)}
.ts-plib__row{display:flex;align-items:center;gap:6px;flex:0 0 auto;min-width:0}
.ts-plib__row .ts-ui-select{flex:1 1 0;min-width:0}
.ts-plib__nav{flex:0 0 auto}
/* Прокрутка вынесена из потока (как у TS Style Prompt): иначе естественная
   высота карточки становится высотой ноды в Nodes 2.0 и раздувает её. */
.ts-plib__body{position:relative;flex:1 1 0;min-height:0}
.ts-plib__scroll{position:absolute;inset:0;overflow-y:auto;overflow-x:hidden;
  scrollbar-gutter:stable;box-sizing:border-box;padding-right:2px;
  display:flex;flex-direction:column;gap:8px}
.ts-plib__message{padding:18px 12px;text-align:center;color:var(--ts-muted);line-height:1.5}
.ts-plib__preview{width:100%;max-height:200px;object-fit:contain;border-radius:var(--ts-radius);
  background:var(--ts-sunken);flex:0 0 auto}
.ts-plib__head{display:flex;align-items:baseline;gap:8px;min-width:0}
.ts-plib__code{font-weight:700;color:var(--ts-accent);font-variant-numeric:tabular-nums;flex:0 0 auto}
.ts-plib__title{font-weight:600;color:var(--ts-text);flex:1 1 auto;min-width:0}
.ts-plib__badge{flex:0 0 auto;font-size:var(--ts-fs-xs);color:var(--ts-accent);
  background:var(--ts-accent-soft);border:1px solid var(--ts-accent-line);
  border-radius:999px;padding:0 7px;white-space:nowrap}
.ts-plib__summary{color:var(--ts-muted);line-height:1.45}
.ts-plib__prompt{background:var(--ts-sunken);border:1px solid var(--ts-border-soft);
  border-radius:var(--ts-radius);padding:8px 10px;line-height:1.55;color:var(--ts-text);
  white-space:pre-wrap;word-break:break-word;user-select:text;cursor:text}
.ts-plib__slot{border-radius:3px;padding:0 2px}
.ts-plib__slot--empty{color:var(--ts-warning);border:1px dashed var(--ts-warning)}
.ts-plib__slot--filled{color:var(--ts-accent);background:var(--ts-accent-soft)}
.ts-plib__fields{display:flex;flex-direction:column;gap:6px}
.ts-plib__field{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px 6px;align-items:center}
.ts-plib__field-name{grid-column:1 / -1;font-size:var(--ts-fs-xs);color:var(--ts-muted)}
.ts-plib__field-name b{color:var(--ts-text);font-weight:600;margin-right:6px}
.ts-plib__field .ts-ui-input{min-width:0;width:100%;box-sizing:border-box}
.ts-plib__more{color:var(--ts-muted);line-height:1.45}
.ts-plib__more summary{cursor:pointer;color:var(--ts-text);font-weight:600;margin-bottom:4px}
.ts-plib__more ul{margin:4px 0 0 18px;padding:0}
.ts-plib__more a{color:var(--ts-accent)}
.ts-plib__footer{display:flex;align-items:center;gap:8px;flex:0 0 auto;min-width:0}
.ts-plib__footer .ts-ui-btn--primary{flex:0 0 auto}
.ts-plib__status{flex:1 1 auto;min-width:0;font-size:var(--ts-fs-xs);color:var(--ts-muted);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ts-plib__status--warn{color:var(--ts-warning)}
.ts-plib__footer .ts-ui-launch{margin-left:auto}
.ts-plib__wired{flex:0 0 auto;padding:6px 8px;line-height:1.45;font-size:var(--ts-fs-xs);
  color:var(--ts-warning);border:1px dashed var(--ts-warning);border-radius:var(--ts-radius)}
.ts-plib__wired[hidden]{display:none}
.ts-plib__list.is-disabled{opacity:.5;pointer-events:none}

/* Оболочка держит место в ноде, пока содержимое живёт в полноэкранном окне. */
.ts-plib-shell{width:100%;height:100%;min-height:0;display:flex}
.ts-plib-shell > .ts-plib{flex:1 1 auto;min-width:0}
.ts-plib__main{display:flex;gap:12px;flex:1 1 0;min-height:0;min-width:0}
.ts-plib__content{display:flex;flex-direction:column;gap:6px;flex:1 1 0;min-width:0;min-height:0}
/* Колонка списка — только во весь экран: в ноде ей нет места, и миниатюры
   там не должны грузиться вовсе. Строка поиска одна на оба режима и
   переезжает: в ноде она над пресетом, во весь экран — над списком. */
.ts-plib__side{display:none}
.ts-plib__search .ts-ui-input{flex:1 1 0;min-width:0}
.ts-plib__search .ts-plib__count{flex:0 0 auto;white-space:nowrap}
.ts-plib__count:empty{display:none}
/* Найденное в ноде показывается на месте карточки, прокручивает его она. */
.ts-plib__list--inline{flex:0 0 auto;overflow:visible}
.ts-plib__item.is-highlight{border-color:var(--ts-accent)}

.ts-plib.is-fullscreen{height:100%;padding:10px 16px 14px;font-size:var(--ts-fs);gap:10px}
.ts-plib.is-fullscreen .ts-plib__side{display:flex;flex-direction:column;gap:6px;
  flex:0 0 clamp(260px,28%,420px);min-height:0}
.ts-plib.is-fullscreen .ts-plib__content{max-width:1100px}
.ts-plib.is-fullscreen .ts-plib__preview{max-height:46vh}
.ts-plib.is-fullscreen .ts-plib__prompt{font-size:var(--ts-fs);line-height:1.6;padding:10px 12px}
.ts-plib__count{font-size:var(--ts-fs-xs);color:var(--ts-muted)}
.ts-plib__list{flex:1 1 0;min-height:0;overflow-y:auto;overflow-x:hidden;
  border:1px solid var(--ts-border-soft);border-radius:var(--ts-radius);
  background:var(--ts-sunken);padding:4px}
.ts-plib__group-title{font-size:var(--ts-fs-xs);color:var(--ts-muted);text-transform:uppercase;
  letter-spacing:.04em;margin:8px 6px 3px}
.ts-plib__item{display:flex;align-items:center;gap:8px;width:100%;box-sizing:border-box;
  text-align:left;background:none;border:1px solid transparent;border-radius:var(--ts-radius-sm);
  padding:5px 6px;color:var(--ts-text);cursor:pointer;font:inherit}
.ts-plib__item:hover{background:var(--ts-surface-hover)}
.ts-plib__item.is-active{background:var(--ts-accent-soft);border-color:var(--ts-accent-line)}
.ts-plib__thumb{width:44px;height:44px;object-fit:cover;border-radius:var(--ts-radius-sm);
  flex:0 0 auto;background:var(--ts-surface)}
.ts-plib__item-code{font-weight:700;color:var(--ts-accent);font-variant-numeric:tabular-nums;
  flex:0 0 auto;min-width:2.6em}
.ts-plib__item-title{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
`;
    document.head.appendChild(style);
}

// Каталог один на страницу: его отдаёт сервер, он описывает папку с JSON.
let catalogRequest = null;

function loadCatalog(force = false) {
    if (force || !catalogRequest) {
        catalogRequest = api.fetchApi("/ts_prompt_library/catalog")
            .then((response) => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return response.json();
            })
            .catch((error) => {
                catalogRequest = null;
                throw error;
            });
    }
    return catalogRequest;
}

function el(tag, className = "", text = "") {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
}

function option(value, label) {
    const node = document.createElement("option");
    node.value = value;
    node.textContent = label;
    return node;
}

function writeWidget(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return;
    widget.value = value;
    try {
        widget.callback?.(value);
    } catch (error) {
        console.warn("[TS PromptLibrary] widget callback failed", error);
    }
    node.setDirtyCanvas?.(true, true);
}

async function copyText(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch {
        // Буфер обмена недоступен (не HTTPS, нет разрешения) — старый путь.
        const area = el("textarea");
        area.value = text;
        area.style.position = "fixed";
        area.style.left = "-9999px";
        document.body.appendChild(area);
        area.select();
        let ok = false;
        try {
            ok = document.execCommand("copy");
        } catch {
            ok = false;
        }
        area.remove();
        return ok;
    }
}

function setupPromptLibrary(node) {
    if (node.__tsPromptLibrary) return;
    ensureStyles();
    const t = pickLocaleStrings(STRINGS);
    const lang = getUiLanguage();

    // ⚠️ У `preset` гнездо остаётся: пресет можно подать проводом. Спрятан
    // только ряд виджета — выбором ведает панель ниже.
    hideWidget(node, "preset", { keepInput: true });
    hideWidget(node, "fields");

    const root = el("div", `${TS_UI_CLASS} ts-plib`);
    const rowTop = el("div", "ts-plib__row");
    const sectionSelect = el("select", "ts-ui-select");
    sectionSelect.title = t.section;
    const collectionSelect = el("select", "ts-ui-select");
    collectionSelect.title = t.model;
    rowTop.append(sectionSelect, collectionSelect);

    const rowPreset = el("div", "ts-plib__row");
    const presetSelect = el("select", "ts-ui-select");
    presetSelect.title = t.preset;
    const prev = el("button", "ts-ui-btn ts-ui-btn--icon ts-plib__nav", "‹");
    prev.type = "button";
    prev.title = t.previous;
    const next = el("button", "ts-ui-btn ts-ui-btn--icon ts-plib__nav", "›");
    next.type = "button";
    next.title = t.next;
    rowPreset.append(presetSelect, prev, next);

    const body = el("div", "ts-plib__body");
    const scroll = el("div", "ts-plib__scroll");
    body.appendChild(scroll);

    // Поиск один на оба режима: в ноде он над пресетом, во весь экран
    // переезжает в колонку каталога (openFullscreen / onClose).
    const searchRow = el("div", "ts-plib__row ts-plib__search");
    const search = el("input", "ts-ui-input");
    search.type = "search";
    search.placeholder = t.search;
    search.title = t.searchHint;
    search.spellcheck = false;
    const count = el("div", "ts-plib__count");
    searchRow.append(search, count);

    // Колонка каталога — видна только во весь экран (см. стили).
    const side = el("div", "ts-plib__side");
    const list = el("div", "ts-plib__list");
    side.append(list);

    const content = el("div", "ts-plib__content");
    content.append(searchRow, rowPreset, body);
    const main = el("div", "ts-plib__main");
    main.append(side, content);

    const footer = el("div", "ts-plib__footer");
    const copyButton = el("button", "ts-ui-btn ts-ui-btn--primary", t.copy);
    copyButton.type = "button";
    copyButton.title = t.copyHint;
    const status = el("div", "ts-plib__status");
    const openButton = createOpenInterfaceButton(() => openFullscreen(), {
        lang,
        description: t.openHint,
    });
    footer.append(copyButton, status, openButton);

    // Провод на `preset` отменяет выбор панели — говорим об этом прямо.
    const wiredNote = el("div", "ts-plib__wired", t.wired);
    wiredNote.hidden = true;

    root.append(rowTop, wiredNote, main, footer);

    // Виджет держит оболочку, а переезжает наполнение: так закрытие окна
    // возвращает всё на место вместе с состоянием (приём TS Song Creator).
    const shell = el("div", "ts-plib-shell");
    shell.appendChild(root);

    // picking — в ноде вместо карточки показан список найденного;
    // highlight — подсвеченная в нём строка (Enter выбирает её).
    const state = { catalog: null, error: null, fullscreen: null, query: "", picking: false, highlight: 0 };
    let promptBlock = null;
    let copyTimer = null;

    const currentKey = () => String(getWidget(node, "preset")?.value ?? "");
    const currentValues = () => parseFields(getWidget(node, "fields")?.value);
    // Подключённый провод: сервер возьмёт пресет с него, а не из виджета.
    const isWired = () => (node.inputs || []).some((i) => i?.name === "preset" && i.link != null);
    // Флаг — класс, а не state.fullscreen: onOpen зовётся ИЗНУТРИ
    // openFullscreenOverlay, когда его результат ещё не присвоен.
    const isFullscreen = () => root.classList.contains("is-fullscreen");
    const searching = () => queryWords(state.query).length > 0;

    /**
     * Честная панель при проводе: карточка показывает выбор виджета, а с выхода
     * уходит пресет с провода. Выбор и «Копировать» тогда вводили бы в
     * заблуждение — они выключены, над карточкой пояснение. Поля остаются:
     * сервер применяет их к пресету с провода.
     */
    function applyWiredState() {
        const wired = isWired();
        wiredNote.hidden = !wired;
        for (const control of [sectionSelect, collectionSelect, presetSelect, prev, next, copyButton]) {
            control.disabled = wired;
        }
        copyButton.title = wired ? t.copyWired : t.copyHint;
        list.classList.toggle("is-disabled", wired);
    }

    function setKey(key) {
        if (isWired()) return;
        if (!key || key === currentKey()) return;
        writeWidget(node, "preset", key);
        render();
    }

    // Шаг по списку. Сохранённого пресета может не быть (граф из другой версии
    // библиотеки) — тогда соседа не найти, и кнопки со стрелками молчали бы.
    // Шаг из «ниоткуда» приводит к показанному первому пресету.
    // С поиском листается только найденное (neighbourKey).
    function stepKey(step) {
        if (!state.catalog) return;
        state.picking = false;
        if (findPreset(state.catalog, currentKey())) {
            const key = neighbourKey(state.catalog, currentKey(), step, state.query);
            if (key === currentKey()) render();
            else setKey(key);
            return;
        }
        const first = state.catalog.sections?.[0]?.collections?.[0]?.groups?.[0]?.presets?.[0];
        setKey(first?.key);
    }

    function setValue(name, value) {
        const values = currentValues();
        if (String(value).trim()) values[name] = value;
        else delete values[name];
        writeWidget(node, "fields", JSON.stringify(values));
        paintPrompt();
    }

    function paintPrompt() {
        const found = state.catalog && findPreset(state.catalog, currentKey());
        if (!found || !promptBlock) return;
        const values = currentValues();
        promptBlock.textContent = "";
        for (const part of promptSegments(found.preset.prompt, values)) {
            if (!part.field) {
                promptBlock.appendChild(document.createTextNode(part.text));
                continue;
            }
            const slot = el("span",
                `ts-plib__slot ts-plib__slot--${part.filled ? "filled" : "empty"}`, part.text);
            slot.title = part.field;
            promptBlock.appendChild(slot);
        }
        const empty = emptyFields(found.preset.prompt, values);
        status.textContent = empty.length ? t.unfilled(empty.join(", ")) : "";
        status.classList.toggle("ts-plib__status--warn", empty.length > 0);
    }

    function fillSelects(found) {
        const { catalog } = state;
        sectionSelect.textContent = "";
        for (const section of catalog.sections) {
            sectionSelect.appendChild(option(section.id, pickText(section.title, lang)));
        }
        sectionSelect.value = found.section.id;

        collectionSelect.textContent = "";
        for (const collection of found.section.collections) {
            collectionSelect.appendChild(option(collection.id, collection.title));
        }
        collectionSelect.value = found.collection.id;

        presetSelect.textContent = "";
        for (const group of found.collection.groups) {
            const holder = document.createElement("optgroup");
            holder.label = pickText(group.title, lang);
            for (const preset of group.presets) {
                holder.appendChild(option(preset.key, `${preset.id} · ${pickText(preset.title, lang)}`));
            }
            presetSelect.appendChild(holder);
        }
        presetSelect.value = found.preset.key;
    }

    function renderFields(found) {
        const { preset, collection } = found;
        if (!preset.fields.length) return null;
        const values = currentValues();
        const box = el("div", "ts-plib__fields");
        box.appendChild(el("div", "ts-ui-label", t.fields));
        for (const name of preset.fields) {
            const spec = collection.placeholders?.[name] || {};
            const example = preset.examples?.[name] || spec.example || "";
            const row = el("label", "ts-plib__field");
            const caption = el("div", "ts-plib__field-name");
            caption.appendChild(el("b", "", `{${name}}`));
            caption.appendChild(document.createTextNode(pickText(spec.label, lang)));
            const input = el("input", "ts-ui-input");
            input.type = "text";
            input.spellcheck = false;
            input.value = values[name] ?? "";
            input.placeholder = example;
            input.addEventListener("input", () => setValue(name, input.value));
            row.append(caption, input);
            if (example) {
                const use = el("button", "ts-ui-btn ts-ui-btn--ghost", t.useExample);
                use.type = "button";
                use.title = t.useExampleHint;
                use.addEventListener("click", (event) => {
                    event.preventDefault();
                    input.value = example;
                    setValue(name, example);
                });
                row.appendChild(use);
            }
            box.appendChild(row);
        }
        return box;
    }

    function renderMore(title, build) {
        const box = el("details", "ts-plib__more");
        box.appendChild(el("summary", "", title));
        build(box);
        return box;
    }

    function link(url, text) {
        const anchor = el("a", "", text);
        anchor.href = url;
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
        return anchor;
    }

    function renderCard(found) {
        const { preset, collection } = found;
        if (preset.preview) {
            const image = el("img", "ts-plib__preview");
            image.loading = "lazy";
            image.alt = pickText(preset.title, lang);
            image.src = api.apiURL(preset.preview);
            scroll.appendChild(image);
        }
        const head = el("div", "ts-plib__head");
        head.appendChild(el("span", "ts-plib__code", preset.id));
        head.appendChild(el("span", "ts-plib__title", pickText(preset.title, lang)));
        const badge = el("span", "ts-plib__badge", t.inputs(preset.inputs));
        if (preset.inputs > 1 && collection.references) badge.title = t.inputsHint(collection.references);
        head.appendChild(badge);
        scroll.appendChild(head);

        const summary = pickText(preset.summary, lang);
        if (summary) scroll.appendChild(el("div", "ts-plib__summary", summary));

        promptBlock = el("div", "ts-plib__prompt");
        scroll.appendChild(promptBlock);

        const fields = renderFields(found);
        if (fields) scroll.appendChild(fields);

        const note = pickText(preset.note, lang);
        if (note || preset.links?.length) {
            scroll.appendChild(renderMore(t.note, (box) => {
                if (note) box.appendChild(el("div", "", note));
                for (const item of preset.links || []) {
                    const line = el("div");
                    line.appendChild(link(item.url, item.title));
                    box.appendChild(line);
                }
            }));
        }

        scroll.appendChild(renderMore(t.about, (box) => {
            box.appendChild(el("div", "", `${collection.title} — ${collection.model}`));
            const about = pickText(collection.summary, lang);
            if (about) box.appendChild(el("div", "", about));
            const guide = collection.guide?.[lang] || collection.guide?.en || [];
            if (guide.length) {
                const list = el("ul");
                for (const line of guide) list.appendChild(el("li", "", line));
                box.appendChild(list);
            }
            if (collection.source?.url) {
                const line = el("div", "", `${t.source}: `);
                line.appendChild(link(collection.source.url, collection.source.title || collection.source.url));
                box.appendChild(line);
            }
        }));
    }

    function showMessage(text) {
        scroll.textContent = "";
        promptBlock = null;
        scroll.appendChild(el("div", "ts-plib__message", text));
        status.textContent = "";
    }

    /**
     * Список промптов текущей модели во весь экран.
     *
     * Рисуется ТОЛЬКО пока окно открыто: миниатюры превью — это запросы к
     * серверу, и компактная нода их не делает вовсе.
     */
    function renderList(found) {
        // Флаг — класс, а не state.fullscreen: onOpen зовётся ИЗНУТРИ
        // openFullscreenOverlay, когда его результат ещё не присвоен.
        if (!root.classList.contains("is-fullscreen") || !found) return;
        list.textContent = "";
        const shown = updateCount(found);
        if (!shown) {
            list.appendChild(el("div", "ts-plib__message", t.nothingFound));
            return;
        }
        const active = renderItems(list, found, { thumbs: true });
        active?.scrollIntoView({ block: "nearest" });
    }

    /** «N из M» — во весь экран всегда, в ноде только пока идёт поиск. */
    function updateCount(found) {
        const total = found.collection.groups.reduce((sum, g) => sum + g.presets.length, 0);
        const shown = searchPresets(found.collection, state.query).length;
        count.textContent = isFullscreen() || searching() ? t.found(shown, total) : "";
        return shown;
    }

    /**
     * Строки найденного по группам. Возвращает строку, которую надо держать в
     * поле зрения: подсвеченную, иначе текущий пресет.
     */
    function renderItems(container, found, { thumbs = false, highlightKey = null } = {}) {
        let active = null;
        let lit = null;
        for (const group of filterGroups(found.collection, state.query)) {
            container.appendChild(el("div", "ts-plib__group-title", pickText(group.title, lang)));
            for (const preset of group.presets) {
                const item = el("button", "ts-plib__item");
                item.type = "button";
                item.title = pickText(preset.summary, lang);
                // Миниатюры — только во весь экран: компактная нода превью не грузит.
                if (thumbs && preset.preview) {
                    const thumb = el("img", "ts-plib__thumb");
                    thumb.loading = "lazy";
                    thumb.alt = "";
                    thumb.src = api.apiURL(preset.preview);
                    item.appendChild(thumb);
                }
                item.appendChild(el("span", "ts-plib__item-code", preset.id));
                item.appendChild(el("span", "ts-plib__item-title", pickText(preset.title, lang)));
                if (preset.key === found.preset.key) {
                    item.classList.add("is-active");
                    active = item;
                }
                if (preset.key === highlightKey) {
                    item.classList.add("is-highlight");
                    lit = item;
                }
                item.addEventListener("click", () => choose(preset.key));
                container.appendChild(item);
            }
        }
        return lit || active;
    }

    /** Найденное на месте карточки — поиск в ноде. */
    function renderResults(found) {
        scroll.textContent = "";
        promptBlock = null;
        const presets = searchPresets(found.collection, state.query);
        updateCount(found);
        status.textContent = presets.length ? t.searchKeys : "";
        status.classList.remove("ts-plib__status--warn");
        if (!presets.length) {
            scroll.appendChild(el("div", "ts-plib__message", t.nothingFound));
            return;
        }
        state.highlight = Math.min(Math.max(state.highlight, 0), presets.length - 1);
        const box = el("div", "ts-plib__list ts-plib__list--inline");
        box.classList.toggle("is-disabled", isWired());
        const lit = renderItems(box, found, { highlightKey: presets[state.highlight].key });
        scroll.appendChild(box);
        lit?.scrollIntoView({ block: "nearest" });
    }

    /** Выбор из найденного: карточка возвращается, запрос остаётся в поле. */
    function choose(key) {
        if (isWired() || !key) return;
        state.picking = false;
        if (key === currentKey()) render();
        else setKey(key);
    }

    /**
     * Найденное сейчас, по порядку каталога. Сохранённого пресета может не
     * быть — тогда ищем по показанной коллекции, как и render().
     */
    function currentResults() {
        if (!state.catalog?.sections?.length) return [];
        const first = state.catalog.sections[0].collections[0].groups[0].presets[0];
        const found = findPreset(state.catalog, currentKey()) || findPreset(state.catalog, first.key);
        return found ? searchPresets(found.collection, state.query) : [];
    }

    function openFullscreen() {
        if (state.fullscreen?.isOpen()) return;
        root.classList.add("is-fullscreen");
        // Во весь экран найденное — в колонке слева, карточка остаётся видна.
        state.picking = false;
        side.insertBefore(searchRow, list);
        render();
        state.fullscreen = openFullscreenOverlay(root, {
            label: t.fullscreenTitle,
            closeTitle: t.close,
            trigger: openButton,
            onOpen: () => {
                renderList(state.catalog && findPreset(state.catalog, currentKey()));
            },
            onKey: (event, { typing = false } = {}) => {
                // В поле поиска и в полях подстановки стрелки — это ввод.
                if (typing) return;
                const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[event.key];
                if (!step || !state.catalog) return;
                event.preventDefault();
                stepKey(step);
            },
            onClose: () => {
                root.classList.remove("is-fullscreen");
                state.fullscreen = null;
                list.textContent = "";
                content.insertBefore(searchRow, rowPreset);
                shell.appendChild(root);
                render();
                node.setDirtyCanvas?.(true, true);
            },
        });
    }

    search.addEventListener("input", () => {
        state.query = search.value;
        state.highlight = 0;
        // В ноде найденное встаёт на место карточки; во весь экран — в колонку.
        state.picking = !isFullscreen() && searching();
        // Во весь экран карточка от запроса не зависит — перестраиваем только
        // колонку, иначе превью мигало бы на каждой букве.
        if (isFullscreen()) renderList(state.catalog && findPreset(state.catalog, currentKey()));
        else render();
    });

    search.addEventListener("focus", () => {
        if (isFullscreen() || state.picking || !searching()) return;
        state.picking = true;
        render();
    });

    // Клавиши поля поиска. Во весь экран стрелки сразу листают найденное
    // (карточка рядом), в ноде — двигают подсветку в списке, Enter выбирает.
    // ⚠️ Esc во весь экран закрывает окно раньше нас (общий обработчик в фазе
    // перехвата) — так и задумано: «Закрыть (Esc)» обещано на кнопке.
    search.addEventListener("keydown", (event) => {
        const step = { ArrowDown: 1, ArrowUp: -1 }[event.key];
        if (event.key === "Escape") {
            if (!state.query) return;
            event.preventDefault();
            event.stopPropagation();
            search.value = "";
            state.query = "";
            state.picking = false;
            render();
            return;
        }
        if (!step && event.key !== "Enter") return;
        event.preventDefault();
        event.stopPropagation();
        const results = currentResults();
        if (isFullscreen()) {
            if (step) stepKey(step);
            else if (results.length && !results.some((p) => p.key === currentKey())) choose(results[0].key);
            return;
        }
        if (!searching()) return;
        if (!state.picking) {
            state.picking = true;
            render();
            return;
        }
        if (!results.length) return;
        if (step) {
            state.highlight = (state.highlight + step + results.length) % results.length;
            render();
        } else {
            choose(results[Math.min(state.highlight, results.length - 1)].key);
        }
    });

    function render() {
        applyWiredState();
        if (state.error) return showMessage(t.failed);
        if (!state.catalog) return showMessage(t.loading);
        if (!state.catalog.sections?.length) return showMessage(t.empty);
        let found = findPreset(state.catalog, currentKey());
        if (!found) {
            // Ключа нет (граф из другой версии библиотеки): показываем первый
            // пресет, но в виджет его НЕ пишем — пусть человек решит сам, а
            // прогон честно скажет, что сохранённого пресета нет.
            const first = state.catalog.sections[0].collections[0].groups[0].presets[0];
            found = findPreset(state.catalog, first.key);
            fillSelects(found);
            // Ничего не выделено: иначе выбор именно первого пункта не дал бы
            // события change, и выйти из этого состояния было бы нечем.
            presetSelect.selectedIndex = -1;
            // Поиск выводит и отсюда: найденное — по показанной коллекции.
            if (state.picking) renderResults(found);
            else {
                showMessage(t.missing(currentKey()));
                updateCount(found);
            }
            renderList(found);
            return;
        }
        fillSelects(found);
        if (state.picking) {
            renderResults(found);
            return;
        }
        scroll.textContent = "";
        renderCard(found);
        paintPrompt();
        scroll.scrollTop = 0;
        updateCount(found);
        renderList(found);
    }

    async function refresh(force = false) {
        try {
            state.catalog = await loadCatalog(force);
            state.error = null;
            // Новая нода без выбора — берём первый пресет каталога.
            if (!currentKey() && state.catalog.sections?.length) {
                writeWidget(node, "preset", state.catalog.sections[0].collections[0].groups[0].presets[0].key);
            }
        } catch (error) {
            console.error("[TS PromptLibrary] catalog request failed", error);
            state.error = error;
        }
        render();
    }

    sectionSelect.addEventListener("change", () =>
        setKey(switchKey(state.catalog, currentKey(), sectionSelect.value)));
    collectionSelect.addEventListener("change", () =>
        setKey(switchKey(state.catalog, currentKey(), sectionSelect.value, collectionSelect.value)));
    presetSelect.addEventListener("change", () => choose(presetSelect.value));
    prev.addEventListener("click", () => stepKey(-1));
    next.addEventListener("click", () => stepKey(1));

    copyButton.addEventListener("click", async () => {
        if (isWired()) return;
        const found = state.catalog && findPreset(state.catalog, currentKey());
        if (!found) return;
        const ok = await copyText(fillPrompt(found.preset.prompt, currentValues()));
        clearTimeout(copyTimer);
        copyButton.textContent = ok ? `${t.copied} ✓` : t.copy;
        if (!ok) {
            status.textContent = t.copyFailed;
            status.classList.add("ts-plib__status--warn");
        }
        copyTimer = setTimeout(() => { copyButton.textContent = t.copy; }, 1600);
    });

    // LiteGraph уже посчитал размер по виджетам, и хелпер его уважает — без
    // этой строки новая нода встаёт в минимум. Сохранённый размер приезжает
    // позже, в onConfigure (как у TS Angle Select).
    node.size = [DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT];
    addResizableDomWidget(node, shell, {
        name: DOM_WIDGET_NAME,
        minWidth: 400,
        minHeight: 360,
        defaultWidth: DEFAULT_NODE_WIDTH,
        defaultHeight: DEFAULT_NODE_HEIGHT,
        chromeHeight: 30,
        minWidgetHeight: 300,
    });

    node.__tsPromptLibrary = { render, refresh };
    // «R» в ComfyUI: библиотеку могли поправить на диске.
    node.refreshComboInNode = () => refresh(true);

    // Провод подключили или сняли — панель тут же перестраивается.
    const previousConnections = node.onConnectionsChange;
    node.onConnectionsChange = function tsPromptLibraryConnections(...args) {
        const result = previousConnections?.apply(this, args);
        try {
            applyWiredState();
        } catch (error) {
            console.warn("[TS PromptLibrary] wire state update failed", error);
        }
        return result;
    };

    const previousRemoved = node.onRemoved;
    node.onRemoved = function tsPromptLibraryRemoved(...args) {
        clearTimeout(copyTimer);
        // Нода удалена при открытом окне — окно не должно пережить её.
        state.fullscreen?.close();
        return previousRemoved?.apply(this, args);
    };

    refresh();
}

app.registerExtension({
    name: "ts.promptLibrary",
    nodeCreated(node) {
        if (node?.comfyClass === NODE_TYPE) setupPromptLibrary(node);
    },
    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_TYPE) return;
        // Значения виджетов из workflow пришли только сейчас (§12.5.12):
        // перерисовываем то, что уже собрано, а не собираем заново.
        if (!node.__tsPromptLibrary) setupPromptLibrary(node);
        else node.__tsPromptLibrary.render();
    },
});
