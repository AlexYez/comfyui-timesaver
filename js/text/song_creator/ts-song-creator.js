/**
 * TS Song Creator — текст песни и стиль в одном редакторе.
 *
 * Сверху ряд сегментных тегов ([verse], [chorus], …) — они вставляются кнопкой,
 * потому что набирать их руками в каждой песне и помнить точное написание
 * незачем. Под ним поле текста, ниже поле стиля и библиотека готовых стилей.
 *
 * ⚠️ Ударение ставится ПРАВОЙ КНОПКОЙ на гласной, и ради одного этого пункта
 * пришлось нарисовать всё контекстное меню (`_song_menu.js`): расширить родное
 * меню браузера нельзя ничем, а отнимать у человека «копировать» и «вставить»
 * ради своей команды — плохая сделка.
 *
 * ⚠️ Длинный текст правится в полноэкранном режиме — общим `_fullscreen.js`,
 * своей копии оверлея не заводить. При открытии содержимое ПЕРЕЕЗЖАЕТ в оверлей
 * и возвращается назад при закрытии: так состояние полей, выделение и история
 * ввода переживают открытие-закрытие, чего пересозданная разметка не умеет.
 *
 * Размер виджета — через общий `addResizableDomWidget` (он же держит размер
 * после загрузки workflow).
 */

import { app } from "/scripts/app.js";

import {
    TS_UI_CLASS,
    createOpenInterfaceButton,
    ensureThemeStyles,
    getUiLanguage,
    pickLocaleStrings,
    setOpenInterfaceLabel,
} from "../../_theme.js";
import { addResizableDomWidget, getWidget, hideWidget } from "../../_dom_widget.js";
import { openFullscreenOverlay } from "../../_fullscreen.js";
import { countStress, stripAllStress, toggleStress } from "./_song_stress.js";
import { replaceSegment } from "./_song_compose.js";
import {
    hideContextMenu,
    readClipboardText,
    showContextMenu,
    writeClipboardText,
} from "./_song_menu.js";
import { createStyleLibrary, forgetPresets, loadPresets, pickPreset } from "./_song_styles.js";

const EXTENSION_ID = "ts.songCreator";
const NODE_ID = "TS_SongCreator";
const STYLE_ID = "ts-song-creator-style";
const WIDGET_NAME = "ts_song_creator";

const INPUT_LYRICS = "lyrics";
const INPUT_STYLE = "style";
const INPUT_PRESET = "preset";

const MIN_NODE_WIDTH = 380;
const MIN_NODE_HEIGHT = 420;
const DEFAULT_NODE_WIDTH = 460;
const DEFAULT_NODE_HEIGHT = 560;
const CHROME_HEIGHT = 92;          // заголовок + виджет пресета
const MIN_WIDGET_HEIGHT = 300;

const STRINGS = {
    en: {
        lyrics: "Lyrics",
        style: "Style",
        styleLibrary: "Style library",
        styleLibraryTip: "Ready-made style prompts for the chosen model. Click one to put it "
            + "in the field, then edit it — a preset is a starting point.",
        vocalLibrary: "Voice",
        vocalLibraryTip: "Who sings, and how. Picking a genre brings its usual voice along; "
            + "choose another here and it replaces that one rather than piling up.",
        tabStyle: "Style",
        tabStyleTip: "The genre list: tempo, instruments and the mood of the track.",
        tabVocal: "Vocal",
        tabVocalTip: "The voice list: it replaces the singer the genre brought with it.",
        searchStyles: "Search styles",
        searchStylesTip: "Filter by name, description or by a word in the prompt itself.",
        noStyles: "Nothing matches",
        lyricsPlaceholder: "[verse]\nThe first line of the song…",
        stylePlaceholder: "pop, uplifting, 118 bpm, female vocal, bright synths",
        tagsTip: "Insert a section tag at the cursor. The model reads these to know where a "
            + "part begins.",
        copy: "Copy",
        cut: "Cut",
        paste: "Paste",
        selectAll: "Select all",
        stress: "Mark stressed",
        unstress: "Remove the stress mark",
        stressTip: "Adds a combining accent to the vowel — how a model is told which syllable "
            + "carries the stress.",
        stressNone: "Select a single vowel first",
        stripStress: "Remove every stress mark",
        clipboardDenied: "The browser did not allow reading the clipboard",
        lines: (n) => `${n} line${n === 1 ? "" : "s"}`,
        stressCount: (n) => `${n} stressed`,
        closeEditor: "Close the editor (Esc)",
        editorTitle: "TS Song Creator — editor",
        openTip: "Write the song full screen: the same fields, the whole window.",
    },
    ru: {
        lyrics: "Текст песни",
        style: "Стиль",
        styleLibrary: "Библиотека стилей",
        styleLibraryTip: "Готовые описания стиля для выбранной модели. Нажмите — попадёт в "
            + "поле, дальше правьте: пресет это отправная точка.",
        vocalLibrary: "Голос",
        vocalLibraryTip: "Кто поёт и как. Жанр приводит с собой привычный для него голос; "
            + "выберите здесь другой — он ЗАМЕНИТ прежний, а не добавится к нему.",
        tabStyle: "Стиль",
        tabStyleTip: "Список жанров: темп, инструменты и настроение трека.",
        tabVocal: "Вокал",
        tabVocalTip: "Список голосов: выбранный заменяет того певца, которого привёл жанр.",
        searchStyles: "Поиск стиля",
        searchStylesTip: "Ищет по названию, описанию и по словам самого промпта.",
        noStyles: "Ничего не нашлось",
        lyricsPlaceholder: "[verse]\nПервая строчка песни…",
        stylePlaceholder: "pop, uplifting, 118 bpm, female vocal, bright synths",
        tagsTip: "Вставить тег части песни на место курсора. По ним модель понимает, где "
            + "начинается новая часть.",
        copy: "Копировать",
        cut: "Вырезать",
        paste: "Вставить",
        selectAll: "Выделить всё",
        stress: "Сделать ударной",
        unstress: "Убрать ударение",
        stressTip: "Добавляет к гласной знак ударения — так модели говорят, на какой слог "
            + "падает ударение.",
        stressNone: "Сначала выделите одну гласную",
        stripStress: "Убрать все ударения",
        clipboardDenied: "Браузер не дал прочитать буфер обмена",
        lines: (n) => `${n} стр.`,
        stressCount: (n) => `ударений: ${n}`,
        closeEditor: "Закрыть редактор (Esc)",
        editorTitle: "TS Song Creator — редактор",
        openTip: "Писать песню во весь экран: те же поля, всё окно.",
    },
};

const STYLE_TEXT = `
.ts-song-shell{flex:1 1 auto;min-width:0;min-height:0}
.ts-song{flex:1 1 auto;width:100%;height:100%;min-width:0;min-height:0;display:flex;
    flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;font-family:var(--ts-font)}
/* ⚠️ Раскладка тела нужна В ОБОИХ режимах. Сперва эти правила стояли только
   под .is-fullscreen, и в самой ноде flex:1 у поля текста не за что было
   зацепиться: оба поля схлопывались до двух строк и не росли вместе с нодой.
   ⚠️⚠️ И обратных кавычек в этом комментарии быть не может: STYLE_TEXT —
   шаблонная строка, одна такая кавычка обрывает её и роняет весь модуль,
   причём node --check этого НЕ ловит (CLAUDE.md §12.6). */
.ts-song__body{flex:1 1 auto;min-height:0;display:flex;flex-direction:column;gap:6px}
/* ⚠️ В ноде колонок нет — есть одна колонка сверху вниз, и обе половины
   раскрываются в неё через display:contents. Без этого поле стиля считало свои
   проценты от родителя, чья высота равна содержимому, и навсегда оставалось
   щелью в 47 px, как бы ни растягивали ноду (замерено). Колонками они
   становятся только на весь экран — ниже. */
.ts-song__left,.ts-song__right{display:contents}
.ts-song__bar{flex:0 0 auto;display:flex;align-items:flex-start;gap:6px}
/* ⚠️ Теги ПЕРЕНОСЯТСЯ, а не едут в строку с прокруткой. В одну линию их
   двенадцать, и чтобы увидеть последний, ноду приходилось растягивать шире
   экрана; в узкой ноде они складываются в два-три ряда и видны все.
   Потолок с прокруткой — чтобы ряды тегов не съели поле текста. */
.ts-song__tags{flex:1 1 auto;min-width:0;display:flex;flex-wrap:wrap;gap:4px;
    max-height:76px;overflow-y:auto;scrollbar-width:thin}
.ts-song__tag{flex:0 0 auto;font-size:var(--ts-fs-xs);padding:3px 8px;border-radius:999px;
    border:1px solid var(--ts-border);background:var(--ts-sunken);color:var(--ts-text);
    cursor:pointer;white-space:nowrap}
.ts-song__tag:hover{border-color:var(--ts-accent);color:var(--ts-accent)}
.ts-song__label{flex:0 0 auto;font-size:var(--ts-fs-xs);letter-spacing:.03em;
    text-transform:uppercase;color:var(--ts-muted)}
/* ⚠️ min-height:0 обязателен: у flex-элемента минимум по умолчанию — его
   содержимое, и textarea с rows=2 не давала колонке сжиматься правильно. */
.ts-song__lyrics{flex:1 1 auto;min-height:0;height:100%;resize:none;line-height:1.45;
    box-sizing:border-box}
.ts-song__stylerow{flex:0 0 auto;display:flex;align-items:center;gap:6px}
/* Поле стиля растёт вместе с нодой, но медленнее текста песни: доля высоты,
   а не фиксированные пиксели. Жёсткого потолка нет — он читался как «поле не
   масштабируется», и был неправ: чем выше нода, тем больше места под стиль. */
.ts-song__style{flex:0 1 30%;min-height:52px;resize:none;box-sizing:border-box}
.ts-song__status{flex:0 0 auto;display:flex;gap:10px;font-size:var(--ts-fs-xs);
    color:var(--ts-muted)}
/* Кнопка открытия редактора живёт в своей строке, как у всех нод пака: в ряду
   с тегами она растягивалась вместе с ним и выглядела чужеродно. */
.ts-song__launch{flex:0 0 auto;margin-top:2px}
.ts-song.is-fullscreen .ts-song__launch{display:none}
/* ⚠️ Библиотека держится ВНЕ потока: в Nodes 2.0 высоту виджету выдают по
   min-content его содержимого, и раскрытый список стилей раздувал бы ноду
   на всю свою высоту (§12.5.1 — тот же приём, что у селектора разрешения). */
.ts-song__libwrap{position:relative;flex:0 0 auto;height:0}
.ts-song__libwrap[hidden]{display:none}
.ts-song-lib{position:absolute;left:0;right:0;bottom:0;max-height:260px;display:flex;
    flex-direction:column;gap:6px;padding:6px;border-radius:var(--ts-radius);
    border:1px solid var(--ts-border);background:var(--ts-elevated);
    box-shadow:0 8px 24px rgba(0,0,0,.35);z-index:4}
.ts-song-lib__search{flex:0 0 auto}
.ts-song-lib__tabs{flex:0 0 auto;display:flex;gap:4px}
.ts-song-lib__tab{flex:1 1 0;padding:4px 8px;font-size:var(--ts-fs-xs);
    border-radius:var(--ts-radius-sm);border:1px solid var(--ts-border);
    background:var(--ts-sunken);color:var(--ts-muted);cursor:pointer}
.ts-song-lib__tab.is-active{border-color:var(--ts-accent);color:var(--ts-accent);
    background:var(--ts-accent-soft)}
.ts-song-lib__groups{flex:0 0 auto;display:flex;flex-wrap:wrap;gap:4px}
.ts-song-lib__groups[hidden]{display:none}
.ts-song-lib__group{padding:3px 10px;font-size:var(--ts-fs-xs);border-radius:999px;
    border:1px solid var(--ts-border);background:var(--ts-sunken);color:var(--ts-text);
    cursor:pointer}
.ts-song-lib__group.is-active{border-color:var(--ts-accent);color:var(--ts-accent)}
.ts-song-lib__list{flex:1 1 auto;min-height:0;overflow-y:auto;display:flex;
    flex-direction:column;gap:3px;scrollbar-width:thin}
.ts-song-lib__card{display:flex;flex-direction:column;gap:2px;align-items:flex-start;
    text-align:left;padding:5px 8px;border-radius:var(--ts-radius-sm);
    border:1px solid transparent;background:var(--ts-sunken);color:var(--ts-text);
    cursor:pointer}
.ts-song-lib__card:hover{border-color:var(--ts-accent-line)}
.ts-song-lib__card.is-active{border-color:var(--ts-accent);background:var(--ts-accent-soft)}
.ts-song-lib__name{font-size:var(--ts-fs-sm)}
.ts-song-lib__hint{font-size:var(--ts-fs-xs);color:var(--ts-muted)}
.ts-song-lib__empty{font-size:var(--ts-fs-xs);color:var(--ts-muted);padding:4px 2px}
/* Полноэкранный режим: поля рядом, библиотека всегда открыта. */
.ts-song.is-fullscreen{gap:10px;padding:10px}
.ts-song.is-fullscreen .ts-song__body{flex:1 1 auto;min-height:0;display:grid;
    grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:10px}
.ts-song.is-fullscreen .ts-song__left,
.ts-song.is-fullscreen .ts-song__right{min-height:0;display:flex !important;
    flex-direction:column;gap:6px}
.ts-song.is-fullscreen .ts-song__right{flex:1 1 auto}
.ts-song.is-fullscreen .ts-song__style{flex:0 0 auto;height:auto;min-height:96px;
    max-height:none}
.ts-song.is-fullscreen .ts-song__libwrap{flex:2 1 auto;height:auto;min-height:200px}
.ts-song.is-fullscreen .ts-song-lib{position:static;max-height:none;height:100%;
    box-shadow:none}
/* Контекстное меню живёт в body — у него свой блок правил. */
.ts-song-menu{position:fixed;z-index:2147483000;min-width:190px;padding:4px;
    border-radius:var(--ts-radius);border:1px solid var(--ts-border);
    background:var(--ts-elevated);box-shadow:0 10px 30px rgba(0,0,0,.45);
    font-family:var(--ts-font);display:flex;flex-direction:column;gap:1px}
.ts-song-menu__item{display:block;width:100%;text-align:left;padding:6px 10px;
    border:0;border-radius:var(--ts-radius-sm);background:transparent;color:var(--ts-text);
    font-size:var(--ts-fs-sm);cursor:pointer}
.ts-song-menu__item:hover:not(:disabled){background:var(--ts-accent-soft);color:var(--ts-accent)}
.ts-song-menu__item:disabled{color:var(--ts-faint);cursor:default}
.ts-song-menu__sep{height:1px;margin:3px 4px;background:var(--ts-border)}
`;

function ensureStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = STYLE_TEXT;
    document.head.appendChild(style);
}

function readValue(node, name, fallback) {
    const widget = getWidget(node, name);
    const value = widget?.value;
    return value === undefined || value === null ? fallback : value;
}

function setValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return;
    if (widget.value === value) return;
    widget.value = value;
    if (typeof widget.callback === "function") widget.callback(value);
}

/** Не пускать события мыши и клавиш в LiteGraph: иначе холст «съедает» ввод. */
function stopPropagation(element, events) {
    for (const name of events) {
        element.addEventListener(name, (event) => event.stopPropagation());
    }
}

function setupSongCreator(node) {
    if (node._tsSongCreatorInitialized) return;
    node._tsSongCreatorInitialized = true;

    ensureStyles();
    const lang = getUiLanguage();
    const L = pickLocaleStrings(STRINGS);
    const t = (key) => {
        const value = L[key];
        return typeof value === "function" ? value : value;
    };

    const doc = document;
    const container = doc.createElement("div");
    container.className = `${TS_UI_CLASS} ts-song`;

    // ── верхняя полоса: теги + кнопки ────────────────────────────────────
    const bar = doc.createElement("div");
    bar.className = "ts-song__bar";
    const tags = doc.createElement("div");
    tags.className = "ts-song__tags";
    tags.title = L.tagsTip;

    bar.append(tags);

    // Кнопка редактора — в собственной строке `ts-ui-launchbar`, как её и
    // задумала тема (эталон — TS LaMa Cleanup). В ряду с тегами она делила с
    // ними ширину и разъезжалась.
    const launchRow = doc.createElement("div");
    launchRow.className = "ts-ui-launchbar ts-song__launch";
    const openButton = createOpenInterfaceButton(() => openEditor(), {
        lang,
        description: L.openTip,
    });
    launchRow.append(openButton);

    // ── тело: текст, стиль, библиотека ───────────────────────────────────
    const body = doc.createElement("div");
    body.className = "ts-song__body";

    const left = doc.createElement("div");
    left.className = "ts-song__left";
    const lyrics = doc.createElement("textarea");
    lyrics.className = "ts-ui-textarea ts-song__lyrics";
    lyrics.placeholder = L.lyricsPlaceholder;
    lyrics.spellcheck = false;
    left.append(lyrics);

    const right = doc.createElement("div");
    right.className = "ts-song__right";
    const styleRow = doc.createElement("div");
    styleRow.className = "ts-song__stylerow";
    const styleLabel = doc.createElement("div");
    styleLabel.className = "ts-song__label";
    styleLabel.textContent = L.style;
    const libButton = doc.createElement("button");
    libButton.type = "button";
    libButton.className = "ts-ui-btn ts-ui-btn--ghost";
    libButton.textContent = L.styleLibrary;
    libButton.title = L.styleLibraryTip;
    // Отдельная кнопка на голос: это второе измерение стиля, и искать его во
    // вкладке внутри библиотеки человеку незачем.
    const vocalButton = doc.createElement("button");
    vocalButton.type = "button";
    vocalButton.className = "ts-ui-btn ts-ui-btn--ghost";
    vocalButton.textContent = L.vocalLibrary;
    vocalButton.title = L.vocalLibraryTip;
    styleRow.append(styleLabel, libButton, vocalButton);

    const styleField = doc.createElement("textarea");
    styleField.className = "ts-ui-textarea ts-song__style";
    styleField.placeholder = L.stylePlaceholder;
    styleField.spellcheck = false;

    const libWrap = doc.createElement("div");
    libWrap.className = "ts-song__libwrap";
    libWrap.hidden = true;

    const library = createStyleLibrary({
        t: (key) => L[key] || key,
        lang,
        // Выбор жанра заменяет ТОЛЬКО жанровую часть и заодно подставляет
        // голос, под который стиль написан, — если своего человек ещё не
        // выбирал. Иначе поле оставалось бы без вокала вовсе, а модель сама
        // решала бы, кому петь.
        onPick: (prompt, style) => {
            const suggested = style?.vocal
                ? (state.vocals.find((item) => item.id === style.vocal)?.prompt || "")
                : "";
            let next = replaceSegment(styleField.value, state.lastStyle, prompt);
            if (!state.lastVocal && suggested) {
                next = replaceSegment(next, "", suggested);
                state.lastVocal = suggested;
            }
            state.lastStyle = prompt;
            styleField.value = next;
            commitStyle();
        },
        onPickVocal: (prompt) => {
            styleField.value = replaceSegment(styleField.value, state.lastVocal, prompt);
            state.lastVocal = prompt;
            commitStyle();
        },
    });
    libWrap.appendChild(library.element);

    right.append(styleRow, styleField, libWrap);
    body.append(left, right);

    const status = doc.createElement("div");
    status.className = "ts-song__status";
    const linesStatus = doc.createElement("span");
    const stressStatus = doc.createElement("span");
    status.append(linesStatus, stressStatus);

    container.append(bar, body, status, launchRow);

    const state = {
        fullscreen: null,
        presetName: String(readValue(node, INPUT_PRESET, "")),
        // Что мы сами положили в поле в прошлый раз — чтобы заменить именно
        // это, а не дописать третий голос к двум прежним.
        lastStyle: "",
        lastVocal: "",
        vocals: [],
    };

    // ── запись в виджеты ─────────────────────────────────────────────────
    function commitLyrics() {
        setValue(node, INPUT_LYRICS, lyrics.value);
        refreshStatus();
    }

    function commitStyle() {
        setValue(node, INPUT_STYLE, styleField.value);
        library.setActive(styleField.value.trim());
    }

    function refreshStatus() {
        const count = lyrics.value ? lyrics.value.split("\n").length : 0;
        linesStatus.textContent = L.lines(count);
        const stressed = countStress(lyrics.value);
        stressStatus.textContent = stressed ? L.stressCount(stressed) : "";
    }

    lyrics.addEventListener("input", commitLyrics);
    styleField.addEventListener("input", commitStyle);

    // ── сегментные теги ──────────────────────────────────────────────────
    function insertTag(tag) {
        const start = lyrics.selectionStart ?? lyrics.value.length;
        const end = lyrics.selectionEnd ?? start;
        const before = lyrics.value.slice(0, start);
        const after = lyrics.value.slice(end);
        // Тег — это НАЧАЛО части, поэтому он всегда встаёт на свою строку, а
        // за ним идёт пустая: набирать перевод строки руками после каждой
        // вставки — ровно та морока, ради которой кнопки и сделаны.
        const prefix = before && !before.endsWith("\n") ? "\n" : "";
        const inserted = `${prefix}${tag}\n`;
        lyrics.value = before + inserted + after;
        const caret = (before + inserted).length;
        lyrics.setSelectionRange(caret, caret);
        lyrics.focus();
        commitLyrics();
    }

    function renderTags(list) {
        tags.textContent = "";
        for (const item of list || []) {
            const button = doc.createElement("button");
            button.type = "button";
            button.className = "ts-song__tag";
            const name = item?.name || {};
            button.textContent = String(name[lang] || name.en || item.tag);
            button.title = `${item.tag} — ${L.tagsTip}`;
            button.addEventListener("click", () => insertTag(item.tag));
            tags.appendChild(button);
        }
    }

    // ── контекстное меню ─────────────────────────────────────────────────
    lyrics.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        event.stopPropagation();

        const hasSelection = lyrics.selectionStart !== lyrics.selectionEnd;
        const probe = toggleStress(lyrics.value, lyrics.selectionStart, lyrics.selectionEnd);
        const stressLabel = probe?.stressed === false ? L.unstress : L.stress;

        showContextMenu({ x: event.clientX, y: event.clientY }, [
            {
                label: L.copy,
                disabled: !hasSelection,
                onSelect: () => writeClipboardText(
                    lyrics.value.slice(lyrics.selectionStart, lyrics.selectionEnd)),
            },
            {
                label: L.cut,
                disabled: !hasSelection,
                onSelect: async () => {
                    const start = lyrics.selectionStart;
                    const end = lyrics.selectionEnd;
                    await writeClipboardText(lyrics.value.slice(start, end));
                    lyrics.value = lyrics.value.slice(0, start) + lyrics.value.slice(end);
                    lyrics.setSelectionRange(start, start);
                    lyrics.focus();
                    commitLyrics();
                },
            },
            {
                label: L.paste,
                onSelect: async () => {
                    const text = await readClipboardText();
                    if (!text) {
                        console.warn(`[TS Song Creator] ${L.clipboardDenied}`);
                        return;
                    }
                    const start = lyrics.selectionStart;
                    const end = lyrics.selectionEnd;
                    lyrics.value = lyrics.value.slice(0, start) + text + lyrics.value.slice(end);
                    const caret = start + text.length;
                    lyrics.setSelectionRange(caret, caret);
                    lyrics.focus();
                    commitLyrics();
                },
            },
            {
                label: L.selectAll,
                onSelect: () => {
                    lyrics.focus();
                    lyrics.select();
                },
            },
            { separator: true },
            {
                label: stressLabel,
                hint: probe ? L.stressTip : L.stressNone,
                disabled: !probe,
                onSelect: () => {
                    const result = toggleStress(
                        lyrics.value, lyrics.selectionStart, lyrics.selectionEnd);
                    if (!result) return;
                    lyrics.value = result.text;
                    lyrics.setSelectionRange(result.selectionStart, result.selectionEnd);
                    lyrics.focus();
                    commitLyrics();
                },
            },
            {
                label: L.stripStress,
                hint: L.stressTip,
                disabled: countStress(lyrics.value) === 0,
                onSelect: () => {
                    lyrics.value = stripAllStress(lyrics.value);
                    lyrics.focus();
                    commitLyrics();
                },
            },
        ], TS_UI_CLASS);
    });

    libButton.addEventListener("click", () => {
        libWrap.hidden = !libWrap.hidden;
        if (!libWrap.hidden) library.focusSearch();
    });

    vocalButton.addEventListener("click", () => {
        libWrap.hidden = false;
        library.showVocals();
    });

    // ── полноэкранный режим ──────────────────────────────────────────────
    function openEditor() {
        if (state.fullscreen?.isOpen()) return;
        container.classList.add("is-fullscreen");
        libWrap.hidden = false;
        state.fullscreen = openFullscreenOverlay(container, {
            label: L.editorTitle,
            closeTitle: L.closeEditor,
            onClose: () => {
                container.classList.remove("is-fullscreen");
                libWrap.hidden = true;
                state.fullscreen = null;
                // Контент вернулся в ноду — пусть встанет на своё место.
                shell.appendChild(container);
                node.setDirtyCanvas?.(true, true);
            },
        });
    }

    // Оболочка нужна, чтобы оверлею было куда вернуть содержимое: сам виджет
    // держит именно её, а переезжает наполнение.
    const shell = doc.createElement("div");
    shell.className = "ts-song-shell";
    shell.style.cssText = "width:100%;height:100%;min-height:0;display:flex";
    shell.appendChild(container);

    stopPropagation(shell, [
        "pointerdown", "pointerup", "mousedown", "mouseup", "wheel", "dblclick", "keydown",
    ]);

    // Штатные многострочные виджеты заменены нашими полями; `preset` остаётся
    // обычным combo — его обновляет клавиша «R» вместе со всеми остальными, и
    // свой список пришлось бы поддерживать вручную ради одной строки.
    hideWidget(node, INPUT_LYRICS);
    hideWidget(node, INPUT_STYLE);

    addResizableDomWidget(node, shell, {
        name: WIDGET_NAME,
        minWidth: MIN_NODE_WIDTH,
        minHeight: MIN_NODE_HEIGHT,
        defaultWidth: DEFAULT_NODE_WIDTH,
        defaultHeight: DEFAULT_NODE_HEIGHT,
        chromeHeight: CHROME_HEIGHT,
        minWidgetHeight: MIN_WIDGET_HEIGHT,
    });

    // ── пресет: стили и теги ─────────────────────────────────────────────
    function applyPreset(presets) {
        const preset = pickPreset(presets, state.presetName);
        state.vocals = preset?.vocals || [];
        renderTags(preset?.tags || []);
        library.setStyles(preset?.styles || []);
        library.setVocals(state.vocals, preset?.vocal_groups || []);
        library.setActive(styleField.value.trim());
    }

    function reloadPresets() {
        loadPresets().then(applyPreset);
    }

    const presetWidget = getWidget(node, INPUT_PRESET);
    if (presetWidget) {
        const previousCallback = presetWidget.callback;
        presetWidget.callback = function onPresetChanged(value, ...rest) {
            state.presetName = String(value ?? "");
            reloadPresets();
            return previousCallback?.apply(this, [value, ...rest]);
        };
    }

    // Клавиша «R» перечитывает списки — наш кэш пресетов обязан сброситься
    // вместе с остальными (§12.5.17).
    node.refreshComboInNode = () => {
        forgetPresets();
        state.presetName = String(readValue(node, INPUT_PRESET, state.presetName));
        reloadPresets();
    };

    node._tsSongCreatorRehydrate = () => {
        lyrics.value = String(readValue(node, INPUT_LYRICS, ""));
        styleField.value = String(readValue(node, INPUT_STYLE, ""));
        state.presetName = String(readValue(node, INPUT_PRESET, state.presetName));
        refreshStatus();
        reloadPresets();
    };

    const previousRemoved = node.onRemoved;
    node.onRemoved = function onRemoved(...args) {
        hideContextMenu();
        state.fullscreen?.close();
        return previousRemoved?.apply(this, args);
    };

    node._tsSongCreatorRelabel = () => {
        setOpenInterfaceLabel(openButton, getUiLanguage(), L.openTip);
    };

    node._tsSongCreatorRehydrate();
}

app.registerExtension({
    name: EXTENSION_ID,
    nodeCreated(node) {
        if (node?.comfyClass !== NODE_ID) return;
        setupSongCreator(node);
    },
    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_ID) return;
        if (!node._tsSongCreatorInitialized) {
            setupSongCreator(node);
            return;
        }
        node._tsSongCreatorRehydrate?.();
    },
    refreshComboInNodes() {
        forgetPresets();
    },
});
