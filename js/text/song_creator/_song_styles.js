/**
 * Библиотека стилей: загрузка пресетов и панель выбора.
 *
 * Пресеты приходят с бэкенда (`/ts_song_creator/presets`), потому что живут
 * файлами в `nodes/text/song_presets/` — новая модель добавляется файлом, и
 * фронтенд про неё узнаёт сам, без правки кода здесь.
 *
 * ⚠️ Человеку показывается ПЕРЕВЕДЁННОЕ название стиля, а в поле подставляется
 * английский `prompt`. Это не небрежность локализации: `prompt` — вход модели,
 * она обучалась на этих словах, и «воодушевляющий поп» вместо «uplifting pop»
 * ломает обусловливание, а не переводит его.
 */

import { api } from "/scripts/api.js";

import { contains } from "./_song_compose.js";

/** Один запрос на страницу: список стилей одинаков для всех нод. */
let presetsPromise = null;

/**
 * Пресеты с сервера. Ошибка сети не роняет ноду — вернётся пустой список.
 *
 * @returns {Promise<Array<object>>}
 */
export function loadPresets() {
    if (!presetsPromise) {
        presetsPromise = api.fetchApi("/ts_song_creator/presets")
            .then((response) => (response.ok ? response.json() : { presets: [] }))
            .then((payload) => (Array.isArray(payload?.presets) ? payload.presets : []))
            .catch((error) => {
                console.error("[TS Song Creator] presets failed to load:", error);
                return [];
            });
    }
    return presetsPromise;
}

/** Сбросить кэш — нужен клавише «R» (refreshComboInNodes). */
export function forgetPresets() {
    presetsPromise = null;
}

/**
 * Выбрать пресет по имени; при незнакомом имени — первый доступный.
 *
 * @param {Array<object>} presets что отдал сервер.
 * @param {string} name имя из виджета ноды.
 * @returns {object|null}
 */
export function pickPreset(presets, name) {
    if (!Array.isArray(presets) || !presets.length) return null;
    const wanted = String(name || "").trim().toLowerCase();
    return presets.find((item) => String(item?.name || "").toLowerCase() === wanted)
        || presets[0];
}

/**
 * Панель библиотеки: раздел «Стиль» и раздел «Вокал» под одним поиском.
 *
 * ⚠️ Вокал — второе измерение, а не подпункт жанра: голос человек меняет
 * независимо. Сначала пол (женский / мужской / прочее), внутри — виды и
 * нюансы, потому что «мужской или женский» решается первым, а «хрипловатый
 * или мягкий» — уже внутри выбранного.
 *
 * @param {object} options
 * @param {(prompt:string, style:object) => void} options.onPick выбран жанр.
 * @param {(prompt:string, vocal:object) => void} options.onPickVocal выбран голос.
 * @param {(key:string) => string} options.t переводчик подписей.
 * @param {string} options.lang текущий язык интерфейса.
 * @returns {{element:HTMLElement, setStyles:Function, setVocals:Function,
 *   setActive:Function, focusSearch:Function}}
 */
export function createStyleLibrary({ onPick, onPickVocal, t, lang }) {
    const doc = document;
    const element = doc.createElement("div");
    element.className = "ts-song-lib";

    const search = doc.createElement("input");
    search.type = "search";
    search.className = "ts-ui-input ts-song-lib__search";
    search.placeholder = t("searchStyles");
    search.title = t("searchStylesTip");

    // Две вкладки вместо одного длинного списка: держать тридцать жанров и
    // два десятка голосов в одной простыне — значит листать её каждый раз.
    const tabs = doc.createElement("div");
    tabs.className = "ts-song-lib__tabs";
    const styleTab = doc.createElement("button");
    styleTab.type = "button";
    styleTab.className = "ts-song-lib__tab is-active";
    styleTab.textContent = t("tabStyle");
    const vocalTab = doc.createElement("button");
    vocalTab.type = "button";
    vocalTab.className = "ts-song-lib__tab";
    vocalTab.textContent = t("tabVocal");
    tabs.append(styleTab, vocalTab);

    // Пол — переключатель над списком голосов.
    const groupRow = doc.createElement("div");
    groupRow.className = "ts-song-lib__groups";
    groupRow.hidden = true;

    const list = doc.createElement("div");
    list.className = "ts-song-lib__list";

    const empty = doc.createElement("div");
    empty.className = "ts-song-lib__empty";
    empty.textContent = t("noStyles");
    empty.hidden = true;

    element.append(tabs, search, groupRow, list, empty);

    let allStyles = [];
    let allVocals = [];
    let vocalGroups = [];
    let activePrompt = "";
    let tab = "style";
    let group = "";

    function label(style) {
        const name = style?.name || {};
        return String(name[lang] || name.en || style?.id || "");
    }

    function hint(style) {
        const text = style?.hint || {};
        return String(text[lang] || text.en || "");
    }

    function renderGroups() {
        groupRow.textContent = "";
        for (const item of vocalGroups) {
            const button = doc.createElement("button");
            button.type = "button";
            button.className = "ts-song-lib__group";
            if (item.id === group) button.classList.add("is-active");
            const name = item?.name || {};
            button.textContent = String(name[lang] || name.en || item.id);
            button.addEventListener("click", () => {
                group = item.id;
                renderGroups();
                render();
            });
            groupRow.appendChild(button);
        }
    }

    function card(item, onSelect) {
        const button = doc.createElement("button");
        button.type = "button";
        button.className = "ts-song-lib__card";
        if (contains(activePrompt, item.prompt)) button.classList.add("is-active");
        button.title = hint(item) || item.prompt;

        const title = doc.createElement("span");
        title.className = "ts-song-lib__name";
        title.textContent = label(item);

        const note = doc.createElement("span");
        note.className = "ts-song-lib__hint";
        note.textContent = hint(item);

        button.append(title, note);
        button.addEventListener("click", () => onSelect(item));
        return button;
    }

    function render() {
        const needle = search.value.trim().toLowerCase();
        const vocalMode = tab === "vocal";
        groupRow.hidden = !vocalMode;
        list.textContent = "";

        const source = vocalMode
            ? allVocals.filter((item) => !group || item.group === group)
            : allStyles;

        let shown = 0;
        for (const item of source) {
            // Ищем и по переводу, и по английскому промпту: человек может
            // помнить «trap», а видеть «Трэп».
            const haystack = `${label(item)} ${hint(item)} ${item.prompt} ${item.id}`
                .toLowerCase();
            if (needle && !haystack.includes(needle)) continue;
            shown += 1;
            list.appendChild(card(item, vocalMode
                ? (picked) => onPickVocal?.(picked.prompt, picked)
                : (picked) => onPick?.(picked.prompt, picked)));
        }
        empty.hidden = shown > 0;
    }

    function selectTab(name) {
        tab = name;
        styleTab.classList.toggle("is-active", name === "style");
        vocalTab.classList.toggle("is-active", name === "vocal");
        render();
    }

    styleTab.addEventListener("click", () => selectTab("style"));
    vocalTab.addEventListener("click", () => selectTab("vocal"));
    search.addEventListener("input", render);

    return {
        element,
        setStyles(styles) {
            allStyles = Array.isArray(styles) ? styles : [];
            render();
        },
        setVocals(vocals, groups) {
            allVocals = Array.isArray(vocals) ? vocals : [];
            vocalGroups = Array.isArray(groups) && groups.length
                ? groups
                : [...new Set(allVocals.map((item) => item.group))]
                    .map((id) => ({ id, name: { en: id, ru: id } }));
            if (!vocalGroups.some((item) => item.id === group)) {
                group = vocalGroups[0]?.id || "";
            }
            renderGroups();
            render();
        },
        /** Подсветить карточки, чьи слова сейчас в поле. */
        setActive(prompt) {
            activePrompt = String(prompt || "");
            render();
        },
        showVocals() {
            selectTab("vocal");
        },
        focusSearch() {
            search.focus();
            search.select?.();
        },
    };
}
