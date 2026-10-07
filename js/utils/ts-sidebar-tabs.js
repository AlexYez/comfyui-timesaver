// Левый сайдбар ComfyUI: какие вкладки показывать.
//
// Просьба владельца пака (07.10.2026): в настройках, в разделе пака, по
// галочке на каждую вкладку левого меню — «Ассеты», «Библиотека нод»,
// «Модели», «Рабочие процессы», «Приложения» и вкладки других паков, — чтобы
// лишнее можно было убрать. Кнопки без панели («Шаблоны», «Центр поддержки»,
// «Настройки» и прочие внизу) НЕ трогаются — так решил владелец.
//
// Как: публичное API рабочей области —
// `app.extensionManager.unregisterSidebarTab(id)` / `registerSidebarTab(tab)`.
//
// ⚠️ Вернуть вкладку на место нельзя одним вызовом: `registerSidebarTab`
// дописывает её В КОНЕЦ (замерено на 1.53.10: model-library после
// unregister+register уехала за Artius Browser). Поэтому порядок, в котором
// вкладки были при первом знакомстве, запоминается, и при каждом изменении
// набора неправильно стоящий «хвост» снимается и ставится заново в этом
// порядке. Голова, стоящая верно, не трогается — её панели не пересоздаются.
//
// ⚠️ Вкладки других паков регистрируются в их собственном setup(), порядок
// которого не наш. Узнаём о них подпиской на хранилище вкладок (`$subscribe` —
// публичное API Pinia), а не опросом по таймеру: тогда галочка появляется в
// момент появления вкладки, в том числе через минуту после загрузки.

import { app } from "/scripts/app.js";

import { pickLocaleStrings } from "../_theme.js";

// Only the switches of OTHER packs' tabs need this: they are made at run time,
// so locales/ru/settings.json cannot know their ids. The core switches are
// translated there.
const STRINGS = {
    en: {
        show: (name) => `Show «${name}»`,
        extraTip: "A tab added by another extension. Off: it leaves the left sidebar "
            + "(its panel closes). On: it comes back in its place.",
    },
    ru: {
        show: (name) => `Показывать «${name}»`,
        extraTip: "Вкладка другого расширения. Выключено — уходит из левого меню "
            + "(её панель закрывается). Включено — возвращается на своё место.",
    },
};

const PREFIX = "TS.Sidebar.Show.";
/** sortOrder of the first switch; the rest count down from it. */
const SORT_BASE = 1000;
/**
 * Settings category of one switch.
 *
 * ⚠️ The LAST element must be unique per setting: the settings dialog builds
 * its tree keyed by the whole category path, and settings that share one path
 * overwrite each other — measured on 1.53.10, five switches sharing
 * ["TS Timesaver", "Interface", "Left sidebar items"] showed up as ONE (the last
 * added). The element itself is not displayed; the setting's name is.
 */
function categoryFor(id) {
    return ["TS Timesaver", "Interface", `Left sidebar: ${id}`];
}

/** Core tabs, known up front so their switches exist (and are translated) from the start. */
const CORE_TABS = [
    { id: "assets", name: "Assets" },
    { id: "node-library", name: "Node library" },
    { id: "model-library", name: "Model library" },
    { id: "workflows", name: "Workflows" },
    { id: "apps", name: "Apps" },
];

const state = {
    ready: false,
    scheduled: false,
    /** id → the tab object as the frontend registered it (needed to put it back). */
    known: new Map(),
    /** ids in the order they were first seen — the order put back on every change. */
    order: [],
    /** setting ids already declared, so a late tab is never declared twice. */
    declared: new Set(CORE_TABS.map((tab) => PREFIX + tab.id)),
};

function workspace() {
    return app.extensionManager;
}

function isShown(id) {
    // An unknown or never-touched setting reads as undefined: shown, like before.
    return workspace()?.setting?.get?.(PREFIX + id) !== false;
}

function currentTabs() {
    try {
        return workspace()?.getSidebarTabs?.() || [];
    } catch {
        return [];
    }
}

/** Remember every tab seen, in order; declare a switch for tabs of other packs. */
function learnTabs() {
    for (const tab of currentTabs()) {
        if (!tab?.id || state.known.has(tab.id)) continue;
        state.known.set(tab.id, tab);
        state.order.push(tab.id);
        declareExtraTab(tab);
    }
}

function labelOf(tab) {
    const text = [tab.title, tab.label, tab.tooltip].find((value) => typeof value === "string"
        && value.trim() && !/^[\w]+\.[\w.]+$/.test(value.trim()));  // skip raw i18n keys
    return text || tab.id;
}

/** A tab of another pack (unknown up front) gets its switch the moment it shows up. */
function declareExtraTab(tab) {
    const id = PREFIX + tab.id;
    if (state.declared.has(id)) return;
    state.declared.add(id);
    const t = pickLocaleStrings(STRINGS);
    try {
        app.ui.settings.addSetting({
            id,
            category: categoryFor(tab.id),
            sortOrder: SORT_BASE - CORE_TABS.length - state.order.length,
            name: t.show(labelOf(tab)),
            tooltip: t.extraTip,
            type: "boolean",
            defaultValue: true,
            onChange: () => schedule(),
        });
    } catch (error) {
        console.warn("[TS SidebarTabs] could not add a switch for", tab.id, error);
    }
}

/** Put the visible tabs back in their remembered order; leave alone what is already right. */
function apply() {
    state.scheduled = false;
    // onChange fires while settings load, before the workspace has its tabs;
    // setup() applies once everything is there.
    if (!state.ready) return;
    const manager = workspace();
    if (!manager?.registerSidebarTab || !manager?.unregisterSidebarTab) return;
    learnTabs();

    const present = currentTabs().map((tab) => tab.id).filter((id) => state.known.has(id));
    const wanted = state.order.filter((id) => isShown(id));
    // Idempotent: when everything already stands right nothing is touched — which
    // is also what ends the loop through our own store subscription.
    if (present.length === wanted.length && present.every((id, i) => id === wanted[i])) return;

    // An open panel whose tab is about to go is closed first, or it would stay
    // on screen with no button to close it.
    const sidebar = manager.sidebarTab;
    const active = sidebar?.activeSidebarTabId;
    if (active && !wanted.includes(active)) {
        try { sidebar.toggleSidebarTab(active); } catch { /* panel already gone */ }
    }

    let keep = 0;
    while (keep < present.length && keep < wanted.length && present[keep] === wanted[keep]) keep += 1;
    for (const id of present.slice(keep)) {
        try { manager.unregisterSidebarTab(id); } catch { /* already gone */ }
    }
    for (const id of wanted.slice(keep)) {
        try { manager.registerSidebarTab(state.known.get(id)); } catch (error) {
            console.warn("[TS SidebarTabs] could not restore", id, error);
        }
    }
}

/** One apply per burst of changes (a settings save or a pack registering its tab). */
function schedule() {
    if (state.scheduled) return;
    state.scheduled = true;
    setTimeout(apply, 0);
}

function switchFor(tab, index) {
    return {
        id: PREFIX + tab.id,
        category: categoryFor(tab.id),
        // The dialog lists higher sortOrder first: the switches read top to
        // bottom in the sidebar's own order (measured: without it, reversed).
        sortOrder: SORT_BASE - index,
        name: `Show «${tab.name}»`,
        tooltip: "Off: the tab leaves the left sidebar (its panel closes). "
            + "On: it comes back in its place.",
        type: "boolean",
        defaultValue: true,
        onChange: () => schedule(),
    };
}

app.registerExtension({
    name: "ts.sidebarTabs",
    settings: CORE_TABS.map(switchFor),
    async setup() {
        state.ready = true;
        try {
            workspace()?.sidebarTab?.$subscribe?.(() => schedule());
        } catch (error) {
            console.warn("[TS SidebarTabs] could not watch the sidebar", error);
        }
        apply();
    },
});
