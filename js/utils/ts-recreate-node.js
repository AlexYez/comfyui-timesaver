// «Пересоздать ноду» — пункт меню правой кнопки у ЛЮБОЙ ноды.
//
// Нода встаёт заново, как только что поставленная из меню: все виджеты на
// умолчаниях, название, цвет, размер и свойства — тоже. Остаются место на
// холсте, номер ноды (ссылки на него в графе не едут), режим (байпас/выкл. —
// это состояние графа, а не настройка ноды) и ВСЕ провода, включая точки на
// них. Лечит ноду, сохранённую старой версией пака, и просто сбрасывает
// накрученное.
//
// Это НЕ нода — работа холста, как TS Tidy Layout и значок байпаса на группе.
// Крючок штатный (`getNodeMenuItems`), плюс команда в палитре.
//
// ⚠️ Провода не теряются ни при каком исходе. Связь ищется у новой ноды по
// имени гнезда, затем — тем же номером и типом (вход переименован в новой
// версии). Не нашлась хоть одна — нода возвращается как была, со своими
// значениями и проводами, а человек получает список того, что не сошлось.
//
// ⚠️ Входы подключаются по порядку: у Autogrow-входов (images.image_1, _2 …)
// следующее гнездо появляется, только когда занято предыдущее.

import { app } from "/scripts/app.js";

import { pickLocaleStrings } from "../_theme.js";
import { canRecreate, matchSlot, snapshotLinks } from "./_recreate_node.js";

const COMMAND = "TS.RecreateNode";
const LOG_PREFIX = "[TS RecreateNode]";

const STRINGS = {
    en: {
        menu: "Recreate node",
        menuMany: (n) => `Recreate ${n} nodes`,
        title: "Recreate selected nodes (defaults, wires kept)",
        nothing: "Select a node to recreate.",
        done: (n) => (n === 1 ? "Node recreated with default settings, wires kept."
            : `${n} nodes recreated with default settings, wires kept.`),
        failed: (title, names) => `«${title}» left as it was: the current version of the node has no `
            + `matching socket for ${names}.`,
        error: "Recreating failed — the node was left as it was. See the browser console.",
    },
    ru: {
        menu: "Пересоздать ноду",
        menuMany: (n) => `Пересоздать ноды: ${n}`,
        title: "Пересоздать выбранные ноды (умолчания, провода сохраняются)",
        nothing: "Выберите ноду, которую пересоздать.",
        done: (n) => (n === 1 ? "Нода пересоздана с настройками по умолчанию, провода на месте."
            : `Пересоздано нод: ${n}. Настройки по умолчанию, провода на месте.`),
        failed: (title, names) => `«${title}» оставлена как была: у нынешней версии ноды нет `
            + `подходящего гнезда для ${names}.`,
        error: "Пересоздать не удалось — нода оставлена как была. Подробности в консоли браузера.",
    },
};

function toast(severity, detail) {
    try {
        app.extensionManager?.toast?.add?.({ severity, detail, life: severity === "success" ? 2500 : 6000 });
    } catch (error) {
        console.warn(`${LOG_PREFIX} toast failed`, error);
    }
}

const registered = () => globalThis.LiteGraph?.registered_node_types || {};

function linkGetter(graph) {
    return (id) => (graph.links instanceof Map ? graph.links.get(id) : graph.links?.[id]) || null;
}

/** Подключить снимок к ноде `target`. Возвращает то, что не сошлось. */
function attach(graph, target, { incoming, outgoing }, exact = false) {
    const missed = [];
    // По порядку слотов — ради Autogrow.
    for (const item of [...incoming].sort((a, b) => a.slot - b.slot)) {
        const origin = graph.getNodeById(item.originId);
        const index = exact ? item.slot
            : matchSlot(target.inputs, item, (i) => target.inputs[i]?.link == null);
        if (!origin || index < 0 || !origin.connect(item.originSlot, target, index, item.parentId ?? undefined)) {
            missed.push(item);
        }
    }
    for (const item of outgoing) {
        const destination = graph.getNodeById(item.targetId);
        const index = exact ? item.slot : matchSlot(target.outputs, item);
        if (!destination || index < 0
            || !target.connect(index, destination, item.targetSlot, item.parentId ?? undefined)) {
            missed.push(item);
        }
    }
    return missed;
}

/** Вернуть ноду как была: её же значения, слоты и провода. */
function restore(graph, saved, links) {
    const back = globalThis.LiteGraph.createNode(saved.type);
    back.configure({
        ...saved,
        inputs: (saved.inputs || []).map((input) => ({ ...input, link: null })),
        outputs: (saved.outputs || []).map((output) => ({ ...output, links: [] })),
    });
    back.id = saved.id;
    graph.add(back);
    const lost = attach(graph, back, links, true);
    if (lost.length) console.error(`${LOG_PREFIX} could not restore ${lost.length} wire(s) of node ${saved.id}`, lost);
    return back;
}

/**
 * Пересоздать одну ноду. Возвращает {node} или {missed} — тогда нода
 * возвращена как была.
 */
function recreateOne(old) {
    const graph = old.graph;
    const links = snapshotLinks(old, linkGetter(graph));
    const saved = old.serialize();
    const { id, mode } = old;
    const pos = [old.pos[0], old.pos[1]];

    const fresh = globalThis.LiteGraph.createNode(old.type);
    if (!fresh) throw new Error(`node type ${old.type} cannot be created`);

    graph.remove(old);
    fresh.id = id;
    fresh.pos = pos;
    fresh.mode = mode;
    graph.add(fresh);

    let missed;
    try {
        missed = attach(graph, fresh, links);
    } catch (error) {
        graph.remove(fresh);
        restore(graph, saved, links);
        throw error;
    }
    if (missed.length) {
        graph.remove(fresh);
        restore(graph, saved, links);
        return { missed };
    }
    return { node: fresh };
}

function targets(canvas, node) {
    const selected = Object.values(canvas?.selected_nodes || {});
    // Правый клик по ноде из выделения — пересоздаётся всё выделение.
    const list = node && !selected.includes(node) ? [node] : selected.length ? selected : node ? [node] : [];
    return list.filter((item) => canRecreate(item, registered()));
}

function recreate(canvas, node = null) {
    const t = pickLocaleStrings(STRINGS);
    const list = targets(canvas, node);
    if (!list.length) {
        toast("info", t.nothing);
        return;
    }
    canvas?.emitBeforeChange?.();
    const made = [];
    try {
        for (const item of list) {
            const title = item.title || item.type;
            try {
                const result = recreateOne(item);
                if (result.node) {
                    made.push(result.node);
                    continue;
                }
                const names = result.missed.map((m) => `«${m.name}»`).join(", ");
                toast("warn", t.failed(title, names));
            } catch (error) {
                console.error(`${LOG_PREFIX} ${item.type} failed`, error);
                toast("error", t.error);
            }
        }
    } finally {
        canvas?.emitAfterChange?.();
    }
    if (made.length) {
        canvas?.deselectAll?.();
        for (const fresh of made) canvas?.select?.(fresh);
        toast("success", t.done(made.length));
    }
    canvas?.setDirty?.(true, true);
}

app.registerExtension({
    name: "ts.recreateNode",

    commands: [
        {
            id: COMMAND,
            // Подпись читается один раз: смена языка перезагружает страницу.
            label: pickLocaleStrings(STRINGS).title,
            icon: "pi pi-refresh",
            function: () => recreate(app.canvas),
        },
    ],

    getNodeMenuItems(node) {
        if (!canRecreate(node, registered())) return [];
        const t = pickLocaleStrings(STRINGS);
        const count = targets(app.canvas, node).length;
        return [{
            content: count > 1 ? t.menuMany(count) : t.menu,
            callback: () => recreate(app.canvas, node),
        }];
    },
});
