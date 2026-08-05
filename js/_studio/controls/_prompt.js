// TS Studio kit — контрол «prompt» (Промпт: текст, который человек пишет, и слот под панель инструментов.)
//
// Один вид контрола — один файл. Манифест бэкенда объявляет контролы
// данными, реестр (`../_controls.js`) сводит вид к отрисовщику; ни один
// вид не знает о существовании других.
//
// Контракт отрисовщика: (control, ctx) -> {element, get, set}, и о каждой
// правке он сообщает через ctx.onChange(param, value).

import { deckSection } from "../_shell.js";
import { localized } from "./_shared.js";

export const KIND = "prompt";

export const render = (control, ctx) => {
    const section = deckSection(localized(control.label, ctx.locale,
        control.param === "negative_prompt" ? ctx.t.negativePrompt : ctx.t.prompt));
    const wrap = document.createElement("div");
    wrap.className = "ts-studio__prompt";
    const area = document.createElement("textarea");
    area.className = "ts-ui-textarea";
    area.placeholder = ctx.t.promptPlaceholder;
    area.addEventListener("input", () => ctx.onChange(control.param, area.value));
    wrap.appendChild(area);
    // The prompt toolbar (voice, image, presets, styles) mounts here in the
    // media-combine phase; the slot exists from day one.
    const toolbarSlot = document.createElement("div");
    toolbarSlot.dataset.tsSlot = `prompt-toolbar:${control.param}`;
    wrap.appendChild(toolbarSlot);
    section.appendChild(wrap);
    return {
        element: section,
        get: () => area.value,
        // Setting a control is a value change like any other — every other
        // kind reports one from its set(). The prompt must too: the deck is
        // rebuilt on every model and mode switch, and the text carried across
        // is restored through here. Staying silent would leave the words
        // visible in the box while the run went out without them.
        set: (value) => {
            area.value = String(value ?? "");
            ctx.onChange(control.param, area.value);
        },
    };
};
