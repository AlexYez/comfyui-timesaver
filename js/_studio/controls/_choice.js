// TS Studio kit — контрол «choice» (Выбор: короткий взаимоисключающий набор значений.)
//
// Один вид контрола — один файл. Манифест бэкенда объявляет контролы
// данными, реестр (`../_controls.js`) сводит вид к отрисовщику; ни один
// вид не знает о существовании других.
//
// Контракт отрисовщика: (control, ctx) -> {element, get, set}, и о каждой
// правке он сообщает через ctx.onChange(param, value).

import { deckSection } from "../_shell.js";
import { localized } from "./_shared.js";

// For settings with a handful of meaningful values (an upscale factor, say)
// where a slider would invite meaningless ones in between.
export const KIND = "choice";

export const render = (control, ctx) => {
    const section = deckSection(localized(control.label, ctx.locale, control.param));
    const row = document.createElement("div");
    row.className = "ts-studio__choice";
    const options = Array.isArray(control.options) ? control.options : [];
    const buttons = new Map();
    let value = control.default ?? options[0]?.value;

    function sync(emit = true) {
        for (const [candidate, button] of buttons) {
            button.classList.toggle("is-active", candidate === value);
        }
        if (!emit) return;
        // An option may stand for a set of values rather than one: a quality
        // preset is several numbers that only make sense together (Ideogram's
        // steps, mu and std). Those ride along, so the deck needs no separate
        // control for each and they cannot drift apart.
        const chosen = options.find((option) => option.value === value);
        for (const [param, carried] of Object.entries(chosen?.values || {})) {
            ctx.onChange(param, carried);
        }
        ctx.onChange(control.param, value);
    }

    for (const option of options) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-studio__choicebtn";
        button.textContent = localized(option.label, ctx.locale, String(option.value));
        const tip = localized(option.tooltip, ctx.locale, "");
        if (tip) button.title = tip;
        button.addEventListener("click", () => { value = option.value; sync(); });
        buttons.set(option.value, button);
        row.appendChild(button);
    }
    section.appendChild(row);
    sync();

    return {
        element: section,
        get: () => value,
        set: (next) => {
            if (!buttons.has(next)) return;      // a value this backend lacks
            value = next;
            sync();
        },
        setDisabled: (disabled) => {
            section.classList.toggle("is-disabled", Boolean(disabled));
            for (const button of buttons.values()) button.disabled = Boolean(disabled);
        },
    };
};
