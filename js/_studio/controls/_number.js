// TS Studio kit — контрол «number» (Число: точное значение, когда шкала не нужна.)
//
// Один вид контрола — один файл. Манифест бэкенда объявляет контролы
// данными, реестр (`../_controls.js`) сводит вид к отрисовщику; ни один
// вид не знает о существовании других.
//
// Контракт отрисовщика: (control, ctx) -> {element, get, set}, и о каждой
// правке он сообщает через ctx.onChange(param, value).

import { deckSection } from "../_shell.js";
import { localized } from "./_shared.js";

export const KIND = "number";

export const render = (control, ctx) => {
    const row = document.createElement("div");
    row.className = "ts-studio__numrow";
    const label = document.createElement("span");
    label.textContent = localized(control.label, ctx.locale, control.param);
    const field = document.createElement("input");
    field.type = "number";
    field.className = "ts-ui-input";
    if (control.min !== undefined) field.min = String(control.min);
    if (control.max !== undefined) field.max = String(control.max);
    field.step = String(control.step ?? 1);
    field.addEventListener("input", () => ctx.onChange(control.param, Number(field.value)));
    row.append(label, field);
    return {
        element: row,
        get: () => Number(field.value),
        set: (value) => { field.value = String(value); ctx.onChange(control.param, Number(value)); },
        setDisabled: (disabled) => { field.disabled = Boolean(disabled); },
    };
};
