// TS Studio kit — контрол «toggle» (Переключатель: да или нет.)
//
// Один вид контрола — один файл. Манифест бэкенда объявляет контролы
// данными, реестр (`../_controls.js`) сводит вид к отрисовщику; ни один
// вид не знает о существовании других.
//
// Контракт отрисовщика: (control, ctx) -> {element, get, set}, и о каждой
// правке он сообщает через ctx.onChange(param, value).

import { deckSection } from "../_shell.js";

export const KIND = "toggle";

export const render = (control, ctx) => {
    const row = document.createElement("div");
    row.className = "ts-studio__numrow";
    const label = document.createElement("span");
    label.textContent = (control.label?.[ctx.locale]) || control.label?.en || control.param;
    // A switch rather than an ON/OFF caption: the state reads at a glance and
    // the control stops shouting two letters in the middle of the deck.
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ts-studio__switch";
    button.setAttribute("role", "switch");
    const knob = document.createElement("span");
    knob.className = "ts-studio__switchknob";
    button.appendChild(knob);
    const tip = control.tooltip?.[ctx.locale] || control.tooltip?.en;
    if (tip) { row.title = tip; button.title = tip; }
    let value = Boolean(control.default);
    function sync() {
        button.classList.toggle("is-on", value);
        button.setAttribute("aria-checked", String(value));
        ctx.onChange(control.param, value);
    }
    button.addEventListener("click", () => { value = !value; sync(); });
    row.append(label, button);
    sync();
    return {
        element: row,
        get: () => value,
        set: (v) => { value = Boolean(v); sync(); },
    };
};
