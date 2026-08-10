// TS Studio kit — контрол «slider» (Ползунок: число в границах, читаемое как величина.)
//
// Один вид контрола — один файл. Манифест бэкенда объявляет контролы
// данными, реестр (`../_controls.js`) сводит вид к отрисовщику; ни один
// вид не знает о существовании других.
//
// Контракт отрисовщика: (control, ctx) -> {element, get, set}, и о каждой
// правке он сообщает через ctx.onChange(param, value).

import { deckSection } from "../_shell.js";
import { localized } from "./_shared.js";

// Same contract as "number" (get/set/setDisabled) so a manifest can swap the
// kind without the app noticing.
export const KIND = "slider";

export const render = (control, ctx) => {
    const min = Number(control.min ?? 0);
    const max = Number(control.max ?? 1);
    const step = Number(control.step ?? 0.05);
    const decimals = String(step).includes(".") ? String(step).split(".")[1].length : 0;
    // Умолчание зажимается в диапазон так же, как и любое присвоение. Без
    // этого значение вне [min, max] жило только в JS: ползунок вставал в
    // ближайший конец, подпись показывала третье число, а в граф уходило
    // четвёртое. Один раз это уже стоило часа измерений — манифест просил 60
    // при потолке 50, и молча работали старые 25.
    let value = Math.min(max, Math.max(min, Number(control.default ?? min)));

    const wrap = document.createElement("div");
    wrap.className = "ts-studio__section ts-studio__slider";
    const head = document.createElement("div");
    head.className = "ts-studio__sliderhead";
    const label = document.createElement("span");
    label.className = "ts-studio__sectionhead";
    label.textContent = localized(control.label, ctx.locale, control.param);
    const readout = document.createElement("span");
    readout.className = "ts-studio__slidervalue";
    head.append(label, readout);

    const row = document.createElement("div");
    row.className = "ts-studio__sliderrow";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "ts-ui-slider";
    slider.min = String(min);
    slider.max = String(max);
    slider.step = String(step);
    const tip = localized(control.tooltip, ctx.locale, "");
    if (tip) { slider.title = tip; label.title = tip; }
    row.appendChild(slider);
    wrap.append(head, row);

    function sync(emit = true) {
        slider.value = String(value);
        readout.textContent = value.toFixed(decimals);
        if (emit) ctx.onChange(control.param, value);
    }
    slider.addEventListener("input", () => { value = Number(slider.value); sync(); });
    sync();

    return {
        element: wrap,
        get: () => value,
        set: (next) => {
            const parsed = Number(next);
            if (!Number.isFinite(parsed)) return;
            value = Math.min(max, Math.max(min, parsed));
            sync();
        },
        setDisabled: (disabled) => {
            slider.disabled = Boolean(disabled);
            wrap.classList.toggle("is-disabled", Boolean(disabled));
        },
    };
};
