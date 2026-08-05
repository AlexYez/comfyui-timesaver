// TS Studio kit — контрол «strength» (Сила: одна шкала вместо ползунка и отдельного тумблера.)
//
// Один вид контрола — один файл. Манифест бэкенда объявляет контролы
// данными, реестр (`../_controls.js`) сводит вид к отрисовщику; ни один
// вид не знает о существовании других.
//
// Контракт отрисовщика: (control, ctx) -> {element, get, set}, и о каждой
// правке он сообщает через ctx.onChange(param, value).

import { deckSection } from "../_shell.js";
import { localized } from "./_shared.js";

//
// У перерисовки ровно две задачи, и между ними нет плавного перехода: либо
// доработать то, что есть, либо заменить область целиком. Раньше это были два
// органа управления — ползунок силы и тумблер Replace, — и человеку нужно было
// помнить, что при включённом тумблере ползунок ничего не значит.
//
// Здесь это одна шкала с подписанными ступенями: 10, 20, 30, 45 и Replace.
// Наверх контрол отдаёт одно число — силу; режим замены из него выводится
// (см. REPLACE_DENOISE в nodes/image/_inpaint_crop.py и его зеркало в
// _crop_geometry.js). Ступени неслучайны: до 45% модель дорабатывает пиксели,
// выше — начинает выдумывать, и промежуточные значения там только вводят в
// заблуждение.
export const KIND = "strength";

export const render = (control, ctx) => {
    const stops = Array.isArray(control.stops) && control.stops.length
        ? control.stops.map(Number) : [0.1, 0.2, 0.3, 0.45, 1.0];
    const replaceAt = Number(control.replace_at ?? 0.6);
    const labelFor = (value) => (value >= replaceAt
        ? (localized(control.replace_label, ctx.locale, "") || "Replace")
        : `${Math.round(value * 100)}%`);

    let index = Math.max(0, stops.findIndex(
        (v) => Math.abs(v - Number(control.default ?? stops[0])) < 1e-6));

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
    slider.min = "0";
    slider.max = String(stops.length - 1);
    slider.step = "1";
    const tip = localized(control.tooltip, ctx.locale, "");
    if (tip) { slider.title = tip; label.title = tip; }
    row.appendChild(slider);

    const ticks = document.createElement("div");
    ticks.className = "ts-studio__ticks";
    const tickNodes = stops.map((value, at) => {
        const tick = document.createElement("span");
        tick.className = "ts-studio__tick";
        tick.textContent = labelFor(value);
        tick.title = tip;
        tick.addEventListener("click", () => { index = at; sync(); });
        ticks.appendChild(tick);
        return tick;
    });
    wrap.append(head, row, ticks);

    function sync(emit = true) {
        slider.value = String(index);
        readout.textContent = labelFor(stops[index]);
        tickNodes.forEach((tick, at) => tick.classList.toggle("is-active", at === index));
        if (emit) ctx.onChange(control.param, stops[index]);
    }
    slider.addEventListener("input", () => { index = Number(slider.value); sync(); });
    sync();

    return {
        element: wrap,
        get: () => stops[index],
        set: (next) => {
            const parsed = Number(next);
            if (!Number.isFinite(parsed)) return;
            // Ближайшая ступень: сохранённое значение может быть из прошлой
            // раскладки шкалы, и оно не должно ни пропадать, ни промахиваться.
            let best = 0;
            for (let at = 1; at < stops.length; at += 1) {
                if (Math.abs(stops[at] - parsed) < Math.abs(stops[best] - parsed)) best = at;
            }
            index = best;
            sync();
        },
        setDisabled: (disabled) => {
            slider.disabled = Boolean(disabled);
            wrap.classList.toggle("is-disabled", Boolean(disabled));
        },
    };
};
