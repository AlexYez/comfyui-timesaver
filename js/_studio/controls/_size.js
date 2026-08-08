// TS Studio kit — контрол «size» (Размер кадра: карточки пропорций и ползунок мегапикселей.)
//
// Один вид контрола — один файл. Манифест бэкенда объявляет контролы
// данными, реестр (`../_controls.js`) сводит вид к отрисовщику; ни один
// вид не знает о существовании других.
//
// Контракт отрисовщика: (control, ctx) -> {element, get, set}, и о каждой
// правке он сообщает через ctx.onChange(param, value).

import { createRatioPicker, parseRatioText } from "../../_theme.js";
import { deckSection } from "../_shell.js";

function parseAspect(text) {
    return parseRatioText(text)?.ratio ?? 1;
}

export function sizeFromAspect(aspectText, megapixels, snap) {
    const ratio = parseAspect(aspectText);
    const pixels = megapixels * 1e6;
    const rawWidth = Math.sqrt(pixels * ratio);
    const rawHeight = rawWidth / ratio;
    const step = Math.max(1, snap | 0);
    const width = Math.max(step, Math.round(rawWidth / step) * step);
    const height = Math.max(step, Math.round(rawHeight / step) * step);
    return { width, height };
}

export const KIND = "size";

export const render = (control, ctx) => {
    const aspects = control.aspects || ["1:1"];
    const state = {
        aspect: control.default_aspect || aspects[0],
        mp: Number(control.base_mp || 1.0),
    };
    const [mpMin, mpMax] = control.mp || [0.25, 4.0];
    const snap = Number(control.snap || 16);

    const section = deckSection(ctx.t.format);
    section.classList.add("ts-studio__size");
    // The same cards TS Resolution Selector draws, in the compact form
    // (`createRatioPicker` in js/_theme.js): a trigger showing the chosen shape,
    // the 3x3 grid only while it is open, and the "w:h" field to its right. Nine
    // cards standing open cost more of this panel's one column of vertical room
    // than they are worth — the node, which can be resized, keeps them open.
    const picker = createRatioPicker({
        values: aspects,
        onSelect: (aspect) => { state.aspect = aspect; sync(); },
        onCustom: (aspect) => { state.aspect = aspect; sync(); },
        customPlaceholder: ctx.t.aspectCustom,
        customTitle: ctx.t.aspectCustomTip,
    });
    const grid = document.createElement("div");
    grid.className = "ts-studio__aspects";
    grid.appendChild(picker.element);
    section.appendChild(grid);

    const resTitle = deckSection(ctx.t.resolution);
    const row = document.createElement("div");
    row.className = "ts-studio__sizerow";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "ts-ui-slider";
    slider.min = String(mpMin);
    slider.max = String(mpMax);
    slider.step = "0.05";
    slider.value = String(state.mp);
    slider.addEventListener("input", () => { state.mp = Number(slider.value); sync(); });
    row.appendChild(slider);
    const info = document.createElement("div");
    info.className = "ts-studio__sizeinfo";
    const mpText = document.createElement("span");
    const whText = document.createElement("span");
    info.append(mpText, whText);
    resTitle.append(row, info);
    section.appendChild(resTitle);

    function sync() {
        picker.select(state.aspect);
        const { width, height } = sizeFromAspect(state.aspect, state.mp, snap);
        mpText.textContent = `${state.mp.toFixed(2)} MP`;
        whText.textContent = `${width} × ${height}`;
        ctx.onChange(control.width_param || "width", width);
        ctx.onChange(control.height_param || "height", height);
    }
    sync();

    const note = document.createElement("div");
    note.className = "ts-studio__sizenote";
    note.style.display = "none";
    note.textContent = ctx.t.sizeFromReference;
    section.appendChild(note);

    return {
        element: section,
        get: () => ({ aspect: state.aspect, mp: state.mp }),
        // Edit runs inherit the reference's frame, so the picker would be
        // lying if it stayed live: grey it out and say why.
        setDisabled: (disabled) => {
            const off = Boolean(disabled);
            section.classList.toggle("is-disabled", off);
            slider.disabled = off;
            picker.setDisabled(off);
            note.style.display = off ? "" : "none";
        },
        set: (value) => {
            // A carried-over aspect only applies if this model offers it;
            // otherwise the model's own default stands.
            if (value?.aspect && picker.has(value.aspect)) state.aspect = value.aspect;
            if (value?.mp) {
                state.mp = Math.min(mpMax, Math.max(mpMin, Number(value.mp)));
                slider.value = String(state.mp);
            }
            sync();
        },
    };
};
