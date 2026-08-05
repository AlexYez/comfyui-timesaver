// TS Studio kit — контрол «seed» (Сид: значение вручную или новый на каждый прогон.)
//
// Один вид контрола — один файл. Манифест бэкенда объявляет контролы
// данными, реестр (`../_controls.js`) сводит вид к отрисовщику; ни один
// вид не знает о существовании других.
//
// Контракт отрисовщика: (control, ctx) -> {element, get, set}, и о каждой
// правке он сообщает через ctx.onChange(param, value).

import { deckSection } from "../_shell.js";

export function randomSeed() {
    return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
}

// Same grid and weight as the rail icons. The die shows a real five-pip face
// (the three-pip diagonal read as a digit); "a new seed every run" is a cycle
// rather than a shuffle, whose crossing arrows turned to noise at 14px.
const SEED_ICON_ATTRS = 'viewBox="0 0 24 24" width="14" height="14" fill="none" '
    + 'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
    + 'stroke-linejoin="round"';

const SEED_ICONS = {
    dice: `<svg ${SEED_ICON_ATTRS}><rect x="4" y="4" width="16" height="16" rx="3.2"/><circle cx="8.6" cy="8.6" r="1.15" fill="currentColor" stroke="none"/><circle cx="15.4" cy="8.6" r="1.15" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.15" fill="currentColor" stroke="none"/><circle cx="8.6" cy="15.4" r="1.15" fill="currentColor" stroke="none"/><circle cx="15.4" cy="15.4" r="1.15" fill="currentColor" stroke="none"/></svg>`,
    shuffle: `<svg ${SEED_ICON_ATTRS}><path d="M4.8 12a7.2 7.2 0 0 1 12.3-5.1"/><path d="M19.2 12a7.2 7.2 0 0 1-12.3 5.1"/><path d="M17.6 3.4v3.6H14"/><path d="M6.4 20.6V17H10"/></svg>`,
    lock: `<svg ${SEED_ICON_ATTRS}><rect x="5" y="10.5" width="14" height="9.5" rx="2.2"/><path d="M8.2 10.5V7.4a3.8 3.8 0 0 1 7.6 0v3.1"/></svg>`,
};

// Seed is two decisions, so it gets two explicit controls: the mode (a new
// seed every run vs the one in the field) and a one-shot dice that rolls a
// value NOW and pins it. Typing a seed always means "use exactly this".
export const KIND = "seed";

export const render = (control, ctx) => {
    const state = { value: randomSeed(), randomize: true };
    const section = deckSection(ctx.t.seed);
    const row = document.createElement("div");
    row.className = "ts-studio__seedrow";
    const field = document.createElement("input");
    field.type = "text";
    field.className = "ts-ui-input";
    field.inputMode = "numeric";
    field.title = ctx.t.seedFieldTip;
    field.value = String(state.value);

    const dice = iconButton(SEED_ICONS.dice, ctx.t.seedDice);
    const mode = iconButton(SEED_ICONS.shuffle, ctx.t.randomizeTip);
    const hint = document.createElement("div");
    hint.className = "ts-studio__seedhint";

    function sync(emit = true) {
        mode.innerHTML = state.randomize ? SEED_ICONS.shuffle : SEED_ICONS.lock;
        mode.classList.toggle("is-active", state.randomize);
        mode.title = state.randomize ? ctx.t.randomizeTip : ctx.t.seedFixedTip;
        mode.setAttribute("aria-label", mode.title);
        mode.setAttribute("aria-pressed", state.randomize ? "true" : "false");
        hint.textContent = state.randomize ? ctx.t.seedHintRandom : ctx.t.seedHintFixed;
        if (emit) ctx.onChange(control.param, { ...state });
    }

    field.addEventListener("input", () => {
        const digits = field.value.replace(/\D/g, "");
        if (field.value !== digits) field.value = digits;
        state.value = Number(digits || 0);
        state.randomize = false;
        sync();
    });
    dice.addEventListener("click", () => {
        state.value = randomSeed();
        state.randomize = false;
        field.value = String(state.value);
        sync();
    });
    mode.addEventListener("click", () => {
        state.randomize = !state.randomize;
        sync();
    });

    row.append(field, dice, mode);
    section.append(row, hint);
    sync();

    function iconButton(svg, title) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-studio__seedbtn";
        button.title = title;
        button.setAttribute("aria-label", title);
        button.innerHTML = svg;
        return button;
    }

    return {
        element: section,
        get: () => ({ ...state }),
        set: (value) => {
            if (Number.isFinite(Number(value?.value))) {
                state.value = Number(value.value);
                field.value = String(state.value);
            }
            if (typeof value?.randomize === "boolean") state.randomize = value.randomize;
            sync();
        },
        // The app writes the seed a run actually used, so the field always
        // shows what produced the image on the stage — even in random mode.
        showSeed: (seed) => { field.value = String(seed); state.value = seed; },
    };
};
