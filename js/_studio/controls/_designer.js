// TS Studio kit — контрол «designer» (Дизайнер: семейство отдаёт свой собственный редактор.)
//
// Один вид контрола — один файл. Манифест бэкенда объявляет контролы
// данными, реестр (`../_controls.js`) сводит вид к отрисовщику; ни один
// вид не знает о существовании других.
//
// Контракт отрисовщика: (control, ctx) -> {element, get, set}, и о каждой
// правке он сообщает через ctx.onChange(param, value).

import { deckSection } from "../_shell.js";
import { getEditorProvider } from "../_editors.js";
import { localized } from "./_shared.js";

// The studio does not reimplement an authoring UI that already ships with a
// node — this control opens that editor and keeps whatever state it returns.
export const KIND = "designer";

export const render = (control, ctx) => {
    const provider = getEditorProvider(control.provider);
    const section = deckSection(localized(control.label, ctx.locale, ""));
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ts-ui-btn ts-studio__designer";
    const label = localized(provider?.label, ctx.locale, control.provider || "Editor");
    button.textContent = label;
    const note = document.createElement("div");
    note.className = "ts-studio__seedhint";
    let state = null;

    function sync(emit = true) {
        note.textContent = state ? ctx.t.designerReady : ctx.t.designerEmpty;
        button.classList.toggle("is-active", Boolean(state));
        if (emit) ctx.onChange(control.param, state ? JSON.stringify(state) : "");
    }

    button.addEventListener("click", async () => {
        if (!provider) return;
        button.disabled = true;
        try {
            const next = await provider.open({
                design: state,
                prompt: ctx.getPrompt?.() || "",
                aspect: ctx.getSize?.()?.aspect,
                megapixels: ctx.getSize?.()?.mp,
            });
            if (next) state = next;
            sync();
        } catch (err) {
            console.warn("[TS Studio] editor failed", err);
        } finally {
            button.disabled = false;
        }
    });

    section.append(button, note);
    if (!provider) {
        button.disabled = true;
        button.title = ctx.t.designerMissing;
    }
    sync();

    return {
        element: section,
        get: () => state,
        set: (value) => {
            if (typeof value === "string" && value.trim()) {
                try { state = JSON.parse(value); } catch { state = null; }
            } else if (value && typeof value === "object") {
                state = value;
            } else {
                state = null;
            }
            sync();
        },
    };
};
