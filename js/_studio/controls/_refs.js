// TS Studio kit — контрол «refs» (Слоты референсов: картинки, по которым модель правит кадр.)
//
// Один вид контрола — один файл. Манифест бэкенда объявляет контролы
// данными, реестр (`../_controls.js`) сводит вид к отрисовщику; ни один
// вид не знает о существовании других.
//
// Контракт отрисовщика: (control, ctx) -> {element, get, set}, и о каждой
// правке он сообщает через ctx.onChange(param, value).

import { deckSection } from "../_shell.js";
import { makeDropZone, annotatedImageUrl } from "../_dnd.js";
import { localized } from "./_shared.js";
import { createHiddenFileInput } from "../../_theme.js";

// Each slot maps to an optional image marker (ref_1..ref_N). Filling a slot
// uploads the blob immediately; the run only carries annotated names. Empty
// slots become dropParams so the patcher removes their branch.
export const KIND = "refs";

export const render = (control, ctx) => {
    const max = Math.max(1, Math.min(Number(control.max || 3), 6));
    // A backend may name the slots for what they do there — Inpaint asks for
    // one object to place, not a set of references.
    const section = deckSection(localized(control.label, ctx.locale, ctx.t.references));
    const row = document.createElement("div");
    row.className = "ts-studio__refs";
    section.appendChild(row);

    const slots = [];
    const teardowns = [];

    const fileInput = createHiddenFileInput({ accept: "image/*" });
    document.body.appendChild(fileInput);
    teardowns.push(() => fileInput.remove());
    let pickTarget = -1;
    fileInput.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        if (file && pickTarget >= 0) fill(pickTarget, file, file.name);
        fileInput.value = "";
    });

    async function fill(index, blob, name) {
        try {
            const annotated = await ctx.uploadImage(blob, name || `ref_${index + 1}.png`);
            slots[index].value = annotated;
            // Свой адрес держим сами: чужой blob-адрес может быть отозван
            // владельцем (так и было с кадром, вынутым из ролика), и в слоте
            // оставалась битая картинка.
            releaseUrl(index);
            const url = URL.createObjectURL(blob);
            slots[index].url = url;
            slots[index].img.src = url;
            slots[index].button.classList.add("is-filled");
            emit();
        } catch (err) {
            console.warn("[TS Studio] reference upload failed", err);
        }
    }

    /** Отпустить адрес прошлого превью — иначе они копятся за сессию. */
    function releaseUrl(index) {
        const url = slots[index]?.url;
        if (url) URL.revokeObjectURL(url);
        if (slots[index]) slots[index].url = "";
    }

    function clear(index) {
        slots[index].value = "";
        releaseUrl(index);
        slots[index].img.removeAttribute("src");
        slots[index].button.classList.remove("is-filled");
        emit();
    }

    function emit() {
        const refs = {};
        slots.forEach((slot, i) => { refs[`ref_${i + 1}`] = slot.value; });
        ctx.onChange(control.param || "__refs", refs);
    }

    for (let i = 0; i < max; i += 1) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-studio__ref";
        button.title = ctx.t.refSlotTip(i + 1);
        button.setAttribute("aria-label", ctx.t.refSlotTip(i + 1));
        const img = document.createElement("img");
        img.alt = "";
        const plus = document.createElement("span");
        plus.textContent = "+";
        const x = document.createElement("button");
        x.type = "button";
        x.className = "ts-studio__refx";
        x.textContent = "×";
        x.title = ctx.t.refClear;
        button.append(img, plus, x);
        const index = i;
        button.addEventListener("click", (event) => {
            if (event.target === x) return;
            pickTarget = index;
            fileInput.click();
        });
        x.addEventListener("click", (event) => { event.stopPropagation(); clear(index); });
        teardowns.push(makeDropZone(button, {
            max: 1,
            onDrop: async ([item]) => fill(index, await item.getBlob(), item.name),
        }));
        row.appendChild(button);
        slots.push({ button, img, value: "", url: "" });
    }
    emit();

    return {
        element: section,
        get: () => slots.map((s) => s.value),
        // Values are annotated names, so a restored slot points at the very
        // file the original run used — no re-upload, no copy.
        set: (values) => {
            const list = Array.isArray(values) ? values : [];
            slots.forEach((slot, index) => {
                const annotated = String(list[index] || "");
                slot.value = annotated;
                if (annotated) {
                    releaseUrl(index);
                    slot.img.src = annotatedImageUrl(annotated);
                    slot.button.classList.add("is-filled");
                } else {
                    releaseUrl(index);
                    slot.img.removeAttribute("src");
                    slot.button.classList.remove("is-filled");
                }
            });
            emit();
        },
        teardown: () => teardowns.forEach((fn) => fn()),
    };
};
