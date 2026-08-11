// TS Studio kit — контрол «loras» (Стопка LoRA: список адаптеров с весами.)
//
// Один вид контрола — один файл. Манифест бэкенда объявляет контролы
// данными, реестр (`../_controls.js`) сводит вид к отрисовщику; ни один
// вид не знает о существовании других.
//
// Контракт отрисовщика: (control, ctx) -> {element, get, set}, и о каждой
// правке он сообщает через ctx.onChange(param, value).

import { deckSection } from "../_shell.js";

export const KIND = "loras";

export const render = (control, ctx) => {
    const [lo, hi] = control.strength || [-2.0, 2.0];
    const max = Number(control.max || 8);
    const section = deckSection("LoRA");
    const list = document.createElement("div");
    list.className = "ts-studio__loras";
    const pickWrap = document.createElement("div");
    pickWrap.className = "ts-studio__lorapick";
    const addButton = document.createElement("button");
    addButton.type = "button";
    addButton.className = "ts-studio__loraadd";
    addButton.title = ctx.t.loraAddTitle || "";
    addButton.textContent = ctx.t.loraAdd;
    const pop = document.createElement("div");
    pop.className = "ts-studio__lorapop";
    const search = document.createElement("input");
    search.type = "text";
    search.className = "ts-ui-input";
    search.placeholder = ctx.t.loraSearch;
    const optList = document.createElement("div");
    optList.className = "ts-studio__loralist";
    pop.append(search, optList);
    pickWrap.append(addButton, pop);
    section.append(list, pickWrap);

    const stack = []; // {name, strength}
    const options = ctx.loraOptions || [];

    function emit() {
        ctx.onChange(control.param || "loras", stack.map((l) => ({ ...l })));
        addButton.style.display = stack.length >= max ? "none" : "";
    }

    function renderOptions(query) {
        const needle = query.trim().toLowerCase();
        optList.textContent = "";
        for (const name of options) {
            if (stack.some((l) => l.name === name)) continue;
            if (needle && !name.toLowerCase().includes(needle)) continue;
            const option = document.createElement("button");
            option.type = "button";
            option.className = "ts-studio__loraopt";
            option.textContent = name.replace(/\\/g, "/");
            option.title = name;
            option.addEventListener("click", () => {
                stack.push({ name, strength: 1.0 });
                pop.classList.remove("is-open");
                renderList();
                emit();
            });
            optList.appendChild(option);
        }
    }

    let dragIndex = -1;
    function renderList() {
        list.textContent = "";
        stack.forEach((lora, index) => {
            const row = document.createElement("div");
            row.className = "ts-studio__lora";
            const handle = document.createElement("button");
            handle.type = "button";
            handle.className = "ts-studio__lorahandle";
            handle.textContent = "⋮⋮";
            handle.title = ctx.t.loraDrag;
            const name = document.createElement("span");
            name.className = "ts-studio__loraname";
            name.textContent = lora.name.replace(/\\/g, "/").split("/").pop();
            name.title = lora.name;
            const slider = document.createElement("input");
            slider.type = "range";
            slider.className = "ts-ui-slider";
            slider.min = String(lo);
            slider.max = String(hi);
            slider.step = "0.05";
            slider.value = String(lora.strength);
            slider.title = ctx.t.loraStrength;
            const value = document.createElement("span");
            value.className = "ts-studio__loraval";
            value.textContent = lora.strength.toFixed(2).replace(/0$/, "");
            slider.addEventListener("input", () => {
                lora.strength = Number(slider.value);
                value.textContent = lora.strength.toFixed(2).replace(/0$/, "");
                emit();
            });
            const x = document.createElement("button");
            x.type = "button";
            x.className = "ts-studio__lorax";
            x.textContent = "×";
            x.title = ctx.t.loraRemove;
            x.addEventListener("click", () => {
                stack.splice(index, 1);
                renderList();
                emit();
            });
            row.draggable = true;
            row.addEventListener("dragstart", (event) => {
                dragIndex = index;
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData("text/plain", String(index));
            });
            row.addEventListener("dragover", (event) => {
                if (dragIndex < 0) return;
                event.preventDefault();
                row.classList.add("is-drag-over");
            });
            row.addEventListener("dragleave", () => row.classList.remove("is-drag-over"));
            row.addEventListener("drop", (event) => {
                event.preventDefault();
                row.classList.remove("is-drag-over");
                if (dragIndex < 0 || dragIndex === index) return;
                const [moved] = stack.splice(dragIndex, 1);
                stack.splice(index, 0, moved);
                dragIndex = -1;
                renderList();
                emit();
            });
            row.append(handle, name, slider, value, x);
            list.appendChild(row);
        });
    }

    addButton.addEventListener("click", () => {
        const open = !pop.classList.contains("is-open");
        pop.classList.toggle("is-open", open);
        if (open) {
            renderOptions("");
            search.value = "";
            search.focus();
        }
    });
    search.addEventListener("input", () => renderOptions(search.value));
    const onDocDown = (event) => {
        if (!pickWrap.contains(event.target)) pop.classList.remove("is-open");
    };
    document.addEventListener("pointerdown", onDocDown);

    if (!options.length) {
        addButton.disabled = true;
        addButton.title = ctx.t.loraNone;
    }
    emit();

    return {
        element: section,
        get: () => stack.map((l) => ({ ...l })),
        set: (value) => {
            stack.length = 0;
            for (const lora of value || []) stack.push({ ...lora });
            renderList();
            emit();
        },
        teardown: () => document.removeEventListener("pointerdown", onDocDown),
    };
};
