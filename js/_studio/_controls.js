// TS Studio kit — the control-kind registry (ui-kit layer).
//
// A backend manifest declares controls as data ({param, kind, ...}); this
// registry maps kind -> renderer. Adding a control kind = registering one
// function (plan §3.5); the deck builder walks the manifest and never
// special-cases a family. Every renderer returns {element, get, set} and
// reports edits through onChange so the app stores values per mode.

import { ensureThemeStyles } from "../_theme.js";
import { deckSection } from "./_shell.js";
import { makeDropZone, annotatedImageUrl } from "./_dnd.js";
import { getEditorProvider } from "./_editors.js";

// Controls render inside the shell's TS_UI_CLASS scope; ensureThemeStyles()
// here keeps this module self-sufficient if a control is ever mounted alone.
const STYLE_ID = "ts-studio-controls-styles";

export function ensureControlStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-studio__prompt{position:relative}
.ts-studio__prompt textarea{width:100%;min-height:86px;resize:vertical}
.ts-studio__aspects{display:flex;flex-wrap:wrap;gap:5px}
.ts-studio__aspect{display:flex;flex-direction:column;align-items:center;gap:3px;
    padding:5px 7px 3px;border:1px solid var(--ts-border);border-radius:var(--ts-radius);
    background:none;color:var(--ts-muted);cursor:pointer;font-size:var(--ts-fs-xs)}
.ts-studio__aspect:hover{border-color:var(--ts-border-strong);color:var(--ts-text)}
.ts-studio__aspect.is-active{color:var(--ts-accent)}
.ts-studio__aspect.is-active .ts-studio__aspectshape{background:var(--ts-accent);border-color:var(--ts-accent)}
.ts-studio__aspectshape{border:1px solid var(--ts-muted);border-radius:2px;background:none}
.ts-studio__aspect:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
.ts-studio__sizerow{display:flex;align-items:center;gap:8px}
.ts-studio__sizerow input[type=range]{flex:1}
.ts-studio__sizeinfo{display:flex;justify-content:space-between;font-size:var(--ts-fs-sm);
    color:var(--ts-muted)}
.ts-studio__size.is-disabled .ts-studio__aspects,
.ts-studio__size.is-disabled .ts-studio__sizerow,
.ts-studio__size.is-disabled .ts-studio__sizeinfo{opacity:.4}
.ts-studio__sizenote{font-size:var(--ts-fs-xs);color:var(--ts-muted)}
.ts-studio__seedrow{display:flex;align-items:center;gap:4px}
.ts-studio__seedrow input[type=text]{flex:1;font-variant-numeric:tabular-nums}
.ts-studio__seedbtn{width:26px;height:26px;flex:0 0 auto;display:flex;align-items:center;
    justify-content:center;border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);
    background:none;color:var(--ts-muted);cursor:pointer;padding:0}
.ts-studio__seedbtn:hover{color:var(--ts-text);border-color:var(--ts-border-strong)}
.ts-studio__seedbtn.is-active{color:var(--ts-accent);border-color:var(--ts-accent-line);
    background:var(--ts-accent-soft)}
.ts-studio__seedbtn:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
.ts-studio__seedhint{font-size:var(--ts-fs-xs);color:var(--ts-muted)}
.ts-studio__sliderhead{display:flex;align-items:center;justify-content:space-between;gap:8px}
.ts-studio__slidervalue{font-size:var(--ts-fs-sm);color:var(--ts-text);
    font-variant-numeric:tabular-nums}
.ts-studio__sliderrow{display:flex;align-items:center;gap:6px}
.ts-studio__sliderrow input[type=range]{flex:1;min-width:0}
.ts-studio__slider.is-disabled{opacity:.45}
.ts-studio__designer.is-active{border-color:var(--ts-accent-line);color:var(--ts-accent)}
.ts-studio__numrow{display:flex;align-items:center;justify-content:space-between;gap:8px;
    min-height:26px}
.ts-studio__numrow input{width:76px;text-align:right}
.ts-studio__advanced{border:none;background:none;padding:0;display:flex;align-items:center;gap:5px;
    color:var(--ts-muted);cursor:pointer;font-size:var(--ts-fs-sm)}
.ts-studio__advanced:hover{color:var(--ts-text)}
.ts-studio__refs{display:flex;gap:6px}
.ts-studio__ref{position:relative;width:52px;height:52px;border:1px dashed var(--ts-border-strong);
    border-radius:var(--ts-radius);background:none;color:var(--ts-muted);cursor:pointer;
    display:flex;align-items:center;justify-content:center;font-size:15px;padding:0;overflow:hidden}
.ts-studio__ref.is-drag-over{border-color:var(--ts-accent);color:var(--ts-accent)}
.ts-studio__ref.is-filled{border-style:solid}
.ts-studio__ref img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.ts-studio__refx{position:absolute;top:1px;right:1px;z-index:2;width:15px;height:15px;
    border-radius:50%;border:none;background:var(--ts-elevated);color:var(--ts-text);
    font-size:10px;line-height:1;cursor:pointer;padding:0;display:none}
.ts-studio__ref.is-filled .ts-studio__refx{display:block}
.ts-studio__ref:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
.ts-studio__loras{display:flex;flex-direction:column;gap:3px}
.ts-studio__lora{display:flex;align-items:center;gap:6px;min-height:26px;
    border-radius:var(--ts-radius-sm);padding:1px 2px}
.ts-studio__lora.is-drag-over{background:var(--ts-accent-soft)}
.ts-studio__lorahandle{cursor:grab;color:var(--ts-muted);border:none;background:none;
    padding:0 2px;font-size:11px;letter-spacing:1px}
.ts-studio__loraname{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
    font-size:var(--ts-fs-sm)}
.ts-studio__lora input[type=range]{width:64px}
.ts-studio__loraval{width:34px;text-align:right;font-size:var(--ts-fs-sm);color:var(--ts-muted)}
.ts-studio__lorax{border:none;background:none;color:var(--ts-muted);cursor:pointer;padding:0 3px}
.ts-studio__lorax:hover{color:var(--ts-danger)}
.ts-studio__loraadd{border:1px dashed var(--ts-border-strong);border-radius:var(--ts-radius-sm);
    background:none;color:var(--ts-muted);cursor:pointer;padding:4px;font-size:var(--ts-fs-sm)}
.ts-studio__loraadd:hover{color:var(--ts-text);border-color:var(--ts-border-strong)}
.ts-studio__lorapick{position:relative}
.ts-studio__lorapop{position:absolute;z-index:41;left:0;right:0;top:calc(100% + 3px);
    display:none;flex-direction:column;gap:4px;padding:6px;max-height:220px;
    background:var(--ts-elevated);border:1px solid var(--ts-border);
    border-radius:var(--ts-radius);box-shadow:var(--ts-shadow)}
.ts-studio__lorapop.is-open{display:flex}
.ts-studio__loralist{overflow-y:auto;display:flex;flex-direction:column;min-height:0}
/* flex:0 0 auto is load-bearing: in a scrolling flex column the default
   flex-shrink squeezed every option down to 6px once the list overflowed. */
.ts-studio__loraopt{flex:0 0 auto;min-height:22px;display:flex;align-items:center;
    border:none;background:none;color:var(--ts-text);cursor:pointer;
    text-align:left;padding:3px 5px;border-radius:var(--ts-radius-sm);
    font-size:var(--ts-fs-sm);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ts-studio__loraopt:hover{background:var(--ts-border-soft)}
`;
    document.head.appendChild(style);
}

const RENDERERS = new Map();

/** Extension point: register a renderer for a manifest control kind. */
export function registerControlKind(kind, renderer) {
    RENDERERS.set(kind, renderer);
}

export function getControlRenderer(kind) {
    return RENDERERS.get(kind) || null;
}

function localized(labelSpec, locale, fallback) {
    if (!labelSpec) return fallback;
    if (typeof labelSpec === "string") return labelSpec;
    return labelSpec[locale] || labelSpec.en || fallback;
}

// ── prompt ──────────────────────────────────────────────────────────────── //
registerControlKind("prompt", (control, ctx) => {
    const section = deckSection(localized(control.label, ctx.locale,
        control.param === "negative_prompt" ? ctx.t.negativePrompt : ctx.t.prompt));
    const wrap = document.createElement("div");
    wrap.className = "ts-studio__prompt";
    const area = document.createElement("textarea");
    area.className = "ts-ui-textarea";
    area.placeholder = ctx.t.promptPlaceholder;
    area.addEventListener("input", () => ctx.onChange(control.param, area.value));
    wrap.appendChild(area);
    // The prompt toolbar (voice, image, presets, styles) mounts here in the
    // media-combine phase; the slot exists from day one.
    const toolbarSlot = document.createElement("div");
    toolbarSlot.dataset.tsSlot = `prompt-toolbar:${control.param}`;
    wrap.appendChild(toolbarSlot);
    section.appendChild(wrap);
    return {
        element: section,
        get: () => area.value,
        // Setting a control is a value change like any other — every other
        // kind reports one from its set(). The prompt must too: the deck is
        // rebuilt on every model and mode switch, and the text carried across
        // is restored through here. Staying silent would leave the words
        // visible in the box while the run went out without them.
        set: (value) => {
            area.value = String(value ?? "");
            ctx.onChange(control.param, area.value);
        },
    };
});

// ── size: aspect cards + megapixel slider ───────────────────────────────── //
function parseAspect(text) {
    const [w, h] = String(text).split(":").map(Number);
    return w > 0 && h > 0 ? w / h : 1;
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

registerControlKind("size", (control, ctx) => {
    const aspects = control.aspects || ["1:1"];
    const state = {
        aspect: control.default_aspect || aspects[0],
        mp: Number(control.base_mp || 1.0),
    };
    const [mpMin, mpMax] = control.mp || [0.25, 4.0];
    const snap = Number(control.snap || 16);

    const section = deckSection(ctx.t.format);
    section.classList.add("ts-studio__size");
    const grid = document.createElement("div");
    grid.className = "ts-studio__aspects";
    const buttons = new Map();
    for (const aspect of aspects) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-studio__aspect";
        const shape = document.createElement("span");
        shape.className = "ts-studio__aspectshape";
        const ratio = parseAspect(aspect);
        const base = 18;
        shape.style.width = `${Math.round(ratio >= 1 ? base : base * ratio)}px`;
        shape.style.height = `${Math.round(ratio >= 1 ? base / ratio : base)}px`;
        const caption = document.createElement("span");
        caption.textContent = aspect;
        button.append(shape, caption);
        button.addEventListener("click", () => { state.aspect = aspect; sync(); });
        grid.appendChild(button);
        buttons.set(aspect, button);
    }
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
        for (const [aspect, button] of buttons) {
            button.classList.toggle("is-active", aspect === state.aspect);
        }
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
            for (const button of buttons.values()) button.disabled = off;
            note.style.display = off ? "" : "none";
        },
        set: (value) => {
            // A carried-over aspect only applies if this model offers it;
            // otherwise the model's own default stands.
            if (value?.aspect && buttons.has(value.aspect)) state.aspect = value.aspect;
            if (value?.mp) {
                state.mp = Math.min(mpMax, Math.max(mpMin, Number(value.mp)));
                slider.value = String(state.mp);
            }
            sync();
        },
    };
});

// ── seed: manual value + randomize toggle ───────────────────────────────── //
export function randomSeed() {
    return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
}

const SEED_ICONS = {
    dice: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="4" y="4" width="16" height="16" rx="3"/><circle cx="9" cy="9" r="1.2" fill="currentColor" stroke="none"/><circle cx="15" cy="15" r="1.2" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.2" fill="currentColor" stroke="none"/></svg>',
    shuffle: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M3 7h4l10 10h4M3 17h4l3.2-3.2M14.6 9.2L17 7h4M18 4l3 3-3 3M18 14l3 3-3 3"/></svg>',
    lock: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7"><rect x="5" y="10" width="14" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg>',
};

// Seed is two decisions, so it gets two explicit controls: the mode (a new
// seed every run vs the one in the field) and a one-shot dice that rolls a
// value NOW and pins it. Typing a seed always means "use exactly this".
registerControlKind("seed", (control, ctx) => {
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
});

// ── reference slots ─────────────────────────────────────────────────────── //
// Each slot maps to an optional image marker (ref_1..ref_N). Filling a slot
// uploads the blob immediately; the run only carries annotated names. Empty
// slots become dropParams so the patcher removes their branch.
registerControlKind("refs", (control, ctx) => {
    const max = Math.max(1, Math.min(Number(control.max || 3), 6));
    // A backend may name the slots for what they do there — Inpaint asks for
    // one object to place, not a set of references.
    const section = deckSection(localized(control.label, ctx.locale, ctx.t.references));
    const row = document.createElement("div");
    row.className = "ts-studio__refs";
    section.appendChild(row);

    const slots = [];
    const teardowns = [];

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.className = "ts-ui-file";
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
            slots[index].img.src = URL.createObjectURL(blob);
            slots[index].button.classList.add("is-filled");
            emit();
        } catch (err) {
            console.warn("[TS Studio] reference upload failed", err);
        }
    }

    function clear(index) {
        slots[index].value = "";
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
        slots.push({ button, img, value: "" });
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
                    slot.img.src = annotatedImageUrl(annotated);
                    slot.button.classList.add("is-filled");
                } else {
                    slot.img.removeAttribute("src");
                    slot.button.classList.remove("is-filled");
                }
            });
            emit();
        },
        teardown: () => teardowns.forEach((fn) => fn()),
    };
});

// ── LoRA stack ──────────────────────────────────────────────────────────── //
registerControlKind("loras", (control, ctx) => {
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
});

// ── designer: hand the family's own editor the wheel ────────────────────── //
// The studio does not reimplement an authoring UI that already ships with a
// node — this control opens that editor and keeps whatever state it returns.
registerControlKind("designer", (control, ctx) => {
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
});

// ── toggle ──────────────────────────────────────────────────────────────── //
registerControlKind("toggle", (control, ctx) => {
    const row = document.createElement("div");
    row.className = "ts-studio__numrow";
    const label = document.createElement("span");
    label.textContent = (control.label?.[ctx.locale]) || control.label?.en || control.param;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ts-ui-btn";
    button.textContent = "OFF";
    const tip = control.tooltip?.[ctx.locale] || control.tooltip?.en;
    if (tip) { row.title = tip; button.title = tip; }
    let value = Boolean(control.default);
    function sync() {
        button.classList.toggle("is-active", value);
        button.textContent = value ? "ON" : "OFF";
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
});

// ── slider: a bounded number that reads as a magnitude ──────────────────── //
// Same contract as "number" (get/set/setDisabled) so a manifest can swap the
// kind without the app noticing.
registerControlKind("slider", (control, ctx) => {
    const min = Number(control.min ?? 0);
    const max = Number(control.max ?? 1);
    const step = Number(control.step ?? 0.05);
    const decimals = String(step).includes(".") ? String(step).split(".")[1].length : 0;
    let value = Number(control.default ?? min);

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
});

// ── number ──────────────────────────────────────────────────────────────── //
registerControlKind("number", (control, ctx) => {
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
});
