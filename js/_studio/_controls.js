// TS Studio kit — the control-kind registry (ui-kit layer).
//
// A backend manifest declares controls as data ({param, kind, ...}); this
// registry maps kind -> renderer. Adding a control kind = registering one
// function (plan §3.5); the deck builder walks the manifest and never
// special-cases a family. Every renderer returns {element, get, set} and
// reports edits through onChange so the app stores values per mode.

import { deckSection } from "./_shell.js";

const STYLE_ID = "ts-studio-controls-styles";

export function ensureControlStyles() {
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
.ts-studio__seedrow{display:flex;align-items:center;gap:6px}
.ts-studio__seedrow input[type=text]{flex:1}
.ts-studio__numrow{display:flex;align-items:center;justify-content:space-between;gap:8px;
    min-height:26px}
.ts-studio__numrow input{width:76px;text-align:right}
.ts-studio__advanced{border:none;background:none;padding:0;display:flex;align-items:center;gap:5px;
    color:var(--ts-muted);cursor:pointer;font-size:var(--ts-fs-sm)}
.ts-studio__advanced:hover{color:var(--ts-text)}
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
        set: (value) => { area.value = String(value ?? ""); },
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

    return {
        element: section,
        get: () => ({ aspect: state.aspect, mp: state.mp }),
        set: (value) => {
            if (value?.aspect) state.aspect = value.aspect;
            if (value?.mp) { state.mp = Number(value.mp); slider.value = String(state.mp); }
            sync();
        },
    };
});

// ── seed: manual value + randomize toggle ───────────────────────────────── //
export function randomSeed() {
    return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
}

registerControlKind("seed", (control, ctx) => {
    const state = { value: 0, randomize: true };
    const section = deckSection(ctx.t.seed);
    const row = document.createElement("div");
    row.className = "ts-studio__seedrow";
    const field = document.createElement("input");
    field.type = "text";
    field.className = "ts-ui-input";
    field.inputMode = "numeric";
    field.value = "0";
    field.addEventListener("input", () => {
        const parsed = Number(field.value.replace(/\D/g, "") || 0);
        state.value = parsed;
        state.randomize = false;
        toggle.classList.remove("is-active");
        ctx.onChange(control.param, state);
    });
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "ts-ui-btn is-active";
    toggle.textContent = ctx.t.randomize;
    toggle.title = ctx.t.randomizeTip;
    toggle.addEventListener("click", () => {
        state.randomize = !state.randomize;
        toggle.classList.toggle("is-active", state.randomize);
        ctx.onChange(control.param, state);
    });
    row.append(field, toggle);
    section.appendChild(row);
    ctx.onChange(control.param, state);
    return {
        element: section,
        get: () => ({ ...state }),
        set: (value) => {
            if (typeof value?.value === "number") { state.value = value.value; field.value = String(value.value); }
            if (typeof value?.randomize === "boolean") {
                state.randomize = value.randomize;
                toggle.classList.toggle("is-active", state.randomize);
            }
            ctx.onChange(control.param, state);
        },
        // The app reads/writes the last used seed here after each run.
        showSeed: (seed) => { field.value = String(seed); state.value = seed; },
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
    };
});
