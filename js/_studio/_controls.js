// TS Studio kit — the control-kind registry (ui-kit layer).
//
// A backend manifest declares controls as data ({param, kind, ...}); this
// registry maps kind -> renderer. Adding a control kind = registering one
// function (plan §3.5); the deck builder walks the manifest and never
// special-cases a family. Every renderer returns {element, get, set} and
// reports edits through onChange so the app stores values per mode.

import { ensureThemeStyles } from "../_theme.js";

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
/* Proportions are chosen with the pack's shared control (the .ts-ui-ratiopick
   family in js/_theme.js) — a trigger plus a popover, the same one TS Resolution
   Selector shows open. This wrapper only puts it on its own line.
   NOTE: no backticks in this comment — the whole stylesheet is one template
   literal, and one backtick would end it. */
.ts-studio__aspects{display:flex;align-items:center;gap:6px}
/* A choice that turned out to be a list of proportions shows the same trigger,
   so its row of words is not a row any more. */
.ts-studio__choice--ratios{display:block}
.ts-studio__sizerow{display:flex;align-items:center;gap:8px}
.ts-studio__sizerow input[type=range]{flex:1}
.ts-studio__sizeinfo{display:flex;justify-content:space-between;font-size:var(--ts-fs-sm);
    color:var(--ts-muted)}
.ts-studio__size.is-disabled .ts-studio__aspects,
.ts-studio__size.is-disabled .ts-studio__sizerow,
.ts-studio__size.is-disabled .ts-studio__sizeinfo{opacity:.4}
.ts-studio__sizenote{font-size:var(--ts-fs-xs);color:var(--ts-muted)}
.ts-studio__choice{display:flex;gap:5px}
.ts-studio__choicebtn{flex:1;min-width:0;padding:6px 4px;border-radius:var(--ts-radius-sm);
    border:1px solid var(--ts-border);background:var(--ts-surface);color:var(--ts-muted);
    font-size:var(--ts-fs-sm);cursor:pointer;
    transition:border-color .12s ease,background .12s ease,color .12s ease}
.ts-studio__choicebtn:hover{border-color:var(--ts-border-strong);color:var(--ts-text)}
.ts-studio__choicebtn.is-active{border-color:var(--ts-accent-line);
    background:var(--ts-accent-soft);color:var(--ts-accent)}
.ts-studio__choicebtn:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
.ts-studio__choice.is-disabled{opacity:.45}
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
/* Шкала силы: деления подписаны, потому что это не непрерывная величина, а
   несколько осмысленных ступеней — и последняя из них меняет режим целиком. */
.ts-studio__ticks{display:flex;justify-content:space-between;margin-top:2px;
    font-size:var(--ts-fs-xs);color:var(--ts-muted);user-select:none}
.ts-studio__tick{flex:1 1 0;text-align:center;cursor:pointer;padding:1px 0;
    border-radius:var(--ts-radius-sm)}
.ts-studio__tick:first-child{text-align:left}
.ts-studio__tick:last-child{text-align:right}
.ts-studio__tick:hover{color:var(--ts-text)}
.ts-studio__tick.is-active{color:var(--ts-accent)}
.ts-studio__designer.is-active{border-color:var(--ts-accent-line);color:var(--ts-accent)}
.ts-studio__numrow{display:flex;align-items:center;justify-content:space-between;gap:8px;
    min-height:26px}
.ts-studio__switch{position:relative;width:34px;height:19px;flex:0 0 auto;padding:0;
    border:none;border-radius:999px;cursor:pointer;background:var(--ts-surface-active);
    box-shadow:inset 0 0 0 1px var(--ts-border-soft);transition:background .14s ease}
.ts-studio__switch.is-on{background:var(--ts-accent);box-shadow:none}
.ts-studio__switchknob{position:absolute;top:2px;left:2px;width:15px;height:15px;
    border-radius:50%;background:var(--ts-on-media);box-shadow:var(--ts-shadow-sm);
    transition:transform .14s ease}
.ts-studio__switch.is-on .ts-studio__switchknob{transform:translateX(15px)}
.ts-studio__switch:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:2px}
.ts-studio__numrow input{width:76px;text-align:right}
.ts-studio__advanced{border:none;background:none;padding:0;display:flex;align-items:center;gap:5px;
    color:var(--ts-muted);cursor:pointer;font-size:var(--ts-fs-sm)}
.ts-studio__advanced:hover{color:var(--ts-text)}
.ts-studio__refs{display:flex;gap:8px;flex-wrap:wrap}
/* The border is drawn INSIDE via box-shadow rather than as a real border:
   a bordered box clips its content against the padding box, so a cover image
   left a hairline of border colour along the rounded corners. With an inset
   shadow the image fills the whole button and the frame sits on top of it. */
.ts-studio__ref{position:relative;width:56px;height:56px;border:none;
    border-radius:var(--ts-radius);background:var(--ts-surface);color:var(--ts-muted);
    cursor:pointer;display:flex;align-items:center;justify-content:center;
    font-size:16px;padding:0;overflow:hidden;
    box-shadow:inset 0 0 0 1px var(--ts-border-soft);
    transition:box-shadow .12s ease,color .12s ease}
.ts-studio__ref::after{content:"";position:absolute;inset:0;border-radius:inherit;
    pointer-events:none;box-shadow:inset 0 0 0 1px var(--ts-border-strong)}
.ts-studio__ref.is-filled::after{box-shadow:inset 0 0 0 1px var(--ts-border)}
.ts-studio__ref:hover{color:var(--ts-text);box-shadow:inset 0 0 0 1px var(--ts-border-strong)}
.ts-studio__ref.is-drag-over{color:var(--ts-accent)}
.ts-studio__ref.is-drag-over::after{box-shadow:inset 0 0 0 2px var(--ts-accent)}
.ts-studio__ref img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;
    border-radius:inherit}
/* Sits fully inside the rounded corner instead of straddling it, and only
   appears once there is something to clear. */
.ts-studio__refx{position:absolute;top:4px;right:4px;z-index:2;width:17px;height:17px;
    border-radius:50%;border:none;background:var(--ts-scrim-strong);color:var(--ts-on-media);
    font-size:11px;line-height:17px;text-align:center;cursor:pointer;padding:0;display:none;
    opacity:0;transition:opacity .12s ease}
.ts-studio__ref.is-filled .ts-studio__refx{display:block}
.ts-studio__ref.is-filled:hover .ts-studio__refx,
.ts-studio__refx:focus-visible{opacity:1}
.ts-studio__ref:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:2px}
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

// ── виды контролов ──────────────────────────────────────────────────────── //
//
// По файлу на вид. Реестр знает только их список; сами виды друг о друге не
// знают вовсе, поэтому новый добавляется одним файлом и одной строкой здесь.
import * as promptKind from "./controls/_prompt.js";
import * as sizeKind from "./controls/_size.js";
import * as seedKind from "./controls/_seed.js";
import * as refsKind from "./controls/_refs.js";
import * as lorasKind from "./controls/_loras.js";
import * as designerKind from "./controls/_designer.js";
import * as choiceKind from "./controls/_choice.js";
import * as toggleKind from "./controls/_toggle.js";
import * as sliderKind from "./controls/_slider.js";
import * as strengthKind from "./controls/_strength.js";
import * as numberKind from "./controls/_number.js";

for (const kind of [promptKind, sizeKind, seedKind, refsKind, lorasKind, designerKind, choiceKind, toggleKind, sliderKind, strengthKind, numberKind]) {
    registerControlKind(kind.KIND, kind.render);
}

// Публичный контракт модуля не изменился: студия по-прежнему берёт эти две
// функции отсюда, хотя живут они теперь в файлах своих видов.
export { sizeFromAspect } from "./controls/_size.js";
export { randomSeed } from "./controls/_seed.js";
