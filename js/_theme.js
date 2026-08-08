// Pack-level visual design system for every comfyui-timesaver node that draws
// its own UI (DOM widgets, popovers, fullscreen editors).
//
// Why this exists: each interactive node used to ship its own palette, so the
// pack looked like a dozen unrelated plugins. Everything now reads from one set
// of `--ts-*` tokens declared here, which in turn defer to ComfyUI's own theme
// variables — so our panels follow the user's theme instead of fighting it.
//
// Usage from a node module:
//     import { ensureThemeStyles, TS_UI_CLASS } from "../_theme.js";
//     ensureThemeStyles();
//     container.className = `${TS_UI_CLASS} my-node`;
//
// The module is side-effect free on import (ComfyUI auto-imports every file
// under WEB_DIRECTORY); styles are injected only when a node asks for them.

const STYLE_ID = "ts-theme-styles";

// Root class that carries the tokens. Any element that renders TS UI should
// have it (or descend from something that does) — tokens are also declared on
// :root so inheritance normally covers it either way.
export const TS_UI_CLASS = "ts-ui";

// ---------------------------------------------------------------------------
// THE BRAND KNOB
// ---------------------------------------------------------------------------
// Single source of truth for the accent colour. Every accented surface in the
// pack (primary buttons, sliders, focus rings, selection outlines, active
// tabs, waveform highlights) derives from this one value. Change it here and
// the whole pack follows.
export const TS_ACCENT = "#9a8cc7";

// Static fallbacks for browsers without color-mix(). Custom properties are
// parsed as raw token streams, so an unsupported color-mix() would NOT fall
// back automatically — hence the @supports guard in the stylesheet below,
// which only overrides these once the browser proves it can compute them.
const TS_ACCENT_STRONG = "#8677b6";
const TS_ACCENT_DIM = "#6f6396";
const TS_ACCENT_SOFT = "rgba(154, 140, 199, 0.16)";
const TS_ACCENT_LINE = "rgba(154, 140, 199, 0.42)";
// Text/icon colour placed ON an accent-filled surface.
const TS_ACCENT_CONTRAST = "#15121d";

function themeCss() {
    return `
/* ── Tokens ──────────────────────────────────────────────────────────────
   Only four values are read from ComfyUI: background, surface, text and
   border. Everything else is MIXED FROM THEM rather than hard-coded, which is
   what makes a light palette work: a hover state is "a step from the surface
   TOWARDS the text colour", and that step goes lighter on a dark theme and
   darker on a light one automatically. The literals below are the stock dark
   values and only apply on browsers without color-mix(). */
:root,.${TS_UI_CLASS}{
  --ts-accent:${TS_ACCENT};
  --ts-accent-strong:${TS_ACCENT_STRONG};
  --ts-accent-dim:${TS_ACCENT_DIM};
  --ts-accent-soft:${TS_ACCENT_SOFT};
  --ts-accent-line:${TS_ACCENT_LINE};
  --ts-accent-contrast:${TS_ACCENT_CONTRAST};

  --ts-bg:var(--comfy-menu-bg,#171718);
  --ts-surface:var(--comfy-input-bg,#222);
  --ts-surface-hover:#2e2e30;
  --ts-surface-active:#37373a;
  --ts-elevated:#232325;
  --ts-sunken:#141415;
  /* A dimming scrim stays dark in both themes — that is what "dimmed" means.
     The same is true of chrome that sits ON user media (a clear button over a
     thumbnail, a label plate over a frame): it must read against any picture,
     so these two are deliberately fixed rather than themed. */
  --ts-scrim:rgba(12,12,13,.72);
  --ts-scrim-strong:rgba(12,12,13,.86);
  --ts-on-media:#f2f2f4;
  --ts-modal-bg:var(--comfy-menu-bg,#171718);

  --ts-text:var(--input-text,#ddd);
  --ts-muted:var(--descrip-text,#999);
  --ts-faint:#6d6d70;

  --ts-border:var(--border-color,#4e4e4e);
  --ts-border-soft:#333335;
  --ts-border-strong:#5c5c5f;

  --ts-danger:#d98a86;
  --ts-success:#8fbf9f;
  --ts-warning:#d4b483;

  /* Scrollbars need real contrast against the panel — a thumb tinted like a
     border reads as "no scrollbar at all" on a dark surface. */
  --ts-scrollbar:#5c5c60;
  --ts-scrollbar-hover:#7b7b80;

  --ts-radius-sm:5px;
  --ts-radius:7px;
  --ts-radius-lg:10px;

  /* Room the fullscreen close button occupies in the top-right corner: its
     34px box plus the 12px inset on each side. Any bar or header that runs to
     the top edge reserves this much on its right so the two never overlap. */
  --ts-fs-safe-right:58px;
  --ts-fs-topbar:42px;

  --ts-font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --ts-fs-xs:10px;
  --ts-fs-sm:11px;
  --ts-fs:12px;
  --ts-fs-lg:13px;

  --ts-shadow-tint:rgba(0,0,0,.34);
  --ts-shadow:0 6px 18px var(--ts-shadow-tint);
  --ts-shadow-sm:0 3px 9px var(--ts-shadow-tint);
  /* Neutral transparency checkerboard — shared by every image/media surface. */
  --ts-checker:repeating-conic-gradient(#242426 0% 25%,#1b1b1c 0% 50%) 50%/20px 20px;
}
/* Derived values. Two families:
     • accents — one knob (--ts-accent) drives the whole ramp;
     • neutrals and semantics — mixed against --ts-text / --ts-bg so they
       invert with the user's ComfyUI palette instead of assuming a dark one.
   Custom properties never fall back on their own (an unsupported color-mix
   would compute to garbage at use time, not revert), hence the @supports
   guard: the statics above stand until the browser proves it can do better. */
@supports (color:color-mix(in srgb,red,blue)){
  :root,.${TS_UI_CLASS}{
    --ts-accent-strong:color-mix(in srgb,var(--ts-accent) 84%,#000);
    --ts-accent-dim:color-mix(in srgb,var(--ts-accent) 66%,#000);
    --ts-accent-soft:color-mix(in srgb,var(--ts-accent) 17%,transparent);
    --ts-accent-line:color-mix(in srgb,var(--ts-accent) 45%,transparent);
    --ts-accent-contrast:color-mix(in srgb,var(--ts-accent) 14%,#000);

    /* Toward the text colour = away from the background, in either theme. */
    --ts-surface-hover:color-mix(in srgb,var(--ts-surface) 88%,var(--ts-text));
    --ts-surface-active:color-mix(in srgb,var(--ts-surface) 78%,var(--ts-text));
    --ts-elevated:color-mix(in srgb,var(--ts-bg) 93%,var(--ts-text));
    /* Recessed wells follow ComfyUI's own input background. */
    --ts-sunken:color-mix(in srgb,var(--ts-surface) 94%,var(--ts-bg));
    --ts-faint:color-mix(in srgb,var(--ts-muted) 72%,var(--ts-bg));
    --ts-border-soft:color-mix(in srgb,var(--ts-border) 55%,var(--ts-bg));
    --ts-border-strong:color-mix(in srgb,var(--ts-border) 72%,var(--ts-text));
    --ts-checker:repeating-conic-gradient(
      color-mix(in srgb,var(--ts-bg) 93%,var(--ts-text)) 0% 25%,
      color-mix(in srgb,var(--ts-bg) 98%,var(--ts-text)) 0% 50%) 50%/20px 20px;
    /* Semantic hues are pulled toward the text colour so they stay legible on
       a light background too (where the dark-tuned pastels would wash out). */
    /* Blending the shadow toward the page background keeps it a heavy drop on
       a dark theme and a soft card lift on a light one. */
    --ts-shadow-tint:color-mix(in srgb,color-mix(in srgb,#000 80%,var(--ts-bg)) 30%,transparent);
    --ts-scrollbar:color-mix(in srgb,var(--ts-text) 36%,var(--ts-bg));
    --ts-scrollbar-hover:color-mix(in srgb,var(--ts-text) 58%,var(--ts-bg));
    --ts-danger:color-mix(in srgb,#d0625c 74%,var(--ts-text));
    --ts-success:color-mix(in srgb,#5aa87a 74%,var(--ts-text));
    --ts-warning:color-mix(in srgb,#c99446 74%,var(--ts-text));
  }
}

.${TS_UI_CLASS}{color:var(--ts-text);font-family:var(--ts-font);font-size:var(--ts-fs);box-sizing:border-box}
.${TS_UI_CLASS} *,.${TS_UI_CLASS} *::before,.${TS_UI_CLASS} *::after{box-sizing:border-box}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.ts-ui-btn{display:inline-flex;align-items:center;justify-content:center;gap:5px;
  border:1px solid var(--ts-border);background:var(--ts-surface);color:var(--ts-text);
  border-radius:var(--ts-radius);padding:6px 11px;font-family:var(--ts-font);
  font-size:var(--ts-fs-sm);font-weight:600;letter-spacing:.01em;line-height:1.2;
  cursor:pointer;white-space:nowrap;transition:background .12s ease,border-color .12s ease}
.ts-ui-btn:hover:not([disabled]){background:var(--ts-surface-hover);border-color:var(--ts-border-strong)}
.ts-ui-btn:active:not([disabled]){background:var(--ts-surface-active)}
.ts-ui-btn[disabled]{opacity:.42;cursor:not-allowed}
.ts-ui-btn:focus-visible{outline:2px solid var(--ts-accent);outline-offset:1px}
.ts-ui-btn--primary{background:var(--ts-accent);border-color:var(--ts-accent-strong);color:var(--ts-accent-contrast)}
.ts-ui-btn--primary:hover:not([disabled]){background:var(--ts-accent-strong);border-color:var(--ts-accent-strong)}
.ts-ui-btn--primary:active:not([disabled]){background:var(--ts-accent-dim)}
.ts-ui-btn--danger{color:var(--ts-danger);border-color:var(--ts-border)}
.ts-ui-btn--danger:hover:not([disabled]){background:var(--ts-surface-hover);border-color:var(--ts-danger)}
.ts-ui-btn--ghost{background:transparent;border-color:transparent}
.ts-ui-btn--ghost:hover:not([disabled]){background:var(--ts-surface);border-color:var(--ts-border-soft)}
.ts-ui-btn--icon{padding:0;width:28px;height:28px;flex:0 0 auto}
.ts-ui-btn--icon svg{width:14px;height:14px;fill:currentColor;pointer-events:none}
.ts-ui-btn.is-active{background:var(--ts-accent-soft);border-color:var(--ts-accent-line);color:var(--ts-text)}

/* ── Chrome: toolbars, panels, status bars ───────────────────────────── */
.ts-ui-toolbar{display:flex;align-items:center;gap:8px;padding:6px 8px;
  background:var(--ts-bg);border:1px solid var(--ts-border-soft);border-radius:var(--ts-radius-lg)}
.ts-ui-group{display:flex;align-items:center;gap:6px}
.ts-ui-sep{width:1px;align-self:stretch;background:var(--ts-border-soft);margin:2px 2px}
.ts-ui-panel{background:var(--ts-elevated);border:1px solid var(--ts-border-soft);
  border-radius:var(--ts-radius-lg);box-shadow:var(--ts-shadow);padding:10px}
.ts-ui-title{font-size:var(--ts-fs-xs);color:var(--ts-muted);text-transform:uppercase;
  letter-spacing:.07em;font-weight:700}
.ts-ui-statusbar{display:flex;align-items:center;justify-content:space-between;gap:10px;
  padding:6px 10px;font-size:var(--ts-fs-sm);color:var(--ts-muted);background:var(--ts-bg);
  border:1px solid var(--ts-border-soft);border-radius:var(--ts-radius)}
.ts-ui-statusbar.is-error,.ts-ui-status.is-error{color:var(--ts-danger)}
.ts-ui-statusbar.is-success,.ts-ui-status.is-success{color:var(--ts-success)}
.ts-ui-ellipsis{flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ts-ui-meta{font-variant-numeric:tabular-nums;color:var(--ts-faint);font-size:var(--ts-fs-xs);white-space:nowrap}

/* ── Fields ──────────────────────────────────────────────────────────── */
.ts-ui-field{display:flex;flex-direction:column;gap:4px}
.ts-ui-field__row{display:flex;align-items:center;justify-content:space-between;gap:6px}
.ts-ui-field__name{color:var(--ts-muted);font-size:var(--ts-fs-xs);text-transform:uppercase;
  letter-spacing:.05em;font-weight:600}
.ts-ui-field__value{font-variant-numeric:tabular-nums;font-weight:600;font-size:var(--ts-fs-sm);color:var(--ts-text)}
.ts-ui-input,.ts-ui-select,.ts-ui-textarea{background:var(--ts-sunken);color:var(--ts-text);
  border:1px solid var(--ts-border-soft);border-radius:var(--ts-radius-sm);padding:5px 8px;
  font-family:var(--ts-font);font-size:var(--ts-fs-sm);outline:none;width:100%}
.ts-ui-input:focus,.ts-ui-select:focus,.ts-ui-textarea:focus{border-color:var(--ts-accent-line)}
/* Keyboard users need to SEE the caret's home. A border tint alone is easy to
   miss on a dark surface, so give fields the same ring the buttons get. */
.ts-ui-input:focus-visible,.ts-ui-select:focus-visible,.ts-ui-textarea:focus-visible{
  outline:2px solid var(--ts-accent);outline-offset:1px}
.ts-ui-textarea{resize:vertical;line-height:1.45}
.ts-ui-label{font-size:var(--ts-fs-xs);color:var(--ts-muted);text-transform:uppercase;
  letter-spacing:.06em;white-space:nowrap}

/* ── Sliders ─────────────────────────────────────────────────────────── */
.ts-ui-slider{-webkit-appearance:none;appearance:none;width:100%;height:4px;border-radius:999px;
  background:var(--ts-border-soft);outline:none;cursor:pointer}
.ts-ui-slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:13px;height:13px;
  border-radius:999px;background:var(--ts-accent);border:2px solid var(--ts-modal-bg);cursor:pointer}
.ts-ui-slider::-moz-range-thumb{width:13px;height:13px;border-radius:999px;background:var(--ts-accent);
  border:2px solid var(--ts-modal-bg);cursor:pointer}
.ts-ui-slider:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:3px}

/* ── Fullscreen editor shell ─────────────────────────────────────────── */
.ts-ui-modal{position:fixed;inset:0;z-index:11000;display:flex;background:var(--ts-modal-bg);
  color:var(--ts-text);font-family:var(--ts-font)}
/* Two different shapes share this shell. An EDITOR fills the viewport, so the
   default flex-start/stretch is exactly right. A DIALOG (a report, a summary,
   a confirmation) is a panel of its own size — left unaligned it pins itself to
   the top-left corner and reads as a stray tooltip, while the node the user
   just clicked sits in the middle of the screen. Centring is opt-in via
   openFullscreenOverlay's "center" option so the editors keep their full bleed.
   NOTE: this whole stylesheet is a JS template literal — never put a backtick
   in a comment here, it ends the string and breaks the module. */
.ts-ui-modal--center{align-items:center;justify-content:center;padding:24px}
/* Sized to the panel it wraps, purely so the close button has something its
   own size to sit on instead of the far corner of the screen. */
.ts-ui-fs-frame{position:relative;display:flex;min-height:0;max-width:100%;max-height:100%;
  animation:ts-ui-dialog-in .16s ease-out}
.ts-ui-modal--center .ts-ui-fs-close{box-shadow:none}
@keyframes ts-ui-dialog-in{from{opacity:0;transform:translateY(8px) scale(.985)}
  to{opacity:1;transform:none}}
/* Unified fullscreen close button — every editor opened via openFullscreenOverlay
   gets the SAME control in the SAME spot (top-right), on a bar of its own. */
.ts-ui-fs-topbar{position:absolute;top:0;left:0;right:0;height:var(--ts-fs-topbar,42px);
  z-index:11050;display:flex;align-items:center;justify-content:flex-end;gap:10px;
  padding:0 10px;background:var(--ts-surface);border-bottom:1px solid var(--ts-border)}
.ts-ui-fs-title{margin-right:auto;font-size:var(--ts-fs-sm);color:var(--ts-muted);
  letter-spacing:.02em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* Содержимое начинается ПОД полосой — потому налезать на кнопку больше нечему.
   Отступом, а не margin: содержимое редакторов растянуто на всю коробку по
   высоте, и внешний отступ у такого элемента ничего не сдвинет. */
.ts-ui-modal.ts-ui-fs-hastopbar{padding-top:var(--ts-fs-topbar,42px);box-sizing:border-box}
/* У ДИАЛОГА полосы нет — только кнопка на его собственном углу.
   Полоса придумана для редакторов, растянутых во весь экран: там она —
   единственная поверхность наверху, и содержимому есть подо что уйти. У
   диалога поверхность своя, панель, и полоса вставала НАД ней отдельной серой
   плашкой в пустоте: висит сверху, ни к чему не прилегает, читается как
   сломанная графика. Диалог держит заголовок сам (и место под кнопку справа —
   см. .ts-fdl__head), поэтому имя окна ему передавать не нужно.
   NOTE: обратная кавычка внутри этого комментария оборвала бы шаблонную
   строку — весь stylesheet живёт внутри неё. */
.ts-ui-fs-frame.ts-ui-fs-hastopbar{padding-top:0}
.ts-ui-modal--center .ts-ui-fs-topbar{height:auto;top:8px;left:auto;right:8px;
  padding:0;background:none;border-bottom:0}
.ts-ui-fs-close{width:30px;height:30px;flex:0 0 auto}
/* Под полосой резервировать угол больше не нужно: кнопка закрытия туда не
   попадает. Резерв остаётся объявленным для редакторов без полосы, а здесь
   схлопывается до обычного поля — иначе у каждой шапки справа так и висела бы
   мёртвая полоса в 58 px. */
.ts-ui-modal.ts-ui-fs-hastopbar,.ts-ui-fs-frame.ts-ui-fs-hastopbar{--ts-fs-safe-right:12px}
/* Возврат с внутреннего экрана редактора. Стоит СЛЕВА, у названия экрана:
   справа вверху живёт закрытие всей студии, и две одинаковые кнопки рядом
   заставляли бы гадать, какая из них что закроет. */
.ts-ui-back{width:28px;height:28px;flex:0 0 auto}
/* Hidden focus anchor: parks keyboard focus so ComfyUI's graph hotkeys
   (Ctrl+Z → ChangeTracker) stay out of an open editor. See CLAUDE.md §12.5. */
.ts-ui-keyanchor{position:fixed;left:-9999px;top:-9999px;width:1px;height:1px;opacity:0;
  pointer-events:none;resize:none}
/* Hidden file inputs are parked off-screen, never display:none — some browsers
   refuse a programmatic .click() on a collapsed input (CLAUDE.md §12.5.11). */
.ts-ui-file{position:fixed;left:-9999px;top:-9999px;width:1px;height:1px;opacity:0;pointer-events:none}

/* ── Feedback: scrim, spinner, drop target ───────────────────────────── */
.ts-ui-scrim{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
  flex-direction:column;gap:10px;background:var(--ts-scrim);backdrop-filter:blur(2px);
  color:var(--ts-text);font-size:var(--ts-fs);pointer-events:none;z-index:20}
.ts-ui-scrim.is-active{display:flex}
.ts-ui-spinner{width:26px;height:26px;border-radius:999px;border:3px solid var(--ts-border-soft);
  border-top-color:var(--ts-accent);animation:ts-ui-spin .9s linear infinite}
@keyframes ts-ui-spin{to{transform:rotate(360deg)}}
.ts-ui-checker{background:var(--ts-checker)}
.ts-ui-drop{position:absolute;inset:8px;display:none;align-items:center;justify-content:center;
  border:2px dashed var(--ts-accent-line);border-radius:var(--ts-radius-lg);background:var(--ts-accent-soft);
  color:var(--ts-text);font-size:var(--ts-fs-lg);font-weight:600;pointer-events:none;z-index:24}
.is-drag-over>.ts-ui-drop{display:flex}
.is-drag-over{outline:2px dashed var(--ts-accent-line);outline-offset:-3px}

/* ── Ratio cards: the one way to choose proportions ───────────────────
   Wherever the pack asks "what shape should the frame be" — the resolution
   selector's node, the studio's frame size, the new frame in outpaint — it
   asks with the same control: a grid of cards, each holding a rectangle of
   that exact proportion above its label.

   Three columns, rows flow. Nine values give the 3x3 the selector is built
   around; seven give 3+3+1, and nothing has to be told how many there are.

   The card is a fixed-height box and the rectangle inside is proportional,
   rather than the button itself being the rectangle: a row of naked
   rectangles of wildly different shapes cannot line up, and the labels end
   up at different heights. The frame carries the proportion, the card
   carries the alignment. */
.ts-ui-ratios{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;width:100%}
.ts-ui-ratio{display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:4px;min-width:0;padding:5px 3px;border:1px solid var(--ts-border-soft);
  border-radius:var(--ts-radius);background:var(--ts-surface);color:inherit;cursor:pointer;
  transition:border-color .15s ease,box-shadow .15s ease,background .15s ease}
.ts-ui-ratio:hover:not(:disabled){border-color:var(--ts-border-strong);background:var(--ts-surface-hover)}
.ts-ui-ratio.is-selected{border-color:var(--ts-accent);box-shadow:0 0 0 1px var(--ts-accent-line);
  background:var(--ts-accent-soft)}
.ts-ui-ratio:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
.ts-ui-ratio:disabled{cursor:default}
/* ⚠️ THE WRAP MUST BE SQUARE, and that is the whole trick.
   The frame's two sides are given in per cent, and per cent resolves against
   DIFFERENT bases: width against the wrap's width, height against its height.
   The first version made the wrap full-card-width and 26 px tall, so 16:9 came
   out as roughly 80x15 — a letterbox slot, not a 16:9 frame. Every shape was
   squashed by the wrap's own proportion.
   A square base makes "100% by 56%" mean exactly 16:9. Both sides of the square
   come from one token, so a 9:21 and a 21:9 occupy the same room and every label
   sits on one line. */
.ts-ui-ratio__wrap{display:flex;align-items:center;justify-content:center;
  width:var(--ts-ratio-box,34px);height:var(--ts-ratio-box,34px);margin:0 auto;
  flex:0 0 auto}
.ts-ui-ratio__frame{border-radius:3px;
  border:1px solid var(--ts-muted);background:var(--ts-elevated)}
.ts-ui-ratio.is-selected .ts-ui-ratio__frame{border-color:var(--ts-accent)}
.ts-ui-ratio__label{font-size:var(--ts-fs-xs);letter-spacing:.02em;color:var(--ts-muted);
  line-height:1;white-space:nowrap}
.ts-ui-ratio.is-selected .ts-ui-ratio__label{color:var(--ts-accent)}
.ts-ui-ratios.is-disabled{opacity:.4}

/* Compact form of the same control: a trigger that shows what is chosen, and
   the grid only while it is open. A side panel has one column of vertical room
   and a dozen other controls in it — nine cards standing open all day cost more
   of it than they are worth. */
.ts-ui-ratiopick{position:relative;display:flex;align-items:center;gap:6px}
.ts-ui-ratiopick__trigger{display:flex;align-items:center;gap:6px;padding:3px 7px 3px 5px;
  border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);
  background:var(--ts-surface);color:var(--ts-text);cursor:pointer;
  font-size:var(--ts-fs-sm);line-height:1;flex:0 0 auto}
.ts-ui-ratiopick__trigger:hover{border-color:var(--ts-border-strong)}
.ts-ui-ratiopick__trigger.is-active{border-color:var(--ts-accent);background:var(--ts-accent-soft)}
.ts-ui-ratiopick__trigger:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
.ts-ui-ratiopick__trigger:disabled{cursor:default;opacity:.5}
/* The preview is the same square-based frame as a card's, just smaller — the
   selected shape is the label, so it must be the shape it names. */
.ts-ui-ratiopick__trigger .ts-ui-ratio__wrap{--ts-ratio-box:18px;margin:0}
.ts-ui-ratiopick__value{font-variant-numeric:tabular-nums}
.ts-ui-ratiopick__caret{color:var(--ts-muted);font-size:9px}
.ts-ui-ratiopick__custom{width:58px;flex:0 0 auto;height:24px;padding:0 6px;
  font-size:var(--ts-fs-xs);text-align:center}
.ts-ui-ratiopick__pop{position:absolute;z-index:40;top:calc(100% + 4px);left:0;
  display:none;min-width:186px;padding:8px;background:var(--ts-elevated);
  border:1px solid var(--ts-border);border-radius:var(--ts-radius);box-shadow:var(--ts-shadow)}
.ts-ui-ratiopick__pop.is-open{display:block}

/* ── Launcher: the one way to open a node's fullscreen editor ─────────
   Same label, same look, horizontally centred in every node that has one,
   so the control is found in the same place across the pack. */
.ts-ui-launchbar{display:flex;align-items:center;justify-content:center;gap:8px;
  width:100%;flex:0 0 auto}
.ts-ui-launch{padding:7px 16px;font-size:var(--ts-fs);letter-spacing:.01em}
.ts-ui-launch svg{width:15px;height:15px;fill:currentColor;pointer-events:none}

/* ── Scrollbars inside TS surfaces ─────────────────────────────────────
   Standard properties are authoritative in Chromium 121+, Firefox and
   Safari 18.2+. Chromium IGNORES the -webkit- pseudo-elements as soon as
   scrollbar-width/-color is set, so the legacy block is fenced behind
   @supports instead of sitting alongside them. */
.${TS_UI_CLASS},.${TS_UI_CLASS} *{scrollbar-width:thin;
  scrollbar-color:var(--ts-scrollbar) transparent}
@supports not (scrollbar-color: auto){
  .${TS_UI_CLASS} ::-webkit-scrollbar,.${TS_UI_CLASS}::-webkit-scrollbar{width:10px;height:10px}
  .${TS_UI_CLASS} ::-webkit-scrollbar-track,.${TS_UI_CLASS}::-webkit-scrollbar-track{background:transparent}
  .${TS_UI_CLASS} ::-webkit-scrollbar-thumb,.${TS_UI_CLASS}::-webkit-scrollbar-thumb{
    background:var(--ts-scrollbar);border-radius:999px;border:2px solid transparent;
    background-clip:content-box}
  .${TS_UI_CLASS} ::-webkit-scrollbar-thumb:hover,.${TS_UI_CLASS}::-webkit-scrollbar-thumb:hover{
    background:var(--ts-scrollbar-hover);background-clip:content-box}
}

/* Motion is a preference, not a given: vestibular disorders make sliding and
   spinning genuinely unpleasant. Everything here is decoration — the spinner
   keeps a slow turn so "busy" still reads, and nothing else moves. */
@media (prefers-reduced-motion: reduce){
  .${TS_UI_CLASS} *,.${TS_UI_CLASS}::before,.${TS_UI_CLASS}::after{
    transition-duration:.01ms !important;animation-duration:.01ms !important;
    animation-iteration-count:1 !important;scroll-behavior:auto !important}
  .${TS_UI_CLASS} .ts-ui-spinner{animation:ts-ui-spin 1.8s linear infinite !important}
}
`;
}

// ---------------------------------------------------------------------------
// Ratio cards — one control for every "what shape?" question in the pack
// ---------------------------------------------------------------------------

/**
 * Width / height of a "w:h" string, or null when it is not one.
 *
 * Used both to draw a card and to DECIDE whether a list of options is a list
 * of proportions at all — that is how the studio's generic choice control
 * knows to draw cards for the outpaint frame and plain buttons for
 * Refine / Replace, without a manifest having to say so.
 *
 * @param {string} text e.g. "16:9"
 * @returns {{w: number, h: number, ratio: number}|null}
 */
export function parseRatioText(text) {
    const parts = String(text ?? "").trim().split(":");
    if (parts.length !== 2) return null;
    const w = Number(parts[0]);
    const h = Number(parts[1]);
    if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return null;
    return { w, h, ratio: w / h };
}

/** True when EVERY value is a proportion — one non-ratio and it is a plain list. */
export function isRatioList(values) {
    const list = Array.from(values || []);
    return list.length > 0 && list.every((value) => parseRatioText(value) !== null);
}

/**
 * A grid of proportion cards.
 *
 * @param {object} options
 * @param {string[]} [options.values] Proportions as "w:h".
 * @param {(value: string) => void} [options.onSelect] Called on click, never on `select()`.
 * @param {number} [options.boxSize=34] Side of the SQUARE box each frame is fitted into.
 * @param {(value: string) => string} [options.label] Card caption; defaults to the value.
 * @returns {{element: HTMLElement, select: Function, selected: Function, add: Function,
 *            has: Function, values: Function, setDisabled: Function, buttons: Map}}
 */
/**
 * The square box with one proportional frame inside it.
 *
 * Both sides of the frame are given in per cent OF THAT SQUARE, which is what
 * makes the drawn shape exactly the written one. ⚠️ The square matters: per
 * cent resolves width against the box's width and height against its height, so
 * a non-square box multiplies every shape by its own proportion — the first
 * version was full-card-width by 26 px tall, and 16:9 came out a letterbox slit.
 *
 * @param {string} value Proportion as "w:h".
 * @returns {HTMLElement} The wrap element (empty when the value is not a ratio).
 */
export function createRatioFrame(value) {
    const wrap = document.createElement("div");
    wrap.className = "ts-ui-ratio__wrap";
    const parsed = parseRatioText(value);
    if (!parsed) return wrap;
    const frame = document.createElement("div");
    frame.className = "ts-ui-ratio__frame";
    const long = 100;
    const short = Math.max(12, Math.round((long * Math.min(parsed.w, parsed.h))
        / Math.max(parsed.w, parsed.h)));
    const [w, h] = parsed.ratio >= 1 ? [long, short] : [short, long];
    frame.style.width = `${w}%`;
    frame.style.height = `${h}%`;
    wrap.appendChild(frame);
    return wrap;
}


export function createRatioCards({ values = [], onSelect, boxSize = 34, label } = {}) {
    ensureThemeStyles();
    const element = document.createElement("div");
    element.className = "ts-ui-ratios";
    // Сторона КВАДРАТА, в который вписывается рамка (см. .ts-ui-ratio__wrap):
    // проценты рамки считаются от него, поэтому база обязана быть квадратной.
    element.style.setProperty("--ts-ratio-box", `${boxSize}px`);
    const buttons = new Map();
    let current = "";

    function add(value) {
        const parsed = parseRatioText(value);
        if (!parsed || buttons.has(value)) return buttons.get(value) || null;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-ui-ratio";
        button.dataset.value = value;

        const wrap = createRatioFrame(value);

        const caption = document.createElement("div");
        caption.className = "ts-ui-ratio__label";
        caption.textContent = label ? label(value) : value;

        button.append(wrap, caption);
        button.addEventListener("click", () => {
            select(value);
            onSelect?.(value);
        });
        element.appendChild(button);
        buttons.set(value, button);
        if (value === current) button.classList.add("is-selected");
        return button;
    }

    function select(value) {
        current = String(value ?? "");
        for (const [key, button] of buttons) {
            button.classList.toggle("is-selected", key === current);
        }
    }

    for (const value of values) add(value);

    return {
        element,
        buttons,
        add,
        select,
        selected: () => current,
        has: (value) => buttons.has(value),
        values: () => Array.from(buttons.keys()),
        setDisabled: (disabled) => {
            const off = Boolean(disabled);
            element.classList.toggle("is-disabled", off);
            for (const button of buttons.values()) button.disabled = off;
        },
    };
}


/**
 * The compact form: a trigger that shows the chosen shape, and the grid only
 * while it is open.
 *
 * Same cards, same geometry — this only decides when they are on screen. A side
 * panel has one column of vertical room and a dozen controls competing for it,
 * so nine cards standing open all day cost more than they are worth; a node with
 * a resizable body is the opposite case and keeps the open grid.
 *
 * @param {object} options
 * @param {string[]} [options.values] Proportions as "w:h".
 * @param {(value: string) => void} [options.onSelect] Called on a person's pick.
 * @param {(value: string) => void} [options.onCustom] Enables the small "w:h"
 *   field to the right of the trigger; called with the accepted proportion.
 * @param {string} [options.customPlaceholder="w:h"]
 * @param {string} [options.customTitle]
 * @param {number} [options.boxSize=34] Side of the square inside each card.
 * @returns {{element: HTMLElement, select: Function, selected: Function, add: Function,
 *            has: Function, values: Function, setDisabled: Function, close: Function,
 *            cards: object, trigger: HTMLElement}}
 */
export function createRatioPicker({
    values = [],
    onSelect,
    onCustom,
    customPlaceholder = "w:h",
    customTitle,
    boxSize = 34,
} = {}) {
    ensureThemeStyles();
    const element = document.createElement("div");
    element.className = "ts-ui-ratiopick";

    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "ts-ui-ratiopick__trigger";
    let preview = createRatioFrame("1:1");
    const valueText = document.createElement("span");
    valueText.className = "ts-ui-ratiopick__value";
    const caret = document.createElement("span");
    caret.className = "ts-ui-ratiopick__caret";
    caret.textContent = "▾";
    trigger.append(preview, valueText, caret);

    const pop = document.createElement("div");
    pop.className = "ts-ui-ratiopick__pop";

    const cards = createRatioCards({
        values,
        boxSize,
        onSelect: (value) => {
            paint(value);
            close();
            onSelect?.(value);
        },
    });
    pop.appendChild(cards.element);
    element.append(trigger, pop);

    if (onCustom) {
        const custom = document.createElement("input");
        custom.type = "text";
        custom.className = "ts-ui-input ts-ui-ratiopick__custom";
        custom.placeholder = customPlaceholder;
        if (customTitle) custom.title = customTitle;
        custom.addEventListener("change", () => {
            // "16 9", "16x9", "16,9" all mean the same thing to a person.
            const text = custom.value.trim().replace(/[\s,x×]+/g, ":");
            const parsed = parseRatioText(text);
            custom.value = "";
            if (!parsed) return;
            const value = `${parsed.w}:${parsed.h}`;
            cards.add(value);
            paint(value);
            onCustom(value);
        });
        element.appendChild(custom);
    }

    function paint(value) {
        cards.select(value);
        valueText.textContent = String(value ?? "");
        const fresh = createRatioFrame(value);
        trigger.replaceChild(fresh, preview);
        preview = fresh;
    }

    // The listener lives only while the popover is open, and removes itself if
    // the control was taken off the page in between (the studio rebuilds its
    // deck on every model change, and a listener outliving its panel is a leak).
    function onDocumentDown(event) {
        if (!element.isConnected) return close();
        if (!element.contains(event.target)) close();
    }
    function onKeyDown(event) {
        if (event.key === "Escape") close();
    }

    function open() {
        pop.classList.add("is-open");
        trigger.classList.add("is-active");
        document.addEventListener("pointerdown", onDocumentDown);
        document.addEventListener("keydown", onKeyDown);
    }

    function close() {
        pop.classList.remove("is-open");
        trigger.classList.remove("is-active");
        document.removeEventListener("pointerdown", onDocumentDown);
        document.removeEventListener("keydown", onKeyDown);
    }

    trigger.addEventListener("click", () => {
        if (pop.classList.contains("is-open")) close();
        else open();
    });

    return {
        element,
        trigger,
        cards,
        close,
        add: (value) => cards.add(value),
        has: (value) => cards.has(value),
        values: () => cards.values(),
        selected: () => cards.selected(),
        select: paint,
        setDisabled: (disabled) => {
            const off = Boolean(disabled);
            trigger.disabled = off;
            cards.setDisabled(off);
            for (const field of element.querySelectorAll("input")) field.disabled = off;
            if (off) close();
        },
    };
}

// ---------------------------------------------------------------------------
// Unified editor launcher
// ---------------------------------------------------------------------------
// Nodes whose real UI lives in a fullscreen editor all open it the same way:
// one centred button, worded identically. Keeping the label and the factory
// here means a future node cannot invent its own ("Edit Image", "Редактировать"
// and friends were the previous state of affairs).

const OPEN_INTERFACE_LABELS = {
    en: "Open Interface",
    ru: "Открыть интерфейс",
};

/** Two-letter UI language: ComfyUI's own setting first, then the browser. */
export function getUiLanguage() {
    try {
        const app = window.comfyAPI?.app?.app || window.app;
        const locale = app?.extensionManager?.setting?.get?.("Comfy.Locale");
        if (typeof locale === "string" && locale) return locale.slice(0, 2).toLowerCase();
    } catch {
        // Setting store not ready (or not present on older frontends).
    }
    return String(navigator?.language || "en").slice(0, 2).toLowerCase();
}

/**
 * Label for the launcher button.
 * @param {string} [lang] Force a language (nodes that localise their whole
 *   panel from document data pass their own, so the button matches its
 *   surroundings instead of the global UI locale).
 */
export function getOpenInterfaceLabel(lang) {
    const key = String(lang || getUiLanguage()).slice(0, 2).toLowerCase();
    return OPEN_INTERFACE_LABELS[key] || OPEN_INTERFACE_LABELS.en;
}

/**
 * Resolve a per-node UI dictionary against the current ComfyUI locale.
 *
 * Usage (top of a node module):
 *     const STRINGS = {
 *         en: { load: "Load Image", uploading: "Uploading..." },
 *         ru: { load: "Загрузить изображение", uploading: "Загрузка..." },
 *     };
 *     // inside setup(): resolve lazily so the settings store is ready
 *     const L = pickLocaleStrings(STRINGS);
 *     button.textContent = L.load;
 *
 * Merging is PER KEY: a key missing from the active language falls back to its
 * English value instead of rendering `undefined`. Values may be strings or
 * functions (for parameterised messages: `saved: (name) => \`Saved: ${name}\``).
 *
 * A locale change reloads the ComfyUI page (verified behaviour of the Vue
 * frontend), so resolving once per setup is correct — no live re-render needed.
 * Log messages (console.*, logging) stay English by project convention; this
 * helper is for USER-VISIBLE text only.
 *
 * @template {Record<string, any>} T
 * @param {{en: T} & Record<string, Partial<T>>} dictionaries
 * @returns {T}
 */
export function pickLocaleStrings(dictionaries) {
    const base = dictionaries?.en || {};
    const localized = dictionaries?.[getUiLanguage()] || {};
    return { ...base, ...localized };
}

function launchIconSvg() {
    // Material Design "open_in_full".
    return `<svg viewBox="0 0 24 24"><path d="M21 11V3h-8l3.29 3.29-10 10L3 13v8h8l-3.29-3.29 10-10z"/></svg>`;
}

/**
 * Build the standard "open the fullscreen editor" button.
 * @param {(event: MouseEvent) => void} onOpen
 * @param {{lang?: string, icon?: boolean, description?: string}} [options]
 *   description — optional tooltip explaining what THIS node's editor does.
 *   The label is shared; what the editor is for legitimately differs per node.
 * @returns {HTMLButtonElement}
 */
/**
 * Кнопка «назад» для экрана, живущего внутри редактора (наборы, настройки,
 * справка). Одна стрелка и один размер на весь пак.
 *
 * @param {string} label подпись для подсказки и озвучки
 * @param {() => void} [onBack]
 * @returns {HTMLButtonElement}
 */
export function createPanelBackButton(label, onBack) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ts-ui-btn ts-ui-btn--icon ts-ui-back";
    button.title = label;
    button.setAttribute("aria-label", label);
    button.innerHTML = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none"'
        + ' stroke="currentColor" stroke-width="1.7" stroke-linecap="round"'
        + ' stroke-linejoin="round"><path d="M9.5 3.5 5 8l4.5 4.5"/></svg>';
    if (typeof onBack === "function") button.addEventListener("click", onBack);
    return button;
}

export function createOpenInterfaceButton(onOpen, options = {}) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ts-ui-btn ts-ui-btn--primary ts-ui-launch";
    if (options.icon !== false) button.innerHTML = launchIconSvg();
    const label = document.createElement("span");
    label.textContent = getOpenInterfaceLabel(options.lang);
    button.appendChild(label);
    if (options.description) button._tsTitleOverride = options.description;
    button.title = button._tsTitleOverride || label.textContent;
    // Nodes relabel on locale change by writing to this span.
    button._tsLabelEl = label;
    if (typeof onOpen === "function") {
        button.addEventListener("click", (event) => {
            event.stopPropagation();
            onOpen(event);
        });
    }
    return button;
}

/**
 * Update an existing launcher's wording (locale or document language change).
 * Pass `description` to refresh a localised tooltip alongside it.
 */
export function setOpenInterfaceLabel(button, lang, description) {
    const label = button?._tsLabelEl;
    if (!label) return;
    label.textContent = getOpenInterfaceLabel(lang);
    if (description !== undefined) button._tsTitleOverride = description;
    button.title = button._tsTitleOverride || label.textContent;
}

/** Inject the shared stylesheet once per document. Safe to call repeatedly. */
export function ensureThemeStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = themeCss();
    document.head.appendChild(style);
}

// Canvas-drawn widgets (sliders, waveforms, LiteGraph node bodies) cannot use
// CSS variables, so they read the resolved values instead. Cached per token set
// because getComputedStyle is not free and these run inside draw loops.
let colorCache = null;

/**
 * Resolved theme colours for canvas rendering.
 * @returns {{accent:string,accentStrong:string,accentDim:string,text:string,
 *            muted:string,faint:string,border:string,borderSoft:string,
 *            bg:string,surface:string,sunken:string,danger:string,success:string}}
 */
export function getThemeColors() {
    if (colorCache) return colorCache;
    ensureThemeStyles();
    const probe = document.createElement("div");
    probe.className = TS_UI_CLASS;
    probe.style.cssText = "position:fixed;left:-9999px;top:-9999px;width:1px;height:1px";
    document.body.appendChild(probe);
    const computed = getComputedStyle(probe);
    const read = (name, fallback) => computed.getPropertyValue(name).trim() || fallback;
    colorCache = {
        accent: read("--ts-accent", TS_ACCENT),
        accentStrong: read("--ts-accent-strong", TS_ACCENT_STRONG),
        accentDim: read("--ts-accent-dim", TS_ACCENT_DIM),
        text: read("--ts-text", "#ddd"),
        muted: read("--ts-muted", "#999"),
        faint: read("--ts-faint", "#6d6d70"),
        border: read("--ts-border", "#4e4e4e"),
        borderSoft: read("--ts-border-soft", "#333335"),
        bg: read("--ts-bg", "#171718"),
        surface: read("--ts-surface", "#222"),
        sunken: read("--ts-sunken", "#141415"),
        danger: read("--ts-danger", "#d98a86"),
        success: read("--ts-success", "#8fbf9f"),
    };
    probe.remove();
    return colorCache;
}

/** Drop the cache — call if the tokens are ever changed at runtime. */
export function resetThemeColors() {
    colorCache = null;
}

// Switching the ComfyUI colour palette does NOT reload the page (unlike the
// locale switch), so canvas-drawn widgets kept painting with the colours of the
// theme that was active when they first drew. ComfyUI applies a palette by
// rewriting the CSS variables on <html>/<body> and toggling its theme class, so
// watching those attributes is enough to know the tokens moved.
let themeWatcher = null;

function watchThemeChanges() {
    if (themeWatcher || typeof MutationObserver === "undefined") return;
    themeWatcher = new MutationObserver(() => resetThemeColors());
    // <html> carries the palette's inline custom properties; <body> only its
    // theme class. Watching body's `style` too would invalidate the cache on
    // every unrelated inline tweak, and each invalidation costs a
    // getComputedStyle probe on the next draw.
    themeWatcher.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["class", "style", "data-theme"],
    });
    if (document.body) {
        themeWatcher.observe(document.body, {
            attributes: true,
            attributeFilter: ["class", "data-theme"],
        });
    }
}

watchThemeChanges();
