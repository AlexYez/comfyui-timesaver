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
   Surfaces and text defer to ComfyUI's own theme variables so a user theme
   switch carries through; the literals are the stock dark-theme values and
   act as fallbacks on older frontends. */
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
  --ts-scrim:rgba(12,12,13,.72);
  --ts-modal-bg:#161617;

  --ts-text:var(--input-text,#ddd);
  --ts-muted:var(--descrip-text,#999);
  --ts-faint:#6d6d70;

  --ts-border:var(--border-color,#4e4e4e);
  --ts-border-soft:#333335;
  --ts-border-strong:#5c5c5f;

  --ts-danger:#d98a86;
  --ts-success:#8fbf9f;
  --ts-warning:#d4b483;

  --ts-radius-sm:5px;
  --ts-radius:7px;
  --ts-radius-lg:10px;

  --ts-font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --ts-fs-xs:10px;
  --ts-fs-sm:11px;
  --ts-fs:12px;
  --ts-fs-lg:13px;

  --ts-shadow:0 10px 28px rgba(0,0,0,.45);
  --ts-shadow-sm:0 4px 12px rgba(0,0,0,.35);
  /* Neutral transparency checkerboard — shared by every image/media surface. */
  --ts-checker:repeating-conic-gradient(#242426 0% 25%,#1b1b1c 0% 50%) 50%/20px 20px;
}
/* Derived accents: one knob (--ts-accent) drives the whole ramp wherever
   color-mix() is available. */
@supports (color:color-mix(in srgb,red,blue)){
  :root,.${TS_UI_CLASS}{
    --ts-accent-strong:color-mix(in srgb,var(--ts-accent) 84%,#000);
    --ts-accent-dim:color-mix(in srgb,var(--ts-accent) 66%,#000);
    --ts-accent-soft:color-mix(in srgb,var(--ts-accent) 17%,transparent);
    --ts-accent-line:color-mix(in srgb,var(--ts-accent) 45%,transparent);
    --ts-accent-contrast:color-mix(in srgb,var(--ts-accent) 14%,#000);
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

/* ── Scrollbars inside TS surfaces ───────────────────────────────────── */
.${TS_UI_CLASS} ::-webkit-scrollbar{width:10px;height:10px}
.${TS_UI_CLASS} ::-webkit-scrollbar-track{background:transparent}
.${TS_UI_CLASS} ::-webkit-scrollbar-thumb{background:var(--ts-border-soft);border-radius:999px;
  border:2px solid transparent;background-clip:content-box}
.${TS_UI_CLASS} ::-webkit-scrollbar-thumb:hover{background:var(--ts-border);background-clip:content-box}
`;
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
