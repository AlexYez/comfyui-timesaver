// TS Studio kit — built-in help (ui-kit layer). Plan §9.2.
//
// Two levels: (1) rich hover hints — every control already carries a
// localised title; this module makes them SWITCHABLE by stashing/restoring
// title attributes across the studio root; (2) a full help panel over the
// stage, rendering the app's markdown pages (en/ru) with a tiny built-in
// converter — no external libraries, CSP-clean.

import { TS_UI_CLASS, createPanelBackButton, ensureThemeStyles } from "../_theme.js";

const STYLE_ID = "ts-studio-help-styles";
const HINTS_KEY = "ts-studio.hints";

export function ensureHelpStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-help{position:absolute;inset:0;z-index:8;display:none;flex-direction:column;
    background:var(--ts-bg)}
.ts-help.is-open{display:flex}
/* Своя шапка экрана: возврат слева, дальше название и переключатель. */
.ts-help__head{display:flex;align-items:center;gap:8px;padding:8px 12px;
    border-bottom:1px solid var(--ts-border)}
.ts-help__title{font-weight:700}
.ts-help__body{flex:1;overflow-y:auto;padding:14px 18px;max-width:760px}
.ts-help__body h1{font-size:16px;margin:0 0 8px}
.ts-help__body h2{font-size:14px;margin:14px 0 6px;color:var(--ts-accent)}
.ts-help__body h3{font-size:12px;margin:10px 0 4px}
.ts-help__body p{margin:0 0 8px;line-height:1.5;font-size:var(--ts-fs)}
.ts-help__body li{margin:0 0 4px;line-height:1.45;font-size:var(--ts-fs)}
.ts-help__body code{background:var(--ts-sunken);border-radius:3px;padding:0 4px}
.ts-help__hintrow{display:flex;align-items:center;gap:8px;margin-left:auto;
    font-size:var(--ts-fs-sm);color:var(--ts-muted)}
`;
    document.head.appendChild(style);
}

export function hintsEnabled() {
    try {
        return localStorage.getItem(HINTS_KEY) !== "off";
    } catch {
        return true;
    }
}

/** Stash or restore every [title] under root. Off = quiet UI, on = teaching UI. */
export function applyHintSetting(root, enabled) {
    try {
        localStorage.setItem(HINTS_KEY, enabled ? "on" : "off");
    } catch { /* private mode */ }
    for (const el of root.querySelectorAll("[title], [data-ts-title]")) {
        if (enabled) {
            if (el.dataset.tsTitle) {
                el.title = el.dataset.tsTitle;
                delete el.dataset.tsTitle;
            }
        } else if (el.title) {
            el.dataset.tsTitle = el.title;
            el.removeAttribute("title");
        }
    }
}

// Deliberately small markdown: headings, lists, bold, code, paragraphs.
function mdToHtml(md) {
    const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const inline = (s) => esc(s)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
    const lines = md.split(/\r?\n/);
    const out = [];
    let list = false;
    for (const line of lines) {
        const h = /^(#{1,3})\s+(.*)$/.exec(line);
        const li = /^[-*]\s+(.*)$/.exec(line);
        if (list && !li) { out.push("</ul>"); list = false; }
        if (h) out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`);
        else if (li) {
            if (!list) { out.push("<ul>"); list = true; }
            out.push(`<li>${inline(li[1])}</li>`);
        } else if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
    if (list) out.push("</ul>");
    return out.join("\n");
}

/**
 * @param {object} options {stage, t, locale, pagesBase, studioRoot}
 * @returns {{toggle, isOpen, teardown, element}}
 */
export function createHelpPanel(options) {
    ensureHelpStyles();
    const panel = document.createElement("div");
    panel.className = `${TS_UI_CLASS} ts-help`;
    const head = document.createElement("div");
    head.className = "ts-help__head";
    const title = document.createElement("span");
    title.className = "ts-help__title";
    title.textContent = options.t.help.helpHeader;
    const hintRow = document.createElement("label");
    hintRow.className = "ts-help__hintrow";
    const hintToggle = document.createElement("input");
    hintToggle.type = "checkbox";
    hintToggle.checked = hintsEnabled();
    hintToggle.addEventListener("change", () =>
        applyHintSetting(options.studioRoot, hintToggle.checked));
    hintRow.append(hintToggle, document.createTextNode(options.t.help.hintsToggle));
    const back = createPanelBackButton(options.t.help.closeLabel, () => toggle(false));
    head.append(back, title, hintRow);
    const body = document.createElement("div");
    body.className = "ts-help__body";
    panel.append(head, body);
    options.stage.appendChild(panel);

    let loaded = false;
    async function load() {
        if (loaded) return;
        loaded = true;
        try {
            const url = `${options.pagesBase}/${options.locale}.md`;
            const response = await fetch(url);
            const text = response.ok ? await response.text()
                : (await fetch(`${options.pagesBase}/en.md`)).ok
                    ? await (await fetch(`${options.pagesBase}/en.md`)).text() : "";
            body.innerHTML = mdToHtml(text || options.t.help.missing);
        } catch {
            body.textContent = options.t.help.missing;
        }
    }

    function toggle(open = !panel.classList.contains("is-open")) {
        panel.classList.toggle("is-open", open);
        if (open) load();
    }

    if (!hintsEnabled()) applyHintSetting(options.studioRoot, false);

    return { element: panel, toggle, isOpen: () => panel.classList.contains("is-open"),
             teardown: () => panel.remove() };
}
