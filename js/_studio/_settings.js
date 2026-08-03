// Studio settings — the handful of choices that belong to the person, not to
// a backend or a session: where the asset browser sits, and so on.
//
// Kept in localStorage rather than ComfyUI's own settings store: these are
// preferences of one fullscreen app, and mixing them into the host's settings
// list would put studio-only rows in front of every ComfyUI user.
//
// The panel renders from a declared list, so adding a setting is one entry
// here plus one reader wherever it applies.

import { TS_UI_CLASS, ensureThemeStyles, pickLocaleStrings } from "../_theme.js";

// The panel owns its own wording: it is the only place these lines appear, and
// keeping them here means adding a setting does not touch the app's table.
const STRINGS = {
    en: {
        open: "Settings",
        title: "Settings",
        close: "Close",
        browserSide: "Asset browser side",
        browserSideNote: "Which edge the library and session panel occupy.",
        sideLeft: "Left",
        sideRight: "Right",
        pass: "Subscription",
        passNote: "Installed packs keep working without it — a pass is needed "
            + "only to receive new ones.",
        passActive: (date) => `Active until ${date}`,
        passExpired: "Expired",
        passNone: "No pass on this machine",
        passEnter: "Enter a code",
        passForget: "Forget",
    },
    ru: {
        open: "Настройки",
        title: "Настройки",
        close: "Закрыть",
        browserSide: "Сторона браузера ассетов",
        browserSideNote: "У какого края располагается панель библиотеки и сессии.",
        sideLeft: "Слева",
        sideRight: "Справа",
        pass: "Подписка",
        passNote: "Установленные наборы работают и без неё — ключ нужен только "
            + "чтобы получать новые.",
        passActive: (date) => `Активна до ${date}`,
        passExpired: "Истекла",
        passNone: "Ключа на этой машине нет",
        passEnter: "Ввести код",
        passForget: "Забыть",
    },
};

/** Localised strings for the settings surface (also used for the rail button). */
export function settingsStrings() {
    return pickLocaleStrings(STRINGS);
}

const STORAGE_KEY = "ts.studio.settings";
const STYLE_ID = "ts-studio-settings-style";

/** Every setting the studio understands, with its default. */
export const SETTINGS = {
    browserSide: { default: "left", values: ["left", "right"] },
};

function readAll() {
    try {
        const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
        return raw && typeof raw === "object" ? raw : {};
    } catch {
        return {};                       // private mode, or someone else's key
    }
}

/**
 * One setting's current value, falling back to the default when it was never
 * set or holds something the studio no longer understands.
 *
 * @param {string} key
 */
export function readSetting(key) {
    const spec = SETTINGS[key];
    if (!spec) return undefined;
    const stored = readAll()[key];
    if (spec.values && !spec.values.includes(stored)) return spec.default;
    return stored === undefined ? spec.default : stored;
}

/**
 * Store one setting. Writing is best-effort: a browser that refuses storage
 * still gets a working studio for this sitting.
 *
 * @param {string} key
 * @param {*} value
 */
export function writeSetting(key, value) {
    const next = { ...readAll(), [key]: value };
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch { /* private mode */ }
}

export function ensureSettingsStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-settings{position:absolute;inset:0;z-index:9;display:none;flex-direction:column;
    background:var(--ts-bg)}
.ts-settings.is-open{display:flex}
/* The panel runs to the top edge, so its own Close keeps clear of the
   fullscreen close button in the corner. */
.ts-settings__head{display:flex;align-items:center;gap:8px;padding:8px 12px;
    padding-right:var(--ts-fs-safe-right);border-bottom:1px solid var(--ts-border)}
.ts-settings__title{font-weight:700}
.ts-settings__body{flex:1;overflow-y:auto;padding:16px 18px;max-width:560px;
    display:flex;flex-direction:column;gap:18px}
.ts-settings__row{display:flex;flex-direction:column;gap:6px}
.ts-settings__label{font-size:var(--ts-fs);font-weight:600}
.ts-settings__note{font-size:var(--ts-fs-sm);color:var(--ts-muted);line-height:1.45}
.ts-settings__choice{display:flex;gap:6px}
.ts-settings__opt{flex:1;min-width:0;padding:7px 10px;border-radius:var(--ts-radius);
    border:1px solid var(--ts-border);background:var(--ts-surface);color:var(--ts-text);
    font-size:var(--ts-fs-sm);cursor:pointer;transition:border-color .12s ease,
    background .12s ease,color .12s ease}
.ts-settings__opt:hover{border-color:var(--ts-border-strong)}
.ts-settings__opt.is-active{border-color:var(--ts-accent-line);background:var(--ts-accent-soft);
    color:var(--ts-accent)}
.ts-settings__opt:focus-visible{outline:2px solid var(--ts-accent-line);outline-offset:1px}
`;
    document.head.appendChild(style);
}

/**
 * The settings surface, mounted like the help panel: an overlay inside the
 * studio that opens over the stage.
 *
 * @param {object} options
 * @param {HTMLElement} options.host Element the panel mounts into.
 * @param {(key: string, value: *) => void} options.onChange Applied live.
 * @param {{state: () => object, clear: () => Promise<object>,
 *          prompt: () => void}} [options.pass] Subscription row; omitted in a
 *          studio built without one.
 * @returns {{open: () => void, close: () => void, toggle: () => void,
 *            isOpen: () => boolean, element: HTMLElement, teardown: () => void}}
 */
export function createSettingsPanel(options) {
    ensureSettingsStyles();
    const t = settingsStrings();
    const { onChange } = options;

    const panel = document.createElement("div");
    panel.className = `${TS_UI_CLASS} ts-settings`;

    const head = document.createElement("div");
    head.className = "ts-settings__head";
    const title = document.createElement("span");
    title.className = "ts-settings__title";
    title.textContent = t.title;
    const close = document.createElement("button");
    close.type = "button";
    close.className = "ts-ui-btn";
    close.textContent = t.close;
    close.style.marginLeft = "auto";
    close.addEventListener("click", () => setOpen(false));
    head.append(title, close);

    const body = document.createElement("div");
    body.className = "ts-settings__body";
    panel.append(head, body);

    /** A row of mutually exclusive choices for one setting. */
    function choiceRow(key, label, note, options_) {
        const row = document.createElement("div");
        row.className = "ts-settings__row";
        const caption = document.createElement("span");
        caption.className = "ts-settings__label";
        caption.textContent = label;
        const hint = document.createElement("span");
        hint.className = "ts-settings__note";
        hint.textContent = note;
        const choice = document.createElement("div");
        choice.className = "ts-settings__choice";
        const buttons = new Map();
        for (const option of options_) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "ts-settings__opt";
            button.textContent = option.label;
            button.addEventListener("click", () => {
                writeSetting(key, option.value);
                for (const [value, element] of buttons) {
                    element.classList.toggle("is-active", value === option.value);
                }
                onChange?.(key, option.value);
            });
            buttons.set(option.value, button);
            choice.appendChild(button);
        }
        const current = readSetting(key);
        buttons.get(current)?.classList.add("is-active");
        row.append(caption, hint, choice);
        return row;
    }

    body.appendChild(choiceRow(
        "browserSide", t.browserSide, t.browserSideNote,
        [{ value: "left", label: t.sideLeft },
         { value: "right", label: t.sideRight }],
    ));

    // The pass belongs here rather than in the gate dialog: the dialog is for
    // getting one, this is for seeing what you have — and for dropping it,
    // which matters on a shared or handed-on machine.
    let passState = null;
    const passRow = document.createElement("div");
    passRow.className = "ts-settings__row";
    const passLabel = document.createElement("span");
    passLabel.className = "ts-settings__label";
    passLabel.textContent = t.pass;
    const passNote = document.createElement("span");
    passNote.className = "ts-settings__note";
    passNote.textContent = t.passNote;
    const passLine = document.createElement("div");
    passLine.className = "ts-settings__choice";
    const passStatus = document.createElement("span");
    passStatus.className = "ts-settings__note";
    passStatus.style.flex = "1";
    const passAction = document.createElement("button");
    passAction.type = "button";
    passAction.className = "ts-settings__opt";
    passAction.style.flex = "0 0 auto";
    passLine.append(passStatus, passAction);
    passRow.append(passLabel, passNote, passLine);
    if (options.pass) body.appendChild(passRow);

    function renderPass() {
        const state = passState || {};
        if (state.state === "active") {
            passStatus.textContent = t.passActive(
                new Date(state.expiresAt).toLocaleDateString());
            passAction.textContent = t.passForget;
            passAction.onclick = async () => {
                passState = await options.pass.clear();
                renderPass();
            };
        } else {
            passStatus.textContent = state.state === "expired" || state.state === "revoked"
                ? t.passExpired : t.passNone;
            passAction.textContent = t.passEnter;
            passAction.onclick = () => options.pass.prompt();
        }
    }

    options.host.appendChild(panel);

    function setOpen(open) {
        panel.classList.toggle("is-open", open);
        // Read the pass as the panel opens: it can change in the gate dialog
        // while this panel sits behind it.
        if (open && options.pass) {
            Promise.resolve(options.pass.state()).then((state) => {
                passState = state;
                renderPass();
            });
        }
    }

    return {
        element: panel,
        open: () => setOpen(true),
        close: () => setOpen(false),
        toggle: () => setOpen(!panel.classList.contains("is-open")),
        isOpen: () => panel.classList.contains("is-open"),
        teardown: () => panel.remove(),
    };
}
