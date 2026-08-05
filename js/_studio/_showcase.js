// Showcase — what a subscription actually buys, shown rather than described.
//
// Reads the public catalogue, so it works with no pass and no packs: someone
// who has just installed the pack can see every model, a real example of what
// it does, and what arrived this month. That is the whole job — the buying
// happens in the gate dialog, the installing in one button per card.
//
// Previews are cached by the browser like any image; nothing is downloaded
// until this screen is opened.

import { TS_UI_CLASS, createPanelBackButton, ensureThemeStyles,
    pickLocaleStrings } from "../_theme.js";

const STYLE_ID = "ts-studio-showcase-styles";

const STRINGS = {
    en: {
        open: "Packs",
        title: "Packs",
        close: "Close",
        free: "Free",
        included: "In the pack",
        installed: "Installed",
        update: "Update",
        install: "Install",
        remove: "Remove",
        locked: "Subscribers",
        getAccess: "Get access",
        whatsNew: "New this month",
        offline: "The catalogue could not be reached — showing what is installed.",
        empty: "No packs are published yet.",
        working: "Working…",
        failed: (reason) => `Failed: ${reason}`,
        installedOk: "Installed. The models are in the list.",
        beforeAfter: "before · after",
    },
    ru: {
        open: "Наборы",
        title: "Наборы",
        close: "Закрыть",
        free: "Бесплатно",
        included: "В наборе",
        installed: "Установлено",
        update: "Обновить",
        install: "Установить",
        remove: "Удалить",
        locked: "По подписке",
        getAccess: "Получить доступ",
        whatsNew: "Новое в этом месяце",
        offline: "Каталог недоступен — показано то, что установлено.",
        empty: "Наборы пока не опубликованы.",
        working: "Работаем…",
        failed: (reason) => `Не вышло: ${reason}`,
        installedOk: "Установлено. Модели появились в списке.",
        beforeAfter: "до · после",
    },
};

export function ensureShowcaseStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-showcase{position:absolute;inset:0;z-index:9;display:none;flex-direction:column;
    background:var(--ts-bg)}
.ts-showcase.is-open{display:flex}
.ts-showcase__head{display:flex;align-items:center;gap:8px;padding:8px 12px;
    border-bottom:1px solid var(--ts-border)}
.ts-showcase__title{font-weight:700}
.ts-showcase__body{flex:1;overflow-y:auto;padding:16px 18px;display:flex;
    flex-direction:column;gap:16px}
.ts-showcase__note{font-size:var(--ts-fs-sm);color:var(--ts-muted)}
.ts-showcase__grid{display:grid;gap:14px;
    grid-template-columns:repeat(auto-fill,minmax(240px,1fr))}
.ts-showcase__card{display:flex;flex-direction:column;border:1px solid var(--ts-border);
    border-radius:var(--ts-radius-lg);overflow:hidden;background:var(--ts-elevated)}
.ts-showcase__media{position:relative;aspect-ratio:4/3;background:var(--ts-sunken);
    overflow:hidden}
/* Out of flow inside the aspect-ratio box: a media element that fills its
   height in normal flow is what drives runaway node growth in Nodes 2.0, and
   the before/after pair has to stack anyway. */
.ts-showcase__media img,.ts-showcase__media video{position:absolute;inset:0;
    width:100%;height:100%;object-fit:cover;display:block}
.ts-showcase__after{opacity:1;transition:opacity .18s}
.ts-showcase__card:hover .ts-showcase__after{opacity:0}
.ts-showcase__hint{position:absolute;right:8px;bottom:8px;padding:2px 6px;
    border-radius:999px;font-size:var(--ts-fs-xs);
    background:var(--ts-scrim-strong);color:var(--ts-on-media)}
/* Badges sit over artwork, so each state is mixed into the scrim rather than
   used at full strength — the plate stays dark enough to read on any picture
   while its hue still comes from the theme. */
.ts-showcase__badge{position:absolute;top:8px;left:8px;padding:3px 7px;
    border-radius:999px;font-size:var(--ts-fs-xs);font-weight:600;
    background:var(--ts-scrim-strong);color:var(--ts-on-media)}
.ts-showcase__badge[data-kind="free"]{
    background:color-mix(in srgb,var(--ts-success) 55%,var(--ts-scrim-strong))}
.ts-showcase__badge[data-kind="locked"]{
    background:color-mix(in srgb,var(--ts-accent) 60%,var(--ts-scrim-strong))}
.ts-showcase__badge[data-kind="included"]{
    background:color-mix(in srgb,var(--ts-accent) 35%,var(--ts-scrim-strong))}
.ts-showcase__text{padding:10px 12px;display:flex;flex-direction:column;gap:6px;flex:1}
.ts-showcase__name{font-weight:600}
.ts-showcase__about{font-size:var(--ts-fs-sm);color:var(--ts-muted);line-height:1.45}
.ts-showcase__foot{display:flex;gap:6px;padding:0 12px 12px}
.ts-showcase__foot .ts-ui-btn{flex:1}
.ts-showcase__new{border:1px solid var(--ts-accent-line);border-radius:var(--ts-radius);
    background:var(--ts-accent-soft);padding:12px 14px;display:flex;
    flex-direction:column;gap:6px}
.ts-showcase__newtitle{font-weight:700;color:var(--ts-accent)}
.ts-showcase__status{font-size:var(--ts-fs-sm);min-height:16px}
.ts-showcase__status.is-bad{color:var(--ts-danger)}
`;
    document.head.appendChild(style);
}

function localized(value, locale) {
    if (!value) return "";
    if (typeof value === "string") return value;
    return value[locale] || value.en || "";
}

/**
 * The packs screen.
 *
 * @param {object} options
 * @param {HTMLElement} options.host Where the panel mounts (the stage).
 * @param {object} options.api ComfyUI api object.
 * @param {(data: object) => void} [options.onCatalog] Fired on every read of
 *        the catalogue, so the app can offer what is not installed yet.
 * @param {() => void} [options.onInstalled] Reload backends after a change.
 * @param {() => void} [options.onWantAccess] Open the gate dialog.
 */
export function createShowcase(options) {
    ensureShowcaseStyles();
    const t = pickLocaleStrings(STRINGS);
    const locale = STRINGS.ru === t ? "ru" : "en";
    const { api, onInstalled, onWantAccess } = options;

    const panel = document.createElement("div");
    panel.className = `${TS_UI_CLASS} ts-showcase`;
    const head = document.createElement("div");
    head.className = "ts-showcase__head";
    const title = document.createElement("span");
    title.className = "ts-showcase__title";
    title.textContent = t.title;
    const back = createPanelBackButton(t.close, () => setOpen(false));
    head.append(back, title);

    const body = document.createElement("div");
    body.className = "ts-showcase__body";
    const status = document.createElement("div");
    status.className = "ts-showcase__status";
    panel.append(head, body);
    options.host.appendChild(panel);

    let data = null;

    function card(entry) {
        const element = document.createElement("div");
        element.className = "ts-showcase__card";

        const media = document.createElement("div");
        media.className = "ts-showcase__media";
        if (entry.before && entry.after) {
            // An upscaler or a retoucher is only believable side by side, so
            // the card holds the pair and swaps on hover — no controls, no
            // slider handle to miss at this size.
            const before = document.createElement("img");
            before.src = entry.before;
            before.alt = "";
            before.loading = "lazy";
            const after = document.createElement("img");
            after.src = entry.after;
            after.alt = "";
            after.loading = "lazy";
            after.className = "ts-showcase__after";
            const hint = document.createElement("span");
            hint.className = "ts-showcase__hint";
            hint.textContent = t.beforeAfter;
            media.append(before, after, hint);
        } else if (entry.video) {
            const video = document.createElement("video");
            video.src = entry.video;
            video.muted = true;
            video.loop = true;
            video.playsInline = true;
            video.preload = "none";
            // Motion on hover only: a wall of autoplaying clips is a wall of noise.
            element.addEventListener("pointerenter", () => video.play().catch(() => {}));
            element.addEventListener("pointerleave", () => { video.pause(); });
            media.appendChild(video);
        } else if (entry.preview) {
            const image = document.createElement("img");
            image.src = entry.preview;
            image.alt = "";
            image.loading = "lazy";
            media.appendChild(image);
        }
        const badge = document.createElement("span");
        badge.className = "ts-showcase__badge";
        if (entry.installed) {
            badge.dataset.kind = "installed";
            badge.textContent = t.installed;
        } else if (!entry.tier) {
            badge.dataset.kind = "free";
            badge.textContent = t.free;
        } else if (entry.open) {
            badge.dataset.kind = "included";
            badge.textContent = t.included;
        } else {
            badge.dataset.kind = "locked";
            badge.textContent = t.locked;
        }
        media.appendChild(badge);

        const text = document.createElement("div");
        text.className = "ts-showcase__text";
        const name = document.createElement("div");
        name.className = "ts-showcase__name";
        name.textContent = localized(entry.name, locale) || entry.id;
        const about = document.createElement("div");
        about.className = "ts-showcase__about";
        about.textContent = localized(entry.about, locale);
        text.append(name, about);

        // What the button offers follows what is actually possible here: an
        // installed pack keeps working without a pass, so it is never asked to
        // buy anything — only a fetch it cannot perform sends someone to the
        // subscription.
        const foot = document.createElement("div");
        foot.className = "ts-showcase__foot";
        const wants = entry.installed ? entry.updateAvailable : true;
        const action = document.createElement("button");
        action.type = "button";
        action.className = "ts-ui-btn ts-ui-btn--primary";
        if (!wants) {
            action.textContent = t.installed;
            action.disabled = true;
        } else if (!entry.open) {
            action.textContent = t.getAccess;
            action.addEventListener("click", () => onWantAccess?.());
        } else {
            action.textContent = entry.installed ? t.update : t.install;
            action.addEventListener("click", () => install(entry, action));
        }
        foot.appendChild(action);
        if (entry.installed) {
            const drop = document.createElement("button");
            drop.type = "button";
            drop.className = "ts-ui-btn";
            drop.textContent = t.remove;
            drop.addEventListener("click", () => remove(entry));
            foot.appendChild(drop);
        }

        element.append(media, text, foot);
        return element;
    }

    async function call(path, payload) {
        const response = await api.fetchApi(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const body_ = await response.json();
        if (!response.ok) throw new Error(body_?.error || String(response.status));
        return body_;
    }

    async function install(entry, button) {
        button.disabled = true;
        status.classList.remove("is-bad");
        status.textContent = t.working;
        try {
            const result = await call("/ts_studio/packs/install", { id: entry.id });
            data = result.packs;
            status.textContent = t.installedOk;
            render();
            options.onCatalog?.(data);
            onInstalled?.();
        } catch (err) {
            status.textContent = t.failed(String(err.message || err));
            status.classList.add("is-bad");
        } finally {
            button.disabled = false;
        }
    }

    async function remove(entry) {
        try {
            const result = await call("/ts_studio/packs/remove", { id: entry.id });
            data = result.packs;
            render();
            options.onCatalog?.(data);
            onInstalled?.();
        } catch (err) {
            status.textContent = t.failed(String(err.message || err));
            status.classList.add("is-bad");
        }
    }

    function render() {
        body.textContent = "";
        if (!data) return;

        if (data.offline) {
            const note = document.createElement("div");
            note.className = "ts-showcase__note";
            note.textContent = t.offline;
            body.appendChild(note);
        }
        const news = (data.packs || []).find((entry) => entry.whatsNew);
        if (news) {
            const block = document.createElement("div");
            block.className = "ts-showcase__new";
            const heading = document.createElement("div");
            heading.className = "ts-showcase__newtitle";
            heading.textContent = t.whatsNew;
            const text = document.createElement("div");
            text.className = "ts-showcase__about";
            text.textContent = localized(news.whatsNew, locale);
            block.append(heading, text);
            body.appendChild(block);
        }

        const grid = document.createElement("div");
        grid.className = "ts-showcase__grid";
        for (const entry of data.packs || []) grid.appendChild(card(entry));
        body.appendChild(grid);
        if (!(data.packs || []).length) {
            const empty = document.createElement("div");
            empty.className = "ts-showcase__note";
            empty.textContent = t.empty;
            body.appendChild(empty);
        }
        body.appendChild(status);
    }

    async function refresh() {
        try {
            const response = await api.fetchApi("/ts_studio/packs");
            data = await response.json();
        } catch (err) {
            console.warn("[TS Studio] the catalogue is unavailable", err);
            data = { packs: [], offline: true };
        }
        render();
        options.onCatalog?.(data);
        return data;
    }

    function setOpen(open) {
        panel.classList.toggle("is-open", open);
        if (open) refresh();
    }

    return {
        element: panel,
        open: () => setOpen(true),
        close: () => setOpen(false),
        toggle: () => setOpen(!panel.classList.contains("is-open")),
        isOpen: () => panel.classList.contains("is-open"),
        refresh,
        strings: t,
        teardown: () => panel.remove(),
    };
}
