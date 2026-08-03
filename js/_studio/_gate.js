// Subscription gate — the studio's side of the pass.
//
// Two jobs, and deliberately no more: report what the current pass opens, and
// ask for a code when someone reaches for something it does not. Nothing here
// touches a run: installed material works with no pass and with an expired
// one, which is the whole point of the model (see nodes/_pass.py).
//
// The state is fetched once per studio session and cached: the deck is rebuilt
// on every model and mode switch, and a request per rebuild would be silly.

import { TS_UI_CLASS, ensureThemeStyles, pickLocaleStrings } from "../_theme.js";

const STYLE_ID = "ts-studio-gate-styles";

const STRINGS = {
    en: {
        locked: "Subscribers",
        lockedSuffix: (label) => `${label} — subscribers`,
        title: "Available to subscribers",
        body: "Krea 2, Ideogram and the monthly packs of recipes and models. "
            + "Everything you have already installed keeps working — a pass is "
            + "needed only to receive new material.",
        codeLabel: "Month's code",
        codePlaceholder: "TSV-…",
        activate: "Activate",
        cancel: "Close",
        where: "Where to get the code:",
        working: "Checking…",
        activeUntil: (date, days) => `Active until ${date} · ${days} days left`,
        expired: "The pass has expired — installed material still works",
        none: "No pass",
        failed: (reason) => reason,
        offlineHint: "No internet? Paste the token from the post instead of the code.",
    },
    ru: {
        locked: "По подписке",
        lockedSuffix: (label) => `${label} — по подписке`,
        title: "Доступно подписчикам",
        body: "Krea 2, Ideogram и ежемесячные наборы рецептов и моделей. "
            + "Всё, что уже установлено, продолжает работать — ключ нужен "
            + "только для получения нового.",
        codeLabel: "Код месяца",
        codePlaceholder: "TSV-…",
        activate: "Активировать",
        cancel: "Закрыть",
        where: "Где взять код:",
        working: "Проверяем…",
        activeUntil: (date, days) => `Активен до ${date} · осталось ${days} дн.`,
        expired: "Ключ истёк — установленное продолжает работать",
        none: "Ключа нет",
        failed: (reason) => reason,
        offlineHint: "Нет интернета? Вставьте вместо кода токен из поста.",
    },
};

const STORE_LABELS = { boosty: "Boosty", patreon: "Patreon", vk: "VK" };

export function ensureGateStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-gate{position:fixed;inset:0;z-index:11400;display:none;align-items:center;
    justify-content:center;background:var(--ts-scrim);backdrop-filter:blur(2px)}
.ts-gate.is-open{display:flex}
.ts-gate__card{width:min(440px,92vw);background:var(--ts-modal-bg);
    border:1px solid var(--ts-border);border-radius:var(--ts-radius-lg);
    box-shadow:var(--ts-shadow);padding:20px;display:flex;flex-direction:column;gap:12px}
.ts-gate__title{font-size:var(--ts-fs-lg);font-weight:700}
.ts-gate__body{font-size:var(--ts-fs);color:var(--ts-muted);line-height:1.5}
.ts-gate__label{font-size:var(--ts-fs-sm);color:var(--ts-muted)}
.ts-gate__row{display:flex;gap:6px}
.ts-gate__row input{flex:1;min-width:0;font-family:var(--ts-font);
    letter-spacing:.04em;text-transform:uppercase}
.ts-gate__links{display:flex;gap:8px;flex-wrap:wrap}
.ts-gate__link{font-size:var(--ts-fs-sm);color:var(--ts-accent);text-decoration:none;
    border-bottom:1px solid var(--ts-accent-line)}
.ts-gate__link:hover{border-bottom-color:var(--ts-accent)}
.ts-gate__note{font-size:var(--ts-fs-xs);color:var(--ts-faint);line-height:1.45}
.ts-gate__status{font-size:var(--ts-fs-sm);min-height:16px}
.ts-gate__status.is-bad{color:var(--ts-danger)}
.ts-gate__status.is-good{color:var(--ts-accent)}
.ts-gate__foot{display:flex;justify-content:flex-end;gap:6px;margin-top:2px}
`;
    document.head.appendChild(style);
}

/**
 * Read the current pass state from the server.
 *
 * Never throws: no server, no network, no pass — all mean "free tier only",
 * which is a perfectly good state to be in.
 */
export async function fetchPassState(api) {
    try {
        const response = await api.fetchApi("/ts_pass/status");
        if (!response.ok) throw new Error(String(response.status));
        return await response.json();
    } catch (err) {
        console.warn("[TS Studio] pass status unavailable", err);
        return { state: "none", tier: 0, links: {} };
    }
}

/**
 * The gate: knows what is open, and asks for a code when it is not.
 *
 * @param {object} options
 * @param {object} options.api ComfyUI api object.
 * @param {() => void} [options.onChange] Called after the pass changes.
 */
export function createGate(options) {
    ensureGateStyles();
    const t = pickLocaleStrings(STRINGS);
    const { api, onChange } = options;
    let state = { state: "none", tier: 0, links: {} };

    const overlay = document.createElement("div");
    overlay.className = `${TS_UI_CLASS} ts-gate`;
    const card = document.createElement("div");
    card.className = "ts-gate__card";
    overlay.appendChild(card);

    const title = document.createElement("div");
    title.className = "ts-gate__title";
    title.textContent = t.title;
    const body = document.createElement("div");
    body.className = "ts-gate__body";
    body.textContent = t.body;

    const label = document.createElement("div");
    label.className = "ts-gate__label";
    label.textContent = t.codeLabel;
    const row = document.createElement("div");
    row.className = "ts-gate__row";
    const field = document.createElement("input");
    field.type = "text";
    field.className = "ts-ui-input";
    field.placeholder = t.codePlaceholder;
    field.autocomplete = "off";
    field.spellcheck = false;
    const activate = document.createElement("button");
    activate.type = "button";
    activate.className = "ts-ui-btn ts-ui-btn--primary";
    activate.textContent = t.activate;
    row.append(field, activate);

    const status = document.createElement("div");
    status.className = "ts-gate__status";
    const where = document.createElement("div");
    where.className = "ts-gate__label";
    where.textContent = t.where;
    const links = document.createElement("div");
    links.className = "ts-gate__links";
    const note = document.createElement("div");
    note.className = "ts-gate__note";
    note.textContent = t.offlineHint;

    const foot = document.createElement("div");
    foot.className = "ts-gate__foot";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "ts-ui-btn";
    close.textContent = t.cancel;
    close.addEventListener("click", () => setOpen(false));
    foot.appendChild(close);

    card.append(title, body, label, row, status, where, links, note, foot);
    document.body.appendChild(overlay);

    overlay.addEventListener("pointerdown", (event) => {
        if (event.target === overlay) setOpen(false);
    });

    function renderLinks() {
        links.textContent = "";
        for (const [key, url] of Object.entries(state.links || {})) {
            if (!url) continue;
            const link = document.createElement("a");
            link.className = "ts-gate__link";
            link.href = url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = STORE_LABELS[key] || key;
            links.appendChild(link);
        }
    }

    function renderStatus() {
        status.classList.remove("is-bad", "is-good");
        if (state.state === "active") {
            const date = new Date(state.expiresAt).toLocaleDateString();
            status.textContent = t.activeUntil(date, state.daysLeft);
            status.classList.add("is-good");
        } else if (state.state === "expired" || state.state === "revoked") {
            status.textContent = t.expired;
        } else {
            status.textContent = "";
        }
    }

    async function refresh() {
        state = await fetchPassState(api);
        renderLinks();
        renderStatus();
        return state;
    }

    activate.addEventListener("click", async () => {
        const code = field.value.trim();
        if (!code) return;
        activate.disabled = true;
        status.classList.remove("is-bad", "is-good");
        status.textContent = t.working;
        try {
            const response = await api.fetchApi("/ts_pass/activate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code }),
            });
            const payload = await response.json();
            if (!response.ok) throw new Error(payload?.error || String(response.status));
            state = payload;
            field.value = "";
            renderStatus();
            onChange?.(state);
            if (state.state === "active") setTimeout(() => setOpen(false), 900);
        } catch (err) {
            status.textContent = t.failed(String(err.message || err));
            status.classList.add("is-bad");
        } finally {
            activate.disabled = false;
        }
    });

    field.addEventListener("keydown", (event) => {
        if (event.key === "Enter") activate.click();
        event.stopPropagation();          // Escape/Enter belong to this field
    });

    function setOpen(open) {
        overlay.classList.toggle("is-open", open);
        if (open) {
            renderLinks();
            renderStatus();
            setTimeout(() => field.focus(), 30);
        }
    }

    return {
        element: overlay,
        refresh,
        /** Everything free is open; paid needs an active pass of that tier. */
        opens: (tier) => !tier || (state.state === "active" && (state.tier || 0) >= tier),
        state: () => state,
        prompt: () => setOpen(true),
        close: () => setOpen(false),
        teardown: () => overlay.remove(),
        strings: t,
    };
}
