// Менеджер паков — что у студии есть, чего нет и что из этого показывать.
//
// Пак здесь — одна модельная линия: Z-Image, Flux 2 Klein, Krea 2. Человек
// выбирает модель, а не «набор» непонятного состава, поэтому карточка отвечает
// ровно на те вопросы, которые про модель и задают:
//
//   что это         обложка и две строки текста;
//   какого уровня   Free / Pro / Ultimate — значком, а не мелким шрифтом;
//   есть ли здесь   в сборке, установлен, или нужен доступ;
//   заработает ли   все ли файлы моделей на месте — до первого запуска, а не
//                   после десяти минут ожидания и красной ошибки;
//   нужен ли он     выключатель. Выключенный пак исчезает из студии и
//                   остаётся на диске: шесть моделей в списке, когда работают
//                   тремя, — это шум, но удалять из-за шума нечего.
//
// Каталог читается публично и без пропуска: увидеть, что существует, человек
// должен до того, как решит платить. Покупка живёт в окне доступа, установка —
// в одной кнопке на карточке.

import { TS_UI_CLASS, createPanelBackButton, ensureThemeStyles,
    pickLocaleStrings } from "../_theme.js";

const STYLE_ID = "ts-studio-showcase-styles";

// Лесенка уровней, та же, что в nodes/_pass.py и в каталоге сборки.
const TIER_ORDER = [0, 2, 3];

const STRINGS = {
    en: {
        open: "Packs",
        title: "Packs",
        close: "Close",
        free: "Free",
        pro: "Pro",
        ultimate: "Ultimate",
        tierFree: "Free — ships with the studio",
        tierPro: "Pro — the launcher subscription",
        tierUltimate: "Ultimate — the top tier",
        included: "In the pack",
        installed: "Installed",
        inBuild: "In this build",
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
        modelsReady: "All models are here",
        modelsMissing: (count) => `${count} model file(s) missing`,
        modelsUnknown: "Not here yet — nothing to check",
        needsKey: "Needs an API key, not a model file",
        modesLine: (list) => `Modes: ${list}`,
        shown: "Shown in the studio",
        hidden: "Hidden from the studio",
        hiddenByTier: "Above the tier being previewed",
        show: "Show",
        hide: "Hide",
        viewing: (tier) => `Previewing the studio as: ${tier}`,
        viewingAuthor: "Previewing: everything (author)",
        modes: {
            t2i: "Generate",
            edit: "Edit",
            inpaint: "Inpaint",
            upscale: "Upscale",
        },
    },
    ru: {
        open: "Наборы",
        title: "Наборы",
        close: "Закрыть",
        free: "Free",
        pro: "Pro",
        ultimate: "Ultimate",
        tierFree: "Free — идёт вместе со студией",
        tierPro: "Pro — подписка уровня лаунчера",
        tierUltimate: "Ultimate — старший уровень",
        included: "В наборе",
        installed: "Установлен",
        inBuild: "В сборке",
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
        modelsReady: "Все модели на месте",
        modelsMissing: (count) => `Не хватает файлов моделей: ${count}`,
        modelsUnknown: "Ещё не здесь — проверять нечего",
        needsKey: "Нужен ключ API, а не файл модели",
        modesLine: (list) => `Режимы: ${list}`,
        shown: "Показывается в студии",
        hidden: "Скрыт из студии",
        hiddenByTier: "Выше уровня, которым сейчас смотрим",
        show: "Показать",
        hide: "Скрыть",
        viewing: (tier) => `Студия показана как для уровня: ${tier}`,
        viewingAuthor: "Показано всё — как видит автор",
        modes: {
            t2i: "Генерация",
            edit: "Редактирование",
            inpaint: "Inpaint",
            upscale: "Upscale",
        },
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
.ts-showcase__tier{display:flex;flex-direction:column;gap:10px}
.ts-showcase__tierhead{display:flex;align-items:baseline;gap:8px}
.ts-showcase__tiername{font-weight:700;letter-spacing:.02em}
.ts-showcase__tierabout{font-size:var(--ts-fs-sm);color:var(--ts-muted)}
.ts-showcase__grid{display:grid;gap:14px;
    grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}
.ts-showcase__card{display:flex;flex-direction:column;border:1px solid var(--ts-border);
    border-radius:var(--ts-radius-lg);overflow:hidden;background:var(--ts-elevated)}
/* Выключенный пак не исчезает и не кричит: приглушён ровно настолько, чтобы
   было видно — он здесь, просто сейчас не в работе. */
.ts-showcase__card.is-off{opacity:.55}
.ts-showcase__card.is-off:hover{opacity:1}
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
/* Уровень — справа сверху, отдельно от состояния: это разные вопросы. */
.ts-showcase__tierbadge{position:absolute;top:8px;right:8px;padding:3px 7px;
    border-radius:999px;font-size:var(--ts-fs-xs);font-weight:700;
    letter-spacing:.04em;text-transform:uppercase;
    background:var(--ts-scrim-strong);color:var(--ts-on-media)}
.ts-showcase__text{padding:10px 12px;display:flex;flex-direction:column;gap:6px;flex:1}
.ts-showcase__name{font-weight:600}
.ts-showcase__about{font-size:var(--ts-fs-sm);color:var(--ts-muted);line-height:1.45}
.ts-showcase__facts{display:flex;flex-direction:column;gap:3px;
    font-size:var(--ts-fs-sm);color:var(--ts-muted)}
.ts-showcase__fact{display:flex;align-items:center;gap:6px}
.ts-showcase__dot{width:7px;height:7px;border-radius:50%;flex:0 0 auto;
    background:var(--ts-muted)}
.ts-showcase__dot.is-ok{background:var(--ts-success)}
.ts-showcase__dot.is-bad{background:var(--ts-danger)}
.ts-showcase__foot{display:flex;gap:6px;padding:0 12px 12px}
.ts-showcase__foot .ts-ui-btn{flex:1}
.ts-showcase__switch{display:flex;align-items:center;gap:6px;padding:0 12px 12px;
    font-size:var(--ts-fs-sm);color:var(--ts-muted);cursor:pointer}
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

/** Имена семейств пака — каталог сборки пишет строками, удалённый описаниями. */
export function familyNames(pack) {
    return (pack?.families || [])
        .map((item) => (typeof item === "string" ? item : item?.family))
        .filter(Boolean);
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
 * @param {(pack: object) => object|null} [options.readiness] Готовность пака по
 *        живым бэкендам: {ready, total, missing[]} или null, если его тут нет.
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

    function tierName(tier) {
        if (tier >= 3) return t.ultimate;
        if (tier >= 2) return t.pro;
        return t.free;
    }

    function tierAbout(tier) {
        if (tier >= 3) return t.tierUltimate;
        if (tier >= 2) return t.tierPro;
        return t.tierFree;
    }

    function fact(text, kind) {
        const row = document.createElement("div");
        row.className = "ts-showcase__fact";
        const dot = document.createElement("span");
        dot.className = `ts-showcase__dot${kind ? ` is-${kind}` : ""}`;
        const label = document.createElement("span");
        label.textContent = text;
        row.append(dot, label);
        return row;
    }

    /** Строка про модели: главный вопрос «заработает ли оно вообще». */
    function readinessRow(entry) {
        if (entry.api) return fact(t.needsKey, "");
        const state = options.readiness?.(entry) || null;
        if (!state) return fact(t.modelsUnknown, "");
        if (!state.missing?.length) return fact(t.modelsReady, "ok");
        const row = fact(t.modelsMissing(state.missing.length), "bad");
        // Подробности — в подсказке: на карточке важен вердикт, а список
        // недостающих файлов нужен, только когда идёшь их искать.
        row.title = state.missing.join("\n");
        return row;
    }

    function card(entry) {
        const element = document.createElement("div");
        element.className = "ts-showcase__card";
        if (entry.hidden) element.classList.add("is-off");

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
        } else if (entry.cover || entry.preview) {
            const image = document.createElement("img");
            image.src = entry.cover || entry.preview;
            image.alt = "";
            image.loading = "lazy";
            media.appendChild(image);
        }
        const badge = document.createElement("span");
        badge.className = "ts-showcase__badge";
        if (entry.installed) {
            badge.dataset.kind = "installed";
            badge.textContent = t.installed;
        } else if (entry.builtin) {
            badge.dataset.kind = "free";
            badge.textContent = t.inBuild;
        } else if (entry.open) {
            badge.dataset.kind = "included";
            badge.textContent = t.included;
        } else {
            badge.dataset.kind = "locked";
            badge.textContent = t.locked;
        }
        const tierBadge = document.createElement("span");
        tierBadge.className = "ts-showcase__tierbadge";
        tierBadge.textContent = tierName(Number(entry.tier || 0));
        media.append(badge, tierBadge);

        const text = document.createElement("div");
        text.className = "ts-showcase__text";
        const name = document.createElement("div");
        name.className = "ts-showcase__name";
        name.textContent = localized(entry.name, locale) || entry.id;
        const about = document.createElement("div");
        about.className = "ts-showcase__about";
        about.textContent = localized(entry.about, locale);
        const facts = document.createElement("div");
        facts.className = "ts-showcase__facts";
        const modes = (entry.modes || []).map((mode) => t.modes[mode] || mode);
        if (modes.length) facts.appendChild(fact(t.modesLine(modes.join(" · ")), ""));
        facts.appendChild(readinessRow(entry));
        if (entry.hidden === "tier") facts.appendChild(fact(t.hiddenByTier, ""));
        text.append(name, about, facts);

        // What the button offers follows what is actually possible here: an
        // installed pack keeps working without a pass, so it is never asked to
        // buy anything — only a fetch it cannot perform sends someone to the
        // subscription.
        const foot = document.createElement("div");
        foot.className = "ts-showcase__foot";
        const deliverable = Boolean(entry.file || entry.version);
        const wants = entry.installed ? entry.updateAvailable : !entry.builtin;
        const action = document.createElement("button");
        action.type = "button";
        action.className = "ts-ui-btn ts-ui-btn--primary";
        if (!wants || !deliverable) {
            action.textContent = entry.installed ? t.installed : t.inBuild;
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
        // Выключатель есть только у того, что действительно здесь: предлагать
        // спрятать то, чего на машине нет, — обещание без содержания.
        if (entry.present) element.appendChild(switchRow(entry));
        return element;
    }

    /** «Показывать в студии» — не удаление, а именно видимость. */
    function switchRow(entry) {
        const row = document.createElement("label");
        row.className = "ts-showcase__switch";
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = entry.hidden !== "off";
        // Потолок уровня — не выбор человека, и переключателем его не снять.
        box.disabled = entry.hidden === "tier";
        const label = document.createElement("span");
        label.textContent = box.checked ? t.shown : t.hidden;
        box.addEventListener("change", () => setEnabled(entry, box.checked));
        row.append(box, label);
        return row;
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

    async function setEnabled(entry, enabled) {
        status.classList.remove("is-bad");
        try {
            const result = await call("/ts_studio/packs/enable",
                { id: entry.id, enabled });
            data = result.packs;
            render();
            options.onCatalog?.(data);
            // Список моделей перечитывается сразу: выключатель, который
            // действует «после перезапуска», — это не выключатель.
            onInstalled?.();
        } catch (err) {
            status.textContent = t.failed(String(err.message || err));
            status.classList.add("is-bad");
        }
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

        if (data.viewTier !== null && data.viewTier !== undefined) {
            // Видно всем, кто в режиме тестирования: иначе легко полчаса
            // искать «пропавшую» модель, которую сам же и убрал потолком.
            const note = document.createElement("div");
            note.className = "ts-showcase__note";
            note.textContent = t.viewing(tierName(Number(data.viewTier)));
            body.appendChild(note);
        }
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

        // Раскладка по уровням, а не одной кучей: уровень — первое, что
        // человек хочет знать, и первое, что решает, доступно ли это ему.
        const packs = data.packs || [];
        const tiers = [...new Set([...TIER_ORDER,
            ...packs.map((p) => Number(p.tier || 0))])].sort((a, b) => a - b);
        for (const tier of tiers) {
            const mine = packs.filter((p) => Number(p.tier || 0) === tier);
            if (!mine.length) continue;
            const section = document.createElement("div");
            section.className = "ts-showcase__tier";
            const heading = document.createElement("div");
            heading.className = "ts-showcase__tierhead";
            const name = document.createElement("span");
            name.className = "ts-showcase__tiername";
            name.textContent = tierName(tier);
            const about = document.createElement("span");
            about.className = "ts-showcase__tierabout";
            about.textContent = tierAbout(tier);
            heading.append(name, about);
            const grid = document.createElement("div");
            grid.className = "ts-showcase__grid";
            for (const entry of mine) grid.appendChild(card(entry));
            section.append(heading, grid);
            body.appendChild(section);
        }
        if (!packs.length) {
            const empty = document.createElement("div");
            empty.className = "ts-showcase__note";
            empty.textContent = t.empty;
            body.appendChild(empty);
        }
        body.appendChild(status);
    }

    async function refresh() {
        // Два чтения, и порядок здесь — не оптимизация, а поведение. Первое
        // отвечает по тому, что лежит на этой машине, и рисует все карточки
        // сразу. Второе идёт в сеть за доставкой (версии, архивы) — два адреса,
        // у каждого свой таймаут, и до его ответа экран был бы пустым.
        try {
            const local = await api.fetchApi("/ts_studio/packs/state");
            if (local.ok) {
                data = await local.json();
                render();
                options.onCatalog?.(data);
            }
        } catch (err) {
            console.warn("[TS Studio] local pack state unavailable", err);
        }
        try {
            const response = await api.fetchApi("/ts_studio/packs");
            data = await response.json();
        } catch (err) {
            console.warn("[TS Studio] the catalogue is unavailable", err);
            // Каталог сборки уже показан — не затираем его пустотой.
            if (!data) data = { packs: [], offline: true };
            else data = { ...data, offline: true };
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
