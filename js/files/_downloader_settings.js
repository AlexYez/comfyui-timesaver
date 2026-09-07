// Настройки TS Files Downloader — своей панелью, а не показом штатных виджетов.
//
// ⚠️ Показывать под кнопкой обратно родные виджеты не годится по двум причинам,
// и обе замерены. Первая: в Nodes 2.0 каждый элемент `node.widgets` занимает
// строку сетки ноды, и спрятанный виджет строку НЕ освобождает — сверху ноды
// оставалась пустота в треть её высоты. Вторая: спрятанные виджеты в Vue-режиме
// не показывались обратно вовсе, и кнопка «Настройки» просто ничего не делала.
//
// Поэтому виджеты убираются из ноды общим `hideWidget` (значения при этом живут
// в `node.properties` и доезжают до промпта — см. js/_dom_widget.js), а человек
// правит их ЗДЕСЬ. Имена, типы и границы читаются у самих виджетов, так что
// схема остаётся единственным источником правды: добавится вход — он появится
// в панели, а не разойдётся с ней.

import { TS_UI_CLASS, ensureThemeStyles, pickLocaleStrings } from "../_theme.js";
import { getWidget } from "../_dom_widget.js";

/** Что показываем и в каком порядке. Имена — контракт ноды, не менять. */
const GROUPS = [
    {
        key: "download",
        fields: [
            { name: "enable", kind: "toggle" },
            { name: "skip_existing", kind: "toggle" },
            { name: "unzip_after_download", kind: "toggle" },
            { name: "chunk_size_kb", kind: "number" },
        ],
    },
    {
        key: "integrity",
        fields: [
            { name: "verify_size", kind: "toggle" },
            { name: "integrity_mode", kind: "combo" },
        ],
    },
    {
        key: "access",
        fields: [
            { name: "hf_domain", kind: "text" },
            { name: "proxy_url", kind: "text" },
            { name: "hf_token", kind: "secret" },
            { name: "modelscope_token", kind: "secret" },
        ],
    },
];

const STRINGS = {
    en: {
        title: "Settings",
        done: "Done",
        groups: {
            download: "Downloading",
            integrity: "Integrity",
            access: "Sources and tokens",
        },
        labels: {
            enable: "Run with the workflow",
            skip_existing: "Skip what is already on disk",
            unzip_after_download: "Unpack archives",
            chunk_size_kb: "Chunk size, KB",
            verify_size: "Check the size",
            integrity_mode: "Integrity check",
            hf_domain: "Hugging Face mirrors",
            proxy_url: "Proxy",
            hf_token: "Hugging Face token",
            modelscope_token: "ModelScope token",
        },
        hints: {
            enable: "Off, the node lets the graph pass without downloading anything.",
            skip_existing: "A file already in place is not fetched again.",
            chunk_size_kb: "Bigger chunks are faster on a good link, worse on a flaky one.",
            integrity_mode: "hf_sha256_auto also compares the checksum where the host gives one.",
            hf_domain: "Tried in order, comma separated.",
            hf_token: "Needed for gated repositories.",
        },
        secretShow: "Show",
        secretHide: "Hide",
    },
    ru: {
        title: "Настройки",
        done: "Готово",
        groups: {
            download: "Загрузка",
            integrity: "Целостность",
            access: "Источники и токены",
        },
        labels: {
            enable: "Работать при прогоне воркфлоу",
            skip_existing: "Пропускать то, что уже на диске",
            unzip_after_download: "Распаковывать архивы",
            chunk_size_kb: "Размер куска, КБ",
            verify_size: "Сверять размер",
            integrity_mode: "Проверка целостности",
            hf_domain: "Зеркала Hugging Face",
            proxy_url: "Прокси",
            hf_token: "Токен Hugging Face",
            modelscope_token: "Токен ModelScope",
        },
        hints: {
            enable: "Выключено — нода пропускает прогон, ничего не скачивая.",
            skip_existing: "Файл, лежащий на месте, второй раз не качается.",
            chunk_size_kb: "Большой кусок быстрее на хорошем канале и хуже на рваном.",
            integrity_mode: "hf_sha256_auto дополнительно сверяет контрольную сумму, если сервер её отдаёт.",
            hf_domain: "Перебираются по порядку, через запятую.",
            hf_token: "Нужен для закрытых репозиториев.",
        },
        secretShow: "Показать",
        secretHide: "Скрыть",
    },
};

export function ensureSettingsStyles() {
    // ⚠️ Тема — первой строкой, до раннего возврата (§12.6).
    ensureThemeStyles();
    if (document.getElementById("ts-fdl-settings-styles")) return;
    const style = document.createElement("style");
    style.id = "ts-fdl-settings-styles";
    style.textContent = `
.ts-fdl-settings-panel{
  position:absolute;inset:0;z-index:3;display:none;flex-direction:column;gap:6px;
  padding:8px;border-radius:var(--ts-radius-lg);
  background:var(--ts-elevated);border:1px solid var(--ts-border)}
.ts-fdl-settings-panel.is-open{display:flex}
.ts-fdl-settings-panel__head{display:flex;align-items:center;gap:8px;flex:0 0 auto}
.ts-fdl-settings-panel__title{
  font-size:var(--ts-fs);font-weight:600;color:var(--ts-text);flex:1 1 auto}
.ts-fdl-settings-panel__body{
  flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;
  display:flex;flex-direction:column;gap:10px;padding-right:2px}

.ts-fdl-group{display:flex;flex-direction:column;gap:4px}
.ts-fdl-group__title{
  font-size:var(--ts-fs-xs);text-transform:uppercase;letter-spacing:.06em;
  color:var(--ts-faint)}

/* Подпись слева, поле справа — глаз идёт по одной колонке значений. */
.ts-fdl-field{
  display:grid;grid-template-columns:1fr auto;align-items:center;gap:8px;
  padding:4px 6px;border-radius:var(--ts-radius-sm)}
.ts-fdl-field:hover{background:var(--ts-surface)}
.ts-fdl-field__label{min-width:0;display:flex;flex-direction:column;gap:1px}
.ts-fdl-field__name{
  font-size:var(--ts-fs-sm);color:var(--ts-text);
  overflow:hidden;text-overflow:ellipsis}
.ts-fdl-field__hint{
  font-size:var(--ts-fs-xs);color:var(--ts-faint);line-height:1.35}
.ts-fdl-field__control{display:flex;align-items:center;gap:4px;flex:0 0 auto}
.ts-fdl-field__control input[type="text"],
.ts-fdl-field__control input[type="password"],
.ts-fdl-field__control input[type="number"],
.ts-fdl-field__control select{width:180px;max-width:42vw}
.ts-fdl-field--wide{grid-template-columns:1fr}
.ts-fdl-field--wide .ts-fdl-field__control{width:100%}
.ts-fdl-field--wide .ts-fdl-field__control input,
.ts-fdl-field--wide .ts-fdl-field__control select{width:100%;max-width:none}

/* Переключатель: та же форма, что у точек статуса, — одна система на ноду. */
.ts-fdl-switch{
  width:34px;height:18px;border-radius:999px;border:1px solid var(--ts-border);
  background:var(--ts-sunken);position:relative;cursor:pointer;padding:0;
  transition:background .15s ease,border-color .15s ease}
.ts-fdl-switch::after{
  content:"";position:absolute;top:2px;left:2px;width:12px;height:12px;
  border-radius:50%;background:var(--ts-muted);transition:transform .15s ease,background .15s ease}
.ts-fdl-switch[aria-checked="true"]{background:var(--ts-accent-soft);border-color:var(--ts-accent-line)}
.ts-fdl-switch[aria-checked="true"]::after{transform:translateX(16px);background:var(--ts-accent)}
.ts-fdl-switch:focus-visible{outline:2px solid var(--ts-accent);outline-offset:2px}
@media (prefers-reduced-motion:reduce){
  .ts-fdl-switch,.ts-fdl-switch::after{transition:none}
}
`;
    document.head.appendChild(style);
}

function readWidget(node, name) {
    const widget = getWidget(node, name);
    return widget ? widget.value : undefined;
}

function writeWidget(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return;
    widget.value = value;
    // ⚠️ Колбэк зовём сами: у виджета, вынутого из `node.widgets`, его больше
    // никто не позовёт, а на нём висит и запись в `node.properties`.
    if (typeof widget.callback === "function") {
        try {
            widget.callback(value, null, node);
        } catch (err) {
            console.warn("[TS FilesDownloader] settings callback failed", err);
        }
    }
    node.properties = node.properties || {};
    node.properties[name] = value;
}

function makeToggle(node, name, current) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ts-fdl-switch";
    button.setAttribute("role", "switch");
    const apply = (value) => button.setAttribute("aria-checked", value ? "true" : "false");
    apply(Boolean(current));
    button.addEventListener("click", () => {
        const next = button.getAttribute("aria-checked") !== "true";
        apply(next);
        writeWidget(node, name, next);
    });
    return { element: button, sync: (value) => apply(Boolean(value)) };
}

function makeNumber(node, name, current) {
    const widget = getWidget(node, name);
    const input = document.createElement("input");
    input.type = "number";
    input.className = "ts-ui-input";
    const options = widget?.options || {};
    if (Number.isFinite(Number(options.min))) input.min = String(options.min);
    if (Number.isFinite(Number(options.max))) input.max = String(options.max);
    if (Number.isFinite(Number(options.step))) input.step = String(Math.max(1, options.step / 10));
    input.value = String(current ?? "");
    input.addEventListener("change", () => {
        const value = Number(input.value);
        if (!Number.isFinite(value)) return;
        const min = Number.isFinite(Number(options.min)) ? Number(options.min) : value;
        const max = Number.isFinite(Number(options.max)) ? Number(options.max) : value;
        const clamped = Math.min(Math.max(value, min), max);
        input.value = String(clamped);
        writeWidget(node, name, clamped);
    });
    return { element: input, sync: (value) => { input.value = String(value ?? ""); } };
}

function makeText(node, name, current, secret, L) {
    const wrap = document.createElement("div");
    wrap.style.display = "flex";
    wrap.style.gap = "4px";
    wrap.style.width = "100%";

    const input = document.createElement("input");
    input.type = secret ? "password" : "text";
    input.className = "ts-ui-input";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.value = String(current ?? "");
    input.addEventListener("change", () => writeWidget(node, name, input.value));
    wrap.appendChild(input);

    if (secret) {
        const eye = document.createElement("button");
        eye.type = "button";
        eye.className = "ts-ui-btn ts-ui-btn--icon";
        eye.textContent = "👁";
        eye.title = L.secretShow;
        eye.addEventListener("click", () => {
            const hidden = input.type === "password";
            input.type = hidden ? "text" : "password";
            eye.title = hidden ? L.secretHide : L.secretShow;
        });
        wrap.appendChild(eye);
    }

    return { element: wrap, sync: (value) => { input.value = String(value ?? ""); } };
}

function makeCombo(node, name, current) {
    const widget = getWidget(node, name);
    const select = document.createElement("select");
    select.className = "ts-ui-select";
    const values = widget?.options?.values;
    const list = Array.isArray(values) && values.length ? values : [current];
    for (const value of list) {
        const option = document.createElement("option");
        option.value = String(value);
        option.textContent = String(value);
        select.appendChild(option);
    }
    select.value = String(current ?? list[0] ?? "");
    select.addEventListener("change", () => writeWidget(node, name, select.value));
    return { element: select, sync: (value) => { select.value = String(value ?? ""); } };
}

/**
 * Панель настроек поверх списка.
 *
 * @param {object} node нода LiteGraph.
 * @returns {{element:HTMLElement, open:Function, close:Function, toggle:Function,
 *            isOpen:Function, sync:Function}}
 */
export function createSettingsPanel(node) {
    ensureSettingsStyles();
    const L = pickLocaleStrings(STRINGS);

    const panel = document.createElement("div");
    panel.className = `${TS_UI_CLASS} ts-fdl-settings-panel`;

    const head = document.createElement("div");
    head.className = "ts-fdl-settings-panel__head";
    const title = document.createElement("div");
    title.className = "ts-fdl-settings-panel__title";
    title.textContent = L.title;
    const done = document.createElement("button");
    done.type = "button";
    done.className = "ts-ui-btn ts-ui-btn--primary";
    done.textContent = L.done;
    head.append(title, done);

    const body = document.createElement("div");
    body.className = "ts-fdl-settings-panel__body";

    const syncers = [];

    for (const group of GROUPS) {
        const fields = group.fields.filter((field) => getWidget(node, field.name));
        if (!fields.length) continue;

        const block = document.createElement("div");
        block.className = "ts-fdl-group";
        const caption = document.createElement("div");
        caption.className = "ts-fdl-group__title";
        caption.textContent = L.groups[group.key] || group.key;
        block.appendChild(caption);

        for (const field of fields) {
            const row = document.createElement("div");
            row.className = "ts-fdl-field";
            if (field.kind === "text" || field.kind === "secret") {
                // Адреса и токены длинные — им нужна вся ширина строки.
                row.classList.add("ts-fdl-field--wide");
            }

            const label = document.createElement("div");
            label.className = "ts-fdl-field__label";
            const name = document.createElement("div");
            name.className = "ts-fdl-field__name";
            name.textContent = L.labels[field.name] || field.name;
            label.appendChild(name);
            const hint = L.hints[field.name];
            if (hint) {
                const note = document.createElement("div");
                note.className = "ts-fdl-field__hint";
                note.textContent = hint;
                label.appendChild(note);
            }
            row.appendChild(label);

            const control = document.createElement("div");
            control.className = "ts-fdl-field__control";
            const current = readWidget(node, field.name);
            let built;
            if (field.kind === "toggle") built = makeToggle(node, field.name, current);
            else if (field.kind === "number") built = makeNumber(node, field.name, current);
            else if (field.kind === "combo") built = makeCombo(node, field.name, current);
            else built = makeText(node, field.name, current, field.kind === "secret", L);
            control.appendChild(built.element);
            row.appendChild(control);
            block.appendChild(row);

            syncers.push(() => built.sync(readWidget(node, field.name)));
        }

        body.appendChild(block);
    }

    panel.append(head, body);

    const sync = () => {
        for (const run of syncers) {
            try {
                run();
            } catch (err) {
                console.warn("[TS FilesDownloader] settings sync failed", err);
            }
        }
    };

    const api = {
        element: panel,
        sync,
        isOpen: () => panel.classList.contains("is-open"),
        open() {
            // Значение могло приехать из воркфлоу уже после сборки панели.
            sync();
            panel.classList.add("is-open");
        },
        close() {
            panel.classList.remove("is-open");
        },
        toggle() {
            if (api.isOpen()) api.close();
            else api.open();
        },
    };

    done.addEventListener("click", () => api.close());
    return api;
}
