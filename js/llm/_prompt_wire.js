// Вход `prompt` у TS Super Prompt и TS Super Prompt RT — общая часть интерфейса.
//
// Когда к гнезду `prompt` подключён провод, запуск берёт текст с него, а не из
// поля ноды (правило — `nodes/llm/_prompt_wire.py`). Без подсказки это
// выглядело бы как «поле не работает», поэтому над полем появляется строка об
// этом, а под ней — результат последнего запуска с кнопкой «Копировать».
//
// ⚠️ Результат показывается ЗДЕСЬ, а не записывается в поле: изменённый
// виджет сбил бы кэш ComfyUI, и следующий запуск пересчитал бы весь граф ниже,
// включая сэмплер.

import { ensureThemeStyles, pickLocaleStrings } from "../_theme.js";

/** Ключ ui-выхода ноды (`RESULT_UI_KEY` в `_prompt_wire.py`). */
export const RESULT_UI_KEY = "ts_super_prompt_result";
export const PROMPT_INPUT = "prompt";

const STYLE_ID = "ts-prompt-wire-styles";

const STRINGS = {
    en: {
        noticeAsIs: "The prompt comes from the «prompt» input and goes out as it is. "
            + "The field below is used only when that input is empty.",
        noticeEnhanced: "The prompt comes from the «prompt» input and is enhanced when the "
            + "workflow runs. The field below is used only when that input is empty.",
        toggle: "Enhance the incoming prompt on run",
        toggleHint: "Off — the prompt from the input goes out unchanged. On — the node enhances "
            + "it with the chosen preset when the workflow runs.",
        result: "Last run's result",
        copy: "Copy",
        copied: "Copied",
    },
    ru: {
        noticeAsIs: "Промпт приходит со входа «prompt» и передаётся дальше как есть. "
            + "Поле ниже используется, только если этот вход пуст.",
        noticeEnhanced: "Промпт приходит со входа «prompt» и улучшается при запуске. "
            + "Поле ниже используется, только если этот вход пуст.",
        toggle: "Улучшать входящий промпт при запуске",
        toggleHint: "Выключено — промпт со входа уходит дальше без изменений. Включено — нода "
            + "улучшает его выбранным пресетом при запуске.",
        result: "Результат последнего запуска",
        copy: "Копировать",
        copied: "Скопировано",
    },
};

/** Виджет-переключатель улучшения (вход `enhance_prompt` обеих нод). */
export const ENHANCE_WIDGET = "enhance_prompt";

/** Подключён ли провод к гнезду `prompt`. */
export function isPromptWired(node) {
    return (node?.inputs || []).some(
        (input) => input?.name === PROMPT_INPUT && input.link !== null && input.link !== undefined);
}

/** Текст результата из события `executed`, либо пустая строка. */
export function resultFromOutput(output) {
    const value = output?.[RESULT_UI_KEY];
    if (Array.isArray(value)) return typeof value[0] === "string" ? value[0] : "";
    return typeof value === "string" ? value : "";
}

function ensureStyles(doc) {
    ensureThemeStyles();
    if (doc.getElementById(STYLE_ID)) return;
    const style = doc.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-prompt-wire{flex:0 0 auto;display:flex;flex-direction:column;gap:4px;
  padding:5px 8px;border-radius:var(--ts-radius-sm);
  border:1px solid var(--ts-accent-line);background:var(--ts-accent-soft);
  font-size:var(--ts-fs-xs);color:var(--ts-text);line-height:1.4}
.ts-prompt-wire[hidden]{display:none}
.ts-prompt-wire__result{display:flex;flex-direction:column;gap:3px}
.ts-prompt-wire__result[hidden]{display:none}
.ts-prompt-wire__head{display:flex;align-items:center;justify-content:space-between;gap:6px;
  color:var(--ts-muted)}
.ts-prompt-wire__text{max-height:96px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;
  user-select:text;background:var(--ts-sunken);border-radius:var(--ts-radius-sm);padding:4px 6px}
.ts-prompt-wire__toggle{display:flex;align-items:center;gap:6px;cursor:pointer;
  color:var(--ts-text);font-weight:600;user-select:none}
.ts-prompt-wire__toggle input{margin:0;accent-color:var(--ts-accent);cursor:pointer}
`;
    doc.head.appendChild(style);
}

async function copyText(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch {
        return false;
    }
}

/**
 * Встроить подсказку и результат в интерфейс ноды.
 *
 * Обработчики `onConnectionsChange` и `onExecuted` ставятся на ноду ОДИН раз и
 * ищут текущую панель через `node.__tsPromptWire`: интерфейс ноды пересобирается
 * (смена вкладки, вставка), и повторная обёртка складывала бы обработчики в
 * стопку.
 *
 * @param {object} node          нода LiteGraph.
 * @param {HTMLElement} container корень интерфейса ноды.
 * @param {HTMLElement} before   элемент, перед которым встать (поле ввода).
 * @param {Document} [doc]       документ интерфейса.
 * @param {object} [options]
 * @param {() => boolean} [options.getEnhance] прочитать переключатель улучшения.
 * @param {(on: boolean) => void} [options.setEnhance] записать его. Функции даёт
 *   нода: помощники виджетов тянут `app`, и модуль перестал бы проверяться Node.
 * @returns {{refresh: () => void, dispose: () => void}}
 */
export function attachPromptWire(node, container, before, doc = document, options = {}) {
    ensureStyles(doc);
    const t = pickLocaleStrings(STRINGS);
    const getEnhance = options.getEnhance || (() => false);
    const setEnhance = options.setEnhance || (() => {});

    const root = doc.createElement("div");
    root.className = "ts-prompt-wire";
    root.hidden = true;
    const notice = doc.createElement("div");

    // Переключатель живёт здесь, а не отдельным виджетом: он имеет смысл только
    // при подключённом проводе, и тогда же его видно.
    const toggle = doc.createElement("label");
    toggle.className = "ts-prompt-wire__toggle";
    toggle.title = t.toggleHint;
    const box = doc.createElement("input");
    box.type = "checkbox";
    const toggleText = doc.createElement("span");
    toggleText.textContent = t.toggle;
    toggle.append(box, toggleText);
    box.addEventListener("change", () => {
        setEnhance(box.checked);
        paintNotice();
    });

    function paintNotice() {
        notice.textContent = box.checked ? t.noticeEnhanced : t.noticeAsIs;
    }

    const result = doc.createElement("div");
    result.className = "ts-prompt-wire__result";
    result.hidden = true;
    const head = doc.createElement("div");
    head.className = "ts-prompt-wire__head";
    const label = doc.createElement("span");
    label.textContent = t.result;
    const copy = doc.createElement("button");
    copy.type = "button";
    copy.className = "ts-ui-btn ts-ui-btn--ghost";
    copy.textContent = t.copy;
    head.append(label, copy);
    const text = doc.createElement("div");
    text.className = "ts-prompt-wire__text";
    result.append(head, text);
    root.append(notice, toggle, result);
    container.insertBefore(root, before);

    let copyTimer = 0;
    copy.addEventListener("click", async () => {
        const ok = await copyText(text.textContent || "");
        clearTimeout(copyTimer);
        copy.textContent = ok ? t.copied : t.copy;
        copyTimer = setTimeout(() => { copy.textContent = t.copy; }, 1500);
    });

    const handle = {
        refresh() {
            root.hidden = !isPromptWired(node);
            // Значение приходит из workflow позже создания панели (§12.5.12).
            box.checked = Boolean(getEnhance());
            paintNotice();
        },
        show(value) {
            text.textContent = value;
            result.hidden = !value;
        },
        dispose() {
            clearTimeout(copyTimer);
            root.remove();
            if (node.__tsPromptWire === handle) delete node.__tsPromptWire;
        },
    };
    node.__tsPromptWire = handle;

    if (!node.__tsPromptWireHooked) {
        node.__tsPromptWireHooked = true;
        const previousConnections = node.onConnectionsChange;
        node.onConnectionsChange = function tsPromptWireConnections(...args) {
            const answer = previousConnections?.apply(this, args);
            node.__tsPromptWire?.refresh();
            return answer;
        };
        const previousExecuted = node.onExecuted;
        node.onExecuted = function tsPromptWireExecuted(output, ...rest) {
            const answer = previousExecuted?.call(this, output, ...rest);
            const value = resultFromOutput(output);
            if (value) node.__tsPromptWire?.show(value);
            return answer;
        };
    }

    handle.refresh();
    return { refresh: handle.refresh, dispose: handle.dispose };
}
