// TS Studio kit — панель итераций: ↶ ↷ и «Сохранить».
//
// Поверхность для `_history.js`. Разделение простое: история знает, какие
// версии есть и что из них ещё не сохранено, а панель — как это показать и
// какие кнопки погасить. Ни она о ComfyUI, ни история о DOM ничего не знают,
// поэтому обе проверяются порознь.
//
// ПОЧЕМУ ПАНЕЛЬ ОДНА НА ВСЕ РАЗДЕЛЫ. Апскейл, перерисовка и инпэйнт итерируют
// одинаково: прогнал — посмотрел — вернулся — прогнал иначе. Если каждый
// заведёт свои кнопки, они разойдутся в мелочах (где-то Save гаснет после
// сохранения, где-то нет), и человек перестанет им верить.
//
// ЧЕГО ЗДЕСЬ НЕТ. Панель не показывает картинку и не сохраняет файл — она
// зовёт `onShow` и `onKeep`. Показ версии в разных разделах разный (сцена,
// холст инпэйнта), а сохранение — это сеть.

/**
 * Панель итераций.
 *
 * @param {object} options
 * @param {object} options.history Линейка версий из `createHistory`.
 * @param {object} [options.strings] Подписи: {undo, redo, keep, kept, of}.
 * @param {(version: object) => void} [options.onShow] Показать версию.
 * @param {(draft: object) => Promise<object|null>} [options.onKeep] Перенести
 *   черновик в библиотеку; возвращает то, что ответил сервер.
 * @returns {{element: HTMLElement, sync: Function, destroy: Function}}
 */
export function createIterations({ history, strings = {}, onShow, onKeep } = {}) {
    if (!history) throw new Error("[TS Studio] iterations need a history");
    const t = {
        undo: "Undo", redo: "Redo", keep: "Save", kept: "Saved",
        keepFailed: (why) => `Save failed: ${why}`,
        ...strings,
    };

    const element = document.createElement("div");
    element.className = "ts-iter";
    element.style.display = "none";

    const button = (label, title, cls = "") => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = `ts-iter__btn${cls ? ` ${cls}` : ""}`;
        b.textContent = label;
        b.title = title;
        return b;
    };

    const undo = button("↶", t.undo);
    const redo = button("↷", t.redo);
    const count = document.createElement("span");
    count.className = "ts-iter__count";
    const keep = button(t.keep, t.keep, "ts-iter__keep");
    element.append(undo, redo, count, keep);

    let busy = false;
    let active = true;

    function sync() {
        const state = history.state();
        if (!active) { element.style.display = "none"; return; }
        // Одна версия — это просто исходник: откатывать и сохранять нечего,
        // и полоса кнопок только загораживала бы кадр.
        element.style.display = state.total > 1 || state.canKeep ? "" : "none";
        undo.disabled = busy || !state.canUndo;
        redo.disabled = busy || !state.canRedo;
        keep.disabled = busy || !state.canKeep;
        keep.textContent = state.current?.kept ? t.kept : t.keep;
        count.textContent = state.total > 1 ? `${state.index + 1} / ${state.total}` : "";
    }

    undo.addEventListener("click", () => { onShow?.(history.undo()); sync(); });
    redo.addEventListener("click", () => { onShow?.(history.redo()); sync(); });
    keep.addEventListener("click", async () => {
        const draft = history.draft();
        if (!draft || busy) return;
        busy = true;
        sync();
        try {
            const saved = await onKeep?.(draft);
            // Отметка ставится только на ответ сервера: иначе кнопка гаснет, а
            // в библиотеке пусто, и человек об этом узнаёт назавтра.
            if (saved) history.markKept(saved);
        } finally {
            busy = false;
            sync();
        }
    });

    sync();
    // Панель следит за историей сама: раздел не обязан помнить про `sync()`.
    const unsubscribe = history.subscribe?.(sync) || (() => {});
    return {
        element,
        sync,
        /** Раздел ушёл с экрана. История его при этом цела — вернётся с ним. */
        setActive(value) { active = Boolean(value); sync(); },
        /** Пока идёт прогон, откатывать некуда: результат ещё не пришёл. */
        setBusy(value) { busy = Boolean(value); sync(); },
        destroy() { unsubscribe(); element.remove(); },
    };
}

/** Стиль панели. Раскладка своя, цвета — только из токенов темы. */
export const ITERATIONS_CSS = `
.ts-iter{position:absolute;left:50%;bottom:10px;transform:translateX(-50%);z-index:7;
    display:flex;align-items:center;gap:2px;padding:3px;
    border:1px solid var(--ts-border);border-radius:var(--ts-radius-sm);
    background:var(--ts-elevated);box-shadow:var(--ts-shadow)}
.ts-iter__btn{min-width:26px;height:26px;padding:0 8px;cursor:pointer;
    display:inline-flex;align-items:center;justify-content:center;
    border:none;border-radius:calc(var(--ts-radius-sm) - 2px);background:transparent;
    color:var(--ts-muted);font-size:13px;font-family:inherit}
.ts-iter__btn:hover:not(:disabled){color:var(--ts-text);background:var(--ts-surface-hover)}
.ts-iter__btn:disabled{opacity:.35;cursor:default}
.ts-iter__count{padding:0 6px;font-size:11px;color:var(--ts-muted);
    font-variant-numeric:tabular-nums;min-width:34px;text-align:center}
.ts-iter__keep{color:var(--ts-accent);font-size:12px}
.ts-iter__keep:hover:not(:disabled){color:var(--ts-accent);background:var(--ts-accent-soft)}
`;
