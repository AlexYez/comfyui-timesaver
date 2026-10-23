// Источник файла по проводу: TS Video Loader и TS Audio Loader принимают путь
// на сокет `path` (обычно от TS Batch Source). Модуль общий для обеих нод: он
// отвечает «подключён ли сокет» и «какой файл пришёл последним», а сами ноды
// решают, как это показать.
//
// ⚠️ Подключение меняется не только руками: при загрузке workflow LiteGraph
// восстанавливает связи ПОСЛЕ onNodeCreated, поэтому состояние перечитывается
// и на onConnectionsChange, и на onConfigure, и ещё раз через кадр.

/** Подключён ли вход с этим именем. */
export function isInputLinked(node, name) {
    const input = node?.inputs?.find((item) => item?.name === name);
    return input?.link !== undefined && input?.link !== null;
}

/**
 * Следить за сокетом и за файлом, который по нему пришёл.
 *
 * @param {object} node нода LiteGraph.
 * @param {string} name имя входа.
 * @param {(state: {linked: boolean, file: string}) => void} onChange вызывается
 *     при каждом изменении; `file` — имя последнего файла с провода (из ui
 *     выполнения, ключ `ts_wired_path`), пустое до первого прогона.
 * @returns {() => {linked: boolean, file: string}} чтение текущего состояния.
 */
export function watchWiredSource(node, name, onChange) {
    const state = { linked: false, file: "" };

    const refresh = () => {
        const linked = isInputLinked(node, name);
        if (!linked) state.file = "";
        state.linked = linked;
        try {
            onChange({ ...state });
        } catch (error) {
            console.warn("[TS WiredSource] could not show the wired state", error);
        }
    };

    const chain = (method, after) => {
        const original = node[method];
        node[method] = function chained(...args) {
            const result = original?.apply(this, args);
            after(...args);
            return result;
        };
    };

    chain("onConnectionsChange", () => refresh());
    chain("onConfigure", () => requestAnimationFrame(refresh));
    chain("onExecuted", (message) => {
        // ⚠️ При списковом прогоне ядро склеивает ui всех заданий в ОДИН список
        // и присылает его разом (замерено: clip2, clip10, deep). Последний —
        // тот, что прошёл через ноду последним.
        const files = message?.ts_wired_path;
        const file = Array.isArray(files) ? files[files.length - 1] : undefined;
        if (typeof file === "string" && file) {
            state.file = file;
            refresh();
        }
    });
    requestAnimationFrame(refresh);
    return () => ({ ...state });
}
