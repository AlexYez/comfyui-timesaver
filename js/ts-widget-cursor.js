// Курсор-пальчик над теми виджетами, которые действительно нажимаются.
//
// Ноды классического режима рисуются НА ХОЛСТЕ: кнопки не элементы DOM, и
// `cursor: pointer` из CSS до них не доходит — над всей нодой стоит один курсор
// холста. Отсюда жалоба: наводишь на кнопку, а курсор остаётся стрелкой с
// перекрестием и ничем не показывает, что здесь нажимают.
//
// ⚠️ Мы НИЧЕГО не перехватываем и не подменяем: только читаем позицию, которую
// LiteGraph уже посчитал, и ставим курсор. Слушатель добавлен после родного,
// поэтому `graph_mouse` к этому моменту обновлён.
//
// В Nodes 2.0 (Vue) виджеты — обычный DOM со своим курсором, и делать здесь
// нечего: тогда под указателем просто не окажется холста.

import { app } from "/scripts/app.js";

// Типы, которые реагируют на нажатие. Текстовые поля сюда не входят: они
// открывают редактор, и «пальчик» над строкой ввода читается как ошибка.
const CLICKABLE = new Set(["button", "toggle", "combo"]);

const state = { installed: false, ours: false, previous: "" };

function widgetUnderCursor(canvas) {
    const graph = app?.graph;
    const point = canvas?.graph_mouse;
    if (!graph || !point) return null;

    const [x, y] = point;
    // `visible_nodes` — то, что LiteGraph только что нарисовал; проверять весь
    // граф на каждое движение мыши незачем.
    const node = graph.getNodeOnPos?.(x, y, canvas.visible_nodes);
    if (!node || node.flags?.collapsed) return null;
    return node.getWidgetOnPos?.(x, y) || null;
}

function apply(element, wantPointer) {
    if (wantPointer) {
        if (state.ours) return;
        state.previous = element.style.cursor;
        element.style.cursor = "pointer";
        state.ours = true;
        return;
    }
    if (!state.ours) return;
    // Возвращаем ровно то, что было: LiteGraph ставит свои курсоры (тянем
    // связь, меняем размер), и затирать их нельзя.
    element.style.cursor = state.previous || "";
    state.ours = false;
}

function install() {
    if (state.installed) return;
    const canvas = app?.canvas;
    const element = canvas?.canvas;
    if (!canvas || !element) return;
    state.installed = true;

    const onMove = () => {
        try {
            const widget = widgetUnderCursor(canvas);
            const wanted = Boolean(widget) && CLICKABLE.has(String(widget.type))
                && widget.disabled !== true && widget.computedDisabled !== true;
            apply(element, wanted);
        } catch (error) {
            console.warn("[TS WidgetCursor] pointer check failed", error);
        }
    };

    element.addEventListener("pointermove", onMove, { passive: true });
    // Ушли с холста — курсор больше не наш.
    element.addEventListener("pointerleave", () => apply(element, false), { passive: true });
}

app.registerExtension({
    name: "ts.widgetCursor",
    setup() {
        install();
    },
});
