// Контекстное меню рабочей области — и реестр того, что в нём есть.
//
// Две вещи в одном файле, и это осознанно: меню без команд бесполезно, а
// команда без места, где её показать, — тем более. Дальше сюда будут
// добавляться функции, и добавляться они должны в ОДНО место.
//
// Разделение внутри файла жёсткое:
//
//   createContextMenu()   — как меню выглядит и ведёт себя. Ничего не знает ни
//                           про студию, ни про режимы: ему дают список пунктов.
//   registerStageCommand  — реестр команд. Команда описывается данными, а не
//   commandsFor(ctx)        кодом меню: где показывать, когда включена, что
//                           делает. Новая функция = ещё одна регистрация.
//
// Контракт команды:
//
//   {
//     id:      "studio.send.inpaint",     // уникальный, стабильный
//     group:   "send",                    // соседи по группе идут вместе,
//                                         // между группами — разделитель
//     groupOrder: 10,                     // порядок САМИХ групп; по алфавиту
//                                         // «canvas» встал бы выше «send»
//     order:   10,                        // меньше — выше внутри группы
//     label(ctx) -> string,               // текст пункта (уже локализованный)
//     hint(ctx)  -> string|"",            // подпись под текстом
//     visible(ctx) -> boolean,            // показывать ли вообще
//     enabled(ctx) -> boolean,            // кликабельна ли
//     run(ctx) -> void|Promise            // сама работа
//   }
//
// `ctx` собирает вызывающий: он один знает, что сейчас на экране. Меню передаёт
// его командам как есть и в него не заглядывает.
//
// ⚠️ Команды НЕ хранят состояние. Между двумя открытиями меню может смениться
// режим, картинка и язык — поэтому и `visible`, и `label` спрашиваются заново
// на каждом открытии.

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";

const STYLE_ID = "ts-studio-ctxmenu-styles";

function ensureStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    // Только раскладка; цвета — из токенов --ts-*.
    // NOTE: no backticks in this comment — the whole stylesheet is one template
    // literal, and one backtick would end it.
    style.textContent = `
/* ⚠️ Выше оверлея студии. У полноэкранной модалки пака z-index 11000, и меню с
   шестьюдесятью открывалось ПОД ней: в DOM оно есть, класс is-open стоит, а на
   экране нет ничего — и браузерное меню тоже подавлено. Именно так это и
   выглядело у владельца. Проверять такое надо попаданием курсора
   (elementFromPoint), а не наличием класса. */
.ts-ctxmenu{position:fixed;z-index:11050;min-width:212px;max-width:320px;padding:5px;
  display:none;flex-direction:column;gap:1px;background:var(--ts-elevated);
  border:1px solid var(--ts-border);border-radius:var(--ts-radius);
  box-shadow:var(--ts-shadow);font-family:var(--ts-font);font-size:var(--ts-fs)}
.ts-ctxmenu.is-open{display:flex}
.ts-ctxmenu__item{display:flex;flex-direction:column;align-items:flex-start;gap:1px;
  padding:6px 9px;border:0;border-radius:var(--ts-radius-sm);background:none;
  color:var(--ts-text);font:inherit;text-align:left;cursor:pointer;width:100%}
.ts-ctxmenu__item:hover:not(:disabled),.ts-ctxmenu__item:focus-visible{
  background:var(--ts-surface-hover);outline:none}
.ts-ctxmenu__item:disabled{color:var(--ts-faint);cursor:default}
.ts-ctxmenu__hint{font-size:var(--ts-fs-xs);color:var(--ts-muted)}
.ts-ctxmenu__item:disabled .ts-ctxmenu__hint{color:var(--ts-faint)}
.ts-ctxmenu__sep{height:1px;margin:4px 2px;background:var(--ts-border-soft)}
`;
    document.head.appendChild(style);
}

// ---------------------------------------------------------------------------
// Реестр команд
// ---------------------------------------------------------------------------

/** @type {object[]} */
const commands = [];

/**
 * Добавить команду в меню.
 *
 * @param {object} command См. контракт в шапке файла.
 * @returns {() => void} Снять команду обратно (для тестов и временных пунктов).
 */
export function registerStageCommand(command) {
    if (!command || !command.id || typeof command.run !== "function") {
        throw new Error("[TS Studio] a stage command needs an id and a run()");
    }
    const existing = commands.findIndex((item) => item.id === command.id);
    // Повторная регистрация того же id — замена, а не второй пункт: студия
    // пересобирается при смене модели, и дубли иначе копились бы.
    if (existing >= 0) commands.splice(existing, 1, command);
    else commands.push(command);
    return () => {
        const at = commands.findIndex((item) => item.id === command.id);
        if (at >= 0) commands.splice(at, 1);
    };
}

/** Забыть все команды — только для тестов. */
export function resetStageCommands() {
    commands.length = 0;
}

/**
 * Пункты для текущего положения дел, уже разложенные по группам.
 *
 * @param {object} ctx Что сейчас на экране; смысл знает вызывающий.
 * @returns {object[]} Готовые пункты в порядке показа.
 */
export function commandsFor(ctx) {
    return commands
        .filter((command) => {
            try {
                return command.visible ? Boolean(command.visible(ctx)) : true;
            } catch (err) {
                // Сломанная команда не должна уносить с собой всё меню.
                console.warn(`[TS Studio] command ${command.id} failed visible()`, err);
                return false;
            }
        })
        .map((command) => ({
            id: command.id,
            group: command.group || "",
            groupOrder: Number.isFinite(command.groupOrder) ? command.groupOrder : 100,
            order: Number.isFinite(command.order) ? command.order : 100,
            label: safeText(command, "label", ctx),
            hint: safeText(command, "hint", ctx),
            disabled: command.enabled ? !command.enabled(ctx) : false,
            run: () => command.run(ctx),
        }))
        .sort((a, b) => (a.group === b.group
            ? a.order - b.order
            : a.groupOrder - b.groupOrder
              || String(a.group).localeCompare(String(b.group))));
}

function safeText(command, key, ctx) {
    const value = command[key];
    if (typeof value === "function") {
        try {
            return String(value(ctx) ?? "");
        } catch (err) {
            console.warn(`[TS Studio] command ${command.id} failed ${key}()`, err);
            return "";
        }
    }
    return value === undefined || value === null ? "" : String(value);
}

// ---------------------------------------------------------------------------
// Само меню
// ---------------------------------------------------------------------------

/**
 * Контекстное меню, живущее в конце <body>.
 *
 * В body, а не рядом с областью: студия открывается и полноэкранным оверлеем, и
 * внутри ноды, и меню, обрезанное `overflow:hidden` родителя, — классическая
 * беда таких панелей.
 *
 * @param {object} [options]
 * @param {(error: Error, item: object) => void} [options.onError] Что делать,
 *   когда команда упала: показать это человеку умеет только вызывающий.
 * @param {HTMLElement} [options.host] Куда встроиться. По умолчанию `body`, но
 *   владелец полноэкранного оверлея обязан передать СВОЙ корень: иначе меню
 *   остаётся в слое ниже и оказывается за оверлеем.
 * @returns {{open: Function, close: Function, isOpen: Function, teardown: Function}}
 */
export function createContextMenu({ onError, host } = {}) {
    ensureStyles();
    const element = document.createElement("div");
    element.className = `${TS_UI_CLASS} ts-ctxmenu`;
    (host || document.body).appendChild(element);

    function close() {
        if (!element.classList.contains("is-open")) return;
        element.classList.remove("is-open");
        element.replaceChildren();
        document.removeEventListener("pointerdown", onDocumentDown, true);
        document.removeEventListener("keydown", onKeyDown, true);
        window.removeEventListener("blur", close);
        window.removeEventListener("resize", close);
    }

    function onDocumentDown(event) {
        if (!element.contains(event.target)) close();
    }

    function onKeyDown(event) {
        if (event.key === "Escape") {
            event.stopPropagation();
            close();
            return;
        }
        if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
        const items = [...element.querySelectorAll("button:not(:disabled)")];
        if (!items.length) return;
        event.preventDefault();
        const at = items.indexOf(document.activeElement);
        const step = event.key === "ArrowDown" ? 1 : -1;
        const next = at < 0 ? 0 : (at + step + items.length) % items.length;
        items[next].focus();
    }

    return {
        element,
        /**
         * Показать меню в точке экрана.
         *
         * @param {number} clientX
         * @param {number} clientY
         * @param {object[]} items Пункты из `commandsFor`.
         * @returns {boolean} Показалось ли (пустое меню не показывается).
         */
        open(clientX, clientY, items) {
            close();
            if (!items || !items.length) return false;
            let lastGroup = null;
            for (const item of items) {
                if (lastGroup !== null && item.group !== lastGroup) {
                    const sep = document.createElement("div");
                    sep.className = "ts-ctxmenu__sep";
                    element.appendChild(sep);
                }
                lastGroup = item.group;
                const button = document.createElement("button");
                button.type = "button";
                button.className = "ts-ctxmenu__item";
                button.disabled = Boolean(item.disabled);
                const label = document.createElement("span");
                label.textContent = item.label;
                button.appendChild(label);
                if (item.hint) {
                    const hint = document.createElement("span");
                    hint.className = "ts-ctxmenu__hint";
                    hint.textContent = item.hint;
                    button.appendChild(hint);
                }
                button.addEventListener("click", async () => {
                    close();
                    try {
                        await item.run();
                    } catch (err) {
                        onError?.(err instanceof Error ? err : new Error(String(err)), item);
                    }
                });
                element.appendChild(button);
            }

            // Ставим меню так, чтобы оно не уезжало за край экрана. Это
            // единственный замер в модуле, и он неизбежен: положение курсора
            // существует только в экранных координатах.
            element.classList.add("is-open");
            element.style.left = "0px";
            element.style.top = "0px";
            const box = element.getBoundingClientRect();
            const x = Math.max(4, Math.min(clientX, window.innerWidth - box.width - 4));
            const y = Math.max(4, Math.min(clientY, window.innerHeight - box.height - 4));
            element.style.left = `${x}px`;
            element.style.top = `${y}px`;

            document.addEventListener("pointerdown", onDocumentDown, true);
            document.addEventListener("keydown", onKeyDown, true);
            window.addEventListener("blur", close);
            window.addEventListener("resize", close);
            return true;
        },
        close,
        isOpen: () => element.classList.contains("is-open"),
        teardown() {
            close();
            element.remove();
        },
    };
}

// ---------------------------------------------------------------------------
// Команды студии
// ---------------------------------------------------------------------------

/**
 * Зарегистрировать штатные команды рабочей области.
 *
 * Вынесено в функцию, а не выполняется на импорте: тесты собирают реестр сами,
 * а студия зовёт это один раз при открытии.
 *
 * ⚠️ Список разделов ЖИВОЙ. Он собирается из установленных моделей, а те
 * приезжают асинхронно: снимок, сделанный при открытии студии, был почти
 * всегда пустым — и в меню оставался один пункт «очистить холст». Поэтому
 * `modes` можно (и лучше) передавать функцией, а саму регистрацию повторять
 * при каждой пересборке рельса: команды заменяются по id, дублей не будет.
 *
 * @param {object} deps
 * @param {string[]|(() => string[])} deps.modes Идентификаторы разделов.
 * @param {(modeId: string) => string} deps.modeLabel Название раздела.
 * @param {object} deps.strings Подписи меню: {sendTo, clear, clearHint}.
 */
export function registerStudioStageCommands({ modes, modeLabel, strings }) {
    // ⚠️ Генерация в список НЕ входит. Она делает картинку из текста; всё, что
    // туда можно отправить, ложится в слот референса — а это уже другое
    // действие, и человек узнавал бы о нём только после клика. Решение
    // владельца: пункта быть не должно.
    //
    // Пачка тоже: у неё нет рабочего кадра — она принимает промпты. Картинку
    // в неё бросают прямо на поверхность, и оттуда читают промпт; «отправить»
    // означало бы что-то другое.
    const SKIP = new Set(["generate", "batch"]);
    const liveModes = () => ((typeof modes === "function" ? modes() : modes) || [])
        .filter((id) => !SKIP.has(id));
    for (const target of liveModes()) {
        registerStageCommand({
            id: `studio.send.${target}`,
            group: "send",
            groupOrder: 10,
            order: liveModes().indexOf(target),
            // Отправлять картинку самому себе незачем, и без картинки — тоже.
            // Раздел, из-под которого ушла последняя модель, тоже не предлагаем:
            // список живой, а команда, зарегистрированная однажды, — нет.
            visible: (ctx) => ctx.modeId !== target
                && liveModes().includes(target)
                && Boolean(ctx.image?.url),
            label: () => strings.sendTo(modeLabel(target)),
            run: (ctx) => ctx.actions.sendTo(target),
        });
    }

    registerStageCommand({
        id: "studio.canvas.clear",
        group: "canvas",
        // Ниже отправки: очистка — редкое и разрушительное действие, ему не
        // место первым пунктом под курсором.
        groupOrder: 90,
        order: 0,
        label: () => strings.clear,
        hint: () => strings.clearHint,
        enabled: (ctx) => Boolean(ctx.image?.url) || Boolean(ctx.canClear),
        run: (ctx) => ctx.actions.clear(),
    });
}
