/**
 * Своё контекстное меню для поля текста песни.
 *
 * ⚠️ Меню приходится делать своим целиком, включая «копировать» и «вставить».
 * Родное меню браузера расширить нечем: добавить в него пункт нельзя никак, а
 * ради одного «сделать ударной» человек не должен терять привычные команды —
 * поэтому они здесь все.
 *
 * ⚠️ «Вставить» работает через `navigator.clipboard.readText()`, и это
 * единственный путь: `document.execCommand("paste")` браузеры отключили из
 * соображений безопасности. Чтение буфера может потребовать разрешения — если
 * его не дали, пункт честно говорит об этом, а не делает вид, что вставил.
 */

/** Один экземпляр на страницу: два открытых меню — это уже недоразумение. */
let openMenu = null;

function closeOpenMenu() {
    if (!openMenu) return;
    openMenu.remove();
    openMenu = null;
    document.removeEventListener("pointerdown", onDocumentPointerDown, true);
    document.removeEventListener("keydown", onDocumentKeyDown, true);
    window.removeEventListener("blur", closeOpenMenu);
}

function onDocumentPointerDown(event) {
    if (openMenu && !openMenu.contains(event.target)) closeOpenMenu();
}

function onDocumentKeyDown(event) {
    if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        closeOpenMenu();
    }
}

/**
 * Показать меню у курсора.
 *
 * @param {{x:number, y:number}} at точка в координатах окна.
 * @param {Array<{label:string, hint?:string, disabled?:boolean, separator?:boolean,
 *   onSelect?:() => void}>} items пункты; `separator: true` рисует черту.
 * @param {string} [scopeClass] класс темы, который надо унаследовать.
 */
export function showContextMenu(at, items, scopeClass = "ts-ui") {
    closeOpenMenu();

    const menu = document.createElement("div");
    menu.className = `${scopeClass} ts-song-menu`;
    menu.setAttribute("role", "menu");

    for (const item of items) {
        if (item?.separator) {
            const line = document.createElement("div");
            line.className = "ts-song-menu__sep";
            menu.appendChild(line);
            continue;
        }
        const button = document.createElement("button");
        button.type = "button";
        button.className = "ts-song-menu__item";
        button.textContent = item.label;
        if (item.hint) button.title = item.hint;
        if (item.disabled) {
            button.disabled = true;
        } else {
            button.addEventListener("click", () => {
                closeOpenMenu();
                item.onSelect?.();
            });
        }
        menu.appendChild(button);
    }

    // Рисуем за пределами ноды: внутри виджета меню обрезалось бы переполнением,
    // а на холсте с зумом ещё и масштабировалось бы вместе с нодой.
    document.body.appendChild(menu);

    // Положение считаем ПОСЛЕ вставки: до неё у меню нет размеров, и у нижнего
    // края экрана оно уезжало за границу.
    const rect = menu.getBoundingClientRect();
    const maxLeft = Math.max(4, window.innerWidth - rect.width - 4);
    const maxTop = Math.max(4, window.innerHeight - rect.height - 4);
    menu.style.left = `${Math.min(Math.max(4, at.x), maxLeft)}px`;
    menu.style.top = `${Math.min(Math.max(4, at.y), maxTop)}px`;

    openMenu = menu;
    document.addEventListener("pointerdown", onDocumentPointerDown, true);
    document.addEventListener("keydown", onDocumentKeyDown, true);
    window.addEventListener("blur", closeOpenMenu);
}

/** Закрыть меню снаружи — например, когда нода уходит с холста. */
export function hideContextMenu() {
    closeOpenMenu();
}

/**
 * Прочитать буфер обмена. Возвращает пустую строку, если доступа нет.
 *
 * @returns {Promise<string>}
 */
export async function readClipboardText() {
    try {
        if (navigator.clipboard?.readText) return await navigator.clipboard.readText();
    } catch (error) {
        console.warn("[TS Song Creator] clipboard read refused:", error);
    }
    return "";
}

/**
 * Положить текст в буфер обмена.
 *
 * @param {string} text что копируем.
 * @returns {Promise<boolean>} удалось ли.
 */
export async function writeClipboardText(text) {
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(String(text ?? ""));
            return true;
        }
    } catch (error) {
        console.warn("[TS Song Creator] clipboard write refused:", error);
    }
    return false;
}
