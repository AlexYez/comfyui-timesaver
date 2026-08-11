// Стопка LoRA — состояние без DOM.
//
// Здесь только список и действия над ним: добавить, убрать, переставить,
// поменять силу, прочитать и записать строку для ноды. Ни одной ссылки на
// документ — поэтому проверяется без браузера, а интерфейс (`ts-lora-loader.js`)
// остаётся про пиксели.
//
// ФОРМАТ ХРАНЕНИЯ — одна строка JSON в скрытом виджете `loras_json`:
//
//     [{"name": "detail.safetensors", "strength": 0.8, "on": true}, ...]
//
// Почему строка, а не набор виджетов: `widgets_values` в ComfyUI позиционный, и
// виджеты, приходящие и уходящие вместе со строками списка, сдвигали бы
// значения соседей в каждом сохранённом workflow. Одна строка — одна позиция,
// сколько бы LoRA в ней ни лежало.

// ⚠️ ГРАНИЦЫ, ШАГ И УМОЛЧАНИЕ НЕ ЗАДАНЫ ЗДЕСЬ. Они берутся у родной
// `LoraLoaderModelOnly` — той самой, в которую нода разворачивается. Значение,
// принятое у нас, но отвергнутое там, было бы обманом; а зашитые копии чисел
// разошлись бы с ComfyUI при первом же его обновлении. Спецификацию приносит
// интерфейс (`setStrengthSpec`), прочитав `/object_info`.
//
// Числа ниже — ТОЛЬКО запасной вариант на случай, если спросить не у кого
// (сервер не ответил, тест работает без ComfyUI). Они совпадают с тем, что
// ComfyUI объявляет сегодня, но истиной считается ответ сервера.
const FALLBACK_SPEC = { min: -100, max: 100, step: 0.01, default: 1 };

let spec = withDecimals(FALLBACK_SPEC);

/** Сколько знаков показывать — выводится из шага, как у виджетов ComfyUI. */
function withDecimals(raw) {
    const step = raw.step > 0 ? raw.step : FALLBACK_SPEC.step;
    const decimals = step >= 1 ? 0 : Math.min(6, Math.ceil(-Math.log10(step)));
    return { ...raw, step, decimals };
}

/**
 * Принять спецификацию силы у родной ноды.
 *
 * Берём только то, что пришло числом: неполный или чужой ответ не должен
 * оставлять нас без границ вовсе.
 *
 * @param {{min?: number, max?: number, step?: number, default?: number}} raw
 *   Объект из `/object_info` (`input.required.strength_model[1]`).
 */
export function setStrengthSpec(raw) {
    // ⚠️ Через `Number()` в лоб нельзя: `Number(null)` и `Number("")` — это 0,
    // и «поля нет» превратилось бы в «граница равна нулю».
    const pick = (key) => {
        const value = raw?.[key];
        const ok = typeof value === "number"
            || (typeof value === "string" && value.trim() !== "");
        const number = ok ? Number(value) : NaN;
        return Number.isFinite(number) ? number : FALLBACK_SPEC[key];
    };
    const min = pick("min");
    const max = pick("max");
    spec = withDecimals({
        min: Math.min(min, max),
        max: Math.max(min, max),
        step: pick("step"),
        default: pick("default"),
    });
    return strengthSpec();
}

/** Действующая спецификация силы (копия — менять её можно только через setter). */
export function strengthSpec() {
    return { ...spec };
}

/** Показать силу так же, как её показывает родная нода. */
export function formatStrength(value) {
    return clampStrength(value).toFixed(spec.decimals);
}

/** Шаг стрелкой — ровно шаг родной ноды. */
export function stepStrength(value, direction) {
    return clampStrength(clampStrength(value) + Math.sign(direction) * spec.step);
}

/** Сила в допустимых границах и без хвоста с плавающей точкой. */
export function clampStrength(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return spec.default;
    const bounded = Math.max(spec.min, Math.min(spec.max, number));
    // Округление по шагу: «0.7500000000000001» в сохранённом workflow — мусор,
    // который потом читают глазами.
    const factor = 10 ** spec.decimals;
    return Math.round(bounded * factor) / factor;
}

/**
 * Разобрать сохранённую строку.
 *
 * ⚠️ Битую строку возвращаем пустым списком, а не бросаем: workflow мог прийти
 * из чужой сборки или из будущей версии пака, и терять из-за этого весь граф
 * человек не подписывался.
 *
 * @param {string} raw содержимое виджета
 * @returns {Array<{name: string, strength: number, on: boolean}>}
 */
export function parseStack(raw) {
    let data;
    try {
        data = JSON.parse(raw || "[]");
    } catch {
        return [];
    }
    if (!Array.isArray(data)) return [];
    return data
        .map((entry) => (entry && typeof entry === "object" ? {
            name: String(entry.name || "").trim(),
            strength: clampStrength(entry.strength ?? spec.default),
            on: entry.on !== false,
        } : null))
        .filter((entry) => entry && entry.name);
}

/** Записать список в строку — ровно в том виде, в каком его читает нода. */
export function serialiseStack(stack) {
    return JSON.stringify((stack || []).map((entry) => ({
        name: entry.name,
        strength: clampStrength(entry.strength),
        on: entry.on !== false,
    })));
}

/**
 * Добавить LoRA в конец.
 *
 * Повтор разрешён намеренно: одну и ту же LoRA иногда ставят дважды с разной
 * силой — приём редкий, но настоящий, и запрещать его не наше дело.
 */
export function addLora(stack, name, strength = null) {
    const clean = String(name || "").trim();
    if (!clean) return stack.slice();
    // null — «как по умолчанию у родной ноды», а не «ноль».
    const value = strength === null ? spec.default : strength;
    return [...stack, { name: clean, strength: clampStrength(value), on: true }];
}

/** Убрать строку по номеру. */
export function removeAt(stack, index) {
    if (!(index >= 0 && index < stack.length)) return stack.slice();
    const out = stack.slice();
    out.splice(index, 1);
    return out;
}

/**
 * Переставить строку с места на место.
 *
 * Порядок — не украшение: LoRA накладываются последовательно, и результат от
 * перестановки меняется. Поэтому перетаскивание списка обязано двигать именно
 * то, за что взялись, а не менять две строки местами.
 */
export function moveItem(stack, from, to) {
    const out = stack.slice();
    if (!(from >= 0 && from < out.length)) return out;
    const target = Math.max(0, Math.min(out.length - 1, to));
    if (target === from) return out;
    const [item] = out.splice(from, 1);
    out.splice(target, 0, item);
    return out;
}

/** Поменять силу одной строки. */
export function setStrength(stack, index, value) {
    if (!(index >= 0 && index < stack.length)) return stack.slice();
    const out = stack.slice();
    out[index] = { ...out[index], strength: clampStrength(value) };
    return out;
}

/** Включить или отложить строку, не удаляя её. */
export function setEnabled(stack, index, on) {
    if (!(index >= 0 && index < stack.length)) return stack.slice();
    const out = stack.slice();
    out[index] = { ...out[index], on: Boolean(on) };
    return out;
}

/**
 * Отобрать имена по тому, что человек напечатал в поиске.
 *
 * Сравнение без учёта регистра и по КУСКАМ: «det xl» находит
 * `add_detail_XL.safetensors`. Люди помнят LoRA обрывками, а не целиком.
 */
export function filterNames(names, query) {
    const parts = String(query || "").toLowerCase().split(/\s+/).filter(Boolean);
    if (!parts.length) return names.slice();
    return names.filter((name) => {
        const haystack = String(name).toLowerCase();
        return parts.every((part) => haystack.includes(part));
    });
}

/** Короткое имя для строки списка: без папок и без расширения. */
export function shortName(name) {
    const tail = String(name || "").split(/[\\/]/).pop() || "";
    return tail.replace(/\.(safetensors|ckpt|pt|bin|sft)$/i, "");
}
