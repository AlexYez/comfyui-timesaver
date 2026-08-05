// Что именно уходит в граф — правила, а не проводка.
//
// Между «что человек выставил в деке» и «что получит ComfyUI» стоят три
// правила. Каждое из них добыто измерениями и каждое легко отменить одной
// строкой, не заметив: здесь они отдельно от проводки, с тестами.
//
//   1. в граф идёт только то, что он объявил. Дека собрана под один бэкенд, а
//      прогон может уйти в другой (генерация с референсом уходит в граф
//      редактирования того же семейства). Лишний параметр — не «мелочь»:
//      патчер справедливо ругается на то, чего в графе нет.
//   2. сила и есть режим. Раньше рядом жили ползунок и переключатель Replace,
//      и при включённом переключателе ползунок ничего не значил. Теперь шкала
//      одна, и её последняя ступень И ЕСТЬ замена.
//   3. доработка идёт без размышлений LanPaint. Его внутренний цикл согласует
//      перерисованное с окружением — на полной замене это нужно, а на
//      доработке уводит результат далеко за запрошенную силу.

// Порог живёт там же, где геометрия выреза: он определяет не только режим
// интерфейса, но и то, насколько увеличивается вырез. Вторая копия числа
// разошлась бы с первой молча.
import { REPLACE_DENOISE } from "../../_studio/_crop_geometry.js";

export { REPLACE_DENOISE };

/**
 * Объявляет ли граф такой параметр — прямым входом или литералом.
 *
 * @param {object} target бэкенд, в который уйдёт прогон
 * @param {string} param имя параметра
 */
export function graphDeclares(target, param) {
    return Boolean(target?.spec?.params?.has?.(param)
        || target?.spec?.literals?.has?.(param));
}

/**
 * Значения деки, годные для этого графа.
 *
 * Составные значения (сид, LoRA, референсы) сюда не попадают: у них свои
 * пути в патчере, и класть их рядом с числами значило бы отправить в граф
 * объект там, где он ждёт число.
 *
 * @param {object} values всё, что помнит дека
 * @param {object} target бэкенд прогона
 * @returns {object}
 */
export function collectDeckValues(values, target) {
    const out = {};
    for (const [param, value] of Object.entries(values || {})) {
        if (param === "seed" || param === "loras" || param === "__refs") continue;
        if (typeof value === "object" && value !== null) continue;
        if (!graphDeclares(target, param)) continue;
        out[param] = value;
    }
    return out;
}

/**
 * Правило силы: одна шкала вместо ползунка и тумблера.
 *
 * Меняет переданный объект и возвращает решение — оно же нужно интерфейсу,
 * чтобы показать человеку, чем закончится эта ступень.
 *
 * Числа, ради которых это устроено именно так, измерены на одном сиде:
 * LanPaint с размышлениями двигал пиксель внутри маски в среднем на 69.5
 * против 32.8 у обычного прохода — молодое лицо при силе 0.45 возвращалось
 * пожилым. С нулём размышлений LanPaint вырождается в обычный сэмплер
 * (расхождение с настоящим KSampler — 0.057 из 255), и доработка ускоряется
 * в девять раз: 24 секунды против 209.
 *
 * @param {object} runValues значения прогона (меняются на месте)
 * @param {object} target бэкенд прогона
 * @returns {{replacing: boolean}|null} null, если силы в этом прогоне нет
 */
export function applyStrengthRule(runValues, target) {
    const strength = Number(runValues?.denoise);
    if (!Number.isFinite(strength)) return null;
    const replacing = strength >= REPLACE_DENOISE;
    if (replacing) runValues.denoise = 1.0;
    // Klein принимает режим прямо в ноду; у остальных семейств такого входа в
    // графе нет, и передавать его туда нельзя.
    if (graphDeclares(target, "replace")) runValues.replace = replacing;
    if (!replacing && graphDeclares(target, "think_steps")) runValues.think_steps = 0;
    return { replacing };
}

/**
 * Стили дописываются к промпту так же, как это делает нода-селектор.
 *
 * Иначе в галерее сохранилось бы одно, а посчиталось другое: параметры
 * прогона должны воспроизводить ровно то, что ушло в граф.
 *
 * @param {string} prompt что написал человек
 * @param {string} styleTail стили через запятую
 * @returns {string}
 */
export function withStyles(prompt, styleTail) {
    if (!styleTail) return prompt;
    if (typeof prompt !== "string") return prompt;
    const base = prompt.trim().replace(/[,\s]+$/, "");
    return base ? `${base}, ${styleTail}` : styleTail;
}

/**
 * Референсы: заполненные уходят в граф, пустые — снимаются с него.
 *
 * @param {object} refs слоты референсов
 * @param {object} target бэкенд прогона
 * @returns {{values: object, drop: string[]}}
 */
export function collectRefs(refs, target) {
    const out = { values: {}, drop: [] };
    for (const [name, annotated] of Object.entries(refs || {})) {
        if (!target?.spec?.params?.has?.(name)) continue;
        if (annotated) out.values[name] = annotated;
        else out.drop.push(name);
    }
    return out;
}
