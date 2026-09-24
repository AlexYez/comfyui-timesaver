// Сверка списка TS Files Downloader с загрузчиками открытого графа.
//
// Нода качает туда, куда велит строка списка, а загрузчик ищет туда, куда
// указывает его виджет. Это ДВА независимых места в одном workflow, и
// разъехаться им ничто не мешает: автор поменял подпапку у загрузчика и забыл
// строку — граф уходит подписчику, модель качается, загрузчик её не видит.
// Здесь каждая строка сравнивается с тем, что загрузчики графа реально ждут.
//
// ⚠️ Модуль чистый: ни `app`, ни `api`. Его проверяет настоящий Node в тестах,
// а кнопка «Взять модели из workflow» судит о папках той же функцией
// `sameTarget`, что и сверка, — двух мнений о том, «та ли это папка», в
// одной ноде быть не должно.

// ComfyUI читает две директории для некоторых категорий и держит старое имя
// живым (folder_paths.map_legacy): текстовый энкодер ищется и в `models/clip`,
// и в `models/text_encoders`, UNET — в `models/unet` и `models/diffusion_models`.
// Строка, нацеленная в любое из двух имён, работает, поэтому расхождением это
// не считается.
export const FOLDER_ALIASES = {
    unet: "diffusion_models",
    clip: "text_encoders",
    t2i_adapter: "controlnet",
};

/** Канонический вид папки: разделители, регистр, синонимы, приставка `models/`. */
export function canonTarget(value) {
    const parts = String(value || "")
        .replace(/\\/g, "/")
        .split("/")
        .map((part) => part.trim().toLowerCase())
        .filter((part) => part && part !== "." && part !== "..");
    if (parts[0] === "models") parts.shift();
    if (parts.length) parts[0] = FOLDER_ALIASES[parts[0]] || parts[0];
    return parts.join("/");
}

/** Сравнить две папки, записанные человеком: разделители и регистр гуляют. */
export function sameTarget(a, b) {
    return canonTarget(a) === canonTarget(b);
}

// Файлы, о которых загрузчики вообще могут что-то сказать. Архив, конфиг или
// картинка в списке законны, но ни один загрузчик их не выбирает — судить их
// по графу нельзя.
// ⚠️ `tsmodel` здесь обязателен: запертые модели для подписчиков раздаются
// именно этой нодой.
export const MODEL_EXT = /\.(safetensors|ckpt|pt|pth|bin|gguf|onnx|sft|safetensor|tsmodel)$/i;

// Путь, названный МЕСТОМ, а не папкой модели: буква диска, UNC, корень, `~`,
// переменная окружения. Такая строка ведёт мимо папок ComfyUI, и сравнивать
// её с виджетом загрузчика бессмысленно.
const LOCATION = /^[a-z]:|^[\\/]|^~|%[^%]+%|\$\{?[A-Za-z_]/i;

/** Подпапка в сравнимом виде: прямые разделители, без краёв, без регистра.
 *  Регистр снимается и на Linux, где пути ему чувствительны: здесь лучше
 *  промолчать о `Wan` против `wan`, чем кричать о них там, где их нет. */
function normSub(value) {
    return String(value || "")
        .replace(/\\/g, "/")
        .split("/")
        .map((part) => part.trim().toLowerCase())
        .filter((part) => part && part !== ".")
        .join("/");
}

/**
 * Качает ли строка туда, где ищет загрузчик.
 *
 * Одна функция на всю ноду: ею судят и сверка списка, и кнопка «Взять модели
 * из workflow». Точно — по ключу реестра и подпапке, когда сервер сказал, под
 * каким именем загрузчики найдут строку (`scopes`), а категория загрузчика
 * известна; иначе — по имени папки с учётом синонимов.
 *
 * @param {string} target  папка строки, как её написали.
 * @param {{folder:string, category?:string, sub?:string}} want  где ищет загрузчик.
 * @param {Array<[string, string]>|null|undefined} scopes  ответ сервера для строки.
 */
export function sameDestination(target, want, scopes) {
    if (Array.isArray(scopes) && want?.category) {
        const category = want.category.toLowerCase();
        const sub = normSub(want.sub);
        return scopes.some(([key, prefix]) =>
            String(key || "").toLowerCase() === category && normSub(prefix) === sub);
    }
    return sameTarget(target, want?.folder);
}

/**
 * Что загрузчики графа говорят об одной строке списка.
 *
 * Два способа сравнить, и первый главный:
 *
 * 1. ТОЧНО, по реестру ComfyUI. Сервер говорит, под каким именем загрузчики
 *    найдут то, что скачает строка (`scopes`: `[категория, подпапка]`), а у
 *    загрузчика известна его категория. Совпали категория и подпапка — строка
 *    качает ровно туда, где загрузчик ищет, как бы ни называлась папка на
 *    диске (корень из `extra_model_paths.yaml` может зваться чем угодно).
 * 2. По имени папки — когда категории загрузчика никто не знает (чужая нода,
 *    чья выпадашка пуста на этой машине). С учётом синонимов `unet`/`clip`.
 *
 * @param {string} fileName   имя файла из адреса строки.
 * @param {string} target     папка строки, как её написали.
 * @param {Map<string, Array<{folder:string, category?:string, sub?:string}>>|null} expected
 *   имя файла в нижнем регистре -> где его ждут загрузчики графа. `folder` —
 *   `<категория>/<подпапка>` для показа; пустой `folder` — загрузчик есть, но
 *   откуда он читает, неизвестно.
 * @param {Array<[string, string]>|null|undefined} [scopes]  ответ сервера для
 *   строки: массив — сравнивать точно; `null` — ответа ещё нет, с приговором
 *   «не та папка» подождать; `undefined` — ответа не будет, сравнивать имена.
 * @returns {{kind: "mismatch", wanted: string[]} | {kind: "unused"} | null}
 *   `null` — сказать нечего: всё сходится, либо судить не по чему.
 */
export function judgeLine(fileName, target, expected, scopes = undefined) {
    if (!expected) return null;
    const name = String(fileName || "").trim();
    if (!MODEL_EXT.test(name)) return null;
    const written = String(target || "").trim();
    if (!written || LOCATION.test(written)) return null;

    const wanted = expected.get(name.toLowerCase());
    if (!wanted || !wanted.length) return { kind: "unused" };

    const matches = (want) => sameDestination(written, want, scopes);

    const known = wanted.filter((want) => want.folder);
    if (known.some(matches)) return null;
    // Один из загрузчиков файл выбирает, но из какой папки — неизвестно (чужая
    // нода с пустой выпадашкой). Утверждать расхождение не на чем.
    if (known.length < wanted.length) return null;
    // Сервер ещё не ответил: по одним именам можно ошибиться, а мигнуть
    // красным и тут же погаснуть — хуже, чем подождать долю секунды.
    if (scopes === null) return null;

    const unique = [];
    for (const want of known) {
        if (!unique.some((seen) => sameTarget(seen, want.folder))) unique.push(want.folder);
    }
    return { kind: "mismatch", wanted: unique };
}

/** Отпечаток ожиданий: перерисовывать список, только когда они изменились. */
export function expectationKey(expected) {
    if (!expected) return "";
    return [...expected.entries()]
        .map(([name, wants]) => `${name}=${wants
            .map((want) => `${want.folder}@${want.category || ""}`)
            .sort()
            .join("|")}`)
        .sort()
        .join("\n");
}
