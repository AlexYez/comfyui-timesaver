// TS Studio kit — пачка заданий и порядок их прохождения (core layer, no DOM).
//
// ГЛАВНАЯ МЫСЛЬ, РАДИ КОТОРОЙ ЭТО НЕ «ПРОГНАТЬ N РАЗ». Каждая смена модели —
// это выгрузка одной и загрузка другой: десятки секунд и вся видеопамять. Если
// гнать задания по одному целиком («улучшил промпт — нарисовал — улучшил —
// нарисовал»), машина только и делает, что переставляет модели. Поэтому пачка
// идёт ФАЗАМИ: сначала одна модель проходит по всем заданиям, потом следующая.
//
//     фаза «промпты»:  Qwen загружен один раз → N текстов
//     фаза «картинки»: модель загружена один раз → N кадров
//
// Отсюда всё устройство модуля: очередь знает о фазах и о заданиях, и НИЧЕГО не
// знает ни о промптах, ни о картинках, ни о ComfyUI. Что делать с заданием,
// решает переданная работа (`run`), а модуль отвечает за порядок, остановку,
// ошибки и отчёт. Так же устроен прогон одиночного кадра: он тоже сводится к
// пачке из одного задания, и второй ветки кода на это не нужно.
//
// ОШИБКА ОДНОГО ЗАДАНИЯ НЕ РОНЯЕТ ПАЧКУ. Из двадцати кадров один может не выйти
// (модель не нашлась, файл битый) — остальные девятнадцать человек всё равно
// ждёт. Задание помечается неудачей вместе с причиной, и работа идёт дальше;
// прерывает пачку только явная остановка.

/**
 * Как выдавать сид каждому заданию.
 *
 * `fixed`  — один и тот же: сравнить модели или промпты между собой.
 * `series` — по порядку от базового: воспроизводимо и при этом разнообразно.
 * `random` — свой у каждого: набрать вариантов.
 */
export const SEED_MODES = ["fixed", "series", "random"];

/**
 * Сид для задания под номером `index`.
 *
 * ⚠️ Случайный берётся ЗДЕСЬ и сразу попадает в отчёт задания: иначе повторить
 * понравившийся кадр невозможно — а именно за этим в пачку и идут.
 *
 * @param {"fixed"|"series"|"random"} mode
 * @param {number} base базовый сид из деки
 * @param {number} index номер задания, с нуля
 * @param {() => number} [rng] источник случайности (для тестов)
 * @returns {number}
 */
export function seedFor(mode, base, index, rng = Math.random) {
    const start = Number.isFinite(Number(base)) ? Math.floor(Number(base)) : 0;
    if (mode === "series") return start + index;
    if (mode === "random") return Math.floor(rng() * 0xffffffff);
    return start;
}

/**
 * Задание пачки.
 *
 * @typedef {object} BatchTask
 * @property {string} id       Своё имя задания: по нему отчёт связывается со списком.
 * @property {object} input    Что дано: {prompt} или {image} — смысл знает работа.
 * @property {object} [result] Что вышло; заполняет модуль.
 * @property {string} [error]  Почему не вышло.
 * @property {"waiting"|"running"|"done"|"failed"|"skipped"} status
 */

/**
 * Очередь пачки.
 *
 * @param {object} options
 * @param {Array<{id: string, title?: string, run: Function}>} options.phases
 *   Фазы по порядку. `run(task, context)` делает работу над ОДНИМ заданием и
 *   возвращает то, что положить в `task.result`; бросает — задание помечается
 *   неудачей. Фаза целиком проходит по всем заданиям, и только потом начинается
 *   следующая: ради этого модуль и написан.
 * @param {(state: object) => void} [options.onProgress] Ход изменился.
 * @param {(phaseId: string, state: object) => Promise<void>} [options.onPhaseDone]
 *   Фаза прошла по всем заданиям. Здесь раздел складывает промежуточный итог:
 *   список готовых промптов уходит во временный файл ДО того, как начнётся
 *   рисование, — час работы модели не должен зависеть от того, переживёт ли
 *   вкладка браузера следующие двадцать минут.
 * @returns {object} контракт пачки
 */
export function createBatch({ phases, onProgress, onPhaseDone } = {}) {
    if (!Array.isArray(phases) || !phases.length) {
        throw new Error("[TS Studio] a batch needs at least one phase");
    }

    /** @type {BatchTask[]} */
    let tasks = [];
    let phaseAt = -1;
    let stopped = false;
    let running = false;

    const state = () => ({
        phase: phaseAt >= 0 ? phases[phaseAt]?.id || "" : "",
        phaseIndex: phaseAt,
        phaseTotal: phases.length,
        total: tasks.length,
        done: tasks.filter((t) => t.status === "done").length,
        failed: tasks.filter((t) => t.status === "failed").length,
        running,
        stopped,
        tasks: tasks.map((t) => ({ ...t })),
    });

    const announce = () => { onProgress?.(state()); };

    return {
        state,

        /**
         * Что прогонять. Каждое задание — одна строка списка на экране.
         *
         * @param {Array<object>} inputs что дано каждому заданию
         */
        load(inputs) {
            if (running) throw new Error("[TS Studio] the batch is already running");
            tasks = (Array.isArray(inputs) ? inputs : []).map((input, index) => ({
                id: `task_${index + 1}`,
                input: { ...input },
                status: "waiting",
            }));
            phaseAt = -1;
            stopped = false;
            announce();
            return state();
        },

        /**
         * Пройти все фазы по всем заданиям.
         *
         * @param {object} [context] Что передать работе: модель, сид, роуты.
         * @returns {Promise<object>} итоговое состояние
         */
        async run(context = {}) {
            if (running) throw new Error("[TS Studio] the batch is already running");
            running = true;
            stopped = false;
            try {
                for (let at = 0; at < phases.length; at += 1) {
                    if (stopped) break;
                    phaseAt = at;
                    announce();
                    for (const task of tasks) {
                        if (stopped) break;
                        // Задание, не пережившее прошлую фазу, дальше не идёт:
                        // рисовать по несуществующему промпту нечего.
                        if (task.status === "failed" || task.status === "skipped") continue;
                        task.status = "running";
                        announce();
                        try {
                            const outcome = await phases[at].run(task, {
                                ...context,
                                phase: phases[at].id,
                                index: tasks.indexOf(task),
                            });
                            task.result = { ...(task.result || {}), ...(outcome || {}) };
                            task.status = "done";
                        } catch (err) {
                            task.status = "failed";
                            task.error = String(err?.message || err);
                        }
                        announce();
                    }
                    // Итог фазы — наружу. Ошибка здесь не должна ронять пачку:
                    // не записался промежуточный файл — жаль, но кадры важнее.
                    try {
                        await onPhaseDone?.(phases[at].id, state());
                    } catch (err) {
                        console.warn("[TS Studio] batch phase hook failed", err);
                    }
                    // Между фазами задания снова ждут: следующая фаза их поднимет.
                    if (at < phases.length - 1) {
                        for (const task of tasks) {
                            if (task.status === "done") task.status = "waiting";
                        }
                    }
                }
            } finally {
                running = false;
                phaseAt = -1;
                announce();
            }
            return state();
        },

        /**
         * Остановить пачку.
         *
         * Текущее задание доигрывает до конца — обрывать прогон на половине
         * значит оставить в очереди ComfyUI кадр, за который никто не отвечает.
         * Всё, что не начиналось, помечается пропущенным.
         */
        stop() {
            stopped = true;
            for (const task of tasks) {
                if (task.status === "waiting") task.status = "skipped";
            }
            announce();
            return state();
        },
    };
}
