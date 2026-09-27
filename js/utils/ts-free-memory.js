// Кнопка «Освободить память» в верхней панели ComfyUI.
//
// Зачем, словами владельца пака: ComfyUI не хочется закрывать, а параллельно
// нужно открыть видеоредактор — и ComfyUI держит видеопамять и ОЗУ. Кнопка
// выгружает модели и отпускает память, не останавливая сервер.
//
// Чем она больше встроенной команды «Unload Models and Execution Cache»:
// 1. её видно — встроенная живёт в палитре команд, о ней мало кто знает;
// 2. она выгружает и модели самого пака, которые ComfyUI не учитывает вовсе
//    (Gemma на WebGPU, кэши Qwen, Whisper и др.) — это делает маршрут
//    `/ts_memory/free` (nodes/ts_memory_routes.py);
// 3. она говорит, сколько освободилось, а если идёт прогон — что память
//    отпустят после него, а не делает вид, что всё уже свободно.
//
// Выключается тумблером в настройках (раздел «TS Timesaver»). Кнопка при этом
// не снимается с панели, а прячется классом: список кнопок панели ComfyUI
// собирает один раз при регистрации расширения, а класс срабатывает мгновенно
// и без перезагрузки страницы.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import { ensureThemeStyles, pickLocaleStrings } from "../_theme.js";
import { describeResult } from "./_free_memory.js";

const SETTING_ID = "TS.FreeMemoryButton";
const BUTTON_CLASS = "ts-free-memory-button";
/** Класс на <html>: пока он стоит, кнопки нет. */
const HIDDEN_CLASS = "ts-free-memory-off";
const STYLE_ID = "ts-free-memory-styles";

const STRINGS = {
    en: {
        label: "Free memory",
        tooltip: "Unload every model and clear cached results to hand video memory "
            + "and RAM back to other apps. The next run loads models again.",
        busy: "Freeing memory…",
        done: (list) => `Freed: ${list}.`,
        vram: (size) => `${size} of video memory`,
        ram: (size) => `${size} of RAM`,
        doneShort: "Models unloaded, memory freed.",
        nothing: "Nothing was held — there was no memory to free.",
        busyNote: "A model busy with a generation was left loaded — press again when it finishes.",
        deferred: "A run is in progress. Memory will be freed as soon as it finishes.",
        failed: "Could not free memory — see the console.",
        summary: "TS Timesaver",
    },
    ru: {
        label: "Освободить память",
        tooltip: "Выгрузить все модели и очистить кэш результатов, чтобы отдать "
            + "видеопамять и ОЗУ другим программам. Следующий запуск загрузит модели заново.",
        busy: "Освобождаю память…",
        done: (list) => `Освобождено: ${list}.`,
        vram: (size) => `видеопамять ${size}`,
        ram: (size) => `ОЗУ ${size}`,
        doneShort: "Модели выгружены, память освобождена.",
        nothing: "Освобождать было нечего — память не занята.",
        busyNote: "Модель, занятая генерацией, осталась загруженной — нажмите ещё раз, когда та закончится.",
        deferred: "Идёт прогон. Память освободится, как только он закончится.",
        failed: "Не удалось освободить память — подробности в консоли.",
        summary: "TS Timesaver",
    },
};

// ⚠️ Словарь берётся при загрузке модуля, а не при нажатии: подпись кнопки
// нужна уже при регистрации расширения. Смена языка ComfyUI перезагружает
// страницу, так что устареть он не успевает.
const t = pickLocaleStrings(STRINGS);

function ensureStyles() {
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    // Только видимость: вид кнопки — забота самой панели ComfyUI, чтобы наша
    // стояла в ряду как родная.
    style.textContent = `.${HIDDEN_CLASS} .${BUTTON_CLASS}{display:none !important}`;
    document.head.appendChild(style);
}

/** Включена ли кнопка. Умолчание — включена. */
function buttonEnabled() {
    try {
        const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
        if (value !== undefined && value !== null) return Boolean(value);
    } catch { /* старый фронтенд — ниже запасной путь */ }
    try {
        const value = app?.ui?.settings?.getSettingValue?.(SETTING_ID, true);
        return value === undefined || value === null ? true : Boolean(value);
    } catch {
        return true;
    }
}

function applyVisibility(enabled) {
    ensureStyles();
    document.documentElement.classList.toggle(HIDDEN_CLASS, !enabled);
}

function toast(severity, detail, life = 5000) {
    try {
        app.extensionManager?.toast?.add({ severity, summary: t.summary, detail, life });
    } catch {
        /* уведомление — любезность; сбой в нём не должен ломать очистку */
    }
}

let running = false;

async function freeMemory() {
    // Двойной щелчок не должен выгружать дважды: второй запрос ничего не
    // освободит, а второе уведомление запутает.
    if (running) return;
    running = true;
    toast("info", t.busy, 2500);
    try {
        const response = await api.fetchApi("/ts_memory/free", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const answer = await response.json();
        const locale = document.documentElement.lang || navigator.language;
        const { severity, text } = describeResult(answer, t, locale);
        toast(severity, text);
    } catch (error) {
        console.error("[TS FreeMemory] request failed", error);
        toast("error", t.failed);
    } finally {
        running = false;
    }
}

app.registerExtension({
    name: "ts.freeMemory",
    settings: [
        {
            id: SETTING_ID,
            category: ["TS Timesaver", "Interface", "Free memory button"],
            name: "«Free memory» button in the top bar",
            tooltip: "Unloads every model — ComfyUI's and this pack's own — and "
                + "clears cached results, so another app can use the video "
                + "memory and RAM without closing ComfyUI.",
            type: "boolean",
            defaultValue: true,
            onChange: (value) => applyVisibility(Boolean(value)),
        },
    ],
    commands: [
        {
            id: "TS.FreeMemory",
            label: t.label,
            icon: "pi pi-eraser",
            function: freeMemory,
        },
    ],
    actionBarButtons: [
        {
            icon: "pi pi-eraser",
            label: t.label,
            tooltip: t.tooltip,
            class: BUTTON_CLASS,
            onClick: freeMemory,
        },
    ],
    async setup() {
        applyVisibility(buttonEnabled());
    },
});
