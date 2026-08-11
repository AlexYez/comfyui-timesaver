// TS Video Saver: плеер результата прямо в ноде.
//
// Формат и качество рисует сам ComfyUI (DynamicCombo) — здесь только
// воспроизведение и строка о сохранённом файле.
//
// ⚠️ ЗВУК ЗАПОМИНАЕТСЯ. Но включённый звук и автозапуск несовместимы: браузер
// автозапуск со звуком запрещает. Поэтому при восстановленном звуке ролик
// остаётся на паузе и об этом честно написано, а не делается вид, что играет.

import { api } from "/scripts/api.js";

import { TS_UI_CLASS, pickLocaleStrings } from "../../_theme.js";
import { addResizableDomWidget, getWidget } from "../../_dom_widget.js";
import { openFullscreenOverlay } from "../../_fullscreen.js";
import { createPlayback } from "../../_media/_playback.js";
import { icon, setIcon } from "../../_media/_icons.js";
import { formatBytes, formatDuration } from "../../_media/_ruler.js";
import { ensureVideoStyles } from "../loader/_video_styles.js";

export const NODE_TYPE = "TS_VideoSaver";
export const DOM_WIDGET = "ts_video_saver";
export const UI_KEY = "ts_video_saver";

const PROP_PAYLOAD = "ts_video_saver_payload";
const PROP_MUTED = "ts_video_saver_muted";
const PROP_VOLUME = "ts_video_saver_volume";
const PROP_LOOP = "ts_video_saver_loop";
// Привычка для НОВЫХ нод. Уже сохранённая всегда берёт своё из properties,
// иначе workflow перестал бы воспроизводиться одинаково.
const PREF_KEY = "ts.video.saver.audio";

export const STRINGS_SAVER = {
    en: {
        empty: "Queue the graph once\nto see the result here.",
        play: "Play / pause",
        mute: "Sound on/off",
        volume: "Volume",
        seek: "Drag to move through the clip",
        time: "Position / length",
        loop: "Loop",
        fullscreen: "Full screen",
        close: "Close (Esc)",
        pressPlay: "Sound is on — press play.",
        autoplayBlocked: "The browser blocked autoplay. Press play.",
        proxy: "preview copy",
        download: "Download",
        downloadHint: "Save the file to your computer",
        copyPath: "Copy path",
        copyPathHint: "Copy the full path of the saved file to the clipboard",
        copied: "Copied",
    },
    ru: {
        empty: "Запустите очередь один раз,\nчтобы увидеть результат.",
        play: "Воспроизведение / пауза",
        mute: "Звук вкл/выкл",
        volume: "Громкость",
        seek: "Тяните, чтобы перемещаться по ролику",
        time: "Позиция / длина",
        loop: "Повтор",
        fullscreen: "На весь экран",
        close: "Закрыть (Esc)",
        pressPlay: "Звук включён — нажмите воспроизведение.",
        autoplayBlocked: "Браузер заблокировал автозапуск. Нажмите воспроизведение.",
        proxy: "копия для просмотра",
        download: "Скачать",
        downloadHint: "Сохранить файл к себе на компьютер",
        copyPath: "Копировать путь",
        copyPathHint: "Скопировать полный путь сохранённого файла в буфер",
        copied: "Скопировано",
    },
};

function readPreference() {
    try {
        const raw = localStorage.getItem(PREF_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch {
        return null;
    }
}

function writePreference(value) {
    try {
        localStorage.setItem(PREF_KEY, JSON.stringify(value));
    } catch {
        // Приватный режим браузера — привычка просто не сохранится.
    }
}

function iconButton(iconName, title, onClick) {
    const element = document.createElement("button");
    element.type = "button";
    element.className = "ts-ui-btn ts-ui-btn--icon";
    element.innerHTML = icon(iconName);
    element.title = title;
    element.addEventListener("click", onClick);
    return element;
}

export function setupVideoSaver(node) {
    ensureVideoStyles();
    const L = pickLocaleStrings(STRINGS_SAVER);

    const root = document.createElement("div");
    root.className = `${TS_UI_CLASS} ts-vid`;

    const stage = document.createElement("div");
    stage.className = "ts-vid__stage";
    const video = document.createElement("video");
    video.className = "ts-vid__video";
    video.playsInline = true;
    video.preload = "metadata";
    const empty = document.createElement("div");
    empty.className = "ts-vid__empty";
    empty.textContent = L.empty;
    stage.append(video, empty);

    const transport = document.createElement("div");
    transport.className = "ts-vid__transport";

    const playback = createPlayback(video, {
        getRange: () => ({ left: 0, right: video.duration || 0 }),
        isLooping: () => state.loop,
        getFps: () => state.payload?.fps || 0,
        onTime: () => updateTime(),
    });

    const playButton = iconButton("play", L.play, () => playback.toggle());
    const muteButton = iconButton("muted", L.mute, () => {
        state.muted = !state.muted;
        video.muted = state.muted;
        node.properties[PROP_MUTED] = state.muted;
        writePreference({ muted: state.muted, volume: state.volume });
        setIcon(muteButton, state.muted ? "muted" : "sound");
    });
    const loopButton = iconButton("loop", L.loop, () => {
        state.loop = !state.loop;
        node.properties[PROP_LOOP] = state.loop;
        loopButton.classList.toggle("is-active", state.loop);
    });

    const volume = document.createElement("input");
    volume.type = "range";
    volume.className = "ts-ui-slider ts-vid__volume";
    volume.min = "0";
    volume.max = "1";
    volume.step = "0.01";
    volume.title = L.volume;
    volume.setAttribute("aria-label", L.volume);
    volume.addEventListener("input", () => {
        state.volume = Number(volume.value);
        video.volume = state.volume;
        node.properties[PROP_VOLUME] = state.volume;
        writePreference({ muted: state.muted, volume: state.volume });
    });

    // ⚠️ Полоса перемотки живёт ОТДЕЛЬНОЙ строкой под кадром, а не рядом с
    // громкостью. Два одинаковых ползунка в одном ряду читались как поломка:
    // непонятно, который из них что делает, и оба выглядели точками.
    const seekRow = document.createElement("div");
    seekRow.className = "ts-vid__seekrow";
    const seek = document.createElement("input");
    seek.type = "range";
    seek.className = "ts-ui-slider ts-vid__seek";
    seek.min = "0";
    seek.max = "1000";
    seek.value = "0";
    seek.title = L.seek;
    seek.setAttribute("aria-label", L.seek);
    let scrubbing = false;
    // ⚠️ Пока тянут ползунок, обновлять его положение по timeupdate нельзя —
    // он будет вырываться из-под пальца.
    seek.addEventListener("pointerdown", () => { scrubbing = true; });
    seek.addEventListener("input", () => {
        if (video.duration) playback.seek((Number(seek.value) / 1000) * video.duration);
    });
    seek.addEventListener("change", () => { scrubbing = false; });

    const timeLabel = document.createElement("span");
    timeLabel.className = "ts-vid__time";
    timeLabel.title = L.time;

    const fullscreenButton = iconButton("fullscreen", L.fullscreen, () => toggleFullscreen());

    seekRow.append(seek, timeLabel);
    transport.append(playButton, muteButton, volume, loopButton, fullscreenButton);

    const result = document.createElement("div");
    result.className = "ts-vid__result";
    const resultText = document.createElement("span");
    const downloadLink = document.createElement("a");
    downloadLink.className = "ts-ui-btn ts-ui-btn--ghost";
    downloadLink.textContent = L.download;
    downloadLink.title = L.downloadHint;
    downloadLink.target = "_blank";
    downloadLink.rel = "noopener";
    downloadLink.style.display = "none";
    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.className = "ts-ui-btn ts-ui-btn--ghost";
    copyButton.textContent = L.copyPath;
    copyButton.title = L.copyPathHint;
    copyButton.style.display = "none";
    copyButton.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(state.payload?.saved_path || "");
            copyButton.textContent = L.copied;
            setTimeout(() => { copyButton.textContent = L.copyPath; }, 1500);
        } catch (error) {
            console.warn("[TS Video Saver] clipboard unavailable", error);
        }
    });
    result.append(resultText, downloadLink, copyButton);

    const status = document.createElement("div");
    status.className = "ts-vid__status ts-ui-statusbar";
    const statusLeft = document.createElement("span");
    status.append(statusLeft);

    root.append(stage, seekRow, transport, result, status);

    const preference = readPreference();
    const state = {
        payload: null,
        muted: preference?.muted !== false,
        volume: Number.isFinite(preference?.volume) ? preference.volume : 1,
        loop: true,
    };

    function updateTime() {
        const current = video.currentTime || 0;
        const total = video.duration || 0;
        timeLabel.textContent = `${formatDuration(current)} / ${formatDuration(total)}`;
        if (!scrubbing && total > 0) seek.value = String(Math.round((current / total) * 1000));
    }

    function applyPayload(payload) {
        if (!payload) return;
        state.payload = payload;
        node.properties ||= {};
        node.properties[PROP_PAYLOAD] = payload;

        const params = new URLSearchParams({
            filename: payload.filename || "",
            subfolder: payload.subfolder || "",
            type: payload.type || "output",
            format: payload.format || "video/mp4",
        });
        video.muted = state.muted;
        video.volume = state.volume;
        video.src = api.apiURL(`/view?${params}`);
        video.load();
        empty.style.display = "none";

        // ⚠️ Строка результата и кнопка скачивания говорят о НАСТОЯЩЕМ файле, а
        // не о копии для просмотра: скачать превью вместо ProRes — худшее, что
        // здесь может произойти.
        const savedName = payload.saved_filename || payload.filename || "";
        const savedParams = new URLSearchParams({
            filename: savedName,
            subfolder: payload.saved_subfolder ?? payload.subfolder ?? "",
            type: payload.saved_type || "output",
        });

        const bits = [];
        if (payload.width) bits.push(`${payload.width}×${payload.height}`);
        if (payload.fps) bits.push(`${Number(payload.fps).toFixed(2).replace(/\.00$/, "")} fps`);
        if (payload.frames) bits.push(`${payload.frames}`);
        if (payload.size_bytes) bits.push(formatBytes(payload.size_bytes));
        if (payload.format_key) bits.push(payload.format_key);
        resultText.innerHTML = "";
        const strong = document.createElement("b");
        strong.textContent = savedName;
        resultText.append(strong, document.createTextNode(bits.length ? ` · ${bits.join(" · ")}` : ""));

        downloadLink.href = api.apiURL(`/view?${savedParams}`);
        downloadLink.download = savedName || "video";
        downloadLink.style.display = "";
        copyButton.style.display = payload.saved_path ? "" : "none";

        statusLeft.textContent = payload.is_proxy ? L.proxy : "";
    }

    video.addEventListener("loadedmetadata", () => {
        video.muted = state.muted;
        video.volume = state.volume;
        updateTime();
        if (state.muted) {
            video.play().catch(() => { statusLeft.textContent = L.autoplayBlocked; });
        } else {
            // Звук восстановлен — автозапуск браузер всё равно не даст.
            statusLeft.textContent = L.pressPlay;
        }
    });
    video.addEventListener("play", () => setIcon(playButton, "pause"));
    video.addEventListener("pause", () => setIcon(playButton, "play"));

    let fullscreen = null;
    let host = null;
    function toggleFullscreen() {
        if (fullscreen?.isOpen()) { fullscreen.close(); return; }
        host = root.parentElement;
        fullscreen = openFullscreenOverlay(root, {
            label: node.title || NODE_TYPE,
            closeTitle: L.close,
            onClose: () => { host?.appendChild(root); fullscreen = null; },
        });
    }

    // ⚠️ chromeHeight — заголовок, три входа, два выхода и шесть виджетов над
    // нашим плеером. Без него виджет просит всю высоту ноды и вылезает за край.
    // Размер задаём сами: LiteGraph уже выдал ноде свой, и defaultWidth хелпера
    // до неё не доходит.
    node.size = [Math.max(node.size?.[0] || 0, 420), Math.max(node.size?.[1] || 0, 560)];

    addResizableDomWidget(node, root, {
        name: DOM_WIDGET,
        minWidth: 380,
        minHeight: 480,
        defaultWidth: 460,
        defaultHeight: 560,
        chromeHeight: 240,
        minWidgetHeight: 230,
    });

    node._tsVideoSaverApply = applyPayload;
    node._tsVideoSaverRehydrate = () => {
        node.properties ||= {};
        if (typeof node.properties[PROP_MUTED] === "boolean") {
            state.muted = node.properties[PROP_MUTED];
        }
        if (Number.isFinite(node.properties[PROP_VOLUME])) {
            state.volume = node.properties[PROP_VOLUME];
        }
        if (typeof node.properties[PROP_LOOP] === "boolean") {
            state.loop = node.properties[PROP_LOOP];
        }
        setIcon(muteButton, state.muted ? "muted" : "sound");
        loopButton.classList.toggle("is-active", state.loop);
        volume.value = String(state.volume);
        video.muted = state.muted;
        video.volume = state.volume;
        const saved = node.properties[PROP_PAYLOAD];
        if (saved) applyPayload(saved);
    };

    const previousRemoved = node.onRemoved;
    node.onRemoved = function tsVideoSaverRemoved(...args) {
        fullscreen?.close();
        playback.dispose();
        video.pause();
        video.removeAttribute("src");
        video.load();
        return previousRemoved?.apply(this, args);
    };

    node._tsVideoSaverRehydrate();
}

export { getWidget };
