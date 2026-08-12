// TS Video Loader: связь редактора с нодой.
//
// Здесь виджеты, свойства, восстановление после перезагрузки workflow, загрузка
// файла и полноэкранный режим. Рисование и жесты — в `_video_editor.js`.
//
// ⚠️ ХРАНЕНИЕ. Подрезка живёт в скрытых виджетах (их прячет hideWidget: он же
// зеркалит значение в node.properties и доносит его до промпта). Состояние
// интерфейса — зум, позиция окна, цикл, звук — только в properties: в промпт
// ему незачем, а в сохранённом workflow пригодится.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

import { TS_UI_CLASS, pickLocaleStrings } from "../../_theme.js";
import { addResizableDomWidget, getWidget, hideWidget } from "../../_dom_widget.js";
import { openFullscreenOverlay } from "../../_fullscreen.js";
import { hotkeysAllowed, isTypingTarget } from "../../_keys.js";
import { makeDropZone } from "../../_studio/_dnd.js";
import { icon, setIcon } from "../../_media/_icons.js";
import { formatTimecode } from "../../_media/_ruler.js";
import { createVideoEditor } from "./_video_editor.js";
import { ensureVideoStyles } from "./_video_styles.js";

export const NODE_TYPE = "TS_VideoLoader";
export const DOM_WIDGET = "ts_video_loader";
const ROUTE = "/ts_video";

const INPUT_PATH = "source_path";
const INPUT_START = "start_seconds";
const INPUT_END = "end_seconds";

const PROP_ZOOM = "ts_video_zoom";
const PROP_VIEW = "ts_video_view_start";
const PROP_LOOP = "ts_video_loop";
const PROP_MUTED = "ts_video_muted";
const PROP_WAVE = "ts_video_wave";
const PROP_META = "ts_video_meta";

const VIDEO_ACCEPT = "video/*,.mp4,.mov,.mkv,.webm,.avi,.m4v,.mpg,.mpeg,.m2ts,.mts,.ts";

export const STRINGS_LOADER = {
    en: {
        load: "Load video",
        loadHint: "Pick a video file — it is uploaded into the ComfyUI input folder",
        pathPlaceholder: "…or a path on the ComfyUI machine",
        dropHint: "Drop a video here, paste it,\nor press \"Load video\".",
        noFile: "No file selected",
        uploading: (pct) => `Uploading… ${pct}%`,
        uploadFailed: "Could not upload the file.",
        analyzing: "Reading the video…",
        metadataFailed: "Could not read the video.",
        codecUnavailable: "The browser cannot play this codec — trimming still works.",
        play: "Play / pause (Space)",
        loop: "Loop the selection (L)",
        mute: "Sound on/off (M)",
        reset: "Reset the trim (X)",
        zoomOut: "Zoom out",
        zoomIn: "Zoom in",
        fit: "Fit the whole clip (F)",
        zoomSelection: "Zoom to the selection (Z)",
        wave: "Show the sound track",
        fullscreen: "Open the trimmer full screen",
        fullscreenLabel: "Video trimming",
        close: "Close (Esc)",
        trimWhole: "Trim: whole clip",
        trim: (from, to, length) => `Trim: ${from} → ${to} · ${length}`,
        stats: (w, h, fps) => `${w}×${h} · ${fps} fps`,
        noAudio: "no audio",
        frames: (n) => `${n} frames out`,
        framesCapped: (n, full) => `${n} of ${full} frames out (capped)`,
        stageHint: "Drop a video here, or paste one from the clipboard",
        pathHint: "Path to a file on the ComfyUI machine — the footage does not have to be copied into the input folder",
        waveOff: "This file has no audio track",
    },
    ru: {
        load: "Загрузить видео",
        loadHint: "Выберите видеофайл — он загрузится в папку input ComfyUI",
        pathPlaceholder: "…или путь на машине с ComfyUI",
        dropHint: "Бросьте сюда видео, вставьте из буфера\nили нажмите «Загрузить видео».",
        noFile: "Файл не выбран",
        uploading: (pct) => `Загрузка… ${pct}%`,
        uploadFailed: "Не удалось загрузить файл.",
        analyzing: "Читаем видео…",
        metadataFailed: "Не удалось прочитать видео.",
        codecUnavailable: "Браузер не проигрывает этот кодек — подрезка всё равно работает.",
        play: "Воспроизведение / пауза (Space)",
        loop: "Зациклить выделение (L)",
        mute: "Звук вкл/выкл (M)",
        reset: "Сбросить подрезку (X)",
        zoomOut: "Отдалить",
        zoomIn: "Приблизить",
        fit: "Показать весь ролик (F)",
        zoomSelection: "Приблизить к выделению (Z)",
        wave: "Показывать звуковую дорожку",
        fullscreen: "Открыть подрезку на весь экран",
        fullscreenLabel: "Подрезка видео",
        close: "Закрыть (Esc)",
        trimWhole: "Обрезка: весь ролик",
        trim: (from, to, length) => `Обрезка: ${from} → ${to} · ${length}`,
        stats: (w, h, fps) => `${w}×${h} · ${fps} кадр/с`,
        noAudio: "без звука",
        frames: (n) => `${n} кадров на выходе`,
        framesCapped: (n, full) => `${n} из ${full} кадров на выходе (предел)`,
        stageHint: "Бросьте сюда видео или вставьте из буфера",
        pathHint: "Путь к файлу на машине с ComfyUI — съёмку не обязательно копировать в папку input",
        waveOff: "В этом файле нет звуковой дорожки",
    },
};

// ── мелкая утварь ────────────────────────────────────────────────────────── #

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (widget) {
        widget.value = value;
        if (typeof widget.callback === "function") widget.callback(value);
    }
    node.properties ||= {};
    node.properties[name] = value;
}

function readPersisted(node, name, fallback) {
    const widget = getWidget(node, name)?.value;
    if (widget !== undefined && widget !== null && widget !== "") return widget;
    const stored = node?.properties?.[name];
    if (stored !== undefined && stored !== null && stored !== "") return stored;
    return fallback;
}

function readNumber(node, name, fallback) {
    const value = Number(readPersisted(node, name, fallback));
    return Number.isFinite(value) ? value : fallback;
}

function button(iconName, title, onClick, extra = "") {
    const element = document.createElement("button");
    element.type = "button";
    element.className = `ts-ui-btn ts-ui-btn--icon ${extra}`.trim();
    element.innerHTML = icon(iconName);
    element.title = title;
    element.addEventListener("click", onClick);
    return element;
}

export function setupVideoLoader(node) {
    ensureVideoStyles();
    const L = pickLocaleStrings(STRINGS_LOADER);

    const editor = createVideoEditor({
        api,
        route: ROUTE,
        strings: L,
        onRangeChange: (start, end) => {
            setWidgetValue(node, INPUT_START, Number(start.toFixed(3)));
            setWidgetValue(node, INPUT_END, end < 0 ? -1 : Number(end.toFixed(3)));
            updateStatus();
        },
        onViewportChange: (view) => {
            node.properties ||= {};
            node.properties[PROP_ZOOM] = view.zoom;
            node.properties[PROP_VIEW] = view.viewStart;
        },
    });

    // ── верхняя полоса ───────────────────────────────────────────────────── #
    const bar = document.createElement("div");
    bar.className = "ts-vid__bar ts-ui-toolbar";

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = VIDEO_ACCEPT;
    // ⚠️ Инпут паркуется за экраном, а не схлопывается в 1×1: по схлопнутому
    // некоторые браузеры отказываются кликать программно (§12.5.11).
    fileInput.className = "ts-vid__hidden-input";

    const loadButton = document.createElement("button");
    loadButton.type = "button";
    loadButton.className = "ts-ui-btn";
    loadButton.textContent = L.load;
    loadButton.title = L.loadHint;
    loadButton.addEventListener("click", () => fileInput.click());

    const pathInput = document.createElement("input");
    pathInput.type = "text";
    pathInput.className = "ts-ui-input ts-vid__path";
    pathInput.placeholder = L.pathPlaceholder;
    pathInput.title = L.pathHint;
    pathInput.spellcheck = false;
    pathInput.addEventListener("change", () => {
        const value = pathInput.value.trim();
        if (value) applySource(value);
    });

    const nameLabel = document.createElement("span");
    nameLabel.className = "ts-vid__name";
    nameLabel.textContent = L.noFile;

    const waveToggle = button("wave", L.wave, () => {
        if (!editor.state.hasAudio) return;
        editor.state.showWave = !editor.state.showWave;
        node.properties[PROP_WAVE] = editor.state.showWave;
        waveToggle.classList.toggle("is-active", editor.state.showWave);
        editor.scheduleDraw(editor.NEED_ALL);
    });

    /** Кнопка дорожки честно гаснет, когда включать нечего. */
    function syncWaveToggle() {
        const has = Boolean(editor.state.hasAudio);
        waveToggle.disabled = !has;
        waveToggle.title = has ? L.wave : L.waveOff;
        waveToggle.classList.toggle("is-active", has && editor.state.showWave);
    }

    const zoomOut = button("zoomOut", L.zoomOut,
        () => editor.viewport.setZoom(editor.viewport.zoom / 1.6, playheadSeconds()));
    const zoomIn = button("zoomIn", L.zoomIn,
        () => editor.viewport.setZoom(editor.viewport.zoom * 1.6, playheadSeconds()));
    const fitButton = button("fit", L.fit, () => editor.viewport.fit());
    const zoomSelection = button("zoomSelection", L.zoomSelection, () => {
        const { left, right } = editor.bounds();
        editor.viewport.zoomToRange(left, right);
    });

    // Обычная иконка, а не createOpenInterfaceButton: тот несёт подпись
    // «Открыть интерфейс» и занимает четверть полосы. Здесь интерфейс и так
    // открыт — на весь экран его лишь разворачивают.
    const fullscreenButton = button("fullscreen", L.fullscreen, () => toggleFullscreen());

    // Наверху — только то, что относится к ФАЙЛУ, и полный экран справа.
    // Всё, что управляет таймлайном, стоит у самого таймлайна: тянуться за
    // кнопкой зума через всю ноду наверх — лишнее движение.
    const barSpacer = document.createElement("div");
    barSpacer.className = "ts-vid__spacer";
    bar.append(loadButton, pathInput, nameLabel, barSpacer, fullscreenButton, fileInput);

    // ── транспорт ────────────────────────────────────────────────────────── #
    const transport = document.createElement("div");
    transport.className = "ts-vid__transport";

    const playButton = button("play", L.play, () => editor.playback.toggle());
    const loopButton = button("loop", L.loop, () => {
        editor.state.looping = !editor.state.looping;
        node.properties[PROP_LOOP] = editor.state.looping;
        loopButton.classList.toggle("is-active", editor.state.looping);
    });
    const muteButton = button("muted", L.mute, () => {
        editor.video.muted = !editor.video.muted;
        node.properties[PROP_MUTED] = editor.video.muted;
        setIcon(muteButton, editor.video.muted ? "muted" : "sound");
    });
    const resetButton = button("resetTrim", L.reset, () => editor.setRange(0, -1));

    const timeLabel = document.createElement("span");
    timeLabel.className = "ts-vid__time";

    // Ряд стоит ПРЯМО НАД ТАЙМЛАЙНОМ, поэтому здесь же и его управление.
    // Покадровые кнопки и поля таймкода убраны намеренно: шаг кадром остался на
    // стрелках, а начало и конец куска и так написаны в строке состояния —
    // держать их в трёх местах значит захламлять ноду.
    const transportSpacer = document.createElement("div");
    transportSpacer.className = "ts-vid__spacer";

    transport.append(playButton, loopButton, muteButton, resetButton, timeLabel,
                     transportSpacer, waveToggle, zoomOut, zoomIn, fitButton,
                     zoomSelection);

    // ── статус ───────────────────────────────────────────────────────────── #
    const status = document.createElement("div");
    status.className = "ts-vid__status ts-ui-statusbar";
    const statusLeft = document.createElement("span");
    const statusRight = document.createElement("span");
    status.append(statusLeft, statusRight);

    editor.mount(bar, transport, status);
    editor.element.classList.add(TS_UI_CLASS);

    // ── состояние и обновления ───────────────────────────────────────────── #
    function playheadSeconds() {
        return Number(editor.video.currentTime) || editor.state.cropStart || 0;
    }

    /**
     * Сколько кадров реально уйдёт на выход.
     *
     * Повторяет расчёт бэкенда ровно: сперва выбранный кусок, затем частота,
     * затем прореживание шагом, и только в конце — предел. Человек должен
     * видеть это число ДО прогона, а не узнавать его из результата.
     */
    function outputFrames() {
        const { left, right } = editor.bounds();
        const window = Math.max(0, right - left);
        if (!(window > 0)) return null;

        const asked = Number(getWidget(node, "frame_rate")?.value) || 0;
        const rate = asked > 0 ? asked : editor.state.fps;
        if (!(rate > 0)) return null;

        const step = Math.max(1, Number(getWidget(node, "frame_step")?.value) || 1);
        const cap = Math.max(0, Number(getWidget(node, "max_frames")?.value) || 0);

        const full = Math.ceil(Math.round(window * rate) / step);
        const final = cap > 0 ? Math.min(full, cap) : full;
        return { full, final, capped: cap > 0 && cap < full, rate };
    }

    function updateStatus(message = "", isError = false) {
        const { left, right } = editor.bounds();
        const fps = editor.state.fps;
        const whole = editor.state.cropEnd < 0 && left <= 0.001;
        statusLeft.textContent = message || (whole
            ? L.trimWhole
            : L.trim(formatTimecode(left, { fps }), formatTimecode(right, { fps }),
                     formatTimecode(Math.max(0, right - left), { fps })));
        status.classList.toggle("is-error", Boolean(isError));

        const parts = [];
        const frames = outputFrames();
        if (frames) {
            // Предел кадров именно ОБРЕЗАЕТ хвост выбранного куска — и когда он
            // это делает, молчать нельзя: иначе непонятно, почему кадров меньше.
            parts.push(frames.capped
                ? L.framesCapped(frames.final, frames.full)
                : L.frames(frames.final));
        }
        if (editor.state.width) {
            parts.push(L.stats(editor.state.width, editor.state.height,
                               fps ? fps.toFixed(fps % 1 ? 2 : 0) : "?"));
        }
        if (editor.state.duration && !editor.state.hasAudio) parts.push(L.noAudio);
        statusRight.textContent = parts.join(" · ");

        timeLabel.textContent =
            `${formatTimecode(editor.video.currentTime || 0, { fps })} / `
            + `${formatTimecode(editor.state.duration, { fps })}`;
        editor.badge.textContent = editor.state.duration
            ? formatTimecode(editor.video.currentTime || 0, { fps }) : "";
        editor.empty.style.display = editor.state.path ? "none" : "flex";
    }

    editor.video.addEventListener("timeupdate", () => updateStatus());
    editor.video.addEventListener("play", () => setIcon(playButton, "pause"));
    editor.video.addEventListener("pause", () => setIcon(playButton, "play"));
    editor.video.addEventListener("error", () => {
        if (editor.state.path) updateStatus(L.codecUnavailable, false);
    });

    // ── источник ─────────────────────────────────────────────────────────── #
    let metaToken = 0;

    async function fetchMetadata(path) {
        const token = ++metaToken;
        updateStatus(L.analyzing);
        try {
            const response = await api.fetchApi(
                `${ROUTE}/metadata?filepath=${encodeURIComponent(path)}`);
            if (!response.ok) throw new Error(String(response.status));
            const meta = await response.json();
            if (token !== metaToken) return;
            editor.applyMetadata(meta);
            node.properties ||= {};
            // В свойства кладём ТОЛЬКО скаляры: массив пиков в каждом
            // сохранённом workflow — это десятки килобайт мусора на ноду.
            node.properties[PROP_META] = {
                filename: meta.filename,
                duration: meta.duration,
                fps: meta.fps,
                width: meta.width,
                height: meta.height,
                has_audio: meta.has_audio,
                browser_playable: meta.browser_playable,
            };
            // Дублировать имя рядом с полем, где оно уже написано, незачем —
            // подпись нужна только для файла, выбранного кнопкой.
            const shown = meta.filename || path;
            nameLabel.textContent = pathInput.value.trim() === shown ? "" : shown;
            syncWaveToggle();
            editor.setSource(api.apiURL(`${ROUTE}/view?filepath=${encodeURIComponent(path)}`));
            updateStatus();
        } catch (error) {
            if (token !== metaToken) return;
            console.warn("[TS Video Loader] metadata failed", error);
            updateStatus(L.metadataFailed, true);
        }
    }

    function applySource(path) {
        editor.state.path = path;
        setWidgetValue(node, INPUT_PATH, path);
        editor.setRange(0, -1);
        editor.viewport.fit();
        fetchMetadata(path);
    }

    // ⚠️ Побеждает ПОСЛЕДНЕЕ действие человека, а не самая быстрая отправка.
    //
    // Видео — это гигабайты, и выбрать другой файл, не дождавшись первого,
    // совершенно нормально. Обе отправки доходили до конца, и `applySource`
    // срабатывал дважды: в ноде оставался тот ролик, чья загрузка кончилась
    // позже, — то есть чаще НЕ тот, который выбрали последним. Плюс отправка
    // продолжалась даже после удаления ноды.
    let activeUpload = null;

    function abortActiveUpload() {
        if (!activeUpload) return;
        const xhr = activeUpload;
        activeUpload = null;
        try {
            xhr.abort();
        } catch (error) {
            console.warn("[TS Video Loader] aborting the previous upload failed", error);
        }
    }

    function uploadFile(file) {
        abortActiveUpload();
        return new Promise((resolve, reject) => {
            // XHR, а не fetch: у fetch нет прогресса отправки, а видео — это
            // гигабайты, и молчащая полоса выглядит как зависший интерфейс.
            const xhr = new XMLHttpRequest();
            activeUpload = xhr;
            xhr.onabort = () => reject(new Error("aborted"));
            xhr.open("POST", api.apiURL("/upload/image"));
            xhr.upload.onprogress = (event) => {
                if (event.lengthComputable) {
                    updateStatus(L.uploading(Math.round((event.loaded / event.total) * 100)));
                }
            };
            xhr.onload = () => {
                if (activeUpload === xhr) activeUpload = null;
                if (xhr.status >= 400) { reject(new Error(xhr.statusText)); return; }
                try {
                    const payload = JSON.parse(xhr.responseText);
                    const folder = payload.subfolder ? `${payload.subfolder}/` : "";
                    resolve(`${folder}${payload.name} [${payload.type || "input"}]`);
                } catch (error) { reject(error); }
            };
            xhr.onerror = () => {
                if (activeUpload === xhr) activeUpload = null;
                reject(new Error("network"));
            };
            const form = new FormData();
            form.append("image", file, file.name);
            form.append("type", "input");
            xhr.send(form);
        });
    }

    async function chooseFile(file) {
        if (!file) return;
        try {
            applySource(await uploadFile(file));
        } catch (error) {
            // Прерванная своя же отправка — не сбой: человек просто выбрал
            // другой файл, и про предыдущий рассказывать нечего.
            if (String(error?.message) === "aborted") return;
            console.warn("[TS Video Loader] upload failed", error);
            updateStatus(L.uploadFailed, true);
        }
    }

    fileInput.addEventListener("change", async () => {
        const file = fileInput.files?.[0];
        fileInput.value = "";
        await chooseFile(file);
    });

    // ── перетаскивание ───────────────────────────────────────────────────── #
    // ⚠️ Через ОБЩИЙ механизм пака, а не свой обработчик. Он уже понимает и
    // файлы из системного проводника, и карточки браузера Artius, и превью
    // чужих нод, и — главное — снимает содержимое `dataTransfer` синхронно, до
    // первого await: иначе к моменту разбора оно уже пусто (§12.5.16).
    const dropHint = document.createElement("div");
    dropHint.className = "ts-ui-drop";
    dropHint.textContent = L.dropActive;
    editor.stage.appendChild(dropHint);

    async function acceptDropped(items) {
        const video = items.find((item) => item.type === "video");
        if (!video) {
            updateStatus(L.dropNotVideo, true);
            return;
        }
        try {
            updateStatus(L.uploading(0));
            const blob = await video.getBlob();
            const name = video.name || "dropped.mp4";
            await chooseFile(new File([blob], name, { type: blob.type || "video/mp4" }));
        } catch (error) {
            console.warn("[TS Video Loader] drop failed", error);
            updateStatus(L.uploadFailed, true);
        }
    }

    const teardownDrop = makeDropZone(editor.stage, { onDrop: acceptDropped, max: 1 });

    // ── горячие клавиши ──────────────────────────────────────────────────── #
    let pointerInside = false;
    editor.element.addEventListener("pointerenter", () => { pointerInside = true; });
    editor.element.addEventListener("pointerleave", () => { pointerInside = false; });

    function dispatchKey(event) {
        // Пока печатают в поле таймкода, буквы — это ввод, а не команды.
        if (!hotkeysAllowed(event) || isTypingTarget(event.target)) return false;
        if (!pointerInside && !editor.element.contains(document.activeElement)) return false;

        const { left, right } = editor.bounds();
        switch (event.key) {
            case " ": editor.playback.toggle(); break;
            case "i": case "I": case "ш": case "Ш":
                editor.setRange(playheadSeconds(), right); break;
            case "o": case "O": case "щ": case "Щ":
                editor.setRange(left, playheadSeconds()); break;
            case "l": case "L": loopButton.click(); break;
            case "m": case "M": muteButton.click(); break;
            case "f": case "F": editor.viewport.fit(); break;
            case "z": case "Z": editor.viewport.zoomToRange(left, right); break;
            case "x": case "X": editor.setRange(0, -1); break;
            case "ArrowLeft": editor.playback.stepFrames(event.shiftKey ? -10 : -1); break;
            case "ArrowRight": editor.playback.stepFrames(event.shiftKey ? 10 : 1); break;
            case "Home": editor.playback.seek(left, { immediate: true }); break;
            case "End": editor.playback.seek(right, { immediate: true }); break;
            default: return false;
        }
        return true;
    }

    const onKeyDown = (event) => {
        if (!dispatchKey(event)) return;
        event.preventDefault();
        event.stopPropagation();        // иначе Space ещё и подвинет холст графа
    };
    document.addEventListener("keydown", onKeyDown, true);

    // ── полноэкранный режим ──────────────────────────────────────────────── #
    let fullscreen = null;
    let host = null;

    function toggleFullscreen() {
        if (fullscreen?.isOpen()) { fullscreen.close(); return; }
        host = editor.element.parentElement;
        fullscreen = openFullscreenOverlay(editor.element, {
            label: L.fullscreenLabel,
            closeTitle: L.close,
            closeOnBackdrop: false,
            onKey: (event, { typing = false } = {}) => { if (!typing) dispatchKey(event); },
            onOpen: () => {
                editor.timeline.style.height = "200px";
                editor.scheduleDraw(editor.NEED_ALL);
            },
            onClose: () => {
                // Оверлей только ОТЦЕПЛЯЕТ содержимое — возвращаем его на место.
                // Тот же самый элемент, поэтому видео не перезагружается и
                // продолжает играть с той же секунды.
                editor.timeline.style.height = "";
                host?.appendChild(editor.element);
                editor.scheduleDraw(editor.NEED_ALL);
                fullscreen = null;
            },
        });
    }

    // ── монтаж ───────────────────────────────────────────────────────────── #
    // ⚠️ chromeHeight — это место, которое в классическом рендерере занимает всё
    // НАД нашим виджетом: заголовок, четыре выхода и семь обычных виджетов
    // (частота, предел кадров, размер, кратность, шаг, фильтр). Без него виджет
    // просит всю высоту ноды и вылезает за её нижний край — замерено.
    const CHROME_HEIGHT = 250;

    addResizableDomWidget(node, editor.element, {
        name: DOM_WIDGET,
        minWidth: 460,
        minHeight: 650,
        defaultWidth: 700,
        defaultHeight: 700,
        chromeHeight: CHROME_HEIGHT,
        // Хватает на все ряды даже когда полосы кнопок переносятся по две
        // строки — то есть при самой узкой ноде (замерено).
        minWidgetHeight: 396,
        onResize: () => editor.scheduleDraw(editor.NEED_ALL),
    });

    // ⚠️ LiteGraph уже выдал ноде размер, поэтому defaultWidth/defaultHeight
    // хелпера не применяются — он ставит только нижнюю границу. Разворачиваем
    // ноду сами; загрузка workflow всё равно перезапишет размер своим.
    node.size = [Math.max(node.size?.[0] || 0, 700), Math.max(node.size?.[1] || 0, 750)];

    hideWidget(node, INPUT_PATH);
    hideWidget(node, INPUT_START);
    hideWidget(node, INPUT_END);

    const observer = new ResizeObserver(() => editor.scheduleDraw(editor.NEED_ALL));
    observer.observe(editor.element);

    // Невидимая нода не должна крутить видео и тянуть миниатюры: десять
    // загрузчиков в одном workflow иначе декодируют десять роликов разом.
    const visibility = new IntersectionObserver(([entry]) => {
        if (!entry.isIntersecting) {
            editor.playback.pause();
            editor.strip.abortAll();
            abortActiveUpload();
        }
    });
    visibility.observe(editor.element);

    /**
     * Перечитать состояние из ноды.
     *
     * ⚠️ Отдельно от сборки интерфейса: при загрузке workflow значения виджетов
     * приезжают ПОСЛЕ onNodeCreated (§12.5.12), и собранный один раз редактор
     * навсегда остался бы с дефолтами.
     */
    // Обычные виджеты ноды меняются мимо нас — подписываемся на их изменение,
    // чтобы число кадров в строке состояния не отставало от настроек.
    for (const name of ["frame_rate", "frame_step", "max_frames"]) {
        const widget = getWidget(node, name);
        if (!widget) continue;
        const previous = widget.callback;
        widget.callback = function tsVideoWidgetChanged(...args) {
            const result = previous?.apply(this, args);
            updateStatus();
            return result;
        };
    }

    node._tsVideoLoaderRehydrate = () => {
        const path = String(readPersisted(node, INPUT_PATH, "") || "");
        editor.state.cropStart = readNumber(node, INPUT_START, 0);
        editor.state.cropEnd = readNumber(node, INPUT_END, -1);
        editor.state.looping = node.properties?.[PROP_LOOP] !== false;
        editor.state.showWave = node.properties?.[PROP_WAVE] !== false;
        editor.video.muted = node.properties?.[PROP_MUTED] !== false;
        loopButton.classList.toggle("is-active", editor.state.looping);
        syncWaveToggle();
        setIcon(muteButton, editor.video.muted ? "muted" : "sound");

        const meta = node.properties?.[PROP_META];
        if (meta) {
            // Скаляры из свойств позволяют нарисовать шкалу ДО ответа сервера —
            // нода не мигает пустотой при открытии workflow.
            editor.applyMetadata({ ...meta, peaks: null });
            nameLabel.textContent = meta.filename || path;
        }
        pathInput.value = path.includes("[") ? "" : path;
        editor.viewport.fromJSON({
            zoom: node.properties?.[PROP_ZOOM],
            viewStart: node.properties?.[PROP_VIEW],
        });

        editor.state.path = path;
        pathInput.value = path.includes("[") ? "" : path;
        if (path) fetchMetadata(path);
        updateStatus();
        editor.scheduleDraw(editor.NEED_ALL);
    };

    node._tsVideoLoaderCleanup = () => {
        teardownDrop?.();
        fullscreen?.close();
        document.removeEventListener("keydown", onKeyDown, true);
        observer.disconnect();
        visibility.disconnect();
        editor.dispose();
    };

    const previousRemoved = node.onRemoved;
    node.onRemoved = function tsVideoRemoved(...args) {
        node._tsVideoLoaderCleanup?.();
        return previousRemoved?.apply(this, args);
    };

    node._tsVideoLoaderRehydrate();
    requestAnimationFrame(() => editor.scheduleDraw(editor.NEED_ALL));
}

export { app };
