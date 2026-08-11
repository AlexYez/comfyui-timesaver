// Воспроизведение: цикл по выделению, склейка перемоток, покадровый шаг.
//
// Обёртка над обычным `<video>`, общая для загрузчика и сохранятеля.

/**
 * @param {HTMLVideoElement} video
 * @param {object} options
 * @param {() => {left:number,right:number}} [options.getRange] что зациклить
 * @param {() => boolean} [options.isLooping]
 * @param {() => number} [options.getFps]
 * @param {(time:number) => void} [options.onTime]
 * @param {(playing:boolean) => void} [options.onState]
 */
export function createPlayback(video, {
    getRange = () => ({ left: 0, right: 0 }),
    isLooping = () => false,
    getFps = () => 0,
    onTime,
    onState,
} = {}) {
    // ⚠️ Собственный `loop` у элемента всегда выключен: нам нужен цикл по
    // ВЫДЕЛЕННОМУ куску, а не по всему файлу.
    video.loop = false;

    let frameHandle = 0;
    let rafHandle = 0;
    let seekTarget = null;
    let seekBusy = false;

    const fps = () => {
        const value = Number(getFps?.() ?? 0);
        return value > 0 ? value : 25;
    };

    const stopLoop = () => {
        if (frameHandle && video.cancelVideoFrameCallback) {
            video.cancelVideoFrameCallback(frameHandle);
        }
        cancelAnimationFrame(rafHandle);
        frameHandle = 0;
        rafHandle = 0;
    };

    const tick = (_now, meta) => {
        const time = meta?.mediaTime ?? video.currentTime;
        const { left, right } = getRange();
        const epsilon = 0.5 / fps();

        if (right > left && time >= right - epsilon) {
            if (isLooping()) {
                video.currentTime = left;
            } else {
                video.pause();
                video.currentTime = right;
            }
        }
        onTime?.(time);
        if (!video.paused) schedule();
    };

    function schedule() {
        // requestVideoFrameCallback даёт точное время показанного кадра и не
        // тратит ничего на паузе. Где его нет — обычный кадр анимации.
        if (typeof video.requestVideoFrameCallback === "function") {
            frameHandle = video.requestVideoFrameCallback(tick);
        } else {
            rafHandle = requestAnimationFrame((now) => tick(now, null));
        }
    }

    const handlePlay = () => { onState?.(true); schedule(); };
    const handlePause = () => { onState?.(false); stopLoop(); onTime?.(video.currentTime); };
    const handleSeeked = () => {
        seekBusy = false;
        if (seekTarget !== null && Math.abs(video.currentTime - seekTarget) > 0.02) {
            apply(seekTarget);
        } else {
            seekTarget = null;
        }
        onTime?.(video.currentTime);
    };

    function apply(seconds) {
        seekBusy = true;
        try {
            video.currentTime = seconds;
        } catch {
            seekBusy = false;
        }
    }

    video.addEventListener("play", handlePlay);
    video.addEventListener("pause", handlePause);
    video.addEventListener("seeked", handleSeeked);

    return {
        play() {
            const { left, right } = getRange();
            // Начали не внутри куска — переносим в начало: играть «мимо
            // выделения» человек не просил.
            if (right > left && (video.currentTime < left || video.currentTime >= right - 1e-3)) {
                apply(left);
            }
            return video.play();
        },

        pause() { video.pause(); },

        toggle() { return video.paused ? this.play() : (video.pause(), undefined); },

        /**
         * Перемотать со склейкой.
         *
         * ⚠️ Без склейки скраб по часовому файлу превращается в шторм перемоток:
         * каждое движение мыши шлёт новую, декодер не успевает, картинка стоит.
         * Здесь запоминается только ПОСЛЕДНЯЯ цель, а следующая уходит после
         * `seeked`.
         */
        seek(seconds, { immediate = false, scrub = false } = {}) {
            seekTarget = Math.max(0, seconds);
            if (seekBusy && !immediate) return;
            // При скрабе (тянут ручку подрезки) точность до кадра не нужна, а
            // отзывчивость нужна: fastSeek садится на ближайший ключевой кадр и
            // отвечает заметно быстрее.
            if (scrub && typeof video.fastSeek === "function") {
                seekBusy = true;
                try { video.fastSeek(seekTarget); } catch { seekBusy = false; }
                return;
            }
            apply(seekTarget);
        },

        /** Шаг на кадр вперёд или назад. */
        stepFrames(delta) {
            video.pause();
            this.seek(video.currentTime + delta / fps(), { immediate: true });
        },

        setMuted(value) { video.muted = Boolean(value); },
        setVolume(value) { video.volume = Math.max(0, Math.min(1, Number(value) || 0)); },

        dispose() {
            stopLoop();
            video.removeEventListener("play", handlePlay);
            video.removeEventListener("pause", handlePause);
            video.removeEventListener("seeked", handleSeeked);
        },
    };
}
