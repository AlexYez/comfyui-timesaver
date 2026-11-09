import { formatTimecode } from "../../_media/_ruler.js";

const MAX_PREVIEW_SIDE = 1280;
// The trim end is exclusive. Sample just inside it, including at the end of the file.
const FRAME_INSET_SECONDS = 0.001;

export function createRangePreview(stage, { rangeStart, rangeEnd }) {
    const element = document.createElement("div");
    element.className = "ts-vid__range-preview";
    element.hidden = true;

    function createPane(title) {
        const pane = document.createElement("div");
        pane.className = "ts-vid__range-pane";
        const canvas = document.createElement("canvas");
        canvas.className = "ts-vid__range-frame";
        canvas.hidden = true;
        const label = document.createElement("span");
        label.className = "ts-vid__badge ts-vid__range-label";
        pane.append(canvas, label);
        element.appendChild(pane);
        return { canvas, label, title, target: null, rendered: null };
    }

    const panes = [createPane(rangeStart), createPane(rangeEnd)];
    stage.appendChild(element);

    let url = "";
    let decoder = null;
    let pending = null;
    let nextPane = 0;
    let failed = false;
    let visible = typeof IntersectionObserver !== "function";

    function release() {
        if (!decoder) return;
        const previous = decoder;
        decoder = null;
        pending = null;
        previous.removeAttribute("src");
        previous.load();
    }

    function hide() {
        element.hidden = true;
        stage.classList.remove("has-range-preview");
        release();
    }

    function capture() {
        if (!decoder || !pending || decoder.seeking || decoder.readyState < 2) return;
        const { pane, time } = pending;
        const scale = Math.min(1, MAX_PREVIEW_SIDE / Math.max(decoder.videoWidth, decoder.videoHeight));
        const width = Math.round(decoder.videoWidth * scale);
        const height = Math.round(decoder.videoHeight * scale);
        if (!width || !height) {
            failed = true;
            hide();
            return;
        }
        if (pane.canvas.width !== width) pane.canvas.width = width;
        if (pane.canvas.height !== height) pane.canvas.height = height;
        pane.canvas.getContext("2d").drawImage(decoder, 0, 0, width, height);
        pane.canvas.hidden = false;
        pane.rendered = time;
        pending = null;
        pump();
    }

    function pump() {
        if (element.hidden || !visible || document.hidden || pending) return;
        // Keep one seek in flight. Alternate sides when both are dirty, and read
        // the latest target at dispatch so rapid drags never build a seek queue.
        const index = [nextPane, 1 - nextPane].find((i) => panes[i].target !== panes[i].rendered);
        if (index === undefined) return;
        if (!decoder) {
            const media = document.createElement("video");
            decoder = media;
            media.muted = true;
            media.preload = "metadata";
            media.addEventListener("loadedmetadata", () => { if (media === decoder) pump(); });
            for (const event of ["loadeddata", "seeked"]) {
                media.addEventListener(event, () => { if (media === decoder) capture(); });
            }
            media.addEventListener("error", () => {
                if (media !== decoder) return;
                failed = true;
                hide();
            });
            media.src = url;
            return;
        }
        if (decoder.readyState < 1) return;

        const pane = panes[index];
        nextPane = 1 - index;
        const limit = Number.isFinite(decoder.duration)
            ? Math.max(0, decoder.duration - FRAME_INSET_SECONDS) : Infinity;
        // A seek to exactly zero during metadata loading can reuse an old frame
        // in Chromium. Ask for a time inside the first frame instead.
        const time = Math.min(Math.max(FRAME_INSET_SECONDS, pane.target), limit);
        pending = { pane, time: pane.target };
        if (decoder.currentTime !== time || decoder.readyState < 2) decoder.currentTime = time;
        else capture();
    }

    function syncVisibility() {
        if (!visible || document.hidden) release();
        else pump();
    }
    const observer = typeof IntersectionObserver === "function"
        ? new IntersectionObserver(([entry]) => {
            visible = entry.isIntersecting;
            syncVisibility();
        }) : null;
    observer?.observe(stage);
    document.addEventListener("visibilitychange", syncVisibility);

    return {
        setSource(nextUrl) {
            hide();
            url = nextUrl;
            failed = false;
            for (const pane of panes) {
                pane.target = null;
                pane.rendered = null;
                pane.canvas.hidden = true;
            }
        },
        show(left, right) {
            if (!url || failed) return;
            panes[0].target = left;
            panes[1].target = Math.max(left, right - FRAME_INSET_SECONDS);
            for (const [index, time] of [left, right].entries()) {
                const label = `${panes[index].title} · ${formatTimecode(time)}`;
                if (panes[index].label.textContent !== label) panes[index].label.textContent = label;
            }
            element.hidden = false;
            stage.classList.add("has-range-preview");
            pump();
        },
        hide,
        dispose() {
            hide();
            observer?.disconnect();
            document.removeEventListener("visibilitychange", syncVisibility);
        },
    };
}
