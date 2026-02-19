import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// --- Constants & Identity ---
const EXTENSION_ID = "ts.animationpreview";
const NODE_NAME = "TS_Animation_Preview";
const UI_KEY = "ts_animation_preview";
const STYLE_ID = "ts-animation-preview-styles";

// Layout Configuration
const MIN_NODE_WIDTH = 300;
const MIN_NODE_HEIGHT = 200;
const HEADER_HEIGHT_V1 = 30;
const FOOTER_PADDING_V1 = 10;

// Icons (SVG)
// volumeOn: Звук есть
// volumeOff: Звук выключен (Mute)
// play: Треугольник
// pause: Пауза
const ICONS = {
    play: `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`,
    pause: `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`,
    volumeOn: `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg>`,
    volumeOff: `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/></svg>`
};

/**
 * Injects CSS styles.
 */
function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .ts-anim-preview {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            background: #000;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
            box-sizing: border-box;
        }
        
        /* Video */
        .ts-anim-preview__video {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
            background: #000;
        }
        .ts-anim-preview__video[hidden] { display: none !important; }

        /* Placeholder */
        .ts-anim-preview__placeholder {
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            color: #666;
            font-size: 14px;
            font-family: sans-serif;
            pointer-events: none;
            text-align: center;
            width: 100%;
            z-index: 1;
        }

        /* Controls Container */
        .ts-anim-preview__controls {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 32px;
            background: rgba(15, 15, 15, 0.85);
            backdrop-filter: blur(4px);
            display: flex;
            align-items: center;
            padding: 0 6px;
            box-sizing: border-box;
            opacity: 0;
            transition: opacity 0.2s ease-in-out;
            z-index: 10;
        }
        
        .ts-anim-preview:hover .ts-anim-preview__controls {
            opacity: 1;
        }

        /* Buttons */
        .ts-anim-preview__btn {
            background: none;
            border: none;
            color: #ccc;
            width: 28px;
            height: 28px;
            padding: 5px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0.9;
            transition: color 0.1s;
        }
        .ts-anim-preview__btn:hover {
            color: #fff;
        }
        .ts-anim-preview__btn svg {
            width: 100%;
            height: 100%;
            filter: drop-shadow(0 1px 2px rgba(0,0,0,0.5));
        }

        /* Seek Bar (Slider) */
        .ts-anim-preview__seek {
            flex-grow: 1;
            margin: 0 8px;
            height: 4px;
            -webkit-appearance: none;
            background: #444;
            border-radius: 2px;
            cursor: pointer;
            outline: none;
        }
        .ts-anim-preview__seek::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #fff;
            cursor: pointer;
            box-shadow: 0 0 2px rgba(0,0,0,0.5);
            border: none;
        }
        .ts-anim-preview__seek::-moz-range-thumb {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #fff;
            cursor: pointer;
            border: none;
        }
    `;
    document.head.appendChild(style);
}

function isNodesV2() {
    if (typeof window === "undefined") return false;
    return Boolean(window.comfyAPI?.domWidget?.DOMWidgetImpl);
}

function stopPropagation(element, events) {
    events.forEach((eventName) => {
        element.addEventListener(eventName, (event) => {
            event.stopPropagation();
        });
    });
}

/**
 * Creates the DOM elements and logic for the video player.
 */
function setupAnimationPreview(node) {
    if (!node || node._tsAnimPreviewInit) return;
    node._tsAnimPreviewInit = true;

    ensureStyles();

    // Node configuration
    node.resizable = true;
    if (!node.size || node.size.length < 2) {
        node.size = [Math.max(300, MIN_NODE_WIDTH), Math.max(260, MIN_NODE_HEIGHT)];
    }

    // -- DOM Structure --
    const container = document.createElement("div");
    container.className = "ts-anim-preview";

    // Video Element
    const video = document.createElement("video");
    video.className = "ts-anim-preview__video";
    video.loop = true;
    // CRITICAL: Default muted = true to allow autoplay
    video.muted = true; 
    video.playsInline = true;
    video.controls = false; 
    video.hidden = true;

    // Placeholder
    const placeholder = document.createElement("div");
    placeholder.className = "ts-anim-preview__placeholder";
    placeholder.textContent = "No Media";

    // -- Custom Controls --
    const controls = document.createElement("div");
    controls.className = "ts-anim-preview__controls";

    // Play/Pause Button
    const playBtn = document.createElement("button");
    playBtn.className = "ts-anim-preview__btn";
    playBtn.innerHTML = ICONS.pause; // Default is playing -> show pause icon
    playBtn.title = "Play/Pause";

    // Seek Bar
    const seekSlider = document.createElement("input");
    seekSlider.type = "range";
    seekSlider.className = "ts-anim-preview__seek";
    seekSlider.min = 0;
    seekSlider.max = 100;
    seekSlider.value = 0;
    seekSlider.step = 0.1;

    // Volume Button
    const volBtn = document.createElement("button");
    volBtn.className = "ts-anim-preview__btn";
    volBtn.innerHTML = ICONS.volumeOff; // Default is muted -> show mute icon
    volBtn.title = "Mute/Unmute";

    // Assemble
    controls.appendChild(playBtn);
    controls.appendChild(seekSlider);
    controls.appendChild(volBtn);

    container.appendChild(video);
    container.appendChild(placeholder);
    container.appendChild(controls);

    // -- State Management --
    // Default: Playing (paused: false) AND Muted (muted: true)
    const state = node._tsAnimPreviewState || { paused: false, muted: true };
    node._tsAnimPreviewState = state;

    // Apply initial state to video
    video.muted = state.muted; 

    // -- UI Helpers --
    const updatePlayIcon = () => {
        // If paused -> Show Play icon
        // If playing -> Show Pause icon
        playBtn.innerHTML = video.paused ? ICONS.play : ICONS.pause;
    };

    const updateVolumeIcon = () => {
        // If muted -> Show VolumeOff icon
        // If sound on -> Show VolumeOn icon
        volBtn.innerHTML = video.muted ? ICONS.volumeOff : ICONS.volumeOn;
    };

    // Ensure icons match initial state
    updatePlayIcon();
    updateVolumeIcon();

    // -- Event Handlers --

    stopPropagation(controls, [
        "pointerdown", "pointerup", 
        "mousedown", "mouseup", 
        "wheel", "dblclick", "contextmenu", "click"
    ]);

    // 1. Play/Pause
    playBtn.addEventListener("click", () => {
        if (!video.src) return;
        
        if (video.paused) {
            video.play().catch(e => {});
            state.paused = false;
        } else {
            video.pause();
            state.paused = true;
        }
        updatePlayIcon();
        
        node.properties = node.properties || {};
        node.properties.ts_animation_preview_paused = state.paused;
    });

    // 2. Volume Toggle
    volBtn.addEventListener("click", () => {
        // Toggle property
        video.muted = !video.muted;
        state.muted = video.muted;
        
        updateVolumeIcon();
        
        node.properties = node.properties || {};
        node.properties.ts_animation_preview_muted = state.muted;
    });

    // 3. Seek Bar logic
    video.addEventListener("timeupdate", () => {
        if (!video.duration) return;
        // Only update if not currently being dragged (simple check)
        // For a simple node, updating always is usually fine
        seekSlider.value = (video.currentTime / video.duration) * 100;
    });

    seekSlider.addEventListener("input", () => {
        if (!video.duration) return;
        const time = (seekSlider.value / 100) * video.duration;
        video.currentTime = time;
    });

    // 4. Video Events & Autoplay
    video.addEventListener("play", updatePlayIcon);
    video.addEventListener("pause", updatePlayIcon);
    video.addEventListener("volumechange", updateVolumeIcon);
    
    video.addEventListener("loadedmetadata", () => {
        video.hidden = false;
        placeholder.hidden = true;
        controls.style.display = "flex";
        
        // Enforce State
        video.muted = state.muted;
        
        // Autoplay logic
        if (!state.paused) {
            const playPromise = video.play();
            if (playPromise !== undefined) {
                playPromise.catch((error) => {
                    // console.warn("Autoplay prevented:", error);
                    // If autoplay fails (unlikely if muted), we reflect it in UI
                    state.paused = true;
                    updatePlayIcon();
                });
            }
        } else {
            video.pause();
        }
        
        updatePlayIcon();
        updateVolumeIcon();
    });

    video.addEventListener("error", () => {
        video.hidden = true;
        placeholder.hidden = false;
        placeholder.textContent = "Error Loading Media";
        controls.style.display = "none";
    });

    // -- V1 / V2 Layout Logic --
    const isV2 = isNodesV2();
    const widgetOptions = { serialize: false, hideOnZoom: false };
    if (isV2) widgetOptions.getMinHeight = () => MIN_NODE_HEIGHT;

    const domWidget = node.addDOMWidget("ts_animation_preview", "div", container, widgetOptions);
    const domWidgetEl = domWidget?.element || domWidget?.el || domWidget?.container;

    if (domWidgetEl) {
        domWidgetEl.style.width = "100%";
        if (!isV2) {
            domWidgetEl.style.position = "absolute";
            domWidgetEl.style.left = "10px";
            domWidgetEl.style.top = `${HEADER_HEIGHT_V1}px`;
        }
    }

    const syncSizeV1 = () => {
        if (isV2) return;
        const width = node.size[0];
        const height = node.size[1];
        
        const wHeight = Math.max(MIN_NODE_HEIGHT, height - HEADER_HEIGHT_V1 - FOOTER_PADDING_V1);
        const wWidth = Math.max(MIN_NODE_WIDTH, width - 20);

        if (domWidgetEl) {
            domWidgetEl.style.width = `${wWidth}px`;
            domWidgetEl.style.height = `${wHeight}px`;
            domWidgetEl.style.top = `${HEADER_HEIGHT_V1}px`;
            domWidgetEl.style.left = "10px";
        }
        container.style.width = "100%";
        container.style.height = "100%";
    };

    const prevOnResize = node.onResize;
    node.onResize = function() {
        const r = prevOnResize?.apply(this, arguments);
        syncSizeV1();
        return r;
    };
    requestAnimationFrame(syncSizeV1);

    // -- Data Application --
    node._tsAnimPreviewApply = (payload) => {
        if (!payload || !payload.filename) {
            video.pause();
            video.removeAttribute("src");
            video.load();
            video.hidden = true;
            placeholder.hidden = false;
            placeholder.textContent = "No Media";
            controls.style.display = "none";
            return;
        }

        const params = {
            filename: payload.filename,
            subfolder: payload.subfolder || "",
            type: payload.type || "temp",
            format: payload.format || "video/mp4",
        };
        const url = api.apiURL(`/view?${new URLSearchParams(params).toString()}`);

        if (video.src !== url) {
            video.src = url;
            video.load();
        }

        // Restore properties from saved graph if available
        if (node.properties) {
            if (node.properties.ts_animation_preview_paused !== undefined) {
                state.paused = node.properties.ts_animation_preview_paused;
            }
            if (node.properties.ts_animation_preview_muted !== undefined) {
                state.muted = node.properties.ts_animation_preview_muted;
                video.muted = state.muted;
            }
        }
        
        node.properties = node.properties || {};
        node.properties.ts_animation_preview = payload;
    };

    if (node.properties?.ts_animation_preview) {
        node._tsAnimPreviewApply(node.properties.ts_animation_preview);
    } else {
        controls.style.display = "none";
    }
}

// --- Extension Registration ---

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setupAnimationPreview(this);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message && message[UI_KEY]) {
                this._tsAnimPreviewApply?.(message[UI_KEY][0]);
            }
            return r;
        };
    },
    loadedGraphNode(node) {
        if (node.type !== NODE_NAME) return;
        setupAnimationPreview(node);
    }
});