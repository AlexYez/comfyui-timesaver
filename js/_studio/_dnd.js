// TS Studio kit — the one drag-and-drop service (ui-kit layer).
//
// Every drop target in a studio registers here instead of wiring its own
// listeners (plan §7.1). Sources are normalizers in a registry — OS files,
// Artius assets, ComfyUI preview URLs, the studio's own gallery — so adding
// a source never touches a target. All payloads resolve to Blobs through
// HTTP; disk paths never cross the boundary.

export const STUDIO_ASSET_MIME = "application/x-ts-studio-asset";
const ARTIUS_MIME = "application/x-timesaver-artius-asset";

/** Upload a blob through /upload/image; returns the annotated name. */
export async function uploadImage(api, blob, filename) {
    const form = new FormData();
    form.append("image", blob, filename);
    const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
    if (!response.ok) throw new Error(`upload HTTP ${response.status}`);
    const payload = await response.json();
    const folder = payload.subfolder ? `${payload.subfolder}/` : "";
    return `${folder}${payload.name} [${payload.type || "input"}]`;
}

/**
 * Inverse of uploadImage: turn an annotated name ("sub/name.png [input]")
 * back into a /view URL. Annotated names are how every image param travels,
 * so this is what lets a saved run point at its own sources.
 */
export function annotatedImageUrl(annotated) {
    const text = String(annotated || "").trim();
    if (!text) return "";
    const match = /^(.*?)\s*\[(\w+)\]$/.exec(text);
    const path = (match ? match[1] : text).replace(/\\/g, "/");
    const type = match ? match[2] : "input";
    const slash = path.lastIndexOf("/");
    const filename = slash >= 0 ? path.slice(slash + 1) : path;
    const subfolder = slash >= 0 ? path.slice(0, slash) : "";
    return `/view?${new URLSearchParams({ filename, subfolder, type })}`;
}

const NORMALIZERS = [];

/** Extension point: {id, sniff(dataTransfer) -> bool, extract(dataTransfer) -> Promise<items>} */
export function registerDropSource(normalizer) {
    NORMALIZERS.push(normalizer);
}

/** @typedef {{type: "image", name: string, getBlob: () => Promise<Blob>}} DropItem */

// ── built-in sources ────────────────────────────────────────────────────── //
registerDropSource({
    id: "os-files",
    sniff: (dt) => [...(dt?.types || [])].includes("Files"),
    extract: async (dt) => [...(dt.files || [])]
        .filter((file) => file.type.startsWith("image/"))
        .map((file) => ({ type: "image", name: file.name, getBlob: async () => file })),
});

registerDropSource({
    id: "studio-gallery",
    sniff: (dt) => [...(dt?.types || [])].includes(STUDIO_ASSET_MIME),
    extract: async (dt) => {
        const payload = JSON.parse(dt.getData(STUDIO_ASSET_MIME) || "{}");
        const list = Array.isArray(payload) ? payload : [payload];
        return list.filter((a) => a?.url).map((asset) => ({
            type: "image",
            name: asset.name || "gallery.png",
            getBlob: async () => {
                const response = await fetch(asset.url);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return response.blob();
            },
        }));
    },
});

// Artius Browser cards (drag-only integration level; plan §7.2). Its payload
// carries file_url — HTTP access to the bytes, no disk paths.
registerDropSource({
    id: "artius",
    // The MIME is the contract, but it is not always visible: a drag that
    // starts inside the browser's shadow root can reach a dragover with its
    // types stripped. Artius also parks the payload on window for exactly
    // this case (its canvas bridge relies on the same fallback), so a drag
    // in flight from it is accepted either way.
    sniff: (dt) => [...(dt?.types || [])].includes(ARTIUS_MIME)
        || Boolean(window.__tsArtiusDraggedAsset),
    extract: async (dt) => {
        const raw = dt.getData(ARTIUS_MIME) || window.__tsArtiusDraggedAsset || "";
        const payload = typeof raw === "string" ? JSON.parse(raw || "null") : raw;
        const list = Array.isArray(payload) ? payload : payload ? [payload] : [];
        return list
            .filter((asset) => asset?.file_url && String(asset.type) === "image")
            .map((asset) => ({
                type: "image",
                name: asset.filename || "artius.png",
                getBlob: async () => {
                    const response = await fetch(asset.file_url);
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return response.blob();
                },
            }));
    },
});

// ComfyUI node previews drag as plain image URLs.
registerDropSource({
    id: "uri-list",
    sniff: (dt) => [...(dt?.types || [])].includes("text/uri-list"),
    extract: async (dt) => {
        const uri = (dt.getData("text/uri-list") || "").split("\n")[0]?.trim();
        if (!uri || !/^https?:|^\//.test(uri)) return [];
        return [{
            type: "image",
            name: decodeURIComponent(uri.split("/").pop()?.split("?")[0] || "image.png"),
            getBlob: async () => {
                const response = await fetch(uri);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const blob = await response.blob();
                if (!blob.type.startsWith("image/")) throw new Error("not an image");
                return blob;
            },
        }];
    },
});

export function sniffDrop(dataTransfer) {
    return NORMALIZERS.some((n) => n.sniff(dataTransfer));
}

export async function normalizeDrop(dataTransfer) {
    for (const normalizer of NORMALIZERS) {
        if (!normalizer.sniff(dataTransfer)) continue;
        try {
            const items = await normalizer.extract(dataTransfer);
            if (items.length) return items;
        } catch (err) {
            console.warn(`[TS Studio] drop source ${normalizer.id} failed`, err);
        }
    }
    return [];
}

/**
 * Turn an element into a standard drop zone.
 *
 * @param {HTMLElement} element
 * @param {object} options
 * @param {(items: DropItem[]) => void} options.onDrop
 * @param {number} [options.max] Cap on delivered items.
 * @returns {() => void} teardown
 */
export function makeDropZone(element, options) {
    // Also findable without the drag protocol: the pointer-drag fallback below
    // looks for this attribute under the cursor when a card is released.
    element.dataset.tsDropzone = "1";
    element._tsAcceptItems = (items) => options.onDrop(
        options.max ? items.slice(0, options.max) : items);
    const over = (event) => {
        if (!sniffDrop(event.dataTransfer)) return;
        event.preventDefault();
        event.stopPropagation();
        element.classList.add("is-drag-over");
    };
    const leave = () => element.classList.remove("is-drag-over");
    const drop = async (event) => {
        if (!sniffDrop(event.dataTransfer)) return;
        event.preventDefault();
        event.stopPropagation();
        element.classList.remove("is-drag-over");
        const items = await normalizeDrop(event.dataTransfer);
        if (items.length) options.onDrop(options.max ? items.slice(0, options.max) : items);
    };
    element.addEventListener("dragover", over);
    element.addEventListener("dragenter", over);
    element.addEventListener("dragleave", leave);
    element.addEventListener("drop", drop);
    return () => {
        delete element.dataset.tsDropzone;
        delete element._tsAcceptItems;
        element.removeEventListener("dragover", over);
        element.removeEventListener("dragenter", over);
        element.removeEventListener("dragleave", leave);
        element.removeEventListener("drop", drop);
    };
}

/** Make a gallery card (or any element) draggable as a studio asset. */
export function makeAssetDraggable(element, asset) {
    element.draggable = true;
    const onDragStart = (event) => {
        event.dataTransfer.setData(STUDIO_ASSET_MIME, JSON.stringify(asset));
        event.dataTransfer.effectAllowed = "copy";
    };
    element.addEventListener("dragstart", onDragStart);
    return () => element.removeEventListener("dragstart", onDragStart);
}


// ── pointer drag: the same gesture without the drag protocol ─────────────── //
// HTML5 drag-and-drop is not always available to us — a browser will refuse to
// start one when something inside the page moves focus at the wrong moment, and
// an embedded panel living in a shadow root makes that harder to control. This
// carries a card to a drop zone using plain pointer events, which nothing can
// veto, and hands the result to the same onDrop the zone already has.

const GHOST_ID = "ts-pointer-drag-ghost";
const DRAG_THRESHOLD = 6;

/**
 * Make an element's cards draggable by pointer.
 *
 * @param {HTMLElement} host Container the cards live in (shadow hosts fine).
 * @param {object} options
 * @param {(target: EventTarget[]) => ?object} options.pick Card -> asset, or
 *   null when the pointer did not start on one. Receives the composed path.
 * @param {(asset: object) => string} options.preview Thumbnail URL for the ghost.
 * @param {(asset: object) => object} options.item DropItem for the zone.
 * @returns {() => void} teardown
 */
export function makePointerDragSource(host, options) {
    let asset = null;
    let startX = 0;
    let startY = 0;
    let ghost = null;
    let armed = false;

    function ghostElement(url) {
        const element = document.createElement("div");
        element.id = GHOST_ID;
        element.style.cssText = "position:fixed;z-index:12000;width:88px;height:88px;"
            + "border-radius:8px;pointer-events:none;background-size:cover;"
            + "background-position:center;box-shadow:0 8px 24px rgba(0,0,0,.45);"
            + "opacity:.9;transform:translate(-50%,-50%)";
        if (url) element.style.backgroundImage = `url("${url}")`;
        document.body.appendChild(element);
        return element;
    }

    function zoneUnder(x, y) {
        for (const element of document.elementsFromPoint(x, y)) {
            const zone = element.closest?.("[data-ts-dropzone]");
            if (zone?._tsAcceptItems) return zone;
        }
        return null;
    }

    function onPointerDown(event) {
        if (event.button !== 0) return;
        const picked = options.pick(event.composedPath?.() || [event.target]);
        if (!picked) return;
        asset = picked;
        startX = event.clientX;
        startY = event.clientY;
        armed = true;
    }

    function onPointerMove(event) {
        if (!armed || !asset) return;
        if (!ghost) {
            if (Math.hypot(event.clientX - startX, event.clientY - startY) < DRAG_THRESHOLD) return;
            ghost = ghostElement(options.preview?.(asset));
        }
        ghost.style.left = `${event.clientX}px`;
        ghost.style.top = `${event.clientY}px`;
        const zone = zoneUnder(event.clientX, event.clientY);
        for (const element of document.querySelectorAll("[data-ts-dropzone].is-drag-over")) {
            if (element !== zone) element.classList.remove("is-drag-over");
        }
        zone?.classList.add("is-drag-over");
    }

    function finish(event) {
        const dragged = Boolean(ghost);
        const held = asset;
        armed = false;
        asset = null;
        ghost?.remove();
        ghost = null;
        for (const element of document.querySelectorAll("[data-ts-dropzone].is-drag-over")) {
            element.classList.remove("is-drag-over");
        }
        if (!dragged || !held) return;                 // a click, not a drag
        const zone = zoneUnder(event.clientX, event.clientY);
        if (!zone) return;
        try {
            zone._tsAcceptItems([options.item(held)]);
        } catch (err) {
            console.warn("[TS Studio] pointer drop failed", err);
        }
    }

    host.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("pointermove", onPointerMove, true);
    window.addEventListener("pointerup", finish, true);
    window.addEventListener("pointercancel", finish, true);
    return () => {
        host.removeEventListener("pointerdown", onPointerDown, true);
        window.removeEventListener("pointermove", onPointerMove, true);
        window.removeEventListener("pointerup", finish, true);
        window.removeEventListener("pointercancel", finish, true);
        ghost?.remove();
    };
}
