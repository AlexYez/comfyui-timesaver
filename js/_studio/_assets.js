// TS Studio kit — asset provider registry (ui-kit layer).
//
// The Library tab hosts whichever provider detects itself (plan §7.2).
// ArtiusProvider mounts once Artius ships its embed API; until then its
// cards already drag into studio drop zones through _dnd.js. The fallback
// provider is always available: recent images from the server history plus
// an OS file picker, drawn with the shared grid styling.

import { makeAssetDraggable, makePointerDragSource } from "./_dnd.js";
import { markOverlayAbove } from "../_fullscreen.js";

const PROVIDERS = [];

/** Extension point: {id, label, detect() -> bool|Promise, mount(host, ctx) -> {unmount}} */
export function registerAssetProvider(provider) {
    PROVIDERS.push(provider);
}

export async function pickAssetProvider() {
    for (const provider of PROVIDERS) {
        try {
            if (await provider.detect()) return provider;
        } catch {
            continue;
        }
    }
    return null;
}

// ── Artius Browser (embed level; drag-only until mountPanel ships) ──────── //
registerAssetProvider({
    id: "artius",
    detect: () => typeof window.tsArtiusBrowser?.mountPanel === "function",
    mount(host, ctx) {
        const handle = window.tsArtiusBrowser.mountPanel(host, {
            filter: { types: ["image"] },
            multi: false,
            onPick: (asset) => ctx.onPick?.({
                type: "image",
                name: asset.filename,
                url: asset.file_url,
            }),
        });
        const releaseLightbox = adoptArtiusLightbox();
        // Carry cards by pointer as well. The browser's own HTML5 drag is left
        // in place — this is a second route that no focus handling or shadow
        // boundary can veto, which is what made dragging out of the embedded
        // panel unreliable in the first place.
        const releaseDrag = makePointerDragSource(host, {
            pick: (path) => {
                const card = path.find((node) => node?.dataset?.cardId);
                if (!card) return null;
                const panel = path.find((node) => node?.tagName?.toLowerCase()
                    === "ts-artius-browser-panel");
                const asset = panel?.tsFindItemById?.(Number(card.dataset.cardId));
                return asset?.file_url && String(asset.type) === "image" ? asset : null;
            },
            preview: (asset) => asset.preview_url || asset.file_url,
            item: (asset) => ({
                type: "image",
                name: asset.filename || "artius.png",
                getBlob: async () => {
                    const response = await fetch(asset.file_url);
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return response.blob();
                },
            }),
        });
        return {
            unmount: () => { releaseDrag(); releaseLightbox(); handle?.unmount?.(); },
        };
    },
});

// Artius opens its full-size viewer as a body-level element sized for the
// page, not for a host that is already fullscreen: it would paint UNDER the
// studio and its Escape would close the studio instead of the lightbox. The
// adapter lifts it — the browser stays a plain guest, no patch on its side.
const ARTIUS_VIEWER_TAG = "ts-artius-browser-viewer";

function adoptArtiusLightbox() {
    const adopted = new Map();
    const adopt = (element) => {
        if (adopted.has(element)) return;
        adopted.set(element, markOverlayAbove(element));
    };
    for (const node of document.querySelectorAll(ARTIUS_VIEWER_TAG)) adopt(node);
    const observer = new MutationObserver((records) => {
        for (const record of records) {
            for (const node of record.addedNodes) {
                if (node.nodeType !== 1) continue;
                if (node.tagName?.toLowerCase() === ARTIUS_VIEWER_TAG) adopt(node);
                else node.querySelectorAll?.(ARTIUS_VIEWER_TAG).forEach(adopt);
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    return () => {
        observer.disconnect();
        for (const release of adopted.values()) release();
        adopted.clear();
    };
}

// ── fallback: recent server images + OS picker ──────────────────────────── //
registerAssetProvider({
    id: "fallback",
    detect: () => true,
    mount(host, ctx) {
        const grid = document.createElement("div");
        grid.className = "ts-studio__gallerygrid";
        const note = document.createElement("div");
        note.className = "ts-studio__galleryempty";
        note.textContent = ctx.t.libraryHint;
        host.append(note, grid);

        let cancelled = false;
        (async () => {
            try {
                const response = await ctx.api.fetchApi("/history?max_items=256");
                const history = await response.json();
                const seen = new Set();
                const images = [];
                for (const entry of Object.values(history)) {
                    for (const output of Object.values(entry.outputs || {})) {
                        for (const image of output.images || []) {
                            if (image.type !== "output") continue;
                            const key = `${image.subfolder}/${image.filename}`;
                            if (seen.has(key)) continue;
                            seen.add(key);
                            images.push(image);
                        }
                    }
                }
                if (cancelled) return;
                for (const image of images.slice(-60).reverse()) {
                    const url = "/view?" + new URLSearchParams({
                        filename: image.filename,
                        subfolder: image.subfolder || "",
                        type: "output",
                    });
                    const card = document.createElement("button");
                    card.type = "button";
                    card.className = "ts-studio__card";
                    const img = document.createElement("img");
                    img.loading = "lazy";
                    img.alt = image.filename;
                    img.src = url;
                    card.appendChild(img);
                    const asset = { type: "image", name: image.filename, url };
                    card.addEventListener("dblclick", () => ctx.onPick?.(asset));
                    card.title = ctx.t.libraryPickTip;
                    makeAssetDraggable(card, asset);
                    grid.appendChild(card);
                }
                if (!grid.children.length) note.textContent = ctx.t.libraryEmpty;
            } catch (err) {
                note.textContent = String(err?.message || err);
            }
        })();

        return { unmount: () => { cancelled = true; host.textContent = ""; } };
    },
});
