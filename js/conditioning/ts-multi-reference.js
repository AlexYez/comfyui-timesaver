// TS Multi Reference — drag-and-drop multi-image picker.
//
// The Python node TS_MultiReference exposes three combo widgets
// (image_1, image_2, image_3) that store the selected reference image
// filenames. This extension hides those native combo widgets and
// renders a custom DOM widget on top: three thumbnail slots in a row
// with badges, drag-and-drop file upload, drag-and-drop reordering,
// and a delete button per slot.
//
// The combo widgets remain the source of truth — every UI action
// writes back into them, so workflow JSON serializes/deserializes the
// same way as before (no schema change).

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const EXTENSION_ID = "ts.multiReference";
const NODE_NAME = "TS_MultiReference";
const SLOT_COUNT = 3;
const SLOT_WIDGETS = ["image_1", "image_2", "image_3"];
const STYLE_ID = "ts-multi-reference-styles";
const EMPTY_VALUE = "";
const DOM_WIDGET_NAME = "ts_multi_reference_ui";
const DRAG_MIME = "application/x-ts-multi-reference-slot";
const PLACEHOLDER_LABEL = "Drop image\nor click";
const CONTAINER_HEIGHT = 130;

function tsMrLog(message, ...args) {
    if (args.length) console.log(`[TS Multi Reference] ${message}`, ...args);
    else console.log(`[TS Multi Reference] ${message}`);
}

function tsMrWarn(message, ...args) {
    console.warn(`[TS Multi Reference] ${message}`, ...args);
}

function tsMrEnsureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-mr-container {
    display: flex;
    flex-direction: row;
    gap: 2px;
    padding: 4px;
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    min-height: 0;
    color: #e6e7ea;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    --ts-mr-accent: #4af;
    --ts-mr-border: #2d343f;
    --ts-mr-bg: #1c2129;
    --ts-mr-bg-hover: #232a35;
    pointer-events: auto;
    user-select: none;
}
.ts-mr-slot {
    flex: 1 1 0;
    min-width: 0;
    aspect-ratio: 1 / 1;
    position: relative;
    border: 1px dashed var(--ts-mr-border);
    border-radius: 4px;
    background: var(--ts-mr-bg);
    overflow: hidden;
    cursor: pointer;
    box-sizing: border-box;
    transition: border-color 80ms ease, background 80ms ease;
}
.ts-mr-slot:hover {
    background: var(--ts-mr-bg-hover);
}
.ts-mr-slot.ts-mr-filled {
    border-style: solid;
    border-color: var(--ts-mr-border);
    cursor: grab;
}
.ts-mr-slot.ts-mr-dragover {
    border-color: var(--ts-mr-accent);
    background: rgba(68, 170, 255, 0.12);
}
.ts-mr-slot.ts-mr-dragging {
    opacity: 0.4;
}
.ts-mr-thumb {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    pointer-events: none;
    background: var(--ts-mr-bg);
}
.ts-mr-placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #7a828f;
    font-size: 11px;
    line-height: 1.2;
    pointer-events: none;
    white-space: pre-line;
    padding: 4px;
}
.ts-mr-badge {
    position: absolute;
    top: 3px;
    left: 3px;
    background: rgba(0, 0, 0, 0.72);
    color: #ffffff;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 5px;
    border-radius: 3px;
    line-height: 1;
    pointer-events: none;
}
.ts-mr-remove {
    position: absolute;
    top: 3px;
    right: 3px;
    width: 18px;
    height: 18px;
    line-height: 16px;
    text-align: center;
    background: rgba(0, 0, 0, 0.72);
    color: #ffffff;
    font-size: 14px;
    font-weight: 600;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0;
    transition: opacity 80ms ease;
}
.ts-mr-slot:hover .ts-mr-remove,
.ts-mr-slot.ts-mr-filled.ts-mr-dragover .ts-mr-remove {
    opacity: 1;
}
.ts-mr-remove:hover {
    background: rgba(220, 76, 76, 0.85);
}
.ts-mr-busy {
    pointer-events: none;
    opacity: 0.6;
}
`;
    document.head.appendChild(style);
}

function tsMrFindWidget(node, name) {
    return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function tsMrGetSlotValue(node, index) {
    const widget = tsMrFindWidget(node, SLOT_WIDGETS[index]);
    const raw = widget?.value;
    return typeof raw === "string" ? raw : EMPTY_VALUE;
}

function tsMrSetSlotValue(node, index, name) {
    const widget = tsMrFindWidget(node, SLOT_WIDGETS[index]);
    if (!widget) return;
    const value = typeof name === "string" ? name : EMPTY_VALUE;
    if (widget.value !== value) {
        widget.value = value;
        try {
            widget.callback?.(value);
        } catch (err) {
            tsMrWarn(`callback failed for ${SLOT_WIDGETS[index]}`, err);
        }
    }
}

function tsMrEnsureOptionContains(node, name) {
    if (!name) return;
    for (const widgetName of SLOT_WIDGETS) {
        const widget = tsMrFindWidget(node, widgetName);
        const options = widget?.options;
        const values = options?.values;
        if (Array.isArray(values) && !values.includes(name)) {
            values.push(name);
        }
    }
}

function tsMrThumbUrl(name) {
    if (!name) return "";
    const params = new URLSearchParams({
        filename: name,
        type: "input",
        subfolder: "",
    });
    try {
        return api.apiURL(`/view?${params.toString()}`);
    } catch (err) {
        return `/view?${params.toString()}`;
    }
}

async function tsMrUploadImage(file) {
    const body = new FormData();
    body.append("image", file, file.name || "upload.png");
    body.append("overwrite", "false");
    body.append("type", "input");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (resp.status !== 200) {
        throw new Error(`Upload failed with status ${resp.status}`);
    }
    const data = await resp.json();
    if (!data?.name) {
        throw new Error("Upload response did not include a filename");
    }
    return data.name;
}

function tsMrFirstEmptyIndex(node) {
    for (let index = 0; index < SLOT_COUNT; index += 1) {
        if (!tsMrGetSlotValue(node, index)) return index;
    }
    return -1;
}

function tsMrCollectFilledOrdered(node) {
    const filled = [];
    for (let index = 0; index < SLOT_COUNT; index += 1) {
        const value = tsMrGetSlotValue(node, index);
        if (value) filled.push(value);
    }
    return filled;
}

function tsMrRewriteSlotsFromArray(node, names) {
    for (let index = 0; index < SLOT_COUNT; index += 1) {
        tsMrSetSlotValue(node, index, names[index] || EMPTY_VALUE);
    }
}

function tsMrSetDirty(node) {
    node?.setDirtyCanvas?.(true, true);
    app?.graph?.setDirtyCanvas?.(true, true);
}

async function tsMrUploadFilesToSlot(node, files, targetIndex) {
    if (!files || !files.length) return;
    const container = node?.tsMultiReferenceContainer;
    container?.classList?.add("ts-mr-busy");
    try {
        let nextSlot = targetIndex;
        if (nextSlot == null || nextSlot < 0 || nextSlot >= SLOT_COUNT) {
            nextSlot = tsMrFirstEmptyIndex(node);
        }
        for (const file of files) {
            if (!file || !file.type?.startsWith("image/")) continue;
            if (nextSlot < 0 || nextSlot >= SLOT_COUNT) break;
            try {
                const name = await tsMrUploadImage(file);
                tsMrEnsureOptionContains(node, name);
                tsMrSetSlotValue(node, nextSlot, name);
                tsMrLog(`uploaded ${name} into slot ${nextSlot + 1}`);
            } catch (err) {
                tsMrWarn(`upload failed for ${file.name}`, err);
            }
            nextSlot = tsMrFirstEmptyIndex(node);
        }
    } finally {
        container?.classList?.remove("ts-mr-busy");
        tsMrRender(node);
        tsMrSetDirty(node);
    }
}

function tsMrPickFromLibrary(node, slotIndex) {
    // Fall back to the underlying combo widget's behaviour: when ComfyUI
    // has a combobox open routine, expose the original options via the
    // combo widget's own callback. The simplest approach is to delegate
    // by triggering a click on a hidden file input — combobox UI is
    // non-trivial to summon outside LiteGraph. The drop / click-to-pick
    // workflow already handles the common case.
    tsMrTriggerFilePicker(node, slotIndex);
}

function tsMrTriggerFilePicker(node, slotIndex) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.multiple = slotIndex == null;
    input.style.display = "none";
    input.addEventListener("change", () => {
        const files = Array.from(input.files || []);
        tsMrUploadFilesToSlot(node, files, slotIndex);
        input.remove();
    });
    document.body.appendChild(input);
    input.click();
}

function tsMrSwapSlots(node, fromIndex, toIndex) {
    if (fromIndex === toIndex) return;
    if (fromIndex < 0 || toIndex < 0) return;
    if (fromIndex >= SLOT_COUNT || toIndex >= SLOT_COUNT) return;

    const values = [];
    for (let index = 0; index < SLOT_COUNT; index += 1) {
        values.push(tsMrGetSlotValue(node, index));
    }

    const moved = values.splice(fromIndex, 1)[0];
    values.splice(toIndex, 0, moved);

    tsMrRewriteSlotsFromArray(node, values);
    tsMrRender(node);
    tsMrSetDirty(node);
}

function tsMrClearSlot(node, slotIndex) {
    // When clearing a middle slot we shift trailing values up so badges
    // 1..N stay contiguous.
    const filled = tsMrCollectFilledOrdered(node);
    let counter = 0;
    const result = [];
    for (let index = 0; index < SLOT_COUNT; index += 1) {
        const wasFilled = !!tsMrGetSlotValue(node, index);
        if (index === slotIndex) {
            if (wasFilled) counter += 1; // skip this one
            result.push(EMPTY_VALUE);
            continue;
        }
        if (wasFilled) {
            result.push(filled[counter] || EMPTY_VALUE);
            counter += 1;
        } else {
            result.push(EMPTY_VALUE);
        }
    }
    // Re-pack so empty slot moves to the end after removal.
    const compact = result.filter((value) => value);
    while (compact.length < SLOT_COUNT) compact.push(EMPTY_VALUE);
    tsMrRewriteSlotsFromArray(node, compact);
    tsMrRender(node);
    tsMrSetDirty(node);
}

function tsMrBuildSlotElement(node, slotIndex) {
    const slot = document.createElement("div");
    slot.className = "ts-mr-slot";
    slot.dataset.slotIndex = String(slotIndex);

    const thumb = document.createElement("img");
    thumb.className = "ts-mr-thumb";
    thumb.alt = "";
    thumb.draggable = false;
    slot.appendChild(thumb);

    const placeholder = document.createElement("div");
    placeholder.className = "ts-mr-placeholder";
    placeholder.textContent = PLACEHOLDER_LABEL;
    slot.appendChild(placeholder);

    const badge = document.createElement("div");
    badge.className = "ts-mr-badge";
    badge.textContent = `Image ${slotIndex + 1}`;
    slot.appendChild(badge);

    const remove = document.createElement("div");
    remove.className = "ts-mr-remove";
    remove.textContent = "×";
    remove.title = "Remove this reference";
    remove.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        tsMrClearSlot(node, slotIndex);
    });
    slot.appendChild(remove);

    slot.addEventListener("click", (event) => {
        if (event.target === remove) return;
        const value = tsMrGetSlotValue(node, slotIndex);
        if (value) return;
        tsMrTriggerFilePicker(node, slotIndex);
    });

    slot.addEventListener("dragstart", (event) => {
        const value = tsMrGetSlotValue(node, slotIndex);
        if (!value) {
            event.preventDefault();
            return;
        }
        event.dataTransfer.effectAllowed = "move";
        try {
            event.dataTransfer.setData(DRAG_MIME, String(slotIndex));
            event.dataTransfer.setData("text/plain", String(slotIndex));
        } catch (err) {
            // some browsers require setData for the drag to start
        }
        slot.classList.add("ts-mr-dragging");
    });

    slot.addEventListener("dragend", () => {
        slot.classList.remove("ts-mr-dragging");
    });

    slot.addEventListener("dragenter", (event) => {
        event.preventDefault();
        slot.classList.add("ts-mr-dragover");
    });

    slot.addEventListener("dragover", (event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = event.dataTransfer.types.includes("Files")
            ? "copy"
            : "move";
        slot.classList.add("ts-mr-dragover");
    });

    slot.addEventListener("dragleave", (event) => {
        if (event.target !== slot) return;
        slot.classList.remove("ts-mr-dragover");
    });

    slot.addEventListener("drop", (event) => {
        event.preventDefault();
        event.stopPropagation();
        slot.classList.remove("ts-mr-dragover");

        const files = Array.from(event.dataTransfer?.files || []).filter(
            (file) => file.type?.startsWith("image/"),
        );
        if (files.length) {
            tsMrUploadFilesToSlot(node, files, slotIndex);
            return;
        }

        const sourceRaw =
            event.dataTransfer?.getData(DRAG_MIME) ||
            event.dataTransfer?.getData("text/plain");
        const sourceIndex = parseInt(sourceRaw, 10);
        if (!Number.isFinite(sourceIndex)) return;
        tsMrSwapSlots(node, sourceIndex, slotIndex);
    });

    return { slot, thumb, placeholder, badge, remove };
}

function tsMrRender(node) {
    const refs = node?.tsMultiReferenceSlots;
    if (!refs) return;
    for (let index = 0; index < SLOT_COUNT; index += 1) {
        const value = tsMrGetSlotValue(node, index);
        const ref = refs[index];
        if (!ref) continue;
        const { slot, thumb, placeholder } = ref;
        if (value) {
            slot.classList.add("ts-mr-filled");
            slot.draggable = true;
            placeholder.style.display = "none";
            const url = tsMrThumbUrl(value);
            // Prevent re-loading identical src to avoid flicker
            if (thumb.dataset.tsMrCurrent !== value) {
                thumb.dataset.tsMrCurrent = value;
                thumb.src = url;
            }
            thumb.style.display = "block";
            thumb.title = value;
        } else {
            slot.classList.remove("ts-mr-filled");
            slot.draggable = false;
            placeholder.style.display = "flex";
            thumb.removeAttribute("src");
            delete thumb.dataset.tsMrCurrent;
            thumb.style.display = "none";
            thumb.title = "";
        }
    }
}

function tsMrHideNativeWidgets(node) {
    for (const widgetName of SLOT_WIDGETS) {
        const widget = tsMrFindWidget(node, widgetName);
        if (!widget) continue;
        if (widget.tsMultiReferenceHidden) continue;
        widget.tsMultiReferenceHidden = true;
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
    }
}

function tsMrAttachContainerDropZone(node, container) {
    container.addEventListener("dragenter", (event) => {
        if (!event.dataTransfer?.types.includes("Files")) return;
        event.preventDefault();
    });
    container.addEventListener("dragover", (event) => {
        if (!event.dataTransfer?.types.includes("Files")) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
    });
    container.addEventListener("drop", (event) => {
        const files = Array.from(event.dataTransfer?.files || []).filter(
            (file) => file.type?.startsWith("image/"),
        );
        if (!files.length) return;
        // Only handle drops that landed on the container background, not
        // on an individual slot (which has its own handler).
        if (event.target !== container) return;
        event.preventDefault();
        tsMrUploadFilesToSlot(node, files, null);
    });
}

function tsMrSetupNode(node) {
    if (node?.tsMultiReferenceReady) return;
    if (node?.comfyClass !== NODE_NAME && node?.type !== NODE_NAME) return;

    tsMrEnsureStyles();
    tsMrHideNativeWidgets(node);

    const container = document.createElement("div");
    container.className = "ts-mr-container";

    const slots = [];
    for (let index = 0; index < SLOT_COUNT; index += 1) {
        const ref = tsMrBuildSlotElement(node, index);
        slots.push(ref);
        container.appendChild(ref.slot);
    }

    tsMrAttachContainerDropZone(node, container);

    node.tsMultiReferenceContainer = container;
    node.tsMultiReferenceSlots = slots;

    node.addDOMWidget(DOM_WIDGET_NAME, "div", container, {
        serialize: false,
        getMinHeight: () => CONTAINER_HEIGHT,
        getMaxHeight: () => CONTAINER_HEIGHT,
        getHeight: () => CONTAINER_HEIGHT,
    });

    // Initial render after widgets are ready.
    setTimeout(() => tsMrRender(node), 0);

    node.tsMultiReferenceReady = true;
}

function tsMrSuppressDefaultImagePreview(node) {
    // ComfyUI's default behaviour for any node with an IMAGE output is to
    // render the latest output image inside the node body (`node.imgs`).
    // For TS_MultiReference we already show the references inside the
    // custom DOM widget, so the auto-preview just produces a duplicate
    // big thumbnail under the slots. Suppress it.
    if (Array.isArray(node.imgs)) node.imgs = [];
    node.images = undefined;
    if (typeof node.setSizeForImage === "function") {
        // setSizeForImage adjusts node height for the preview; running it
        // with an empty imgs list collapses the preview area.
        try {
            node.setSizeForImage(true);
        } catch (err) {
            // ignore — different ComfyUI versions expose different signatures
        }
    }
}

app.registerExtension({
    name: EXTENSION_ID,
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            tsMrSetupNode(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            tsMrSetupNode(this);
            tsMrSuppressDefaultImagePreview(this);
            // After workflow JSON restored widget values, refresh thumbs.
            setTimeout(() => tsMrRender(this), 0);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            tsMrSuppressDefaultImagePreview(this);
            tsMrSetDirty(this);
            return result;
        };
    },
    nodeCreated(node) {
        if (node?.comfyClass === NODE_NAME || node?.type === NODE_NAME) {
            tsMrSetupNode(node);
            tsMrSuppressDefaultImagePreview(node);
        }
    },
    loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_NAME && node?.type !== NODE_NAME) return;
        tsMrSetupNode(node);
        tsMrSuppressDefaultImagePreview(node);
        setTimeout(() => tsMrRender(node), 0);
    },
});
