// Shared fullscreen-editor overlay for TS nodes.
//
// Several nodes (Lama Cleanup, Ideogram Designer, SAM Media Loader, …) host a
// heavy editing UI that is too big for the node body, so they mount it into a
// fullscreen overlay on demand. That overlay always needs the SAME plumbing:
//
//   • a .ts-ui-modal shell over the whole viewport;
//   • a hidden read-only textarea ("key anchor") that keyboard focus is parked
//     on, so ComfyUI's window-level, capture-phase hotkey service skips our
//     events (it ignores text-field targets) and Ctrl+Z can't reach the graph's
//     ChangeTracker and delete the node under the open editor
//     (project_memory/reference_modal_hotkeys.md, CLAUDE.md §12.5);
//   • focus re-parking whenever a non-field element inside the editor is focused;
//   • Esc to close, plus a caller-supplied editor-scoped key handler;
//   • tear-down that DETACHES (never destroys) the content so its canvas bitmaps,
//     masks and undo history survive an open → close → reopen cycle.
//
// This module encodes that once. A node builds its editing element, then calls
// openFullscreenOverlay(element, {...}); the returned handle closes it. Do NOT
// reinvent the overlay per node.

import { TS_UI_CLASS } from "./_theme.js";

// Overlays that a fullscreen editor may legitimately stack ON TOP of itself —
// an embedded image browser's lightbox, for one. Two things have to be true
// for that to work: it must paint above us, and Esc must reach IT rather than
// close the editor underneath. Both are handled here so no caller reinvents
// the stacking rules.
const ABOVE_ATTR = "data-ts-overlay-above";
const ABOVE_Z = 11500;   // .ts-ui-modal is 11000; see _theme.js

/**
 * Let `element` sit above the fullscreen overlay and own the Escape key while
 * it is on screen.
 *
 * @param {HTMLElement} element The foreign overlay root.
 * @returns {() => void} Undo, for when the overlay closes.
 */
export function markOverlayAbove(element) {
    if (!element) return () => {};
    element.setAttribute(ABOVE_ATTR, "1");
    const previous = element.style.zIndex;
    element.style.zIndex = String(ABOVE_Z);
    return () => {
        element.removeAttribute(ABOVE_ATTR);
        element.style.zIndex = previous;
    };
}

// A closed lightbox often stays in the DOM as an empty singleton, and an empty
// element can still report a client rect. Requiring real area is what keeps a
// dormant one from swallowing Escape forever.
const OPEN_MIN_SIDE = 40;

/** Is some foreign overlay currently stacked above the editors? */
function overlayAboveIsOpen(doc) {
    for (const node of doc.querySelectorAll(`[${ABOVE_ATTR}]`)) {
        const rect = node.getBoundingClientRect();
        if (rect.width >= OPEN_MIN_SIDE && rect.height >= OPEN_MIN_SIDE) return true;
    }
    return false;
}

// Single × icon shared by every fullscreen editor's close control.
const CLOSE_ICON_SVG =
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
    '<path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" ' +
    'stroke-width="2.2" stroke-linecap="round"/></svg>';

/**
 * @typedef {object} FullscreenHandle
 * @property {() => void} close     Tear the overlay down (idempotent).
 * @property {() => boolean} isOpen True until closed.
 * @property {HTMLElement} modal    The overlay root.
 * @property {() => void} parkFocus Re-park keyboard focus on the key anchor.
 */

/**
 * Mount `content` in a fullscreen TS overlay with the pack's standard focus
 * shielding and Esc-to-close.
 *
 * @param {HTMLElement} content The editing element to show. It is re-parented
 *   into the overlay and, on close, only DETACHED (content.remove()) — the
 *   caller keeps the reference so all in-closure state survives a reopen.
 * @param {object} [options]
 * @param {() => void} [options.onClose] Runs after the overlay is torn down.
 * @param {() => void} [options.onOpen]  Runs after mount (layout is live).
 * @param {(event: KeyboardEvent) => void} [options.onKey] Editor-scoped key
 *   handler (Esc is already handled). It decides whether to ignore keys while a
 *   real text field is focused.
 * @param {boolean} [options.closeOnBackdrop=false] Close when the dimmed area
 *   outside the content is clicked. Leave false for editors that could lose work.
 * @param {boolean} [options.center=false] Centre the content in the viewport
 *   instead of letting it fill the shell. Set it for DIALOGS (reports,
 *   summaries) whose content sizes itself; leave it off for editors that are
 *   meant to be full-bleed.
 * @param {string} [options.extraClass] Extra class on the overlay root.
 * @param {boolean} [options.showClose=true] Render the unified top-right × close
 *   button. Turn off only if the editor supplies its own equivalent control.
 * @param {string} [options.closeTitle] Tooltip / aria-label for the close button
 *   (localise via the caller; defaults to "Close (Esc)").
 * @returns {FullscreenHandle}
 */
export function openFullscreenOverlay(content, options = {}) {
    const {
        onClose, onOpen, onKey, closeOnBackdrop = false, extraClass = "",
        showClose = true, closeTitle = "Close (Esc)", label = "", center = false,
    } = options;
    const doc = content?.ownerDocument || document;
    let open = true;

    const modal = doc.createElement("div");
    modal.className = `${TS_UI_CLASS} ts-ui-modal${center ? " ts-ui-modal--center" : ""}` +
        `${extraClass ? ` ${extraClass}` : ""}`;
    // Announce it as what it is. Assistive technology otherwise reads a
    // fullscreen editor as an ordinary <div> stacked over the page, with no
    // signal that the content behind it is inert.
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    // `label` names the dialog itself. closeTitle is deliberately NOT used for
    // it: that string names the close button, not the window it lives in.
    if (label) modal.setAttribute("aria-label", label);

    // A centred dialog wants its close control on its OWN corner; the shared
    // button is position:fixed, so it needs a positioned ancestor that is the
    // size of the panel. That is all this frame is. It is built per open and
    // dropped with the modal, so `content` is still only detached on close and
    // keeps its state for a reopen.
    const frame = center ? doc.createElement("div") : null;
    if (frame) {
        frame.className = "ts-ui-fs-frame";
        frame.append(content);
        modal.append(frame);
    } else {
        modal.append(content);
    }

    const keyAnchor = doc.createElement("textarea");
    keyAnchor.className = "ts-ui-keyanchor";
    keyAnchor.readOnly = true;
    keyAnchor.tabIndex = -1;
    keyAnchor.setAttribute("aria-hidden", "true");
    modal.append(keyAnchor);

    // Unified close control (top-right). Same button, same place, every editor.
    let closeButton = null;
    if (showClose) {
        closeButton = doc.createElement("button");
        closeButton.type = "button";
        closeButton.className = "ts-ui-btn ts-ui-btn--icon ts-ui-fs-close";
        closeButton.title = closeTitle;
        closeButton.setAttribute("aria-label", closeTitle);
        closeButton.innerHTML = CLOSE_ICON_SVG;
        closeButton.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            close();
        });
        (frame || modal).append(closeButton);
    }

    const parkFocus = () => { try { keyAnchor.focus(); } catch { /* not focusable yet */ } };

    // A drag that starts inside the overlay must keep whatever focus the
    // browser gives it. Re-parking focus on the off-screen anchor mid-gesture
    // cancels the drag before it leaves the source element — which is why
    // dragging a card out of the embedded asset browser did nothing at all.
    let dragging = false;
    const onDragStart = () => { dragging = true; };
    const onDragStop = () => { dragging = false; };

    function onFocusIn(event) {
        if (dragging) return;
        const target = event.target;
        if (target === keyAnchor) return;
        const tag = target?.tagName;
        // Real fields (text inputs, sliders, selects) legitimately keep focus;
        // everything else re-parks so the graph's hotkeys stay disarmed.
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target?.isContentEditable) return;
        parkFocus();
    }

    function onKeyDown(event) {
        if (!open) return;
        // Something is stacked above us (an embedded lightbox): it gets the
        // keystroke, including its own Escape-to-close.
        if (overlayAboveIsOpen(doc)) return;
        if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            close();
            return;
        }
        try { onKey?.(event); } catch (err) { console.warn("[TS Fullscreen] key handler failed", err); }
    }

    function onPointerDown(event) {
        if (closeOnBackdrop && event.target === modal) close();
    }

    function close() {
        if (!open) return;
        open = false;
        doc.defaultView?.removeEventListener("keydown", onKeyDown, true);
        modal.removeEventListener("focusin", onFocusIn);
        modal.removeEventListener("pointerdown", onPointerDown);
        modal.removeEventListener("dragstart", onDragStart, true);
        doc.removeEventListener("dragend", onDragStop, true);
        doc.removeEventListener("drop", onDragStop, true);
        // Detach content (keep it alive for reopen), then drop the overlay.
        content.remove();
        modal.remove();
        try { onClose?.(); } catch (err) { console.warn("[TS Fullscreen] onClose failed", err); }
    }

    modal.addEventListener("focusin", onFocusIn);
    modal.addEventListener("pointerdown", onPointerDown);
    // Capture phase, and on the document for the end: a drag can finish
    // anywhere, including outside the overlay.
    modal.addEventListener("dragstart", onDragStart, true);
    doc.addEventListener("dragend", onDragStop, true);
    doc.addEventListener("drop", onDragStop, true);
    doc.defaultView?.addEventListener("keydown", onKeyDown, true);
    doc.body.appendChild(modal);
    parkFocus();
    try { onOpen?.(); } catch (err) { console.warn("[TS Fullscreen] onOpen failed", err); }

    return { close, isOpen: () => open, modal, parkFocus };
}
