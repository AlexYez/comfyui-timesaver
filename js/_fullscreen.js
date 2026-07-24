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
 * @param {string} [options.extraClass] Extra class on the overlay root.
 * @returns {FullscreenHandle}
 */
export function openFullscreenOverlay(content, options = {}) {
    const { onClose, onOpen, onKey, closeOnBackdrop = false, extraClass = "" } = options;
    const doc = content?.ownerDocument || document;
    let open = true;

    const modal = doc.createElement("div");
    modal.className = `${TS_UI_CLASS} ts-ui-modal${extraClass ? ` ${extraClass}` : ""}`;
    modal.append(content);

    const keyAnchor = doc.createElement("textarea");
    keyAnchor.className = "ts-ui-keyanchor";
    keyAnchor.readOnly = true;
    keyAnchor.tabIndex = -1;
    keyAnchor.setAttribute("aria-hidden", "true");
    modal.append(keyAnchor);

    const parkFocus = () => { try { keyAnchor.focus(); } catch { /* not focusable yet */ } };

    function onFocusIn(event) {
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
        // Detach content (keep it alive for reopen), then drop the overlay.
        content.remove();
        modal.remove();
        try { onClose?.(); } catch (err) { console.warn("[TS Fullscreen] onClose failed", err); }
    }

    modal.addEventListener("focusin", onFocusIn);
    modal.addEventListener("pointerdown", onPointerDown);
    doc.defaultView?.addEventListener("keydown", onKeyDown, true);
    doc.body.appendChild(modal);
    parkFocus();
    try { onOpen?.(); } catch (err) { console.warn("[TS Fullscreen] onOpen failed", err); }

    return { close, isOpen: () => open, modal, parkFocus };
}
