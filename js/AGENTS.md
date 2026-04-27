# js/AGENTS.md — ComfyUI Frontend Extension Rules

This directory contains JavaScript/TypeScript frontend extensions for ComfyUI.

Follow the root `AGENTS.md` first. This file adds frontend-specific rules.

---

## 1. Frontend Operating Loop

For frontend changes, always work in this order:

1. Inspect existing extension IDs and widget contracts.
2. Identify whether code is modern hook/Vue-oriented or legacy LiteGraph.
3. Plan the smallest safe change.
4. Implement focused changes only.
5. Run lint/build/tests if available.
6. Verify in browser/ComfyUI when possible.
7. Check console for new errors.

If browser verification is not possible, state exactly what should be tested manually.

---

## 2. Frontend Direction

Use official ComfyUI frontend extension APIs.

Expected patterns:

- Frontend files live under the configured `WEB_DIRECTORY`.
- Extensions are registered with `app.registerExtension`.
- Extension IDs are stable and unique.
- Vue-oriented APIs and documented hooks are preferred for modern UI behavior.
- Legacy LiteGraph code is maintenance-only unless explicitly required.

Do not rely on private ComfyUI internals.

---

## 3. Stable Frontend Identity

Never change existing:

- JS extension ID.
- Widget IDs.
- Setting IDs.
- Saved config keys.
- DOM data attributes used by saved state.
- Node class names referenced by frontend code.
- Backend node IDs referenced by frontend code.

If a rename is necessary:

- Keep compatibility aliases where possible.
- Support old saved keys.
- Add migration logic.
- Document the migration.
- Add tests if possible.

---

## 4. File and Module Style

Use:

- ES2020+ syntax.
- ES modules where possible.
- Small focused modules.
- Clear lifecycle functions.
- Explicit constants.
- Stable naming.

File naming:

```text
ts-module-name.js
```

Extension ID naming:

```text
ts.moduleName
```

Class naming:

```text
TsClassName
```

Avoid:

- One giant JS file.
- Hidden global state.
- Unclear side effects during import.
- Duplicated UI logic.
- Copy-pasted event handlers.

---

## 5. Official API Usage

Preferred:

```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "ts.exampleExtension",
  async setup() {
    // setup code
  },
});
```

Rules:

- Use documented hooks.
- Keep extension lifecycle predictable.
- Check node class names before applying node-specific logic.
- Fail gracefully when a backend node is not present.
- Do not assume load order unless documented.
- Do not assume internal DOM structure is stable.

---

## 6. Monkey-Patching Policy

Forbidden by default:

- Modifying `LGraphCanvas.prototype`.
- Modifying `LGraphNode.prototype`.
- Modifying ComfyUI app prototypes.
- Modifying browser built-in prototypes.
- Overriding core methods without an official hook.

Allowed only for legacy maintenance:

- A small isolated compatibility shim.
- Clear comments explaining why official hooks are not enough.
- Strict guards around ComfyUI version or feature detection.
- No broad side effects.

If monkey-patching is found during review, mark it as high-risk and suggest a hook-based replacement.

---

## 7. DOM and Vue Rules

Modern ComfyUI frontend is moving toward documented extension hooks and Vue-based UI patterns.

Rules:

- Prefer official extension points over direct DOM mutation.
- Keep DOM queries scoped.
- Avoid brittle selectors.
- Avoid polling loops.
- Clean up event listeners.
- Clean up observers.
- Avoid memory leaks in long-running sessions.
- Store UI state explicitly.

If direct DOM manipulation is necessary:

- Keep it minimal.
- Add comments.
- Feature-detect target elements.
- Fail silently but log a debug/warning message when useful.
- Do not break ComfyUI if the element is missing.

---

## 8. Backend/Frontend Contract

Frontend code must not invent backend contracts.

When JS references backend nodes:

- Use stable backend node IDs.
- Confirm node class names.
- Do not depend on display names.
- Do not depend on category names for logic.
- Keep widget name references synchronized with backend schema.
- Preserve old widget names where possible.

When backend schema changes:

- Update JS deliberately.
- Add compatibility for old workflows.
- Document the change.
- Add tests if tooling exists.

---

## 9. Browser Verification

Frontend work needs browser feedback.

Verify when possible:

- ComfyUI loads.
- Browser console has no new errors.
- Extension initializes once.
- Workflow reload does not duplicate handlers.
- Target node UI works.
- Old workflows still open.
- Settings persist correctly.
- Missing backend node degrades gracefully.
- UI does not block large workflows.

If browser automation or manual browser access is unavailable, state the exact manual steps required.

---

## 10. Error Handling

Good frontend errors:

- Explain what failed.
- Identify the module.
- Avoid breaking the entire ComfyUI UI.
- Avoid alert spam.
- Avoid leaking secrets.
- Degrade gracefully.

Use:

```javascript
console.warn("[TS ModuleName] message");
console.error("[TS ModuleName] message", error);
```

Avoid:

- Silent catch blocks.
- Console spam in normal operation.
- Throwing from lifecycle hooks unless the extension truly cannot function.
- User-facing popups for developer-only issues.

---

## 11. State Management

Rules:

- Avoid global mutable state.
- Use closure/module state only when necessary.
- Keep state small.
- Reset/cleanup state when nodes are removed or workflows change.
- Do not store large binary data in frontend state.
- Do not store secrets in localStorage/sessionStorage.
- Version saved state when schema may evolve.

---

## 12. Performance Rules

ComfyUI workflows can contain many nodes.

Avoid:

- Expensive DOM scans on every frame.
- Per-node polling loops.
- Large synchronous work during UI interactions.
- Blocking main thread with heavy parsing.
- Repeated image/video decoding.
- Unbounded caches.

Prefer:

- Event-based updates.
- Debouncing.
- Throttling.
- Lazy UI construction.
- WeakMap for node-associated metadata.
- Cleanup on node removal.

---

## 13. Security Rules

Do not:

- Inject unsanitized HTML.
- Evaluate user strings as code.
- Use `new Function`.
- Store tokens in frontend code.
- Expose API keys.
- Fetch arbitrary remote URLs without validation.
- Trust metadata from workflow files.

If rendering user-provided text, use `textContent` or safe rendering.

---

## 14. JS Tooling Checks

If JS tooling exists, run:

```bash
npm run lint
npm run test
npm run build
```

If the repo uses another package manager, use the existing project convention.

If no tooling exists, at least manually verify:

- ComfyUI starts.
- Browser console has no new errors.
- Extension loads.
- Target node behavior works.
- Old workflows still open.
- No duplicate event handlers are created after reload.

---

## 15. Frontend Self-Review Checklist

Before finalizing frontend code, check:

- Extension ID unchanged.
- Widget/config keys unchanged.
- No new private internals.
- No new monkey-patching.
- Event listeners are cleaned up.
- Observers are cleaned up.
- UI state is bounded.
- No secrets in frontend code.
- Browser verification performed or manual steps listed.
- Backend schema references remain correct.

---

## 16. Frontend Definition of Done

A frontend task is complete only when:

- Extension IDs are stable.
- Existing widget/config keys are preserved.
- No private internals are newly used.
- No monkey-patching is added unless explicitly justified.
- Event listeners and observers are cleaned up.
- Browser console is clean or manual verification is documented.
- Backend/frontend contract remains compatible.
- Relevant checks were run or limitations are stated.
