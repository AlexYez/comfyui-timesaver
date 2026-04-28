# js/AGENTS.md — ComfyUI Frontend Extension Rules

Эта папка содержит JavaScript/TypeScript frontend extensions для ComfyUI.

Codex всегда отвечает пользователю на русском языке. Код и идентификаторы могут быть на английском, но отчёты, объяснения, риски и результаты проверок — на русском.

Следуй root `AGENTS.md` сначала. Этот файл добавляет frontend-specific правила.

---

## 1. Frontend Operating Loop

For frontend changes:

1. Inspect extension IDs, widget IDs, setting IDs and backend node IDs.
2. Identify modern hook/Vue-oriented code or legacy LiteGraph.
3. Plan the smallest safe change.
4. Keep node-specific frontend in one `.js` file per node.
5. Run all available JS checks.
6. Run Playwright E2E if available.
7. Verify browser console when possible.

If browser verification is not possible, state exact manual steps.

---

## 2. One Node = One Optional JS File

If a node needs frontend behavior, it should have one corresponding `.js` file.

Preferred:

```text
nodes/ts_preview_tools.py
js/ts-preview-tools.js
```

Avoid:

```text
js/ts-preview-tools/menu.js
js/ts-preview-tools/widgets.js
js/ts-preview-tools/state.js
js/ts-preview-tools/events.js
```

Shared JS utilities are allowed only if reused by 2+ modules:

```text
js/shared/dom-utils.js
js/shared/widget-utils.js
```

---

## 3. Frontend Direction

Use official ComfyUI frontend extension APIs.

Expected patterns:

- Files live under configured `WEB_DIRECTORY`.
- Extensions registered with `app.registerExtension`.
- Extension IDs are stable and unique.
- Vue-oriented APIs and documented hooks are preferred.
- Legacy LiteGraph code is maintenance-only unless explicitly required.

Do not rely on private ComfyUI internals.

---

## 4. Stable Frontend Identity

Never change existing:

- JS extension ID.
- Widget IDs.
- Setting IDs.
- Saved config keys.
- DOM data attributes used by saved state.
- Node class names referenced by frontend code.
- Backend node IDs referenced by frontend code.

If rename is necessary, keep compatibility aliases, support old saved keys, add migration logic, docs and tests.

---

## 5. File and Module Style

Use:

- ES2020+.
- ES modules where possible.
- One focused node-specific JS file.
- Clear lifecycle functions.
- Explicit constants.

Avoid:

- One giant JS file for unrelated nodes.
- Many tiny files for one node.
- Hidden global state.
- Unclear import side effects.
- Duplicate UI logic.
- Copy-pasted handlers.

---

## 6. Official API Usage

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
- Keep lifecycle predictable.
- Check node class names before node-specific logic.
- Fail gracefully when backend node is absent.
- Do not assume internal DOM structure is stable.

---

## 7. Monkey-Patching Policy

Forbidden by default:

- `LGraphCanvas.prototype` changes.
- `LGraphNode.prototype` changes.
- ComfyUI app prototype changes.
- Browser built-in prototype changes.
- Core method overrides without official hook.

Allowed only for isolated legacy compatibility with clear comments, feature detection and strict guards.

---

## 8. DOM, Vue, State and Performance

Prefer official extension points over direct DOM mutation.

Rules:

- Scope DOM queries.
- Avoid brittle selectors and polling loops.
- Clean up event listeners and observers.
- Avoid memory leaks.
- Keep UI state bounded.
- Do not store secrets in localStorage/sessionStorage.
- Use debouncing/throttling for frequent UI events.
- Avoid blocking main thread in large workflows.

---

## 9. Backend/Frontend Contract

Frontend code must not invent backend contracts.

When JS references backend nodes:

- Use stable backend node IDs.
- Confirm node class names.
- Do not depend on display names or categories for logic.
- Keep widget names synchronized with backend schema.
- Preserve old widget names when possible.

---

## 10. Mandatory Browser and E2E Verification

Frontend work must be verified like a real user whenever possible.

If Playwright exists, run:

```bash
npm run test:e2e
```

If specific tests exist:

```bash
npx playwright test tests/e2e
npx playwright test tests/e2e/comfyui-load.spec.js
npx playwright test tests/e2e/node-search.spec.js
npx playwright test tests/e2e/workflow-open.spec.js
```

User-like checks should cover:

- ComfyUI opens at `COMFY_URL` or `http://127.0.0.1:8188`.
- Browser console has no new errors.
- Target node can be found/searchable.
- Node widgets can be changed.
- Old workflow JSON opens if fixtures exist.
- Reload does not duplicate handlers.
- Missing backend node degrades gracefully.

If ComfyUI is not running, say in Russian:

```text
Не проверено через браузер: ComfyUI не запущен на 127.0.0.1:8188.
Локально нужно запустить ComfyUI и выполнить npm run test:e2e.
```

---

## 11. Error Handling and Logging

Use:

```javascript
console.warn("[TS ModuleName] message");
console.error("[TS ModuleName] message", error);
```

Avoid silent catch blocks, console spam, secrets, and user popups for developer-only issues.

---

## 12. Security

Do not:

- Inject unsanitized HTML.
- Evaluate user strings as code.
- Use `new Function`.
- Store tokens in frontend code.
- Expose API keys.
- Fetch arbitrary remote URLs without validation.
- Trust metadata from workflow files.

Use `textContent` or safe rendering for user text.

---

## 13. Mandatory JS Checks

If JS tooling exists, run:

```bash
npm run lint
npm run test
npm run build
npm run test:e2e
```

If no tooling exists, manually verify or document:

- ComfyUI starts.
- Browser console has no new errors.
- Extension loads.
- Target node behavior works.
- Old workflows still open.
- Event handlers are not duplicated after reload.

---

## 14. Frontend Definition of Done

Frontend task is complete only when:

- Ответ пользователю на русском.
- Extension IDs stable.
- Widget/config keys preserved.
- One-node-one-JS-file preserved.
- No new private internals or monkey-patching.
- Listeners/observers cleaned up.
- Browser console clean or manual verification documented.
- Playwright E2E run when available.
- Backend/frontend contract compatible.
- All possible checks run or limitations stated.
