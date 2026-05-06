# tests/AGENTS.md — Testing and Quality Rules

Эта папка содержит тесты и quality checks для ComfyUI custom nodes.

Codex всегда отвечает пользователю на русском языке. Названия тестов и код могут быть на английском, но отчёты, объяснения и результаты проверок — на русском.

Следуй root `AGENTS.md` сначала.

---

## 1. Testing Mission

Tests must protect users from broken workflows.

Highest-value tests:

1. Node contract tests.
2. Import/syntax tests.
3. Tensor shape/range tests.
4. Validation tests.
5. Regression tests for fixed bugs.
6. Frontend contract/lint tests.
7. Playwright E2E tests that act like a user.
8. Workflow smoke tests.

Do not write tests that only mirror implementation details.

---

## 2. Mandatory Verification Loop

Every bug fix or feature must answer:

- How can this fail?
- Which test catches that failure?
- What command proves it works?
- Can the test run without CUDA, internet, or large models?
- If frontend changed, can Playwright verify it like a user?
- If not, is it marked as integration/manual?

Never accept "looks correct" as the only verification.

---

## 3. Required Test Categories

### Import Tests

Verify Python modules import without crashing.

### Contract Tests

Verify public node contracts remain stable:

- node ID
- class name
- main Python file path
- optional JS file path
- category
- input names/types/defaults
- output names/types/order
- hidden inputs
- deprecation/experimental flags

### Tensor Tests

Verify:

- IMAGE `[B, H, W, C]`
- MASK `[B, H, W]`
- batch dimension preserved
- dtype/range expected
- input tensors are not mutated

### Validation Tests

Valid inputs pass, invalid inputs return clear messages, missing optional dependencies are graceful.

### Regression Tests

Every fixed bug should get a minimal regression test when practical.

### Playwright E2E Tests

Frontend behavior must be tested like a user when possible:

- ComfyUI opens.
- Browser console has no new errors.
- Target node can be searched/found.
- Widgets can be changed.
- Old workflow fixtures open.
- Reload does not duplicate handlers.

---

## 4. Contract Snapshot Policy

Preferred path:

```text
tests/contracts/node_contracts.json
```

Snapshot should include:

```json
{
  "TS_ExampleNode": {
    "api": "v3",
    "class_name": "TS_ExampleNode",
    "python_file": "nodes/ts_example_node.py",
    "js_file": "js/ts-example-node.js",
    "node_id": "TS_ExampleNode",
    "display_name": "TS Example Node",
    "category": "TS/examples",
    "inputs": {
      "image": {"type": "IMAGE", "optional": false},
      "strength": {"type": "FLOAT", "default": 1.0}
    },
    "outputs": [{"name": "image", "type": "IMAGE"}]
  }
}
```

Do not update snapshots casually. Snapshot changes imply public contract changes and require explicit approval.

---

## 5. Test Style

Use:

- `pytest`
- small focused tests
- temporary directories for file I/O
- deterministic CPU-safe tensors
- clear assertion messages
- Playwright for user-like frontend checks

Avoid internet, model downloads, CUDA requirement, real user output folders, slow default tests, and implementation-detail-only tests.

---

## 6. Mandatory Commands

Backend:

```bash
python -m compileall .
python -m pytest tests
```

If available:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy .
python -m pytest tests/test_node_contracts.py
python -m pytest tests/test_tensor_shapes.py
```

Frontend:

```bash
npm run lint
npm run test
npm run build
npm run test:e2e
```

Playwright examples:

```bash
npx playwright test tests/e2e
npx playwright test tests/e2e/comfyui-load.spec.js
npx playwright test tests/e2e/node-search.spec.js
npx playwright test tests/e2e/workflow-open.spec.js
```

If no tooling exists, document manual checks.

---

## 7. Tensor Test Examples

```python
image = torch.zeros((2, 16, 16, 3), dtype=torch.float32)
mask = torch.zeros((2, 16, 16), dtype=torch.float32)

before = image.clone()
result = run_node(image)
assert torch.equal(image, before)
assert result.shape[0] == image.shape[0]
```

---

## 8. Playwright Test Expectations

Preferred path:

```text
tests/e2e/
  comfyui-load.spec.js
  node-search.spec.js
  workflow-open.spec.js
  widget-interaction.spec.js
```

Tests should collect `page.on("console")` errors, collect `page.on("pageerror")` errors, open `COMFY_URL` or `http://127.0.0.1:8188`, search for target node, interact with widgets, load fixture workflow if available, and assert no browser errors.

Avoid fragile selectors if accessible roles/text can be used.

---

## 9. Workflow Smoke Tests

Preferred path:

```text
tests/workflows/
```

Smoke checks should verify workflow JSON loads, old node IDs resolve, connections remain valid, output count is unchanged, and migration works.

Do not store huge assets in tests.

---

## 10. Test Definition of Done

A test change is complete only when:

- Ответ пользователю на русском.
- Tests protect public behavior.
- No internet/model download required.
- Tests are deterministic and CPU-safe unless marked otherwise.
- Frontend UI changes have Playwright/E2E coverage when available.
- Contract snapshots are updated only intentionally.
- Commands are documented.
- Failing tests provide actionable messages.
- One-node-one-file contract is protected where appropriate.
