# tests/AGENTS.md — Testing and Quality Rules

This directory contains tests and quality checks for ComfyUI custom nodes.

Follow the root `AGENTS.md` first. This file adds test-specific rules.

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
7. Workflow smoke tests.

Do not write tests that only mirror implementation details.

---

## 2. Verification Loop

Every bug fix or feature should answer:

- How can this fail?
- Which test catches that failure?
- What command proves it works?
- Can the test run without CUDA, internet, or large models?
- If not, is it marked as integration/manual?

Never accept "looks correct" as the only verification.

---

## 3. Required Test Categories

### Import Tests

Verify Python modules can be imported without crashing.

Purpose:

- Catch syntax errors.
- Catch missing imports.
- Catch heavy import side effects.
- Catch circular imports.

### Contract Tests

Verify public node contracts remain stable:

- node ID
- class name
- category
- input names
- input types
- input defaults
- output names
- output types
- output order
- hidden inputs
- deprecation/experimental flags

### Tensor Tests

Verify:

- IMAGE input/output uses `[B, H, W, C]`.
- MASK input/output uses `[B, H, W]`.
- Batch dimension is preserved.
- dtype is expected.
- range is valid when applicable.
- input tensors are not mutated.

### Validation Tests

Verify:

- Valid inputs pass.
- Invalid inputs return clear messages.
- Missing optional dependencies are handled cleanly.

### Regression Tests

Every fixed bug should get a minimal regression test when practical.

---

## 4. Contract Snapshot Policy

If contract snapshots exist, they are the source of truth for workflow compatibility.

Preferred path:

```text
tests/contracts/node_contracts.json
```

A contract snapshot should include:

```json
{
  "TS_ExampleNode": {
    "api": "v3",
    "class_name": "TS_ExampleNode",
    "node_id": "TS_ExampleNode",
    "display_name": "TS Example Node",
    "category": "TS/examples",
    "inputs": {
      "image": {
        "type": "IMAGE",
        "optional": false
      },
      "strength": {
        "type": "FLOAT",
        "default": 1.0
      }
    },
    "outputs": [
      {
        "name": "image",
        "type": "IMAGE"
      }
    ]
  }
}
```

Rules:

- Do not update snapshots casually.
- Snapshot changes imply public contract changes.
- Public contract changes require explicit user approval.
- Migration code must be tested.

---

## 5. Test Style

Use:

- `pytest`.
- Small focused tests.
- Clear test names.
- Temporary directories for file I/O.
- Deterministic data.
- CPU-safe tensors.
- Minimal synthetic tensors.
- Clear assertion messages for contract failures.

Avoid:

- Downloading models.
- Requiring CUDA.
- Requiring internet.
- Depending on the user's real ComfyUI output folder.
- Slow tests in the default suite.
- Tests that pass only because of execution order.

---

## 6. Suggested Python Commands

Run after backend changes:

```bash
python -m compileall .
python -m pytest tests
```

Run contract tests:

```bash
python -m pytest tests/test_node_contracts.py
```

Run tensor tests:

```bash
python -m pytest tests/test_tensor_shapes.py
```

Run one regression test:

```bash
python -m pytest tests/test_regressions.py -k "bug_name"
```

---

## 7. Suggested JS Commands

If frontend tooling exists:

```bash
npm run lint
npm run test
npm run build
```

If no tooling exists:

- Manually open ComfyUI.
- Confirm the extension loads.
- Confirm browser console has no new errors.
- Confirm old workflows still open.
- Confirm event handlers are not duplicated after reload.

---

## 8. Mocking and ComfyUI Imports

ComfyUI internals may be difficult to import in isolated tests.

Preferred order:

1. Test pure utilities directly.
2. Test schema contracts with lightweight imports.
3. Mock ComfyUI APIs only when necessary.
4. Keep mocks close to the real public API.
5. Do not over-mock behavior that matters.

If a test needs a real ComfyUI runtime, mark it as an integration/smoke test and do not require it in fast unit tests.

---

## 9. Tensor Test Examples

Use tiny synthetic tensors.

Example IMAGE:

```python
image = torch.zeros((2, 16, 16, 3), dtype=torch.float32)
```

Example MASK:

```python
mask = torch.zeros((2, 16, 16), dtype=torch.float32)
```

Check no mutation:

```python
before = image.clone()
result = run_node(image)
assert torch.equal(image, before)
```

Check batch preservation:

```python
assert result.shape[0] == image.shape[0]
```

---

## 10. Workflow Compatibility Smoke Tests

When possible, keep minimal workflow JSON files for critical nodes.

Preferred path:

```text
tests/workflows/
```

Smoke checks should verify:

- Workflow JSON loads.
- Old node IDs still resolve.
- Connections remain valid.
- Output count is unchanged.
- Optional migration works.

Do not store huge assets in tests.

---

## 11. Review-Oriented Tests

For high-risk changes, create tests before refactoring.

High-risk examples:

- V1 to V3 migration.
- Node ID or schema changes.
- Tensor shape transformations.
- Cache/fingerprint logic.
- File path handling.
- Frontend widget persistence.
- Optional dependency fallback.

The test should fail before the fix and pass after the fix whenever practical.

---

## 12. Test Self-Review Checklist

Before finalizing tests, check:

- Does this protect public behavior?
- Does this avoid testing private implementation details?
- Is it deterministic?
- Is it CPU-safe?
- Does it avoid internet/model downloads?
- Does it use temporary directories for file I/O?
- Does it include clear assertion messages?
- Would it catch the original bug?
- Does it run in the default test command?

---

## 13. Test Definition of Done

A test change is complete only when:

- Tests protect public behavior, not private implementation details.
- No internet/model download is required.
- Tests are deterministic.
- Tests are CPU-safe unless marked otherwise.
- Contract snapshots are updated only intentionally.
- Commands are documented.
- Failing tests provide actionable messages.
