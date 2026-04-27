# nodes/AGENTS.md — Python Backend Node Rules

This directory contains Python backend code for ComfyUI custom nodes.

Follow the root `AGENTS.md` first. This file adds backend-specific rules.

---

## 1. Backend Operating Loop

For backend changes, always work in this order:

1. Inspect existing node contracts.
2. Identify V1 or V3 API.
3. Plan the smallest safe change.
4. Implement only the requested change.
5. Run import/contract/tensor checks where possible.
6. Self-review the diff for workflow breakage.

If the implementation reveals new architectural risk, stop and re-plan.

---

## 2. Default Backend Direction

New nodes must use the ComfyUI V3 Node API unless the task explicitly says otherwise.

Preferred development import:

```python
from comfy_api.latest import ComfyExtension, io, ui
```

Use a pinned version such as `comfy_api.v0_0_2` only when the repository has deliberately standardized on it for release reproducibility.

V1 nodes are legacy maintenance only.

Do not create new V1 nodes unless explicitly requested.

Do not migrate V1 nodes to V3 unless explicitly requested.

---

## 3. V3 Node Structure

A V3 node must:

- Inherit from `io.ComfyNode`.
- Implement `define_schema(cls)` as `@classmethod`.
- Implement `execute(cls, ...)` as `@classmethod`.
- Return `io.NodeOutput`.
- Avoid `__init__`.
- Avoid instance state.
- Avoid mutable class state unless it is a deliberate, documented cache.
- Use official ComfyUI APIs only.

Example:

```python
from comfy_api.latest import ComfyExtension, io, ui


class TS_ExampleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TS_ExampleNode",
            display_name="TS Example Node",
            category="TS/examples",
            description="Short user-facing description.",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, image, strength: float) -> io.NodeOutput:
        result = image.clone()
        return io.NodeOutput(result)


class TsBackendExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TS_ExampleNode]


async def comfy_entrypoint() -> TsBackendExtension:
    return TsBackendExtension()
```

---

## 4. Stable Node Identity

Never change existing:

- Python class name.
- V3 `node_id`.
- V1 `NODE_CLASS_MAPPINGS` key.
- Input names.
- Output names.
- Output order.
- Output types.
- Default widget values.
- Category.
- `execute()` parameter names.
- Hidden input semantics.
- Cache/fingerprint semantics.

If a node must be renamed or restructured:

- Preserve old node ID if possible.
- Add `search_aliases`.
- Use `io.NodeReplace` through `ComfyAPI`.
- Keep old aliases where practical.
- Add contract tests.
- Add migration notes.

---

## 5. V3 Schema Rules

Schema must be explicit and stable.

Required schema quality:

- `node_id` is globally unique and prefixed.
- `display_name` is human-readable.
- `category` is stable.
- `description` is useful and concise.
- Inputs have clear names and defaults.
- Advanced inputs are marked with `advanced=True` when appropriate.
- Optional inputs are truly optional.
- Outputs have stable display names.
- Deprecated nodes use `is_deprecated=True`.
- Experimental nodes use `is_experimental=True`.

Avoid:

- Ambiguous input names like `value`, `data`, `x`, unless context is obvious.
- Hidden behavior based on magic strings.
- Unbounded values without reason.
- Changing defaults to "improve" results.

---

## 6. V1 Legacy Maintenance

For V1 nodes, preserve:

- `INPUT_TYPES`.
- `RETURN_TYPES`.
- `RETURN_NAMES`.
- `FUNCTION`.
- `CATEGORY`.
- `OUTPUT_NODE`.
- `NODE_CLASS_MAPPINGS`.
- `NODE_DISPLAY_NAME_MAPPINGS`.
- Input names.
- Widget defaults.
- Output order and types.

Safe V1 changes:

- Fix clear bugs without changing public contract.
- Add internal helper functions.
- Add validation only if it does not reject previously valid workflows unexpectedly.
- Improve logging.
- Improve error messages.
- Add compatibility aliases.

Unsafe V1 changes:

- Changing mappings.
- Changing input names.
- Changing output order.
- Changing defaults.
- Changing function names.
- Removing old mappings.
- Moving to V3 without explicit migration request.

---

## 7. Tensor Format and Batch Handling

ComfyUI conventions:

```text
IMAGE  -> [B, H, W, C], float32, range [0, 1]
MASK   -> [B, H, W],    float32, range [0, 1]
LATENT -> latent["samples"]
```

Rules:

- Always check `image.ndim == 4` for IMAGE when relevant.
- Always check `mask.ndim == 3` for MASK when relevant.
- Always preserve batch dimension.
- Never silently process only the first item in a batch.
- Never mutate input tensors directly.
- Clone before modification.
- Preserve device unless conversion is necessary.
- Preserve dtype where possible.
- Clamp outputs to valid range only when the operation can exceed range.
- Document shape-changing behavior.

Good:

```python
result = image.clone()
result = result.clamp(0.0, 1.0)
```

Bad:

```python
image *= 2
return io.NodeOutput(image[0])
```

---

## 8. Validation Rules

Use `validate_inputs` for user-facing validation.

```python
@classmethod
def validate_inputs(cls, **kwargs) -> bool | str:
    if kwargs["strength"] < 0:
        return "strength must be greater than or equal to 0."
    return True
```

Rules:

- Return `True` or a clear error string.
- Mention the exact input name.
- Do not leak secrets.
- Do not expose unnecessary full user paths.
- Do not perform expensive model loading in validation.
- Do not mutate inputs during validation.

---

## 9. Cache and Fingerprint Rules

Use `fingerprint_inputs` when node output depends on external state.

```python
@classmethod
def fingerprint_inputs(cls, **kwargs):
    return meaningful_hash
```

Rules:

- Never return a constant unless permanent caching is intentional.
- Include file modification time/hash when output depends on a file.
- Include config version when output depends on config.
- Avoid hashing huge tensors manually.
- Keep fingerprint deterministic.

Common mistakes:

- Returning `True` means the node may run once and then reuse cache forever.
- Returning random values forces reruns and breaks caching.
- Ignoring external files can return stale outputs.

---

## 10. Lazy Evaluation Rules

Use `check_lazy_status` only when needed.

Rules:

- Request optional/lazy inputs only when they are actually needed.
- Keep logic simple and deterministic.
- Do not perform heavy work in `check_lazy_status`.
- Return input names, not display names.

---

## 11. Hidden Inputs

Declare hidden inputs explicitly in V3 schema when needed.

Common hidden values:

- `io.Hidden.unique_id`
- `io.Hidden.prompt`
- `io.Hidden.extra_pnginfo`
- `io.Hidden.dynprompt`

Rules:

- Access hidden values through `cls.hidden`.
- Do not rely on hidden inputs that are not declared.
- Do not serialize private prompt data unnecessarily.
- Do not log hidden prompt data unless required and safe.

---

## 12. UI Helpers

Use official `ui` helpers:

- `ui.PreviewImage(images, cls=cls)`
- `ui.PreviewMask(mask, cls=cls)`
- `ui.PreviewAudio(audio, cls=cls)`
- `ui.PreviewText("text")`
- `ui.ImageSaveHelper.get_save_images_ui(...)`
- `ui.AudioSaveHelper.get_save_audio_ui(...)`

Rules:

- Pass `cls=cls` where required to preserve metadata.
- Do not invent custom UI dicts if a helper exists.
- Keep preview generation lightweight.
- Do not save files outside ComfyUI-approved locations.

---

## 13. Model Loading and Heavy Dependencies

Rules:

- Do not load models at import time.
- Do not import torch-heavy or model-heavy libraries at import time unless unavoidable.
- Load lazily inside the node operation.
- Use ComfyUI folder/model path APIs.
- Respect ComfyUI model management.
- Provide clear missing-model messages.
- Avoid permanent references to large tensors.
- Avoid duplicated model instances.

No auto-downloads unless the node explicitly documents and validates this behavior.

No package installation from node code.

---

## 14. File I/O and Security

Use `pathlib.Path`.

Validate:

- Existence.
- File extension.
- Directory boundary.
- Write permissions.
- Filename safety.

Avoid:

- Path traversal.
- Writing outside configured output/cache folders.
- Opening arbitrary user-provided paths without validation.
- Extracting archives without safe path checks.
- Logging secrets or tokens.

Never use shell commands for normal file operations.

---

## 15. Backend Verification

After backend changes, run:

```bash
python -m compileall .
python -m pytest tests
```

If contract tests exist:

```bash
python -m pytest tests/test_node_contracts.py
```

If tensor tests exist:

```bash
python -m pytest tests/test_tensor_shapes.py
```

For new nodes, add or update tests for:

- Schema contract.
- Input validation.
- Batch tensor behavior.
- Output shape.
- Output dtype/range.
- Missing optional dependency behavior.
- Error paths.

If the runtime cannot execute tests, list what should be run locally.

---

## 16. Backend Self-Review Checklist

Before finalizing backend code, check:

- No public node identity changed.
- No input/output names changed.
- No output order/types changed.
- No defaults changed.
- No category changed.
- No input tensors mutated.
- Batch dimension preserved.
- Optional dependencies are lazy and graceful.
- No heavy model load at import time.
- No unsafe path/shell behavior.
- Logging uses `logging`, not `print`.
- Import path is portable.
- Verification was performed or limitations are stated.

---

## 17. Backend Definition of Done

A backend task is complete only when:

- Node identity is preserved or migration is provided.
- Code imports without syntax errors.
- Inputs/outputs/defaults are correct.
- Tensor batch handling is correct.
- No input mutation occurs.
- Logging uses `logging`.
- Optional dependencies are graceful.
- No unsafe path/shell behavior is introduced.
- Relevant tests/checks are run or limitations are stated.
