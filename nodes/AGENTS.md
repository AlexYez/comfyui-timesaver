# nodes/AGENTS.md — Python Backend Node Rules

Эта папка содержит Python backend-код для ComfyUI custom nodes.

Codex всегда отвечает пользователю на русском языке. Код и идентификаторы могут быть на английском, но отчёты, объяснения, риски и результаты проверок — на русском.

Следуй root `AGENTS.md` сначала. Этот файл добавляет backend-specific правила.

---

## 1. Backend Operating Loop

Всегда:

1. Изучи существующие node contracts.
2. Определи V1 или V3 API.
3. Составь минимальный безопасный план.
4. Сохрани структуру: одна публичная нода = один основной `.py` файл.
5. Реализуй только запрошенное.
6. Запусти все возможные backend-проверки.
7. Перечитай diff на предмет workflow breakage.

Если обнаружился новый архитектурный риск — остановись и перепланируй.

---

## 2. One Node = One Python File

Каждая публичная ComfyUI-нода должна иметь один основной `.py` файл.

Preferred:

```text
nodes/ts_resize_image.py
nodes/ts_audio_preview.py
nodes/ts_video_metadata.py
```

Avoid:

```text
nodes/ts_resize_image/schema.py
nodes/ts_resize_image/execute.py
nodes/ts_resize_image/validation.py
nodes/ts_resize_image/types.py
```

Shared utilities разрешены только если логика используется 2+ нодами:

```text
utils/image_ops.py
utils/path_utils.py
utils/tensor_checks.py
```

Правило:

- Простая нода — один самодостаточный `.py` файл.
- Средняя нода — один `.py` файл + private helpers внутри него.
- Повторяемая логика — вынести только reusable часть в `utils/`.
- Не дробить ноду ради “чистой архитектуры”.

### Big-node exception: `nodes/<категория>/<feature>/`

Если нода стала god-file (~1000+ строк) **или** у неё значительный набор приватных helpers, которые не нужны другим нодам (HTTP routes, DOM widget logic, voice/audio pipelines, model loaders), допустимо вынести её в подпапку:

```text
nodes/llm/super_prompt/
├── __init__.py            # пустой
├── ts_super_prompt.py     # единственный публичный entry: класс + schema + execute
├── _voice.py              # приватный helper (Whisper/voice pipeline)
├── _qwen.py               # приватный helper (Qwen prompt enhance)
├── _routes.py             # приватный helper (aiohttp routes)
└── _helpers.py            # приватный helper (logger, prompt_server, common)
```

- Публичный класс ноды — **только в `ts_<name>.py`**, не в `_`-prefixed файлах. Loader игнорирует любой путь с `_`-prefixed компонентом.
- В одной подпапке могут жить несколько публичных нод (как `nodes/audio/loader/` с `TS_AudioLoader` + `TS_AudioPreview`).
- Не дробить одну ноду на `schema.py + execute.py + types.py` — публичный класс остаётся одним файлом.
- Подпапка разрешена только если есть реальная причина (god-file, обширные приватные helpers). Для простой ноды overhead не оправдан.

Существующие примеры: `nodes/image/lama_cleanup/`, `nodes/audio/loader/`, `nodes/image/keying/`. Следуй им.

---

## 3. Default Backend Direction

New nodes must use ComfyUI V3 Node API unless explicitly requested otherwise.

Preferred import:

```python
from comfy_api.latest import ComfyExtension, io, ui
```

Use pinned `comfy_api.v0_0_2` only when the project release target requires it.

V1 nodes are legacy maintenance only. Do not create new V1 nodes and do not migrate V1 to V3 without explicit request.

---

## 4. V3 Node Structure

A V3 node must:

- Inherit from `io.ComfyNode`.
- Implement `define_schema(cls)` as `@classmethod`.
- Implement `execute(cls, ...)` as `@classmethod`.
- Return `io.NodeOutput`.
- Avoid `__init__`.
- Avoid instance state.
- Avoid mutable class state unless it is a deliberate documented cache.
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
            outputs=[io.Image.Output(display_name="image")],
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

## 5. Stable Node Identity

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

If a node must be renamed/restructured:

- Preserve old node ID if possible.
- Add `search_aliases`.
- Use `io.NodeReplace` through `ComfyAPI`.
- Keep aliases where practical.
- Add contract tests and migration notes.

---

## 6. Schema Quality

Schema must be explicit and stable:

- globally unique prefixed `node_id`
- human-readable `display_name`
- stable `category`
- concise `description`
- clear input names/defaults
- `advanced=True` for advanced inputs
- stable output display names
- `is_deprecated=True` for deprecated nodes
- `is_experimental=True` for experimental nodes

Avoid ambiguous input names, hidden magic strings, unbounded values without reason, and default changes that alter existing workflows.

---

## 7. V1 Legacy Maintenance

For V1 nodes preserve:

- `INPUT_TYPES`
- `RETURN_TYPES`
- `RETURN_NAMES`
- `FUNCTION`
- `CATEGORY`
- `OUTPUT_NODE`
- `NODE_CLASS_MAPPINGS`
- `NODE_DISPLAY_NAME_MAPPINGS`
- input names, defaults, output order/types

Safe V1 changes: internal helpers in same file, logging, error messages, critical bug fixes without public contract change.

Unsafe: mapping/name/default/output changes or V1→V3 migration without explicit request.

---

## 8. Tensor and Batch Rules

ComfyUI conventions:

```text
IMAGE  -> [B, H, W, C], float32, range [0, 1]
MASK   -> [B, H, W],    float32, range [0, 1]
LATENT -> latent["samples"]
```

Rules:

- Check `image.ndim == 4` when relevant.
- Check `mask.ndim == 3` when relevant.
- Preserve batch dimension.
- Never process only the first batch item silently.
- Clone before modification.
- Preserve dtype/device where possible.
- Clamp only when operation may exceed valid range.
- Document shape-changing behavior.

Bad:

```python
image *= 2
return io.NodeOutput(image[0])
```

---

## 9. Validation, Fingerprint, Lazy Inputs

Use `validate_inputs` for user-facing validation. Return `True` or a clear error string.

Use `fingerprint_inputs` when output depends on external state. Never return a constant unless permanent caching is intentional.

Use `check_lazy_status` only when needed. Do not do heavy work there.

Declare hidden inputs explicitly and access them through `cls.hidden`.

---

## 10. Model Loading, File I/O, Security

- Do not load models at import time.
- Use lazy imports for heavy dependencies.
- Use ComfyUI folder/model APIs.
- Respect ComfyUI model management.
- No auto-downloads unless explicitly documented and validated.
- No package installation from node code.
- Use `pathlib.Path`.
- Validate paths, extensions, directory boundaries and write permissions.
- Avoid path traversal and arbitrary writes.
- Never use shell commands for normal file operations.

---

## 11. Mandatory Backend Verification

After backend changes, run every available relevant check.

Minimum:

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
python -m pytest tests/test_workflows.py
```

For new nodes, add/update tests for schema contract, validation, batch behavior, output shape, dtype/range, missing optional dependency, error paths, and no input tensor mutation.

If tests cannot run, list exact local commands.

---

## 12. Backend Definition of Done

Backend task is complete only when:

- Ответ пользователю на русском.
- Node identity preserved or migration provided.
- Code imports without syntax errors.
- Inputs/outputs/defaults correct.
- Tensor batch handling correct.
- Input tensors not mutated.
- Logging uses `logging`.
- Optional dependencies fail gracefully.
- No unsafe path/shell behavior.
- One-node-one-file preserved.
- All possible relevant checks run or limitations stated.
