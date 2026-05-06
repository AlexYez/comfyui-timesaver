# AGENTS.md — ComfyUI Custom Nodes Engineering Rules

Этот репозиторий содержит production-quality custom nodes и frontend extensions для ComfyUI.

Codex должен всегда отвечать пользователю на русском языке. Код, имена файлов, классы, API, docstrings и technical comments могут быть на английском, если это лучше для проекта, но планы, отчёты, риски, объяснения, результаты тестов и инструкции пользователю — всегда на русском.

Главная цель: максимальное качество итогового кода без поломки старых ComfyUI workflows.

Кредо проекта:

> Readable. Stable. Testable.

ComfyUI-кредо:

> Stability. Identity. Scalability. Predictability.

Принцип рефакторинга:

> Primum non nocere — first, do no harm.

---

## 0. Instruction Hierarchy

Соблюдай этот root-файл и ближайший локальный `AGENTS.md`:

```text
nodes/AGENTS.md
js/AGENTS.md
doc/AGENTS.md
tests/AGENTS.md
```

Локальные инструкции могут уточнять правила, но не могут ослаблять совместимость, безопасность, тестирование и стабильность node contracts.

---

## 1. Language Rules

Обязательно:

- Всегда отвечай пользователю на русском.
- Все планы, пояснения, отчёты, риски и verification summary пиши на русском.
- Не заявляй, что задача завершена, если проверки не запускались и это не объяснено.
- Если тесты невозможно запустить, честно укажи причину и дай точные команды для локальной проверки.
- Для сложных задач сначала дай короткий план.
- Для рефакторинга всегда пиши: анализ, риски, изменения, проверка.

Разрешено:

- Писать код, API, имена переменных, классов, файлов и docstrings на английском.
- Использовать английские технические идентификаторы ComfyUI/Python/JS.

---

## 2. Operating Loop: Plan → Implement → Verify → Review

Для любой нетривиальной задачи:

1. **Plan**
   - Изучи релевантные файлы.
   - Определи API нод (после 8.9 — всегда V3).
   - Выпиши публичные контракты: node_id, inputs, outputs, defaults, category, frontend IDs.
   - До редактирования определи стратегию тестирования.
   - Выбери минимальное безопасное изменение.

2. **Implement**
   - Делай только запрошенное.
   - Не добавляй unrelated features.
   - Не переписывай рабочий код ради архитектурной красоты.
   - Сохраняй правило: одна нода = один основной `.py` файл + один `.js` файл только при необходимости.

3. **Verify**
   - Запусти все доступные релевантные проверки.
   - Если тест падает — исправь минимально и повтори.
   - Если проверка невозможна — явно напиши почему.

4. **Review**
   - Перечитай diff.
   - Проверь workflow compatibility.
   - Проверь безопасность, hidden side effects, dead code, private API, tensor mutation.

Если план оказался неверным — остановись и перепланируй.

---

## 3. Mandatory Maximum Verification Policy

Codex должен запускать все возможные релевантные тесты перед заявлением о завершении.

Python backend minimum:

```bash
python -m compileall .
python -m pytest tests
```

Python quality checks, если доступны:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy .
```

Frontend checks, если доступны:

```bash
npm run lint
npm run test
npm run build
```

Frontend E2E / browser checks, если доступны:

```bash
npm run test:e2e
npx playwright test tests/e2e
```

Verification summary в ответе обязателен:

```text
Проверено:
- python -m compileall .
- python -m pytest tests

Не проверено:
- npm run test:e2e — ComfyUI не запущен на 127.0.0.1:8188.
```

Нельзя писать "готово" только потому, что код выглядит правильным.

---

## 4. One Node = One Python File, One Optional JS File

Правило структуры:

> Одна публичная ComfyUI-нода = один основной `.py` файл. Если ноде нужен frontend, она может иметь один соответствующий `.js` файл.

Preferred:

```text
nodes/ts_example_node.py
js/ts-example-node.js
```

Не дроби одну ноду на кучу файлов:

```text
nodes/ts_example_node/schema.py
nodes/ts_example_node/execute.py
nodes/ts_example_node/validation.py
nodes/ts_example_node/types.py
```

Shared utilities разрешены только если логика используется 2+ нодами:

```text
utils/image_ops.py
utils/path_utils.py
utils/tensor_checks.py
```

Для простой/средней ноды предпочитай один самодостаточный файл с приватными helper-функциями внутри. Архитектура не должна ухудшать дебаг.

---

## 5. Project Layout

```text
/
├─ AGENTS.md
├─ nodes/                 # one .py file per public node
├─ js/                    # one optional .js file per node needing frontend
├─ configs/               # user-editable config
├─ utils/                 # shared reusable utilities only
├─ docs/                  # user/developer docs
├─ tests/                 # unit, contract, smoke, regression, e2e
├─ tools/                 # dev scripts
├─ .codex/                # optional Codex skills/config
├─ pyproject.toml
├─ package.json           # only if JS tooling exists
└─ README.md
```

Do not create one massive file with unrelated nodes. Do not create many tiny files for one node.

---

## 6. Non-Negotiable Workflow Compatibility Rules

Never change existing values unless the user explicitly requests a breaking change or migration:

- Python node class names.
- V3 `node_id`.
- V1 `NODE_CLASS_MAPPINGS` keys.
- JS extension IDs.
- Input names.
- Output names.
- Output order.
- Output types.
- `execute()` parameter names.
- Default widget values.
- `CATEGORY` values.
- Saved configuration keys.
- Widget IDs used by frontend code.
- Existing semantics.

If rename/migration is needed:

- Keep old node ID where possible.
- Use V3 `search_aliases`.
- Use `io.NodeReplace` via `ComfyAPI`.
- Keep V1 aliases in mappings.
- Add docs migration note.
- Add/update contract tests.

---

## 7. API Strategy: V3 only

Since `8.9` the whole pack is on V3 — `grep RETURN_TYPES nodes/` returns nothing, all 57 nodes use `IO.ComfyNode + define_schema + execute`.

All nodes:

- ComfyUI V3 schema.
- `from comfy_api.latest import IO`.
- Class inherits from `IO.ComfyNode`.
- `define_schema(cls) -> IO.Schema(...)` (`@classmethod`).
- `execute(cls, ...) -> IO.NodeOutput(...)` (`@classmethod`).
- All helpers — `@classmethod` / `@staticmethod` (no `__init__`; runtime state lives at class level).
- `validate_inputs`, `fingerprint_inputs`, `check_lazy_status` when needed.
- Hidden inputs via `cls.hidden` + `IO.Hidden.<name>`.
- Custom IO types: `IO.Custom("MY_TYPE")`. Wildcards: `IO.AnyType.Input/Output`.
- Output node without outputs: `outputs=[]` + `is_output_node=True`.
- `INPUT_IS_LIST` → `is_input_list=True` in `IO.Schema`. `OUTPUT_IS_LIST` → `IO.X.Output(is_output_list=True)`.

---

## 8. Naming Convention

Default prefix: `TS`.

Python V3:

- File: `ts_example_node.py`
- Class: `TS_ExampleNode`
- Schema `node_id`: `"TS_ExampleNode"`
- Display: `"TS Example Node"`
- Category: `"TS/subcategory"`

JavaScript:

- File: `ts-example-node.js`
- Extension ID: `ts.exampleNode`

If the repository already uses another prefix, preserve it.

---

## 9. Python Standards

Use:

- Python 3.10+.
- Type hints for public functions.
- Google-style docstrings for non-trivial functions.
- `pathlib.Path`.
- `logging.getLogger(__name__)`.
- Lazy imports for heavy optional dependencies.
- Specific exception types.

Avoid/forbid:

- `print()` for logging.
- Bare `except:`.
- Broad `except Exception:` without useful handling.
- `global` and hidden mutable module state.
- Import-time side effects.
- Auto-installing packages.
- `eval`, `exec`, unsafe `subprocess`, `os.system` with user input.
- Unsafe `pickle.load`.
- Hardcoded absolute paths.

---

## 10. JavaScript Standards

Use:

- ES2020+.
- ES modules where possible.
- Official ComfyUI extension APIs.
- Stable extension IDs.
- Small focused modules.

Avoid:

- Global variables.
- Prototype monkey-patching.
- Direct mutation of ComfyUI internals.
- Undocumented private fields.
- Silent catch blocks.
- Console spam.

Legacy LiteGraph hooks are allowed only for maintaining compatibility and must be isolated.

---

## 11. Tensor and GPU Rules

ComfyUI tensor conventions:

```text
IMAGE  -> [B, H, W, C], float32, range [0, 1]
MASK   -> [B, H, W],    float32, range [0, 1]
LATENT -> latent["samples"]
```

Always:

- Validate shape.
- Preserve batch dimension.
- Clone before mutating inputs.
- Preserve dtype/range unless documented.
- Respect ComfyUI device/memory management.

Never:

- Process only `image[0]` unless explicitly documented.
- Mutate input tensors in place.
- Assume CUDA.
- Hardcode device names.
- Load large models at import time.

Use:

```python
import comfy.model_management

device = comfy.model_management.get_torch_device()
```

Inference:

```python
with torch.no_grad():
    ...
```

---

## 12. Logging

Python:

```python
logger = logging.getLogger(__name__)
```

Prefixes:

```text
[TS NodeName]
[TS ModuleName]
```

Rules:

- Plain text only.
- No ANSI colors.
- No emojis.
- No secrets.
- Avoid leaking full private user paths.
- DEBUG for internals, INFO for operations, WARNING for recoverable issues, ERROR for failures.

---

## 13. Portability and Optional Dependencies

This project must remain portable, especially for Windows portable ComfyUI builds.

- Use `comfy.folder_paths` for ComfyUI-managed paths.
- Use relative/config paths.
- Validate config values.
- Optional dependencies must fail gracefully.
- Do not run pip from node code.
- Do not mutate permanent environment variables.
- Do not bake secrets into frontend code.

---

## 14. Security and Permissions

Do not perform dangerous operations unless explicitly requested:

- deleting files/directories
- force-push or history rewrite
- global package installs
- permanent environment changes
- editing outside repository
- modifying ComfyUI core
- downloading/executing remote code

Prefer read-only inspection, dry runs, sandboxed commands, and explicit confirmation for destructive actions.

---

## 15. Refactoring Protocol

Before editing existing code:

1. Read all relevant files.
2. Confirm V3 schema (since 8.9 the whole pack is V3).
3. Map public contracts.
4. Identify frontend dependencies.
5. Separate bugs from refactoring opportunities.
6. Choose the smallest safe change.

Safe refactoring:

- Extract private helpers inside the same node file.
- Move only truly shared logic to `utils/`.
- Add type hints/docstrings.
- Replace `print()` with logging.
- Improve validation and tensor operations.
- Fix resource leaks.

Do not:

- Split one node into many files.
- Rewrite stable code for purity.
- Change behavior during refactor unless requested.
- Add features during refactor.
- Delete commented-out code without confirmation.
- Change node identifiers.
- Re-introduce V1 patterns in this pack (legacy V1 only kept as a reading reference for other plugins).

---

## 16. Self-Review Gate

Before finalizing, check:

- Ответ на русском?
- Все возможные тесты запущены?
- Незапущенные проверки явно перечислены?
- Node identity не изменён?
- Inputs/outputs/defaults/category не изменены?
- One-node-one-file сохранён?
- Tensor batch сохранён?
- Input tensors не мутируются?
- Нет private ComfyUI API?
- Нет heavy import/model load at import time?
- Нет unsafe path/shell behavior?
- Frontend E2E запущен при изменениях UI?

---

## 17. Response Format

For new features, answer in Russian:

1. Что сделано.
2. Какие файлы изменены.
3. Дизайн-решения.
4. Предположения.
5. Что проверено.
6. Что не проверено и почему.
7. Риски.

For refactoring:

### Анализ
### Риски
### Изменение
### Проверка

---

## 18. Definition of Done

Task is complete only when:

- Codex answered in Russian.
- Requested change is implemented.
- Workflow contracts are preserved or migration is provided.
- All relevant available checks were run.
- Unrun checks are listed with reasons.
- Code imports successfully.
- Logging is plain text.
- Optional dependencies fail gracefully.
- No unsafe APIs/secrets are introduced.
- Docs are updated if needed.
- One-node-one-file structure is preserved unless a shared utility is justified.

---

## 19. Final Priority Order

1. Preserve existing workflows.
2. Preserve compatibility.
3. Preserve user-facing behavior.
4. Improve correctness.
5. Improve testability.
6. Improve maintainability.
7. Improve performance.
8. Improve architecture.
9. Add features only when requested.
