# CLAUDE.md — Engineering Rules for comfyui-timesaver

Этот файл — операционные правила для Claude (и других AI-ассистентов) при работе над `comfyui-timesaver`. Он построен на основе [AGENTS.md](AGENTS.md) и привязан к ComfyUI custom-node экосистеме (V1/V3 API).

Главная цель: **максимальное качество итогового кода без поломки старых ComfyUI workflows**.

Кредо проекта:

> Readable. Stable. Testable.

ComfyUI-кредо:

> Stability. Identity. Scalability. Predictability.

Принцип рефакторинга:

> Primum non nocere — first, do no harm.

Язык общения с пользователем — **русский**. Код, идентификаторы, docstrings — английский.

---

## 0. Иерархия инструкций

Соблюдай этот корневой `CLAUDE.md` и ближайший локальный `AGENTS.md`:

```text
nodes/AGENTS.md
js/AGENTS.md
doc/AGENTS.md
tests/AGENTS.md
```

Локальные `AGENTS.md` могут уточнять правила, но не могут ослаблять совместимость, безопасность, тестирование и стабильность node contracts.

При конфликте: workflow compatibility > безопасность > тесты > всё остальное.

---

## 1. Контекст проекта

`comfyui-timesaver` — production-quality пак custom nodes и frontend extensions для ComfyUI.

- Версия: `8.8` (`pyproject.toml`).
- Репозиторий: https://github.com/AlexYez/comfyui-timesaver.
- 56 нод в категориях: image / video / audio / llm / text / files / utils / conditioning.
- conditioning/ содержит 1 ноду: TS_MultiReference.
- Корневой загрузчик: [`__init__.py`](__init__.py) — рекурсивно сканирует `nodes/**/*.py`, оборачивает entrypoints через `TSDependencyManager` и печатает startup-таблицу.
- Dependency guard: [`ts_dependency_manager.py`](ts_dependency_manager.py).
- Snapshot контрактов: [`tests/contracts/node_contracts.json`](tests/contracts/node_contracts.json) (генерится `tools/build_node_contracts.py`).

Layout (с релиза 8.8):

```text
comfyui-timesaver/
├─ AGENTS.md / CLAUDE.md
├─ __init__.py            # рекурсивное auto-discovery + import audit
├─ ts_dependency_manager.py
├─ pyproject.toml
├─ requirements.txt
├─ README.md / README.ru.md
├─ nodes/
│  ├─ __init__.py
│  ├─ _shared.py          # pack-level helpers (TS_Logger)
│  ├─ image/  (25)
│  │  ├─ _keying_helpers.py
│  │  └─ ts_*.py
│  ├─ video/  (7)
│  ├─ audio/  (6)
│  │  └─ _audio_helpers.py
│  ├─ llm/    (2)
│  ├─ text/   (4)
│  ├─ files/  (8)         # TS_FileBrowser удалён
│  ├─ utils/  (4)
│  └─ conditioning/ (1)  # TS_MultiReference
├─ js/                    # WEB_DIRECTORY = "./js", сканируется рекурсивно
│  ├─ image/ video/ audio/ llm/ text/ files/ utils/
│  └─ utils/_slider_helpers.js   # shared ES module
├─ tools/
│  ├─ __init__.py
│  └─ build_node_contracts.py
├─ doc/
│  ├─ AGENTS.md
│  ├─ TS_DEPENDENCY_POLICY.md
│  └─ migration.md
├─ tests/
│  ├─ contracts/node_contracts.json
│  ├─ conftest.py         # ts_tmp_path fixture
│  ├─ test_pack_imports.py
│  ├─ test_node_contracts.py
│  └─ test_super_prompt_contract.py / test_voice_recognition_audio.py
└─ .github/workflows/     # ComfyRegistry publish
```

Loader-правила:
- Файл-нода обязан начинаться с `ts_`.
- Любой путь, содержащий часть с `_`-префиксом, пропускается → используется для shared private модулей (`_shared.py`, `_keying_helpers.py`, `_audio_helpers.py`).

---

## 2. Языковые правила

Обязательно:

- Всегда отвечай пользователю **на русском**.
- Все планы, пояснения, отчёты, риски, verification summary — на русском.
- Не заявляй, что задача завершена, если проверки не запускались (или явно объясни, почему).
- Если тесты невозможно запустить — честно скажи и дай точные команды для локальной проверки.
- Для сложных задач сначала дай короткий план.
- Для рефакторинга всегда пиши: **Анализ / Риски / Изменение / Проверка**.

Разрешено английским:

- Код, имена файлов/классов/переменных, docstrings.
- Технические идентификаторы ComfyUI/Python/JS.

---

## 3. Operating Loop: Plan → Implement → Verify → Review

Для любой нетривиальной задачи:

1. **Plan**
   - Прочитай релевантные файлы.
   - Определи API ноды: V1 или V3.
   - Выпиши публичные контракты: `node_id`, inputs, outputs, defaults, category, frontend IDs.
   - Выбери стратегию тестирования заранее.
   - Выбери минимальное безопасное изменение.

2. **Implement**
   - Делай только запрошенное.
   - Не добавляй unrelated features.
   - Не переписывай рабочий код ради «чистой архитектуры».
   - Сохраняй правило: одна публичная нода = один основной `.py` (+ один опциональный `.js`).

3. **Verify**
   - Запусти все доступные релевантные проверки (см. §10).
   - Если тест падает — исправь минимально и повтори.
   - Если проверка невозможна — явно напиши почему.

4. **Review**
   - Перечитай diff.
   - Проверь workflow compatibility.
   - Проверь безопасность, hidden side effects, dead code, private API, tensor mutation.

Если план оказался неверным — остановись и перепланируй.

---

## 4. Non-Negotiable: Workflow Compatibility

Никогда не меняй без явного запроса/миграции:

- Python class names.
- V3 `node_id`.
- V1 ключи `NODE_CLASS_MAPPINGS`.
- JS extension IDs.
- Имена inputs / outputs.
- Порядок и типы outputs.
- Имена параметров `execute()` / `FUNCTION`.
- Default widget values.
- `CATEGORY`.
- Saved configuration keys и widget IDs, на которые ссылается фронтенд.
- Существующую семантику поведения.

Если переименование действительно необходимо:

- Сохрани старый `node_id` где возможно.
- V3: добавь `search_aliases`; используй `io.NodeReplace` через `ComfyAPI`.
- V1: оставь старый ключ в `NODE_CLASS_MAPPINGS` как alias.
- Добавь docs migration note.
- Добавь/обнови contract test.

---

## 5. API: V3 по умолчанию, V1 — frozen

Состояние пака: смешанное. Часть нод уже на V3 (`comfy_api.latest.IO`), часть остаётся V1.

Новые ноды:

- По умолчанию **ComfyUI V3 schema**.
- Импорт: `from comfy_api.latest import IO` (или `IO, UI`).
- Класс наследует `IO.ComfyNode`.
- `define_schema(cls)` → `IO.Schema(...)` (`@classmethod`).
- `execute(cls, ...)` → `IO.NodeOutput(...)` (`@classmethod`).
- При необходимости: `validate_inputs`, `fingerprint_inputs`, `check_lazy_status`, скрытые входы через `cls.hidden`.
- Пинить `comfy_api.v0_0_2` только если этого требует release target.

Существующие V1 ноды:

- Frozen public contracts.
- Не мигрировать V1 → V3 без явного запроса пользователя.
- Не смешивать V1 и V3 паттерны в одном файле без переходного слоя.

V3 шаблон (минимальный, в стиле проекта):

```python
import logging
from comfy_api.latest import IO

class TS_ExampleNode(IO.ComfyNode):
    _LOGGER = logging.getLogger("comfyui_timesaver.ts_example")
    _LOG_PREFIX = "[TS Example]"

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ExampleNode",
            display_name="TS Example Node",
            category="TS/examples",
            description="Short user-facing description.",
            inputs=[
                IO.Image.Input("image"),
                IO.Float.Input("strength", default=1.0, min=0.0, max=10.0),
            ],
            outputs=[IO.Image.Output(display_name="image")],
            search_aliases=["example", "ts example"],
        )

    @classmethod
    def execute(cls, image, strength: float) -> IO.NodeOutput:
        result = image.clone()
        return IO.NodeOutput(result)


NODE_CLASS_MAPPINGS = {"TS_ExampleNode": TS_ExampleNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ExampleNode": "TS Example Node"}
```

V1 шаблон (для legacy maintenance):

```python
class TS_LegacyExample:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "TS/examples"

    def process(self, image):
        return (image.clone(),)


NODE_CLASS_MAPPINGS = {"TS_LegacyExample": TS_LegacyExample}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LegacyExample": "TS Legacy Example"}
```

---

## 6. Naming Convention (TS-prefix)

| Артефакт | Пример |
| --- | --- |
| Файл Python | `ts_example_node.py` |
| Класс Python | `TS_ExampleNode` |
| V3 `node_id` | `"TS_ExampleNode"` |
| Display name | `"TS Example Node"` |
| Category | `"TS/<подкатегория>"` (`TS/image`, `TS/audio`, `TS/LLM`, `TS/Color`, `TS Tools/Sliders`) |
| JS-файл | `ts-example-node.js` |
| JS extension ID | `ts.exampleNode` |

Существующие исключения уважаем (не переименовываем): `TSWhisper`, `TSCropToMask`, `TSRestoreFromCrop`, `TSAutoTileSize`, `TS Cube to Equirectangular`, `TS Equirectangular to Cube`, и др.

---

## 7. One Node = One File

> Одна публичная ComfyUI-нода = один основной `.py` файл в подходящей `nodes/<категория>/`. При необходимости фронтенда — один соответствующий `.js` в `js/<категория>/`.

Запрещено:

```text
nodes/ts_example/schema.py        # не дробим одну ноду на много файлов
nodes/ts_example/execute.py
```

```text
nodes/ts_combined.py              # не объединяем несколько нод в один файл
class TS_NodeA: ...
class TS_NodeB: ...
```

Категории: `image / video / audio / llm / text / files / utils`. Если новая нода реально не вписывается ни в одну — обсудить с пользователем; не плодить новые категории молча.

Shared-логика разрешена только когда используется 2+ нодами:

- В пределах одной категории — `nodes/<категория>/_<name>.py` (приватный модуль; loader пропускает по `_`-префиксу).
- В пределах всего пакета — `nodes/_shared.py`.

Фронтенд аналогично: `js/<категория>/_<name>.js` для shared ES-модулей.

Существующие примеры shared-модулей: `nodes/_shared.py` (TS_Logger), `nodes/image/_keying_helpers.py` (gaussian_blur_4d), `nodes/audio/_audio_helpers.py` (probe/decode/preview helpers + aiohttp routes), `js/utils/_slider_helpers.js` (slider config logic).

---

## 8. Tensor & GPU Rules

ComfyUI-конвенции:

```text
IMAGE  -> [B, H, W, C], float32, range [0, 1]
MASK   -> [B, H, W],    float32, range [0, 1]
LATENT -> latent["samples"]  (shape [B, C, H, W])
AUDIO  -> {"waveform": [B, C, T], "sample_rate": int}
```

Всегда:

- Проверяй `image.ndim == 4`, `mask.ndim == 3`, когда это релевантно.
- Сохраняй batch dimension (никаких `image[0]` без явной документации).
- Клонируй перед мутацией: `image.clone()`.
- Сохраняй dtype/device.
- Clamp только если операция действительно может выйти за валидный диапазон.
- Используй `torch.no_grad()` для inference.

Никогда:

- Не мутируй входные тензоры.
- Не предполагай CUDA. Используй `comfy.model_management.get_torch_device()`.
- Не хардкодь `cuda:0`.
- Не загружай большие модели на module level — только лениво в `execute`/`process`.

Пример:

```python
import comfy.model_management as mm
import torch

device = mm.get_torch_device()
with torch.no_grad():
    out = some_model(image.to(device)).clamp(0.0, 1.0)
```

---

## 9. Validation, Fingerprint, Lazy

V3-ноды (рекомендуется):

- `validate_inputs(cls, ...)` → `True` или строка с понятной ошибкой.
- `fingerprint_inputs(cls, ...)` → детерминированный hash для cache invalidation. Никогда не возвращай константу, если результат зависит от внешнего состояния.
- `check_lazy_status(...)` — только когда нужно; никакой тяжёлой работы.
- Скрытые входы декларируй явно и читай через `cls.hidden`.

V1:

- Эквиваленты: `VALIDATE_INPUTS`, `IS_CHANGED` (см. ComfyUI core).

Пример из проекта — см. `nodes/ts_audio_loader_node.py` (`TS_AudioLoader.fingerprint_inputs` хеширует mode + crop + размер/mtime файла).

---

## 10. Mandatory Verification

Минимум backend перед заявлением «готово»:

```bash
python -m compileall .
python -m pytest tests
```

Если доступны:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy .
```

Frontend (npm tooling в репо отсутствует — добавлять только если действительно нужно):

```bash
npm run lint
npm run test
npm run build
npm run test:e2e
```

Frontend E2E через Playwright (если ComfyUI запущен на `127.0.0.1:8188`):

```bash
npx playwright test tests/e2e
```

Verification summary в ответе пользователю — обязателен и пишется на русском:

```text
Проверено:
- python -m compileall .
- python -m pytest tests

Не проверено:
- npm run test:e2e — ComfyUI не запущен на 127.0.0.1:8188.
```

Никогда не пиши «готово» только потому, что код выглядит правильным.

---

## 11. Tests

- `tests/test_super_prompt_contract.py` — образец V3 contract-теста с monkeypatch-стабами для `comfy_api.latest`, `folder_paths`, `aiohttp`. Используй этот паттерн для новых V3 нод.
- `tests/test_voice_recognition_audio.py` — пример behavior-теста подсистемы.
- Тесты должны быть CPU-safe, без интернета, без скачивания моделей.
- Tensor-тесты обязаны проверять: shape, batch preservation, dtype/range, **отсутствие мутации входа** (`assert torch.equal(image, before)`).
- Snapshot контрактов (если потребуются) — `tests/contracts/node_contracts.json`.

См. полные правила в [tests/AGENTS.md](tests/AGENTS.md).

---

## 12. Frontend (js/)

`WEB_DIRECTORY = "./js"` в `__init__.py`. Каждый `.js` подключается автоматически.

- Используй `app.registerExtension({ name: "ts.<id>", ... })` через `import { app } from "/scripts/app.js"`.
- Стабильные extension IDs (НЕ менять): `ts.bookmark`, `ts.resolutionselector`, `ts.audioLoader`, `ts.animationpreview`, `ts.prompt_builder`, `ts_suite.style_prompt_selector`, `ts.superPrompt`, `ts.float-slider`, `ts.int-slider`.
- Один публичный узел с фронтендом = один `.js`. Не дроби на `menu.js / widgets.js / state.js`.
- Никаких глобалов, monkey-patch, eval, `new Function`, blind `innerHTML` с пользовательским текстом.
- Логи: `console.warn("[TS ModuleName] ...")` / `console.error("[TS ModuleName] ...", err)` — без эмоджи и спама.
- Legacy LiteGraph (`registerCustomNodes` + `LGraphNode`) разрешён только для compatibility — пример: `ts-bookmark.js`.
- Когда задеваешь UI и ComfyUI запущен — проверь browser console на чистоту.

См. полные правила в [js/AGENTS.md](js/AGENTS.md).

---

## 13. Python Standards

Используй:

- Python 3.10+.
- Type hints для публичных функций.
- Google-style docstrings для нетривиальных функций.
- `pathlib.Path`.
- `logging.getLogger(__name__)` — префиксы `[TS NodeName]` / `[TS ModuleName]`, plain text, без ANSI/emoji/секретов.
- Lazy imports для тяжёлых optional-зависимостей через `TSDependencyManager.import_optional(...)`.
- Specific exception types, понятные сообщения с TS-префиксом.

Запрещено:

- `print()` для логирования.
- Голый `except:` или `except Exception:` без полезной обработки.
- `global` и скрытое мутируемое module state.
- Side effects на module-level (загрузка моделей, открытие файлов и т.п.).
- Авто-установка пакетов из кода.
- `eval`, `exec`, `subprocess` с пользовательским вводом, `os.system`.
- `pickle.load` без strict-режима/доверенного источника.
- Хардкод абсолютных путей.

---

## 14. Dependency Resilience (TSDependencyManager)

Политика — `doc/TS_DEPENDENCY_POLICY.md`:

1. Optional-зависимости → `TSDependencyManager.import_optional("module.path")`.
2. Проверяй наличие в runtime entrypoint (`execute` или метод из `FUNCTION`) и кидай `RuntimeError` с TS-префиксом, если зависимость нужна.
3. Не делай hard module-level imports для optional-пакетов.
4. Логи — plain text, actionable.

Runtime guard:

- `__init__.py` оборачивает все зарегистрированные ноды через `TSDependencyManager.wrap_node_runtime(...)`.
- V1: при исключении возвращает typed fallback на основе `RETURN_TYPES`.
- V3: нормализует ошибку и пробрасывает `RuntimeError` с TS-префиксом.

---

## 15. Logging

```python
import logging
logger = logging.getLogger(__name__)  # или getLogger("comfyui_timesaver.ts_<name>")
LOG_PREFIX = "[TS NodeName]"

logger.info("%s started: %s", LOG_PREFIX, payload)
```

- DEBUG — внутреннее, INFO — операции, WARNING — recoverable, ERROR — failure.
- Plain text, без ANSI, без emoji, без секретов.
- Не утекать полные приватные пути пользователя, если можно — обрезать.

---

## 16. Portability

Пак должен оставаться портативным (включая Windows portable ComfyUI):

- Используй `folder_paths` (`folder_paths.get_input_directory()`, `folder_paths.models_dir`, и т.п.) для путей.
- Конфиг — относительный или валидируемый.
- Optional-зависимости — gracefully fail.
- Не запускай `pip` из кода узла.
- Не модифицируй системные env-переменные постоянно.
- Не зашивай секреты во frontend.

---

## 17. Security & Permissions

Не выполняй опасные операции без явного запроса пользователя:

- Удаление файлов/директорий, force-push, history rewrite.
- Глобальная установка пакетов, постоянные изменения env.
- Редактирование вне репозитория.
- Модификация ComfyUI core.
- Скачивание/выполнение удалённого кода.

Предпочитай read-only inspection, dry-run, sandbox-команды и явное подтверждение для destructive действий.

---

## 18. Refactoring Protocol

Перед редактированием существующего кода:

1. Прочитай все релевантные файлы.
2. Определи V1 vs V3.
3. Зафиксируй публичные контракты ноды.
4. Найди зависимости фронтенда.
5. Раздели «баг» и «refactor opportunity».
6. Выбери минимальное безопасное изменение.

Safe:

- Извлечь private helpers в тот же файл.
- Перенести только реально shared-логику в `utils/`.
- Добавить type hints / docstrings.
- Заменить `print()` на logging.
- Улучшить validation и tensor-операции.
- Починить resource leaks.

Запрещено в рамках «чистого refactor»:

- Дробить ноду на много файлов.
- Переписывать стабильный код ради чистоты.
- Менять поведение.
- Добавлять фичи.
- Удалять закомментированный код без подтверждения.
- Менять идентификаторы.
- Мигрировать V1 → V3 без явного запроса.

Формат ответа для рефакторинга — на русском, четырьмя секциями:

```markdown
### Анализ
### Риски
### Изменение
### Проверка
```

---

## 19. Self-Review Gate (перед финалом)

- [ ] Ответ на русском?
- [ ] Все возможные тесты запущены?
- [ ] Незапущенные проверки явно перечислены с причинами?
- [ ] Node identity не изменён (class name, `node_id`, маппинги, JS ID)?
- [ ] Inputs/outputs/defaults/category не изменены?
- [ ] One-node-one-file сохранён?
- [ ] Tensor batch сохранён, входной тензор не мутируется?
- [ ] Нет private ComfyUI API?
- [ ] Нет heavy import / model load на module-level?
- [ ] Нет unsafe path/shell behavior?
- [ ] Frontend E2E запущен/задокументирован при изменениях UI?
- [ ] Logging plain text, без секретов?
- [ ] Optional dependencies fail gracefully?

---

## 20. Response Format

Для новых фич, отвечай на русском по структуре:

1. Что сделано.
2. Какие файлы изменены.
3. Дизайн-решения.
4. Предположения.
5. Что проверено.
6. Что не проверено и почему.
7. Риски.

Для рефакторинга — `### Анализ / ### Риски / ### Изменение / ### Проверка`.

Verification summary всегда в формате:

```text
Проверено:
- python -m compileall .
- python -m pytest tests

Не проверено:
- npm run test:e2e — ComfyUI не запущен на 127.0.0.1:8188.
```

---

## 21. Definition of Done

Задача завершена только когда:

- Ответ пользователю на русском.
- Запрошенное изменение реализовано.
- Workflow contracts сохранены или предоставлен план миграции.
- Все доступные релевантные проверки запущены.
- Незапущенные проверки перечислены с причинами.
- Код импортируется без ошибок (`python -m compileall .`).
- Logging plain text.
- Optional dependencies fail gracefully.
- Никаких unsafe API/секретов.
- Документация обновлена при необходимости.
- One-node-one-file сохранён (или shared-utility обоснован).

---

## 22. Final Priority Order

1. Preserve existing workflows.
2. Preserve compatibility.
3. Preserve user-facing behavior.
4. Improve correctness.
5. Improve testability.
6. Improve maintainability.
7. Improve performance.
8. Improve architecture.
9. Add features only when requested.

---

## 23. Полезные ссылки внутри репо

- [README.md](README.md) — пользовательская документация (русский + английский).
- [AGENTS.md](AGENTS.md) — root engineering rules (источник этого файла).
- [nodes/AGENTS.md](nodes/AGENTS.md) — backend-specific правила.
- [js/AGENTS.md](js/AGENTS.md) — frontend-specific правила.
- [tests/AGENTS.md](tests/AGENTS.md) — testing rules.
- [doc/AGENTS.md](doc/AGENTS.md) — documentation rules.
- [doc/TS_DEPENDENCY_POLICY.md](doc/TS_DEPENDENCY_POLICY.md) — политика зависимостей.
- [pyproject.toml](pyproject.toml) — версия, dependencies, ComfyRegistry metadata.
- [requirements.txt](requirements.txt) — runtime dependencies.
