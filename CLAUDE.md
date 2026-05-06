# CLAUDE.md — Engineering Rules for comfyui-timesaver

Этот файл — операционные правила для Claude (и других AI-ассистентов) при работе над `comfyui-timesaver`. Он построен на основе [AGENTS.md](AGENTS.md) и привязан к ComfyUI custom-node V3 API (`comfy_api.v0_0_2.IO` — pinned namespace, не `latest`).

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

- Версия: `9.1` (`pyproject.toml`).
- Репозиторий: https://github.com/AlexYez/comfyui-timesaver.
- 57 нод (все на V3) в категориях: image / video / audio / llm / text / files / utils / conditioning.
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

## 4.5. Обязательные инструменты (skills + ComfyUI Python)

### 4.5.1. Skill `comfyui-custom-nodes`

Перед любой нетривиальной работой над нодой подключай skill **`/comfyui-custom-nodes`** (плагин). Это база знаний по ComfyUI custom nodes API:

- `comfyui-custom-nodes:comfyui-node-basics` — базовая структура V3 ноды.
- `comfyui-custom-nodes:comfyui-node-frontend` — JS extensions: hooks, widgets, sidebar tabs, commands, settings, DOM widgets, V2 (Vue) layout, addDOMWidget options (`getMinHeight`/`getHeight`/`computeSize`), suppressing default previews.
- `comfyui-custom-nodes:comfyui-node-inputs` — типы входов, виджеты, скрытые/опциональные/lazy.
- `comfyui-custom-nodes:comfyui-node-outputs` — `IO.NodeOutput`, UI outputs, preview-ноды.
- `comfyui-custom-nodes:comfyui-node-datatypes` — IMAGE / MASK / LATENT / AUDIO / VIDEO / 3D conventions.
- `comfyui-custom-nodes:comfyui-node-lifecycle` — caching, fingerprint_inputs, validate_inputs, check_lazy_status.
- `comfyui-custom-nodes:comfyui-node-advanced` — MatchType, Autogrow, DynamicCombo, MultiType, wildcard inputs.
- `comfyui-custom-nodes:comfyui-node-packaging` — структура пакета, `__init__.py`, registration, WEB_DIRECTORY.
- `comfyui-custom-nodes:comfyui-node-migration` — миграция V1 → V3.

Используй skill **до** того, как писать код, особенно при работе с frontend (DOM widgets, Vue render, V2 layout, IMAGEUPLOAD виджеты, image preview suppression).

### 4.5.2. ComfyUI Python для GPU-тестов

Системный/test Python может не содержать numpy/torch/PIL — тесты, которые требуют этих зависимостей, под ним просто скипаются.

Чтобы реально прогнать тесты ноды (особенно с GPU/inference/тензорами), используй ComfyUI-овский portable Python (на Windows portable обычно `python_embeded/python.exe` в корне ComfyUI):

```bash
python -m pytest tests/test_<node>.py -v
python -m compileall .
python tools/build_node_contracts.py
```

Этот Python:

- Содержит `numpy`, `torch` (с CUDA), `PIL`, `comfy_api`, `folder_paths` и весь runtime ComfyUI.
- Используется для регенерации `tests/contracts/node_contracts.json`.
- Должен использоваться для всех verification-команд, которые касаются GPU/inference/contract snapshot.

Для чисто CPU-тестов (схема, paths, fingerprint без torch) подойдёт и обычный test Python — там тесты с numpy/torch скипаются через `pytest.importorskip`.

В verification summary всегда указывай, под каким Python запускались тесты:

```text
Проверено (под ComfyUI Python):
- python -m compileall .
- python -m pytest tests/test_lama_cleanup_contract.py
```

### 4.5.3. Playwright + Chromium для frontend GUI testing

В ComfyUI Python установлен **`playwright` + Chromium** (браузер уже загружен). Используй его как **default** для всего, что требует автоматического GUI рендера:

- Headless screenshots для документации (canvas + DOM widgets composited корректно — `canvas.toDataURL()` теряет DOM widgets).
- Frontend smoke tests: создание ноды → проверка console → snapshot.
- E2E workflow прогон.

**Готовый helper:** [`tools/screenshot_nodes.py`](tools/screenshot_nodes.py).

```bash
# Все 57 нод
python tools/screenshot_nodes.py

# Конкретные (по node_id или file stem)
python tools/screenshot_nodes.py TS_Keyer ts_audio_loader

# Видимое окно для отладки
python tools/screenshot_nodes.py --no-headless
```

**Critical: всегда forced `locale='en-US'`** — иначе Chromium берёт OS-locale (русский) и ComfyUI переведёт UI labels (например, `IMAGE` → `ИЗОБРАЖЕНИЕ`):

```python
context = browser.new_context(
    viewport={"width": 1920, "height": 1080},
    device_scale_factor=1,
    locale="en-US",
    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
)
```

**Playwright vs Chrome MCP:**

| Сценарий | Используй |
|---|---|
| Headless screenshot для README/docs | Playwright |
| Frontend smoke test после изменений (createNode + console) | Playwright |
| E2E workflow runs | Playwright |
| Проверка поведения в реальном браузере пользователя | Chrome MCP |
| Интерактивная отладка с user state/cookies | Chrome MCP |

**Default для frontend automation — Playwright, не Chrome MCP.** Chrome MCP оставляй для случаев, где явно нужен реальный браузер пользователя со всеми его расширениями/настройками.

---

## 5. API: только V3

С релиза `8.9` **весь пак на V3**. V1-нод не осталось — `grep RETURN_TYPES nodes/` пуст, все 57 нод используют `IO.ComfyNode + define_schema + execute`.

Новые ноды:

- Только **ComfyUI V3 schema**.
- Импорт: `from comfy_api.v0_0_2 import IO` (или `IO, UI`). **Не использовать `comfy_api.latest`** — это нестабильный alias; production пина к `v0_0_2`.
- Класс наследует `IO.ComfyNode`.
- `define_schema(cls)` → `IO.Schema(...)` (`@classmethod`).
- `execute(cls, ...)` → `IO.NodeOutput(...)` (`@classmethod`).
- Все вспомогательные методы — `@classmethod` или `@staticmethod` (`__init__` не используется, состояние — class-level).
- При необходимости: `validate_inputs`, `fingerprint_inputs`, `check_lazy_status`, скрытые входы через `cls.hidden` + `IO.Hidden.<name>`.
- Custom IO-типы: `IO.Custom("MY_TYPE")`. Wildcard вход/выход: `IO.AnyType.Input/Output`.
- Output-нода без выходов: `outputs=[]` + `is_output_node=True`.
- INPUT_IS_LIST: `is_input_list=True` в `IO.Schema(...)`. OUTPUT_IS_LIST: `IO.X.Output(is_output_list=True)`.

V1 шаблон (раздел ниже) оставлен только как reference на случай чтения чужих legacy-плагинов; новый код в этом паке всегда V3.

V3 шаблон (минимальный, в стиле проекта):

```python
import logging
from comfy_api.v0_0_2 import IO

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

### 7.1. Big-node split: `nodes/<категория>/<feature>/`

Допустимое исключение из «one node = one file» — когда нода становится god-file (~1000+ строк) **или** имеет существенный набор приватных helpers (HTTP routes, DOM widget logic, voice/audio pipelines, model loaders), которые нельзя переиспользовать в других нодах. Тогда:

1. Создаём подпапку `nodes/<категория>/<feature>/` (например, `nodes/image/lama_cleanup/`, `nodes/audio/loader/`, `nodes/image/keying/`, `nodes/llm/super_prompt/`).
2. **Один публичный `ts_<name>.py`** в этой подпапке содержит класс ноды (schema + execute) и `NODE_CLASS_MAPPINGS` — это единственный entry point для loader-а.
3. **Все остальные модули в подпапке должны начинаться с `_`** (`_helpers.py`, `_voice.py`, `_routes.py`, `_qwen.py`). Loader игнорирует любой путь с `_`-prefixed компонентом, поэтому они не регистрируются как самостоятельные ноды.
4. `__init__.py` в подпапке оставляем пустым.
5. Если в подпапке находится **несколько публичных нод** (как `nodes/audio/loader/` с `TS_AudioLoader` + `TS_AudioPreview`), у каждой свой `ts_<name>.py`; общие helpers — в `_<name>.py` той же подпапки.

Этот паттерн уже применён в проекте — следуй существующим примерам, не изобретай новый layout.

Запрещено и в подпапке:
- Дробить одну ноду на `schema.py + execute.py + types.py` — публичный класс остаётся в одном `ts_<name>.py`.
- Помещать публичный класс в файл с `_`-prefix — loader его не подхватит.
- Поднимать подпапку без явной причины (god-file, обширные helpers) — для простой ноды overhead подпапки не оправдан.

Frontend парный layout: `js/<категория>/<feature>/ts-<name>.js` (entry с `app.registerExtension`) + `_<name>.js` (приватные ES-модули). См. `js/image/lama_cleanup/`, `js/audio/loader/`.

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

Используй именно ComfyUI-овский Python (см. §4.5.2). Под обычным test Python тесты, требующие numpy/torch/PIL, скипаются — это **не** настоящая проверка. Для регенерации `tests/contracts/node_contracts.json` после изменения схем используй тот же Python:

```bash
python tools/build_node_contracts.py
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

- `tests/test_super_prompt_contract.py` — образец V3 contract-теста с monkeypatch-стабами для `comfy_api.v0_0_2`, `folder_paths`, `aiohttp`. Используй этот паттерн для новых V3 нод.
- `tests/test_voice_recognition_audio.py` — пример behavior-теста подсистемы.
- Тесты должны быть CPU-safe, без интернета, без скачивания моделей.
- Tensor-тесты обязаны проверять: shape, batch preservation, dtype/range, **отсутствие мутации входа** (`assert torch.equal(image, before)`).
- Snapshot контрактов (если потребуются) — `tests/contracts/node_contracts.json`.

См. полные правила в [tests/AGENTS.md](tests/AGENTS.md).

---

## 12. Frontend (js/)

`WEB_DIRECTORY = "./js"` в `__init__.py`. Каждый `.js` подключается автоматически.

- Используй `app.registerExtension({ name: "ts.<id>", ... })` через `import { app } from "/scripts/app.js"`.
- Стабильные extension IDs (НЕ менять): `ts.bookmark`, `ts.resolutionselector`, `ts.audioLoader`, `ts.lamaCleanup`, `ts.animationpreview`, `ts.prompt_builder`, `ts_suite.style_prompt_selector`, `ts.superPrompt`, `ts.float-slider`, `ts.int-slider`.
- Один публичный узел с фронтендом = один `.js`. Не дроби на `menu.js / widgets.js / state.js`.
- Никаких глобалов, monkey-patch, eval, `new Function`, blind `innerHTML` с пользовательским текстом.
- Логи: `console.warn("[TS ModuleName] ...")` / `console.error("[TS ModuleName] ...", err)` — без эмоджи и спама.
- Legacy LiteGraph (`registerCustomNodes` + `LGraphNode`) разрешён только для compatibility — пример: `ts-bookmark.js`.
- Когда задеваешь UI и ComfyUI запущен — проверь browser console на чистоту.

См. полные правила в [js/AGENTS.md](js/AGENTS.md).

---

## 12.5. Интерактивные ноды с DOM widgets

Если делаешь полноценный in-node UI (canvas, кисти, drag-drop и т.п.) через `addDOMWidget` — соблюдай эти правила. Каждое — реальная ошибка из истории. Полные объяснения и code snippets — в memory `reference_dom_widget_pitfalls.md`.

### 12.5.1. Layout: НЕ переопределять `widget.computeSize`

ComfyUI core (>=1.34) для DOM widgets использует `computeLayoutSize()`. Установка `widget.computeSize` **выталкивает** widget в фиксированную ветку и ломает layout, давая infinite height growth в V2 Vue.

```javascript
const widgetOptions = {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => 220,
    getMaxHeight: () => 8192,
    afterResize: () => requestRedraw(),
};
const domWidget = node.addDOMWidget(name, "div", container, widgetOptions);
// НЕ делать domWidget.computeSize = ... !
```

### 12.5.2. Не использовать IMAGEUPLOAD/upload= если есть свой UI

`IO.Combo.Input(..., upload=IO.UploadType.image)` добавляет нежелательные UI-элементы в обоих режимах:
- V1: кнопку "choose file to upload" над нодой.
- V2 Vue: image preview **под** нодой (через `node.imgs`, который Vue render не пропускает через `Object.defineProperty`).

Решение: `IO.String.Input("source_path", default="", socketless=True)` + ручной upload через `/upload/image` из JS. Также убирай `advanced=True` с inputs скрытых JS — иначе V2 показывает "Show advanced inputs" toggle.

### 12.5.3. Координаты курсора: viewport vs local CSS pixels

`event.clientX` и `getBoundingClientRect()` — viewport (post-transform). `cursor.style.left` — local CSS pixels (pre-transform). Parent `transform: scale(s)` (LiteGraph zoom, Vue node scale) разводит эти системы → cursor drifts.

Compensation через ratio offsetWidth:

```javascript
const containerRect = container.getBoundingClientRect();
const layoutWidth = container.offsetWidth || containerRect.width;
const parentScale = layoutWidth > 0 ? containerRect.width / layoutWidth : 1;
const inverseScale = parentScale > 0.001 ? 1 / parentScale : 1;
const xLocal = (clientX - containerRect.left) * inverseScale - (container.clientLeft || 0);
const yLocal = (clientY - containerRect.top) * inverseScale - (container.clientTop || 0);
cursor.style.left = `${xLocal - radius}px`;  // НЕ transform: translate — sub-pixel ошибки
cursor.style.top = `${yLocal - radius}px`;
```

### 12.5.4. Mask compositing: НЕ использовать `source-in` с fillRect

Антипаттерн: `drawImage(image)` → `drawImage(maskCanvas)` → `globalCompositeOperation = "source-in"` → `fillRect(...)`. Это **стирает image** даже когда маска пустая, потому что source-in fillRect полностью покрывает destination.

Правильно: tinted mask — отдельный offscreen canvas с тем же paint operations что и mask, но dark color. Redraw делает простой `drawImage(tintedMaskCanvas, ...)` без compositing трюков.

### 12.5.5. Image padding под floating overlays — через JS scale, НЕ CSS canvas inset

Для in-node UI с floating toolbar/statusbar: НЕ позиционировать canvas через `top:56px;bottom:44px` — `<canvas>` это replaced element, нестабилен на resize.

Канвас держать full-bleed (`inset:0`), padding делать в `state.scale/offsetX/offsetY`:

```javascript
const IMAGE_PAD_TOP = 56;
const IMAGE_PAD_BOTTOM = 44;
const IMAGE_PAD_SIDE = 8;

function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    if (state.imageWidth > 0 && rect.width > 0) {
        const usableWidth = Math.max(1, rect.width - IMAGE_PAD_SIDE * 2);
        const usableHeight = Math.max(1, rect.height - IMAGE_PAD_TOP - IMAGE_PAD_BOTTOM);
        state.scale = Math.min(usableWidth / state.imageWidth, usableHeight / state.imageHeight);
        state.offsetX = IMAGE_PAD_SIDE + (usableWidth - state.imageWidth * state.scale) / 2;
        state.offsetY = IMAGE_PAD_TOP + (usableHeight - state.imageHeight * state.scale) / 2;
    }
}
```

`rebuildImageCache` должен использовать `state.offsetX/Y/scale` (не пересчитывать) — иначе cache и mask blit рассинхронизируются.

### 12.5.6. Performance recipe для big images

Три ключевых паттерна, дающих плавную работу даже на 8K:

1. **Image render cache** — pre-rendered image at display resolution в offscreen canvas, rebuild только на resize/image-change. Каждый redraw — cheap blit.
2. **Incremental tinted mask** — рисовать в обе offscreen canvas (mask + tintedMask) во время `drawSegment`/`drawBrushAt`. Redraw = blit готового tinted без full rebuild.
3. **HTML cursor element** — `<div class="cursor">` с `position:absolute; pointer-events:none`. Обновлять через style.left/top на pointer move. Cursor-only движения **НЕ** вызывают `requestRedraw`.

### 12.5.7. Cursor visibility

`cursor:none` на canvas скрывает native cursor. Если custom HTML cursor показывается только при `state.image` — без image над canvas получается "мёртвая зона". Решение: условный CSS класс:

```css
.ts-lama__canvas{cursor:default}
.ts-lama__canvas.has-image{cursor:none}
```

```javascript
canvas.classList.toggle("has-image", Boolean(state.image));
```

### 12.5.8. Backend для интерактивных нод

- **Per-session asyncio.Lock** для длительных jobs:
  ```python
  _session_locks: dict[str, asyncio.Lock] = {}
  def _get_session_lock(session_id):
      lock = _session_locks.get(session_id)
      if lock is None:
          lock = asyncio.Lock()
          _session_locks[session_id] = lock
      return lock
  
  async def handler(request):
      lock = _get_session_lock(safe_session_id)
      async with lock:
          # serialised per session
  ```
- **Versioned working files** — `{session}_{tag}_{nanos:020d}.png`. Никогда не overwrite — нужно для undo/redo.
- **Cleanup** на `/seed`, `/reset`, history overflow. Только `path.name.startswith(f"{safe_session}_")` — защита от удаления чужих файлов.

### 12.5.9. Output organization

Сохранять результаты в подпапку с тегом в имени:
- `output/<feature_name>/<source_stem>_<feature_name>_<timestamp>.png`.
- Response `{"subfolder": "<feature_name>", "type": "output"}`.

### 12.5.10. Folder registration для моделей

```python
def _register_model_folder():
    base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(base))

_register_model_folder()  # на module import — поддержка extra_model_paths.yaml
```

### 12.5.11. Hidden file input + drag-drop + paste

- `<input type="file">` скрыть через `position:fixed; left:-9999px; top:-9999px;` (НЕ `width:1px;height:1px` — некоторые браузеры блокируют программный `.click()`).
- Container-level dragenter/dragover/dragleave/drop для drag-and-drop.
- Document-level paste с проверкой `pointerOverContainer()` чтобы избежать конфликта между несколькими нодами одного типа.
- В `node._tsCleanup` обязательно `document.removeEventListener("paste", ...)` — иначе утечка listener'ов.

### 12.5.12. Эталонная реализация

`nodes/image/lama_cleanup/` + `js/image/lama_cleanup/` — TS_LamaCleanup, наиболее полный пример всех паттернов из этого раздела.

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
