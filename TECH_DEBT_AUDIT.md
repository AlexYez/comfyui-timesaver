# Tech debt audit — comfyui-timesaver

Read-only аудит технического долга. Файл — единственный артефакт, ничего другого не модифицировалось.

---

## 1. Mental model

`comfyui-timesaver` v9.1 — production-ready пак из 57 V3-нод для ComfyUI, покрывающий image/video/audio/LLM/text/files/utils/conditioning. Опубликован через ComfyRegistry, основной потребитель — конечные пользователи ComfyUI (Windows portable, Linux desktop, Mac). Тяжёлые ноды (`TS_BGRM_BiRefNet`, `TS_LamaCleanup`, `TS_Qwen3_VL_V3`, `TS_Whisper`) загружают модели из HuggingFace по требованию через `huggingface_hub`. Несколько нод (`TS_LamaCleanup`, `TS_AudioLoader`, `TS_PromptBuilder`, `TS_StylePromptSelector`, `TS_SuperPrompt`) регистрируют HTTP-routes на `PromptServer.instance.routes` и имеют интерактивный JS-фронтенд через `addDOMWidget`.

Loader в [`__init__.py`](__init__.py) рекурсивно сканирует `nodes/**/ts_*.py`, пропуская любые пути с компонентом, начинающимся с `_`, и оборачивает каждый зарегистрированный класс в [`TSDependencyManager.wrap_node_runtime`](ts_dependency_manager.py:84). Опциональные тяжёлые зависимости (bitsandbytes, demucs, silero, openai-whisper) выделены в `[project.optional-dependencies]` и lazily импортируются.

## 2. Schema mode

**Pure V3.** `grep RETURN_TYPES nodes/` пуст, все 57 нод используют `IO.ComfyNode + define_schema + execute → IO.NodeOutput`. Внутри [`nodes/llm/ts_super_prompt.py`](nodes/llm/ts_super_prompt.py) лежит backwards-compat shim для тестов, который re-export'ит публичный класс из `nodes/llm/super_prompt/ts_super_prompt.py`; loader подхватывает оба пути, `dict.update` с одинаковым ключом и idempotency-флаг `_ts_runtime_guard_wrapped` гарантируют отсутствие двойной регистрации.

## 3. Version compatibility matrix

- `requires-python` — **не объявлен** в [pyproject.toml](pyproject.toml). README показывает badge "python-3.10+", CI прогоняется на 3.10/3.11/3.12, в коде используется PEP 585 (`list[...]`, `dict[...]`) и PEP 604 (`X | None`, требуют 3.10+).
- `requires-comfyui` — **не объявлен** в [pyproject.toml](pyproject.toml) `[tool.comfy]`, при том что весь пак импортирует `comfy_api.latest`, который по документации ComfyUI намеренно нестабилен.
- `comfyui-frontend-package` — **не объявлен**, фронтенд использует современные API (`addDOMWidget`, `app.registerExtension`, прямой импорт `/scripts/app.js`).
- `comfy_api` — везде используется `comfy_api.latest`, ни один файл не закреплён на `comfy_api.v0_0_2`. См. finding #1.
- `classifiers` — отсутствуют, OS/GPU declarations не задекларированы. Часть нод требует CUDA, но это не выражено машиночитаемо.
- `LICENSE.txt` — **отсутствует** в репозитории, при том что [`pyproject.toml:5`](pyproject.toml:5) объявляет `license = { file = "LICENSE.txt" }`. См. finding #4.

## 4. Convention compliance summary

57 публичных классов нод, по одному классу в файле — конвенция "one node = one file" соблюдена везде. Подпапки (`nodes/image/lama_cleanup/`, `nodes/image/keying/`, `nodes/audio/loader/`, `nodes/llm/super_prompt/`) корректно используют `_*.py` для приватных helpers. Frontend `js/<категория>/<feature>/` зеркалирует backend layout. **Конвенциональных нарушений one-file-per-node не найдено.**

## 5. Excluded paths

- `nodes/frame_interpolation_models/`, `nodes/video_depth_anything/` — vendored third-party model code (не TS-prefixed, loader пропускает).
- `nodes/luts/`, `nodes/prompts/`, `nodes/styles/` — статические ассеты (`.cube`, `.txt`, `.json`).
- `.cache/`, `.test_input/`, `.test_models/`, `__pycache__/` — генерируемое.
- `doc/screenshots/` — PNG-ассеты для README.
- `tools/`, `tests/` — не относятся к runtime пака.

## 6. Top takeaways

1. Все 57 нод закреплены на `comfy_api.latest` (документация ComfyUI прямо называет его нестабильным) — finding #1.
2. `TS_FilmGrain` мутирует входной тензор при определённых device/dtype-комбинациях — finding #2.
3. `TS_BGRM_BiRefNet` принудительно перебивает выбор устройства из `model_management.get_torch_device()` на `cuda` — finding #3.
4. Отсутствует `LICENSE.txt`, на который ссылается `pyproject.toml` и README badge — finding #4.
5. `pyproject.toml` без `requires-python` и `requires-comfyui` — finding #5.

---

## Findings

### 1. [HIGH · M] Весь пак закреплён на `comfy_api.latest`

Files: [nodes/audio/ts_music_stems.py:10](nodes/audio/ts_music_stems.py:10),
       [nodes/audio/ts_whisper.py:15](nodes/audio/ts_whisper.py:15),
       [nodes/image/ts_bgrm_birefnet.py:19](nodes/image/ts_bgrm_birefnet.py:19),
       [nodes/llm/ts_qwen3_vl.py:17](nodes/llm/ts_qwen3_vl.py:17),
       [nodes/conditioning/ts_multi_reference.py:52](nodes/conditioning/ts_multi_reference.py:52),
       … и ещё 52 файла, итого все 57 классов нод.

**What's wrong**: каждый файл V3-ноды импортирует `from comfy_api.latest import IO`. По документации ComfyUI (`https://docs.comfy.org/custom-nodes/v3_migration`) `latest` намеренно нестабилен: туда попадают breaking changes, а stable API замораживается в датированных модулях типа `comfy_api.v0_0_2`. Внутренний документ пака [`nodes/AGENTS.md:96`](nodes/AGENTS.md:96) сам пишет: *"Use pinned `comfy_api.v0_0_2` only when the project release target requires it"* — но ни один файл этого не делает. Кроме того, в [`pyproject.toml`](pyproject.toml) нет ни `requires-comfyui`, ни pin'а на конкретную версию, поэтому пользователь, обновивший ComfyUI, может получить ImportError во всех 57 нодах одновременно.

**Why it matters here**: ComfyRegistry-публикация автоматическая ([`.github/workflows/publish_action.yml`](.github/workflows/publish_action.yml)), и при breaking change в `comfy_api.latest` без сопутствующего bump в этом паке ВСЕ workflows пользователей с этими нодами будут падать. Это пакет на 57 нод — потенциальный blast radius огромен.

**Recommendation**: заменить `from comfy_api.latest import IO` на `from comfy_api.v0_0_2 import IO` (или другую датированную версию, поддерживающую все используемые фичи) во всех 57 файлах + примеры в `nodes/AGENTS.md`/`CLAUDE.md`. Добавить в [`pyproject.toml`](pyproject.toml) `[tool.comfy] requires-comfyui = ">=X.Y"` с минимальной версией ComfyUI, экспортирующей `comfy_api.v0_0_2`. Регенерировать снапшот через `python tools/build_node_contracts.py`.

---

### 2. [HIGH · S] `TS_FilmGrain` мутирует входной тензор при no-op преобразовании

Files: [nodes/image/ts_film_grain.py:74-75](nodes/image/ts_film_grain.py:74),
       [nodes/image/ts_film_grain.py:162-163](nodes/image/ts_film_grain.py:162)

**What's wrong**: блок `if images.device != target_device or images.dtype != target_dtype: images = images.to(...)` выполняет `.to()` (создающий новый тензор) только при изменении device или dtype. Когда вход уже находится на нужном устройстве и dtype (например, `force_gpu=True` с CUDA-float16 входом или `force_gpu=False` с CPU-float32 входом), `images` остаётся ссылкой на оригинальный тензор от вызывающей ноды. Затем строка 162 `output_images = images.add_(final_grain)` — in-place add, который мутирует тот самый исходный тензор; следующая `output_images.clamp_(0.0, 1.0)` усугубляет ситуацию.

**Trigger**: вход `IMAGE` тензор `[B,H,W,3] float32 device='cpu'` при `force_gpu=False` и input уже на CPU (типичный случай для CPU-only/MPS setup). Также воспроизводится при `force_gpu=True`, если предыдущая нода уже вернула CUDA-float16. Все остальные параметры — defaults.

**Why it matters here**: ComfyUI кэширует IMAGE-тензоры между нодами; мутация ломает идемпотентность. Если тот же тензор подаётся в TS_FilmGrain параллельно с другой нодой или повторно при cache hit, грейн будет применён несколько раз, и пользователь получит "повышенный шум на каждом ре-ране". CLAUDE.md §8 это запрещает прямым текстом, и в [`tests/test_image_nodes.py:11`](tests/test_image_nodes.py:11) docstring рекламирует *"TS_FilmGrain: schema + tensor invariants (no input mutation, range, shape)"*, но фактически нет ни одного `def test_film_grain_*` — заявленный инвариант не проверяется.

**Recommendation**: либо безусловно клонировать вход (`images = images.clone().to(target_device, dtype=target_dtype, non_blocking=True)`), либо заменить in-place операции на не-mutating (`output_images = images + final_grain` затем `output_images = output_images.clamp(0.0, 1.0)`). Добавить в `tests/test_image_nodes.py` тест `assert torch.equal(image, before)` после execute с параметрами по умолчанию (по аналогии с [`tests/test_bgrm_node.py:143`](tests/test_bgrm_node.py:143)).

---

### 3. [HIGH · M] `TS_BGRM_BiRefNet._get_target_device` перебивает `comfy.model_management`

Files: [nodes/image/ts_bgrm_birefnet.py:178-189](nodes/image/ts_bgrm_birefnet.py:178)

**What's wrong**: функция вызывает `model_management.get_torch_device()` (строка 180), но затем в строке 185 проверяет `if torch.cuda.is_available() and getattr(target_device, "type", str(target_device)) == "cpu"` и **перезаписывает** результат на `torch.device("cuda")` (строка 187). Это явно противоречит контракту ComfyUI: `get_torch_device()` уже учитывает флаги `--cpu`, `--directml`, multi-GPU index'ы и vram management.

**Trigger**: пользователь запустил ComfyUI с `--cpu` (или установил vram management в "Lowvram"/"NoVram"), CUDA-устройство физически доступно. `get_torch_device()` возвращает `torch.device("cpu")`, узел молча игнорирует это и грузит модель на cuda:0. На системах с несколькими GPU аналогичная проблема: ComfyUI может направить узел на `cuda:1`, а здесь жёстко берётся `cuda` без индекса.

**Why it matters here**: пользователь, явно выбравший CPU-режим, получает OOM/нестабильность; multi-GPU пользователь получает работу не на том устройстве, на которое ComfyUI ожидает разделить нагрузку. ComfyUI documentation прямо предписывает использовать `model_management.get_torch_device()` без перебивания. Все остальные узлы пака (например, [`nodes/image/ts_color_match.py:364-372`](nodes/image/ts_color_match.py:364), [`nodes/audio/ts_silero_tts.py:158-167`](nodes/audio/ts_silero_tts.py:158)) делают это правильно — здесь явное локальное отступление.

**Recommendation**: удалить блок строк 185-187. Если есть реальная необходимость переопределять CPU-фоллбек ComfyUI (что само по себе сомнительно), то добавить пользовательский input `device` (`auto/cpu/gpu`) и уважать `--cpu` флаг через `model_management.get_torch_device()`.

---

### 4. [HIGH · S] `LICENSE.txt` отсутствует, но `pyproject.toml` и README на него ссылаются

Files: [pyproject.toml:5](pyproject.toml:5),
       [README.md:14](README.md:14)

**What's wrong**: [`pyproject.toml:5`](pyproject.toml:5) объявляет `license = { file = "LICENSE.txt" }`, [`README.md:14`](README.md:14) показывает badge `[![License](.../license-see%20LICENSE.txt-...)](LICENSE.txt)`. Однако ни в рабочем дереве, ни в истории git (`git log --all --oneline -- LICENSE.txt LICENSE`) такого файла нет. `git ls-files | grep -i licens` пуст.

**Why it matters here**: (a) ComfyRegistry publish action в [.github/workflows/publish_action.yml](.github/workflows/publish_action.yml) вызывает `Comfy-Org/publish-node-action@main`, который читает `pyproject.toml` и валидирует license-ссылку; пуш с изменённым `pyproject.toml` может сломать публикацию. (b) `pip install` из исходников при включённой PEP 639 будет ругаться, что `License-File` указан, но файл недоступен. (c) Юридически — пользователи, форкающие пак, не имеют легитимного licensing statement.

**Recommendation**: либо закоммитить `LICENSE.txt` с реальным текстом лицензии, либо заменить строку в [`pyproject.toml:5`](pyproject.toml:5) на inline-объявление типа `license = "MIT"` (или подходящий SPDX-идентификатор) и убрать ссылку на файл. README badge подправить на корректный URL.

---

### 5. [MEDIUM · S] `pyproject.toml` без `requires-python` и `requires-comfyui`

Files: [pyproject.toml:1-23](pyproject.toml:1)

**What's wrong**: в `[project]` нет `requires-python`, в `[tool.comfy]` нет `requires-comfyui`. `pip install` соглашается ставить пак на любую Python-версию, после чего модули падают на import-time из-за PEP 585/604 синтаксиса (`list[str]`, `int | None`). CI Workflow [`.github/workflows/ci.yml:17`](.github/workflows/ci.yml:17) тестируется на 3.10/3.11/3.12 — то есть 3.10 является де-факто нижней границей, но pyproject это не отражает.

**Why it matters here**: пользователь на Python 3.9 (всё ещё поддерживается некоторыми ComfyUI portable сборками) получит `SyntaxError` или `TypeError: 'type' object is not subscriptable` после успешной установки — некорректный диагностический experience. Аналогично, ComfyRegistry рекомендует `requires-comfyui` для V3-паков, чтобы registry не предлагал пак пользователям на устаревших ComfyUI без `comfy_api.latest`.

**Recommendation**: в [pyproject.toml](pyproject.toml) добавить:

```toml
[project]
requires-python = ">=3.10"

[tool.comfy]
requires-comfyui = ">=X.Y"   # минимальная версия с comfy_api.v0_0_2, см. также finding #1
```

После пина `comfy_api.v0_0_2` (см. #1) `requires-comfyui` берётся из ComfyUI release notes, где появился `v0_0_2`.

---

### 6. [MEDIUM · S] `TS_Qwen3_VL_V3` определяет `__init__` и instance-state вопреки V3-конвенции

Files: [nodes/llm/ts_qwen3_vl.py:20-28](nodes/llm/ts_qwen3_vl.py:20),
       [nodes/llm/ts_qwen3_vl.py:55-61](nodes/llm/ts_qwen3_vl.py:55)

**What's wrong**: класс `TS_Qwen3_VL_V3` (наследник `IO.ComfyNode`) определяет `def __init__(self)` (строки 55-61), который инициализирует `self._logger`, `self._cache`, `self._cache_order`, `self._cache_max_items`, `self._snapshot_endpoint_supported`. По документации ComfyUI V3 `__init__` на классе ноды бессмыслен (класс санитизируется до execution, instance state не сохраняется). Код пака обходит это через `_get_instance()` (строки 23-28): `cls._instance = cls.__new__(cls); cls._instance.__init__()`. Это ручной singleton-pattern, который противоречит [`nodes/AGENTS.md:110-111`](nodes/AGENTS.md:110) (*"Avoid `__init__`. Avoid instance state."*) и [`CLAUDE.md`](CLAUDE.md) §5 (*"`__init__` не используется, состояние — class-level"*).

**Why it matters here**: workaround функционально работает — singleton живёт в `cls._instance`, методы вроде `process()` обращаются к `self._cache`, и Python кэш ссылок поддерживает state. Но: (a) читателю кода **сложнее** понять lifecycle (`define_schema/fingerprint_inputs/execute` — classmethod'ы, а `process` — instance метод на singleton); (b) если ComfyUI ужесточит V3-санитизацию (например, начнёт проксировать класс), workaround сломается; (c) правило в собственных `AGENTS.md`/`CLAUDE.md` теряет вес, если самая большая нода пака его нарушает.

**Recommendation**: убрать `__init__`, перенести `_logger`, `_cache_order`, `_snapshot_endpoint_supported` в class-level атрибуты, `_cache` оставить class-level dict. `process(self, ...)` и его helpers переписать как `process(cls, ...)` classmethod'ы. `_get_instance` удалить. Это согласуется с тем, как `TS_VideoDepth` ([`nodes/video/ts_video_depth.py:187-191`](nodes/video/ts_video_depth.py:187)) хранит `_loaded_model_instance`/`_loaded_model_filename`/`_model_on_device_type_str` чисто на class-level.

---

### 7. [MEDIUM · M] Декораторы `_register_get`/`_register_post` дублируются в 5 локациях

Files: [nodes/text/ts_prompt_builder.py:27-50](nodes/text/ts_prompt_builder.py:27),
       [nodes/text/ts_style_prompt_selector.py:24-34](nodes/text/ts_style_prompt_selector.py:24),
       [nodes/audio/loader/_audio_helpers.py:108-131](nodes/audio/loader/_audio_helpers.py:108),
       [nodes/image/lama_cleanup/_lama_helpers.py:116-139](nodes/image/lama_cleanup/_lama_helpers.py:116),
       [nodes/llm/super_prompt/_helpers.py:204-217](nodes/llm/super_prompt/_helpers.py:204)

**What's wrong**: пять модулей определяют идентичные по логике пары `_register_get(path)` / `_register_post(path)` декораторов. Каждая копия резолвит `PromptServer.instance` (через `try/except` или через ранее закэшированный `_PROMPT_SERVER`), wrap'ает декоратор в try/except для устойчивости, логирует warning при отсутствии `PromptServer`. Различия — только в имени логгера и префиксе сообщения. Логика идентична на ~25 LOC × 5 = ~125 LOC дублирования.

**Why it matters here**: пять копий означают, что любой fix (например, добавление CSRF-проверки, корректная обработка hot-reload `PromptServer`, выравнивание поведения при недоступности в standalone-режиме) надо синхронизировать в пяти местах. Это уже привело к мелким расхождениям: в `_audio_helpers.py` `_PROMPT_SERVER` кэшируется на module-import, в `ts_prompt_builder.py` resolution делается лениво внутри декоратора. Добавление новой ноды с HTTP routes сейчас требует копировать тот же блок шестой раз — вместо `from .._shared import register_get, register_post`. CLAUDE.md §7 прямо санкционирует shared-логику в `nodes/_shared.py` если используется ≥2 нодами; здесь ≥5.

**Recommendation**: добавить в [`nodes/_shared.py`](nodes/_shared.py) функции `register_get(path, *, log_prefix)` и `register_post(path, *, log_prefix)`, которые принимают `log_prefix` параметром (для сохранения per-node prefix'ов в warnings). Заменить локальные определения в пяти файлах на `from .._shared import register_get, register_post` (или `from ..._shared import ...` для двухуровневых подпапок). Параметр `log_prefix` оставляет per-node идентичность, не плодя экземпляры одного и того же кода.

---

### 8. [MEDIUM · L] `nodes/llm/ts_qwen3_vl.py` (1270 LOC) — кандидат на промоушен в подпапку

Files: [nodes/llm/ts_qwen3_vl.py](nodes/llm/ts_qwen3_vl.py) (1270 строк)

**What's wrong**: один файл содержит: (a) class definition + schema (~120 LOC), (b) `_load_presets` / `_load_processor_or_tokenizer` (preset loading, ~80 LOC), (c) `_load_model` / `_load_model_with_loader` + bnb quant config + attention mode resolver (~250 LOC), (d) HF model availability + download + offline-mode handling (~150 LOC), (e) image/video preprocess + size reduction + frame sampling (~200 LOC), (f) `process` + chat template + generation params (~250 LOC), (g) cache management + memory cleanup (~120 LOC), (h) bitsandbytes detection / dtype helpers (~100 LOC). [`CLAUDE.md`](CLAUDE.md) §7.1 прямо предписывает: *"когда нода становится god-file (~1000+ строк) **или** имеет существенный набор приватных helpers ... вынести её в подпапку"*. Файл пересекает оба критерия.

**Why it matters here**: это самый большой файл пака. Каждое изменение в одной из восьми перечисленных областей требует прокручивать 1270 LOC и удерживать их в голове. В пакете уже есть прецедент ровно такого split'а — `TS_SuperPrompt` был вынесен в [`nodes/llm/super_prompt/`](nodes/llm/super_prompt/) с `_helpers.py`/`_voice.py`/`_qwen.py`/`ts_super_prompt.py` (commit `deac607`). Тот же шаблон применим здесь: один публичный класс остаётся в `ts_qwen3_vl.py`, остальное — в `_loader.py`/`_preprocess.py`/`_quant.py`/`_helpers.py`.

**Recommendation**: создать подпапку `nodes/llm/qwen3_vl/` с layout по аналогии с `super_prompt/`:
- `ts_qwen3_vl.py` — публичный класс `TS_Qwen3_VL_V3` (`define_schema`/`execute`/`fingerprint_inputs` + `NODE_CLASS_MAPPINGS`).
- `_helpers.py` — preset loading, `_presets_path`, log_prefix, общие константы.
- `_loader.py` — HF download + offline mode + model class resolution + processor loading.
- `_quant.py` — bitsandbytes detection + dtype maps + attention mode helpers.
- `_preprocess.py` — image/video resize + frame sampling.
- `__init__.py` пустой.
- Loader подхватит автоматически благодаря `_`-prefix фильтру в [`__init__.py:38-42`](__init__.py:38).

После рефакторинга регенерировать `tests/contracts/node_contracts.json` через `python tools/build_node_contracts.py` — поле `python_file` обновится на `"nodes/llm/qwen3_vl/ts_qwen3_vl.py"`, остальной snapshot останется без изменений (это нормально и единственное публичное изменение).

---

## Quick wins

- [ ] #2 — `images.clone()` или замена in-place операций на out-of-place (полчаса работы + тест).
- [ ] #3 — удалить override блок в `_get_target_device` (5 строк + ручная проверка на CPU box'е).
- [ ] #4 — закоммитить `LICENSE.txt` или заменить на inline `license = "MIT"`.
- [ ] #5 — добавить `requires-python = ">=3.10"` и `requires-comfyui = ">=X.Y"` в pyproject.

---

## Things that look bad but are actually fine

- **Большие файлы как `ts_whisper.py` (1063 LOC), `ts_downloader.py` (806 LOC), `ts_color_match.py` (654 LOC)** — конвенция one-node-one-file. `ts_whisper.py` находится прямо у границы 1000 LOC (см. Open questions). `ts_downloader.py` и `ts_color_match.py` ниже порога.
- **Backwards-compat shim в [`nodes/llm/ts_super_prompt.py`](nodes/llm/ts_super_prompt.py)** — re-exports + `NODE_CLASS_MAPPINGS = {"TS_SuperPrompt": TS_SuperPrompt}` дублируют регистрацию. Автор явно задокументировал, почему второй проход loader'а безопасен (Python module cache + `_ts_runtime_guard_wrapped` flag в [`ts_dependency_manager.py:69`](ts_dependency_manager.py:69)). Корректно работает, нужен живым тестам [`tests/test_super_prompt_contract.py:117`](tests/test_super_prompt_contract.py:117) и [`tests/test_voice_recognition_audio.py:114`](tests/test_voice_recognition_audio.py:114).
- **`fingerprint_inputs` возвращает `float("nan")` или `mtime`** — корректные V3-идиомы для "always recompute" / "invalidate when file changes". См. [`nodes/image/ts_resolution_selector.py:51`](nodes/image/ts_resolution_selector.py:51) (NaN при наличии image), [`nodes/llm/ts_qwen3_vl.py:113-116`](nodes/llm/ts_qwen3_vl.py:113) (mtime presets-файла).
- **Class-level state как `_loaded_model_instance`/`_model_on_device_type_str`** в [`nodes/video/ts_video_depth.py:188-191`](nodes/video/ts_video_depth.py:188) — каноничный V3-singleton паттерн.
- **`node_id` со пробелами**: `"TS Cube to Equirectangular"`, `"TS Equirectangular to Cube"`, `"TS Files Downloader"`, `"TS Youtube Chapters"` — исторические identity, перечислены в [`CLAUDE.md`](CLAUDE.md) §6 как preserved exceptions; менять нельзя без миграции.
- **`_register_birefnet_folder()` вызов на module-load** в [`nodes/image/ts_bgrm_birefnet.py:40`](nodes/image/ts_bgrm_birefnet.py:40) — это лёгкая регистрация в `folder_paths`, обёрнута в try/except. Не тяжёлый side effect.
- **`subprocess.Popen(...)` без `shell=True` в [`nodes/audio/loader/_audio_helpers.py:318`](nodes/audio/loader/_audio_helpers.py:318)** — вызов ffmpeg с argument list, есть hard wall-clock cap (10 мин), есть kill-on-error path. Корректно.
- **HuggingFace `trust_remote_code=True` в [`nodes/llm/ts_qwen3_vl.py:948`](nodes/llm/ts_qwen3_vl.py:948), `importlib.util.spec_from_file_location` для BiRefNet config в [`nodes/image/ts_bgrm_birefnet.py:452-468`](nodes/image/ts_bgrm_birefnet.py:452)** — стандартный HF pattern для моделей с custom code; модели загружаются из жёстко указанных репозиториев, ответственность за trust лежит на пользователе, выбирающем модель.
- **`bare except:` в [`nodes/video_depth_anything/utils/dc_utils.py:12`](nodes/video_depth_anything/utils/dc_utils.py:12)** — vendored DepthCrafter код, не TS.
- **Loader walks `_PACKAGE_DIR.rglob("*.py")` без отдельного фильтра `.cache/`/`.test_*/`** — в этих папках 0 `.py` файлов на практике (проверено), startup cost негligible.
- **Loud startup table (~80 INFO строк)** в [`__init__.py:265-336`](__init__.py:265) — диагностический контент, помогает troubleshoot'ить missing deps.
- **`requirements.txt` без `requests`/`tqdm`** — оба попадают как транзитивные зависимости от `huggingface_hub`/`transformers` (которые declared); см. Open questions.

---

## Open questions for the maintainer

1. **`nodes/audio/ts_whisper.py` 1063 LOC — ровно у границы 1000 LOC порога из CLAUDE.md §7.1**. Есть смысл вынести в `nodes/audio/whisper/` подпапку (как `super_prompt/`) или оставить плоско? Файл реже меняется чем `ts_qwen3_vl.py` и пограничен по размеру.
2. **Несогласованность `torch.cuda.empty_cache()` vs `comfy.model_management.soft_empty_cache()`**. В пакете ~12 сайтов первого, ~5 второго (включая `ts_frame_interpolation.py:452/649/656`, `ts_whisper.py:975`). `soft_empty_cache` device-aware и ничего не ломает на CPU/MPS — но текущий код не падает, просто делает no-op на не-CUDA. Стоит ли пройтись и нормализовать на единый стиль?
3. **Документная несогласованность в `CLAUDE.md`**: в разделе Layout пишется "image (25)" и "audio (6)", но фактический подсчёт — 26 image + 5 audio (всего 57). README согласован с фактом. Поправить CLAUDE.md или это намеренная разметка?
4. **`requests` и `tqdm` импортируются на module-top в [`nodes/files/ts_downloader.py:12-15`](nodes/files/ts_downloader.py:12), но не задекларированы в [`requirements.txt`](requirements.txt) / `[project.dependencies]`**. На практике установлены транзитивно от `huggingface_hub`/`transformers`. Декларировать явно для надёжности или это неважно?
5. **`TS_AnimationPreview` и `TS_Whisper` хардкодят `"auto"/"cpu"/"cuda"` в device dropdown'ах** (см. [`nodes/audio/ts_whisper.py:763`](nodes/audio/ts_whisper.py:763)). MPS/ROCm пользователи видят бесполезные опции "cuda", при этом "auto" падает на CPU. Стоит расширить опции (`mps`, `auto-via-model_management`) или оставить — на pure-Mac/AMD платформах пак ограниченно полезен и без этого?
6. **JS-only нода `TS_Bookmark`** ([`js/utils/ts-bookmark.js`](js/utils/ts-bookmark.js)) не считается в публичном "57 нод" totalcount нигде в README/CLAUDE.md/migration.md. Технически это `LiteGraph` virtual node. Намеренно ли её исключают из счёта или стоит документировать?
7. **`ruff check` стоит `continue-on-error: true`** в [`.github/workflows/ci.yml:47`](.github/workflows/ci.yml:47). Комментарий обещает: *"Drop continue-on-error once the existing baseline is fixed"*. Какие именно baseline issues остались, и есть ли план их добить?
