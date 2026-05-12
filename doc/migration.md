# Migration Notes

> **Update v9.2:** import surface pinned from `comfy_api.latest` to `comfy_api.v0_0_2`. Все примеры в этом документе с `from comfy_api.latest import IO` относятся к момиту 8.9; в текущем коде используется `from comfy_api.v0_0_2 import IO`. Поведение API идентично, namespace pinned для предсказуемости при обновлениях ComfyUI core.

## 8.8 → 8.9 — Full V1 → V3 API migration

Релиз `8.9` переводит **все 57 нод** на ComfyUI V3 API (`comfy_api.latest.IO`). До этого пак был смешанным (V1 + V3). После миграции `grep RETURN_TYPES nodes/` пуст.

### Что НЕ изменилось (workflow совместимость)

`/object_info` сравнение «до vs после» идентично с точностью до V3-косметики (`api_node`/`deprecated`/`experimental` → false вместо null, COMBO в формате `["COMBO", {options, multiselect}]` вместо `[[options]]`, STRING explicit `multiline:false`). Это wire-format V3, поведение и значения совпадают:

- `node_id` — без изменений у всех 57 нод.
- Имена Python-классов — без изменений.
- Имена входов / выходов / типы / defaults / min/max/step / COMBO options / BOOLEAN labels — без изменений.
- `CATEGORY` — без изменений.
- JS-extension IDs (`ts.bookmark`, `ts.audioLoader`, `ts.lamaCleanup`, `ts.float-slider`, `ts.int-slider`, `ts.prompt_builder`, `ts_suite.style_prompt_selector`, `ts.superPrompt`, `ts.animationpreview`, `ts.resolutionselector`) — без изменений.
- TILE_INFO / CROP_DATA — переведены на `IO.Custom("...")` с теми же type-ID именами; sockets совместимы.

Сохранённые workflow JSON открываются без ручных правок.

### Что изменилось внутри (API)

- `INPUT_TYPES` (V1 dict) → `define_schema(cls) -> IO.Schema(...)` с явными `IO.X.Input(...)` объектами.
- `RETURN_TYPES` / `RETURN_NAMES` / `FUNCTION` / `CATEGORY` (V1 class-attrs) → поглощены в `IO.Schema(outputs=[...], category=...)`.
- `def execute(self, ...)` (V1 instance method, имя из `FUNCTION`) → `@classmethod def execute(cls, ...) -> IO.NodeOutput`.
- `IS_CHANGED` → `fingerprint_inputs` (`@classmethod`).
- `VALIDATE_INPUTS` → `validate_inputs` (`@classmethod`).
- `OUTPUT_NODE = True` → `IO.Schema(is_output_node=True)` (с `outputs=[]` если у ноды нет графовых выходов).
- `INPUT_IS_LIST = True` → `IO.Schema(is_input_list=True)`.
- `OUTPUT_IS_LIST = (True, False, ...)` → `IO.X.Output(is_output_list=True)` per-output.
- Hidden inputs (`"hidden": {"prompt_graph": "PROMPT"}`) → `IO.Schema(hidden=[IO.Hidden.prompt])`, читаются через `cls.hidden.prompt`.
- Wildcard `"*"` → `IO.AnyType.Input(...)` / `IO.AnyType.Output(...)`.
- Custom типы (TILE_INFO, CROP_DATA) → `IO.Custom("TILE_INFO")` / `IO.Custom("CROP_DATA")` модульные синглтоны.
- `__init__` для runtime state (model cache, resamplers, pipeline) → перенесён на class-level (`cls._model`, `cls._cache` и т.п.) с lazy-инициализацией в `execute`. ComfyNode V3 не вызывает `__init__` для каждого запуска.
- `display: "slider"` (V1 widget option) → `display_mode=IO.NumberDisplay.slider` в `IO.Int.Input` / `IO.Float.Input`.
- `display_name` для каждого Output задаётся явно (V3 не fallback'ает на тип) — для нод с пустым `RETURN_NAMES` использовали имя типа: `"IMAGE"`, `"MODEL"`.

### Pre-existing баги, починены попутно

| Что | Файл(ы) | Симптом до 8.9 |
|---|---|---|
| `from ..ts_dependency_manager` (две точки вместо трёх) | `nodes/audio/ts_music_stems.py`, `nodes/image/ts_cube_to_equirect.py`, `nodes/image/ts_equirect_to_cube.py` | Все три ноды тихо SKIPPED loader-ом (Missing dependency) |
| `from .frame_interpolation_models` (одна точка вместо двух) | `nodes/video/ts_frame_interpolation.py` | Нода SKIPPED |
| `from typing import Any` отсутствовал | `nodes/text/ts_silero_stress.py` | NameError при загрузке, нода SKIPPED |
| `_slider_helpers.tsSnapToStep` snap к min при огромном диапазоне | `js/utils/sliders/_slider_helpers.js` | `TS_FloatSlider` default 0.5 → 1.0 при добавлении на canvas |
| `_slider_helpers.tsApplyConfig` precision = countDecimals(step) | `js/utils/sliders/_slider_helpers.js` | UI отображал float как int если step после ComfyUI ×10 терял дробную часть |
| `from collections.abc import Mapping` отсутствовал | `nodes/video/ts_animation_preview.py` | NameError на редком audio-input branch |
| Stale тесты на старую категорию `Timesaver/Image Tools` | `tests/test_bgrm_node.py` | Pytest падал на assertion |
| Stale тесты на устаревший comfy_api (`Input.kwargs`, `NodeOutput.values`) | `tests/test_multi_reference_node.py` | Pytest падал на attribute |

### Pending

- `nodes/video/ts_frame_interpolation.py` хелпер-пакет лежит в `nodes/frame_interpolation_models/`. Поднимать его в `nodes/video/frame_interpolation_models/` рискованнее, чем оставить относительный импорт `..frame_interpolation_models` — отложено до отдельной задачи.

### Verification

```bash
python -m compileall .
python -m pytest tests                # 93 passed
python tools/build_node_contracts.py  # 57 V3 contracts
grep -rn "RETURN_TYPES" nodes/        # пусто
```

После обновления старые workflow JSON открываются без правок.

---

## 8.7 → 8.8 — One-node-one-file restructuring

Релиз `8.8` фиксирует строгое правило проекта: **одна публичная нода = один основной `.py` файл, плюс опционально один `.js`**.

### Что НЕ изменилось (workflow совместимость)

Всё, на что ссылаются сохранённые workflow JSON, осталось 1-в-1:

- `node_id` (ключи в `NODE_CLASS_MAPPINGS`) — без изменений у всех 57 нод.
- Имена Python-классов — без изменений.
- `INPUT_TYPES` / `RETURN_TYPES` / `RETURN_NAMES` / `CATEGORY` — без изменений.
- Default widget values — без изменений.
- V3 schema (`define_schema`) — без изменений.
- Имена параметров `execute()` / `FUNCTION` — без изменений.
- `WEB_DIRECTORY = "./js"` — без изменений (ComfyUI сканирует подпапки рекурсивно).

Ваши старые workflows должны открываться без ручных правок.

### Что изменилось

#### 1. Структура каталогов

`nodes/` теперь делится на категории:

```text
nodes/
├─ image/    — 25 нод (resize, color, keyer, mask, tile, 360°, и т.п.)
├─ video/    — 7 нод (depth, upscale, frame interpolation, animation preview, free memory, LTX, RTX upscaler)
├─ audio/    — 5 нод (loader, preview, whisper, silero TTS, music stems)
├─ llm/      — 2 ноды (Qwen3 VL, Super Prompt)
├─ text/     — 4 ноды (prompt builder, batch loader, style selector, silero stress)
├─ files/    — 8 нод (downloader, model converters, scanner, LoRA merger, EDL→YouTube, file path loader)
├─ utils/    — 4 ноды (float/int slider, smart switch, math int)
└─ conditioning/ — 1 нода (multi reference)
```

`js/` зеркальная структура.

Виртуальная нода `TS_Bookmark` (только JS, без backend) живёт в `js/utils/ts-bookmark.js`.

#### 2. Расщепление multi-class файлов

Старые «суммарные» файлы (`ts_image_tools_node.py`, `ts_image_resize_node.py`, и т.д.) разделены на отдельные файлы по одной ноде. Маппинг старого пути на новый — см. `tests/contracts/node_contracts.json`.

#### 3. Имя файла Qwen3 VL

`nodes/ts_qwen3_vl_v3_node.py` → `nodes/llm/ts_qwen3_vl.py`.

`node_id` остался `TS_Qwen3_VL_V3` (V3 в имени относится к версии модели Qwen, не к ComfyUI Node API).

#### 4. JS extension IDs — изменения для slider

| Старый ID | Новый ID | Нода |
| --- | --- | --- |
| `ts.slider-settings` | `ts.float-slider` | TS_FloatSlider |
| `ts.slider-settings` | `ts.int-slider` | TS_Int_Slider |

Было одно общее расширение, стало два по одному на ноду. Сохранённые workflow от этого не страдают (extension IDs не сериализуются в workflow JSON; пользовательские property-значения `min/max/step/default` сохраняются на самой ноде).

Остальные extension IDs не изменились:

| Файл (новый путь) | Extension ID |
| --- | --- |
| `js/utils/ts-bookmark.js` | `ts.bookmark` |
| `js/image/ts-resolution-selector.js` | `ts.resolutionselector` |
| `js/audio/ts-audio-loader.js` | `ts.audioLoader` |
| `js/video/ts-animation-preview.js` | `ts.animationpreview` |
| `js/text/ts-prompt-builder.js` | `ts.prompt_builder` |
| `js/text/ts-style-prompt.js` | `ts_suite.style_prompt_selector` |
| `js/llm/ts-super-prompt.js` | `ts.superPrompt` |

#### 5. Новые приватные модули (не зарегистрированы как ноды)

Модули с префиксом `_` пропускаются loader-ом и используются как shared helpers:

- `nodes/_shared.py` — `TS_Logger` (используется slider/switch/math/animation_preview).
- `nodes/image/_keying_helpers.py` — `gaussian_blur_4d`, `CHANNEL_TO_INDEX` (Keyer + Despill).
- `nodes/audio/_audio_helpers.py` — module-level helpers и aiohttp routes для Audio Loader/Preview.

#### 6. Loader (`__init__.py`) изменения

- Сканирование `nodes/` теперь рекурсивное (`rglob`).
- Файл-нода обязан начинаться с `ts_` (другие .py пропускаются — это защищает `frame_interpolation_models/`, `video_depth_anything/` и приватные `_*.py`).

### Pending tasks

- **`js/audio/ts-audio-loader.js`** — пока обслуживает обе ноды (`TS_AudioLoader` и `TS_AudioPreview`) одним extension'ом `ts.audioLoader`. Полное разделение этого 1100-строчного UI-модуля отложено в отдельную задачу: его нужно разрабатывать с проверкой через ComfyUI в браузере.

### Verification

```bash
python -m compileall .
python -m pytest tests
python tools/build_node_contracts.py --check
```

После обновления:

1. Откройте старый workflow JSON.
2. Убедитесь, что все `TS_*` ноды резолвятся (нет красных «missing»).
3. Проверьте, что значения widget'ов на сдвинутых нодах те же, что были.
4. Если используете `TS_FloatSlider` / `TS_Int_Slider` — убедитесь, что property `min/max/step/default` сохранены на нодах.

При проблемах:

- `python tools/build_node_contracts.py --check` покажет, какой контракт сломан.
- `tests/contracts/node_contracts.json` содержит точное соответствие `node_id → python_file`.

---

## Disk cleanup notes (после удалённых нод)

После удаления некоторых нод на диске пользователя могут остаться кэш-каталоги, не покрытые `.gitignore`. Они безвредны, но занимают место. Удалить вручную при необходимости:

| Каталог | Откуда | Безопасно удалить |
| --- | --- | --- |
| `nodes/.cache/tsfb_thumbnails/` | `TS_FileBrowser` (удалена ранее) | Да |
| `nodes/files/.cache/tsfb_thumbnails/` | `TS_FileBrowser` (старое расположение) | Да |
| `nodes/.cache/ts_audio_loader/` | `TS_AudioLoader` preview cache | Да, пересоздастся при первом использовании |

`TS_DeflickerNode` удалена в одной из revision'ов после 8.8 (импорт `nodes/rife/` отсутствовал и был сломан). Соответствующих кэш-каталогов на диске нода не оставляла. Сохранённые workflow с этой нодой теперь покажут «missing node» — ремап не предусмотрен.

---

## После 8.9 — релизы 9.x

Эти изменения не ломают workflow JSON, но меняют внутреннее устройство пакета и/или ассортимент нод. Полный список — `git log --oneline`.

| Релиз | Что | Workflow impact |
| --- | --- | --- |
| 9.1 | Version bump + tests/tools re-tracked | Нет |
| 9.2 | Pin `comfy_api.v0_0_2` + V3 lock_class fix + static invariants | Внутренний refactor; ноды с per-class кешем (`TS_BGRM_BiRefNet`, `TS_Whisper`, `TS_VideoDepth`, `TS_Qwen3_VL_V3`) переведены на module-level `_NodeState`. Симптом до фикса: `AttributeError: Cannot modify class attribute '...' on locked class` |
| 9.3 | `TS_LamaCleanup` — pure PyTorch + safetensors loader | Нет (та же schema). Architecture перенесена в `nodes/image/lama_cleanup/_lama_arch.py`, отказ от внешнего `lama-cleaner` package. Веса: `.safetensors` в `models/lama/` (раньше pickled `.ckpt`). |
| 9.4 | `TS_VideoDepth` GPU overhaul + новые image-ноды (**TS_Matting_ViTMatte**, **TS_SAM_MediaLoader**) | Новые публичные ноды; image count 26 → 28; total 57 → 59. `TS_BGRM_BiRefNet` — расширены опции (refine_foreground убран, новые controls — schema совместима со старыми workflow через defaults). |
| 9.5 | `TS_DownloadFilesNode` hardening + LLM stack refactor | (1) `TS_DownloadFilesNode`: zip-slip защита, sanitize filename, per-mirror connectivity probe, SHA256 progress UI. (2) **LLM stack refactor**: `nodes/llm/_qwen_engine.py` — новый shared Qwen pipeline (~1250 строк), используется и в `TS_Qwen3_VL_V3` (`nodes/llm/ts_qwen3_vl.py` стал тонкой обёрткой), и в `TS_SuperPrompt` (`nodes/llm/super_prompt/_qwen.py`). Public schema/выходы обеих нод не изменились. `nodes/llm/super_prompt/` split-подпакет; легаси-путь `nodes/llm/ts_super_prompt.py` — backward-compat re-export shim с пустым `NODE_CLASS_MAPPINGS`. |
| 9.6 | `TS_AudioLoader` / `TS_AudioPreview` — фикс сброса crop при переключении workflow-вкладок + документация | Hidden widgets (`crop_start_seconds`, `crop_end_seconds`, `preview_state_json`) теперь читаются с fallback на `node.properties` (защита от V2 Vue serialize quirks). `loadedGraphNode` вызывает `_tsAudioLoaderRehydrate` вместо повторного `setupAudioLoader` — DOM не пересоздаётся, не двоится верхняя часть ноды. `applyMediaPayload` больше не сбрасывает crop при `cropEnd === -1` (sentinel «до конца»). Документация: CLAUDE.md обновлён под v9.6, добавлены §12.5.12 (rehydrate) и §12.5.13 (hidden widgets fallback). README/migration.md актуализированы под текущие 59 нод. |

Для всех 9.x релизов: `node_id`, имена входов/выходов, defaults, category, JS extension IDs не менялись. Контракты ловятся через `tests/contracts/node_contracts.json` (59 V3 нод на v9.6) и `tests/test_node_contracts.py` (snapshot test).
