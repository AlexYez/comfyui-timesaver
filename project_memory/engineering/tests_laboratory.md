---
name: tests_laboratory
description: "История локальной тестовой лаборатории: фикстуры, окружение, импорты и измеренные прогоны."
metadata:
  type: reference
---

История локальной тестовой лаборатории: фикстуры, окружение, импорты и измеренные прогоны.

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

⚠️ **Устарело в части состава нод.** 11.08.2026 с пака сняты 16 нод (см. [reference_removed_nodes_2026_08.md](../history/reference_removed_nodes_2026_08.md)); упомянутые ниже снятые ноды и их файлы больше не существуют. Остальное в записи верно.
Тесты пака — `tests/` (root):

**Существующие contract/behavior файлы:**
- `tests/conftest.py` — `ts_tmp_path` fixture (project-local temp под `tests/.cache/`, обходит ограничения `%LOCALAPPDATA%\Temp`).
- `tests/test_pack_imports.py` — auto-discovery loader (3 теста).
- `tests/test_node_contracts.py` — snapshot vs `tests/contracts/node_contracts.json` (4 теста). Регенерация: `tools/build_node_contracts.py`.
- `tests/test_no_mojibake.py` — анти-cp1251 регрессия для UTF-8 русских строк.
- `tests/test_super_prompt_contract.py` — TS_SuperPrompt contract.
- `tests/test_voice_recognition_audio.py` — voice recognition подсистема SuperPrompt.
- `tests/test_whisper.py` — TSWhisper helpers (12 тестов; **self-contained** — стабит `comfy_api.v0_0_2.IO` + `folder_paths` + `srt`, БЕЗ ручной предзагрузки `comfy`). Покрывает `_safe_float`/`_safe_int`/`_parse_temperature`/`_normalize_text`/`_remove_unwanted_phrases`/`_seconds_to_ttml_time`/`_create_ttml_content` + schema. Переписан 2026-06-23: старые тесты ссылались на удалённые при рефакторинге `_merge_segments`/`_build_generate_kwargs`/`_is_oom_error` и требовали реального `comfy` на path.
- `tests/test_downloader.py` — TS_DownloadFilesNode parsing helpers (no network).
- `tests/test_bgrm_node.py` — TS_BGRM_BiRefNet contract + GPU-conditional paths.
- `tests/test_lama_cleanup_contract.py` — TS_LamaCleanup schema/helpers/save/cleanup/lock.
- `tests/test_multi_reference_node.py` — TS_MultiReference behavior + tensor invariants.

**Категориальные файлы (добавлены в session 2026-05-06):**
- `tests/test_utils_nodes.py` — TS_Math_Int / TS_Smart_Switch / TS_Int_Slider / TS_FloatSlider (33 теста).
- `tests/test_image_nodes.py` — TS_GetImageMegapixels, TS_GetImageSizeSide, TS_ImageBatchCut, TS_Color_Grade, TS_Film_Emulation, TS_ImageResize, TS_ImageBatchToImageList (39 тестов).
- `tests/test_image_advanced_nodes.py` — TS_QwenSafeResize, TS_WAN_SafeResize, TS_QwenCanvas, TS_ResolutionSelector, TS_ImageTileSplitter/Merger (round-trip), TSAutoTileSize, TSCropToMask, TS_Keyer, TS_Despill, TS_ImagePromptInjector, TS_RestoreFromCrop helpers (45 тестов). Cube/Equirect — snapshot-only (relative imports `from ...ts_dependency_manager` ломают standalone import).
- `tests/test_text_files_audio.py` — TS_BatchPromptLoader, TS_PromptBuilder helpers, TS_StylePromptSelector, TS_SileroStress, TS_FilePathLoader, TS Youtube Chapters, TS_ModelConverter/Scanner schema, audio nodes contract (31 тест).
- `tests/test_video_llm_nodes.py` — TS_RTX_Upscaler, TS_Video_Upscale_With_Model schema, video deps snapshot-only (TS_Free_Video_Memory + его тесты удалены в 9.31).
- `tests/test_comfyui_live_api.py` — live ComfyUI smoke (127.0.0.1:8188): /api/object_info сравнение со snapshot, /api/system_stats, widget defaults drift, **+ res_2s/bong_tangent зарегистрированы в KSampler**. Скипается если ComfyUI не запущен.
- `tests/test_browser_smoke.py` — headless playwright (Chromium, locale=en-US): каждая нода переживает LiteGraph.createNode, ноль console/page errors, DOM-виджеты, HTTP-роуты, **+ res_2s/bong_tangent видны в виджетах KSampler во фронтенде**. Требует запущенного ComfyUI + playwright.
- `tests/test_res4lyf_inject.py` — unit: регистрация res_2s (samplers) / bong_tangent (schedulers) через stub comfy; кривая bong_tangent (steps+1, убывание, scale на sigma_max); численная корректность res_2s (экспоненциальный RK2: точность на константном денойзере + 2-й порядок сходимости).

**Total на v9.94: 835 passed под ComfyUI Python при ЗАПУЩЕННОМ ComfyUI (:8188) + playwright; 738 passed offline (0 skipped)** (live API + browser smoke прогоняются и зелёные). Без запущенного ComfyUI — `test_comfyui_live_api.py` + `test_browser_smoke.py` авто-скипаются (или `--ignore=...`). Включая test_static_invariants.py, test_qwen_engine.py, test_video_depth_helpers.py, test_image_advanced_nodes.py, test_res4lyf_inject.py.

**✅ FIXED 2026-06-23 (`task_faa75928`): 11 устаревших тестов синхронизированы с кодом** в `test_utils_nodes.py`, `test_downloader.py`, `test_super_prompt_helpers.py`. Что было и как приведено к текущему контракту нод:
- TS Smart Switch (`ts_smart_switch.py`) при НИ ОДНОМ валидном входе теперь **райзит RuntimeError** (не return None). `test_smart_switch_*` переписаны: «both invalid» → `pytest.raises(RuntimeError)`; «rejects wrong type / 3d / plain dict / bool» → дают валидный `input_2` и проверяют **failover** на него (точнее доказывает rejection, чем зависимость от None).
- TS Math Int (`ts_math_int.py`) при делении на ноль райзит **ZeroDivisionError**, при неизвестной операции — **ValueError** (не silent 0). Тесты → `pytest.raises`.
- Downloader (`ts_downloader.py`): UA сменён со spoofed-Chrome (`Mozilla…`) на `comfyui-timesaver/{version}` (чтобы не триггерить registry security scanner). Тест ассертит `comfyui-timesaver` в UA.
- `is_model_cached`: делегирует в `_whisper_engine.is_model_cached` → `model_file_path`. Тест теперь патчит `voice._engine.model_file_path` (реальный `_whisper_engine`, он НЕ стабится — стабится только `_qwen_engine`), а не `voice._model_file_path` (тонкая обёртка, которую `is_model_cached` не вызывает).

Полный прогон после фикса: **557 passed / 1 failed** (единственный fail — browser-smoke THREE.Object3D, см. ниже).

**Browser smoke fail-фильтр:** `test_no_unfiltered_console_errors_during_smoke` падает на `THREE.Object3D.add ... MeshModelAdapter/MainModel` — это сторонний 3D node-pack в custom_nodes (`qwenmultiangle`/`WanVideoWrapper`/core Load3D), не TS-нода. Окружение-специфично.

**Паттерн contract-тестов:**
- pytest + `monkeypatch.setitem(sys.modules, ...)` для подмены `comfy_api`, `comfy_api.latest`, `folder_paths`, `aiohttp`/`aiohttp.web`, `comfy.model_management`, `comfy.utils`, `server`.
- Стабы `_IO`, `_Schema`, `_Input`, `_Output`, `_NodeOutput`, `_UploadType`, `_NumberDisplay`, `_UI` (для preview), `_AnyType`, `_Custom` имитируют V3 API без зависимостей ComfyUI.
- `_install_stubs()` — guard `if "comfy_api.latest" in sys.modules: return` сохраняет реальный API при наличии (ComfyUI Python).
- `monkeypatch.syspath_prepend(root)` чтобы импорты `nodes.image.*` работали из корня пака.
- Skip tests, требующие numpy/torch/PIL, через `pytest.importorskip("numpy")` etc.
- Для тестов file-IO: `ts_tmp_path` fixture + `_make_working_file` локальный helper.
- Для тестов с временем: `monkeypatch.setattr(helpers.time, "strftime", lambda fmt: "FIXED")`.
- Для тестов с asyncio.Lock: `asyncio.run(runner())` где `runner` — async-обёртка.

**Limitation: relative imports beyond top-level**
Ноды, использующие `from ...ts_dependency_manager import TSDependencyManager` (cube/equirect, music_stems), нельзя импортировать в standalone test runner — pkg родитель не загружен. Используй snapshot-only тест: проверь что node_id присутствует в `tests/contracts/node_contracts.json`.

**Запуск:**
```bash
D:/AiApps/ComfyUI/comfyui/python/python.exe -m pytest tests
D:/AiApps/ComfyUI/comfyui/python/python.exe -m compileall .
D:/AiApps/ComfyUI/comfyui/python/python.exe tools/build_node_contracts.py --check
```

**How to apply:**
- Для новых V3 нод повторять паттерн стабов из `test_super_prompt_contract.py` / `test_lama_cleanup_contract.py` / `test_utils_nodes.py`.
- Не добавлять тесты, требующие реальных моделей или сети, в обычный набор. Помечать как integration/manual.
- Snapshot-тесты контрактов: после изменения V3 schema регенерировать через `tools/build_node_contracts.py`.
- Live API smoke (`test_comfyui_live_api.py`) — обязателен после любого изменения схемы; покажет drift между snapshot и фактической регистрацией.

---

## Инфраструктура: как вообще запускается и что мешает

(перенесено из `testing-infrastructure.md` при консолидации 2026-08-04 —
факты те же, файл больше не нужен)

Тесты живут в `tests/` (сознательно untracked — см. .gitignore) и запускаются ТОЛЬКО
embedded-питоном ComfyUI: `D:/AiApps/ComfyUI/comfyui/python/python.exe -m pytest tests/ -q`
(системный python не имеет psutil/torch-зависимостей comfy).

**Импорт пака в тестах** — через синтетический пакет, иначе `nodes` конфликтует
с ComfyUI-шным `nodes.py`:
```python
_PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMFY = os.path.dirname(os.path.dirname(_PACK))
sys.path.insert(0, _COMFY)
pkg = types.ModuleType("tspack"); pkg.__path__ = [_PACK]; sys.modules["tspack"] = pkg
mod = importlib.import_module("tspack.nodes.files.ts_model_converter")
```
Корневой `__init__.py` грузить через `spec_from_file_location(..., submodule_search_locations=None)`
— тогда `_STANDALONE_IMPORT=True` и лоадер нодов не запускается.

**Ограничения песочницы:** pytest-фикстура `tmp_path` падает с PermissionError
(нет доступа к %TEMP%) — для чистых path-функций использовать литеральные пути.

**Смоук всего пака** (62 модуля, 59 нод на 2026-06-11):
```python
sys.path.insert(0, "."); sys.path.insert(0, "custom_nodes")  # из корня ComfyUI
importlib.import_module("comfyui-timesaver")  # читать _MODULE_LOAD_RESULTS
```

Ruff-базлайн (НЕ мои регрессии, существовали до фиксов 2026-06-11): T201 prints в
_ideogram_helpers (CLI-хелперы), F841 `n` и
F821 guarded-del `frames_gpu` в ts_video_depth, F401 в super_prompt/ts_super_prompt.
(F841 weights_filename в birefnet исправлен в series-5, 2026-06-11.)

Series-5 (ревью-фиксы W1-W4, S3-S6, S8) закоммичены 2026-06-11; их тесты —
tests/test_series5_review_fixes.py (untracked, как и вся tests/). CI получил
smoke-import джоб (клонирует ComfyUI, CPU torch, импорт пака через лоадер).

**Фикс smoke-import (2026-06-19, коммит e7b8921):** job падал с момента добавления
(S8) — CI ставит CPU-only torch, а `comfy.model_management` при импорте дёргает
`torch.cuda.current_device()` → `AssertionError: Torch not compiled with CUDA enabled`
→ все ~21 модуля с `import comfy.model_management` уходили в ERROR (nodes=40, errors=21).
Лечится `from comfy.cli_args import args; args.cpu = True` ДО импорта пака (model_management
читает `args.cpu` на строке 156-157 при импорте, ставит cpu_state=CPU). CPU-путь
воспроизводится локально тем же `args.cpu=True` даже на CUDA-машине: `nodes=60 errors=0`.

**JS-виджеты тоже тестируемы** — node v24 есть в системе. Чистые функции из
`js/.../_slider_helpers.js` (нет top-level импортов кроме как внутри
`tsRegisterSliderExtension`) импортируются напрямую в `.mjs`-тест и гоняются
`node tests/<file>.mjs`. Пример: `tests/test_slider_step.mjs` мокает node+widget с
РЕАЛЬНЫМИ опциями V3-фабрики (`step=realStep*10`, `step2=realStep`, `round`,
`precision`) и проверяет округление формулой litegraph
`Number((Math.round(v/round)*round).toFixed(precision))`. Истину по фронтенду
смотреть в source-maps `comfyui_frontend_package/static/assets/*.js.map`
(litegraph — `api-*.js.map`, число-виджет — `WidgetInputNumber-*`, фабрика —
`dialogService-*`).
