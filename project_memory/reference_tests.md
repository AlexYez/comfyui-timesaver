---
name: Tests location and stubs
description: tests/ — pytest с monkeypatch-стабами для comfy_api/folder_paths/aiohttp; ts_tmp_path fixture; live API smoke на 127.0.0.1:8188; покрытие по категориям; 427 collected (418 passing + 9 skipped без live API) under ComfyUI Python (v9.12)
type: reference
originSessionId: 8022fd27-bafd-461a-97d9-dc12a4035284
---
Тесты пака — `tests/` (root):

**Существующие contract/behavior файлы:**
- `tests/conftest.py` — `ts_tmp_path` fixture (project-local temp под `tests/.cache/`, обходит ограничения `%LOCALAPPDATA%\Temp`).
- `tests/test_pack_imports.py` — auto-discovery loader (3 теста).
- `tests/test_node_contracts.py` — snapshot vs `tests/contracts/node_contracts.json` (4 теста). Регенерация: `tools/build_node_contracts.py`.
- `tests/test_no_mojibake.py` — анти-cp1251 регрессия для UTF-8 русских строк.
- `tests/test_super_prompt_contract.py` — TS_SuperPrompt contract.
- `tests/test_voice_recognition_audio.py` — voice recognition подсистема SuperPrompt.
- `tests/test_whisper.py` — TSWhisper helpers (CPU-safe, torch+transformers stubbed).
- `tests/test_downloader.py` — TS_DownloadFilesNode parsing helpers (no network).
- `tests/test_bgrm_node.py` — TS_BGRM_BiRefNet contract + GPU-conditional paths.
- `tests/test_lama_cleanup_contract.py` — TS_LamaCleanup schema/helpers/save/cleanup/lock.
- `tests/test_multi_reference_node.py` — TS_MultiReference behavior + tensor invariants.

**Категориальные файлы (добавлены в session 2026-05-06):**
- `tests/test_utils_nodes.py` — TS_Math_Int / TS_Smart_Switch / TS_Int_Slider / TS_FloatSlider (33 теста).
- `tests/test_image_nodes.py` — TS_GetImageMegapixels, TS_GetImageSizeSide, TS_ImageBatchCut, TS_Color_Grade, TS_Film_Emulation, TS_ImageResize, TS_ImageBatchToImageList (39 тестов).
- `tests/test_image_advanced_nodes.py` — TS_QwenSafeResize, TS_WAN_SafeResize, TS_QwenCanvas, TS_ResolutionSelector, TS_ImageTileSplitter/Merger (round-trip), TSAutoTileSize, TSCropToMask, TS_Keyer, TS_Despill, TS_ImagePromptInjector, TS_RestoreFromCrop helpers (45 тестов). Cube/Equirect — snapshot-only (relative imports `from ...ts_dependency_manager` ломают standalone import).
- `tests/test_text_files_audio.py` — TS_BatchPromptLoader, TS_PromptBuilder helpers, TS_StylePromptSelector, TS_SileroStress, TS_FilePathLoader, TS Youtube Chapters, TS_ModelConverter/Scanner schema, audio nodes contract (31 тест).
- `tests/test_video_llm_nodes.py` — TS_Free_Video_Memory, TS_RTX_Upscaler, TS_Video_Upscale_With_Model schema, video deps snapshot-only (13 тестов).
- `tests/test_comfyui_live_api.py` — live ComfyUI smoke (127.0.0.1:8188): /api/object_info полный сравнительный test, /api/system_stats, widget defaults drift (9 тестов). Скипается если ComfyUI не запущен.

**Total на v9.12: 427 collected, 418 passing + 9 skipped под ComfyUI Python без live API** (исключая test_browser_smoke.py + test_comfyui_live_api.py — оба требуют запущенного ComfyUI на 127.0.0.1:8188). Включая test_static_invariants.py, test_qwen_engine.py, test_video_depth_helpers.py, test_image_advanced_nodes.py. Запуск: `--ignore=tests/test_browser_smoke.py --ignore=tests/test_comfyui_live_api.py` когда ComfyUI не активен.

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
