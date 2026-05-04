# TECH_DEBT_AUDIT.md — comfyui-timesaver

Аудит выполнен в read-only режиме на ветке `master`, релиз `8.8`. Артефакты не модифицировались, кроме самого этого файла. Все цитаты — `путь:строка`.

## Executive summary

1. **Уязвимость локальной утечки файлов** — `/ts_audio_loader/view` и `/ts_audio_loader/metadata` принимают произвольный `filepath` и отдают любой файл, доступный процессу ComfyUI. ComfyUI слушает по умолчанию `127.0.0.1`, но в LAN-конфигурации это уже эксплуатируемый path traversal. **Critical.**
2. **`TS_DeflickerNode` режим `rife_interpolation` падает на любом запуске** — внутри `nodes/video/ts_deflicker.py:14,20` импорт `from ..rife.warplayer import warp` и `from ..rife.IFNet_HDv3 import IFNet`, но каталог `nodes/rife/` отсутствует в репозитории. Любой пользователь, выбравший этот метод в дропдауне, получает рантайм-краш. **Critical.**
3. **Mojibake в UI 8 нод** — русские tooltips и descriptions сохранены двойной кодировкой (UTF-8 над cp1251). Пользователи видят текст вида `Р РµС„РµСЂРµРЅСЃ` вместо `Референс` (см. `ts_color_match.py:359-372`). **High.**
4. **`torch.load()` без `weights_only=True`** в `ts_deflicker.py:22` — небезопасное исполнение pickle при загрузке checkpoint, особенно после смены дефолта в torch 2.6. **High.**
5. **CATEGORY rot** — 19 нод из 57 используют категории вне правила CLAUDE.md §6 (`TS/<подкатегория>`): `image`, `image/resize`, `image/processing`, `Image/Color`, `Image Adjustments/Grain`, `Video PostProcessing`, `video`, `Tools/Video`, `conditioning/video_models`, `conversion`, `Model Conversion`, `file_utils`, `utils/Tile Size`, `utils/text`, `utils/model_analysis`, `Tools/TS_Image`, `Tools/TS_IO`, `Tools/TS_Video`, `Timesaver/Image Tools`. Из-за фриза §4 миграция требует явного решения, но текущий хаос документирован как факт. **Medium.**
5. **Массовое нарушение §13 «print() запрещён»** — `print()` для логирования встречается в 16 файлах (см. таблицу). **Medium.**
6. **Module-level side effects** — `nodes/audio/_audio_helpers.py:69-71` создаёт три директории (`nodes/.cache/ts_audio_loader`, `input/ts_audio_loader_recordings`, `nodes/.cache/ts_audio_loader/generated_audio`) при импорте. Импорт ноды для контракт-теста или для read-only анализа уже мутирует файловую систему. **High.**
7. **Дрейф `pyproject.toml` ↔ `requirements.txt`** — `silero-stress` указан в `requirements.txt:20`, но отсутствует в `pyproject.toml [project] dependencies`. ComfyRegistry-публикация недополучит зависимость, и нода `TS_SileroStress` упадёт у пользователей, ставивших через registry. **High.**
8. **Документационный дрейф удалённой `TS_FileBrowser`** — `README.md:323`, `doc/migration.md:34`, `CLAUDE.md:421` всё ещё ссылаются на нод/extension `TS_FileBrowser` / `ts.filebrowser`, хотя сама нода удалена из репозитория. **Medium.**
9. **Мёртвый legacy-loader** — `__init__.py:24,27-28,58-72` пытается обнаружить `*_node.py` и `ts_resolution_selector.py` в корне пакета, но после рефакторинга 8.7→8.8 в корне нет ни одного такого файла. Код недостижим. **Low.**
10. **ANSI-коды в логах** — `nodes/audio/ts_whisper.py:21-25,87-93` хранит `\x1b[36m`/`\x1b[31m` константы и инжектит их в строки лога, нарушая CLAUDE.md §15. **Low.**

## Архитектурная ментальная модель

`comfyui-timesaver` — production-quality пак ComfyUI custom nodes (57 нод, mix V1+V3 API). Точка входа — корневой `__init__.py`, делает рекурсивный auto-discovery `nodes/**/ts_*.py`, регистрирует `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS` из каждого модуля, оборачивает их через `TSDependencyManager.wrap_node_runtime()` (V1 → typed fallback, V3 → нормализованный `RuntimeError`). На старте печатает табличный отчёт «Module/Status/Nodes/Details + Import audit».

Layout новых V3-нод (после релиза 8.8) — строгий: одна нода = один `ts_<name>.py` в `nodes/<категория>/`, плюс опциональный `js/<категория>/ts-<name>.js`. Категории: `image/`, `video/`, `audio/`, `llm/`, `text/`, `files/`, `utils/`, `conditioning/`. Приватные shared-модули с префиксом `_` пропускаются loader'ом (`_shared.py`, `_keying_helpers.py`, `_audio_helpers.py`).

Frontend — простые extension-файлы под `WEB_DIRECTORY = "./js"`, регистрируются через `app.registerExtension({ name: "ts.<id>", ... })`. Большая часть UI — DOM/canvas, без npm-сборки.

Реалии расходятся с моделью в трёх местах: (1) фриз CATEGORY конфликтует с тем, что ~⅓ нод используют не-`TS/` категории; (2) часть V1-нод хранит legacy-стиль, дублируя инфраструктуру (`TS_Logger` через `print`, ANSI-коды в whisper, `__import__("torch")` в super_prompt); (3) после 8.7→8.8 рефакторинга остался мёртвый код в loader и stale ссылки на удалённую `TS_FileBrowser` в документации.

## Excluded from audit

- `.git/`, `.cache/`, `.claude/`, `__pycache__/`, `tests/.cache/`, `nodes/.cache/`, `nodes/files/.cache/`, `.tracking` — игнорируются `.gitignore`, либо являются генерируемыми артефактами.
- `nodes/video_depth_anything/**` (16 файлов, ~2700 LOC) — vendored DINOv2/DPT/motion module под Apache 2.0. Используется только нодой `TS_VideoDepthNode`. Изменять внутри запрещено по правилам ComfyUI vendoring.
- `nodes/frame_interpolation_models/**` (3 файла, ~626 LOC) — vendored FILM-Net + IFNet. Используется только `TS_Frame_Interpolation`.
- `nodes/luts/*.cube`, `nodes/prompts/*.txt`, `nodes/styles/img/*.png`, `doc/img/*.png`, `icon.png` — ассеты.
- `tests/contracts/node_contracts.json` — генерируемый снапшот (через `tools/build_node_contracts.py`).
- `README.ru.md` — авто-генерируется из `README.md` через `doc/generate_readme_ru.py`.

## Findings table

| ID | Category | File:Line | Severity | Effort | Description | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| F-01 | Security | nodes/audio/_audio_helpers.py:509-517 | Critical | S | `/ts_audio_loader/view` отдаёт любой файл по `?filepath=`. Проверка только `os.path.isfile`. Любой клиент HTTP-сервера ComfyUI читает `/etc/passwd`, ssh keys, prod-секреты с диска. | Ограничить `filepath` до подкаталогов `folder_paths.get_input_directory()`, `RECORDINGS_DIR`, `GENERATED_AUDIO_DIR`. Проверка через `Path.resolve().is_relative_to()`. |
| F-02 | Security | nodes/audio/_audio_helpers.py:498-506 | Critical | S | `/ts_audio_loader/metadata` — тот же path traversal: ffmpeg-метаданные любого файла на диске. | Тот же allow-list, что F-01. |
| F-03 | Correctness | nodes/video/ts_deflicker.py:14,20 | Critical | M | Импорты `from ..rife.warplayer import warp` и `from ..rife.IFNet_HDv3 import IFNet`. Каталог `nodes/rife/` отсутствует. Метод `rife_interpolation` крашится на каждом вызове (V1 fallback вернёт типизированный нуль через TSDependencyManager — но user-experience сломан). | Либо удалить `rife_interpolation` из дропдауна `INPUT_TYPES` и связанные ветки, либо мигрировать на `nodes/frame_interpolation_models/IFNet`, который уже в репо для `ts_frame_interpolation.py`. |
| F-04 | Security | nodes/video/ts_deflicker.py:22 | High | S | `state_dict = torch.load(model_path)` без `weights_only=True`. Для weights, скачанных с HuggingFace, любой PT-checkpoint выполняет произвольный код через pickle. PyTorch 2.6 уже сменил дефолт. | Передать `weights_only=True`. |
| F-05 | UI/Localization | nodes/image/ts_color_match.py:359-372 | High | M | Все русские `description`/`tooltip` в `INPUT_TYPES` — двойная кодировка (UTF-8 → cp1251 → UTF-8). В UI пользователь видит `Р РµС„РµСЂРµРЅСЃ` вместо `Референс`. То же в `ts_qwen_canvas.py:30,69`, `ts_qwen_safe_resize.py`, `ts_image_resize.py`, `ts_batch_prompt_loader.py`, `ts_file_path_loader.py`, `ts_deflicker.py:67`, `ts_model_converter_advanced.py`. | Перекодировать строки этих файлов: `bytes.decode('latin-1').encode('latin-1').decode('utf-8')` (или `cp1251`-эквивалент в зависимости от шага), либо переписать tooltips заново. |
| F-06 | Tensor Hygiene | nodes/video/ts_deflicker.py:21,30,31 | High | S | Хардкод `.cuda()` (`IFNet().cuda()`, `img0/255.).…cuda()`). Нарушает CLAUDE.md §8 «не предполагай CUDA». На CPU-only машинах нода крашится с `AssertionError: Torch not compiled with CUDA enabled`. | Использовать `comfy.model_management.get_torch_device()` и `.to(device)`. |
| F-07 | Module-Level Side Effects | nodes/audio/_audio_helpers.py:69-71 | High | S | `CACHE_DIR.mkdir(...)`, `RECORDINGS_DIR.mkdir(...)`, `GENERATED_AUDIO_DIR.mkdir(...)` при импорте. Нарушает CLAUDE.md §13. Тесты, контракт-сборщик, любой статический инструмент мутируют ФС. | Перенести `mkdir` в момент первой записи в каждый каталог (lazy в `_write_cached_preview`, `transcribe_endpoint`, etc). |
| F-08 | Dependency Drift | pyproject.toml:5-26 vs requirements.txt:20 | High | S | `silero-stress` указан в `requirements.txt`, но отсутствует в `[project] dependencies` `pyproject.toml`. ComfyRegistry-инсталляция получит `requirements.txt` или `pyproject` в зависимости от метода — расхождение приведёт к ImportError на `TS_SileroStress` после реgistry-pull. | Добавить `silero-stress>=0.0.0` в `pyproject.toml` (с тем же ограничением, что в requirements). |
| F-09 | Logging | 16 файлов | Medium | M | Использование `print()` вместо `logging.getLogger`. Нарушает CLAUDE.md §13. Затрагивает: `nodes/_shared.py:13`, `nodes/audio/ts_music_stems.py` (7 print), `nodes/image/ts_film_grain.py:110,118,123-132`, `nodes/image/ts_color_match.py:33`, `nodes/image/ts_crop_to_mask.py:38,53,55,58,70`, `nodes/image/ts_get_image_megapixels.py:29`, `nodes/image/ts_get_image_size_side.py:37`, `nodes/image/ts_image_batch_cut.py:31`, `nodes/image/ts_image_batch_to_list.py:30`, `nodes/image/ts_image_list_to_batch.py:30`, `nodes/image/ts_image_prompt_injector.py:33`, `nodes/image/ts_image_tile_merger.py:28`, `nodes/image/ts_image_tile_splitter.py:32`, `nodes/image/ts_resolution_selector.py:52`, `nodes/image/ts_restore_from_crop.py:82`, `nodes/video/ts_deflicker.py:70,77,80,92,223`, `nodes/video/ts_video_depth.py` (12 print, включая module-top banners), `nodes/video/ts_video_upscale_with_model.py` (8 print), `nodes/video/ts_rtx_upscaler.py:74,96,171`, `nodes/video/ts_ltx_first_last_frame.py:45,127`, `nodes/video/ts_free_video_memory.py:30,42,47`, `nodes/files/ts_edl_chapters.py:23,62`, `nodes/files/ts_downloader.py:128,171,179,183,274,278,423,439`. | Завести в каждой ноде `logger = logging.getLogger("comfyui_timesaver.<name>")` + `LOG_PREFIX`. Заменить `print(f"[X] ...")` → `logger.info("%s %s", LOG_PREFIX, ...)`. Из 16 файлов критичны 5 крупных (downloader, video_depth, video_upscale, music_stems, deflicker) — остальные мелочь. |
| F-10 | Logging | nodes/_shared.py:11-18 | Medium | S | `TS_Logger.log(node_name, message, color="cyan")` принимает `color`, но игнорирует — внутри `print(f"[TS {node_name}] {message}")`. Параметр `color` ничего не делает, вводит в заблуждение использующих (`ts_smart_switch.py:89,91,114,122`). | Удалить `color` из сигнатуры (или удалить весь `TS_Logger` и заменить на `logging`). |
| F-11 | Logging | nodes/audio/ts_whisper.py:21-25,87-93 | Medium | S | Хардкод ANSI escape codes (`_COLOR_CYAN = "\x1b[36m"` etc) в логах. CLAUDE.md §15 «Plain text, без ANSI, без emoji». Лог-файлы и UI-консоли ComfyUI без TTY получают мусор. | Удалить ANSI-константы; логировать `"%s shape=%s dtype=%s device=%s"`. |
| F-12 | God Files | nodes/llm/ts_super_prompt.py | Medium | L | 1705 LOC: voice recognition + Qwen runtime + аудио-препроцессинг + HTTP routes + UI events + node class. Файл нарушает «one node = one file» в духе (одна нода, но 6 подсистем). Любой fix требует загрузить весь контекст. | Локально приватные модули (нарушает right тоже): `nodes/llm/_super_prompt_voice.py`, `nodes/llm/_super_prompt_qwen.py`. Это запрещено CLAUDE.md §7 для публичных нод, но допустимо как `_`-префиксы (приватные shared, см. `_audio_helpers.py`). Решение требует обсуждения. Альтернатива: оставить как есть и принять — нода гигантская но стабильная. |
| F-13 | God Files | nodes/llm/ts_qwen3_vl.py | Medium | L | 1271 LOC: model loader + memory management + dtype/precision/attention resolver + multi-modal chat templating. То же замечание что F-12. | То же. |
| F-14 | God Files | nodes/audio/ts_whisper.py | Medium | L | 1079 LOC: Whisper model + SRT/TTML генерация + resampler cache + audio prep. То же. | То же. |
| F-15 | God Files | nodes/files/ts_downloader.py | Medium | L | 832 LOC: HTTP retry session + connectivity check + 4 mirror handlers + HF SHA256 + unzip. + 8 print + bare User-Agent. | Допустимо разбить внутрь файла на классы; `_create_session_with_retries`, `_replace_hf_domain`, и т.п. уже есть как методы. Главный fix — заменить print на logger. |
| F-16 | Dead Code | __init__.py:24,27-28,58-72 | Medium | S | `_LEGACY_NODE_FILENAMES = {"ts_resolution_selector.py"}` и петля по `_PACKAGE_DIR.glob("*.py")` обнаруживают legacy-файлы в корне. После 8.7→8.8 рефакторинга в корне нет файлов с префиксом `ts_` (только `ts_dependency_manager.py`, явно исключённый). Код недостижим, тесты `test_pack_imports.py:62-115` его всё ещё прогоняют. | Удалить `_LEGACY_NODE_FILENAMES`, `_is_legacy_node_file`, петлю в `_discover_module_entries`. Тесты `test_discover_module_entries_handles_subpackages` уже покрывают актуальный путь. |
| F-17 | Doc Drift | README.md:323; doc/migration.md:34; CLAUDE.md:421 | Medium | S | Отсылки к удалённому `TS_FileBrowser` / extension `ts.filebrowser` остались в документации. README EN говорит «file browser, downloader, converters»; migration.md перечисляет 9 нод в `files/`; CLAUDE.md держит `ts.filebrowser` в списке стабильных IDs. Но самой ноды и JS-файла нет (см. CLAUDE.md:73 «TS_FileBrowser удалён»). | Удалить «file browser» из обоих README и migration.md; снять `ts.filebrowser` из стабильного списка в CLAUDE.md:421 и js/AGENTS.md (если там тоже). |
| F-18 | Doc Drift | AGENTS.md:30; doc/AGENTS.md:1,23-26 | Medium | S | Корневой `AGENTS.md` ссылается на `docs/AGENTS.md`, doc/AGENTS.md имеет заголовок `# docs/AGENTS.md` и упоминает `docs/ai-lessons.md`, `docs/troubleshooting.md`, `docs/developer-notes.md`. Реальный каталог — `doc/` (singular, без `s`). | `s/docs\//doc\//g` в обоих файлах. Также удалить ссылки на `ai-lessons.md`, `troubleshooting.md`, `developer-notes.md`, которых нет в репо. |
| F-19 | Category Inconsistency | 19 нод | Medium | M | CLAUDE.md §6 фиксирует категорию `TS/<подкатегория>`. Реально используется ≥10 разных схем: `TS/Image Tools` (12), `TS/audio` (3), `TS/image` (3 V3), `TS/Color`, `TS/Conditioning`, `TS/Math`, `TS/LLM`, `TS/Prompt`, `TS/Resolution`, `TS/Upscaling`, `TS/Video`, `TS/Model Tools`, `TS/Model Conversion`, `TS/Interface Tools`, `TS Tools/Sliders`, `TS Tools/Logic`, `TS Qwen`, плюс не-`TS/` варианты `image`, `image/resize`, `image/processing`, `Image/Color`, `Image Adjustments/Grain`, `video`, `Video PostProcessing`, `Tools/Video`, `Tools/TS_Image`, `Tools/TS_IO`, `Tools/TS_Video`, `Timesaver/Image Tools`, `conditioning/video_models`, `conversion`, `Model Conversion`, `file_utils`, `utils/Tile Size`, `utils/text`, `utils/model_analysis`. CLAUDE.md §4 фризит CATEGORY как часть контракта — миграция требует осознанного решения и переноса всех старых workflow. | Выпустить документированный план миграции категорий: цель — единая иерархия `TS/<категория>/<подкатегория>`. Делать одной волной, обновить snapshot контрактов. До тех пор — задокументировать как known issue в README. |
| F-20 | Dead Comments | nodes/utils/ts_smart_switch.py:129-130 | Low | S | `# Node 4: TS Math Int` — артефакт мульти-классового файла до 8.8 рефакторинга. Сейчас в файле только `TS_Smart_Switch`. | Удалить блок. |
| F-21 | Dead Imports | nodes/image/ts_get_image_megapixels.py:7 | Low | S | `import time` — не используется. | Удалить. |
| F-22 | Inefficient Cache Key | nodes/image/ts_get_image_megapixels.py:68-74 | Low | S | `IS_CHANGED` вычисляет `image.mean()` при каждом вызове — синхронный CUDA→CPU + O(B·H·W·C). Ключу достаточно `shape` + `dtype`, мегапиксели уже инвариантны. | `return f"{tuple(image.shape)}_{image.dtype}"`. |
| F-23 | Silent Errors | 9 файлов | Medium | M | `except Exception: pass` без логирования и без причин: `nodes/audio/_audio_helpers.py:143`, `nodes/llm/ts_super_prompt.py:1487,1575`, `nodes/llm/ts_qwen3_vl.py:373,650,739,817,1133`, `nodes/audio/ts_whisper.py:995`, `nodes/video/ts_animation_preview.py:94,99,147,152,180,185`, `nodes/files/ts_model_converter_advanced.py:217`, `nodes/files/ts_downloader.py:784`. CLAUDE.md §13 «никаких голых except». | Поставить `logger.debug` или `logger.warning` на каждый swallow с указанием контекста; либо специфичный `except` (OSError, RuntimeError) вместо широкого. |
| F-24 | Bare Except | nodes/video_depth_anything/utils/dc_utils.py:12 | Low | S | `except:` (без типа) в vendored-коде. Vendored-код обычно не трогаем, но это статический сигнал. | Допустимо оставить как есть (vendored). Зафиксировать в `doc/migration.md` как vendor-deviation. |
| F-25 | Encoding | 8 файлов | High | M | Mojibake (двойная UTF-8 над cp1251) встречается в комментариях/строках: `nodes/files/ts_model_converter_advanced.py`, `nodes/image/ts_qwen_canvas.py:30,69`, `nodes/image/ts_qwen_safe_resize.py`, `nodes/image/ts_image_resize.py`, `nodes/text/ts_batch_prompt_loader.py`, `nodes/files/ts_file_path_loader.py`, `nodes/video/ts_deflicker.py:67,84`, `nodes/image/ts_color_match.py:359-372`. См. F-05. | Покрыть в одном PR: восстановить корректный UTF-8 во всех 8 файлах. Идентификация через `rg "Р[А-Я]Р|РЎ[А-Я]"`. |
| F-26 | API Misuse | nodes/llm/ts_super_prompt.py:1527,1532,1535,1543 | Low | S | `__import__("torch")` для отложенного импорта внутри функции. Используется как обходной путь для тестовых стабов, но вводит в заблуждение и ломает type-checkers. | Сделать модульный `_lazy_torch()` через `TSDependencyManager.import_optional("torch")` (один paren) или просто `import torch` локально внутри функции. |
| F-27 | Stale Pycache | nodes/__pycache__/ts_super_prompt_node.cpython-312.pyc | Low | S | `.pyc` для удалённого файла `ts_super_prompt_node.py` (ныне `nodes/llm/ts_super_prompt.py`). | Один раз вычистить (`-Force` или `git clean -fdX`). Гарантировано не попадает в репо (gitignore covers __pycache__). |
| F-28 | Stale Cache Dir | nodes/.cache/tsfb_thumbnails | Low | S | Каталог thumbnails от удалённой `TS_FileBrowser`. Не tracked в git, но висит на диске пользователя после удаления ноды. | Можно ничего не делать (gitignore покрывает); можно дописать секцию «cleanup» в `doc/migration.md`. |
| F-29 | Wildcard Type | nodes/utils/ts_smart_switch.py:24,25,29 | Low | S | `("*",)` — неофициальный wildcard для V1 input/output. Работает, но это hack. | Допустимо оставить; в случае миграции на V3 заменить на `IO.AnyType`/`IO.Combo` с явными типами. |
| F-30 | Inconsistent V1 Logging | nodes/llm/ts_qwen3_vl.py:225,242,247,361 | Medium | S | Логгер использует имя `"TS_Qwen3_VL_V3"` (`logging.getLogger("TS_Qwen3_VL_V3")`) — не совпадает с конвенцией `comfyui_timesaver.<name>` (CLAUDE.md §15). | Перейти на `logging.getLogger("comfyui_timesaver.ts_qwen3_vl")`. |
| F-31 | Code Banner Spam | nodes/video/ts_video_depth.py:310-316 | Medium | S | На module-top уровне 6 `print(...)` с баннером «Custom Node: TS Video Depth - Loaded Successfully». Печатается при каждом импорте — даже когда `__init__.py` уже выводит свою таблицу. | Заменить на `logger.info` или удалить (loader уже отчитывается). |
| F-32 | Debug Spam | nodes/image/ts_film_grain.py:123-132 | Medium | S | На каждый вызов `apply_grain` печатается `--- TS_FilmGrain Debug Info ---` блок (10 строк, включая VRAM). Это debug-output, оставленный в production. | Снести; либо убрать за `if logger.isEnabledFor(logging.DEBUG):`. |
| F-33 | CI Coverage Gap | .github/workflows/ci.yml:28-39 | Low | S | CI ставит только `pytest`, не запускает ruff/mypy, хотя CLAUDE.md §10 их рекомендует. Frontend проверок (lint/build/E2E) нет вообще. | Добавить `ruff check .` (без `--fix`) шагом, не блокирующим merge. Mypy/Frontend lint — на усмотрение. |
| F-34 | Missing Timeout | nodes/video/ts_deflicker.py:71 | Medium | S | `requests.get(url, stream=True)` без `timeout=`. Скачивание зависнет на медленном соединении. | `requests.get(url, stream=True, timeout=(15, 300))`. |
| F-35 | Mojibake in Comment | nodes/video/ts_deflicker.py:67,84 | Low | S | Docstrings `"""РЎРєР°С‡РёРІР°РµС‚..."""` — неработающий комментарий по сути. См. F-25. | Перепиши на корректный русский или английский. |
| F-36 | HTTP Endpoint Without Validation | nodes/llm/ts_super_prompt.py:1582-1605 | Medium | M | `/ts_super_prompt/enhance` принимает `text`, `system_preset`, `image` без валидации длины/структуры. Тяжёлый Qwen-вызов запускается на каждый запрос; нет rate-limit, нет idempotency. Локальный пользователь может закидать DoS. | Добавить ограничение длины `text` (≤8KB), валидацию `system_preset` через `_preset_options()`, single-flight через `_MODEL_LOCK` уже есть — можно опубликовать ещё `429 Busy` если занят. |
| F-37 | Unicode in Source | nodes/utils/ts_smart_switch.py:114 | Low | S | `TS_Logger.error("SmartSwitch", "Warning: Both inputs are None.")` через `error` API при том что это warning. Семантика искажена. | `TS_Logger.log(..., "Warning: ...")` или `logger.warning`. |
| F-38 | Loader Print | __init__.py:320-351 | Low | S | Все `[TS Startup]` шаги через `print(...)` (не logger). Привычно для ComfyUI custom_nodes loader, но нарушает §13 буквально. | Допустимо оставить, ComfyUI core тоже печатает. Документировать как exception. |
| F-39 | Missing display_mode etc | tests/contracts/node_contracts.json | Low | S | Snapshot не отслеживает defaults/min/max widget'ов — он зависит от стабильности класс-имени и пути. Любая смена `default` / `min` пройдёт незамеченной. | Расширить collector (`tools/build_node_contracts.py`) собирать `default`/`min`/`max` для V1 `INPUT_TYPES` и V3 `IO.<Type>.Input`. Это High-value test-coverage. |
| F-40 | Test Skips | tests/test_bgrm_node.py:140,149 | Low | S | 2 GPU-тест skip'ятся при `not torch.cuda.is_available()`. На CPU CI они никогда не запускаются — тестовое покрытие GPU-путей нулевое. | Принимаем (CI на ubuntu-latest без CUDA). Зафиксировать в `tests/AGENTS.md`. |
| F-41 | Russian-only docstrings in tooltip | nodes/llm/ts_super_prompt.py:1626-1652 | Low | S | Tooltip'ы только на русском («Поле промпта…»). Ожидаемо для русскоязычного пользователя, но англоязычные пользователи видят мусор. README двуязычный, tooltips — нет. | Принимаем как осознанное решение проекта. |
| F-42 | Test Coverage Gap | nodes/files/ts_downloader.py | Medium | M | 832 LOC — нет unit-теста. Парсер `file_list` (line 273) делает ad-hoc split по пробелам — лёгкий случай для regression. | Добавить минимум: тест парсинга `file_list`, тест `_replace_hf_domain`, тест `_check_connectivity_to_targets` с monkeypatch. |
| F-43 | Test Coverage Gap | nodes/audio/ts_whisper.py | Medium | M | 1079 LOC — есть `tests/test_voice_recognition_audio.py` (для super_prompt's voice subsystem), но самой `TSWhisper` контракт не тестируется (только snapshot фиксирует node_id/category/class). | Добавить behavior-тест с monkeypatch `transformers`, проверяющий ветки разных moduel/precision/attn путей. |
| F-44 | Inefficient Mojibake on disk | tests/contracts/node_contracts.json | Low | S | Snapshot v3 содержит 401 строку плотного JSON, читается каждым `pytest` запуском. На 57 нод — терпимо. На 200+ начнёт быть медленно. | Принимаем. |
| F-45 | Frontend size | js/audio/ts-audio-loader.js (931 LOC), js/llm/ts-super-prompt.js (778 LOC), js/text/ts-style-prompt.js (579 LOC), js/text/ts-prompt-builder.js (487 LOC), js/video/ts-animation-preview.js (464 LOC) | Medium | L | По логике CLAUDE.md §12 «один публичный узел = один .js». Файлы большие, но соответствуют правилу: `ts-audio-loader.js` обслуживает 2 ноды (`TS_AudioLoader` + `TS_AudioPreview`) — `migration.md:88` отмечает это как pending task. | `ts-audio-loader.js` следует разделить — вынести `TS_AudioPreview` UI в `js/audio/ts-audio-preview.js`. Документировано в migration.md. |

> Хвостовые наблюдения, которые НЕ попали в таблицу из-за лимита: 5 случаев `os.path.join(__file__)` вместо `Path(__file__).resolve().parent` (мелкая стилистика), документированные как `tooltip` строки длиной >2KB (UX-вопрос, не debt), наличие `TS_Smart_Switch` как двойного типа (`*`, обсуждается выше), `LANCZOS = Image.Resampling.LANCZOS if hasattr(...)` fallback на устаревшую константу Pillow в `ts_image_resize.py:24` (Pillow 9.1+ имеет Resampling).

## Top 5 «if you fix nothing else, fix these»

### 1. Закрыть path traversal в `/ts_audio_loader/{view,metadata}` (F-01, F-02)

`nodes/audio/_audio_helpers.py:498-517` сейчас:

```python
@_register_get("/ts_audio_loader/view")
async def ts_audio_loader_view(request: web.Request) -> web.StreamResponse:
    filepath = _normalize_selected_path(urllib.parse.unquote(request.query.get("filepath", "")))
    if not filepath or not os.path.isfile(filepath):
        return web.Response(status=404)
    try:
        return web.FileResponse(filepath)
```

Заменить на allow-list корней. Скетч изменения (≤30 строк, не переписывать функционал):

```python
_ALLOWED_VIEW_ROOTS = (
    Path(folder_paths.get_input_directory()).resolve(),
    RECORDINGS_DIR.resolve(),
    GENERATED_AUDIO_DIR.resolve(),
)

def _is_inside_allowed_root(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return any(resolved.is_relative_to(root) for root in _ALLOWED_VIEW_ROOTS)


@_register_get("/ts_audio_loader/view")
async def ts_audio_loader_view(request: web.Request) -> web.StreamResponse:
    raw = urllib.parse.unquote(request.query.get("filepath", ""))
    candidate = Path(_normalize_selected_path(raw))
    if not raw or not candidate.is_file() or not _is_inside_allowed_root(candidate):
        return web.Response(status=404)
    return web.FileResponse(str(candidate))
```

То же — для `_metadata_endpoint`. Тест: добавить `tests/test_audio_loader_routes.py`, проверяющий 404 на `?filepath=/etc/passwd` и `?filepath=../../../somewhere`.

### 2. Починить broken RIFE-import в `TS_DeflickerNode` (F-03)

`nodes/video/ts_deflicker.py:14,20` — выбрать одно из:

**A.** Удалить mode `rife_interpolation` (минимальный путь):

```diff
-                "method": (["temporal_median", "temporal_gaussian", "adaptive_histogram", "rife_interpolation"], {"default": "temporal_gaussian"}),
+                "method": (["temporal_median", "temporal_gaussian", "adaptive_histogram"], {"default": "temporal_gaussian"}),
```

И удалить класс `RIFE`, методы `rife_deflicker`, `download_model`, `get_model_path`, ветку `elif method == "rife_interpolation"` в `deflicker`. Контракт snapshot обновить.

**B.** Подключить уже существующий `nodes.frame_interpolation_models.IFNet` (использует `ts_frame_interpolation.py`), переписав загрузку модели через ComfyUI `folder_paths` + `safetensors` (как в `ts_frame_interpolation.py:60-61`). Это значительно крупнее.

Рекомендация: **A** (как минимально безопасное), сопроводить migration note.

### 3. Восстановить кодировку tooltips/comments в 8 файлах (F-05, F-25)

Mojibake разрушает UX 8 V1-нод. Скрипт восстановления (запускается локально, не коммитится):

```python
# не для CI — для разовой починки
import sys
from pathlib import Path

for path in [
    "nodes/image/ts_color_match.py",
    "nodes/image/ts_qwen_canvas.py",
    "nodes/image/ts_qwen_safe_resize.py",
    "nodes/image/ts_image_resize.py",
    "nodes/text/ts_batch_prompt_loader.py",
    "nodes/files/ts_file_path_loader.py",
    "nodes/files/ts_model_converter_advanced.py",
    "nodes/video/ts_deflicker.py",
]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    fixed = text.encode("latin-1", errors="ignore").decode("cp1251", errors="ignore")
    # smoke check: Кириллица найдена?
    if any(0x0410 <= ord(c) <= 0x044F for c in fixed):
        p.write_text(fixed, encoding="utf-8")
```

⚠️ Перед запуском убедиться, что строки реально UTF-8 поверх cp1251 (иногда cp866 или Win1252). Проверять по выборке. Покрыть результат `python -m compileall .` и `pytest`.

### 4. Восстановить парность `pyproject.toml` ↔ `requirements.txt` (F-08)

`pyproject.toml:5-26` сейчас:

```toml
dependencies = [
    "accelerate>=0.21.0",
    ...
    "silero"
]
```

Добавить:

```diff
-    "silero"
+    "silero",
+    "silero-stress"
 ]
```

Дополнительно: разовая ручная сверка двух файлов и решение, какой из них главный. Вариант — генерировать `requirements.txt` из `pyproject.toml` через `pip-compile`. Это вне минимального fix.

### 5. Удалить mass `print()` логирование в 5 крупных нодах (F-09)

Минимальный сабсет (5 наиболее громких):

- `nodes/files/ts_downloader.py` — 8 print
- `nodes/video/ts_video_depth.py` — 12 print, включая module-top banners
- `nodes/video/ts_video_upscale_with_model.py` — 8 print
- `nodes/audio/ts_music_stems.py` — 7 print
- `nodes/image/ts_film_grain.py:123-132` — debug-block

Стандартизированная замена:

```python
import logging
logger = logging.getLogger("comfyui_timesaver.ts_<name>")
LOG_PREFIX = "[TS <Name>]"

# было: print(f"[TS Music Stems] Initializing model: {model_name}")
# стало:
logger.info("%s Initializing model: %s", LOG_PREFIX, model_name)
```

Каждый файл: ≤30 строк изменений. Прогнать `pytest tests` после каждого.

## Quick wins

- [ ] Удалить мёртвый legacy-loader: `__init__.py:24,27-28,58-72` + `_LEGACY_NODE_FILENAMES` (F-16, S/Medium).
- [ ] Удалить orphan-комментарий `# Node 4: TS Math Int` в `ts_smart_switch.py:129-130` (F-20, S/Low).
- [ ] Удалить unused `import time` в `ts_get_image_megapixels.py:7` (F-21, S/Low).
- [ ] Заменить `image.mean()` в `IS_CHANGED` на `f"{shape}_{dtype}"` (F-22, S/Low).
- [ ] Удалить ANSI-коды и `_log_tensor_shape` colored из `ts_whisper.py:21-25,87-93` (F-11, S/Medium).
- [ ] Чистка module-top print-баннера в `ts_video_depth.py:310-316` (F-31, S/Medium).
- [ ] Удалить debug-блок в `ts_film_grain.py:123-132` (F-32, S/Medium).
- [ ] Добавить `timeout=` в `requests.get` в `ts_deflicker.py:71` (F-34, S/Medium).
- [ ] `s/docs\//doc\//g` в `AGENTS.md:30` и `doc/AGENTS.md:1,23-26` (F-18, S/Medium).
- [ ] Удалить ссылки на `TS_FileBrowser` / `ts.filebrowser` из `README.md:323`, `doc/migration.md:34`, `CLAUDE.md:421` (F-17, S/Medium).
- [ ] Заменить `logging.getLogger("TS_Qwen3_VL_V3")` на `logging.getLogger("comfyui_timesaver.ts_qwen3_vl")` (F-30, S/Medium).
- [ ] Добавить `silero-stress` в `pyproject.toml` (F-08, S/High).

## Things that look bad but are actually fine

- **`subprocess.run(...)` в нескольких узлах** (`_audio_helpers.py:215,274,355`, `ts_super_prompt.py:654`, `ts_animation_preview.py:334`) — все используют list-args без `shell=True`, аргументы — пути из `folder_paths` или `imageio_ffmpeg.get_ffmpeg_exe()`. Не command injection. Принимаем.
- **`importlib.util.spec_from_file_location` в `ts_bgrm_birefnet.py:422`** — динамическая загрузка скачанного с HF python-файла модели. Похоже на arbitrary code execution, но это часть HF-трекинга веса BiRefNet (модель приходит вместе со своим `BiRefNet_<config>.py`). Стандартный паттерн ComfyUI. Принимаем при условии trusted-source HF-репо.
- **Module-level `_register_model_folder` в `ts_frame_interpolation.py:60-61`** — регистрация в `folder_paths` при импорте. Side effect, но это конвенциональный паттерн ComfyUI custom-nodes для добавления `models/rife/`, `models/film/` в whitelist. Допустимо.
- **`ts_smart_switch.py` `("*",)` wildcard** — V1 API hack, но широко используемый в ComfyUI ecosystem (`Bypass`, `Reroute` аналогичны). До миграции на V3 `IO.AnyType` — допустимо.
- **Гигантские LLM/audio файлы (1705/1271/1079 LOC)** — нарушают «one node = one file in spirit», но формально соответствуют (одна публичная нода, плюс приватные helpers внутри файла). Дальнейшая декомпозиция через `_<name>.py` приватные модули — обсуждаемо, но требует обоснования и осторожной миграции. Сейчас — управляемый risk.
- **Mass V1 API в 38 нодах** — CLAUDE.md явно фризит V1 (§5 «Existing V1 nodes: frozen»). Не debt — это сознательная политика.
- **`comfyui-timesaver/.tracking`** — внутренний файл ComfyUI Manager (см. содержимое — это manifest), не наш debt.
- **Vendored `nodes/video_depth_anything/` и `nodes/frame_interpolation_models/`** — нельзя править: это Apache 2.0 licensed forked code (DINOv2/DPT/FILM-Net/IFNet). Print'ы и `bare except:` внутри — приемлемые vendor-deviations.
- **`nodes/_shared.py:13` `print()`** — формально debt, но это мелкий fallback-логгер, возможно сохранён сознательно для случаев, когда `logging` не настроен. Всё же: см. F-10 (стоит сменить).
- **`__init__.py:320-351` через `print()`** — startup-таблица в stdout, аналогично ComfyUI core. Конвенция.

## Open questions for the maintainer

1. **CATEGORY миграция (F-19)**: маппинг ~19 нод на единую `TS/<...>` иерархию ломает поиск/сортировку в saved workflows. Хотите разовый migration release с явным `display_name` фриз?
2. **CATEGORY conflict с CLAUDE.md §4**: §4 явно говорит «CATEGORY не менять без явного запроса». §6 даёт правило именования. Конфликт намеренный — сейчас CATEGORY стабилен но непоследователен. Стоит зафиксировать одно правило или оставить как есть?
3. **`TS_DeflickerNode rife_interpolation`** (F-03): удалять метод (потеря функции) или подключить новую `IFNet` инфраструктуру (M-task)? Какой путь выбираете?
4. **`/ts_audio_loader/view` allow-list (F-01)**: ограничить `input/`, `output/`, `RECORDINGS_DIR`, `GENERATED_AUDIO_DIR` — или нужен ещё какой-то путь, который вы используете?
5. **`ts_super_prompt.py` декомпозиция (F-12)**: 1705 LOC — допустимо как «одна нода», но шесть подсистем внутри. Разделить на приватные `_*.py` помощники в `nodes/llm/`?
6. **`silero-stress` в `pyproject.toml` (F-08)**: была ли причина не добавлять? Возможно, проблема pip-resolver на 3.10/3.11 в CI?
7. **Mojibake (F-05, F-25)**: вы знаете оригинальные тексты? Можем починить алгоритмически, но если кодировка нестандартная (cp866/win1252) — нужно подтверждение по выборке.
8. **Test coverage `ts_downloader.py` и `ts_whisper.py` (F-42, F-43)**: добавить unit-тесты или принимать как ручной QA?
9. **CI без ruff (F-33)**: добавлять `ruff check .` без `--fix` блокирующим шагом? У вас есть существующие conventions, которых ruff с дефолтом может не знать.
10. **`js/audio/ts-audio-loader.js` разделение (F-45)**: уже зафиксировано в migration.md как pending. Рассматриваете отдельную задачу?
