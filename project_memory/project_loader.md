---
name: Loader and dependency policy
description: __init__.py авто-обнаруживает nodes/*.py и оборачивает каждую ноду TSDependencyManager.wrap_node_runtime
type: project
originSessionId: 3ceba1d8-67f6-454f-b13a-b662edeed0db
---
Загрузчик пака (`__init__.py`):

1. Рекурсивно сканирует `nodes/**/*.py` через `rglob` (после релиза 8.8 поддерживаются подпапки).
2. Пропускает: `__init__.py`, любую часть пути с префиксом `_` (`_shared.py`, `_audio_helpers.py`, `_keying_helpers.py`, и т.п.), и файлы, чьё имя НЕ начинается с `ts_` (это защищает helper-пакеты `frame_interpolation_models/`, `video_depth_anything/`).
3. Импортирует каждый модуль, читает его `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`, сливает в глобальные.
4. Каждую зарегистрированную ноду оборачивает `TSDependencyManager.wrap_node_runtime(node_name, node_cls, logger)`.
5. На выходе печатает startup-таблицу: модули (Module/Status/Nodes/Details), внешние импорты (Import/Available/Severity/Source) и summary.

Legacy-слой для `_LEGACY_NODE_FILENAMES` в корне пакета сохранён как есть, но в чистом 8.8-состоянии не используется.

`TSDependencyManager` (`ts_dependency_manager.py`):
- `import_optional(module_name)` — кешированный мягкий импорт; при ошибке возвращает `None`.
- `extract_missing_dependency(exc)` — извлекает имя отсутствующего модуля из `ModuleNotFoundError`/текста.
- `wrap_node_runtime(...)` — оборачивает метод, на который указывает `FUNCTION` (V1) или `execute` (V3):
  - Для V1 при исключении возвращает typed fallback на основе `RETURN_TYPES` (zeros tensor для IMAGE/MASK, пустой dict для AUDIO/LATENT, дефолты для STRING/INT/FLOAT/BOOLEAN).
  - Для V3 нормализует ошибку и пробрасывает как `RuntimeError` с TS prefix.
- Маркер `_ts_runtime_guard_wrapped = True` предотвращает двойную обёртку.

Политика зависимостей (`doc/TS_DEPENDENCY_POLICY.md`):
- Не падать всем паком из-за одной отсутствующей optional-зависимости.
- Тяжёлые/опциональные либы — через `TSDependencyManager.import_optional(...)`, не на module level.
- Валидировать наличие optional-deps в runtime entry method и кидать понятный `RuntimeError` с TS-префиксом.
- Логи — plain text, без цветов и emoji.
- НЕ запускать pip из кода ноды.

Why: пользователи устанавливают пак на разных конфигурациях (Windows portable, Linux, разные GPU). Минимум падений = больше доверия и меньше тикетов.

How to apply:
- При добавлении новой optional-зависимости: import через `TSDependencyManager.import_optional`, проверка в `execute`/функции, понятное сообщение.
- При изменении `RETURN_TYPES` в V1-ноде убедиться, что fallback в `TSDependencyManager.fallback_value_for_type` корректно обработает каждый тип.
- Если нужна централизованная фоновая обёртка для нового сценария — расширять `TSDependencyManager`, а не дублировать try/except в каждой ноде.
