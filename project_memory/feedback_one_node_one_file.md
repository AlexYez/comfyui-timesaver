---
name: One node = one file
description: Каждая публичная ComfyUI-нода = один основной .py файл; не дробить на schema/execute/types
type: feedback
originSessionId: 3ceba1d8-67f6-454f-b13a-b662edeed0db
---
Структурное правило (см. `AGENTS.md` §4 и `nodes/AGENTS.md` §2):

> Одна публичная ComfyUI-нода = один основной `.py` файл в `nodes/`. Если ноде нужен фронтенд — один соответствующий `.js` файл в `js/`.

Запрещено дробить ноду на:
```
nodes/ts_example/schema.py
nodes/ts_example/execute.py
nodes/ts_example/validation.py
nodes/ts_example/types.py
```

Allowed: private helpers внутри того же файла, либо `utils/` ТОЛЬКО если логика реально используется 2+ нодами.

С релиза 8.8 правило строгое: каждая публичная нода = один файл. Multi-class файлы расщеплены по подпапкам (`nodes/image/`, `nodes/video/`, `nodes/audio/`, `nodes/llm/`, `nodes/text/`, `nodes/files/`, `nodes/utils/`).

Why: пользователь явно потребовал «дробим всё» и категориальное деление. Backwards compatibility сохранена через стабильные `node_id` и `NODE_CLASS_MAPPINGS` ключи.

How to apply:
- Новые ноды — один файл в соответствующей подпапке `nodes/<категория>/ts_<name>.py`.
- Shared логика для 2+ нод одной категории — в `nodes/<категория>/_*.py` (приватный модуль, loader пропускает по `_`-префиксу).
- Pack-level shared (типа `TS_Logger`) — в `nodes/_shared.py`.
- При появлении новой категории — создаём подпапку с `__init__.py` (loader работает рекурсивно через `rglob`).
