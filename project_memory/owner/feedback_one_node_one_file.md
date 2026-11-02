---
name: feedback_one_node_one_file
description: "Каждая публичная ComfyUI-нода = один основной .py файл; не дробить на schema/execute/types"
metadata:
  type: feedback
---

Каждая публичная ComfyUI-нода = один основной .py файл; не дробить на schema/execute/types

Структурное правило: актуальная навигация — [AGENTS](../../AGENTS.md) и [правила V3-нод](../../doc/engineering/node-api.md). Прежнего nodes/AGENTS.md в репозитории нет.

> Одна публичная ComfyUI-нода = один основной `.py` файл в `nodes/`. Если ноде нужен фронтенд — один соответствующий `.js` файл в `js/`.

Запрещено дробить ноду на:
```
nodes/ts_example/schema.py
nodes/ts_example/execute.py
nodes/ts_example/validation.py
nodes/ts_example/types.py
```

Allowed: private helpers внутри того же файла, либо `utils/` ТОЛЬКО если логика реально используется 2+ нодами.

Историческое описание миграции 8.8: каждая публичная нода = один файл. Multi-class файлы расщеплены по подпапкам (`nodes/image/`, `nodes/video/`, `nodes/audio/`, `nodes/llm/`, `nodes/text/`, `nodes/files/`, `nodes/utils/`).

Why: пользователь явно потребовал «дробим всё» и категориальное деление. Backwards compatibility сохранена через стабильные `node_id` и `NODE_CLASS_MAPPINGS` ключи.

How to apply:
- Новые ноды — один файл в соответствующей подпапке `nodes/<категория>/ts_<name>.py`.
- Shared логика для 2+ нод одной категории — в `nodes/<категория>/_*.py` (приватный модуль, loader пропускает по `_`-префиксу).
- Pack-level shared (типа `TS_Logger`) — в `nodes/_shared.py`.
- При появлении новой категории — создаём подпапку с `__init__.py` (loader работает рекурсивно через `rglob`).

## Проверенное исключение текущего кода

На 2026-11-02 две публичные H3 Regenerate-ноды находятся в [одном исходнике](../../nodes/video/latent_upscale/ts_h3_regenerate.py). Это зафиксированное существующее исключение, не разрешение расширять его или молча менять исходное требование владельца. Код этой документационной задачей не менялся.
