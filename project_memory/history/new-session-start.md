---
name: new-session-start
description: "История: Готовый стартовый промпт для новой сессии из dev-папки C: — продолжение работы над нодой TS Smart Inpaint"
metadata:
  type: reference
  status: historical
---

История: Готовый стартовый промпт для новой сессии из dev-папки C: — продолжение работы над нодой TS Smart Inpaint

Этот стартовый промпт относился к конкретной работе над Smart Inpaint 2026-06-22. Не вставлять его как актуальную задачу: начать с [AGENTS.md](../../AGENTS.md) и текущего [индекса памяти](../MEMORY.md).

**Историческая запись, не описание текущей реализации и не действующее разрешение.** Даты и результаты ниже относятся к указанным сессиям. Пути `tests/`, `tools/`, `dist/` и `archive/` с артефактами — локальная лаборатория; они не гарантированы в чистом клоне.

# Стартовый промпт для новой сессии (вставить первым сообщением)

> Работаем в dev-папке проекта comfyui-timesaver (это `C:\Users\Sanchez\Documents\timesaver\projects\cloude-projects\comfyui-timesaver` — правильная папка: тут `.git`, `tests/`, полная `project_memory/`, `CLAUDE.md`; разработка/commit/push только отсюда).
>
> Сначала прочитай в `project_memory/`: `ts-smart-inpaint-handoff.md` (полный контекст задачи), `ts-smart-inpaint-node.md` (архитектура ноды), `feedback_runtime_only_locations.md` (правило dev=C: / runtime=D:, не пушить из D:).
>
> Статус: оптимизация **A0+A1** (crop-only VAE-путь) для ноды TS Smart Inpaint уже сделана и запушена — commit `deccb54` на GitHub, в C: и в runtime D:. НЕ переделывай это.
>
> Задача: продолжаем работу над нодой **TS Smart Inpaint** (`nodes/image/ts_smart_inpaint.py`). Приоритет — фикс «мыла» на свежесгенерированной области (feather/blur, «опция C» из handoff): развести жёсткую маску генерации (как в 5D-ветке) и узкий feather-только-в-композите, смещённый наружу, реализовать как ОПЦИЮ, не меняя поведение по умолчанию. План и детали — в `ts-smart-inpaint-handoff.md`.
>
> Перед правками прочитай текущий `nodes/image/ts_smart_inpaint.py` целиком (он уже с A0/A1). Ядро Angelo X-Refine не ломать. Workflow: правка → verify → commit → push из C:, затем runtime D: обновляется `git pull` / ComfyUI Manager. Отвечай на русском.
