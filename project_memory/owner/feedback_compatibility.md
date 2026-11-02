---
name: feedback_compatibility
description: "Не менять node_id, входы/выходы, defaults, category, widget IDs без явного запроса пользователя"
metadata:
  type: feedback
---

Не менять node_id, входы/выходы, defaults, category, widget IDs без явного запроса пользователя

Главная цель проекта ([AGENTS](../../AGENTS.md), [совместимость](../../doc/engineering/workflow.md)): **не сломать существующие workflows пользователей**.

Никогда не менять без явного запроса/миграции:
- Python class names.
- V3 `node_id`.
- V1 `NODE_CLASS_MAPPINGS` keys.
- JS extension IDs.
- Input names.
- Output names, output order, output types.
- `execute()` parameter names (порядок и имена).
- Default widget values.
- `CATEGORY` values.
- Saved configuration keys.
- Widget IDs, на которые ссылается фронтенд.
- Существующая семантика поведения (что делает нода при тех же входах).

Если переименование действительно нужно:
- V3: добавить `search_aliases` в Schema, использовать `io.NodeReplace` через `ComfyAPI`.
- V1: сохранить старый ключ в `NODE_CLASS_MAPPINGS` как alias (старый ключ → класс).
- Добавить миграционную заметку в docs.
- Добавить/обновить contract test.

Why: воркфлоу в ComfyUI сериализуют node_id и widget names. Любое переименование = поломка чужих сохранённых workflow. Пак — production, у пользователей есть рабочие графы.

How to apply:
- Refactor ≠ rename. Переименование, перестановка, изменение default — всё это breaking changes, требуют явного согласия пользователя.
- Перед редактированием убедиться, что diff не задевает публичный контракт (см. [контракт workflow](../../doc/engineering/workflow.md)).
- Сохранённый порядок приоритетов исходной инструкции: Preserve workflows > Preserve compatibility > Preserve UX > Correctness > Testability > Maintainability > Performance > Architecture > New features.
