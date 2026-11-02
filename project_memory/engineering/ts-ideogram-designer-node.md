---
name: ts-ideogram-designer-node
description: "Ideogram Designer: архитектура, caption JSON, редактор и причины прежних ошибок."
metadata:
  type: project
---

Ideogram Designer: архитектура, caption JSON, редактор и причины прежних ошибок.

Текущий backend — [ts_ideogram_designer.py](../../nodes/ideogram/ts_ideogram_designer.py), UI — [js/ideogram](../../js/ideogram/), нормализация caption — [nodes/_caption_json.py](../../nodes/_caption_json.py). Правила build/test/sync/restart берутся из [operations](../../doc/operations/README.md), не из старого Build/dev приложения.

## Темы

- [Ideogram Designer: выходы, пресеты, редактор, preview и известные ограничения.](ideogram_designer_architecture.md)
- [Ideogram Designer: история редактора промпта, координаты JSON, палитры и raw desc.](ideogram_designer_caption.md)
