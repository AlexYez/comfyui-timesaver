---
name: Naming convention TS
description: Префикс TS_ для классов и node_id, ts_ для имён файлов; новые имена не должны конфликтовать со старыми
type: feedback
originSessionId: 3ceba1d8-67f6-454f-b13a-b662edeed0db
---
Жёсткое правило именования (см. `AGENTS.md` §8):

- Файл Python: `ts_example_node.py` (snake_case, префикс `ts_`).
- Класс Python: `TS_ExampleNode` (PascalCase с префиксом `TS_`).
- V3 `node_id`: `"TS_ExampleNode"` (совпадает с именем класса).
- Display name: `"TS Example Node"`.
- Category: `"TS/<подкатегория>"` (например, `TS/image`, `TS/audio`, `TS/LLM`, `TS/Color`, `TS Tools/Sliders`).
- JS-файл: `ts-example-node.js` (kebab-case).
- JS extension ID: `ts.exampleNode` (camelCase или kebab часть после `ts.`).

Why: префикс `TS` защищает от конфликтов с другими паками; устоявшаяся схема позволяет фронтенду надёжно находить узлы по типу. Любое отклонение ломает сериализованные workflow пользователей.

How to apply:
- При создании новой ноды строго следовать схеме.
- Никогда не переименовывать существующие классы / `node_id` / display names — это breaking change. Если миграция нужна, использовать `search_aliases` (V3) или сохранять алиас в `NODE_CLASS_MAPPINGS` (V1).
- Уточнения: `TS Cube to Equirectangular` и `TS Equirectangular to Cube` — display names с пробелами; `TSWhisper`, `TSCropToMask`, `TSRestoreFromCrop`, `TSAutoTileSize` — исторически без подчёркивания. Эти имена не менять.
