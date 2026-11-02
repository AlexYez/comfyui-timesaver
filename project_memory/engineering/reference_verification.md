---
name: reference_verification
description: "Каноника проверок находится в operations/testing; здесь сохранены прежние сценарии и их причины."
metadata:
  type: reference
originSessionId: 3ceba1d8-67f6-454f-b13a-b662edeed0db
---

Каноника проверок находится в operations/testing; здесь сохранены прежние сценарии и их причины.

**Текущая опора (2026-11-02):** Текущие проверки и разрешения описаны в [testing](../../doc/operations/testing.md). Длинный сценарий студии ниже относится к снятой TS Image Studio; он не является обязательным порядком работы с нынешним паком.

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

**Используй ComfyUI Python** (`D:/AiApps/ComfyUI/comfyui/python/python.exe`) — обычный test Python без numpy/torch/PIL, тесты с importorskip скипнутся (см. `reference_comfyui_python.md`).

Минимум для backend перед заявлением "готово":

```bash
D:/AiApps/ComfyUI/comfyui/python/python.exe -m compileall .
D:/AiApps/ComfyUI/comfyui/python/python.exe -m pytest tests/test_<тема>.py
```

⚠️ **Объём по умолчанию — затронутые ноды и общие модули, не весь набор**
([feedback_tests_on_demand](../owner/feedback_tests_on_demand.md)): тесты самой ноды, тесты общих модулей, которые она
задействует, и тесты ДРУГИХ нод, использующих те же модули. Полный прогон занимает
машину, на которой человек работает.

⚠️ **Весь набор — ТОЛЬКО по прямой просьбе.** Оговорка «или перед публикацией
версии» снята 14.09.2026 по указанию владельца: бамп и пуш не повод. Лишний
прогон, уже запущенный, — остановить.

⚠️ **Ровно один прогон за раз.** Два параллельных pytest'а сталкиваются на общем
`tests/.cache/tmp` и убивают друг друга; результату такого прогона верить нельзя.

⚠️ **Простой `pytest tests` против ЗАНЯТОГО ComfyUI может повиснуть** — живой файл
ждёт ответа и не дожидается (наблюдалось 2026-08-08, набор встал на старте). Полный
прогон вместе с живыми файлами делать через `tools/preflight.py --full` и после
перезапуска сервера.

Проверка на Apple Silicon — `.github/workflows/macos_smoke.yml` по кнопке
(`gh workflow run macos_smoke.yml --ref <ветка>`), см. [reference_macos_compat](reference_macos_compat.md).

После изменения V3 schemas — регенерировать contract snapshot:

```bash
D:/AiApps/ComfyUI/comfyui/python/python.exe tools/build_node_contracts.py
```

Опционально, если установлены инструменты:

```bash
D:/AiApps/ComfyUI/comfyui/python/python.exe -m ruff check .
D:/AiApps/ComfyUI/comfyui/python/python.exe -m ruff format --check .
D:/AiApps/ComfyUI/comfyui/python/python.exe -m mypy .
```

Frontend (если есть npm tooling — в текущем виде package.json в корне отсутствует):

```bash
npm run lint
npm run test
npm run build
```

Frontend E2E (Playwright, если настроен):

```bash
npm run test:e2e
npx playwright test tests/e2e
```

Если ComfyUI запущен локально, фактическая проверка фронтенда:
- Открыть `http://127.0.0.1:8188` (или `COMFY_URL`).
- Проверить, что browser console чистая.
- Найти ноду через поиск, изменить виджеты.
- Открыть workflow JSON, если есть фикстуры.

Verification summary в ответе пользователю обязателен и пишется на русском в формате:

```text
Проверено:
- python -m compileall .
- python -m pytest tests

Не проверено:
- npm run test:e2e — ComfyUI не запущен на 127.0.0.1:8188.
```

Не писать "готово" только потому, что код выглядит правильным.

Where: тесты — в `tests/`. Pytest cache: `pytest-cache-files-*`.

## Полная проверка подсистемы студии (эталон прогона 2026-08-04)

Порядок обязателен: E2E лаунчера поднимает свой ComfyUI на 8188 и гасит живой.

```bash
# 1. Доехало ли до ComfyUI
python tools/verify_live_sync.py --sync

# 2. Весь пак (браузерный смоук требует запущенного ComfyUI)
python -m pytest tests -q

# 3. Живые генерации: каждое семейство в каждом режиме через интерфейс
#    (скрипт-прогон держать в scratchpad, он одноразовый)

# 4. Цикл автора в админке лаунчера
cd ../comfyui-launcher && node tools/workflow-admin/server.js
#    ключи -> сборка (host и github) -> публикация -> установка в студии

# 5. Гейты лаунчера
cd launcher && npm test && npm run lint && npm run typecheck
node scripts/admin-gui.js && node scripts/gui-smoke.js && node scripts/gui-scenarios.js

# 6. И только теперь E2E лаунчера
node tests/e2e/boot-embed.e2e.js
```

Что считать за «проверено» в цикле подписки: выпуск ключей на новый месяц;
сборка набора в обеих схемах; попытка открыть зашифрованный набор ключами
соседних уровней и другого месяца (обязаны отказать); активация по коду на
всех трёх уровнях; установка набора студией с расшифровкой; отказ установки
при `simulate: none`.

Живые тайминги семейств — [studio-model-behaviour](../history/studio-model-behaviour.md). Апскейл 4x на 20B-модели
идёт десятки минут: короткий бюджет ожидания в прогоне даёт «не дождались», а
не поломку.
