---
name: Verification commands
description: Минимальные обязательные backend-проверки и опциональные frontend/E2E команды для пака
type: reference
originSessionId: 3ceba1d8-67f6-454f-b13a-b662edeed0db
---
**Используй ComfyUI Python** (`D:/AiApps/ComfyUI/comfyui/python/python.exe`) — обычный test Python без numpy/torch/PIL, тесты с importorskip скипнутся (см. `reference_comfyui_python.md`).

Минимум для backend перед заявлением "готово" (root `AGENTS.md` §3, `nodes/AGENTS.md` §11):

```bash
D:/AiApps/ComfyUI/comfyui/python/python.exe -m compileall .
D:/AiApps/ComfyUI/comfyui/python/python.exe -m pytest tests
```

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
