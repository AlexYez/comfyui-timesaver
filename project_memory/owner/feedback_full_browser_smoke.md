---
name: feedback_full_browser_smoke
description: "После рестарта ComfyUI обязателен Playwright smoke (LiteGraph.createNode + read console errors). Скриншоты в smoke не делаем — они только для README через tools/screenshot_nodes.py"
metadata:
  type: feedback
---

После рестарта ComfyUI обязателен Playwright smoke (LiteGraph.createNode + read console errors). Скриншоты в smoke не делаем — они только для README через tools/screenshot_nodes.py

## Границы требования

Проверять фронтенд в браузере, когда меняется его поведение; проверка регистрации по HTTP не заменяет видимый результат. Это не разрешение запускать/перезапускать сервер: действует [прямое разрешение владельца](comfyui-lifecycle.md). Указанный ниже `tools/screenshot_nodes.py` — локальный исторический инструмент, не обязательный файл чистого клона. Полный набор тестов запускается только по [отдельной команде](feedback_tests_on_demand.md).

## Исходная запись

После каждого мигрированного батча browser smoke-test **обязателен** — не только curl /object_info.

**Why:** пользователь явно сказал «и браузерные тесты не забывай делать, всегда делай все возможные тесты». curl показывает только wire-format регистрацию, но не: фактический рендер на canvas, JS-ошибки, DOM widgets, JS-extension hooks, конфликты с другими расширениями.

`tools/screenshot_nodes.py` — это helper для генерации README-картинок при добавлении новых нод, **не smoke-test**. Запускать его в тестах не нужно (пользователь явно поправил 2026-05-06).

**How to apply (после каждого рестарта ComfyUI):**
1. Curl `/object_info` — быстрый preflight: убедиться что наши node_id зарегистрированы.
2. Playwright (через ComfyUI Python, `locale="en-US"`) inline-скриптом:
   - открыть `http://127.0.0.1:8188`
   - дождаться `app` и `LiteGraph` в window
   - подписаться на `console` события и собрать errors
   - для каждой ноды из списка: `app.graph.clear()` → `LiteGraph.createNode(id)` → `app.graph.add(node)` → проверить что `node` not null и без exception
   - вернуть { ok: [...], failed: [...], console_errors: [...] }
3. Только после этого считать этап завершённым.

Curl остаётся быстрым preflight; Playwright даёт реальный браузерный сигнал; screenshots не нужны для тестов.
