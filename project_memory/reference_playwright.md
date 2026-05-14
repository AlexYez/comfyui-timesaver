---
name: Playwright with Chromium available
description: ComfyUI Python имеет playwright + Chromium для headless GUI скриншотов и frontend-тестов; обязательно forced en-US locale; tools/screenshot_nodes.py — готовый helper
type: reference
originSessionId: 0f7f3f61-1a62-42f4-bd4d-68eb550d8c88
---
**Playwright установлен в ComfyUI Python:**

- `D:/AiApps/ComfyUI/comfyui/python/python.exe` имеет `playwright==1.59.0` + Chromium (загружен через `python -m playwright install chromium`).
- Chromium лежит в `C:/Users/Sanchez/AppData/Local/ms-playwright/chromium-*/`.

**Когда использовать Playwright:**
- Headless GUI screenshots для документации (canvas + DOM widgets composited correctly).
- Автоматическое frontend testing (создание нод, проверка console, e2e workflows).
- CDP-level capture: `page.screenshot(clip={...})` — финальный composited frame, не только canvas pixel buffer (≠ `canvas.toDataURL()`).

**Critical: всегда форсить английский locale** — иначе Chromium берёт OS-locale (русский на этой машине) и ComfyUI переведёт labels (например `IMAGE` → `ИЗОБРАЖЕНИЕ`):

```python
context = browser.new_context(
    viewport={"width": 1920, "height": 1080},
    device_scale_factor=1,
    locale="en-US",
    extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
)
```

**`tools/screenshot_nodes.py` — это helper для генерации README/doc-картинок при добавлении новой ноды, НЕ smoke-test.** Запускать только когда добавлена новая нода и нужен её скриншот в `doc/screenshots/`. Для тестов писать отдельные Playwright-скрипты с createNode + read console (без screenshots). Пользователь поправил это явно 2026-05-06.

```bash
# Все 57
D:/AiApps/ComfyUI/comfyui/python/python.exe tools/screenshot_nodes.py

# Конкретные (по node_id или file stem)
D:/AiApps/ComfyUI/comfyui/python/python.exe tools/screenshot_nodes.py TS_Keyer ts_audio_loader

# Видимое окно для отладки
D:/AiApps/ComfyUI/comfyui/python/python.exe tools/screenshot_nodes.py --no-headless
```

Скрипт использует `tools.build_node_contracts.collect_contracts()` для AST-discovery нод — не нужен ручной mapping.

**Playwright vs Chrome MCP (расширение Claude в браузере пользователя):**

| | Playwright | Chrome MCP |
|---|---|---|
| Headless screenshots с composited DOM+canvas | ✅ | ✅ (но `save_to_disk` путь не возвращается надёжно) |
| Программное управление без UI | ✅ headless | ❌ требует видимое окно пользователя |
| Reproducible тесты | ✅ изолированный context | ❌ может зависеть от user state, cookies, открытых вкладок |
| Принудительный locale | ✅ `locale='en-US'` | ❌ берёт user browser settings |
| Запуск из Python скрипта | ✅ напрямую через sync_playwright | ❌ только через MCP tool calls |
| User session / тестирование UX в реальном браузере пользователя | ❌ | ✅ |

**Default: для frontend GUI testing использовать Playwright, не Chrome MCP.**
Chrome MCP оставить только для случаев, где явно нужно проверить поведение в реальном пользовательском браузере (например, его установка с конкретными расширениями/настройками).

**How to apply:**
- Для smoke test workflow (создать ноду → проверить console → snapshot) пиши Python скрипт в `tools/` или ad-hoc через `playwright`, а не через Chrome MCP в conversation.
- Frontend E2E тесты (если будут добавлены) — также через playwright, в `tests/e2e/` или подобной папке.
- Если запросили GUI screenshot для документации — `tools/screenshot_nodes.py` или новый headless playwright скрипт. Не использовать `canvas.toDataURL()` для нод с DOM widgets — потеряется их UI.
