---
name: User manages ComfyUI lifecycle
description: пользователь сам запускает и перезагружает ComfyUI; тестовые скрипты только подключаются к уже запущенному 127.0.0.1:8188
type: feedback
originSessionId: 735fb877-19df-48cd-bb11-714ae6800a94
---
Тестовые скрипты и инструменты НЕ должны запускать, останавливать или перезагружать ComfyUI. Пользователь делает это сам через UI/Restart или Ctrl+C+relaunch. Тесты (`tests/test_browser_smoke.py`, `tests/test_comfyui_live_api.py`, любые playwright/HTTP API проверки) подключаются к уже работающему серверу на 127.0.0.1:8188 и должны skip (а не fail), если он недоступен.

**Why:** пользователь управляет своим инстансом ComfyUI вручную (модели, GPU, расширения уже загружены, перезапуск стоит времени); скрипты, которые сами запускают/убивают сервер, ломают его рабочий процесс и могут конфликтовать за порт. Это устоявшийся подход в этом репо — `test_browser_smoke.py:47` и `test_comfyui_live_api.py:33` уже делают `pytest.skip(...)` при недоступности.

**How to apply:**
- Не предлагать `subprocess.Popen("python main.py")` или `comfy launch` в новых тестах/скриптах.
- Не предлагать GitHub CI matrix, который ставит ComfyUI с нуля (это другая модель — для solo-dev она избыточна).
- Любая проверка работы пакета на реальном ComfyUI = пользователь делает Restart сам, потом запускает скрипт.
- После правки production-кода в `nodes/` напоминать: «перезапусти ComfyUI, чтобы новые модули загрузились» — иначе сервер держит старые в памяти.
- Документация инструментов (preflight, smoke-скрипты) должна явно писать «connects only; never starts the server» / «restart ComfyUI manually after editing».
