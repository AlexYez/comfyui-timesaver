---
name: Comfy Registry — .comfyignore и scanner scope
description: comfy-cli zip = git ls-files − .comfyignore (рабочее, но не задокументировано); ВАЖНО — security scanner анализирует ПОЛНЫЙ git checkout, .comfyignore ему не указ
type: reference
originSessionId: d94f9eb6-fcc1-4ae7-9326-c4cdeb36a1bb
---
## КРИТИЧНО: scanner ≠ zip

Comfy registry security scanner анализирует **весь git checkout** (то, что лежит в репо на момент `actions/checkout`), а не финальный zip. Поэтому `.comfyignore` помогает только end-user'у (меньше файлов в установленном паке), но **не защищает от Flagged-статусов**. Подтверждено мейнтейнером Comfy-Org в Discord (2026-05-13): «.comfyignore doesn't ignore during the checks, even if the files aren't distributed».

Следствие: любой подозрительный паттерн в `tests/`, `tools/`, `doc/*.py` — даже исключённый из zip — всё равно увидит сканер.

Триггеры в этом паке исторически:
- `exec(compile(source, ..., "exec"), namespace)` в `tests/test_pack_imports.py` — был с v9.2 до v9.8, вызывал bandit `B102:exec_used`. Починен в v9.9 заменой на `importlib.util.spec_from_file_location` + `exec_module` с явным `module.__package__ = ""`.

Возможные триггеры в production коде (пока не подтверждены как Flagged-причина, но потенциально):
- `subprocess.run/Popen` в `nodes/audio/loader/_audio_helpers.py`, `nodes/llm/super_prompt/_voice.py`, `nodes/video/ts_animation_preview.py` — для ffmpeg/ffprobe (в 8.6–9.1 эти же вызовы проходили Active).
- `aiohttp.web.Route` handlers в lama_cleanup, sam_media_loader, audio loader, super_prompt — для in-node DOM widgets.

## .comfyignore (рабочее)

`comfy node publish` (Comfy-Org/publish-node-action → comfy-cli) собирает zip следующим образом (`comfy_cli/file_utils.py::zip_files`):

1. Базовый список — `git -C . ls-files`. Всё, что в `.gitignore`, автоматически отсекается.
2. Поверх него применяется `.comfyignore` из корня репо (gitwildmatch через `pathspec`). Это поле **не задокументировано в docs.comfy.org**, но реализовано и работает.
3. `[tool.comfy].includes` в pyproject.toml — force-include, обходит `.comfyignore`.
4. Hardcoded skip — только `.git` в fallback-ветке без git.

В этом паке `.comfyignore` исключает: `tests/`, `tools/`, `.github/`, `doc/AGENTS.md`, `doc/TS_DEPENDENCY_POLICY.md`, `doc/migration.md`, `doc/generate_readme_ru.py`. Симуляция: 259/295 файлов остаются в pack zip, подтверждено скачиванием опубликованного 9.8 zip.

## Проверка после релиза

`GET https://api.comfy.org/nodes/comfyui-timesaver` → если новая версия в `latest_version` со `status: NodeVersionStatusActive` — сработало. Версии в статусе `NodeVersionStatusFlagged` не отдаются как latest и скрыты от ComfyUI Manager. `status_reason` в API не возвращается — причину Flagged надо запрашивать у Comfy-Org Discord/GitHub.
