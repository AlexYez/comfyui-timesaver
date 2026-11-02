---
name: feedback_runtime_only_locations
description: "История разделения dev и runtime; требование хранить память вне Git отменено консолидацией 2026-11-02."
metadata:
  type: feedback
---

История разделения dev и runtime; требование хранить память вне Git отменено консолидацией 2026-11-02.

## Статус на 2026-11-02

Новый запрос владельца о консолидации явно изменил прежнее правило об исключении документации и памяти из Git:

> Everything the project needs lives inside the repository: persistent memory (for example a `project_memory/` folder with an index), documentation, decisions and their reasons.

Поэтому память, корневые инструкции и документация теперь должны быть доступны из самого репозитория. Это не разрешает переносить в Git токены, модели, кэши и локальные тестовые артефакты. Разработка из dev checkout и запрет писать/пушить из установленного клона остаются в силе. Пути машины владельца и текущий порядок синхронизации — [описание окружения](../../doc/operations/environment.md) и [разработка](../../doc/operations/development.md).

Ниже — историческое правило хранения до этой консолидации. Упоминания обязательных `.tracking`, `.comfyignore`, `@main` и синхронизации «только через Git» устарели; фактические источники — [`.gitignore`](../../.gitignore), [workflow публикации](../../.github/workflows/publish_action.yml) и приведённые выше текущие процедуры. Примеры `reset --hard` в истории не являются рекомендацией стирать локальные правки.

## Историческая запись

> ⚠️ **ОБНОВЛЕНО 9.30:** runtime D: — обычный **git clone** GitHub-репо, обновляется `git pull` / ComfyUI Manager. Поэтому в D: ЕСТЬ `.git`/`.github`/`.gitignore` (это норма для клона, безвредно для ComfyUI). Синхронизация идёт **только через git**: push из dev C: → GitHub, затем `git pull` в D:. Dev-мусор (`tests/`/`project_memory/`/`CLAUDE.md`/`.claude/`) в клон не попадает (gitignored).

## Правило

GitHub-репозиторий (`https://github.com/AlexYez/comfyui-timesaver`) и runtime-папка ComfyUI (`D:\AiApps\ComfyUI\comfyui\ComfyUI\custom_nodes\comfyui-timesaver`) должны содержать **только runtime-файлы**, нужные конечному пользователю. Все dev-артефакты живут отдельно — в dev-location в Documents.

**Claude Code сессии открываются из dev-location**:

```
C:\Users\Sanchez\Documents\timesaver\projects\cloude-projects\comfyui-timesaver
```

Это даёт автоматически:
- Чтение этого `CLAUDE.md` и `project_memory/MEMORY.md` в startup context.
- Прямые git операции без `-C <path>` обходов.
- Работающие `pytest`, `bandit`, `compileall` без переключения cwd.

Никогда не открывать сессию из `D:\AiApps\...\custom_nodes\comfyui-timesaver` — это runtime-клон для ComfyUI, не для разработки (нет тестов, памяти, CLAUDE.md).

## Три места и что в них

| Локация | Назначение | Что лежит |
|---------|------------|-----------|
| **Dev-location** `C:\Users\Sanchez\Documents\timesaver\projects\cloude-projects\comfyui-timesaver\` | Полный git checkout, разработка. Сюда `cd` для всех правок и git операций. | `.git/`, всё runtime + `tests/`, `tools/`, `project_memory/`, `CLAUDE.md`, `.github/` |
| **GitHub** `https://github.com/AlexYez/comfyui-timesaver` | Публичный source для Comfy registry + контрибьюторов. | Только runtime + `.github/workflows/publish_action.yml` + `.gitignore` + `.comfyignore`. `tests/`, `tools/`, `project_memory/`, `CLAUDE.md` исключены через `.gitignore` (commit `893ed90`, untracked с 9.13). |
| **Runtime** `D:\AiApps\ComfyUI\comfyui\ComfyUI\custom_nodes\comfyui-timesaver\` | Загружается ComfyUI при старте. Обычный git clone, обновляется `git pull`. | Runtime + git-инфра клона (`.git/`, `.github/`, `.gitignore`); НЕ содержит тестов, CI-секретов, memory (всё gitignored, поэтому `git pull` их не тянет). |

## Runtime — точный список

Что должно быть в GitHub и custom_nodes:

- Файлы корня: `__init__.py`, `ts_dependency_manager.py`, `pyproject.toml`, `requirements.txt`, `README.md`, `README.ru.md`, `LICENSE.txt`, `icon.png`, `.tracking`
- Папки: `nodes/`, `js/`, `doc/screenshots/`
- **Только в GitHub дополнительно (обязательно, не трогать):**
  - `.github/workflows/publish_action.yml` — **автоматическая публикация в Comfy registry**. Триггер: `on: push: paths: pyproject.toml` к `master`. Без этого файла ни один bump версии не пойдёт в реестр — релизы остановятся. Использует `Comfy-Org/publish-node-action@main` + secret `REGISTRY_ACCESS_TOKEN` из repo settings.
  - `.github/workflows/ci.yml`, `.github/ISSUE_TEMPLATE/*`, `.github/pull_request_template.md` — GitHub-инфраструктура.
  - `.gitignore`, `.comfyignore` — публикации репо и Comfy zip нужны.

Что **никогда** не должно быть в GitHub и custom_nodes:

- Папки: `tests/`, `tools/`, `project_memory/`, `.claude/`, `.codex_tmp/`, `.cache/`, `.pytest_cache/`, `__pycache__/`, `.test_models/`, `.test_input/`
- Файлы: `CLAUDE.md`, `AGENTS.md`, `MODERNIZATION_PLAN.md`
- `doc/AGENTS.md`, `doc/TS_DEPENDENCY_POLICY.md`, `doc/migration.md`, `doc/generate_readme_ru.py` — только в dev

## Sync dev → runtime (после правки runtime)

Когда правишь `nodes/`, `js/`, `__init__.py`, `ts_dependency_manager.py`, `pyproject.toml`, `requirements.txt`, etc. — синхронизация идёт **только через git**:

1. В dev-location (`C:`): `git commit` → `git push origin master`.
2. В runtime (`D:`): `git -C "D:\AiApps\ComfyUI\comfyui\ComfyUI\custom_nodes\comfyui-timesaver" pull --ff-only origin master` (или ComfyUI Manager → Update). Один раз в конце, после всех push-ей.
3. Пользователь перезапускает ComfyUI, чтобы перечитать модули.

Dev-мусор (`tests/`, `tools/`, `project_memory/`, `CLAUDE.md`, dev-doc) gitignored — `git pull` его не тянет, runtime остаётся чистым автоматически. `.cache/` (HF model cache) в D: git не трогает (gitignored), он переживает любой pull. Подробнее — [feedback_commit_then_sync](feedback_commit_then_sync.md).

## ОПАСНОСТЬ — не сноси custom_nodes целиком

`Remove-Item -Recurse -Force` на `custom_nodes\comfyui-timesaver` фейлится посередине из-за залоченных permission'ами `.cache\models_native\rife\.hf_cache\.../tmp*` папок и **оставляет custom_nodes наполовину уничтоженным** (без top-level файлов, но с залоченным `.cache/`). Не сноси клон целиком; если структура в D: повреждена — восстанавливай через git (`git -C <D:> checkout -- .` или `git -C <D:> reset --hard origin/master`), а не delete+recreate.

## Junction-альтернатива

Можно сделать junction `custom_nodes\comfyui-timesaver` → `Documents\...\comfyui-timesaver` (single source of truth), но для этого нужно успешно очистить custom_nodes (включая залоченные `.cache/`) — а это требует admin/перезагрузки. Пока остаётся git-clone + `git pull` workflow.

## Если сессия открыта из runtime-копии `D:`

Это бывает: в `D:\…\project_memory\` лежит обрывок памяти, и стартовый
контекст не показывает правил отсюда. Порядок действий:

1. Просят править код — **стоп**. Сначала попросить перезапустить Claude из
   dev-локации `C:` (там `.git`, тесты, полная память, CLAUDE.md).
2. Никогда не делать `git commit` / `git push` в `D:` и не заводить там веток.
3. Если правка в `D:` всё же случилась — перенести её в `C:` (cherry-pick или
   копирование), закоммитить и запушить из `C:`, затем вернуть `D:` в зеркало:
   `git -C <D> reset --hard origin/master`. Перед reset — ветка `backup-*`.

**Инцидент 2026-06-22 (так и вышло).** Сессия из `D:`: правка
`nodes/image/ts_smart_inpaint.py` закоммичена прямо в runtime (`37ae6d4`), рядом
висел незапушенный `fa14091`. Починено cherry-pick'ом обоих в `C:` поверх 9.58,
вычисткой трейлера `Co-Authored-By` (см. [feedback_no_coauthor_footer](feedback_no_coauthor_footer.md)),
push из `C:` (`deccb54`) и `reset --hard origin/master` в `D:`. Блобы всех шести
файлов совпали побайтово.
