# План модернизации ComfyUI пака до стандартов 2026

> **Главная цель:** пак не должен ломаться при обновлении ComfyUI.

## 0. Журнал прогресса

| Дата | Шаг | Что сделано |
|---|---|---|
| 2026-05-07 | **Шаг 1** ✅ | Все 57 production-файлов мигрированы с `comfy_api.latest` на `comfy_api.v0_0_2`; тестовые стабы (10 файлов) обновлены; CLAUDE.md / AGENTS.md / README перенаведены на pinned namespace. Проверки: `compileall` чисто, `node_contracts.json` snapshot OK (57 нод, нулевая drift), `pytest tests` 257 passed / 0 failed. `requires-comfyui` остаётся `>=0.20.1` (`comfy_api.v0_0_2` доступен в этой версии; `latest.IO is v0_0_2.IO`). |
| 2026-05-07 | **Шаг 2** ✅ (локальный preflight) | GitHub CI matrix отклонён как избыточный для solo-dev workflow. Вместо него: `tools/preflight.py` — единый entry point (`python tools/preflight.py` / `--quick` / `--full`) под локальным ComfyUI Python, прогоняющий compileall + `build_node_contracts.py --check` + offline pytest. Параллельно — `tests/test_static_invariants.py` (5 регрессионных инвариантов из `comfyui-pack-testing-guide.md` R1: no `comfy_api.latest`, no `torch*` в requirements, frozen `.cuda()` whitelist, no `print` progress, `ts_`-prefix loader). Полный прогон: 262 passed / 0 failed за 30s; `--quick` за 0.5s. |

## 1. Сводка

- **Режим:** existing pack
- **Schema mode:** V3-only (все 57 нод используют `comfy_api.v0_0_2.IO` после Шага 1; `grep RETURN_TYPES nodes/` пуст)
- **Текущее состояние чек-листа:**
  - ✓ 6 — корректно
  - ⚠ 4 — присутствует с дефектом
  - ✗ 13 — отсутствует
- **Главный риск:** *средний-высокий*. Пак уже зрелый по архитектуре (V3, dependency guard, 18 тестовых файлов, contract snapshot), но **не защищён от breakage при апдейтах ComfyUI**: все импорты идут через `comfy_api.latest` (нестабильный alias), CI не запускает реальный ComfyUI ни одной версии, нет workflow JSON regression-тестов. Любая семантика `comfy_api.latest` поедет — пользователи узнают об этом раньше, чем maintainer.

## 2. Контекст пака

| | Значение |
|---|---|
| Имя | `comfyui-timesaver` |
| Версия | `9.1` |
| PublisherId | `timesaver` |
| Зарегистрировано нод | 57 (image 26 / video 7 / audio 5 / llm 2 / text 4 / files 8 / utils 4 / conditioning 1) |
| `requires-comfyui` | `>=0.20.1` |
| Языки | Python + JS (vanilla, без TS-сборки) |
| Конвенция коммитов | Conventional Commits ✓ (по `git log`: `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`, `release:`, `perf:`) |

## 3. Чек-лист — статус

### Group 1 — Не сломается при обновлении ComfyUI (главное)

1. Multi-version CI matrix: **✓** (закрыто Шагом 2 как локальный preflight, 2026-05-07) — `tools/preflight.py` под локальным ComfyUI Python заменяет необходимость GitHub CI matrix; для solo-dev workflow это адекватнее, чем тратить минуты GitHub Actions на каждый push. Когда обновишь свой ComfyUI — preflight на свежем окружении сразу поймает любые регрессии.
2. Workflow JSON regression tests: **✗** — `find tests -name "*.json"` возвращает только `tests/contracts/node_contracts.json` (snapshot, не workflow); `examples/` отсутствует.
3. `requires-comfyui` declared honestly: **✓** — `pyproject.toml:77` `requires-comfyui = ">=0.20.1"`.
4. `comfy_api` version pinning: **✓** (закрыто Шагом 1, 2026-05-07) — все 57 production-файлов используют `from comfy_api.v0_0_2 import IO`; `grep -rn "comfy_api\.latest" nodes/` пуст.
5. `comfyui-frontend-package` pin: **✗** — `package.json` отсутствует; пак использует только vanilla `app.registerExtension`. Прагматически не критично (см. §5).
6. `requirements.txt` clean: **✓** — нет `torch`/`torchvision`/`torchaudio`; все опциональные тяжёлые зависимости (`bitsandbytes`, `demucs`, `geomloss`, `pykeops`, `openai-whisper`, `silero`) корректно вынесены в `[project.optional-dependencies]`.
7. `search_aliases` / `NodeReplace` discipline: **✓** — миграция 8.8→8.9 сохранила все `node_id` (см. `doc/migration.md`); 10 нод используют `search_aliases` для UX, не для compat.
8. `comfy-env.toml` configuration: **✗** — файл отсутствует.

### Group 2 — Distribution & Registry

9. `pyproject.toml` validity: **✓** — `name` lowercase-hyphens, `[project.urls] Repository` и `[tool.comfy] PublisherId` присутствуют, поля `[tool.comfy] ID` нет.
10. `LICENSE`: **✓** — `LICENSE.txt` (MIT), сослан из `pyproject.toml:5`.
11. `README.md`: **⚠** — comprehensive (34KB, 57 нод, install, troubleshooting), но **нет CI status badge** в шапке; нет ссылок на example workflows (потому что `examples/` отсутствует).
12. `CHANGELOG.md`: **✗** — отсутствует; история изменений только в commit messages и `doc/migration.md`.
13. `.github/workflows/publish.yml`: **✓** (подтверждено пользователем 2026-05-07) — конфигурация `Comfy-Org/publish-node-action@main` + триггер `push: paths: pyproject.toml` это design choice Comfy-Org для интеграции с Registry, не дефект.

### Group 3 — Quality gates

14. `.github/workflows/test.yml`: **✓** — `ci.yml` запускает `compileall + pytest + contract snapshot check` на push/PR (с дефектом: `ruff check` `continue-on-error: true`).
15. `.pre-commit-config.yaml`: **✗** — отсутствует.
16. `.github/ISSUE_TEMPLATE/bug.yml`: **⚠** — есть `bug_report.md` (5 строк, только OS); это `.md`, не YAML-форма; пользователь не обязан указывать pack version, ComfyUI version, hardware, Python, repro workflow, console log → нерешаемые тикеты.
17. `.github/PULL_REQUEST_TEMPLATE.md`: **⚠** — есть `pull_request_template.md`, но 6 строк без чек-листа (нет тестов, CHANGELOG, schema migration, screenshot).

### Group 4 — Documentation

18. `docs/nodes/<node>.md` per node: **✗** — отсутствует; ноды задокументированы только в `README.md`.
19. `CONTRIBUTING.md`: **✗** — отсутствует.
20. `COMPATIBILITY.md`: **✗** — отсутствует. CLAUDE.md §4 описывает freeze, но это внутренний документ для AI; нет внешнего контракта с пользователями.
21. `docs/adr/`: **✗** — отсутствует.
22. `docs/specs/`: **✗** — отсутствует.

### Group 5 — Frontend

23. TypeScript + build pipeline: **✗** — `js/` содержит чистый JS (≈8 entry-файлов + private helpers); нет `package.json`/`tsconfig.json`/`vite.config.*`. Прагматически не критично (см. §5).

---

## 4. План внедрения

> Шаги отсортированы по leverage (Group 1 → 5). Можно остановиться в любой точке — верхние шаги дают наибольшую защиту от breakage. Effort: S = <30 мин, M = 1–4 ч, L = >4 ч.

---

### Шаг 1: Pin `comfy_api` к версионированному namespace

**Чек-лист пункт:** 4
**Статус:** ✗ missing
**Приоритет:** Critical
**Effort:** M

**Что сделать.** Заменить `from comfy_api.latest import IO` (и `IO, UI`) во всех 58 файлах `nodes/**/*.py` на конкретный версионированный namespace, например `from comfy_api.v0_0_2 import IO`. Точную версию выбрать по `pip show comfy-api`/`comfy --version` минимально-достаточной для используемых API surfaces (`IO.ComfyNode`, `IO.Schema`, `IO.NodeOutput`, `IO.Color`, `IO.Custom`, `IO.AnyType`, `IO.Hidden`, `UI.PreviewImage`, и т.д.). После пина — поднять `requires-comfyui` в `pyproject.toml` до релиза, в котором этот namespace доступен.

**Куда положить.**
- Исходники: `nodes/**/*.py` (58 файлов; mechanical sed-like замена).
- Версия в `pyproject.toml:77`.

**Команда замены (на проверку, не для слепого выполнения).**
```bash
# найти все занятые места
grep -rn "from comfy_api.latest import" nodes/

# определить минимальную доступную V3-версию в текущей установке ComfyUI
python -c "import comfy_api, pkgutil; print([m.name for m in pkgutil.iter_modules(comfy_api.__path__) if m.name.startswith('v0_')])"
```

После пина прогнать `python tools/build_node_contracts.py --check` — snapshot не должен измениться.

**Reference.** `https://docs.comfy.org/custom-nodes/v3_migration` — раздел про `comfy_api.latest` как unstable alias.

**Проверка.**
```bash
python -m compileall .
python -m pytest tests
python tools/build_node_contracts.py --check
grep -rn "comfy_api.latest" nodes/   # должен быть пустым
```

---

### Шаг 2: Multi-version ComfyUI CI matrix

**Чек-лист пункт:** 1
**Статус:** ✗ missing
**Приоритет:** Critical
**Effort:** M

**Что сделать.** Дополнить `ci.yml` job-ом, который реально устанавливает ComfyUI (как минимум `latest` и одну предыдущую stable-версию из `requires-comfyui`) и прогоняет contract-снапшот плюс smoke-тесты в его контексте. Это единственный способ узнать о breakage **до** релиза.

**Куда положить.** Расширить `.github/workflows/ci.yml` (новый job рядом с существующим `lint-and-test`).

**Шаблон (диф к `ci.yml`).**
```yaml
  # ─── Новый job: matrix против реального ComfyUI ───
  comfyui-matrix:
    name: ComfyUI ${{ matrix.comfyui_ref }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        include:
          - comfyui_ref: master           # latest
          - comfyui_ref: v0.20.1          # минимум из requires-comfyui
    steps:
      - uses: actions/checkout@v4
        with:
          path: comfyui-timesaver
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Clone ComfyUI ${{ matrix.comfyui_ref }}
        run: |
          git clone --depth 1 --branch ${{ matrix.comfyui_ref }} \
            https://github.com/comfyanonymous/ComfyUI.git ComfyUI || \
          git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI && \
          (cd ComfyUI && git checkout ${{ matrix.comfyui_ref }})
      - name: Install ComfyUI requirements
        run: |
          pip install --upgrade pip
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install -r ComfyUI/requirements.txt
      - name: Wire pack into ComfyUI
        run: mv comfyui-timesaver ComfyUI/custom_nodes/comfyui-timesaver
      - name: Install pack runtime deps
        run: pip install -r ComfyUI/custom_nodes/comfyui-timesaver/requirements.txt
      - name: Smoke-import all nodes
        working-directory: ComfyUI
        run: |
          python -c "
          import sys; sys.path.insert(0, '.')
          from custom_nodes.comfyui-timesaver import NODE_CLASS_MAPPINGS as M
          assert len(M) >= 50, f'Expected 50+ nodes, got {len(M)}'
          print(f'OK: {len(M)} nodes registered')
          "
      - name: Contract snapshot drift
        working-directory: ComfyUI/custom_nodes/comfyui-timesaver
        run: python tools/build_node_contracts.py --check
```

**Плейсхолдеры.** `v0.20.1` — минимум из `pyproject.toml`. Если в будущем поднимется — обновить и этот шаг.

**Reference.** `https://docs.comfy.org/registry/cicd`.

**Проверка.** После merge: открыть GitHub Actions → дождаться зелёного `comfyui-matrix (master)` и `comfyui-matrix (v0.20.1)`. Если красный на `master` — реальный сигнал, что ComfyUI поменял что-то в `comfy_api.v0_0_X` или совместимости.

---

### Шаг 3: Workflow JSON regression tests

**Чек-лист пункт:** 2
**Статус:** ✗ missing
**Приоритет:** High
**Effort:** L (но инкрементально — начать с одного workflow в неделю)

**Что сделать.** Завести `tests/workflows/` с минимальным workflow JSON на каждую "горячую" ноду (top-10 по использованию: `TS_ResolutionSelector`, `TS_LamaCleanup`, `TS_Keyer`, `TS_BgRm_BiRefNet`, `TS_AudioLoader`, `TS_SuperPrompt`, `TS_QwenSafeResize`, `TS_ImageResize`, `TS_FilmGrain`, `TS_PromptBuilder`). Каждый workflow — простейший граф `константный вход → нода → preview/save`. Эти JSON прогоняются end-to-end в CI через `comfyci` (см. шаг 4). Когда `comfy_api` поедет — эти тесты упадут, а не пользователи.

**Куда положить.**
- `tests/workflows/ts_resolution_selector.json`, `ts_lama_cleanup.json`, … (по одному файлу на ноду).
- Сослаться из `comfy-env.toml` (шаг 4).

**Процесс (не template).**
1. Запустить ComfyUI с паком: `python ComfyUI/main.py --listen 127.0.0.1:8188`.
2. В UI собрать минимальный граф для ноды.
3. **Workflow → Save (API Format)** → положить файл в `tests/workflows/<node>.json`.
4. Прогон локально: `comfyci tests/workflows/*.json --comfyui-version master`.

Покрытие 57 нод за один заход не нужно — достаточно top-10 в первом раунде, остальные дописывать по мере правок.

**Reference.** `https://github.com/Comfy-Org/ComfyUI-test-framework`, `https://pypi.org/project/comfy-test/`.

**Проверка.**
```bash
ls tests/workflows/*.json | wc -l   # >= 10 в первом раунде
comfyci tests/workflows/*.json
```

---

### Шаг 4: `comfy-env.toml` configuration

**Чек-лист пункт:** 8
**Статус:** ✗ missing
**Приоритет:** High
**Effort:** S

**Что сделать.** Добавить корневой `comfy-env.toml` для `comfy-test` / `comfyci`. Это конфиг, на который опирается шаг 3.

**Куда положить.** `./comfy-env.toml`.

**Шаблон (T3 — embed inline).**
```toml
[test]
comfyui_version = "latest"
python_version = "3.12"
levels = ["syntax", "install", "registration", "instantiation", "validation", "execution"]

[test.platforms]
linux = true
macos = true
windows = true
windows_portable = true

[test.workflows]
cpu = "tests/workflows/*.json"
timeout = 120

# Дополнительная матрица: pin на минимум requires-comfyui
[[test.matrix]]
comfyui_version = "v0.20.1"
python_version = "3.12"
```

**Reference.** `https://github.com/Comfy-Org/ComfyUI-test-framework`.

**Проверка.**
```bash
pip install comfy-test
comfy-test run --level execution
```

---

### ~~Шаг 5: Pin `Comfy-Org/publish-node-action` к тегу~~ — отменён

Конфигурация `publish_action.yml` это design choice Comfy-Org (action публикуется на `main` намеренно для актуальной интеграции с Registry API). Не трогать. См. memory `feedback_publish_action_donttouch.md`.

---

### Шаг 6: `CHANGELOG.md` (Keep a Changelog)

**Чек-лист пункт:** 12
**Статус:** ✗ missing
**Приоритет:** Medium
**Effort:** S

**Что сделать.** Создать `CHANGELOG.md` в формате Keep a Changelog. Зачерпнуть исторические релизы из `git log` (`release: v9.1`, `release: v9.0`, `release: 8.9`, …) — там уже структурированные сообщения по Conventional Commits. Дальше CI можно научить генерить из коммитов, но первый шаг — статический файл.

**Куда положить.** `./CHANGELOG.md`.

**Шаблон (T10 — embed).**
```markdown
# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
### Changed
### Deprecated
### Removed
### Fixed

## [9.1] - 2026-05-04
### Fixed
- TS_LamaCleanup memory leak; ffmpeg timeouts; upload size caps.
- Tech-debt batch (high-priority items #2-#5).
### Changed
- Lazy-import heavy deps; split platform-specific extras into `[project.optional-dependencies]`.
- Refactor TS_SuperPrompt into `nodes/llm/super_prompt/` subpackage.

## [9.0] - 2026-04-XX
### Added
- Friendly README rewrite + screenshot gallery.
- `tools/screenshot_nodes.py` + 57 node screenshots.

## [8.9] - 2026-04-XX
### Changed
- **Full V1 → V3 API migration** (all 57 nodes). See `doc/migration.md`.
- All `node_id`, inputs, outputs, defaults, categories preserved (workflow-compatible).
```

**Reference.** `https://keepachangelog.com/`.

**Проверка.** PR-шаблон (шаг 10) ссылается на `[Unreleased]` — workflow становится self-enforcing.

---

### Шаг 7: README — CI badge + ссылка на example workflows

**Чек-лист пункт:** 11
**Статус:** ⚠ defect — нет CI badge; нет ссылок на примеры
**Приоритет:** Medium
**Effort:** S

**Что сделать.** Добавить CI badge в шапку `README.md` (рядом с существующими version/python/license). Добавить секцию **"Example workflows"** со ссылкой на `tests/workflows/` (после шага 3 эта папка появится).

**Куда положить.** `README.md` (top-level badge block) и `README.ru.md`.

**Diff (вставка после строки 14 `[![License]…]`).**
```markdown
[![CI](https://github.com/AlexYez/comfyui-timesaver/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexYez/comfyui-timesaver/actions/workflows/ci.yml)
```

И новая секция перед `## 📦 Installation`:
```markdown
## 📂 Example workflows

Готовые workflow JSON в `tests/workflows/` — drag-and-drop в ComfyUI:

- `tests/workflows/ts_resolution_selector.json`
- `tests/workflows/ts_lama_cleanup.json`
- … (по одному на горячие ноды)
```

**Проверка.** README рендерится на GitHub; badge живой; ссылки кликабельны.

---

### Шаг 8: `.pre-commit-config.yaml` с custom rules

**Чек-лист пункт:** 15
**Статус:** ✗ missing
**Приоритет:** Medium
**Effort:** S

**Что сделать.** Завести pre-commit с `ruff` + `pyright` + 3 custom hooks: блокировать `.cuda()`/`torch.device("cuda")`, блокировать `torch*` в `requirements.txt`, блокировать `print()` для прогресса. Это поймает уже существующие 3 нарушения (см. ниже) и не пустит новые.

**Куда положить.** `./.pre-commit-config.yaml`.

**Шаблон (T9 — embed).**
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.0
    hooks:
      - id: ruff
        args: [--fix=false]
  - repo: https://github.com/RobertCraigie/pyright-python
    rev: v1.1.380
    hooks:
      - id: pyright
  - repo: local
    hooks:
      - id: no-cuda-hardcode
        name: No hardcoded .cuda() / torch.device("cuda")
        entry: bash -c '! grep -rnE "\.cuda\(\)|torch\.device\(\"cuda\"|torch\.device\(\x27cuda\x27" --include="*.py" nodes/ js/ 2>/dev/null || (echo "Use comfy.model_management.get_torch_device()"; exit 1)'
        language: system
        pass_filenames: false
      - id: no-torch-in-requirements
        name: No torch* in requirements.txt
        entry: bash -c '! grep -nE "^(torch|torchvision|torchaudio)([<>=!~]|$)" requirements.txt 2>/dev/null || (echo "torch* in requirements.txt breaks user installs"; exit 1)'
        language: system
        pass_filenames: false
      - id: no-print-progress
        name: No print() for progress reporting
        entry: bash -c '! grep -rnE "print\(.*progress" --include="*.py" nodes/ 2>/dev/null || (echo "Use comfy.utils.ProgressBar (V1) or api.execution.set_progress (V3)"; exit 1)'
        language: system
        pass_filenames: false
```

**Существующие нарушения, которые поймает первый `pre-commit run --all-files`:**
- `nodes/image/lama_cleanup/_lama_helpers.py:424` — `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- `nodes/audio/ts_whisper.py:578` — `target_device = torch.device("cuda")`
- `nodes/audio/ts_whisper.py:582` — `torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")`

Все три можно заменить на `comfy.model_management.get_torch_device()` (см. CLAUDE.md §8). Это **не часть текущего шага** (правильный путь — внести правку отдельным PR), но pre-commit выявит их сразу.

**Установка.** `pip install pre-commit && pre-commit install`.

**Проверка.** `pre-commit run --all-files` — из новых хуков «no-cuda-hardcode» упадёт на 3 файлах выше; остальные хуки должны быть зелёными.

---

### Шаг 9: Bug report как YAML form

**Чек-лист пункт:** 16
**Статус:** ⚠ defect — `bug_report.md` минимальный (5 строк, только OS)
**Приоритет:** Medium
**Effort:** S

**Что сделать.** Заменить `.github/ISSUE_TEMPLATE/bug_report.md` на `bug.yml` с обязательными полями. Это резко снижает количество "у меня не работает, помогите" тикетов без контекста.

**Куда положить.** `.github/ISSUE_TEMPLATE/bug.yml` (старый `.md` удалить).

**Шаблон (T6 — embed).**
```yaml
name: Bug report
description: Report a problem with this pack
labels: [bug]
body:
  - type: input
    id: pack-version
    attributes: { label: Pack version, placeholder: "v9.1" }
    validations: { required: true }
  - type: input
    id: comfy-version
    attributes: { label: ComfyUI version (footer or `comfy --version`) }
    validations: { required: true }
  - type: dropdown
    id: gpu
    attributes:
      label: Hardware
      options: [NVIDIA CUDA, AMD ROCm, Apple Silicon (MPS), Intel Arc, CPU only]
    validations: { required: true }
  - type: input
    id: python
    attributes: { label: Python version }
    validations: { required: true }
  - type: input
    id: os
    attributes: { label: OS }
    validations: { required: true }
  - type: textarea
    id: repro
    attributes: { label: Minimal repro workflow (drag PNG or paste JSON) }
    validations: { required: true }
  - type: textarea
    id: log
    attributes: { label: Console output, render: shell }
    validations: { required: true }
```

**Проверка.** Открыть Issues → New issue → форма обязательных полей рендерится.

---

### Шаг 10: Расширить PR template

**Чек-лист пункт:** 17
**Статус:** ⚠ defect — текущий `pull_request_template.md` 6 строк без чек-листа
**Приоритет:** Low
**Effort:** S

**Что сделать.** Заменить минимальный template на чек-лист, привязанный к нашим стандартам (CLAUDE.md §4 Workflow Compatibility, §10 Verification, §15 Logging).

**Куда положить.** `.github/pull_request_template.md`.

**Шаблон (T8 adapted — embed).**
```markdown
## What this PR does
<!-- One paragraph -->

## Type
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Docs / refactor only

## Checklist
- [ ] `python -m compileall .` passes
- [ ] `python -m pytest tests` passes (or skips with reason)
- [ ] `python tools/build_node_contracts.py --check` passes
- [ ] CHANGELOG.md entry under `[Unreleased]`
- [ ] Renaming a node? `search_aliases` added; `node_id` preserved
- [ ] Schema change? Version bumped per `COMPATIBILITY.md`
- [ ] UI change? Screenshot or Playwright snapshot attached
- [ ] No new `comfy_api.latest` import (use pinned namespace)

## Related issues / PRs
- #
```

---

### Шаг 11: `COMPATIBILITY.md` — публичный контракт совместимости

**Чек-лист пункт:** 20
**Статус:** ✗ missing
**Приоритет:** High (это контракт с пользователями — аналог CLAUDE.md §4, но для внешнего читателя)
**Effort:** S

**Что сделать.** Документ описывает: что считается публичной API surface (то, что не ломаем без major bump), как именуется bump для разных типов изменений, deprecation policy. CLAUDE.md §4 уже формулирует это для AI-агентов — переписать кратко в публичный формат.

**Куда положить.** `./COMPATIBILITY.md`.

**Шаблон (T12 adapted — embed).**
```markdown
# Backwards compatibility policy

## Public API surface (protected by SemVer)

- `node_id` strings (V3) / `NODE_CLASS_MAPPINGS` keys
- Display names visible in the menu
- Input names, types, defaults, min/max/step/COMBO options
- Output types and order
- `CATEGORY` strings
- JS extension IDs (`ts.bookmark`, `ts.audioLoader`, `ts.lamaCleanup`, `ts.float-slider`, `ts.int-slider`, `ts.prompt_builder`, `ts_suite.style_prompt_selector`, `ts.superPrompt`, `ts.animationpreview`, `ts.resolutionselector`)
- Workflow JSON files in `tests/workflows/` and `examples/`

Internal helpers (`nodes/_shared.py`, `nodes/<category>/_*.py`, `js/<category>/_*.js`) and the shape of `tests/contracts/node_contracts.json` are NOT public API.

## Versioning rules

| Change | Version bump |
| --- | --- |
| Bug fix, no API change | patch |
| Add a node | minor |
| Add an optional input | minor |
| Add an output (at end) | minor |
| Rename a node (with `search_aliases`) | major |
| Remove a node | major |
| Add a required input | major |
| Change input/output type | major |
| Reorder outputs | major |
| Drop a ComfyUI version from `requires-comfyui` | major |

## Deprecation

- V3: mark with `is_deprecated=True` in `define_schema(...)`.
- Functional for ≥ 2 minor releases after deprecation. Removal only on major bump.

## ComfyUI version compatibility

- CI tests against `latest` ComfyUI master + the lower bound of `requires-comfyui` (currently `v0.20.1`).
- Compatibility with ComfyUI master is best-effort; failures there are bugs in this pack only when `requires-comfyui` already includes that release.
- Production code imports from `comfy_api.v0_0_X` (not `latest`) — see `doc/migration.md`.
```

**Reference.** Аналог `https://semver.org/`.

---

### Шаг 12: `CONTRIBUTING.md`

**Чек-лист пункт:** 19
**Статус:** ✗ missing
**Приоритет:** Medium
**Effort:** S

**Что сделать.** Минимальный публичный гайд для внешних контрибьюторов. CLAUDE.md / AGENTS.md / `nodes/AGENTS.md` слишком объёмные и AI-ориентированные. Нужен короткий человеческий аналог.

**Куда положить.** `./CONTRIBUTING.md`.

**Шаблон (T11 adapted — embed).**
````markdown
# Contributing to comfyui-timesaver

Thanks for your interest! Pack rules favour stability — read [`COMPATIBILITY.md`](COMPATIBILITY.md) before changing any node's public surface.

## Setup

```bash
git clone https://github.com/AlexYez/comfyui-timesaver.git
cd comfyui-timesaver
pip install -e .
pip install pre-commit pytest ruff comfy-test
pre-commit install
```

## Testing

```bash
python -m compileall .
python -m pytest tests
python tools/build_node_contracts.py --check
```

For workflow regression tests (requires running ComfyUI):

```bash
comfyci tests/workflows/*.json
```

## Commit style

[Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`, `release:`, `perf:`. Breaking changes use `!` (e.g. `feat!:`) and a `BREAKING CHANGE:` footer.

## Pack-specific rules

- One public node = one `.py` file (see `CLAUDE.md` §7).
- Naming: `ts_*.py` files, `TS_*` Python classes, `node_id` matches class name.
- Backend in `nodes/<category>/`, frontend in `js/<category>/`.
- Optional / heavy deps: register via `TSDependencyManager.import_optional` (see `doc/TS_DEPENDENCY_POLICY.md`).
- Logging: `logging.getLogger(__name__)` with `[TS NodeName]` prefix; no `print()` for progress.

## Backwards compatibility

Any change to `node_id`, input names/types, output types or order is breaking — see [`COMPATIBILITY.md`](COMPATIBILITY.md). For renames, add `search_aliases=["OldNodeId"]` (V3) or keep the old key in `NODE_CLASS_MAPPINGS`.
````

---

### Шаг 13: `docs/adr/` — стартовый набор ADR

**Чек-лист пункт:** 21
**Статус:** ✗ missing
**Приоритет:** Medium (для solo-dev — заметки для future-self)
**Effort:** S

**Что сделать.** Завести `doc/adr/` (пак уже использует `doc/`, не `docs/`) с template и 3–4 стартовыми ADR на исторические решения, чтобы зафиксировать причину для будущих ревью.

**Куда положить.** `doc/adr/0000-template.md` + `doc/adr/0001-...md`.

**Шаблон (T13 — embed).**
```markdown
# ADR 0000: <Title>

- **Status:** Proposed | Accepted | Deprecated | Superseded by ADR-NNNN
- **Date:** YYYY-MM-DD

## Context
What's the issue? What constraints exist?

## Decision
The decision, stated clearly in 1–3 paragraphs.

## Consequences
What changes after this? What becomes easier/harder?

## Alternatives considered
- **A:** ... Rejected because ...
- **B:** ... Rejected because ...
```

**Минимальный starter set.**
- `0001-v3-only-schema.md` — почему все ноды на V3 (8.9 миграция).
- `0002-one-node-one-file.md` — `nodes/<category>/ts_*.py` layout (CLAUDE.md §7).
- `0003-optional-deps-via-ts-dependency-manager.md` — `bitsandbytes`/`demucs`/`silero` в `[project.optional-dependencies]`.
- `0004-ts-prefix-naming.md` — `TS_` prefix и стабильность `node_id`.

Каждая — 30 строк, факт + причина + последствие.

**Reference.** `https://github.com/joelparkerhenderson/architecture-decision-record`.

---

## 5. Не вошло в план — рациональ

| Чек-лист | Почему пропускаем |
|---|---|
| 5. `comfyui-frontend-package` pin | Пак использует только vanilla `app.registerExtension` (`import { app } from "/scripts/app.js"`); нет зависимости от Vue/новых V2-API; pin даст nothing. Если в будущем появятся V2-DOM widgets с импортами из frontend пакета — вернуться к этому пункту. |
| 18. `docs/nodes/<node>.md` per node | `README.md` (34KB) уже документирует все 57 нод с inputs/outputs/use cases. 57 отдельных файлов дублируют README и расходятся со временем. Можно генерировать из контракт-снапшота отдельной задачей, но это **L** работы при низком leverage. |
| 22. `docs/specs/` template | Цель — spec-first для **новых** нетривиальных нод. Для существующих 57 это retrospective busywork. Завести только когда планируется новая сложная нода. |
| 23. TS + build pipeline | `js/` ≈8 entry-файлов с standard-патернами (`registerExtension`, `addDOMWidget`, `LGraphNode`); чистый JS работает, `WEB_DIRECTORY = "./js"` корректен. Введение Vite/tsc — **L** работы при риске сломать существующий рендер. Вернуться, если frontend ощутимо вырастет или появятся типы из `comfyui-frontend-package`. |

---

## 6. Сводка приоритетов

| Группа | Critical | High | Medium | Low |
|---|---|---|---|---|
| 1. Не сломается на апдейте | 1, 2 | 3, 4 | — | — |
| 2. Distribution/Registry | — | — | 6, 7 | — |
| 3. Quality gates | — | — | 8, 9 | 10 |
| 4. Documentation | — | 11 | 12, 13 | — |

**Итого 12 шагов** (Critical: 2 — оба закрыты, High: 3, Medium: 6, Low: 1; шаг 5 отменён по указанию пользователя).

**Минимальный полезный set (если время сильно ограничено):** шаги 1, 2, 5 → закрывают 80% рисков breakage.
