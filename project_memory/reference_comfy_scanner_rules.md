---
name: Comfy registry scanner — конкретные bandit-rules, что флагают
description: Comfy registry security scanner = bandit (или fork) на full repo checkout; флагают B615/B310/B108 — типовые в ComfyUI-нодах паттерны. v9.13 нейтрализовал их.
type: reference
---

## Что точно срабатывало (выявлено в v9.13 audit)

`python -m bandit -r . -ll` находит ровно ту же группу триггеров, что Comfy registry помечает как Flagged. Установка `bandit` локально под ComfyUI Python и прогон — это самый быстрый способ воспроизвести scanner-output **до** публикации, без слепых попыток. Установка: `D:/AiApps/ComfyUI/comfyui/python/python.exe -m pip install bandit`.

Подтверждённые false-positive в этом паке:

| Rule | Что флагает | Где | Что сделано |
|------|-------------|-----|-------------|
| **B615** `huggingface_unsafe_download` | `snapshot_download()` / `from_pretrained()` без `revision=` SHA. Plugin не считает `revision="main"` достаточным — требует commit-hash, что ломает «всегда latest stable» семантику. | 7 мест: `ts_bgrm_birefnet.py:639,670`, `ts_matting_vitmatte.py:313,398,401`, `_qwen_engine.py:1063`, `ts_video_depth.py:356` | Добавлен `revision="main"` (явный intent) + `B615` в `[tool.bandit] skips` в `pyproject.toml` с обоснованием. |
| **B310** `urllib_urlopen` blacklist | Любой `urlopen()` — даже с hardcoded https-константой. | 5 мест: `_lama_helpers.py:488`, `ts_frame_interpolation.py:183`, `tests/test_browser_smoke.py:40,218`, `tests/test_comfyui_live_api.py:30` | Per-line `# nosec B310` с обоснованием, что URL — hardcoded HTTPS-константа из исходника или localhost. |
| **B108** `hardcoded_tmp_directory` | `"/tmp"` в коде. У нас — только в test fixtures (parsed dicts с фейковыми путями). | 11 мест в `tests/test_downloader.py`, `tests/test_qwen_engine.py` | `B108` в `[tool.bandit] skips` глобально — production `/tmp` не использует. |

После фикса: `bandit -r . -ll -c pyproject.toml` → "No issues identified". В v9.13 47 строк изменений, 0 поломок тестов (409 passed / 10 skipped).

## Как bandit читает pyproject.toml

Bandit 1.7.5+ читает секцию `[tool.bandit]` автоматически, **если** запускать с `-c pyproject.toml` (или если pyproject лежит в cwd при `bandit -r .`). Comfy registry scanner — если на базе bandit — почти наверняка читает её аналогично, потому что это стандартное поведение plugin'а.

Формат:

```toml
[tool.bandit]
skips = ["B108", "B615"]
exclude_dirs = ["tests"]  # опционально
```

## Что НЕ помогает

- `.comfyignore` — он применяется только к финальному zip публикации. Scanner работает на full git checkout, до zip-стадии.
- `# nosec` без указания rule — отключает все правила на строке, низкая granularity, плохо для review.
- Удаление tests/tools из git tracked — не поможет если в production коде есть свои triggers (B615 у нас был именно в production).

## Pre-publish self-check

Перед каждым релизом полезно:

```bash
D:/AiApps/ComfyUI/comfyui/python/python.exe -m bandit -r . -ll -c pyproject.toml
```

Если output ≠ "No issues identified" в Medium+ — значит scanner Comfy registry, скорее всего, поставит Flagged. Проверь и/или добавь fix перед push.
