---
name: feedback_ci_disabled
description: "CI-workflow выключен по просьбе пользователя (лимиты Actions); publish в реестр оставлен активным — не включать CI без явной просьбы"
metadata:
  type: feedback
---

CI-workflow выключен по просьбе пользователя (лимиты Actions); publish в реестр оставлен активным — не включать CI без явной просьбы

Состояние удалённых workflow повторно проверено root-агентом 2026-11-02 командой `gh workflow list --all`: CI — disabled_manually, macOS smoke и публикация — active. Локальные YAML определяют триггеры, но сами по себе не доказывают состояние enabled/disabled на GitHub.

**CI выключен намеренно.** 2026-07-30 пользователь попросил отключить CI — у него
закончились лимиты GitHub Actions. Состояние:

| Workflow | Состояние |
|---|---|
| `CI` (`.github/workflows/ci.yml`, id 271310816) | **disabled_manually** |
| `Publish to Comfy registry` (`publish_action.yml`, id 156526583) | **active** — оставлен по явному подтверждению |
| `Dependency Graph` (id 267000455) | active; через API не гасится (HTTP 422), это встроенная фича GitHub, тумблер в *Settings → Code security* |

**Why:** публикация в Comfy registry — единственный путь релиза к пользователям, её
выключение дало бы тихий провал (бампнули версию, а до реестра не доехало). CI же на
каждый пуш крутит три матричных джоба и smoke-import — именно он и ест минуты.

**How to apply:**
- **Не включать CI обратно** без явной просьбы («включи CI»). Команда: `gh workflow enable 271310816`.
- Пока CI выключен, **автоматической проверки на пуш нет** → локально проверяются затронутые ноды и их общие модули. Полный `pytest tests` — только по прямой команде владельца, согласно более позднему [feedback_tests_on_demand](feedback_tests_on_demand.md).
- Publish не трогать вообще, отдельно см. [feedback_publish_action_donttouch](feedback_publish_action_donttouch.md).

**Заметка на будущее:** репозиторий публичный, а публичным репозиториям Actions даёт
безлимитные минуты на стандартных раннерах — упёршийся лимит, скорее всего, общий по
аккаунту (приватные репо / Codespaces), а не от этого проекта. Проверяется в
*Settings → Billing*.

## macOS-прогон по кнопке (2026-08-08)

Владелец добавил лимитов Actions и попросил проверить пак на macOS. Заведён
`.github/workflows/macos_smoke.yml`: матрица `macos-14` + `macos-15` (Apple
Silicon), клон ComfyUI, torch с обычного PyPI (⚠️ индекс `/whl/cpu` не содержит
колёс под macOS arm64), импорт пака его же загрузчиком, проверка бинарника
ffmpeg. Единственный триггер — `workflow_dispatch`, поэтому выключённый `CI`
трогать не пришлось и автозапусков не появилось.

⚠️ **workflow_dispatch виден только если файл есть в ВЕТКЕ ПО УМОЛЧАНИЮ.**
`gh workflow run ... --ref image-studio` отвечал `HTTP 404: workflow not found
on the default branch`, пока файл лежал только в рабочей ветке. Пришлось
отдельным коммитом положить его в `master` (`0b6949c`) — после этого запуск на
любой ветке работает. Публикацию это не будит: `publish_action.yml` срабатывает
только на изменение `pyproject.toml`.

Чего прогон не может: Metal (у раннера нет GPU; `mps available` печатается, но
не требуется), скачивание моделей и собственный набор тестов — `tests/` и
`tools/` не в гите и до раннера не доезжают.
