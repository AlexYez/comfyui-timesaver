---
name: project_builtin_updater
description: "Отложенная идея — вкладка Updates в настройках пака с уведомлением о новой версии; разобранные варианты, риски и готовые точки опоры"
metadata:
  type: project
---

Отложенная идея — вкладка Updates в настройках пака с уведомлением о новой версии; разобранные варианты, риски и готовые точки опоры

Исходники темы: [ts-group-badge.js](../../js/utils/group_bypasser/ts-group-badge.js), [ts_downloader.py](../../nodes/files/ts_downloader.py), [_shared.py](../../nodes/_shared.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

# Встроенный апдейтер: идея на паузе (15.09.2026)

Владелец предложил вкладку **Updates** в настройках пака: показывать, что вышла
новая версия, и по возможности обновляться оттуда же. Разбор сделан, решение —
**отложить**, вернуться позже. Ниже всё, что выяснено, чтобы не выяснять снова.

## Точки опоры, которые уже есть

- **Свой раздел в настройках ComfyUI существует.**
  `js/utils/group_bypasser/ts-group-badge.js` регистрирует настройку с
  `category: ["TS Timesaver", "Canvas", "Group bypass badge"]`. Вкладка Updates —
  это просто `["TS Timesaver", "Updates", ...]`, отдельной инфраструктуры не надо.
- **Пак умеет читать свою версию:** `_resolve_pack_version()` в
  `nodes/files/ts_downloader.py` (регэкспом из `pyproject.toml`, без tomllib —
  работает на 3.10). При работе над апдейтером вынести в `nodes/_shared.py`.
- Тост показывается через `app.extensionManager?.toast?.add(...)` — уже
  используется в `ts-files-downloader.js` и `ts-tidy-layout.js`.

## ⚠️ Источником «последней версии» реестр быть НЕ МОЖЕТ

`api.comfy.org/nodes/comfyui-timesaver` отдаёт `latest_version: 12.1.0`, потому
что всё новее в статусе Flagged (см. [reference_comfy_scanner_rules](reference_comfy_scanner_rules.md)). То есть
апдейтер на реестре честно сообщал бы пользователям, что они опережают
«последнюю» версию на девять релизов. Брать `pyproject.toml` из master на
GitHub (raw), а «что нового» — из `CHANGELOG.md` там же.

⚠️ Побочно: `raw.githubusercontent.com` уже даёт сканеру находку
`contains_blacklisted_url` (в `_studio_packs.py`). Второй такой URL добавит ещё
одну — на сам вердикт не влияет, но в отчёте будет.

## Три варианта и чем они отличаются

| Вариант | Что делает | Цена |
|---|---|---|
| **Только уведомление** | вкладка с текущей/доступной версией и выдержкой из CHANGELOG + один тост при старте с выключателем; кнопка «как обновить» показывает `git pull` и ссылку на Manager | для сканера обычный GET, ничего не пишется на диск |
| **Через Manager** | кнопка дёргает API ComfyUI Manager, код качает и раскладывает ОН | работает только там, где Manager есть; его API меняется |
| **Свой апдейтер** | `git pull` или zip с GitHub поверх `custom_nodes/comfyui-timesaver` | ⚠️ самозапись кода — тот самый класс, за который реестр банит (`FETCH_AND_RUN_DOWNLOADED_EXECUTABLE`); плюс Windows не даёт перезаписать уже загруженные `.pyd`/`.dll`, git может отсутствовать, локальные правки дают конфликт |

⚠️ Отдельная ирония, которую стоит помнить: в 12.10.0 мы СПЕЦИАЛЬНО закрыли
`custom_nodes` как цель записи для загрузчика ([reference_security_boundaries](reference_security_boundaries.md)).
Свой апдейтер открывает ту же дверь обратно.

Связано: [reference_comfy_scanner_rules](reference_comfy_scanner_rules.md), [reference_security_boundaries](reference_security_boundaries.md),
[reference_theme_system](reference_theme_system.md).
