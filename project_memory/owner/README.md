---
name: owner_index
description: "Дословные исправления владельца и причины правил. Более поздняя датированная команда выше старого разрешения; текущая задача выше общих предпочтений."
metadata:
  type: reference
---

Дословные исправления владельца и причины правил. Более поздняя датированная команда выше старого разрешения; текущая задача выше общих предпочтений.

# Требования владельца

[Общий индекс](../MEMORY.md)

- [Уборка папок](folder-cleanup.md) — сначала инвентаризация и подтверждение; только обратимый архив, защита текущих файлов и отдельные уточнения по крупным остаткам.

- [private-prompt-packs](private-prompt-packs.md) — Не публиковать пользовательские nsfw*-паки; отсутствие ignore-правила не отменяет требования владельца.

- [comfyui-lifecycle](comfyui-lifecycle.md) — Только прямая команда владельца разрешает остановку, запуск или перезапуск ComfyUI.
- [comfyui-windows](comfyui-windows.md) — Использовать одно окно сервера и существующую страницу, не засорять рабочее место тестами.
- [feedback_ci_disabled](feedback_ci_disabled.md) — CI-workflow выключен по просьбе пользователя (лимиты Actions); publish в реестр оставлен активным — не включать CI без явной просьбы
- [feedback_commit_then_sync](feedback_commit_then_sync.md) — Коммит и пуш только по прямой команде; до неё — проверка и синхронизация установленной копии.
- [feedback_compatibility](feedback_compatibility.md) — Не менять node_id, входы/выходы, defaults, category, widget IDs без явного запроса пользователя
- [feedback_full_browser_smoke](feedback_full_browser_smoke.md) — После рестарта ComfyUI обязателен Playwright smoke (LiteGraph.createNode + read console errors). Скриншоты в smoke не делаем — они только для README через tools/screenshot_nodes.py
- [feedback_god_files_ok](feedback_god_files_ok.md) — Maintainer rejects "large file = debt" framing — only multi-class files count as structural debt
- [feedback_merge_after_each_stage](feedback_merge_after_each_stage.md) — После каждого мигрированного батча/этапа сразу делать merge в master, чтобы ComfyUI (запущен из main repo) видел обновления
- [feedback_naming](feedback_naming.md) — Префикс TS_ для классов и node_id, ts_ для имён файлов; новые имена не должны конфликтовать со старыми
- [feedback_no_coauthor_footer](feedback_no_coauthor_footer.md) — не добавлять Co-Authored-By: Claude в коммиты — GitHub отображает второго автора с иконкой Claude
- [feedback_no_subgraphs](feedback_no_subgraphs.md) — Владелец не пользуется сабграфами и считает их тупиковой веткой ComfyUI — поломки только внутри сабграфов не в приоритете
- [feedback_no_version_bumps](feedback_no_version_bumps.md) — Не поднимать версию в pyproject.toml без ведома пользователя — версию назначает он
- [feedback_one_node_one_file](feedback_one_node_one_file.md) — Каждая публичная ComfyUI-нода = один основной .py файл; не дробить на schema/execute/types
- [feedback_preserve_preset_purpose](feedback_preserve_preset_purpose.md) — Do not replace a user's preset purpose with unrequested content restrictions
- [feedback_publish_action_donttouch](feedback_publish_action_donttouch.md) — Не менять работающий workflow публикации без задачи владельца; техническая запись про @main историческая.
- [feedback_quiet_startup](feedback_quiet_startup.md) — Пак не должен засорять консоль ComfyUI при загрузке — одна строка, если всё хорошо
- [feedback_runtime_only_locations](feedback_runtime_only_locations.md) — История разделения dev и runtime; требование хранить память вне Git отменено консолидацией 2026-11-02.
- [feedback_terse_summaries](feedback_terse_summaries.md) — Пользователь просит короткие сводки и предложения «что дальше», без длинных табличных отчётов и развёрнутых планов
- [feedback_tests_on_demand](feedback_tests_on_demand.md) — полный тестовый набор гонять ТОЛЬКО по прямой команде пользователя — исключений нет, релиз тоже не повод; по умолчанию тестируются затронутые ноды и общие модули, которые в них …
- [feedback_update_existing_llm_presets](feedback_update_existing_llm_presets.md) — Update existing LLM presets in place; do not add Gemma-prefixed copies
- [feedback_use_comfyui_skill](feedback_use_comfyui_skill.md) — Перед нетривиальной работой над ComfyUI нодой подключать skill /comfyui-custom-nodes (frontend, basics, lifecycle, etc.)
- [feedback_verify_what_the_user_sees](feedback_verify_what_the_user_sees.md) — Замеры DOM — не доказательство того, что человек что-то видит. Не говорить «работает», когда видимый результат проверить нечем
- [language](language.md) — Объяснения и отчёты владельцу писать по-русски, технические идентификаторы сохранять.
