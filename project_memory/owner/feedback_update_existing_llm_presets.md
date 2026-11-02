---
name: feedback_update_existing_llm_presets
description: "Update existing LLM presets in place; do not add Gemma-prefixed copies"
metadata:
  type: feedback
---

Update existing LLM presets in place; do not add Gemma-prefixed copies

01.11.2026 владелец уточнил:
«Новые пресеты создавать не надо, ты должен обновить существующие, чтобы не
добавлялось новых пресетов. И доделай все остальные пресеты в списке».

При настройке Gemma/Qwen обновлять существующие15 ключей в
nodes/qwen_3_vl_presets.json. Сохранять названия и порядок. Новые варианты
с префиксом модели не создавать без отдельной просьбы. Три ранее добавленных
Gemma - … удалены из меню; скрытые алиасы поддерживают уже сохранённые значения.
Инструкции подбирать на небольших E2B/E4B; не вставлять целые исследования или
Skills. Короткость сама по себе не равна качеству: нужны реальные проверки.
