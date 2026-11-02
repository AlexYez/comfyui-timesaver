---
name: engineering_index_language
description: "Промпты, языковые модели и Ideogram"
metadata:
  type: reference
---

# Промпты, языковые модели и Ideogram

Каждая строка — отдельная тема; начните с нужного hook.

- [ideogram4-caption-format](ideogram4-caption-format.md) — Verified Ideogram 4 structured-JSON caption format + Cyrillic/font gotchas, used by the TS_IdeogramDesigner node.
- [ideogram_designer_architecture](ideogram_designer_architecture.md) — Ideogram Designer: выходы, пресеты, редактор, preview и известные ограничения.
- [ideogram_designer_caption](ideogram_designer_caption.md) — Ideogram Designer: история редактора промпта, координаты JSON, палитры и raw desc.
- [reference_gemma_prompt_tuning](reference_gemma_prompt_tuning.md) — История трёх экспериментальных Gemma-пресетов; текущий подход обновляет существующие пункты.
- [reference_ideogram_inpaint](reference_ideogram_inpaint.md) — Как Ideogram 4 устроен в перерисовке — JSON-подпись, асимметричный CFG, сигмы по вырезу; и что ему не лечится.
- [reference_image_prompt_extraction](reference_image_prompt_extraction.md) — Чтение позитивного промпта из метаданных картинки (TS Batch Source, режим Prompts from images in folder) — порядок….
- [reference_litert_engine](reference_litert_engine.md) — Gemma/LiteRT: обновление на 0.19.0, сравнительный benchmark и проверенные ограничения речи, музыки и кадров; старые замеры датированы.
- [reference_llm_presets_refresh](reference_llm_presets_refresh.md) — All 15 existing presets refreshed, JSON capture fix, two-model tests and live verification.
- [reference_llm_prompt_comparison](reference_llm_prompt_comparison.md) — 31.10.2026 live comparison LFM3 VL/Omni, Gemma LiteRT E2B/E4B, Qwen3.5 2B/4B; Qwen EOS bug fixed.
- [reference_llm_stack](reference_llm_stack.md) — TS_Qwen3_VL_V3 и TS_SuperPrompt используют общий nodes/llm/_qwen_engine.py (с v9.5); это где живёт пайплайн Qwen, model….
- [reference_prompt_designer](reference_prompt_designer.md) — TS Prompt Designer — СНЯТА 14.08.2026, лежит в archive/. Заметка ради измеренных фактов о H3, Gemma и разворачивающихся….
- [reference_prompt_library](reference_prompt_library.md) — TS Prompt Library — библиотека промптов на данных (раздел/модель/collection.json), поля подстановки, Qwen Image 2.1….
- [reference_prompt_packs](reference_prompt_packs.md) — Текущий TXT-алгоритм Prompt Builder и отличие от старой семантической схемы.
- [prompt_builder_semantic_history](prompt_builder_semantic_history.md) — История отсутствующей семантической реализации, роли, профили и требования владельца.
- [reference_prompt_presets_2026_08](reference_prompt_presets_2026_08.md) — Пересборка пресетов TS Super Prompt 11.08.2026 — H3 (два), LTX, универсальный дефолт, чистка с откатом, кнопка HQ….
- [reference_qwen21_turbo](reference_qwen21_turbo.md) — TS Shifted Sigmas + TS LoRA Unmerged — своя замена нодам Viggle для турбо-LoRA Qwen Image 2.1; что замерено и где грабли.
- [reference_style_prompt_library](reference_style_prompt_library.md) — Контракт библиотеки стилей TS_StylePromptSelector — чистые модификаторы с хвостовой ", " и двуязычные поля (голый ключ….
- [reference_super_prompt_downloads](reference_super_prompt_downloads.md) — TS Super Prompt — согласие на скачивание модели и проверка пригодности transformers до загрузки.
- [reference_super_prompt_two_images](reference_super_prompt_two_images.md) — TS Super Prompt — две референсные картинки (первый/последний кадр), перетаскивание из Artius и системные промпты под….
- [reference_super_prompt_wire](reference_super_prompt_wire.md) — Вход `prompt` у TS Super Prompt и RT — отдельное гнездо (не поле-виджет), приоритет над полем, стандартная нода….
- [reference_voice_pipeline](reference_voice_pipeline.md) — Поведение Whisper-пайплайна в TS_SuperPrompt — hysteresis VAD, anti-hallucination фильтр, async record-button; всё….
- [ts-ideogram-designer-node](ts-ideogram-designer-node.md) — Ideogram Designer: архитектура, caption JSON, редактор и причины прежних ошибок.
