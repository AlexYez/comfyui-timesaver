---
name: reference_image_prompt_extraction
description: "Чтение позитивного промпта из метаданных картинки (TS Batch Source, режим Prompts from images in folder) — порядок источников и почему обход проводов идёт по белому списку"
metadata:
  type: reference
---

Чтение позитивного промпта из метаданных картинки (TS Batch Source, режим Prompts from images in folder) — порядок источников и почему обход проводов идёт по белому списку

Исходники темы: [ts_batch_source.py](../../nodes/utils/ts_batch_source.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Режим `Prompts from images in folder` у TS Batch Source (23.10.2026), код —
`nodes/utils/ts_batch_source.py` (`_image_prompt`, `_comfy_positive`).

**Порядок источников:** A1111 `parameters` (PNG) / EXIF UserComment (JPEG,
WebP; UTF-16 без объявленного порядка байт — первый байт 0 = BE) → граф
ComfyUI `prompt` (PNG или EXIF `prompt:{…}` в теге Model/Make) → NovelAI
`Description`. `.text` у PNG читается только когда в `info` пусто — текстовые
куски после IDAT.

**Граф:** корни — вход `positive` сэмплеров (сначала тот, чей `latent_image`
от `Empty*` — первый проход), иначе `conditioning` у `*Guider`. Энкодеры
негатива вычитаются, НО если остаётся пусто (негатив = ConditioningZeroOut
того же энкодера, граф Artius) — берутся все.

⚠️ **Провода — только по белому списку** (`_STRING_SOURCES`, `_PASSTHROUGH`,
switch с bool, StringConcatenate, TS_SuperPrompt, TS_StylePromptSelector).
Общий обход по полям `text`/`prompt` выдавал у `TextGenerate` ИНСТРУКЦИЮ для
LLM вместо промпта картинки (замерено на графах Moss). Неизвестное = `None`,
пустая строка — известный результат (пустой стиль в конкатенации).

**Замер на 1500 картинках `output/`:** промпт у 1369; остальные честно без
него (апскейлы, DLSS-тесты, music, TS_PromptLibrary и TextGenerate — текст
собирается во время прогона и в файл не попадает, если нет TS Image Prompt
Injector).

Связано: [reference_batch_semantics](reference_batch_semantics.md).
