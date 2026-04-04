# ComfyUI Timesaver Nodes

[English](README.md) | [Русский](README.ru.md)

Полное и дружелюбное описание **всех нод текущего пака**. Каждая нода оформлена отдельной раскрывающейся карточкой, чтобы README было удобно читать даже новичку.

Репозиторий: https://github.com/AlexYez/comfyui-timesaver

## Установка

1. Поместите папку в `ComfyUI/custom_nodes/comfyui-timesaver`.
2. При необходимости установите зависимости из `requirements.txt`.
3. Перезапустите ComfyUI.

## ????????? ???????

```text
comfyui-timesaver/
?? nodes/                     # ??? ???? ? node-related ???????
?  ?? *.py                    # ?????????? ???
?  ?? luts/                   # LUT-?????
?  ?? prompts/                # ????? Prompt Builder
?  ?? styles/                 # ?????? style-????????
?  ?? video_depth_anything/   # ????? depth-??????
?  ?? qwen_3_vl_presets.json  # ??????? ??? Qwen-????
?? js/                        # Frontend-??????????
?? doc/                       # ?????????? ????????????
?? requirements.txt
?? pyproject.toml
?? __init__.py                # ????????? + startup audit
```


## Что есть в этом README

- Задокументировано нод: **52**
- Для каждой ноды есть раскрывающаяся карточка `<details>`
- Для каждой ноды добавлен плейсхолдер скриншота
- Описание написано простым языком, без перегруза терминами

## Каталог нод

| Node ID | Для чего нужна | Категория | Типы выходов |
| --- | --- | --- | --- |
| `TS_Qwen3_VL_V3` | Основная мультимодальная нода Qwen (текст + изображение/видео) с пресетами, управлением precision и offline-режимом. | `TS/LLM` | `STRING` |
| `TSWhisper` | Нода Whisper для транскрибации и перевода аудио с выводом SRT и обычного текста. | `TS/Audio` | `STRING` |
| `TS_SileroTTS` | Русская TTS-нода на базе Silero с чанкингом и выходом AUDIO. | `TS/audio` | `AUDIO` |
| `TS_MusicStems` | Разделяет музыку на стемы (vocals, bass, drums, others, instrumental). | `TS/Audio` | `AUDIO` |
| `TS_PromptBuilder` | Собирает структурированные промпты из JSON-конфига и seed для воспроизводимых вариаций. | `TS/Prompt` | `STRING` |
| `TS_BatchPromptLoader` | Читает многострочные промпты и выдаёт один промпт по индексу/шагу. | `utils/text` | `STRING` |
| `TS_StylePromptSelector` | Загружает текст style-промпта из библиотеки по ID или имени. | `TS/Prompt` | `STRING` |
| `TS_ImageResize` | Гибкая нода resize: точный размер, масштаб по стороне, scale factor или целевые мегапиксели. | `image` | `IMAGE` |
| `TS_QwenSafeResize` | Безопасный resize-пресет под ограничения препроцессинга Qwen. | `image/resize` | `IMAGE` |
| `TS_WAN_SafeResize` | Safe-resize для WAN-пайплайнов с размером, дружелюбным к моделям. | `image/resize` | `IMAGE` |
| `TS_QwenCanvas` | Создаёт canvas с разрешением, удобным для Qwen, и при необходимости размещает image/mask. | `TS Qwen` | `IMAGE` |
| `TS_ResolutionSelector` | Выбирает целевое разрешение по aspect-пресетам или custom ratio и может вернуть подготовленный canvas. | `TS/Resolution` | `IMAGE` |
| `TS_Color_Grade` | Быстрый первичный color grading: hue, temperature, saturation, contrast, gamma и tone-контроли. | `TS/Color` | `IMAGE` |
| `TS_Film_Emulation` | Нода для киношной стилизации: пресеты, LUT, warmth, fade и контроль зерна. | `Image/Color` | `IMAGE` |
| `TS_FilmGrain` | Добавляет управляемое плёночное зерно с настройкой размера, интенсивности, цвета и движения. | `Image Adjustments/Grain` | `IMAGE` |
| `TS_Color_Match` | Переносит цветовое настроение с референса на целевое изображение, сохраняя структуру сцены. | `TS/Color` | `IMAGE` |
| `TS_Keyer` | ???????????????? color-difference keyer ??? green/blue/red screen ? ?????? ?????? ? ?????????? despill. | `TS/image` | `IMAGE, MASK, IMAGE` |
| `TS_Despill` | ????????? ???? ?????????? spill ? ??????????? classic, balanced, adaptive ? hue_preserve. | `TS/image` | `IMAGE, MASK, IMAGE` |
| `TS_BGRM_BiRefNet` | AI-удаление фона через BiRefNet: быстро, чисто и удобно для прозрачных композиций. | `Timesaver/Image Tools` | `IMAGE` |
| `TSCropToMask` | Кадрирует область вокруг маски, чтобы ускорить локальные правки и сэкономить память. | `image/processing` | `IMAGE` |
| `TSRestoreFromCrop` | Возвращает обработанный crop обратно в исходный кадр с опциональным смешиванием. | `image/processing` | `IMAGE` |
| `TS_ImageBatchToImageList` | Преобразует batched IMAGE tensor в покадровый list-поток. | `TS/Image Tools` | `IMAGE` |
| `TS_ImageListToImageBatch` | Собирает list-поток изображений обратно в batched IMAGE tensor. | `TS/Image Tools` | `IMAGE` |
| `TS_ImageBatchCut` | Обрезает кадры с начала и/или конца image batch. | `TS/Image Tools` | `IMAGE` |
| `TS_GetImageMegapixels` | Возвращает значение мегапикселей для быстрой оценки качества и производительности. | `TS/Image Tools` | `FLOAT` |
| `TS_GetImageSizeSide` | Возвращает размер выбранной стороны изображения для логики графа и автоконфигурации. | `TS/Image Tools` | `INT` |
| `TS_ImagePromptInjector` | Встраивает текст промпта в image-flow, чтобы контекст шёл вместе с веткой изображения. | `TS/Image Tools` | `IMAGE` |
| `TS_ImageTileSplitter` | Делит изображение на перекрывающиеся тайлы для тяжёлой обработки в высоком качестве. | `TS/Image Tools` | `IMAGE` |
| `TS_ImageTileMerger` | Склеивает тайлы обратно в целое изображение по tile metadata и blending. | `TS/Image Tools` | `IMAGE` |
| `TSAutoTileSize` | Автоматически рассчитывает размер тайлов (width/height) под вашу целевую сетку. | `utils/Tile Size` | `INT` |
| `TS Cube to Equirectangular` | Конвертирует шесть граней куба в одну equirectangular 360-панораму. | `Tools/TS_Image` | `IMAGE` |
| `TS Equirectangular to Cube` | Конвертирует 360-панораму в шесть граней куба для редактирования и проекций. | `Tools/TS_Image` | `IMAGE` |
| `TS_VideoDepthNode` | Строит depth maps по последовательности кадров для композитинга, relighting и depth-эффектов. | `Tools/Video` | `IMAGE` |
| `TS_Video_Upscale_With_Model` | Апскейлит последовательность кадров с выбранной моделью апскейла и memory-стратегиями. | `video` | `IMAGE` |
| `TS_RTX_Upscaler` | Нода NVIDIA RTX Upscaler для быстрого и качественного апскейла на поддерживаемых системах. | `TS/Upscaling` | `IMAGE` |
| `TS_DeflickerNode` | Снижает временное мерцание яркости и цвета в видеопоследовательностях. | `Video PostProcessing` | `IMAGE` |
| `TS_Free_Video_Memory` | Pass-through нода, которая агрессивно освобождает RAM/VRAM между тяжёлыми видео-шагами. | `video` | `IMAGE` |
| `TS_LTX_FirstLastFrame` | Добавляет guidance первого и последнего кадра в latent-пайплайн (полезно для LTX video control). | `conditioning/video_models` | `LATENT` |
| `TS_Animation_Preview` | Создаёт быстрый превью-ролик из кадров с опциональным объединением аудио. | `TS/Interface Tools` | `-` |
| `TS_FileBrowser` | Встроенный media picker: загружает image/video/audio/mask с диска прямо в ваш граф. | `TS/Input` | `IMAGE` |
| `TS_FilePathLoader` | Возвращает путь к файлу и имя файла по индексу из списка папки. | `file_utils` | `STRING` |
| `TS Files Downloader` | Массово скачивает модели и ассеты: resume, mirrors, proxy и опциональная распаковка. | `Tools/TS_IO` | `-` |
| `TS Youtube Chapters` | Конвертирует EDL-тайминги в готовые timestamp-главы для YouTube. | `Tools/TS_Video` | `STRING` |
| `TS_ModelScanner` | Сканирует файлы моделей и возвращает читаемую сводку структуры и метаданных. | `utils/model_analysis` | `STRING` |
| `TS_ModelConverter` | Конвертер модели в один клик для сценариев смены precision. | `conversion` | `MODEL` |
| `TS_ModelConverterAdvanced` | Расширенный конвертер моделей с детальным контролем формата, пресета и результата. | `Model Conversion` | `STRING` |
| `TS_ModelConverterAdvancedDirect` | Расширенный конвертер, работающий напрямую от подключённого входа MODEL. | `TS/Model Conversion` | `STRING` |
| `TS_CPULoraMerger` | ?????????? ?? ??????? LoRA ? ??????? ?????? ?? CPU ? ????????? ????? safetensors-????. | `TS/Model Tools` | `STRING, STRING` |
| `TS_FloatSlider` | Простой UI float slider для аккуратного управления параметрами графа. | `TS Tools/Sliders` | `FLOAT` |
| `TS_Int_Slider` | Простой UI integer slider для детерминированных целочисленных параметров. | `TS Tools/Sliders` | `INT` |
| `TS_Smart_Switch` | Переключает между двумя входами по режиму и помогает держать граф компактным. | `TS Tools/Logic` | `*` |
| `TS_Math_Int` | Нода целочисленной математики для счётчиков, смещений и простой логики графа. | `TS/Math` | `INT` |

## Подробные карточки нод

### Все ноды

<details>
<summary><strong>TS_Qwen3_VL_V3</strong> - Основная мультимодальная нода Qwen (текст + изображение/видео) с пресетами, управлением precision и offline-режимом.</summary>

![Плейсхолдер скриншота для TS_Qwen3_VL_V3](docs/img/placeholders/ts-qwen3-vl-v3.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Основная мультимодальная нода Qwen (текст + изображение/видео) с пресетами, управлением precision и offline-режимом.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `model_name`, `custom_model_id`, `hf_token`, `system_preset`, `prompt`, `seed`, `max_new_tokens`, `precision`, `attention_mode`, `offline_mode`, `unload_after_generation`, `enable`, `max_image_size`, `video_max_frames`
- Опциональные: `image`, `video`, `custom_system_prompt`

**Выходы**
- `STRING`

**Техническая информация**
- Internal id: `TS_Qwen3_VL_V3`
- Class: `TS_Qwen3_VL_V3`
- File: `nodes/ts_qwen3_vl_v3_node.py`
- Category: `TS/LLM`
- Function: `process`
- Примечание по зависимостям: Использует `transformers` и опционально acceleration libraries.

</details>

<details>
<summary><strong>TSWhisper</strong> - Нода Whisper для транскрибации и перевода аудио с выводом SRT и обычного текста.</summary>

![Плейсхолдер скриншота для TSWhisper](docs/img/placeholders/tswhisper.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Нода Whisper для транскрибации и перевода аудио с выводом SRT и обычного текста.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `audio`, `model`, `output_filename_prefix`, `task`, `source_language`, `timestamps`, `save_srt_file`, `precision`
- Опциональные: `output_dir`

**Выходы**
- `STRING`

**Техническая информация**
- Internal id: `TSWhisper`
- Class: `TSWhisper`
- File: `nodes/ts_whisper_node.py`
- Category: `TS/Audio`
- Function: `generate_srt_and_text`
- Примечание по зависимостям: Использует `transformers` and `torchaudio`.

</details>

<details>
<summary><strong>TS_SileroTTS</strong> - Русская TTS-нода на базе Silero с чанкингом и выходом AUDIO.</summary>

![Плейсхолдер скриншота для TS_SileroTTS](docs/img/placeholders/ts-silerotts.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Русская TTS-нода на базе Silero с чанкингом и выходом AUDIO.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `text`, `input_format`, `speaker`, `run_device`, `enable_chunking`, `max_chunk_chars`, `chunk_pause_ms`, `put_accent`, `put_yo`, `put_stress_homo`, `put_yo_homo`
- Опциональные: *(none)*

**Выходы**
- `AUDIO`

**Техническая информация**
- Internal id: `TS_SileroTTS`
- Class: `TS_SileroTTS`
- File: `nodes/ts_silero_tts_node.py`
- Category: `TS/audio`
- Function: `execute`

</details>

<details>
<summary><strong>TS_MusicStems</strong> - Разделяет музыку на стемы (vocals, bass, drums, others, instrumental).</summary>

![Плейсхолдер скриншота для TS_MusicStems](docs/img/placeholders/ts-musicstems.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Разделяет музыку на стемы (vocals, bass, drums, others, instrumental).

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `audio`, `model_name`, `device`, `shifts`, `overlap`, `jobs`
- Опциональные: *(none)*

**Выходы**
- `AUDIO`

**Техническая информация**
- Internal id: `TS_MusicStems`
- Class: `TS_MusicStems`
- File: `nodes/ts_music_stems_node.py`
- Category: `TS/Audio`
- Function: `process_stems`
- Примечание по зависимостям: Requires `demucs`.

</details>

<details>
<summary><strong>TS_PromptBuilder</strong> - Собирает структурированные промпты из JSON-конфига и seed для воспроизводимых вариаций.</summary>

![Плейсхолдер скриншота для TS_PromptBuilder](docs/img/placeholders/ts-promptbuilder.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Собирает структурированные промпты из JSON-конфига и seed для воспроизводимых вариаций.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `seed`, `config_json`
- Опциональные: *(none)*

**Выходы**
- `STRING`

**Техническая информация**
- Internal id: `TS_PromptBuilder`
- Class: `TS_PromptBuilder`
- File: `nodes/ts_prompt_builder_node.py`
- Category: `TS/Prompt`
- Function: `build_prompt`

</details>

<details>
<summary><strong>TS_BatchPromptLoader</strong> - Читает многострочные промпты и выдаёт один промпт по индексу/шагу.</summary>

![Плейсхолдер скриншота для TS_BatchPromptLoader](docs/img/placeholders/ts-batchpromptloader.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Читает многострочные промпты и выдаёт один промпт по индексу/шагу.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `text`
- Опциональные: *(none)*

**Выходы**
- `STRING`

**Техническая информация**
- Internal id: `TS_BatchPromptLoader`
- Class: `TS_BatchPromptLoader`
- File: `nodes/ts_text_tools_node.py`
- Category: `utils/text`
- Function: `process_prompts`

</details>

<details>
<summary><strong>TS_StylePromptSelector</strong> - Загружает текст style-промпта из библиотеки по ID или имени.</summary>

![Плейсхолдер скриншота для TS_StylePromptSelector](docs/img/placeholders/ts-stylepromptselector.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Загружает текст style-промпта из библиотеки по ID или имени.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `style_id`
- Опциональные: *(none)*

**Выходы**
- `STRING`

**Техническая информация**
- Internal id: `TS_StylePromptSelector`
- Class: `TS_StylePromptSelector`
- File: `nodes/ts_style_prompt_node.py`
- Category: `TS/Prompt`
- Function: `get_prompt`

</details>

<details>
<summary><strong>TS_ImageResize</strong> - Гибкая нода resize: точный размер, масштаб по стороне, scale factor или целевые мегапиксели.</summary>

![Плейсхолдер скриншота для TS_ImageResize](docs/img/placeholders/ts-imageresize.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Гибкая нода resize: точный размер, масштаб по стороне, scale factor или целевые мегапиксели.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `pixels`, `target_width`, `target_height`, `smaller_side`, `larger_side`, `scale_factor`, `keep_proportion`, `upscale_method`, `divisible_by`, `megapixels`, `dont_enlarge`
- Опциональные: `mask`

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_ImageResize`
- Class: `TS_ImageResize`
- File: `nodes/ts_image_resize_node.py`
- Category: `image`
- Function: `resize`

</details>

<details>
<summary><strong>TS_QwenSafeResize</strong> - Безопасный resize-пресет под ограничения препроцессинга Qwen.</summary>

![Плейсхолдер скриншота для TS_QwenSafeResize](docs/img/placeholders/ts-qwensaferesize.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Безопасный resize-пресет под ограничения препроцессинга Qwen.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_QwenSafeResize`
- Class: `TS_QwenSafeResize`
- File: `nodes/ts_image_resize_node.py`
- Category: `image/resize`
- Function: `safe_resize`

</details>

<details>
<summary><strong>TS_WAN_SafeResize</strong> - Safe-resize для WAN-пайплайнов с размером, дружелюбным к моделям.</summary>

![Плейсхолдер скриншота для TS_WAN_SafeResize](docs/img/placeholders/ts-wan-saferesize.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Safe-resize для WAN-пайплайнов с размером, дружелюбным к моделям.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`, `quality`
- Опциональные: `interconnection_in`

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_WAN_SafeResize`
- Class: `TS_WAN_SafeResize`
- File: `nodes/ts_image_resize_node.py`
- Category: `image/resize`
- Function: `safe_resize`

</details>

<details>
<summary><strong>TS_QwenCanvas</strong> - Создаёт canvas с разрешением, удобным для Qwen, и при необходимости размещает image/mask.</summary>

![Плейсхолдер скриншота для TS_QwenCanvas](docs/img/placeholders/ts-qwencanvas.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Создаёт canvas с разрешением, удобным для Qwen, и при необходимости размещает image/mask.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `resolution`
- Опциональные: `image`, `mask`

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_QwenCanvas`
- Class: `TS_QwenCanvas`
- File: `nodes/ts_image_resize_node.py`
- Category: `TS Qwen`
- Function: `make_canvas`

</details>

<details>
<summary><strong>TS_ResolutionSelector</strong> - Выбирает целевое разрешение по aspect-пресетам или custom ratio и может вернуть подготовленный canvas.</summary>

![Плейсхолдер скриншота для TS_ResolutionSelector](docs/img/placeholders/ts-resolutionselector.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Выбирает целевое разрешение по aspect-пресетам или custom ratio и может вернуть подготовленный canvas.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `aspect_ratio`, `resolution`, `custom_ratio`, `original_aspect`
- Опциональные: `image`

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_ResolutionSelector`
- Class: `TS_ResolutionSelector`
- File: `nodes/ts_resolution_selector.py`
- Category: `TS/Resolution`
- Function: `select_resolution`

</details>

<details>
<summary><strong>TS_Color_Grade</strong> - Быстрый первичный color grading: hue, temperature, saturation, contrast, gamma и tone-контроли.</summary>

![Плейсхолдер скриншота для TS_Color_Grade](docs/img/placeholders/ts-color-grade.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Быстрый первичный color grading: hue, temperature, saturation, contrast, gamma и tone-контроли.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`, `hue`, `temperature`, `saturation`, `contrast`, `gain`, `lift`, `gamma`, `brightness`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_Color_Grade`
- Class: `TS_Color_Grade`
- File: `nodes/ts_color_node.py`
- Category: `TS/Color`
- Function: `process`

</details>

<details>
<summary><strong>TS_Film_Emulation</strong> - Нода для киношной стилизации: пресеты, LUT, warmth, fade и контроль зерна.</summary>

![Плейсхолдер скриншота для TS_Film_Emulation](docs/img/placeholders/ts-film-emulation.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Нода для киношной стилизации: пресеты, LUT, warmth, fade и контроль зерна.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`, `enable`, `film_preset`, `lut_choice`, `lut_strength`, `gamma_correction`, `film_strength`, `contrast_curve`, `warmth`, `grain_intensity`, `grain_size`, `fade`, `shadow_saturation`, `highlight_saturation`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_Film_Emulation`
- Class: `TS_Film_Emulation`
- File: `nodes/ts_color_node.py`
- Category: `Image/Color`
- Function: `process`

</details>

<details>
<summary><strong>TS_FilmGrain</strong> - Добавляет управляемое плёночное зерно с настройкой размера, интенсивности, цвета и движения.</summary>

![Плейсхолдер скриншота для TS_FilmGrain](docs/img/placeholders/ts-filmgrain.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Добавляет управляемое плёночное зерно с настройкой размера, интенсивности, цвета и движения.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `images`, `force_gpu`, `grain_size`, `grain_intensity`, `grain_speed`, `grain_softness`, `color_grain_strength`, `mid_tone_grain_bias`, `seed`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_FilmGrain`
- Class: `TS_FilmGrain`
- File: `nodes/ts_film_grain_node.py`
- Category: `Image Adjustments/Grain`
- Function: `apply_grain`

</details>

<details>
<summary><strong>TS_Color_Match</strong> - Переносит цветовое настроение с референса на целевое изображение, сохраняя структуру сцены.</summary>

![Плейсхолдер скриншота для TS_Color_Match](docs/img/placeholders/ts-color-match.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Переносит цветовое настроение с референса на целевое изображение, сохраняя структуру сцены.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `reference`, `target`, `mode`, `device`, `strength`, `enable`, `match_mask`, `mask_size`, `compute_max_side`, `mkl_sample_points`, `sinkhorn_max_points`, `reuse_reference`, `chunk_size`, `logging`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_Color_Match`
- Class: `TS_Color_Match`
- File: `nodes/ts_color_match_node.py`
- Category: `TS/Color`
- Function: `process`

</details>

<details>
<summary><strong>TS_Keyer</strong> - ???????????????? color-difference keyer ??? green/blue/red screen ? ?????? ?????? ? ?????????? despill.</summary>

![Плейсхолдер скриншота для TS_Keyer](docs/img/placeholders/ts-keyer.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
???????????????? color-difference keyer ??? green/blue/red screen ? ?????? ?????? ? ?????????? despill.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`, `enable`, `key_color`, `key_channel`, `screen_balance`, `key_strength`, `black_point`, `white_point`, `matte_gamma`, `matte_preblur`, `edge_softness`, `despill_strength`, `despill_edge_only`, `despill_compensate`, `invert_alpha`
- Опциональные: *(none)*

**Выходы**
- `IMAGE, MASK, IMAGE`

**Техническая информация**
- Internal id: `TS_Keyer`
- Class: `TS_Keyer`
- File: `nodes/ts_keyer_node.py`
- Category: `TS/image`
- Function: `execute`

</details>

<details>
<summary><strong>TS_Despill</strong> - ????????? ???? ?????????? spill ? ??????????? classic, balanced, adaptive ? hue_preserve.</summary>

![Плейсхолдер скриншота для TS_Despill](docs/img/placeholders/ts-despill.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
????????? ???? ?????????? spill ? ??????????? classic, balanced, adaptive ? hue_preserve.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`, `enable`, `screen_color`, `algorithm`, `strength`, `spill_threshold`, `spill_softness`, `compensation`, `preserve_luma`, `use_input_alpha_for_edges`, `edge_boost`, `edge_blur`, `skin_protection`, `saturation_restore`, `invert_spill_mask`
- Опциональные: `spill_mask`

**Выходы**
- `IMAGE, MASK, IMAGE`

**Техническая информация**
- Internal id: `TS_Despill`
- Class: `TS_Despill`
- File: `nodes/ts_keyer_node.py`
- Category: `TS/image`
- Function: `execute`

</details>

<details>
<summary><strong>TS_BGRM_BiRefNet</strong> - AI-удаление фона через BiRefNet: быстро, чисто и удобно для прозрачных композиций.</summary>

![Плейсхолдер скриншота для TS_BGRM_BiRefNet](docs/img/placeholders/ts-bgrm-birefnet.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
AI-удаление фона через BiRefNet: быстро, чисто и удобно для прозрачных композиций.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`, `enable`, `model`
- Опциональные: `use_custom_resolution`, `process_resolution`, `mask_blur`, `mask_offset`, `invert_output`, `refine_foreground`, `background`, `background_color`

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_BGRM_BiRefNet`
- Class: `TS_BGRM_BiRefNet`
- File: `nodes/ts_bgrm_node.py`
- Category: `Timesaver/Image Tools`
- Function: `process_image`
- Примечание по зависимостям: Requires BiRefNet model files (auto-download when available).

</details>

<details>
<summary><strong>TSCropToMask</strong> - Кадрирует область вокруг маски, чтобы ускорить локальные правки и сэкономить память.</summary>

![Плейсхолдер скриншота для TSCropToMask](docs/img/placeholders/tscroptomask.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Кадрирует область вокруг маски, чтобы ускорить локальные правки и сэкономить память.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `images`, `mask`, `padding`, `divide_by`, `max_resolution`, `fixed_mask_frame_index`, `interpolation_window_size`, `force_gpu`, `fixed_crop_size`, `fixed_width`, `fixed_height`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TSCropToMask`
- Class: `TSCropToMask`
- File: `nodes/ts_crop_to_mask_node.py`
- Category: `image/processing`
- Function: `crop`

</details>

<details>
<summary><strong>TSRestoreFromCrop</strong> - Возвращает обработанный crop обратно в исходный кадр с опциональным смешиванием.</summary>

![Плейсхолдер скриншота для TSRestoreFromCrop](docs/img/placeholders/tsrestorefromcrop.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Возвращает обработанный crop обратно в исходный кадр с опциональным смешиванием.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `original_images`, `cropped_images`, `crop_data`, `blur`, `blur_type`, `force_gpu`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TSRestoreFromCrop`
- Class: `TSRestoreFromCrop`
- File: `nodes/ts_crop_to_mask_node.py`
- Category: `image/processing`
- Function: `restore`

</details>

<details>
<summary><strong>TS_ImageBatchToImageList</strong> - Преобразует batched IMAGE tensor в покадровый list-поток.</summary>

![Плейсхолдер скриншота для TS_ImageBatchToImageList](docs/img/placeholders/ts-imagebatchtoimagelist.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Преобразует batched IMAGE tensor в покадровый list-поток.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_ImageBatchToImageList`
- Class: `TS_ImageBatchToImageList`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_ImageListToImageBatch</strong> - Собирает list-поток изображений обратно в batched IMAGE tensor.</summary>

![Плейсхолдер скриншота для TS_ImageListToImageBatch](docs/img/placeholders/ts-imagelisttoimagebatch.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Собирает list-поток изображений обратно в batched IMAGE tensor.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `images`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_ImageListToImageBatch`
- Class: `TS_ImageListToImageBatch`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_ImageBatchCut</strong> - Обрезает кадры с начала и/или конца image batch.</summary>

![Плейсхолдер скриншота для TS_ImageBatchCut](docs/img/placeholders/ts-imagebatchcut.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Обрезает кадры с начала и/или конца image batch.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`, `first_cut`, `last_cut`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_ImageBatchCut`
- Class: `TS_ImageBatchCut`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_GetImageMegapixels</strong> - Возвращает значение мегапикселей для быстрой оценки качества и производительности.</summary>

![Плейсхолдер скриншота для TS_GetImageMegapixels](docs/img/placeholders/ts-getimagemegapixels.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Возвращает значение мегапикселей для быстрой оценки качества и производительности.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`
- Опциональные: *(none)*

**Выходы**
- `FLOAT`

**Техническая информация**
- Internal id: `TS_GetImageMegapixels`
- Class: `TS_GetImageMegapixels`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_GetImageSizeSide</strong> - Возвращает размер выбранной стороны изображения для логики графа и автоконфигурации.</summary>

![Плейсхолдер скриншота для TS_GetImageSizeSide](docs/img/placeholders/ts-getimagesizeside.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Возвращает размер выбранной стороны изображения для логики графа и автоконфигурации.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`, `large_side`
- Опциональные: *(none)*

**Выходы**
- `INT`

**Техническая информация**
- Internal id: `TS_GetImageSizeSide`
- Class: `TS_GetImageSizeSide`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_ImagePromptInjector</strong> - Встраивает текст промпта в image-flow, чтобы контекст шёл вместе с веткой изображения.</summary>

![Плейсхолдер скриншота для TS_ImagePromptInjector](docs/img/placeholders/ts-imagepromptinjector.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Встраивает текст промпта в image-flow, чтобы контекст шёл вместе с веткой изображения.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`, `prompt`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_ImagePromptInjector`
- Class: `TS_ImagePromptInjector`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_ImageTileSplitter</strong> - Делит изображение на перекрывающиеся тайлы для тяжёлой обработки в высоком качестве.</summary>

![Плейсхолдер скриншота для TS_ImageTileSplitter](docs/img/placeholders/ts-imagetilesplitter.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Делит изображение на перекрывающиеся тайлы для тяжёлой обработки в высоком качестве.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`, `tile_width`, `tile_height`, `overlap`, `feather`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_ImageTileSplitter`
- Class: `TS_ImageTileSplitter`
- File: `nodes/ts_image_tile_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_ImageTileMerger</strong> - Склеивает тайлы обратно в целое изображение по tile metadata и blending.</summary>

![Плейсхолдер скриншота для TS_ImageTileMerger](docs/img/placeholders/ts-imagetilemerger.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Склеивает тайлы обратно в целое изображение по tile metadata и blending.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `images`, `tile_data`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_ImageTileMerger`
- Class: `TS_ImageTileMerger`
- File: `nodes/ts_image_tile_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TSAutoTileSize</strong> - Автоматически рассчитывает размер тайлов (width/height) под вашу целевую сетку.</summary>

![Плейсхолдер скриншота для TSAutoTileSize](docs/img/placeholders/tsautotilesize.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Автоматически рассчитывает размер тайлов (width/height) под вашу целевую сетку.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `tile_count`, `padding`, `divide_by`
- Опциональные: `image`, `width`, `height`

**Выходы**
- `INT`

**Техническая информация**
- Internal id: `TSAutoTileSize`
- Class: `TSAutoTileSize`
- File: `nodes/ts_image_resize_node.py`
- Category: `utils/Tile Size`
- Function: `calculate_grid`

</details>

<details>
<summary><strong>TS Cube to Equirectangular</strong> - Конвертирует шесть граней куба в одну equirectangular 360-панораму.</summary>

![Плейсхолдер скриншота для TS Cube to Equirectangular](docs/img/placeholders/ts-cube-to-equirectangular.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Конвертирует шесть граней куба в одну equirectangular 360-панораму.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `front`, `right`, `back`, `left`, `top`, `bottom`, `output_width`, `output_height`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS Cube to Equirectangular`
- Class: `TS_CubemapFacesToEquirectangularNode`
- File: `nodes/ts_cube_to_equirect_node.py`
- Category: `Tools/TS_Image`
- Function: `convert`
- Примечание по зависимостям: Requires `py360convert`.

</details>

<details>
<summary><strong>TS Equirectangular to Cube</strong> - Конвертирует 360-панораму в шесть граней куба для редактирования и проекций.</summary>

![Плейсхолдер скриншота для TS Equirectangular to Cube](docs/img/placeholders/ts-equirectangular-to-cube.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Конвертирует 360-панораму в шесть граней куба для редактирования и проекций.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `image`, `cube_size`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS Equirectangular to Cube`
- Class: `TS_EquirectangularToCubemapFacesNode`
- File: `nodes/ts_equirect_to_cube_node.py`
- Category: `Tools/TS_Image`
- Function: `convert`
- Примечание по зависимостям: Requires `py360convert`.

</details>

<details>
<summary><strong>TS_VideoDepthNode</strong> - Строит depth maps по последовательности кадров для композитинга, relighting и depth-эффектов.</summary>

![Плейсхолдер скриншота для TS_VideoDepthNode](docs/img/placeholders/ts-videodepthnode.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Строит depth maps по последовательности кадров для композитинга, relighting и depth-эффектов.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `images`, `model_filename`, `input_size`, `max_res`, `precision`, `colormap`, `dithering_strength`, `apply_median_blur`, `upscale_algorithm`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_VideoDepthNode`
- Class: `TS_VideoDepth`
- File: `nodes/ts_video_depth_node.py`
- Category: `Tools/Video`
- Function: `execute_process_unified`

</details>

<details>
<summary><strong>TS_Video_Upscale_With_Model</strong> - Апскейлит последовательность кадров с выбранной моделью апскейла и memory-стратегиями.</summary>

![Плейсхолдер скриншота для TS_Video_Upscale_With_Model](docs/img/placeholders/ts-video-upscale-with-model.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Апскейлит последовательность кадров с выбранной моделью апскейла и memory-стратегиями.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `model_name`, `images`, `upscale_method`, `factor`, `device_strategy`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_Video_Upscale_With_Model`
- Class: `TS_Video_Upscale_With_Model`
- File: `nodes/ts_video_upscale_node.py`
- Category: `video`
- Function: `upscale_video`
- Примечание по зависимостям: Requires `spandrel` for model loading.

</details>

<details>
<summary><strong>TS_RTX_Upscaler</strong> - Нода NVIDIA RTX Upscaler для быстрого и качественного апскейла на поддерживаемых системах.</summary>

![Плейсхолдер скриншота для TS_RTX_Upscaler](docs/img/placeholders/ts-rtx-upscaler.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Нода NVIDIA RTX Upscaler для быстрого и качественного апскейла на поддерживаемых системах.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `images`, `resize_type`, `scale`, `width`, `height`, `quality`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_RTX_Upscaler`
- Class: `TS_RTX_Upscaler`
- File: `nodes/ts_rtx_upscaler_node.py`
- Category: `TS/Upscaling`
- Function: `upscale`
- Примечание по зависимостям: Requires RTX/VFX runtime components in your environment.

</details>

<details>
<summary><strong>TS_DeflickerNode</strong> - Снижает временное мерцание яркости и цвета в видеопоследовательностях.</summary>

![Плейсхолдер скриншота для TS_DeflickerNode](docs/img/placeholders/ts-deflickernode.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Снижает временное мерцание яркости и цвета в видеопоследовательностях.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `images`, `method`, `window_size`, `intensity`, `preserve_details`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_DeflickerNode`
- Class: `TS_DeflickerNode`
- File: `nodes/ts_deflicker_node.py`
- Category: `Video PostProcessing`
- Function: `deflicker`

</details>

<details>
<summary><strong>TS_Free_Video_Memory</strong> - Pass-through нода, которая агрессивно освобождает RAM/VRAM между тяжёлыми видео-шагами.</summary>

![Плейсхолдер скриншота для TS_Free_Video_Memory](docs/img/placeholders/ts-free-video-memory.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Pass-through нода, которая агрессивно освобождает RAM/VRAM между тяжёлыми видео-шагами.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `images`, `aggressive_cleanup`, `report_memory`
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_Free_Video_Memory`
- Class: `TS_Free_Video_Memory`
- File: `nodes/ts_video_upscale_node.py`
- Category: `video`
- Function: `cleanup_memory`

</details>

<details>
<summary><strong>TS_LTX_FirstLastFrame</strong> - Добавляет guidance первого и последнего кадра в latent-пайплайн (полезно для LTX video control).</summary>

![Плейсхолдер скриншота для TS_LTX_FirstLastFrame](docs/img/placeholders/ts-ltx-firstlastframe.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Добавляет guidance первого и последнего кадра в latent-пайплайн (полезно для LTX video control).

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `vae`, `latent`, `first_strength`, `last_strength`, `enable_last_frame`
- Опциональные: `first_image`, `last_image`

**Выходы**
- `LATENT`

**Техническая информация**
- Internal id: `TS_LTX_FirstLastFrame`
- Class: `TS_LTX_FirstLastFrame`
- File: `nodes/ts_ltx_tools_node.py`
- Category: `conditioning/video_models`
- Function: `execute`

</details>

<details>
<summary><strong>TS_Animation_Preview</strong> - Создаёт быстрый превью-ролик из кадров с опциональным объединением аудио.</summary>

![Плейсхолдер скриншота для TS_Animation_Preview](docs/img/placeholders/ts-animation-preview.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Создаёт быстрый превью-ролик из кадров с опциональным объединением аудио.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `images`, `fps`
- Опциональные: `audio`

**Выходы**
- `-`

**Техническая информация**
- Internal id: `TS_Animation_Preview`
- Class: `TS_Animation_Preview`
- File: `nodes/ts_interface_tools_node.py`
- Category: `TS/Interface Tools`
- Function: `preview`
- Примечание по зависимостям: Использует `imageio` / `imageio-ffmpeg` for video writing.

</details>

<details>
<summary><strong>TS_FileBrowser</strong> - Встроенный media picker: загружает image/video/audio/mask с диска прямо в ваш граф.</summary>

![Плейсхолдер скриншота для TS_FileBrowser](docs/img/placeholders/ts-filebrowser.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Встроенный media picker: загружает image/video/audio/mask с диска прямо в ваш граф.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: *(dynamic in code/UI)*
- Опциональные: *(none)*

**Выходы**
- `IMAGE`

**Техническая информация**
- Internal id: `TS_FileBrowser`
- Class: `TS_FileBrowser`
- File: `nodes/ts_file_browser_node.py`
- Category: `TS/Input`
- Function: `get_selected_media`

</details>

<details>
<summary><strong>TS_FilePathLoader</strong> - Возвращает путь к файлу и имя файла по индексу из списка папки.</summary>

![Плейсхолдер скриншота для TS_FilePathLoader](docs/img/placeholders/ts-filepathloader.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Возвращает путь к файлу и имя файла по индексу из списка папки.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `folder_path`, `index`
- Опциональные: *(none)*

**Выходы**
- `STRING`

**Техническая информация**
- Internal id: `TS_FilePathLoader`
- Class: `TS_FilePathLoader`
- File: `nodes/ts_file_path_node.py`
- Category: `file_utils`
- Function: `get_file_path`

</details>

<details>
<summary><strong>TS Files Downloader</strong> - Массово скачивает модели и ассеты: resume, mirrors, proxy и опциональная распаковка.</summary>

![Плейсхолдер скриншота для TS Files Downloader](docs/img/placeholders/ts-files-downloader.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Массово скачивает модели и ассеты: resume, mirrors, proxy и опциональная распаковка.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `file_list`, `skip_existing`, `verify_size`, `chunk_size_kb`
- Опциональные: `hf_token`, `hf_domain`, `proxy_url`, `modelscope_token`, `unzip_after_download`, `enable`

**Выходы**
- `-`

**Техническая информация**
- Internal id: `TS Files Downloader`
- Class: `TS_DownloadFilesNode`
- File: `nodes/ts_downloader_node.py`
- Category: `Tools/TS_IO`
- Function: `execute_downloads`

</details>

<details>
<summary><strong>TS Youtube Chapters</strong> - Конвертирует EDL-тайминги в готовые timestamp-главы для YouTube.</summary>

![Плейсхолдер скриншота для TS Youtube Chapters](docs/img/placeholders/ts-youtube-chapters.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Конвертирует EDL-тайминги в готовые timestamp-главы для YouTube.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `edl_file_path`
- Опциональные: *(none)*

**Выходы**
- `STRING`

**Техническая информация**
- Internal id: `TS Youtube Chapters`
- Class: `TS_EDLToYouTubeChaptersNode`
- File: `nodes/ts_edl_chapters_node.py`
- Category: `Tools/TS_Video`
- Function: `convert_edl_to_youtube_chapters`

</details>

<details>
<summary><strong>TS_ModelScanner</strong> - Сканирует файлы моделей и возвращает читаемую сводку структуры и метаданных.</summary>

![Плейсхолдер скриншота для TS_ModelScanner](docs/img/placeholders/ts-modelscanner.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Сканирует файлы моделей и возвращает читаемую сводку структуры и метаданных.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `model_name`
- Опциональные: `model`, `summary_only`

**Выходы**
- `STRING`

**Техническая информация**
- Internal id: `TS_ModelScanner`
- Class: `TS_ModelScanner`
- File: `nodes/ts_models_tools_node.py`
- Category: `utils/model_analysis`
- Function: `scan_model`

</details>

<details>
<summary><strong>TS_ModelConverter</strong> - Конвертер модели в один клик для сценариев смены precision.</summary>

![Плейсхолдер скриншота для TS_ModelConverter](docs/img/placeholders/ts-modelconverter.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Конвертер модели в один клик для сценариев смены precision.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `model`
- Опциональные: *(none)*

**Выходы**
- `MODEL`

**Техническая информация**
- Internal id: `TS_ModelConverter`
- Class: `TS_ModelConverterNode`
- File: `nodes/ts_models_tools_node.py`
- Category: `conversion`
- Function: `convert_to_fp8`

</details>

<details>
<summary><strong>TS_ModelConverterAdvanced</strong> - Расширенный конвертер моделей с детальным контролем формата, пресета и результата.</summary>

![Плейсхолдер скриншота для TS_ModelConverterAdvanced](docs/img/placeholders/ts-modelconverteradvanced.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Расширенный конвертер моделей с детальным контролем формата, пресета и результата.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `model_name`, `fp8_mode`, `conversion_preset`, `shard_subdir`, `final_filename`
- Опциональные: `model`

**Выходы**
- `STRING`

**Техническая информация**
- Internal id: `TS_ModelConverterAdvanced`
- Class: `TS_ModelConverterAdvancedNode`
- File: `nodes/ts_models_tools_node.py`
- Category: `Model Conversion`
- Function: `convert_model`

</details>

<details>
<summary><strong>TS_ModelConverterAdvancedDirect</strong> - Расширенный конвертер, работающий напрямую от подключённого входа MODEL.</summary>

![Плейсхолдер скриншота для TS_ModelConverterAdvancedDirect](docs/img/placeholders/ts-modelconverteradvanceddirect.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Расширенный конвертер, работающий напрямую от подключённого входа MODEL.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `model`, `fp8_mode`, `conversion_preset`, `shard_subdir`, `final_filename`
- Опциональные: *(none)*

**Выходы**
- `STRING`

**Техническая информация**
- Internal id: `TS_ModelConverterAdvancedDirect`
- Class: `TS_ModelConverterAdvancedDirectNode`
- File: `nodes/ts_models_tools_node.py`
- Category: `TS/Model Conversion`
- Function: `convert_model`

</details>

<details>
<summary><strong>TS_CPULoraMerger</strong> - ?????????? ?? ??????? LoRA ? ??????? ?????? ?? CPU ? ????????? ????? safetensors-????.</summary>

![Плейсхолдер скриншота для TS_CPULoraMerger](docs/img/placeholders/ts-cpu-lora-merger.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
?????????? ?? ??????? LoRA ? ??????? ?????? ?? CPU ? ????????? ????? safetensors-????.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `base_model`, `lora_1_name`, `lora_1_strength`, `lora_2_name`, `lora_2_strength`, `lora_3_name`, `lora_3_strength`, `lora_4_name`, `lora_4_strength`, `output_model_name`
- Опциональные: *(none)*

**Выходы**
- `STRING, STRING`

**Техническая информация**
- Internal id: `TS_CPULoraMerger`
- Class: `TS_CPULoraMergerNode`
- File: `nodes/ts_models_tools_node.py`
- Category: `TS/Model Tools`
- Function: `merge_to_file`
- Примечание по зависимостям: Использует ComfyUI model loading and `safetensors` for CPU-side merge and save.

</details>

<details>
<summary><strong>TS_FloatSlider</strong> - Простой UI float slider для аккуратного управления параметрами графа.</summary>

![Плейсхолдер скриншота для TS_FloatSlider](docs/img/placeholders/ts-floatslider.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Простой UI float slider для аккуратного управления параметрами графа.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `value`
- Опциональные: *(none)*

**Выходы**
- `FLOAT`

**Техническая информация**
- Internal id: `TS_FloatSlider`
- Class: `TS_FloatSlider`
- File: `nodes/ts_interface_tools_node.py`
- Category: `TS Tools/Sliders`
- Function: `get_value`

</details>

<details>
<summary><strong>TS_Int_Slider</strong> - Простой UI integer slider для детерминированных целочисленных параметров.</summary>

![Плейсхолдер скриншота для TS_Int_Slider](docs/img/placeholders/ts-int-slider.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Простой UI integer slider для детерминированных целочисленных параметров.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `value`
- Опциональные: *(none)*

**Выходы**
- `INT`

**Техническая информация**
- Internal id: `TS_Int_Slider`
- Class: `TS_Int_Slider`
- File: `nodes/ts_interface_tools_node.py`
- Category: `TS Tools/Sliders`
- Function: `get_value`

</details>

<details>
<summary><strong>TS_Smart_Switch</strong> - Переключает между двумя входами по режиму и помогает держать граф компактным.</summary>

![Плейсхолдер скриншота для TS_Smart_Switch](docs/img/placeholders/ts-smart-switch.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Переключает между двумя входами по режиму и помогает держать граф компактным.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `data_type`, `switch`
- Опциональные: `input_1`, `input_2`

**Выходы**
- `*`

**Техническая информация**
- Internal id: `TS_Smart_Switch`
- Class: `TS_Smart_Switch`
- File: `nodes/ts_interface_tools_node.py`
- Category: `TS Tools/Logic`
- Function: `smart_switch`

</details>

<details>
<summary><strong>TS_Math_Int</strong> - Нода целочисленной математики для счётчиков, смещений и простой логики графа.</summary>

![Плейсхолдер скриншота для TS_Math_Int](docs/img/placeholders/ts-math-int.png)

> Плейсхолдер: замените этот блок вашим скриншотом ноды.

**Что делает эта нода**
Нода целочисленной математики для счётчиков, смещений и простой логики графа.

**Быстрый старт**
1. Добавьте ноду в граф и подключите обязательные входы.
2. Сначала оставьте дефолты и меняйте параметры постепенно.
3. Подключите выход к следующей ноде и сравните результат.

**Основные параметры**
- Обязательные: `a`, `b`, `operation`
- Опциональные: *(none)*

**Выходы**
- `INT`

**Техническая информация**
- Internal id: `TS_Math_Int`
- Class: `TS_Math_Int`
- File: `nodes/ts_interface_tools_node.py`
- Category: `TS/Math`
- Function: `calculate`

</details>
