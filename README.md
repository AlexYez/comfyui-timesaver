# ComfyUI Timesaver Nodes

[Русский](#русский) | [English](#english)

## Русский

Timesaver - это набор полезных нод для ComfyUI. Он помогает быстрее работать с изображениями, видео, аудио, текстом, моделями и файлами прямо внутри графа.

README написан для тех, кто только осваивает ComfyUI. Здесь нет необходимости знать внутреннее устройство ComfyUI: достаточно понимать, что нода получает данные на входе, что-то с ними делает и отдает результат на выходе.

Версия пака: `8.6`

Репозиторий: https://github.com/AlexYez/comfyui-timesaver

## Что Есть В Паке

В паке сейчас **57 нод**. Их можно условно разделить так:

- **изображения**: resize, цвет, фон, маски, тайлы, 360-панорамы;
- **видео**: глубина, апскейл, дефликер, интерполяция кадров, предпросмотр;
- **аудио и речь**: загрузка аудио, запись с микрофона, Whisper, TTS, разделение музыки на дорожки;
- **текст и промпты**: сборка промптов, стили, пакетная выдача строк;
- **модели и файлы**: браузер файлов, загрузчик моделей, конвертеры, сканер моделей;
- **маленькие помощники**: слайдеры, переключатель, простая математика.

Если вы только начинаете, не нужно читать весь список сразу. Проще выбрать задачу, найти подходящую ноду в таблице ниже и попробовать ее на простом примере.

## Установка

1. Поместите папку пака сюда:

```text
ComfyUI/custom_nodes/comfyui-timesaver
```

2. Установите зависимости:

```bash
python -m pip install -r ComfyUI/custom_nodes/comfyui-timesaver/requirements.txt
```

3. Перезапустите ComfyUI.

Если используется portable-сборка ComfyUI на Windows, запускайте `pip` из того Python, который идет вместе с этой сборкой. Иначе зависимости могут установиться не туда.

## Обновление

Если пак уже установлен через git:

```bash
cd ComfyUI/custom_nodes/comfyui-timesaver
git pull
python -m pip install -r requirements.txt
```

После обновления перезапустите ComfyUI.

## Первые Шаги

1. Запустите ComfyUI.
2. Откройте меню добавления ноды.
3. В поиске начните вводить `TS`.
4. Выберите нужную ноду.
5. Подключите входы и выходы.
6. Сначала попробуйте настройки по умолчанию, а затем меняйте по одному параметру.

Так проще понять, какая настройка за что отвечает.

## Как Читать Названия Выходов

В таблицах ниже встречаются типы вроде `IMAGE`, `MASK`, `AUDIO`, `STRING`, `INT`.

- `IMAGE` - изображение или пачка кадров.
- `MASK` - маска, чаще всего черно-белая область.
- `AUDIO` - звук.
- `STRING` - текст.
- `INT` и `FLOAT` - числа.
- `LATENT` и `MODEL` - специальные данные для других нод ComfyUI.

Это не нужно запоминать сразу. ComfyUI сам подсказывает, какие выходы можно соединять с какими входами.

## Быстрый Выбор Ноды

| Если нужно | Попробуйте |
| --- | --- |
| Загрузить картинку, видео, аудио или маску из файла | `TS_FileBrowser` |
| Загрузить или записать аудио | `TS_AudioLoader` |
| Посмотреть и обрезать аудио перед работой | `TS_AudioPreview` |
| Распознать речь из аудио | `TSWhisper` |
| Наговорить идею и улучшить ее в AI prompt | `TS_SuperPrompt` |
| Озвучить русский текст | `TS_SileroTTS` |
| Поставить ударения для Silero | `TS_SileroStress` |
| Разделить песню на вокал, бас, барабаны и другое | `TS_MusicStems` |
| Собрать промпт из готовых блоков | `TS_PromptBuilder` |
| Выбрать художественный стиль для промпта | `TS_StylePromptSelector` |
| Изменить размер изображения | `TS_ImageResize` |
| Подготовить картинку под Qwen | `TS_QwenSafeResize`, `TS_QwenCanvas` |
| Подготовить размер под WAN | `TS_WAN_SafeResize` |
| Удалить фон | `TS_BGRM_BiRefNet` |
| Работать с green screen или blue screen | `TS_Keyer`, `TS_Despill` |
| Исправить цвет или добавить пленочный вид | `TS_Color_Grade`, `TS_Film_Emulation`, `TS_FilmGrain` |
| Разбить большое изображение на тайлы | `TS_ImageTileSplitter`, `TS_ImageTileMerger` |
| Сделать глубину по видео-кадрам | `TS_VideoDepthNode` |
| Апскейлить видео или кадры | `TS_Video_Upscale_With_Model`, `TS_RTX_Upscaler` |
| Уменьшить мерцание видео | `TS_DeflickerNode` |
| Сделать предпросмотр анимации | `TS_Animation_Preview` |
| Скачать модели или ассеты | `TS Files Downloader` |
| Посмотреть информацию о модели | `TS_ModelScanner` |
| Конвертировать модель | `TS_ModelConverter`, `TS_ModelConverterAdvanced` |
| Добавить удобный числовой ползунок | `TS_FloatSlider`, `TS_Int_Slider` |
| Переключаться между двумя ветками графа | `TS_Smart_Switch` |

## Каталог Нод

### Изображения

| Нода | Простое описание | Выходы |
| --- | --- | --- |
| `TS_ImageResize` | Меняет размер изображения: точно, по стороне, по масштабу или по мегапикселям. | `IMAGE`, `INT`, `INT`, `MASK` |
| `TS_QwenSafeResize` | Быстро приводит изображение к размеру, удобному для Qwen. | `IMAGE` |
| `TS_WAN_SafeResize` | Подбирает размер изображения для WAN-пайплайнов. | `IMAGE`, `INT`, `INT`, `STRING` |
| `TS_QwenCanvas` | Создает холст нужного размера и может разместить на нем изображение или маску. | `IMAGE`, `INT`, `INT` |
| `TS_ResolutionSelector` | Помогает выбрать разрешение по соотношению сторон. | `IMAGE` |
| `TS_Color_Grade` | Базовая цветокоррекция: яркость, контраст, оттенок, температура и насыщенность. | `IMAGE` |
| `TS_Film_Emulation` | Добавляет кинематографичный или пленочный вид. | `IMAGE` |
| `TS_FilmGrain` | Добавляет на изображение управляемое пленочное зерно. | `IMAGE` |
| `TS_Color_Match` | Переносит цветовое настроение с одного изображения на другое. | `IMAGE` |
| `TS_Keyer` | Вырезает объект с зеленого, синего или красного фона. | `IMAGE`, `MASK`, `IMAGE` |
| `TS_Despill` | Убирает цветные засветки по краям после хромакея. | `IMAGE`, `MASK`, `IMAGE` |
| `TS_BGRM_BiRefNet` | Удаляет фон с помощью BiRefNet. | `IMAGE`, `MASK`, `IMAGE` |
| `TSCropToMask` | Обрезает изображение вокруг маски, чтобы быстрее обрабатывать нужную область. | `IMAGE`, `MASK`, `CROP_DATA`, `INT`, `INT` |
| `TSRestoreFromCrop` | Возвращает обработанный фрагмент обратно в исходный кадр. | `IMAGE` |
| `TS_ImageBatchToImageList` | Разбирает пачку изображений на отдельные элементы. | `IMAGE` |
| `TS_ImageListToImageBatch` | Собирает отдельные изображения обратно в пачку. | `IMAGE` |
| `TS_ImageBatchCut` | Убирает лишние кадры в начале или конце пачки. | `IMAGE` |
| `TS_GetImageMegapixels` | Показывает размер изображения в мегапикселях. | `FLOAT` |
| `TS_GetImageSizeSide` | Возвращает ширину или высоту изображения. | `INT` |
| `TS_ImagePromptInjector` | Прикрепляет текстовый промпт к ветке с изображением. | `IMAGE` |
| `TS_ImageTileSplitter` | Делит большое изображение на части для обработки. | `IMAGE`, `TILE_INFO` |
| `TS_ImageTileMerger` | Собирает части изображения обратно. | `IMAGE` |
| `TSAutoTileSize` | Подбирает размер тайла для выбранной сетки. | `INT`, `INT` |
| `TS Cube to Equirectangular` | Собирает шесть граней куба в 360-панораму. | `IMAGE` |
| `TS Equirectangular to Cube` | Разбирает 360-панораму на шесть граней куба. | `IMAGE` x6 |

### Видео

| Нода | Простое описание | Выходы |
| --- | --- | --- |
| `TS_Frame_Interpolation` | Добавляет промежуточные кадры, чтобы движение выглядело плавнее. | `IMAGE` |
| `TS_VideoDepthNode` | Строит карты глубины для последовательности кадров. | `IMAGE` |
| `TS_Video_Upscale_With_Model` | Апскейлит кадры видео выбранной моделью. | `IMAGE` |
| `TS_RTX_Upscaler` | Использует NVIDIA RTX Upscaler, если система поддерживает его. | `IMAGE` |
| `TS_DeflickerNode` | Уменьшает мерцание яркости и цвета между кадрами. | `IMAGE` |
| `TS_Free_Video_Memory` | Помогает освободить память между тяжелыми шагами видео-графа. | `IMAGE` |
| `TS_LTX_FirstLastFrame` | Добавляет первый и последний кадр в latent-ветку для LTX-процессов. | `LATENT` |
| `TS_Animation_Preview` | Делает быстрый предпросмотр анимации и может добавить звук. | UI |

### Аудио И Речь

| Нода | Простое описание | Выходы |
| --- | --- | --- |
| `TS_AudioLoader` | Загружает аудио или аудиодорожку из видео, умеет записывать звук с микрофона. | `AUDIO`, `INT` |
| `TS_AudioPreview` | Показывает аудио с формой волны и помогает выбрать фрагмент. | UI |
| `TSWhisper` | Распознает речь из аудио, может сохранить SRT и TTML. | `STRING`, `STRING`, `STRING` |
| `TS_SileroTTS` | Озвучивает русский текст через Silero. | `AUDIO` |
| `TS_SileroStress` | Добавляет ударения и букву "ё" для более аккуратного русского TTS. | `STRING` |
| `TS_MusicStems` | Разделяет музыку на вокал, барабаны, бас, другие инструменты и минус. | `AUDIO` x5 |

### Текст, Промпты И LLM

| Нода | Простое описание | Выходы |
| --- | --- | --- |
| `TS_Qwen3_VL_V3` | Отправляет текст, изображение или видео в Qwen и возвращает ответ. | `STRING`, `IMAGE` |
| `TS_SuperPrompt` | Записывает речь в текстовое поле, показывает progressbar и улучшает промпт через Huihui-Qwen3.5-2B-abliterated с пресетами из `qwen_3_vl_presets.json`. | `STRING` |
| `TS_PromptBuilder` | Собирает промпт из готовых категорий и поддерживает seed для повторяемости. | `STRING` |
| `TS_BatchPromptLoader` | Берет одну строку из многострочного списка по индексу. | `STRING`, `INT` |
| `TS_StylePromptSelector` | Подставляет готовый стиль из локальной библиотеки стилей. | `STRING` |

### Файлы, Модели И Служебные Ноды

| Нода | Простое описание | Выходы |
| --- | --- | --- |
| `TS_FileBrowser` | Удобный выбор файла прямо внутри ноды: изображение, видео, аудио, маска или путь. | `IMAGE`, `VIDEO`, `AUDIO`, `MASK`, `STRING` |
| `TS_FilePathLoader` | Возвращает путь и имя файла из папки по индексу. | `STRING`, `STRING` |
| `TS Files Downloader` | Скачивает модели и другие файлы, умеет пропускать уже скачанное. | UI |
| `TS Youtube Chapters` | Превращает EDL-разметку в таймкоды глав для YouTube. | `STRING` |
| `TS_ModelScanner` | Показывает понятную сводку по файлу модели. | `STRING` |
| `TS_ModelConverter` | Конвертирует модель в другой формат или точность. | `MODEL` |
| `TS_ModelConverterAdvanced` | Расширенный вариант конвертера с большим числом настроек. | `STRING` |
| `TS_ModelConverterAdvancedDirect` | Конвертирует подключенную модель напрямую из графа. | `STRING` |
| `TS_CPULoraMerger` | Объединяет LoRA с базовой моделью на CPU и сохраняет результат. | `STRING`, `STRING` |

### Простые Помощники Для Графа

| Нода | Простое описание | Выходы |
| --- | --- | --- |
| `TS_FloatSlider` | Удобный ползунок для дробных чисел. | `FLOAT` |
| `TS_Int_Slider` | Удобный ползунок для целых чисел. | `INT` |
| `TS_Smart_Switch` | Переключает граф между двумя входами. | `*` |
| `TS_Math_Int` | Делает простые операции с целыми числами. | `INT` |

## Зависимости И Модели

Некоторые ноды работают сразу после установки пака. Другим нужны дополнительные библиотеки или модели.

Чаще всего это касается:

- Whisper и распознавания речи;
- Qwen;
- Silero TTS;
- BiRefNet;
- Demucs для разделения музыки;
- видео-апскейла и интерполяции кадров.

Если в консоли ComfyUI появляется сообщение о недостающей зависимости, сначала выполните:

```bash
python -m pip install -r ComfyUI/custom_nodes/comfyui-timesaver/requirements.txt
```

После установки перезапустите ComfyUI.

Модели обычно скачиваются или выбираются отдельно. Не все ноды обязаны скачивать модели автоматически, поэтому при первом запуске тяжелых нод возможна пауза или сообщение о нужном файле.

## Полезные Советы Для Новичков

- Начинайте с маленького изображения или короткого аудио. Так проще понять результат и не ждать слишком долго.
- Меняйте одну настройку за раз. Если изменить сразу десять параметров, сложнее понять, что именно повлияло на результат.
- Смотрите на типы входов и выходов. `IMAGE` соединяется с `IMAGE`, `AUDIO` с `AUDIO`, текст чаще всего идет в `STRING`.
- Если нода работает с видео, помните, что видео в ComfyUI часто представлено как пачка изображений.
- Если ComfyUI зависает на тяжелой ноде, попробуйте меньший размер кадра, меньше кадров или более легкую модель.

## Частые Вопросы

### Ноды не появились в меню

Проверьте, что папка лежит именно здесь:

```text
ComfyUI/custom_nodes/comfyui-timesaver
```

После этого перезапустите ComfyUI и посмотрите консоль. Если там есть ошибка импорта, установите зависимости из `requirements.txt`.

### Появилась ошибка `No module named ...`

Это значит, что не установлена нужная Python-библиотека. Установите зависимости командой:

```bash
python -m pip install -r requirements.txt
```

Команду нужно запускать из папки `comfyui-timesaver` или указывать полный путь к `requirements.txt`.

### Микрофон не работает в TS Super Prompt

Браузер должен разрешить доступ к микрофону. Обычно это работает на:

```text
http://localhost:8188
```

Если открыть ComfyUI по IP-адресу в локальной сети, браузер может запретить микрофон без HTTPS.

### Видео-ноды используют слишком много памяти

Попробуйте уменьшить размер кадров, сократить количество кадров или добавить `TS_Free_Video_Memory` между тяжелыми шагами.

## Структура Пака

```text
comfyui-timesaver/
  nodes/              Python-ноды
  js/                 интерфейсные расширения для ComfyUI
  doc/                служебная документация и скрипты
  requirements.txt    зависимости
  pyproject.toml      сведения о паке
  __init__.py         загрузчик нод
```

## Для Разработчиков

Быстрая проверка после изменений:

```bash
python -m compileall .
python -m pytest tests
```

Если тестов или зависимостей нет в текущей среде, проверьте хотя бы импорт и запуск ComfyUI. Для frontend-нод дополнительно проверьте браузерную консоль.

## Лицензия И Репозиторий

Основной репозиторий:

```text
https://github.com/AlexYez/comfyui-timesaver
```

Если вы только начинаете изучать ComfyUI, лучший путь - собрать простой граф, добавить одну ноду из этого пака и посмотреть, как меняется результат. Постепенно станет понятно, какие ноды нужны именно для вашего рабочего процесса.
## English

Timesaver is a collection of practical ComfyUI nodes. It helps with images, video, audio, text, models, and files directly inside a ComfyUI graph.

This README is written for people who are just starting with ComfyUI. There is no need to know ComfyUI internals first. A node receives something, does a job, and returns a result that can be connected to the next node.

Pack version: `8.6`

Repository: https://github.com/AlexYez/comfyui-timesaver

## What Is Included

The pack currently contains **57 nodes**. They can be grouped like this:

- **images**: resizing, color work, background removal, masks, tiles, 360 panoramas;
- **video**: depth maps, upscaling, deflicker, frame interpolation, preview;
- **audio and speech**: audio loading, microphone recording, Whisper, TTS, music stem separation;
- **text and prompts**: prompt building, style prompts, batch prompt lines;
- **models and files**: file browser, downloader, converters, model scanner;
- **small helpers**: sliders, switches, simple math.

For a first try, it is best to pick one task from the table below, add one node, and test it with a small file or a short prompt.

## Installation

1. Place the pack folder here:

```text
ComfyUI/custom_nodes/comfyui-timesaver
```

2. Install the dependencies:

```bash
python -m pip install -r ComfyUI/custom_nodes/comfyui-timesaver/requirements.txt
```

3. Restart ComfyUI.

For portable ComfyUI builds on Windows, use the Python that comes with that ComfyUI build. Otherwise the dependencies may be installed into a different Python environment.

## Updating

If the pack was installed with git:

```bash
cd ComfyUI/custom_nodes/comfyui-timesaver
git pull
python -m pip install -r requirements.txt
```

Restart ComfyUI after updating.

## First Steps

1. Start ComfyUI.
2. Open the add-node menu.
3. Search for `TS`.
4. Add the node that matches the task.
5. Connect the inputs and outputs.
6. Start with the default settings, then change one option at a time.

Changing one setting at a time makes it much easier to understand what each control does.

## Output Types In Plain Language

Some tables below use short type names:

- `IMAGE` means an image or a batch of frames.
- `MASK` means a mask, usually a black-and-white area.
- `AUDIO` means sound.
- `STRING` means text.
- `INT` and `FLOAT` mean numbers.
- `LATENT` and `MODEL` are special ComfyUI data types used by other nodes.

ComfyUI usually shows which connections are valid, so there is no need to memorize all of this at once.

## Quick Node Picker

| Task | Try |
| --- | --- |
| Load an image, video, audio file, or mask | `TS_FileBrowser` |
| Load or record audio | `TS_AudioLoader` |
| Preview and trim audio | `TS_AudioPreview` |
| Transcribe speech from an audio input | `TSWhisper` |
| Dictate an idea and improve it into an AI prompt | `TS_SuperPrompt` |
| Generate Russian speech from text | `TS_SileroTTS` |
| Add Russian stress marks for Silero | `TS_SileroStress` |
| Split music into vocals, drums, bass, and more | `TS_MusicStems` |
| Build prompts from reusable blocks | `TS_PromptBuilder` |
| Pick a ready-made prompt style | `TS_StylePromptSelector` |
| Resize an image | `TS_ImageResize` |
| Prepare an image for Qwen | `TS_QwenSafeResize`, `TS_QwenCanvas` |
| Prepare an image size for WAN workflows | `TS_WAN_SafeResize` |
| Remove a background | `TS_BGRM_BiRefNet` |
| Work with green screen or blue screen footage | `TS_Keyer`, `TS_Despill` |
| Adjust color or add a film-like look | `TS_Color_Grade`, `TS_Film_Emulation`, `TS_FilmGrain` |
| Split a large image into tiles | `TS_ImageTileSplitter`, `TS_ImageTileMerger` |
| Create depth maps from video frames | `TS_VideoDepthNode` |
| Upscale video frames | `TS_Video_Upscale_With_Model`, `TS_RTX_Upscaler` |
| Reduce video flicker | `TS_DeflickerNode` |
| Make an animation preview | `TS_Animation_Preview` |
| Download models or assets | `TS Files Downloader` |
| Inspect a model file | `TS_ModelScanner` |
| Convert model files | `TS_ModelConverter`, `TS_ModelConverterAdvanced` |
| Add a convenient number slider | `TS_FloatSlider`, `TS_Int_Slider` |
| Switch between two graph branches | `TS_Smart_Switch` |

## Node Catalog

### Images

| Node | Plain Description | Outputs |
| --- | --- | --- |
| `TS_ImageResize` | Resizes an image by exact size, side length, scale, or megapixels. | `IMAGE`, `INT`, `INT`, `MASK` |
| `TS_QwenSafeResize` | Quickly prepares an image size that works well with Qwen. | `IMAGE` |
| `TS_WAN_SafeResize` | Chooses a model-friendly image size for WAN workflows. | `IMAGE`, `INT`, `INT`, `STRING` |
| `TS_QwenCanvas` | Creates a canvas and can place an image or mask on it. | `IMAGE`, `INT`, `INT` |
| `TS_ResolutionSelector` | Helps choose a resolution by aspect ratio. | `IMAGE` |
| `TS_Color_Grade` | Basic color grading: brightness, contrast, hue, temperature, and saturation. | `IMAGE` |
| `TS_Film_Emulation` | Adds a cinematic or film-like look. | `IMAGE` |
| `TS_FilmGrain` | Adds controlled film grain. | `IMAGE` |
| `TS_Color_Match` | Matches the color mood of one image to another. | `IMAGE` |
| `TS_Keyer` | Cuts out an object from a green, blue, or red screen. | `IMAGE`, `MASK`, `IMAGE` |
| `TS_Despill` | Cleans colored edge spill after keying. | `IMAGE`, `MASK`, `IMAGE` |
| `TS_BGRM_BiRefNet` | Removes the background with BiRefNet. | `IMAGE`, `MASK`, `IMAGE` |
| `TSCropToMask` | Crops around a mask so only the important area is processed. | `IMAGE`, `MASK`, `CROP_DATA`, `INT`, `INT` |
| `TSRestoreFromCrop` | Places a processed crop back into the original frame. | `IMAGE` |
| `TS_ImageBatchToImageList` | Splits an image batch into separate items. | `IMAGE` |
| `TS_ImageListToImageBatch` | Combines separate images back into a batch. | `IMAGE` |
| `TS_ImageBatchCut` | Removes frames from the beginning or end of a batch. | `IMAGE` |
| `TS_GetImageMegapixels` | Returns the image size in megapixels. | `FLOAT` |
| `TS_GetImageSizeSide` | Returns image width or height. | `INT` |
| `TS_ImagePromptInjector` | Attaches prompt text to an image branch. | `IMAGE` |
| `TS_ImageTileSplitter` | Splits a large image into smaller tiles. | `IMAGE`, `TILE_INFO` |
| `TS_ImageTileMerger` | Merges image tiles back together. | `IMAGE` |
| `TSAutoTileSize` | Suggests tile size for a chosen grid. | `INT`, `INT` |
| `TS Cube to Equirectangular` | Combines six cube faces into a 360 panorama. | `IMAGE` |
| `TS Equirectangular to Cube` | Splits a 360 panorama into six cube faces. | `IMAGE` x6 |

### Video

| Node | Plain Description | Outputs |
| --- | --- | --- |
| `TS_Frame_Interpolation` | Adds in-between frames for smoother motion. | `IMAGE` |
| `TS_VideoDepthNode` | Creates depth maps from frame sequences. | `IMAGE` |
| `TS_Video_Upscale_With_Model` | Upscales video frames with a selected model. | `IMAGE` |
| `TS_RTX_Upscaler` | Uses NVIDIA RTX Upscaler when supported by the system. | `IMAGE` |
| `TS_DeflickerNode` | Reduces brightness and color flicker between frames. | `IMAGE` |
| `TS_Free_Video_Memory` | Helps free memory between heavy video steps. | `IMAGE` |
| `TS_LTX_FirstLastFrame` | Adds first and last frame guidance for LTX workflows. | `LATENT` |
| `TS_Animation_Preview` | Creates a quick animation preview and can include audio. | UI |

### Audio And Speech

| Node | Plain Description | Outputs |
| --- | --- | --- |
| `TS_AudioLoader` | Loads audio or a video's audio track, and can record microphone audio. | `AUDIO`, `INT` |
| `TS_AudioPreview` | Shows a waveform preview and helps choose an audio fragment. | UI |
| `TSWhisper` | Transcribes speech from audio and can save SRT and TTML subtitles. | `STRING`, `STRING`, `STRING` |
| `TS_SileroTTS` | Generates Russian speech with Silero. | `AUDIO` |
| `TS_SileroStress` | Adds stress marks and "yo" letters for cleaner Russian TTS. | `STRING` |
| `TS_MusicStems` | Splits music into vocals, drums, bass, other instruments, and instrumental. | `AUDIO` x5 |

### Text, Prompts, And LLM

| Node | Plain Description | Outputs |
| --- | --- | --- |
| `TS_Qwen3_VL_V3` | Sends text, image, or video to Qwen and returns a response. | `STRING`, `IMAGE` |
| `TS_SuperPrompt` | Records speech into a text field, shows a progress bar, and enhances prompts with Huihui-Qwen3.5-2B-abliterated using presets from `qwen_3_vl_presets.json`. | `STRING` |
| `TS_PromptBuilder` | Builds a prompt from reusable categories and supports seeds. | `STRING` |
| `TS_BatchPromptLoader` | Selects one line from a multiline prompt list by index. | `STRING`, `INT` |
| `TS_StylePromptSelector` | Inserts a ready-made style from the local style library. | `STRING` |

### Files, Models, And Utilities

| Node | Plain Description | Outputs |
| --- | --- | --- |
| `TS_FileBrowser` | Picks files inside a node: image, video, audio, mask, or path. | `IMAGE`, `VIDEO`, `AUDIO`, `MASK`, `STRING` |
| `TS_FilePathLoader` | Returns a file path and file name from a folder by index. | `STRING`, `STRING` |
| `TS Files Downloader` | Downloads models and files, and can skip files that already exist. | UI |
| `TS Youtube Chapters` | Converts EDL timing into YouTube chapter timestamps. | `STRING` |
| `TS_ModelScanner` | Shows a readable summary of a model file. | `STRING` |
| `TS_ModelConverter` | Converts a model to another format or precision. | `MODEL` |
| `TS_ModelConverterAdvanced` | Advanced model converter with more options. | `STRING` |
| `TS_ModelConverterAdvancedDirect` | Converts a connected model directly from the graph. | `STRING` |
| `TS_CPULoraMerger` | Merges LoRA files into a base model on CPU and saves the result. | `STRING`, `STRING` |

### Small Graph Helpers

| Node | Plain Description | Outputs |
| --- | --- | --- |
| `TS_FloatSlider` | A convenient slider for decimal numbers. | `FLOAT` |
| `TS_Int_Slider` | A convenient slider for whole numbers. | `INT` |
| `TS_Smart_Switch` | Switches a graph between two inputs. | `*` |
| `TS_Math_Int` | Runs simple operations on whole numbers. | `INT` |

## Dependencies And Models

Some nodes work immediately after installation. Others need extra libraries or model files.

This most often applies to:

- Whisper and TS Super Prompt speech recording;
- Qwen;
- Silero TTS;
- BiRefNet;
- Demucs for music stem separation;
- video upscaling and frame interpolation.

If the ComfyUI console reports a missing dependency, run:

```bash
python -m pip install -r ComfyUI/custom_nodes/comfyui-timesaver/requirements.txt
```

Then restart ComfyUI.

Models are usually downloaded or selected separately. Not every node downloads models automatically, so the first run of a heavy node may pause or show a message about a required file.

## Beginner Tips

- Start with a small image or short audio file. It is easier to understand the result and faster to test.
- Change one setting at a time. This makes it clear which option changed the result.
- Watch input and output types. `IMAGE` connects to `IMAGE`, `AUDIO` connects to `AUDIO`, and text usually uses `STRING`.
- Many video workflows in ComfyUI are actually batches of images.
- If a heavy node uses too much memory, try a smaller frame size, fewer frames, or a lighter model.

## FAQ

### Nodes Do Not Appear In The Menu

Check that the folder is placed here:

```text
ComfyUI/custom_nodes/comfyui-timesaver
```

Restart ComfyUI and check the console. If there is an import error, install the dependencies from `requirements.txt`.

### `No module named ...`

A required Python package is missing. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the command from the `comfyui-timesaver` folder or provide the full path to `requirements.txt`.

### Microphone Does Not Work In TS Super Prompt

The browser must allow microphone access. This usually works on:

```text
http://localhost:8188
```

If ComfyUI is opened through a local network IP address, the browser may require HTTPS before allowing microphone access.

### Video Nodes Use Too Much Memory

Try reducing frame size, reducing the number of frames, or adding `TS_Free_Video_Memory` between heavy steps.

## Pack Structure

```text
comfyui-timesaver/
  nodes/              Python nodes
  js/                 ComfyUI frontend extensions
  doc/                documentation and helper scripts
  requirements.txt    dependencies
  pyproject.toml      pack metadata
  __init__.py         node loader
```

## For Developers

Quick checks after changes:

```bash
python -m compileall .
python -m pytest tests
```

If tests or dependencies are not available in the current environment, at least check import and ComfyUI startup. For frontend nodes, also check the browser console.

## Repository

```text
https://github.com/AlexYez/comfyui-timesaver
```

For a first ComfyUI experiment, build a simple graph, add one node from this pack, and observe how the result changes. Step by step, it becomes clearer which nodes are useful for each workflow.
