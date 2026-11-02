# Карта подсистем

Найдите публичную ноду в справке, затем её private engine; не загружайте весь проект для локальной задачи.

Публичные имена и Python-файлы перечислены в
[контрактах](../../tests/contracts/node_contracts.json), пользовательские параметры —
в [справке нод](../nodes/README.md). Таблица ниже — навигация по владельцам,
а не второй каталог всех полей и defaults.

| Подсистема | Основной код | Граница ответственности |
| --- | --- | --- |
| Загрузка и зависимости | [__init__.py](../../__init__.py), [TSDependencyManager](../../ts_dependency_manager.py) | Discovery, registration, guard; не inference. |
| Общие пути, отмена и память | [nodes/_shared.py](../../nodes/_shared.py), [ffmpeg](../../nodes/_ffmpeg.py) | Общие маленькие контракты для разных нод. |
| PyTorch LLM | [Qwen engine](../../nodes/llm/_qwen_engine.py), [Qwen node](../../nodes/llm/ts_qwen3_vl.py) | Каталог/скачивание, load/unload, vision/text generation. |
| Prompt/voice интерфейс | [super_prompt](../../nodes/llm/super_prompt/), [super_prompt_rt](../../nodes/llm/super_prompt_rt/) | Две публичные версии с разными inference backends. |
| LiteRT | [LiteRT engine](../../nodes/llm/_litert_engine.py) | Gemma, .litertlm, WebGPU/native runtime, собственная память. |
| Whisper | [shared engine](../../nodes/_whisper_engine.py), [TS Whisper](../../nodes/audio/ts_whisper.py) | Общая речь для Whisper-ноды и prompt-пайплайна. |
| Медиа | [video/media](../../nodes/video/media/), [audio/loader](../../nodes/audio/loader/) | Probe/decode/encode, свои routes, preview и loader. |
| HDR LTX | [video/hdr](../../nodes/video/hdr/) | Цветовые кривые, EXR I/O, FP32 decode; математика отдельно от UI-нод. |
| H3 latent upscale | [latent_upscale](../../nodes/video/latent_upscale/) | Native H3 latents, spatial upscale, resample и отдельная Regenerate-ветка. |
| DLSS | [dlss](../../nodes/video/dlss/) | v9 in-process C ABI; дополнительный v21 через отдельный официальный CLI. |
| NVIDIA VFX | [TS RTX Upscaler](../../nodes/video/ts_rtx_upscaler.py) | Optional NVIDIA SDK, cache сессии и отмена. |
| Удаление фона/маски | [LaMa](../../nodes/image/lama_cleanup/), [SAM loader](../../nodes/image/sam_media_loader/) | Свои model helpers, маршруты и интерактивный frontend. |
| Глубина | [общий depth engine](../../nodes/_depth_core.py) | Общая модель для image/video половин; vendor код в video_depth_anything. |
| Batch | [источник](../../nodes/utils/ts_batch_source.py), [загрузка](../../nodes/image/ts_batch_load_image.py), [запись](../../nodes/files/ts_batch_write.py) | Список заданий и one_per_run; не путать list с tensor batch. |
| Files Downloader | [backend](../../nodes/files/ts_downloader.py), [frontend](../../js/files/ts-files-downloader.js) | Перечень файлов, загрузка, проверка ссылок, прогресс/отмена. |
| Prompt Builder | [TS Prompt Builder](../../nodes/text/ts_prompt_builder.py) | Чтение *.txt из nodes/prompts, порядок блоков и сборка по seed/config_json. |
| Ideogram | [ideogram](../../nodes/ideogram/), [JS](../../js/ideogram/) | Сценовый layout и собственный редактор; не Image Studio. |
| Locked loaders | [loaders](../../nodes/loaders/) | Три публичных загрузчика, общий private формат/clock. |

## Память моделей

PyTorch ModelPatcher, собственный PyTorch cache и LiteRT WebGPU — разные
владельцы. LiteRT unload по умолчанию выбран потому, что память этого backend
не учитывается ComfyUI как обычная CUDA-модель: разогретый native engine может
мешать следующему sampler. Не переносить эти измерения на все устройства;
актуальное поведение задаёт _litert_engine и вход unload соответствующей ноды.
Общий реестр освобождения памяти находится в nodes/_shared.py, маршруты кнопки —
[nodes/ts_memory_routes.py](../../nodes/ts_memory_routes.py). Сам модуль routes
не является ещё одной публичной нодой.

## Медиа и H3

TS Video Saver пишет файл на сервере; открытая вкладка нужна для интерфейса,
не для самого encode. Зависимость результата от фокуса браузера — дефект,
а не ожидаемая модель работы. Декодированные картинки передаются IMAGE-батчем,
а финальный soundtrack задаётся через AUDIO и fps/число кадров согласованно.

TS Latent Upscale и TS H3 Regenerate — разные пути. Regenerate Model получает
чистую H3 base до генерационных LoRA, подключает свои patches/Sharp/reference
таблицу; Regenerate выполняет refinement и может принять экспериментальный
первый кадр через обычный video VAE. Reference checkpoint в текущей реализации
читается для небольшой AdaLN reference table, не загружается целиком на GPU.
Конкретные шаги, размеры, overlap/denoise и ограничения — в справке соответствующих
нод и коде, не в этом обзоре. Chunked mode отмечен экспериментальным; длинный
ролик не равен независимой обработке кусочков без пограничных условий.

## Данные и ассеты

Не все папки nodes/ содержат ноды. frame_interpolation_models и
video_depth_anything — vendor implementations; prompts, styles, luts и JSON
пресетов — данные. Их пути вычисляются относительно модуля или folder_paths,
не зависят от каталога запуска терминала.

- [Qwen presets](../../nodes/qwen_3_vl_presets.json) используются существующими
  LLM-нодами. Имена/порядок пресетов участвуют в старых workflow; новые имена
  не добавляют вместо корректировки существующих без запроса.
- [Стили](../../nodes/styles/styles.json): prompt — модификатор, заканчивающийся
  на запятую с пробелом для подстановки перед пользовательским prompt. Новые
  поля JSON должны пройти whitelist backend-нормализатора.
- [Prompt data](../../nodes/prompts/) — текущий TS Prompt Builder сканирует текстовые
  файлы непосредственно в этой папке и использует порядок блоков из config_json
  или ts-prompt-builder-config.json. Прежние заметки о папках с семантической
  картой/_packs/_scene описывают другую реализацию и не подтверждаются нынешним
  ts_prompt_builder.py. Пользовательские данные не удаляются лишь потому, что
  отсутствуют в другой копии.
- [LUT](../../nodes/luts/) и [Film Emulation](../../nodes/image/film_emulation/)
  имеют свои форматы/legacy preset names; визуальная близость не разрешает
  незаметно поменять семантику сохранённого пресета.

Исторические измерения и причины редких исключений ищутся по владельцу через
[индекс памяти](../../project_memory/MEMORY.md), а не по копии абсолютного пути
на компьютере автора.
