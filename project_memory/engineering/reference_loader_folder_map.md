---
name: reference_loader_folder_map
description: "Сканер моделей берёт карту «нода → папка» у сервера; таблица типов принципиально не поспевает"
metadata:
  type: reference
---

Сканер моделей берёт карту «нода → папка» у сервера; таблица типов принципиально не поспевает

Исходники темы: [_downloader_search.py](../../nodes/files/_downloader_search.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Правка 14.08.2026. Жалоба: «TS File Downloader при сканировании не отслеживает
модели в ноде Load Latent Upscale Model».

**Масштаб мимо жалобы.** Считано по `/object_info` живой установки: **49
установленных типов нод имеют виджет с именем модели, которого в
`TYPE_TO_FOLDER` нет**. Две из них — родные ноды ComfyUI
(`LatentUpscaleModelLoader` и `LowVRAMLatentUpscaleModelLoader`), и читают они
категорию `latent_upscale_models`, которой в таблице не было вовсе. Остальные —
из пакетов (GGUF-загрузчики, Advanced ControlNet, IPAdapter, ReActor, LTXV…).
Написанная руками таблица за этим не поспеет никогда.

**Как теперь.** Роут `/ts_downloader/loader_folders` в
[`nodes/files/_downloader_search.py`](../../nodes/files/_downloader_search.py)
строит карту по работающему серверу: выпадающий список загрузчика заполнен
`folder_paths.get_filename_list(<категория>)`, значит сами варианты и говорят,
из какой папки они. Сопоставление — `match_category()`: сперва точное
совпадение множеств, затем наименьший список, СОДЕРЖАЩИЙ варианты (так
отфильтрованный по `.gguf` список находит свою папку). На этой машине карта
покрывает 58 типов, ответ 3.5 КБ за ~0.15 с.

Тонкости, каждая — отдельная ловушка:

- **Карта ПО ВИДЖЕТАМ, а не по нодам.** `LTXAVTextEncoderLoader` держит
  `text_encoder` (→ `clip`) и `ckpt_name` (→ `checkpoints`). Одна папка на ноду
  отправила бы половину скачивания не туда.
- **Скачивать надо в ДИРЕКТОРИЮ, а не в имя ключа реестра.** `clip_gguf` и
  `unet_gguf` читаются из `models/clip` и `models/diffusion_models`,
  `ultralytics_bbox` — из `models/ultralytics/bbox`. Роут отдаёт путь
  относительно `models/` у самой полной директории категории.
- **Пустая категория не опознаётся** — пустое множество подмножество любого,
  такие пропускаются. Для них остаётся статическая таблица во фронтенде (в неё
  дописаны `latent_upscale_models` и `audio_encoders`).
- **Форматов описания входа два** (см. [reference_node_size_and_refresh](reference_node_size_and_refresh.md) §3),
  `_combo_options()` читает оба.
- Кэш карты в браузере сбрасывается по «R».

Проверка до/после на живом графе с `LatentUpscaleModelLoader`: было «No models
found in this workflow», стало строка `latent_upscale_models/ltx` с подпапкой.

Гарды: `tests/test_loader_folder_map.py` (разбор и сопоставление, без ComfyUI),
`tests/test_workflow_scan_loader_map.py` (сканер в браузере).
