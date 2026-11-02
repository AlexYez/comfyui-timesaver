---
name: reference_missing_media_scanner
description: "Как фронтенд решает, что медиа-вход «пропал», и почему TS Audio Loader краснел после перезагрузки вкладки"
metadata:
  type: reference
---

Как фронтенд решает, что медиа-вход «пропал», и почему TS Audio Loader краснел после перезагрузки вкладки

Исходники темы: [_audio_helpers.js](../../js/audio/loader/_audio_helpers.js).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Смежное: [reference_pasted_media_fix](reference_pasted_media_fix.md), [reference_widgets_values_positional](reference_widgets_values_positional.md),
[reference_live_copy_sync](reference_live_copy_sync.md).

## Правило сканера (прочитано в исходниках фронтенда, не угадано)

`comfyui_frontend_package/static/assets/settingStore-*.js.map` →
`src/platform/missingMedia/missingMediaScan.ts`, там же `mediaPathDetectionUtil.ts`.
Для каждой ноды графа:

1. берутся ТОЛЬКО виджеты из `node.widgets` — тип `combo`;
   ⚠️ виджет, унесённый нашим `hideWidget` в `_tsHiddenWidgets`, сканер не видит
   ВООБЩЕ, и никакой красноты по нему не будет;
2. спецификация виджета должна нести `image_upload` / `video_upload` /
   `audio_upload` / `animated_image_upload` — то есть `upload=IO.UploadType.*` в
   схеме. Во всём паке такой вход ровно один: `source_path` у TS Audio Loader;
3. значение сравнивается со СПИСКОМ ОПЦИЙ этого виджета **буквально**. Снимается
   только хвост по `/\s+\[(input|output)\]$/`, то есть проверяются два написания:
   `sub/file.wav [input]` и `sub/file.wav`. Ни косые, ни регистр не приводятся;
4. не нашлось — `node.has_errors = true`, красная нода и «A required media input
   has no file selected». Значение `… [output]` уходит на отдельную проверку по
   истории генераций.

## Что из-за этого сломалось (замерено 2026-09-16 на живом сервере)

`folder_paths.recursive_search` называет файл в подпапке по-виндовому:
`ts_audio_loader_recordings\take.wav`. А значение виджета всюду пишется через
прямую косую — так его возвращает `/upload/image`, так его отдаёт маршрут
`/ts_audio_loader/upload_recording`, так оно лежит в файле workflow. Итог по
четырём случаям (все четыре проверены в браузере):

| значение виджета | после перезагрузки |
| --- | --- |
| `cyberpunk.mp3` | чисто |
| `cyberpunk.mp3 [input]` | чисто |
| `ts_audio_loader_recordings/take.wav [input]` | **КРАСНОЕ** |
| `ts_audio_loader_recordings\take.wav` | чисто |

То есть страдал ровно тот, кто записал дорожку с микрофона или держит аудио в
подпапке `input`. Файл при этом на месте, и прогон проходит: бэкенд
(`_normalize_selected_path` → `get_annotated_filepath`) понимает оба написания.

## Починка (16.09.2026, обе половины нужны)

- `_get_uploadable_media_options()` отдаёт список с прямыми косыми на любой ОС —
  то же решение и по той же причине, что в `ts_pasted_media_fix.py`.
- `js/audio/loader/_audio_helpers.js`: `normalizeSourcePathValue()` приводит к
  этому виду путь, уже сохранённый в чужом графе (иначе после смены списка
  краснели бы СТАРЫЕ графы, где значение выбрано из выпадашки с обратными
  косыми), а `rememberSourceOption()` кладёт свежезагруженный файл в
  `widget.options.values` — ровно как делает сам ComfyUI в своём upload-виджете,
  потому что `/object_info` фронтенд спрашивает при открытии страницы, а файл
  появляется позже.
- Чинить значение нужно в `loadedGraphNode`/setup: сканер идёт ПОСЛЕ загрузки
  графа, эти хуки — внутри неё.

Гард: `tests/test_audio_loader_missing_media.py` (бэкенд + живая проверка обоих
написаний в браузере).
