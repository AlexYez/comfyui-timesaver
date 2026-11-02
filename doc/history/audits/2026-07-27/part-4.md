# Ревью 2026-07-27, v9.87: часть 4

Исторический материал. [Оглавление](README.md).

### M3. Video Depth: дефолтный «bilateral» denoise — на самом деле обычный Gaussian
`nodes/video/ts_video_depth.py:1019` (внутри — `:257-294`)
- **Суть:** при `rgb_chunk=None` range-веса становятся `torch.ones_like(...)` → чистый пространственный Gaussian, ноль edge preservation, вопреки tooltip.
- **Фикс:** передавать low-res RGB guide либо считать range-веса от depth-патчей.

### M4. Qwen engine: `unload_after_generation` не освобождает VRAM
`nodes/llm/_qwen_engine.py:731-742` + `ts_qwen3_vl.py:587-598` + `super_prompt/_qwen.py:592-602`
- **Суть:** `_unload_cached_key` вызывается из `finally`, пока локалы `model, processor` живы — refcount не обнуляется; лог врёт «Model unloaded».
- **Фикс:** обнулять локальные ссылки до `gc.collect()/empty_cache()`.

### M5. Stale cache: fingerprint_inputs игнорирует внешние файлы
`nodes/text/ts_prompt_builder.py:279-287`, `nodes/text/ts_style_prompt_selector.py:155-157`
- **Фикс:** добавить в fingerprint mtime внешних файлов по образцу `ts_qwen3_vl.py:286-290`.

### M6. SAM Media Loader: блокирующий видео-декод замораживает весь ComfyUI
`nodes/image/sam_media_loader/_sam_media_helpers.py:588-626`, `:629-660`
- **Фикс:** обернуть декод в `await asyncio.to_thread(...)`; убрать двойное открытие VideoCapture.

### M7. Downloader: raw proxy_url с возможными кредами уходит в лог
`nodes/files/ts_downloader.py:140`
- **Фикс:** логировать замаскированный URL.

## Minor

m1. Film Grain: жёсткий module-level `import kornia.filters` (`ts_film_grain.py:5`).
m2. Ad-hoc optional-импорты вместо `import_optional` (`ts_video_upscale_with_model.py:21-25`, `ts_rtx_upscaler.py:10-15`).
m3. Video Upscale: memory-probe читает GPU 0 вместо выбранного устройства (`:213-215`).
m4. Слайдеры: snap после clamp может выкинуть значение за `[min, max]` (`_slider_helpers.js:189-190`).
m5. Music Stems: device mismatch при CUDA-входе (`ts_music_stems.py:197,209-216`).
m6. Ideogram: утечка capture-listener при teardown с открытым inline-редактором (`_ideogram_editor.js:1136`).
m7. Ideogram: file inputs скрыты `display:none` вместо паттерна §12.5.11.
m8. Audio Loader: file input скрыт ровно запрещённым способом (`_audio_helpers.js:80`).
m9. Архитектура: приватные helpers одной ноды импортируются другой (`super_prompt/_qwen.py:50`, `ts_matting_vitmatte.py:32-40`).
m10. Doc drift: CLAUDE.md §5 vs ARCHITECTURE.md (class-dict pattern).

## Архитектурные улучшения (по убыванию эффекта)

1. `nodes/image/_image_utils.py` — вынести 7 общих helpers из `ts_bgrm_birefnet.py`.
2. Перенести `_load_presets` в `_qwen_engine.py`.
3. Общий HF-download helper (`nodes/_hf_download.py`) — реальный gap: зеркала есть только у Qwen/birefnet.
4. Единый паттерн optional-импортов (kornia, spandrel, nvvfx).
5. Синхронизировать документы правил (m10 + §12.5.11).

## Отклонено при валидации

- **Class-level кэши** — санкционированный ARCHITECTURE.md паттерн №2.
- **`width:1px` на hidden inputs lama/sam** — off-screen присутствует; шум.
- **«4 из 5 HF-загрузчиков падают TypeError»** — опровергнуто.
- **Краш CUDA-probe на CPU-only** — блок под `torch.cuda.is_available()`.
- Плюс ~десяток низкоуверенных кандидатов (в т.ч. нулевой tile при экстремальном `divide_by`, orphan `.part`) — **два из них подтверждены в ревью v9.87, см. пп. 5 и 57**.

## Покрытие

Прочитано агентами: весь `nodes/` (кроме vendored), весь `js/`, `__init__.py`, `ts_dependency_manager.py`, `requirements.txt`, конфиги. Тесты/линтеры не запускались.
