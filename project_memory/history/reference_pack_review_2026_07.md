---
name: reference_pack_review_2026_07
description: "История: Whole-pack reviews 2026-07-03 (Ready) и 2026-07-15 (Needs Attention → всё исправлено в v9.77); что где живёт и что не перепроверять"
metadata:
  type: reference
  status: historical
---

История: Whole-pack reviews 2026-07-03 (Ready) и 2026-07-15 (Needs Attention → всё исправлено в v9.77); что где живёт и что не перепроверять

**Историческая запись, не описание текущей реализации и не действующее разрешение.** Даты и результаты ниже относятся к указанным сессиям. Пути `tests/`, `tools/`, `dist/` и `archive/` с артефактами — локальная лаборатория; они не гарантированы в чистом клоне.

Полный отчёт всегда: `code-review-report.md` в корне dev-репо (gitignored, только dev).

## Ревью №2 — 2026-07-15, v9.76 → исправлено в v9.77

Вердикт до фиксов: **Needs Attention** (7 подтверждённых багов). Метод: 6 шардов × (compliance + bugs) + сквозные architecture/platform агенты, каждая находка перепроверена отдельным валидатором. Все находки исправлены (`b34ad9f`, `869cd60`, `04ce6d3`, `2c4c74c`, `c20bda7`), `node_contracts.json` — diff пустой (API нод не тронут).

Что чинили (не регрессировать):
- **ts_smart_inpaint**: маска сводится к batch 1, а патч держит батч картинки → `_color_correct_patch` булев-индексировал `[B*h*w]` маской `[h*w]` → IndexError на любом B>1 (CC всегда включён). Alpha расширяется до B; B==1 бит-в-бит прежний.
- **ts_frame_interpolation**: буфер `[N,3,H,W]` крашился на RGBA (его реально отдаёт `TS_BGRM_BiRefNet` с `background="Alpha"`). Теперь альфа отбрасывается с warning.
- **ts_video_depth**: дефолтный `denoise_method="bilateral"` всегда звался с `rgb_chunk=None` → range-веса = 1 → обычный Gaussian вопреки tooltip. Теперь передаётся luma-guide, ресемплированный в разрешение глубины.
- **_qwen_engine unload**: вызывающие держат живые локалы `model/processor` в своём `finally`, поэтому pop+del+empty_cache не освобождал ничего («Model unloaded» врал). Фикс: `model.to("cpu")` в `_unload_cached_key` (освобождает CUDA-storage независимо от refcount) + обнуление локалов у вызывающих.
- **fingerprint_inputs**: style selector и prompt builder (seed>0) не учитывали внешние `styles.json` / `.txt` → стейл-промпт из кэша. Теперь mtime, как в `ts_qwen3_vl`.
- **sam_media_loader**: `/probe` (авто-триггер на restore workflow), `/preview`, `/upload`, `/preview_mask` декодили видео синхронно на event loop → фриз всего ComfyUI. Обёрнуто в `asyncio.to_thread`; `/probe` больше не открывает VideoCapture дважды.
- **ts_downloader**: `proxy_url` с `user:password@` логировался как есть → `_redact_proxy_url`.
- **JS**: слайдер snap-ался за `max` (int-слайдер выдавал 2147483648 > INT32) → re-clamp; ideogram `close()` не делал teardown открытого inline-редактора → утечка document-listener; hidden file inputs приведены к §12.5.11.

Новые shared-модули (одна причина — реальное переиспользование, не «чистота»):
- `nodes/image/_image_utils.py` — 7 хелперов, которые vitmatte тянул из birefnet; birefnet их re-export-ит (старые импорты живы).
- `nodes/_hf_download.py` — зеркала (HF_ENDPOINT) + kwarg-совместимость хаба для birefnet/vitmatte/video_depth. **`_qwen_engine` намеренно оставлен со своей реализацией** — у него доп. состояние `_snapshot_endpoint_supported` для download-оценок.
- `nodes/_deps.py` — см. [reference_node_import_paths](../engineering/reference_node_import_paths.md) (важный урок про `...`-импорты и тесты).
- Пресеты Qwen переехали в `_qwen_engine.load_presets(path=None)`; `ts_qwen3_vl._presets_path/_load_presets` остались тонкими делегатами (патчи и тесты работают).

Отклонено при валидации (**не «чинить» повторно**): class-level dict-кэши (`ts_music_stems`/`ts_silero_tts`/`ts_silero_stress`) — санкционированный паттерн, подтверждён по `lock_class`, теперь явно записан в CLAUDE.md §5; `width:1px` на hidden inputs lama/sam — off-screen там уже есть; «TypeError-асимметрия» HF-загрузчиков — опровергнута (хрупкие kwargs только у Qwen).

Тесты: `tests/test_review_regressions.py` (27 тестов; 20 краснеют на pre-fix коде). Известное падение окружения: `test_memory_load_status_is_short_for_button_label` — numba не принимает NumPy 2.5, воспроизводится и на чистом HEAD, к паку отношения не имеет.

## Ревью №1 — 2026-07-03, v9.74

Вердикт **Ready**. Все находки исправлены в `74ac53e` (+ ideogram-ревью тем же днём в `c45621e`, детали в [ts-ideogram-designer-node](../engineering/ts-ideogram-designer-node.md)):
- print→logging: ts_smart_inpaint fine-upscale, vendored motion_module/attention (module-level). Закреплено guard-тестом `tests/test_no_print_logging.py` (allowlist: только self-test ideogram helpers).
- DOM `computeSize` в prompt-builder / style-prompt / resolution-selector теперь ставится ТОЛЬКО в V1 (`if (!isV2)`); V2 идёт через getMinHeight/getMaxHeight (§12.5.1). При правке этих виджетов сохранять гейт.
- `video_depth_anything/utils/dc_utils.py` УДАЛЁН (мёртвый vendored, 0 импортов). `infer_video_depth` — НЕ мёртвый: документированный legacy-API для внешних скриптов; был сломан (str-device → AttributeError на device.type), починен coercion'ом `torch.device(device)`. Не удалять.
- `_patched_prompt_server` (sam helpers): race с реальным промптом — ПРИНЯТ и задокументирован в docstring (эффект только на progress-reporting). Не «чинить» сериализацией без нужды.
- TSDependencyManager: `functools.wraps` на guard-обёртке execute; кэш неудачного `import_optional` — намеренный (см. комментарий).

**Здоровые зоны, проверенные глубоко** (не перепроверять без причины): downloader (traversal guard, token host-allowlist), audio-helpers (stderr-drain, kill-таймеры), SAM/LaMa (bounded LRU локов, off-loop model load), qwen-engine, инжекторы, loader AST-аудит. Единственный `torch.load` — bgrm, `weights_only=True`; files/* на `safe_open`.
