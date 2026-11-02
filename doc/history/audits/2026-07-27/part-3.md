# Ревью 2026-07-27, v9.87: часть 3

Исторический материал. [Оглавление](README.md).

## Suggestion

**Мёртвый код и недостижимые ветки**

24. `nodes/image/ts_restore_from_crop.py:47-55` — ветка `else: return` недостижима: ранний возврат при `blur_amount <= 0` делает следующее условие тавтологией.
25. `nodes/image/ts_crop_to_mask.py:190-193` — «коррекция для изображения меньше тайла» переприсваивает уже гарантированный ноль (ветка достижима, но бессмысленна; реальную работу делает `F.pad` ниже).
26. `nodes/image/ts_image_resize.py:14-19, 164-165` — `TF` и `InterpolationMode` импортируются и не используются; `TORCHVISION_AVAILABLE` гейтит `antialias`, родной параметр `torch.nn.functional.interpolate`. Практического вреда нет: torchvision — жёсткая зависимость ядра, флаг всегда истинен.
27. `js/_dom_widget.js:33` — экспортируемый `isNodesV2` не импортирует никто; три идентичные приватные копии (`_lama_helpers.js`, `ts-style-prompt.js`, `ts-animation-preview.js`) дублируют его. Копия в `_audio_helpers.js` расходится **намеренно** и задокументирована с обеих сторон — её трогать нельзя.
28. `js/ideogram/_ideogram_editor.js` — собственная реализация полноэкранного оверлея (~45 строк) вместо `openFullscreenOverlay`, хотя докстринг `js/_fullscreen.js` называет Ideogram потребителем и требует «Do NOT reinvent the overlay per node». Функционально редактор исправен: парковка фокуса и защита Ctrl+Z свои и работают. Цена — единственный редактор без унифицированной кнопки закрытия и риск расхождения.
29. `nodes/llm/ts_qwen3_vl.py:70-92` и `nodes/llm/super_prompt/_qwen.py:353-375` — 23 строки (`_GENERATION_PARAM_ALIASES`, `_UNSUPPORTED_GENERATION_PARAMS`, `_normalize_generation_params`) совпадают побайтово; оба файла уже импортируют общий движок, где этих помощников нет. Добавление алиаса в один файл молча оставит вторую ноду отвергать тот же пресет.
30. `nodes/image/sam_media_loader/_sam_media_helpers.py:406-409` — единственный в паке кросс-категорийный импорт приватного модуля (`from ...audio.loader._audio_helpers import ...`). Импорт отложенный и обёрнут в try/except, но переименование `_probe_media` тихо превратит звук в тишину (ошибка демотирована до warning).
31. `nodes/llm/ts_super_prompt.py` — 208-строчный compat-шим, ре-экспортирующий ~144 символа (45 приватных) без потребителей в поставляемом коде; загрузчик импортирует его при каждом старте. Цена ниже заявленной: при поломке импорта модуль получает статус SKIPPED, а не ERROR, и `TS_SuperPrompt` продолжает грузиться из настоящего пути. Единственный доказанный потребитель — собственные тесты пака.
32. `js/_theme.js:427-430` — `resetThemeColors()` не импортирует никто, поэтому `colorCache` невозможно инвалидировать. Из-за этого canvas-отрисовка (волновая форма аудио, канвас Ideogram) после переключения светлой/тёмной темы продолжает рисовать прежними цветами до перезагрузки страницы, хотя §12.6 требует поддержки светлой темы.
33. `nodes/text/ts_prompt_builder.py:35-46` и `nodes/text/ts_style_prompt_selector.py:32-43` — fallback-ветка со сканированием `folder_paths` недостижима (`parents[2]` для существующего модуля всегда каталог); из-за этого мёртв и сам `import folder_paths`.
34. `nodes/text/ts_prompt_builder.py:219-221` — роут `/ts_prompt_builder/files` не имеет вызывающих; `/state` уже отдаёт те же данные. Удаление — небольшое изменение публичной HTTP-поверхности.
35. `nodes/audio/loader/_audio_helpers.py:489-503` — `_audio_duration_seconds()` и `_crop_audio_input()` не вызываются нигде.
36. `nodes/video/ts_video_upscale_with_model.py:33` — `upscale_methods = _UPSCALE_METHODS`, остаток V1; схема использует константу напрямую, а ядро читает такой атрибут только из `INPUT_TYPES` самого класса.
37. `nodes/image/lama_cleanup/ts_lama_cleanup.py:153-156` — ветка fallback по `session_id` недостижима: файл вида `{session}.png` не создаёт ни один путь (все писатели используют `{safe}_{tag}_{nanos}.png`).
38. `nodes/image/lama_cleanup/_lama_helpers.py:183-191` — `_get_uploadable_image_options` объявлен в `__all__`, но не вызывается (остаток отказа от штатного `IMAGEUPLOAD`).
39. `nodes/image/lama_cleanup/_lama_helpers.py:364-372` — `_to_input_annotation` без вызовов (одноимённые функции в аудио- и SAM-модулях используются, эта — нет).
40. `nodes/image/sam_media_loader/_sam_media_helpers.py:75-78` — `_comfy_utils` импортируется и не читается (соседние `_comfy_sd`/`_comfy_model_management` используются).
41. `nodes/image/ts_bgrm_birefnet.py:228-238` — `_progress_pulse` не читает параметры `cap_step` и `interval`, при этом все пять вызовов вычисляют и передают `cap_step`.
42. `nodes/conditioning/ts_multi_reference.py:321-356` — параметр `mask` и вся ветка композита недостижимы (единственный вызов передаёт один аргумент; комментарий подтверждает намеренность).
43. `nodes/conditioning/ts_multi_reference.py:67` — `_IMAGE_SLOT_COUNT` не используется (слоты захардкожены).
44. `nodes/ideogram/ts_ideogram_designer.py:24` — неиспользуемый `from pathlib import Path`.
45. `nodes/files/ts_downloader.py:628, 655, 673-681` — `used_get_probe` и `status_code` возвращаются, но не читаются.
46. `nodes/llm/_qwen_engine.py:701-715` — `QwenEngine.apply_chat_template` не вызывается (оба места используют `..._no_thinking`).
47. `nodes/llm/super_prompt/ts_super_prompt.py:135-151` — `_attached_image_as_tensor` не вызывается и не ре-экспортируется шимом.
48. `nodes/llm/super_prompt/_voice.py:127-133` — `_get_ffmpeg_executable` дублирует `get_ffmpeg_executable()` из общего движка. **Удалять нельзя** — шим его ре-экспортирует; корректная чистка — сделать его ре-экспортом.
49. `nodes/llm/super_prompt/_voice.py:120-124` — `_resolve_voice_model` игнорирует `requested_model` (`_ = requested_model`). Видимой пользователю ошибки нет: фронтенд вычисляет модель по тому же правилу. Остаётся latent-несоответствие контракта для сторонних HTTP-клиентов.
50. `nodes/llm/super_prompt/_helpers.py:18` — неиспользуемый `import folder_paths`.
51. `nodes/llm/super_prompt/_voice.py:19, 53` — неиспользуемые `threading` и `VOICE_MODEL_CACHE`.
52. `js/text/ts-prompt-builder.js:380-393` — класс `is-dragging` переключается, но ни в одном стиле пака не определён: подсветка перетаскивания не работает.
53. `js/text/ts-style-prompt.js:938-984` — `keyAnchor` создаётся и монтируется, но никогда не получает фокус (и не имеет `readOnly`/`tabIndex`), хотя комментарий рядом утверждает, что он «парсит горячие клавиши графа». Защита фактически не работает, элемент лишь занимает место в tab-порядке.
54. `js/text/ts-style-prompt.js:50-65` — ключ локализации `closeTitle` объявлен в обоих словарях и не читается (кнопки закрытия у модалки нет).
55. `js/text/ts-style-prompt.js:524, 1205` — `_tsStyleSelectorInitialized` пишется и сбрасывается, но не читается; реальный guard использует другое условие.

**Утечки ресурсов**

56. `nodes/video/ts_animation_preview.py:95-97` — временный WAV создаётся до кодирования видео, а его удаление находится в блоке, достижимом лишь после успеха. При отсутствии рабочего кодировщика H.265 файл (мегабайты) остаётся в temp на каждый прогон.
57. `nodes/image/sam_media_loader/_sam_media_helpers.py:568-593` — при обрыве загрузки осиротевший `.part` остаётся в `input/ts_sam_media/`: обработчик `except` чистит `saved_path`, который на тот момент ещё `None`. Ветка 413 чистит корректно. Рост ограничен — повтор с тем же именем перезапишет файл. *(Кандидат был отброшен в v9.76; теперь подтверждён с точной цепочкой.)*

**Осознанные компромиссы (не дефекты, зафиксировано для протокола)**

58. `validate_inputs(cls, **_) -> True` в `ts_image_resize.py:123-125`, `ts_rtx_upscaler.py:82-84` и др. — это **не** мёртвый код. Наличие `**kwargs` включает в ядре `validate_has_kwargs`, из-за чего подавляется штатная проверка диапазонов и принадлежности combo-списку для всех входов ноды. Приём применяется намеренно, чтобы старые воркфлоу загружались (см. комментарий в `ts_qwen3_vl.py:39`). Цена — пользователь не получает внятной ошибки валидации; выполнение защищено собственными клампами в `execute`. Удалять нельзя без осознанного решения.
59. `nodes/files/ts_downloader.py:39-66` — чтение `pyproject.toml` при импорте ради User-Agent. Формально §13, практически безвредно: обёрнуто в try/except с фолбэком `"dev"`, стоимость — один разбор небольшого файла на запуск.

---

## Отклонено при валидации (для прозрачности)

- **Клик по развёрнутому превью стиля сбрасывает выбор** — намеренное поведение, задокументированное в трёх местах кода и в подсказке схемы («или пробел, если стиль не выбран»).
- **Отсутствие `torch.no_grad()` в `ts_video_upscale_with_model.py`** — ядро ComfyUI оборачивает каждый `execute()` в `torch.inference_mode()` (`execution.py:746`); родная нода апскейла ядра написана точно так же.
- **Опциональные импорты через голый try/except** (`ts_silero_stress.py`, `ts_animation_preview.py`, `_audio_helpers.py`, `ts_image_resize.py`, `ts_downloader.py`, `_image_utils.py`) — все деградируют корректно и проверяются в рантайме; §14 применяется в паке непоследовательно, поэтому это вопрос единообразия, а не дефект конкретного файла.
- **`validate_inputs` как «мёртвый код»** — отклонено дважды независимыми валидаторами; переклассифицировано в п. 58.

## Проверено и чисто

Загрузчик и его фильтр обнаружения, разделение `ImportError` → SKIPPED против прочих → ERROR, идемпотентность `wrap_node_runtime`; оба инжектора (защита от повторного применения, математика `res_2s` и `bong_tangent` перепроверена); защита от path traversal в загрузчике, конвертерах и view-роутах; отсутствие `shell=True`, хардкода абсолютных путей и литералов `cuda`; отсутствие секретов в логах и во фронтенде; дисциплина клонирования тензоров и сохранения батча; `fingerprint_inputs` нигде не константен; извлечение аудио (проверка `returncode`, таймауты, отсутствие взаимоблокировок пайпов, снятие процессов); мьютекс общей модели Qwen и пути выгрузки VRAM; стабильность идентификаторов расширений; подключение темы, локализации и очистки слушателей в остальных GUI-нодах; отсутствие осиротевших модулей (у каждого `_`-префиксного модуля есть импортёр).

## Покрытие

Прочитано агентами: весь `nodes/` (кроме vendored `video_depth_anything/`, `frame_interpolation_models/`), весь `js/`, `__init__.py`, `ts_dependency_manager.py`, `CLAUDE.md`, `doc/`. Тесты и линтеры не запускались — ревью read-only по ТЗ. Рекомендуемая проверка после фиксов: `python -m compileall .` + `python -m pytest tests` под ComfyUI Python, затем живой прогон `tests/test_all_nodes_roundtrip.py`.

---
---

# АРХИВ: предыдущий цикл ревью (v9.76 → закрыт в v9.77)

> Сохранено целиком: файл не отслеживается git, поэтому история существует только здесь.

# Whole-Project Code Review — comfyui-timesaver v9.76

**Вердикт: Needs Attention** — критических дефектов нет, но подтверждено 7 реальных багов (2 детерминированных краша на реалистичных входах, неработающий VRAM-unload, залипание кэша, фриз event loop) плюс 4 policy-нарушения и 2 архитектурных долга.

> **Статус (2026-07-15, v9.77): ВСЕ находки исправлены.** M1–M7, m1–m10 и архитектурные улучшения 1–5 реализованы; API нод не менялся (`node_contracts.json` — diff пустой, 61 нода). Добавлен `tests/test_review_regressions.py` (27 тестов; 20 из них краснеют на pre-fix коде, остальные 7 фиксируют неизменность поведения). Ключевые решения: `_qwen_engine` оставлен со своей HF-реализацией (у него доп. состояние для download-оценок) — общий `nodes/_hf_download.py` подключён к birefnet/vitmatte/video_depth; `nodes/_deps.py` введён потому, что `from ...ts_dependency_manager` резолвится в ComfyUI, но ломает тестовую загрузку `nodes.*` как top-level пакета.

Ревью 2026-07-15, Claude Fable 5. Процесс: 6 шардов × (compliance + bugs) + сквозные architecture/platform агенты → каждая находка перепроверена отдельным валидирующим агентом по исходнику. Неподтверждённое отброшено (см. «Отклонено при валидации» внизу). API нод заморожен — все фиксы ниже его не трогают.

## Critical

Нет.

## Major

### M1. Smart Inpaint: детерминированный краш при батче B > 1
`nodes/image/ts_smart_inpaint.py:761-776` (корень — `_color_correct_patch`, `:482-489`)
- **Суть:** пайплайн полностью батчевый, но маска принудительно сведена к батчу 1 (`m = m[:1]`, `:929-936`). В `_color_correct_patch` булев индекс `preserved` имеет длину `h*w`, а `ref_flat` — `B*h*w` → `IndexError` при B>1. Color-correct всегда включён.
- **Фикс:** расширить `pixel_alpha_crop` до B перед вызовом CC, либо проверять ведущую размерность alpha в guard.

### M2. Frame Interpolation: краш на RGBA-входе (4 канала)
`nodes/video/ts_frame_interpolation.py:551,554` (+ `_prepare_frame`, `:304`)
- **Суть:** выходной буфер жёстко `torch.empty((N, 3, H, W))`; RGBA-кадр даёт shape-mismatch. Реалистичный продюсер RGBA есть в паке: `TS_BGRM_BiRefNet` при `background="Alpha"`.
- **Фикс:** `images = images[..., :3]` в начале execute.
