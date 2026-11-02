---
name: engineering_index_runtime
description: "Загрузчик, зависимости, безопасность и проверки"
metadata:
  type: reference
---

# Загрузчик, зависимости, безопасность и проверки

Каждая строка — отдельная тема; начните с нужного hook.

- [project_api_mix](project_api_mix.md) — V3 API и module-level состояние; причина запрета записи в заблокированный класс ноды.
- [project_builtin_updater](project_builtin_updater.md) — Отложенная идея — вкладка Updates в настройках пака с уведомлением о новой версии; разобранные варианты, риски и….
- [project_loader](project_loader.md) — Рекурсивный loader, runtime guard и тихий старт; подробный отчёт включается TS_VERBOSE_STARTUP.
- [reference_artius_integration](reference_artius_integration.md) — История двух интеграций Artius со снятой TS Image Studio; старые маршруты не являются текущим API.
- [reference_comfy_scanner_rules](reference_comfy_scanner_rules.md) — История вердиктов Registry и границы локальной диагностики; не спецификация внешнего сервиса.
- [reference_comfyignore](reference_comfyignore.md) — История упаковки comfy-cli и отличия zip от checkout; .comfyignore в текущем паке отсутствует.
- [reference_comfyui_python](reference_comfyui_python.md) — Для проверок нужен Python окружения ComfyUI; абсолютный D:-путь — пример установки владельца.
- [reference_device_policy](reference_device_policy.md) — Пак не хардкодит CUDA — устройство через mm.get_torch_device(); регрессия охраняется test_no_new_cuda_hardcodes….
- [reference_fast_test_suite](reference_fast_test_suite.md) — История ускорения локальных тестов 2026-09-28; не разрешение запускать полный набор.
- [reference_ffmpeg_resolver](reference_ffmpeg_resolver.md) — Где пак берёт ffmpeg (свой бинарник из imageio-ffmpeg, не PATH) и почему фоллбэка на PyTorch быть не может.
- [reference_free_memory](reference_free_memory.md) — Кнопка «Освободить память» в верхней панели — флаги ComfyUI + реестр кэшей моделей пака (14 владельцев), блокировки….
- [reference_krea2_text_fusion](reference_krea2_text_fusion.md) — Krea 2 сводит 12 слоёв текстового энкодера одним Linear(12→1) — и как в ComfyUI умножить вес патчем.
- [reference_lazy_inputs](reference_lazy_inputs.md) — Ленивые входы ComfyUI — как заставить ноду не считать невыбранную ветку, чем отличается None от «не подключён» и почему….
- [reference_live_copy_sync](reference_live_copy_sync.md) — Разные checkout разработки и ComfyUI: сверка содержимого и история ошибок доставки исправлений.
- [reference_locked_loaders](reference_locked_loaders.md) — Формат .tsmodel и шесть загрузчиков TS/Loaders — замок без криптографии, чтение тем же mmap-путём, что у ComfyUI.
- [reference_lora_loader](reference_lora_loader.md) — TS LoRA Loader — стопка LoRA одной нодой; разворот в родные загрузчики и грабли с фокусом в поле силы.
- [reference_macos_compat](reference_macos_compat.md) — Совместимость пака с macOS — что измерено на раннере, что починено и что осталось известным ограничением.
- [reference_manager_visibility](reference_manager_visibility.md) — Built-in Manager button depends on launch flag and cached system stats; helper now enables it.
- [reference_missing_media_scanner](reference_missing_media_scanner.md) — Как фронтенд решает, что медиа-вход «пропал», и почему TS Audio Loader краснел после перезагрузки вкладки.
- [reference_music_stems_roformer](reference_music_stems_roformer.md) — TS Music Stems на BS-RoFormer SW и Mel-Band RoFormer — контракты моделей, чанкование, остаточный стем, замеры fp16, все….
- [reference_nag](reference_nag.md) — NAG работает только там, где есть настоящий cross-attention — обзор пяти архитектур и как патчить модуль в ComfyUI.
- [reference_node_import_paths](reference_node_import_paths.md) — `from ...ts_dependency_manager` резолвится в ComfyUI, но ломает тесты; использовать nodes/_deps.py. Патчить надо место….
- [reference_optional_xformers_warnings](reference_optional_xformers_warnings.md) — DINOv2 optional xFormers warnings downgraded to debug; widgetInputs import absent in our pack.
- [reference_pasted_media_fix](reference_pasted_media_fix.md) — Единственная заплатка пака на ядро ComfyUI — штатный Load Image начинает видеть картинки в подпапках input (вставленные….
- [reference_proactor_guard](reference_proactor_guard.md) — Заплатка на asyncio Windows — ложная ConnectionResetError 10054 при закрытии websocket.
- [reference_schedulers_injector](reference_schedulers_injector.md) — Side-effect модули, регистрирующие доп. schedulers (beta57, bong_tangent) и samplers (res_2s) в comfy.samplers через….
- [reference_security_boundaries](reference_security_boundaries.md) — Границы «сеть → диск» после аудита 11.08.2026 — куда пак вправе писать, кому отдаёт токен и что показывает по HTTP.
- [reference_template_workflow_building](reference_template_workflow_building.md) — Как собирать большие workflow из встроенных шаблонов ComfyUI только родными нодами — общий размер на API-ноды через….
- [reference_tests](reference_tests.md) — Локальная тестовая лаборатория: исторические замеры; актуальные команды — в operations/testing.
- [reference_verification](reference_verification.md) — Каноника проверок находится в operations/testing; здесь сохранены прежние сценарии и их причины.
- [tests_laboratory](tests_laboratory.md) — История локальной тестовой лаборатории: фикстуры, окружение, импорты и измеренные прогоны.
- [tests_retired_studio](tests_retired_studio.md) — История тестов снятой TS Image Studio и ловушки стабов/параллельных прогонов.
