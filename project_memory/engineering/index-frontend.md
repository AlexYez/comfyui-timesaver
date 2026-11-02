---
name: engineering_index_frontend
description: "Фронтенд, виджеты и интерфейс"
metadata:
  type: reference
---

# Фронтенд, виджеты и интерфейс

Каждая строка — отдельная тема; начните с нужного hook.

- [dom_widget_controls](dom_widget_controls.md) — DOM-виджеты: подписи, сравнение, fullscreen и колесо мыши.
- [dom_widget_layout](dom_widget_layout.md) — DOM-виджеты: размеры, координаты, маска, курсор и производительность.
- [dom_widget_lifecycle](dom_widget_lifecycle.md) — DOM-виджеты: backend-сессии, загрузка файлов, full-bleed, рост ноды и слои интерфейса.
- [reference_dom_widget_pitfalls](reference_dom_widget_pitfalls.md) — DOM-виджеты: причины ошибок размеров, координат, масок, lifecycle и fullscreen.
- [reference_dom_widget_rehydrate](reference_dom_widget_rehydrate.md) — Pattern для interactive DOM widgets чтобы state переживал переключение workflow-вкладок ComfyUI без дублирования UI.
- [reference_drop_snapshot](reference_drop_snapshot.md) — DataTransfer умирает на первом await — снимок дропа снимается синхронно первой строкой обработчика (captureDrop в….
- [reference_extension_ids](reference_extension_ids.md) — Стабильные JS extension IDs пака — менять нельзя, на них завязаны сохранённые workflow (актуально на v9.5).
- [reference_fullscreen_overlay](reference_fullscreen_overlay.md) — Общий хелпер полноэкранного редактора js/_fullscreen.js — openFullscreenOverlay; используют Lama Cleanup и SAM Media….
- [reference_group_bypasser](reference_group_bypasser.md) — TS Group Bypasser — пульт по группам workflow; факты о группах в движке ComfyUI и почему состояние выводится, а не….
- [reference_guide_and_compare](reference_guide_and_compare.md) — TS Compare — шторка на ОДНОМ сложенном файле вместо двух плееров; ловушки живых замеров (кэш ComfyUI, следы оверлея).
- [reference_log_encoding_and_hotkeys](reference_log_encoding_and_hotkeys.md) — Две находки аудита 28.09.2026 — символ вне cp1251 в логе убивает поток очереди ComfyUI; readonly-якорь полноэкранного….
- [reference_modal_hotkeys](reference_modal_hotkeys.md) — Modal editors MUST park focus on a hidden textarea — event-order shielding against ComfyUI hotkeys is unwinnable….
- [reference_node_size_and_refresh](reference_node_size_and_refresh.md) — Два крючка ComfyUI, мимо которых живут наши ноды — размер после configure и обновление списков по «R».
- [reference_onboarding_wave1](reference_onboarding_wave1.md) — Волна 1 «пак объясняет себя» (v9.95+, коммит a00e9cf) — описания нод, встроенная справка js/docs, ru-описания,….
- [reference_pane_width_grows_node](reference_pane_width_grows_node.md) — ⚠️ Сужение ноды растит её вниз — ряд панели переносится и увеличивает min-content; потолок высоты читает высоту ноды.….
- [reference_playback_guard](reference_playback_guard.md) — Видео в ноде декодирует та же видеокарта, на которой считает ComfyUI — почему появился общий сторож воспроизведения и….
- [reference_playwright](reference_playwright.md) — ComfyUI Python имеет playwright + Chromium для headless GUI скриншотов и frontend-тестов; обязательно forced en-US….
- [reference_recreate_node](reference_recreate_node.md) — «Пересоздать ноду» — расширение холста js/utils/ts-recreate-node.js; как сохраняются провода, номер ноды, точки на….
- [reference_seed_widget_starvation](reference_seed_widget_starvation.md) — ⚠️ seed без control_after_generate — фронтенд добавляет лишний виджет, и ПОСЛЕДНИЙ виджет ноды теряет значение при….
- [reference_sidebar_tabs](reference_sidebar_tabs.md) — Галочки вкладок левого сайдбара ComfyUI в настройках пака (js/utils/ts-sidebar-tabs.js) — API, порядок, ловушка….
- [reference_sliders](reference_sliders.md) — TS Int/Float Slider — рабочий диапазон в node.properties, жёсткие границы в схеме, миграция старых ±1e9 и ловушка….
- [reference_subgraph_unpack](reference_subgraph_unpack.md) — Распаковка сабграфа падает от ОДНОЙ мёртвой ссылки в outputs; и даже успешная теряет значения повышенных виджетов.
- [reference_theme_system](reference_theme_system.md) — Единая тема пака js/_theme.js — токены --ts-*, один акцент TS_ACCENT (мягкий фиолетовый), серая база из переменных….
- [reference_tidy_layout](reference_tidy_layout.md) — Команда «Разложить аккуратно» — крючки меню холста, раскладка по слоям и почему размер ноды нельзя брать из node.size.
- [reference_typing_vs_hotkeys](reference_typing_vs_hotkeys.md) — Пока фокус в поле ввода, простые клавиши — это текст; горячей может быть только комбинация с Ctrl/Cmd (js/_keys.js).
- [reference_widgets_values_positional](reference_widgets_values_positional.md) — ⚠️ hideWidget удаляет виджет из node.widgets, а widgets_values позиционный — старые workflow сдвигаются; миграция в….
