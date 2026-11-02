---
name: engineering_index_files
description: "Файлы, батчи и скачивание моделей"
metadata:
  type: reference
---

# Файлы, батчи и скачивание моделей

Каждая строка — отдельная тема; начните с нужного hook.

- [reference_batch_files_and_wired_path](reference_batch_files_and_wired_path.md) — TS Batch Source «Files in folder» (переименование с псевдонимом) и сокет path у TS Video Loader / TS Audio Loader — как….
- [reference_batch_semantics](reference_batch_semantics.md) — Батч в ComfyUI — списки вместо циклов, почему превью появляются только в конце, почему писатель не может быть списковым….
- [reference_downloader_check_links](reference_downloader_check_links.md) — Anonymous Check Links button; bounded GET probes, public redirects, UI lifecycle and activation after restart.
- [reference_downloader_fixes](reference_downloader_fixes.md) — TS Files Downloader — root causes of the 2026-07-15 user bug reports (redownload loops, false OFFLINE, models-dir….
- [reference_downloader_gated_repos](reference_downloader_gated_repos.md) — TS Files Downloader — закрытый (gated) репозиторий HF распознаётся по X-Error-Code и показывается окном со ссылками;….
- [reference_downloader_loader_lookup](reference_downloader_loader_lookup.md) — TS Files Downloader — «скачана» значит «загрузчик графа её найдёт»; поиск через folder_paths.get_full_path, состояние….
- [reference_downloader_progress](reference_downloader_progress.md) — TS Files Downloader — прогресс живёт на сервере, а operation_id на странице; после F5 индикация молча пропадала.
- [reference_downloader_stop_and_folders](reference_downloader_stop_and_folders.md) — TS Files Downloader — почему остановка прогона не останавливала скачивания, почему папка предлагалась пустая, и как….
- [reference_hf_no_proxy](reference_hf_no_proxy.md) — «All mirrors failed. Invalid port: ':1]'» — запись [::1] в NO_PROXY (её пишет лаунчер владельца) роняет httpx внутри….
- [reference_hf_search](reference_hf_search.md) — Поиск ссылки на модель по имени файла (роут /ts_downloader/hf_search) — почему он ничего не находил и как устроен после….
- [reference_loader_folder_map](reference_loader_folder_map.md) — Сканер моделей берёт карту «нода → папка» у сервера; таблица типов принципиально не поспевает.
- [reference_workflow_model_scan](reference_workflow_model_scan.md) — Как TS Files Downloader собирает модели из workflow — properties.models первичны, загрузчики живут в сабграфах, кнопка….
