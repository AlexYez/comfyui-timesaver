---
name: engineering_index_video
description: "Видео, звук, LTX, H3 и апскейл"
metadata:
  type: reference
---

# Видео, звук, LTX, H3 и апскейл

Каждая строка — отдельная тема; начните с нужного hook.

- [dlss_legacy_v5](dlss_legacy_v5.md) — История v5: исходный протокол, раскладка, ограничения распространения и вердикты Registry.
- [dlss_runtime_history](dlss_runtime_history.md) — История v9/v11: C ABI, память, битые плитки, возврат на v9 и фактура кожи.
- [dlss_v21](dlss_v21.md) — Дополнительный внешний v21: официальное приложение, CLI, форматы транспорта и проверенные ограничения.
- [ltx_hdr_integration](ltx_hdr_integration.md) — LTX HDR: локализация, workflow, проверки и историческое состояние установки.
- [ltx_hdr_pipeline](ltx_hdr_pipeline.md) — LTX HDR: две технологии, карта модулей, численные параметры и цвет/EXR.
- [reference_dlss_upscaler](reference_dlss_upscaler.md) — DLSS: текущий v9, внешний v21 и отдельные журналы старых протоколов и GPU-ошибок.
- [reference_h3_audio_inject](reference_h3_audio_inject.md) — TS H3 Audio Inject — липсинк MiniMax H3 закреплением аудио-потока латента; ⚠️ латент хранит НЕмасштабированное аудио….
- [reference_h3_regenerate_character](reference_h3_regenerate_character.md) — H3 Regenerate first-frame guidance at final resolution and continuation with locked overlap; installed workflow and….
- [reference_latent_upscale_h3](reference_latent_upscale_h3.md) — TS Latent Upscale — перенос ядра MMH3 под MIT, подпапки моделей, замеры fp16 против bf16 и что из оригинала намеренно….
- [reference_ltx25_vae_decode_stall](reference_ltx25_vae_decode_stall.md) — LTX 2.5 VAE Decode «зависает» при полной VRAM — у декодера disable_offload=True и крупный резерв; лечит TS Free VRAM,….
- [reference_ltx_first_last_frame](reference_ltx_first_last_frame.md) — LTX 2.5 — две родные механики закрепления кадра; забытый LTXVCropGuides даёт «артефакты в конце».
- [reference_ltx_hdr](reference_ltx_hdr.md) — HDR-ветка LTX в паке целиком — устройство, обе технологии, все замеренные капканы, воркфлоу.
- [reference_ltx_preset_2026_08](reference_ltx_preset_2026_08.md) — Пресет Video Prompt Enhance LTX — официальные системники LTX 2.5, и почему 2B терял русский текст.
- [reference_ltx_tiled_decode](reference_ltx_tiled_decode.md) — VAE Decode (Tiled) на LTX 2.5 — что значат четыре числа, почему шаблоны расходятся и как подобрать их по видеопамяти.
- [reference_vfx_static_comparison_20261102](reference_vfx_static_comparison_20261102.md) — Статичное сравнение апскейлеров 2026-11-02; VFX 0.2 и DLSS v21 выпущены в 12.13.6.
- [reference_video_guide_failure](reference_video_guide_failure.md) — TS Video Guide снята 07.09.2026 — видео в ноде не показалось у владельца, причина не установлена; что проверено и что….
- [reference_video_loader_formats](reference_video_loader_formats.md) — TS Video Loader — «ProRes: звук есть, картинки нет» это плеер браузера, а не декодер; расширение не решает ничего.
- [reference_video_loader_memory](reference_video_loader_memory.md) — Потолок памяти TS Video Loader — считается от машины, плюс фоллбек на отображённый в память файл.
- [reference_video_media](reference_video_media.md) — Видео-ноды: текущая карта backend/UI и датированная история PyAV, таймлайна и звука.
- [video_media_audio](video_media_audio.md) — Видео и звук: длительность контейнера, склейки, звуковая дорожка и конечный таймлайн.
- [video_media_backend](video_media_backend.md) — Видео backend: PyAV, запись, DynamicCombo и причины выбора тракта.
- [video_media_interface](video_media_interface.md) — Видео UI: раскладка, миниатюры, tooltip, значки и перетаскивание.
