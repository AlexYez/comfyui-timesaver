---
name: engineering_index_image
description: "Изображения, маски, цвет и глубина"
metadata:
  type: reference
---

# Изображения, маски, цвет и глубина

Каждая строка — отдельная тема; начните с нужного hook.

- [reference_angle_select](reference_angle_select.md) — TS Angle Select — 3D-превью на вендоренном Three.js, три регулятора, пресеты-JSON; грабли раскладки и оба режима нод.
- [reference_edit_align](reference_edit_align.md) — Смещение правок Qwen Image 2.1 в «Image Gen Edit Qwen 2» (жалоба пользователя 15.10.2026) — две причины, замеры, и….
- [reference_film_emulation](reference_film_emulation.md) — TS Film Emulation 12.13 — модель плёнки (кривая по слоям, 6 полос оттенка, защита кожи, халация, ч/б), 24 плёнки,….
- [reference_film_grain](reference_film_grain.md) — Плёночное зерно в TS Film Emulation — почему шум применяется в логарифмическом пространстве, замеры по тонам, скорость….
- [reference_inpaint_refine_presets](reference_inpaint_refine_presets.md) — Готовые промпты инпэйнта написаны под Refine (низкий denoise) — описывают то, что уже есть, и просят сохранить….
- [reference_inpaint_technology](reference_inpaint_technology.md) — Технология перерисовки по вырезу — почему так, какие числа откуда взяты, что уже проверено живьём и что менять нельзя.
- [reference_lama_cleanup](reference_lama_cleanup.md) — Lama-нода переехала с pickled .ckpt на safetensors + чистый PyTorch код вместо упакованного оригинала; модели в….
- [reference_model_resize](reference_model_resize.md) — TS Model Resize (15.10.2026) — откуда взяты таблицы разрешений моделей, где сообщество ошибается, и два вида пресетов….
- [reference_multi_reference_mask](reference_multi_reference_mask.md) — TS_MultiReference (v9.12+) — маска используется ТОЛЬКО как bbox-hint, не для силуэт-cutout; auto-detect convention +….
- [reference_rgba_images](reference_rgba_images.md) — IMAGE бывает RGBA — VAE Qwen Image 2.1 в ядре ComfyUI объявлен RGBA, VAEDecode отдаёт 4 канала; итоги аудита пака….
- [reference_sam3_detect_contract](reference_sam3_detect_contract.md) — Родная ComfyUI SAM3_Detect ожидает positive_coords/negative_coords как STRING JSON в пикселях, не custom-type с….
- [reference_universal_depth](reference_universal_depth.md) — Две ноды глубины (TS Video Depth и TS Image Depth), общий движок _depth_core, замеры против эталона Kijai, веса fp16 на….
- [reference_vae_fix](reference_vae_fix.md) — TS VAE Fix (бывш. рабочее имя DeGrid) — какие VAE оставляют решётку 2 px (замеры 09.10.2026 на output владельца),….
- [ts-smart-inpaint-node](ts-smart-inpaint-node.md) — Architecture & provenance of the TS Smart Inpaint node (headless ComfyUI-Angelo Xtra-Fine port).
