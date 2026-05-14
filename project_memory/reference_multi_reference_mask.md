---
name: TS_MultiReference mask handling
description: TS_MultiReference (v9.12+) — маска используется ТОЛЬКО как bbox-hint, не для силуэт-cutout; auto-detect convention + бинаризация по 0.5
type: reference
---
В `nodes/conditioning/ts_multi_reference.py` маска (`mask_N` input) обрабатывается так:

## Конвейер

В `_resize_reference_image`:

1. **Step 0 — `_binarize_and_orient_mask`**: бинаризация по `_BINARIZE_THRESHOLD = 0.5` + auto-detect convention по углам.
   - Sample 4 угловых квадрата (`min(H,W) // _CORNER_SAMPLE_FRACTION (16)` px).
   - Если ≥ 2 углов «светлые» (>0.5) → ComfyUI native (1=transparent), пропускаем.
   - Если < 2 углов светлые → segmentation convention (1=subject, BiRefNet/SAM/RemBG/Mask Editor) → инвертируем.
   - Возвращает каноничный native-convention binary `[B, H, W, 1]`.

2. **Step 1 — `_crop_to_mask_bbox`**: bbox по «opaque» пикселям (1-mask > 0.01) + 16 px padding (`_BBOX_PADDING`). Только для CROP, форму не вырезает.

3. **Step 2 — `_normalize_image_tensor(image)` БЕЗ mask**: composite на белый делается **только** если IMAGE сам RGBA (embedded alpha, защита от premultiplied black halos). MASK input в композит **не передаётся** — пиксели внутри bbox preserved as-is.

## Почему

Пользователь явно сказал: «маска для bbox, форму не вырезать». Силуэт-cutout был неудачным дефолтом — давал «полупрозрачный белый круг сверху» когда mask convention не совпадала, и обрезал контекст вокруг subject в любом случае. Теперь VAE видит subject + натуральный контекст в пределах bbox-padding'а.

## Что было фиксить

- Auto-detect mask convention (углы) — без новых widget'ов.
- Hard binarisation по 0.5 — убирает edge-fuzz от soft segmentation outputs.
- Composite НЕ через mask — pixels inside bbox остаются original RGB.

## Где править

- Логика — `nodes/conditioning/ts_multi_reference.py:_binarize_and_orient_mask` и `_resize_reference_image`.
- Тесты — `tests/test_multi_reference_node.py`:
  - `test_resize_handles_segmentation_convention_mask` — белый круг → bbox 152x152.
  - `test_resize_native_convention_mask_unchanged` — чёрный круг (legacy) → bbox 152x152.
  - `test_resize_binarises_soft_edge_mask` — soft fade → hard 64x64 после бинаризации.
  - `test_resize_preserves_rgb_inside_bbox_no_silhouette_cutout` — red subject + blue surrounding, оба сохранены внутри bbox.
  - `test_binarize_and_orient_mask_inverts_segmentation` / `_keeps_native` — unit tests на helper.

## Edge cases

- **Subject касается всех 4 углов** → auto-detect неоднозначен, default = native, bbox = почти всё image. Это и есть единственный осмысленный ответ.
- **Маска полностью прозрачная / opaque** → `_crop_to_mask_bbox` не сжимает до нуля, возвращает оригинал.
- **RGBA IMAGE без MASK input** → composite по embedded alpha остался (legacy, защита от black halos).

## Если когда-нибудь нужно вернуть силуэт-cutout

- Добавить `mask=mask` обратно в вызов `_normalize_image_tensor` в `_resize_reference_image`.
- Лучше — отдельным toggle (новый BOOL вход `silhouette_cutout` default=False).
