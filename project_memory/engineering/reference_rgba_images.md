---
name: reference_rgba_images
description: "IMAGE бывает RGBA — VAE Qwen Image 2.1 в ядре ComfyUI объявлен RGBA, VAEDecode отдаёт 4 канала; итоги аудита пака 11.10.2026 и правило для image-нод"
metadata:
  type: reference
---

IMAGE бывает RGBA — VAE Qwen Image 2.1 в ядре ComfyUI объявлен RGBA, VAEDecode отдаёт 4 канала; итоги аудита пака 11.10.2026 и правило для image-нод

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

**IMAGE в ComfyUI бывает с 4 каналами — и это теперь обычный случай, а не экзотика.**

`comfy/sd.py`: VAE Qwen Image 2.1 (64-канальный латент, раскладка Wan 2.2) и его
TAESD объявлены RGBA (`self.output_channels = 4`). `VAEDecode` на любом графе
Qwen 2.1 — включая `texture_fix_vae_for_qwen_image_2.1` — отдаёт `[B,H,W,4]`.
Замер 11.10.2026 (encode→decode настоящей картинки): альфа 0.992–1.0 у
штатного VAE, 0.996–1.0 у texture-fix — то есть непрозрачная, но НЕ ровно 1.

Ядро на ENCODE каналы подгоняет само (`vae_encode_crop_pixels`: лишние режет,
недостающую альфу у Qwen 2.1 добивает 1.0). Опасно на DECODE: число каналов
решает VAE, а не холст.

Инцидент 11.10.2026: «Image Gen Edit Qwen 2» — VAEDecode → TS DLSS Upscaler →
TS Film Emulation + `Fuji.cube` падал (LUT сводил 4 канала к 3, `torch.lerp`).

**Аудит всего пака 11.10.2026** (статический разбор + живые прогоны на GPU,
включая модели). Исправлено:
- Film Emulation (падение с LUT), Color Match (отказ «3 channels»);
- Smart Inpaint — падение, когда VAE и холст расходятся каналами (RGBA-картинка
  + Flux/Wan VAE, RGB-картинка + Qwen 2.1 VAE): `_match_channels` после декода;
- Image Depth / Video Depth — падение на RGBA и сером (`run_depth` в
  `_depth_core.py` приводит к RGB);
- LTX HDR Guide — явный отказ на RGBA-подсказке;
- Image List → Batch — серое в цветном списке обрезало ВЕСЬ пакет до красного канала;
- Resolution Selector — непрозрачная RGBA шла как вырез поверх белого и
  светлела до 2/255 (`_OPAQUE_ALPHA = 0.98`);
- Batch Write — превью RGBA в JPEG падало внутри сервера;
- LiteRT (Gemma) — кадр уходил RGBA-PNG; теперь RGB, как у Qwen;
- серый (1 канал): Qwen-движок, Compare, ViTMatte, Frame Interpolation,
  BiRefNet, RTX, DLSS — разворачивают в три канала.

Уже было правильно: DLSS и RTX (альфа отдельно и обратно), BiRefNet/ViTMatte
(альфу заменяет маска), Video Saver, Crop/Restore, Tile Splitter/Merger, Image
Resize, Multi Reference (на белое), LTX First/Last (ядро режет `[:3]`).
Загрузчики пака (Batch Load Image, LaMa, SAM, Video Loader) всегда отдают RGB,
альфу — отдельной MASK.

**Правило для image-нод пака:** цвет считать по `[..., :3]`, остальное вернуть
как пришло (`torch.cat` в конце); серый — `[..., :1].expand(..., 3)`; входной
тензор не трогать. Гард: `tests/test_rgba_inputs.py` (+ Film Emulation и Color
Match в `test_image_nodes.py`, VAE Fix в `test_vae_fix.py`).

⚠️ Мои замеры решётки в [reference_vae_fix](reference_vae_fix.md) шли по PNG через
`convert("RGB")` — альфу они не видели.
