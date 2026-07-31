# TS Matting (ViTMatte)

Guided alpha matting через Hugging Face ViTMatte. На вход — изображение + грубая маска (например, из SAM3 Detect), нода авто-строит trimap и уточняет маску до фотореалистичного альфа-канала. Такой же набор пост-обработки `mask_blur`/`mask_offset`/`background`, как у TS Remove Background, — drop-in замена, когда важны края, волосы и полупрозрачность. Модели кэшируются в `models/vitmatte/`.

**Когда использовать:** получить чистый cut-out из SAM-style маски без захода в Photoshop.

---

Полный справочник нод: [README](https://github.com/AlexYez/comfyui-timesaver/blob/master/README.ru.md)
