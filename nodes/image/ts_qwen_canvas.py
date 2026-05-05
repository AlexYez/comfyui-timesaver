"""TS Qwen Canvas — generate a Qwen-friendly canvas with optional image/mask placement.

node_id: TS_QwenCanvas
"""

import math

import torch
import numpy as np
from PIL import Image

try:
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS = [
    (1344, 1344, "1:1"),
    (1792, 1008, "16:9"),
    (1008, 1792, "16:9 Vertical"),
    (1456, 1088, "4:3"),
    (1088, 1456, "4:3 Vertical"),
    (1568, 1056, "3:2"),
    (1056, 1568, "3:2 Vertical"),
]

# Список для dropdown в UI
ASPECT_OPTIONS = [name for (_, _, name) in QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS]

class TS_QwenCanvas:
    """
    TS Qwen Canvas
    Вписывает изображение в канвас под безопасное разрешение Qwen.
    Поддерживает маску ComfyUI.
    Если изображение не подано, создает пустой белый канвас выбранного разрешения.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # image перемещен в optional
                "resolution": (ASPECT_OPTIONS, {"default": "1:1"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("canvas_image", "width", "height")
    FUNCTION = "make_canvas"
    CATEGORY = "TS/Image"

    def make_canvas(self, resolution="1:1", image=None, mask=None):
        # Получаем target размеры по выбранному имени
        target_w, target_h = None, None
        for w, h, name in QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS:
            if name == resolution:
                target_w, target_h = w, h
                break
        if target_w is None:
            raise ValueError(f"Resolution {resolution} not found")

        # Создаём белый канвас (база)
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))

        # Если изображение есть, обрабатываем и накладываем его
        if image is not None:
            # Конвертация изображения в PIL
            img_tensor = image[0]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)

            # Если маска есть и не пустая, обрезаем изображение по bounding box
            if mask is not None:
                mask_tensor = mask[0]
                mask_np = mask_tensor.detach().cpu().numpy()
                if mask_np.ndim == 4:  # (B,1,H,W)
                    mask_np = mask_np[0,0]
                elif mask_np.ndim == 3:  # (H,W,1) или (1,H,W)
                    mask_np = mask_np.squeeze(0).squeeze(-1)
                elif mask_np.ndim != 2:
                    raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

                # Проверка на пустоту/однотонность
                if mask_np.max() > 0 and mask_np.min() < 1:
                    # Находим bounding box
                    ys, xs = np.where(mask_np > 0)
                    if ys.size > 0 and xs.size > 0:
                        top, left = ys.min(), xs.min()
                        bottom, right = ys.max(), xs.max()
                        img = img.crop((left, top, right + 1, bottom + 1))

            # Масштабируем изображение с сохранением пропорций
            img_w, img_h = img.size
            scale_w = target_w / img_w
            scale_h = target_h / img_h

            # Определяем ориентацию (хотя логика scale ниже одинакова для обоих веток)
            # scale всегда берется по минимуму, чтобы вписать целиком
            scale = min(scale_w, scale_h)

            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

            # Центрирование
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            canvas.paste(img, (paste_x, paste_y))

        # Конвертация в формат ComfyUI (B,H,W,C)
        # Если image был None, вернется просто белый прямоугольник
        out_img = torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0).unsqueeze(0)

        return (out_img, target_w, target_h)



NODE_CLASS_MAPPINGS = {"TS_QwenCanvas": TS_QwenCanvas}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_QwenCanvas": "TS Qwen Canvas"}
