"""TS Qwen Safe Resize — clamp image dimensions to a Qwen-friendly aspect/size table.

node_id: TS_QwenSafeResize
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


QWEN_IMAGE_SUPPORTED_RESOLUTIONS = [
    (1344, 1344, 1.0),   # 1:1 (ближайшее к 1328x1328)
    (1792, 1008, 1.778), # 16:9 (ближайшее к 1664x928)
    (1008, 1792, 0.562), # 9:16
    (1456, 1088, 1.338), # 4:3 (ближайшее к 1472x1140)
    (1088, 1456, 0.747), # 3:4
    (1568, 1056, 1.484), # 3:2 (ближайшее к 1584x1056)
    (1056, 1568, 0.673), # 2:3
]

def closest_supported_resolution(width, height):
    aspect = width / height
    best_res = None
    best_diff = 999
    for w, h, a in QWEN_IMAGE_SUPPORTED_RESOLUTIONS:
        diff = abs(aspect - a)
        if diff < best_diff:
            best_diff = diff
            best_res = (w, h)
    return best_res

class TS_QwenSafeResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "safe_resize"
    CATEGORY = "image/resize"

    def safe_resize(self, image):
        # image: torch.Tensor, shape (B,H,W,C), dtype float32, range 0..1
        b, h, w, c = image.shape
        assert c in [3, 4], f"Expected 3 or 4 channels, got {c}"

        output_images = []

        for i in range(b):
            # в†’ NumPy (0..255)
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # ---- Выбираем ближайшее разрешение ----
            target_w, target_h = closest_supported_resolution(w, h)

            # ---- Масштабируем ----
            scale = max(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)

            # ---- Кроп по центру ----
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            cropped = resized.crop((left, top, right, bottom))

            # → обратно в тензор float32 (0..1)
            img_out = torch.from_numpy(np.array(cropped)).float() / 255.0
            # (H,W,C) в†’ (1,H,W,C)
            output_images.append(img_out.unsqueeze(0))

        # Собираем батч
        output = torch.cat(output_images, dim=0)
        return (output,)
    


# Все разрешения Qwen: горизонтальные и вертикальные


NODE_CLASS_MAPPINGS = {"TS_QwenSafeResize": TS_QwenSafeResize}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_QwenSafeResize": "TS Qwen Safe Resize"}
