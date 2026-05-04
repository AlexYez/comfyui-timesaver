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

# РЎРїРёСЃРѕРє РґР»СЏ dropdown РІ UI
ASPECT_OPTIONS = [name for (_, _, name) in QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS]

class TS_QwenCanvas:
    """
    TS Qwen Canvas
    Р’РїРёСЃС‹РІР°РµС‚ РёР·РѕР±СЂР°Р¶РµРЅРёРµ РІ РєР°РЅРІР°СЃ РїРѕРґ Р±РµР·РѕРїР°СЃРЅРѕРµ СЂР°Р·СЂРµС€РµРЅРёРµ Qwen.
    РџРѕРґРґРµСЂР¶РёРІР°РµС‚ РјР°СЃРєСѓ ComfyUI.
    Р•СЃР»Рё РёР·РѕР±СЂР°Р¶РµРЅРёРµ РЅРµ РїРѕРґР°РЅРѕ, СЃРѕР·РґР°РµС‚ РїСѓСЃС‚РѕР№ Р±РµР»С‹Р№ РєР°РЅРІР°СЃ РІС‹Р±СЂР°РЅРЅРѕРіРѕ СЂР°Р·СЂРµС€РµРЅРёСЏ.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # image РїРµСЂРµРјРµС‰РµРЅ РІ optional
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
    CATEGORY = "TS Qwen"

    def make_canvas(self, resolution="1:1", image=None, mask=None):
        # РџРѕР»СѓС‡Р°РµРј target СЂР°Р·РјРµСЂС‹ РїРѕ РІС‹Р±СЂР°РЅРЅРѕРјСѓ РёРјРµРЅРё
        target_w, target_h = None, None
        for w, h, name in QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS:
            if name == resolution:
                target_w, target_h = w, h
                break
        if target_w is None:
            raise ValueError(f"Resolution {resolution} not found")

        # РЎРѕР·РґР°С‘Рј Р±РµР»С‹Р№ РєР°РЅРІР°СЃ (Р±Р°Р·Р°)
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))

        # Р•СЃР»Рё РёР·РѕР±СЂР°Р¶РµРЅРёРµ РµСЃС‚СЊ, РѕР±СЂР°Р±Р°С‚С‹РІР°РµРј Рё РЅР°РєР»Р°РґС‹РІР°РµРј РµРіРѕ
        if image is not None:
            # РљРѕРЅРІРµСЂС‚Р°С†РёСЏ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ РІ PIL
            img_tensor = image[0]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)

            # Р•СЃР»Рё РјР°СЃРєР° РµСЃС‚СЊ Рё РЅРµ РїСѓСЃС‚Р°СЏ, РѕР±СЂРµР·Р°РµРј РёР·РѕР±СЂР°Р¶РµРЅРёРµ РїРѕ bounding box
            if mask is not None:
                mask_tensor = mask[0]
                mask_np = mask_tensor.detach().cpu().numpy()
                if mask_np.ndim == 4:  # (B,1,H,W)
                    mask_np = mask_np[0,0]
                elif mask_np.ndim == 3:  # (H,W,1) РёР»Рё (1,H,W)
                    mask_np = mask_np.squeeze(0).squeeze(-1)
                elif mask_np.ndim != 2:
                    raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

                # РџСЂРѕРІРµСЂРєР° РЅР° РїСѓСЃС‚РѕС‚Сѓ/РѕРґРЅРѕС‚РѕРЅРЅРѕСЃС‚СЊ
                if mask_np.max() > 0 and mask_np.min() < 1:
                    # РќР°С…РѕРґРёРј bounding box
                    ys, xs = np.where(mask_np > 0)
                    if ys.size > 0 and xs.size > 0:
                        top, left = ys.min(), xs.min()
                        bottom, right = ys.max(), xs.max()
                        img = img.crop((left, top, right + 1, bottom + 1))

            # РњР°СЃС€С‚Р°Р±РёСЂСѓРµРј РёР·РѕР±СЂР°Р¶РµРЅРёРµ СЃ СЃРѕС…СЂР°РЅРµРЅРёРµРј РїСЂРѕРїРѕСЂС†РёР№
            img_w, img_h = img.size
            scale_w = target_w / img_w
            scale_h = target_h / img_h

            # РћРїСЂРµРґРµР»СЏРµРј РѕСЂРёРµРЅС‚Р°С†РёСЋ (С…РѕС‚СЏ Р»РѕРіРёРєР° scale РЅРёР¶Рµ РѕРґРёРЅР°РєРѕРІР° РґР»СЏ РѕР±РѕРёС… РІРµС‚РѕРє)
            # scale РІСЃРµРіРґР° Р±РµСЂРµС‚СЃСЏ РїРѕ РјРёРЅРёРјСѓРјСѓ, С‡С‚РѕР±С‹ РІРїРёСЃР°С‚СЊ С†РµР»РёРєРѕРј
            scale = min(scale_w, scale_h)

            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

            # Р¦РµРЅС‚СЂРёСЂРѕРІР°РЅРёРµ
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            canvas.paste(img, (paste_x, paste_y))

        # РљРѕРЅРІРµСЂС‚Р°С†РёСЏ РІ С„РѕСЂРјР°С‚ ComfyUI (B,H,W,C)
        # Р•СЃР»Рё image Р±С‹Р» None, РІРµСЂРЅРµС‚СЃСЏ РїСЂРѕСЃС‚Рѕ Р±РµР»С‹Р№ РїСЂСЏРјРѕСѓРіРѕР»СЊРЅРёРє
        out_img = torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0).unsqueeze(0)

        return (out_img, target_w, target_h)



NODE_CLASS_MAPPINGS = {"TS_QwenCanvas": TS_QwenCanvas}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_QwenCanvas": "TS Qwen Canvas"}
