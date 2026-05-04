"""TS WAN Safe Resize — pick a model-friendly resolution for WAN pipelines.

node_id: TS_WAN_SafeResize
"""

import math

import torch
import numpy as np
from PIL import Image


class TS_WAN_SafeResize:
    WAN_RESOLUTIONS = {
        "high quality": {
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "1:1": (720, 720),
        },
        "standard quality": {
            "16:9": (832, 480),
            "9:16": (480, 832),
            "1:1": (480, 480),
        },
        "low quality": {
            "16:9": (426, 240),
            "9:16": (240, 426),
            "1:1": (240, 240),
        },
    }

    QUALITY_MAP = {
        "Fast quality": "low quality",
        "Standard quality": "standard quality",
        "High quality": "high quality",
    }

    @staticmethod
    def detect_aspect_ratio(width, height):
        aspect = width / height
        if aspect > 1.3:
            return "16:9"
        elif aspect < 0.8:
            return "9:16"
        else:
            return "1:1"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "quality": (
                    ["Fast quality", "Standard quality", "High quality"],
                    {"default": "Standard quality"},
                ),
            },
            "optional": {
                "interconnection_in": ("STRING",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "interconnection_out")
    FUNCTION = "safe_resize"
    CATEGORY = "image/resize"

    def safe_resize(self, image, quality, interconnection_in=None):
        # РџСЂРёРѕСЂРёС‚РµС‚ interconnection
        if interconnection_in in self.WAN_RESOLUTIONS:
            internal_quality = interconnection_in
        else:
            internal_quality = self.QUALITY_MAP[quality]

        b, h, w, c = image.shape
        assert c in [3, 4], f"Expected 3 or 4 channels, got {c}"

        aspect_key = self.detect_aspect_ratio(w, h)
        target_w, target_h = self.WAN_RESOLUTIONS[internal_quality][aspect_key]

        output_images = []

        for i in range(b):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            scale = max(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)

            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            cropped = resized.crop((left, top, left + target_w, top + target_h))

            img_out = torch.from_numpy(np.array(cropped)).float() / 255.0
            output_images.append(img_out.unsqueeze(0))

        output = torch.cat(output_images, dim=0)

        return (output, target_w, target_h, internal_quality)




NODE_CLASS_MAPPINGS = {"TS_WAN_SafeResize": TS_WAN_SafeResize}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_WAN_SafeResize": "TS WAN Safe Resize"}
