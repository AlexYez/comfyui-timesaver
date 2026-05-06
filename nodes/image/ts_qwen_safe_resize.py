"""TS Qwen Safe Resize — clamp image dimensions to a Qwen-friendly aspect/size table.

node_id: TS_QwenSafeResize
"""

import torch
import numpy as np
from PIL import Image

from comfy_api.v0_0_2 import IO


QWEN_IMAGE_SUPPORTED_RESOLUTIONS = [
    (1344, 1344, 1.0),
    (1792, 1008, 1.778),
    (1008, 1792, 0.562),
    (1456, 1088, 1.338),
    (1088, 1456, 0.747),
    (1568, 1056, 1.484),
    (1056, 1568, 0.673),
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


class TS_QwenSafeResize(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_QwenSafeResize",
            display_name="TS Qwen Safe Resize",
            category="TS/Image",
            inputs=[IO.Image.Input("image")],
            outputs=[IO.Image.Output(display_name="IMAGE")],
        )

    @classmethod
    def execute(cls, image) -> IO.NodeOutput:
        b, h, w, c = image.shape
        assert c in [3, 4], f"Expected 3 or 4 channels, got {c}"

        output_images = []

        for i in range(b):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            target_w, target_h = closest_supported_resolution(w, h)

            scale = max(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)

            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            right = left + target_w
            bottom = top + target_h
            cropped = resized.crop((left, top, right, bottom))

            img_out = torch.from_numpy(np.array(cropped)).float() / 255.0
            output_images.append(img_out.unsqueeze(0))

        output = torch.cat(output_images, dim=0)
        return IO.NodeOutput(output)


NODE_CLASS_MAPPINGS = {"TS_QwenSafeResize": TS_QwenSafeResize}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_QwenSafeResize": "TS Qwen Safe Resize"}
