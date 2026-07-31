"""TS Qwen Safe Resize — clamp image dimensions to a Qwen-friendly aspect/size table.

node_id: TS_QwenSafeResize
"""

import numpy as np
import torch
from comfy_api.v0_0_2 import IO
from PIL import Image

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
            description="Resize to the nearest official Qwen-Image resolution.",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Image to fit to the nearest Qwen-supported resolution (resize + center crop).",
                )
            ],
            outputs=[
                IO.Image.Output(
                    display_name="IMAGE",
                    tooltip="Image resized and cropped to a Qwen-friendly resolution.",
                )
            ],
        )

    @classmethod
    def execute(cls, image) -> IO.NodeOutput:
        b, h, w, c = image.shape
        # Explicit validation (a bare ``assert`` is stripped under ``python -O``).
        if c not in (3, 4):
            raise ValueError(f"[TS Qwen Safe Resize] Expected 3 or 4 channels, got {c}.")

        output_images = []

        for i in range(b):
            # Clip before the cast: VAEDecode output is not clamped to [0, 1],
            # and an out-of-range value wraps modulo 256 (1.03 -> 6, -0.02 -> 251).
            img_np = np.clip(image[i].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            target_w, target_h = closest_supported_resolution(w, h)

            scale = max(target_w / w, target_h / h)
            # round(), and floor at the target: int() truncation could yield
            # target_w - 1 for the limiting axis, making the crop box start at
            # -1 and leaving a 1px black seam on the edge.
            new_w = max(target_w, int(round(w * scale)))
            new_h = max(target_h, int(round(h * scale)))

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
