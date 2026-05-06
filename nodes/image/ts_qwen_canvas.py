"""TS Qwen Canvas — generate a Qwen-friendly canvas with optional image/mask placement.

node_id: TS_QwenCanvas
"""

import torch
import numpy as np
from PIL import Image

from comfy_api.v0_0_2 import IO


QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS = [
    (1344, 1344, "1:1"),
    (1792, 1008, "16:9"),
    (1008, 1792, "16:9 Vertical"),
    (1456, 1088, "4:3"),
    (1088, 1456, "4:3 Vertical"),
    (1568, 1056, "3:2"),
    (1056, 1568, "3:2 Vertical"),
]

ASPECT_OPTIONS = [name for (_, _, name) in QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS]


class TS_QwenCanvas(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_QwenCanvas",
            display_name="TS Qwen Canvas",
            category="TS/Image",
            inputs=[
                IO.Combo.Input("resolution", options=ASPECT_OPTIONS, default="1:1"),
                IO.Image.Input("image", optional=True),
                IO.Mask.Input("mask", optional=True),
            ],
            outputs=[
                IO.Image.Output(display_name="canvas_image"),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, resolution="1:1", image=None, mask=None) -> IO.NodeOutput:
        target_w, target_h = None, None
        for w, h, name in QWEN_IMAGE_CANVAS_SUPPORTED_RESOLUTIONS:
            if name == resolution:
                target_w, target_h = w, h
                break
        if target_w is None:
            raise ValueError(f"Resolution {resolution} not found")

        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))

        if image is not None:
            img_tensor = image[0]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)

            if mask is not None:
                mask_tensor = mask[0]
                mask_np = mask_tensor.detach().cpu().numpy()
                if mask_np.ndim == 4:
                    mask_np = mask_np[0, 0]
                elif mask_np.ndim == 3:
                    mask_np = mask_np.squeeze(0).squeeze(-1)
                elif mask_np.ndim != 2:
                    raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

                if mask_np.max() > 0 and mask_np.min() < 1:
                    ys, xs = np.where(mask_np > 0)
                    if ys.size > 0 and xs.size > 0:
                        top, left = ys.min(), xs.min()
                        bottom, right = ys.max(), xs.max()
                        img = img.crop((left, top, right + 1, bottom + 1))

            img_w, img_h = img.size
            scale_w = target_w / img_w
            scale_h = target_h / img_h

            scale = min(scale_w, scale_h)

            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            canvas.paste(img, (paste_x, paste_y))

        out_img = torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0).unsqueeze(0)

        return IO.NodeOutput(out_img, target_w, target_h)


NODE_CLASS_MAPPINGS = {"TS_QwenCanvas": TS_QwenCanvas}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_QwenCanvas": "TS Qwen Canvas"}
