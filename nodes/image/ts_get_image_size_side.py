"""TS Get Image Size.

node_id: TS_GetImageSizeSide
"""

from typing import Optional
import logging

import torch

from comfy_api.latest import IO


logger = logging.getLogger("comfyui_timesaver.ts_get_image_size_side")
LOG_PREFIX = "[TS Get Image Size Side]"


class TS_GetImageSizeSide(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_GetImageSizeSide",
            display_name="TS Get Image Size",
            category="TS/Image",
            inputs=[
                IO.Image.Input("image"),
                IO.Boolean.Input(
                    "large_side",
                    default=True,
                    label_on="Large Side",
                    label_off="Small Side",
                ),
            ],
            outputs=[IO.Int.Output(display_name="size")],
        )

    @staticmethod
    def _log(message: str) -> None:
        logger.info("%s %s", LOG_PREFIX, message)

    @classmethod
    def _log_tensor(cls, label: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None:
            cls._log(f"{label}: None")
            return
        if not isinstance(tensor, torch.Tensor):
            cls._log(f"{label}: invalid type={type(tensor)}")
            return
        cls._log(
            f"{label} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )

    @classmethod
    def execute(cls, image: torch.Tensor, large_side: bool) -> IO.NodeOutput:
        cls._log_tensor("Input", image)

        if image is None:
            cls._log("Input is None.")
            return IO.NodeOutput(0)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim == 3:
            image = image.unsqueeze(0)
            cls._log_tensor("Input normalized", image)

        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 4 dims [B,H,W,C], got {image.ndim}")

        height = int(image.shape[1])
        width = int(image.shape[2])
        size = max(height, width) if large_side else min(height, width)

        cls._log(f"Computed size={size} (large_side={large_side})")
        return IO.NodeOutput(size)

    @classmethod
    def fingerprint_inputs(cls, image: torch.Tensor, large_side: bool) -> str:
        if image is None or not isinstance(image, torch.Tensor):
            return f"none_{large_side}"
        try:
            return f"{tuple(image.shape)}_{image.dtype}_{float(image.mean())}_{large_side}"
        except Exception:
            return f"{tuple(image.shape)}_{image.dtype}_{large_side}"


NODE_CLASS_MAPPINGS = {"TS_GetImageSizeSide": TS_GetImageSizeSide}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_GetImageSizeSide": "TS Get Image Size"}
