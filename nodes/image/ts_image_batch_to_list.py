"""TS Image Batch to Image List.

node_id: TS_ImageBatchToImageList
"""

from typing import Optional
import time

import torch
import comfy.utils
import logging


logger = logging.getLogger("comfyui_timesaver.ts_image_batch_to_list")
LOG_PREFIX = "[TS Image Batch to Image List]"


class TS_ImageBatchToImageList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "TS/Image"

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

    def execute(self, image: torch.Tensor):
        self._log_tensor("Input", image)

        if image is None:
            self._log("Input is None, returning empty list.")
            return ([],)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim == 3:
            image = image.unsqueeze(0)
            self._log_tensor("Input normalized", image)

        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 4 dims [B,H,W,C], got {image.ndim}")

        images = [image[i : i + 1, ...] for i in range(image.shape[0])]

        if images:
            self._log(f"Output list length={len(images)}")
            self._log_tensor("Output[0]", images[0])

        return (images,)

    @classmethod
    def IS_CHANGED(cls, image: torch.Tensor) -> str:
        if image is None or not isinstance(image, torch.Tensor):
            return "none"
        try:
            return f"{tuple(image.shape)}_{image.dtype}_{float(image.mean())}"
        except Exception:
            return f"{tuple(image.shape)}_{image.dtype}"




NODE_CLASS_MAPPINGS = {"TS_ImageBatchToImageList": TS_ImageBatchToImageList}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ImageBatchToImageList": "TS Image Batch to Image List"}
