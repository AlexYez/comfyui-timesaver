"""TS Get Image Megapixels.

node_id: TS_GetImageMegapixels
"""

from typing import Optional

import torch
import logging


logger = logging.getLogger("comfyui_timesaver.ts_get_image_megapixels")
LOG_PREFIX = "[TS Get Image Megapixels]"


class TS_GetImageMegapixels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("megapixels",)
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
            self._log("Input is None.")
            return (0.0,)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim == 3:
            image = image.unsqueeze(0)
            self._log_tensor("Input normalized", image)

        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 4 dims [B,H,W,C], got {image.ndim}")

        height = int(image.shape[1])
        width = int(image.shape[2])
        megapixels = float(width * height) / 1_000_000.0

        self._log(f"Computed megapixels={megapixels}")
        return (megapixels,)

    @classmethod
    def IS_CHANGED(cls, image: torch.Tensor) -> str:
        if image is None or not isinstance(image, torch.Tensor):
            return "none"
        # Megapixels depend only on shape; do not read pixel data here.
        return f"{tuple(image.shape)}_{image.dtype}"




NODE_CLASS_MAPPINGS = {"TS_GetImageMegapixels": TS_GetImageMegapixels}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_GetImageMegapixels": "TS Get Image Megapixels"}
