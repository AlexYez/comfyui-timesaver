"""TS Get Image Megapixels.

node_id: TS_GetImageMegapixels
"""

from typing import Optional
import logging

import torch

from comfy_api.latest import IO


logger = logging.getLogger("comfyui_timesaver.ts_get_image_megapixels")
LOG_PREFIX = "[TS Get Image Megapixels]"


class TS_GetImageMegapixels(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_GetImageMegapixels",
            display_name="TS Get Image Megapixels",
            category="TS/Image",
            inputs=[IO.Image.Input("image")],
            outputs=[IO.Float.Output(display_name="megapixels")],
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
    def execute(cls, image: torch.Tensor) -> IO.NodeOutput:
        cls._log_tensor("Input", image)

        if image is None:
            cls._log("Input is None.")
            return IO.NodeOutput(0.0)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim == 3:
            image = image.unsqueeze(0)
            cls._log_tensor("Input normalized", image)

        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 4 dims [B,H,W,C], got {image.ndim}")

        height = int(image.shape[1])
        width = int(image.shape[2])
        megapixels = float(width * height) / 1_000_000.0

        cls._log(f"Computed megapixels={megapixels}")
        return IO.NodeOutput(megapixels)

    @classmethod
    def fingerprint_inputs(cls, image: torch.Tensor) -> str:
        if image is None or not isinstance(image, torch.Tensor):
            return "none"
        return f"{tuple(image.shape)}_{image.dtype}"


NODE_CLASS_MAPPINGS = {"TS_GetImageMegapixels": TS_GetImageMegapixels}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_GetImageMegapixels": "TS Get Image Megapixels"}
