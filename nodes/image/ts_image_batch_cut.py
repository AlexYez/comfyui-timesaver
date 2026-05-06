"""TS Image Batch Cut.

node_id: TS_ImageBatchCut
"""

from typing import Optional
import logging

import torch

from comfy_api.v0_0_2 import IO


logger = logging.getLogger("comfyui_timesaver.ts_image_batch_cut")
LOG_PREFIX = "[TS Image Batch Cut]"


class TS_ImageBatchCut(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ImageBatchCut",
            display_name="TS Image Batch Cut",
            category="TS/Image",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("first_cut", default=0, min=0, max=4096),
                IO.Int.Input("last_cut", default=0, min=0, max=4096),
            ],
            outputs=[IO.Image.Output(display_name="image")],
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

    @staticmethod
    def _normalize_cut(value: int) -> int:
        try:
            return max(0, int(value))
        except Exception:
            return 0

    @classmethod
    def execute(cls, image: torch.Tensor, first_cut: int, last_cut: int) -> IO.NodeOutput:
        cls._log_tensor("Input", image)

        if image is None:
            cls._log("Input is None, returning as-is.")
            return IO.NodeOutput(image)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim == 3:
            image = image.unsqueeze(0)
            cls._log_tensor("Input normalized", image)

        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 4 dims [B,H,W,C], got {image.ndim}")

        total = int(image.shape[0])
        cut_start = cls._normalize_cut(first_cut)
        cut_end = cls._normalize_cut(last_cut)

        cls._log(f"Total frames={total} first_cut={cut_start} last_cut={cut_end}")

        if cut_start == 0 and cut_end == 0:
            cls._log("No cut applied.")
            return IO.NodeOutput(image)

        if cut_start + cut_end >= total:
            cls._log("Cut exceeds batch length, returning empty batch.")
            empty = image[:0, ...]
            cls._log_tensor("Output", empty)
            return IO.NodeOutput(empty)

        trimmed = image[cut_start : total - cut_end, ...]
        cls._log_tensor("Output", trimmed)
        return IO.NodeOutput(trimmed)

    @classmethod
    def fingerprint_inputs(cls, image: torch.Tensor, first_cut: int, last_cut: int) -> str:
        if image is None or not isinstance(image, torch.Tensor):
            return f"none_{first_cut}_{last_cut}"
        try:
            return (
                f"{tuple(image.shape)}_{image.dtype}_{float(image.mean())}_{first_cut}_{last_cut}"
            )
        except Exception:
            return f"{tuple(image.shape)}_{image.dtype}_{first_cut}_{last_cut}"


NODE_CLASS_MAPPINGS = {"TS_ImageBatchCut": TS_ImageBatchCut}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ImageBatchCut": "TS Image Batch Cut"}
