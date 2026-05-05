"""TS Image List to Image Batch.

node_id: TS_ImageListToImageBatch
"""

from typing import Optional
import time

import torch
import comfy.utils
import logging


logger = logging.getLogger("comfyui_timesaver.ts_image_list_to_batch")
LOG_PREFIX = "[TS Image List to Image Batch]"


class TS_ImageListToImageBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
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

    @staticmethod
    def _ensure_bhwc(image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            return image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 3 or 4 dims, got {image.ndim}")
        return image

    def execute(self, images):
        if images is None:
            self._log("Input list is None.")
            return ()

        if not isinstance(images, list):
            images = [images]

        self._log(f"Input list length={len(images)}")

        if len(images) == 0:
            self._log("Input list is empty.")
            return ()

        valid_images = [img for img in images if img is not None]
        if len(valid_images) == 0:
            self._log("All input images are None.")
            return ()

        normalized = []
        for idx, img in enumerate(valid_images):
            if not isinstance(img, torch.Tensor):
                raise ValueError(f"Image {idx} is not a torch.Tensor: {type(img)}")
            norm = self._ensure_bhwc(img)
            normalized.append(norm)

        base = normalized[0]
        target_h, target_w = base.shape[1], base.shape[2]
        target_c = min(img.shape[3] for img in normalized)
        target_dtype = base.dtype
        target_device = base.device

        self._log_tensor("Input[0]", base)
        self._log(f"Target size={target_w}x{target_h} channels={target_c}")

        resized = []
        for idx, img in enumerate(normalized):
            if img.device != target_device:
                self._log(f"Image {idx} moved to {target_device}")
                img = img.to(target_device)
            if img.dtype != target_dtype:
                img = img.to(target_dtype)
            if img.shape[1] != target_h or img.shape[2] != target_w:
                self._log(
                    f"Image {idx} resized from {img.shape[2]}x{img.shape[1]} to {target_w}x{target_h}"
                )
                img = comfy.utils.common_upscale(
                    img.movedim(-1, 1), target_w, target_h, "lanczos", "center"
                ).movedim(1, -1)
            if img.shape[3] != target_c:
                self._log(f"Image {idx} channels trimmed to {target_c}")
                img = img[..., :target_c]
            resized.append(img)

        batch = torch.cat(resized, dim=0)
        self._log_tensor("Output", batch)
        return (batch,)

    @classmethod
    def IS_CHANGED(cls, images) -> str:
        if images is None:
            return "none"
        if not isinstance(images, list):
            images = [images]
        if len(images) == 0:
            return "empty"
        shapes = []
        for img in images:
            if isinstance(img, torch.Tensor):
                shapes.append(tuple(img.shape))
        try:
            sums = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    sums.append(float(img.mean()))
            return f"{shapes}_{sums}"
        except Exception:
            return f"{shapes}"




NODE_CLASS_MAPPINGS = {"TS_ImageListToImageBatch": TS_ImageListToImageBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ImageListToImageBatch": "TS Image List to Image Batch"}
