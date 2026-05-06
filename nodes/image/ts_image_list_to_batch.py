"""TS Image List to Image Batch.

node_id: TS_ImageListToImageBatch
"""

from typing import Optional
import logging

import torch
import comfy.utils

from comfy_api.latest import IO


logger = logging.getLogger("comfyui_timesaver.ts_image_list_to_batch")
LOG_PREFIX = "[TS Image List to Image Batch]"


class TS_ImageListToImageBatch(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ImageListToImageBatch",
            display_name="TS Image List to Image Batch",
            category="TS/Image",
            is_input_list=True,
            inputs=[IO.Image.Input("images")],
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
    def _ensure_bhwc(image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            return image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 3 or 4 dims, got {image.ndim}")
        return image

    @classmethod
    def execute(cls, images) -> IO.NodeOutput:
        if images is None:
            cls._log("Input list is None.")
            return IO.NodeOutput()

        if not isinstance(images, list):
            images = [images]

        cls._log(f"Input list length={len(images)}")

        if len(images) == 0:
            cls._log("Input list is empty.")
            return IO.NodeOutput()

        valid_images = [img for img in images if img is not None]
        if len(valid_images) == 0:
            cls._log("All input images are None.")
            return IO.NodeOutput()

        normalized = []
        for idx, img in enumerate(valid_images):
            if not isinstance(img, torch.Tensor):
                raise ValueError(f"Image {idx} is not a torch.Tensor: {type(img)}")
            norm = cls._ensure_bhwc(img)
            normalized.append(norm)

        base = normalized[0]
        target_h, target_w = base.shape[1], base.shape[2]
        target_c = min(img.shape[3] for img in normalized)
        target_dtype = base.dtype
        target_device = base.device

        cls._log_tensor("Input[0]", base)
        cls._log(f"Target size={target_w}x{target_h} channels={target_c}")

        resized = []
        for idx, img in enumerate(normalized):
            if img.device != target_device:
                cls._log(f"Image {idx} moved to {target_device}")
                img = img.to(target_device)
            if img.dtype != target_dtype:
                img = img.to(target_dtype)
            if img.shape[1] != target_h or img.shape[2] != target_w:
                cls._log(
                    f"Image {idx} resized from {img.shape[2]}x{img.shape[1]} to {target_w}x{target_h}"
                )
                img = comfy.utils.common_upscale(
                    img.movedim(-1, 1), target_w, target_h, "lanczos", "center"
                ).movedim(1, -1)
            if img.shape[3] != target_c:
                cls._log(f"Image {idx} channels trimmed to {target_c}")
                img = img[..., :target_c]
            resized.append(img)

        batch = torch.cat(resized, dim=0)
        cls._log_tensor("Output", batch)
        return IO.NodeOutput(batch)

    @classmethod
    def fingerprint_inputs(cls, images) -> str:
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
