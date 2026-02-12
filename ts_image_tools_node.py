from typing import Optional

import torch
import comfy.utils


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
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Image Batch to Image List] {message}")

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
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Image List to Image Batch] {message}")

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
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Get Image Megapixels] {message}")

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
        try:
            return f"{tuple(image.shape)}_{image.dtype}_{float(image.mean())}"
        except Exception:
            return f"{tuple(image.shape)}_{image.dtype}"


class TS_GetImageSizeSide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "large_side": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Large Side",
                        "label_off": "Small Side",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("size",)
    FUNCTION = "execute"
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Get Image Size Side] {message}")

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

    def execute(self, image: torch.Tensor, large_side: bool):
        self._log_tensor("Input", image)

        if image is None:
            self._log("Input is None.")
            return (0,)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        if image.ndim == 3:
            image = image.unsqueeze(0)
            self._log_tensor("Input normalized", image)

        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE with 4 dims [B,H,W,C], got {image.ndim}")

        height = int(image.shape[1])
        width = int(image.shape[2])
        size = max(height, width) if large_side else min(height, width)

        self._log(f"Computed size={size} (large_side={large_side})")
        return (size,)

    @classmethod
    def IS_CHANGED(cls, image: torch.Tensor, large_side: bool) -> str:
        if image is None or not isinstance(image, torch.Tensor):
            return f"none_{large_side}"
        try:
            return f"{tuple(image.shape)}_{image.dtype}_{float(image.mean())}_{large_side}"
        except Exception:
            return f"{tuple(image.shape)}_{image.dtype}_{large_side}"


NODE_CLASS_MAPPINGS = {
    "TS_ImageBatchToImageList": TS_ImageBatchToImageList,
    "TS_ImageListToImageBatch": TS_ImageListToImageBatch,
    "TS_GetImageMegapixels": TS_GetImageMegapixels,
    "TS_GetImageSizeSide": TS_GetImageSizeSide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ImageBatchToImageList": "TS Image Batch to Image List",
    "TS_ImageListToImageBatch": "TS Image List to Image Batch",
    "TS_GetImageMegapixels": "TS Get Image Megapixels",
    "TS_GetImageSizeSide": "TS Get Image Size",
}
