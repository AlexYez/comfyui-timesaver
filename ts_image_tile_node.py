import math
from typing import Optional, Dict, Any, List

import torch


class TS_ImageTileSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "tile_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 512, "step": 8}),
                "feather": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TILE_INFO")
    RETURN_NAMES = ("tiles", "tile_data")
    FUNCTION = "execute"
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Image Tile Splitter] {message}")

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
            raise ValueError(f"Expected IMAGE with 3 or 4 dims [B,H,W,C], got {image.ndim}")
        return image

    @staticmethod
    def _clamp_overlap(tile_w: int, tile_h: int, overlap: int) -> int:
        max_overlap = max(0, min(tile_w - 1, tile_h - 1))
        return max(0, min(overlap, max_overlap))

    def execute(
        self,
        image: torch.Tensor,
        tile_width: int,
        tile_height: int,
        overlap: int,
        feather: float,
    ):
        self._log_tensor("Input", image)

        if image is None or not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        image = self._ensure_bhwc(image)
        batch_size, img_h, img_w, _ = image.shape

        tile_width = max(1, min(int(tile_width), img_w))
        tile_height = max(1, min(int(tile_height), img_h))
        overlap = self._clamp_overlap(tile_width, tile_height, int(overlap))

        stride_w = max(1, tile_width - overlap)
        stride_h = max(1, tile_height - overlap)

        cols = max(1, math.ceil((img_w - overlap) / stride_w))
        rows = max(1, math.ceil((img_h - overlap) / stride_h))

        tile_data: Dict[str, Any] = {
            "original_height": img_h,
            "original_width": img_w,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "overlap": overlap,
            "feather": float(feather),
            "batch_size": batch_size,
            "rows": rows,
            "cols": cols,
            "positions": [],
        }

        results_tiles: List[torch.Tensor] = []

        for b in range(batch_size):
            for r in range(rows):
                for c in range(cols):
                    x = c * stride_w
                    y = r * stride_h

                    if x + tile_width > img_w:
                        x = img_w - tile_width
                    if y + tile_height > img_h:
                        y = img_h - tile_height
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0

                    crop = image[b : b + 1, y : y + tile_height, x : x + tile_width, :]
                    results_tiles.append(crop)

                    tile_data["positions"].append(
                        {"batch_index": b, "x": x, "y": y, "row": r, "col": c}
                    )

        if not results_tiles:
            self._log("No tiles produced, returning original image.")
            return (image, tile_data)

        final_tiles = torch.cat(results_tiles, dim=0)

        self._log_tensor("Output tiles", final_tiles)
        self._log(f"Tiles={final_tiles.shape[0]} Grid={rows}x{cols}")

        return (final_tiles, tile_data)

    @classmethod
    def IS_CHANGED(
        cls,
        image: torch.Tensor,
        tile_width: int,
        tile_height: int,
        overlap: int,
        feather: float,
    ) -> str:
        if image is None or not isinstance(image, torch.Tensor):
            return "none"
        try:
            return (
                f"{tuple(image.shape)}_{image.dtype}_{tile_width}_{tile_height}_"
                f"{overlap}_{feather}_{float(image.mean())}"
            )
        except Exception:
            return f"{tuple(image.shape)}_{image.dtype}_{tile_width}_{tile_height}_{overlap}_{feather}"


class TS_ImageTileMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tile_data": ("TILE_INFO",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "TS/Image Tools"

    @staticmethod
    def _log(message: str) -> None:
        print(f"[TS Image Tile Merger] {message}")

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
    def _ensure_nhwc(images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            return images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected IMAGE with 3 or 4 dims [N,H,W,C], got {images.ndim}")
        return images

    @staticmethod
    def _build_weight_mask(
        height: int,
        width: int,
        feather_ratio: float,
        x: int,
        y: int,
        img_w: int,
        img_h: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.ones((height, width, 1), dtype=dtype, device=device)
        if feather_ratio <= 0.0:
            return mask

        feather_w = int(width * feather_ratio)
        feather_h = int(height * feather_ratio)

        if feather_w > 0:
            grad_x = torch.linspace(0.0, 1.0, feather_w, device=device, dtype=dtype)
            if x > 0:
                mask[:, :feather_w, 0] *= grad_x
            if x + width < img_w:
                mask[:, -feather_w:, 0] *= torch.flip(grad_x, dims=(0,))

        if feather_h > 0:
            grad_y = torch.linspace(0.0, 1.0, feather_h, device=device, dtype=dtype)
            if y > 0:
                mask[:feather_h, :, 0] *= grad_y.unsqueeze(1)
            if y + height < img_h:
                mask[-feather_h:, :, 0] *= torch.flip(grad_y, dims=(0,)).unsqueeze(1)

        return mask

    def execute(self, images: torch.Tensor, tile_data: Dict[str, Any]):
        self._log_tensor("Input tiles", images)

        if images is None or not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(images)}")

        if tile_data is None or not isinstance(tile_data, dict):
            raise ValueError(f"Expected TILE_INFO dict, got {type(tile_data)}")

        images = self._ensure_nhwc(images)

        orig_h = int(tile_data.get("original_height", images.shape[1]))
        orig_w = int(tile_data.get("original_width", images.shape[2]))
        batch_size = int(tile_data.get("batch_size", 1))
        tile_width = int(tile_data.get("tile_width", images.shape[2]))
        tile_height = int(tile_data.get("tile_height", images.shape[1]))
        feather_ratio = float(tile_data.get("feather", tile_data.get("feather_ratio", 0.0)))
        positions = tile_data.get("positions", [])

        device = images.device
        dtype = images.dtype if images.is_floating_point() else torch.float32

        channels = images.shape[-1]
        output = torch.zeros((batch_size, orig_h, orig_w, channels), dtype=dtype, device=device)
        weights = torch.zeros((batch_size, orig_h, orig_w, 1), dtype=dtype, device=device)

        for idx, pos in enumerate(positions):
            if idx >= images.shape[0]:
                break
            if not isinstance(pos, dict):
                continue

            tile = images[idx]
            b_idx = int(pos.get("batch_index", 0))
            x = int(pos.get("x", 0))
            y = int(pos.get("y", 0))

            if b_idx < 0 or b_idx >= batch_size:
                continue

            tile_h, tile_w, _ = tile.shape

            if tile_h != tile_height or tile_w != tile_width:
                tile_bchw = tile.permute(2, 0, 1).unsqueeze(0)
                tile_resized = torch.nn.functional.interpolate(
                    tile_bchw,
                    size=(tile_height, tile_width),
                    mode="bilinear",
                    align_corners=False,
                )
                tile = tile_resized.squeeze(0).permute(1, 2, 0)
                tile_h, tile_w, _ = tile.shape

            if x + tile_w > orig_w:
                x = max(0, orig_w - tile_w)
            if y + tile_h > orig_h:
                y = max(0, orig_h - tile_h)

            weight_mask = self._build_weight_mask(
                tile_h,
                tile_w,
                feather_ratio,
                x,
                y,
                orig_w,
                orig_h,
                device,
                dtype,
            )

            output[b_idx, y : y + tile_h, x : x + tile_w, :] += tile.to(dtype) * weight_mask
            weights[b_idx, y : y + tile_h, x : x + tile_w, :] += weight_mask

        weights[weights == 0] = 1.0
        output = output / weights

        self._log_tensor("Output", output)
        return (output,)

    @classmethod
    def IS_CHANGED(cls, images: torch.Tensor, tile_data: Dict[str, Any]) -> str:
        if images is None or not isinstance(images, torch.Tensor):
            return "none"
        if tile_data is None or not isinstance(tile_data, dict):
            return "none"
        try:
            return (
                f"{tuple(images.shape)}_{images.dtype}_{tile_data.get('original_width')}_"
                f"{tile_data.get('original_height')}_{tile_data.get('tile_width')}_"
                f"{tile_data.get('tile_height')}_{tile_data.get('overlap')}_"
                f"{tile_data.get('feather', tile_data.get('feather_ratio'))}"
            )
        except Exception:
            return f"{tuple(images.shape)}_{images.dtype}"


NODE_CLASS_MAPPINGS = {
    "TS_ImageTileSplitter": TS_ImageTileSplitter,
    "TS_ImageTileMerger": TS_ImageTileMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ImageTileSplitter": "TS Image Tile Splitter",
    "TS_ImageTileMerger": "TS Image Tile Merger",
}
