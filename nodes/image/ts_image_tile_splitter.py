"""TS Image Tile Splitter — split a large image into overlapping tiles.

node_id: TS_ImageTileSplitter
"""

import math
from typing import Any, Dict, List, Optional
import logging

import torch

from comfy_api.latest import IO


logger = logging.getLogger("comfyui_timesaver.ts_image_tile_splitter")
LOG_PREFIX = "[TS Image Tile Splitter]"


_TileInfo = IO.Custom("TILE_INFO")


class TS_ImageTileSplitter(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ImageTileSplitter",
            display_name="TS Image Tile Splitter",
            category="TS/Image",
            inputs=[
                IO.Image.Input("image"),
                IO.Int.Input("tile_width", default=1024, min=64, max=8192, step=8),
                IO.Int.Input("tile_height", default=1024, min=64, max=8192, step=8),
                IO.Int.Input("overlap", default=128, min=0, max=512, step=8),
                IO.Float.Input("feather", default=0.1, min=0.0, max=0.5, step=0.01),
            ],
            outputs=[
                IO.Image.Output(display_name="tiles"),
                _TileInfo.Output(display_name="tile_data"),
            ],
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
        cls._log(f"{label} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")

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

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        tile_width: int,
        tile_height: int,
        overlap: int,
        feather: float,
    ) -> IO.NodeOutput:
        cls._log_tensor("Input", image)

        if image is None or not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(image)}")

        image = cls._ensure_bhwc(image)
        batch_size, img_h, img_w, _ = image.shape

        tile_width = max(1, min(int(tile_width), img_w))
        tile_height = max(1, min(int(tile_height), img_h))
        overlap = cls._clamp_overlap(tile_width, tile_height, int(overlap))

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
            cls._log("No tiles produced, returning original image.")
            return IO.NodeOutput(image, tile_data)

        final_tiles = torch.cat(results_tiles, dim=0)

        cls._log_tensor("Output tiles", final_tiles)
        cls._log(f"Tiles={final_tiles.shape[0]} Grid={rows}x{cols}")

        return IO.NodeOutput(final_tiles, tile_data)

    @classmethod
    def fingerprint_inputs(
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


NODE_CLASS_MAPPINGS = {"TS_ImageTileSplitter": TS_ImageTileSplitter}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ImageTileSplitter": "TS Image Tile Splitter"}
