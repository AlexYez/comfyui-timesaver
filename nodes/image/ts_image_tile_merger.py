"""TS Image Tile Merger — recombine tiles produced by TS_ImageTileSplitter using TILE_INFO metadata.

node_id: TS_ImageTileMerger
"""

from typing import Any, Dict, Optional
import logging

import torch

from comfy_api.v0_0_2 import IO


logger = logging.getLogger("comfyui_timesaver.ts_image_tile_merger")
LOG_PREFIX = "[TS Image Tile Merger]"


_TileInfo = IO.Custom("TILE_INFO")


class TS_ImageTileMerger(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ImageTileMerger",
            display_name="TS Image Tile Merger",
            category="TS/Image",
            inputs=[
                IO.Image.Input("images"),
                _TileInfo.Input("tile_data"),
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
        cls._log(f"{label} shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")

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

    @classmethod
    def execute(cls, images: torch.Tensor, tile_data: Dict[str, Any]) -> IO.NodeOutput:
        cls._log_tensor("Input tiles", images)

        if images is None or not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected IMAGE tensor, got {type(images)}")

        if tile_data is None or not isinstance(tile_data, dict):
            raise ValueError(f"Expected TILE_INFO dict, got {type(tile_data)}")

        images = cls._ensure_nhwc(images)

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

            weight_mask = cls._build_weight_mask(
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

        cls._log_tensor("Output", output)
        return IO.NodeOutput(output)

    @classmethod
    def fingerprint_inputs(cls, images: torch.Tensor, tile_data: Dict[str, Any]) -> str:
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


NODE_CLASS_MAPPINGS = {"TS_ImageTileMerger": TS_ImageTileMerger}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ImageTileMerger": "TS Image Tile Merger"}
