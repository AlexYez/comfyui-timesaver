"""TS Batch Load Image — one picture per call, read from a path.

node_id: TS_BatchLoadImage

The companion to ``TS Batch Source``. The source hands out paths; this node
turns one path into one image, and ComfyUI calls it once per item in the list.

⚠️ That split is what keeps the memory flat. A source that emitted the pictures
themselves would hold the whole set in its output cache — around 10 GB for a
hundred 4K frames — before the first result is written. Here the peak is a
single frame, no matter how long the job is.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from comfy_api.v0_0_2 import IO
from PIL import Image, ImageOps

from ._image_utils import pil2tensor

logger = logging.getLogger("comfyui_timesaver.ts_batch_load_image")
LOG_PREFIX = "[TS Batch Load Image]"


def _clean_path(path: str) -> Path:
    """The user's path, with the quotes Explorer's "Copy as path" adds removed."""
    return Path(str(path).strip().strip('"'))


class TS_BatchLoadImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_BatchLoadImage",
            display_name="TS Batch Load Image",
            category="TS/Image/Batch",
            description=(
                "Read one image from a file path. Wire it to TS Batch Source to walk a "
                "folder one picture at a time, keeping memory to a single frame."
            ),
            inputs=[
                IO.String.Input(
                    "path",
                    default="",
                    tooltip="Full path to an image file. Usually the 'item' output of TS Batch Source.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image", tooltip="The loaded image [1,H,W,C]."),
                IO.Mask.Output(
                    display_name="mask",
                    tooltip="Alpha channel as a mask, or a fully opaque mask when the file has none.",
                ),
                IO.String.Output(
                    display_name="name",
                    tooltip="File name without its extension — handy for naming the result after the source.",
                ),
            ],
            search_aliases=["load image from path", "batch image", "image path"],
        )

    @classmethod
    def fingerprint_inputs(cls, path) -> str:
        """Re-read when the file itself changed, not just when the path did."""
        target = _clean_path(path)
        try:
            info = target.stat()
            return f"{target}|{info.st_mtime_ns}|{info.st_size}"
        except OSError:
            return f"{target}|missing"

    @classmethod
    def execute(cls, path) -> IO.NodeOutput:
        target = _clean_path(path)
        if not str(target).strip():
            raise ValueError(f"{LOG_PREFIX} No path given.")
        if not target.is_file():
            raise FileNotFoundError(f"{LOG_PREFIX} Not a file: {target}")

        with Image.open(target) as opened:
            # EXIF orientation first: a phone photo is stored sideways, and a
            # caption written from the sideways version describes the wrong thing.
            image = ImageOps.exif_transpose(opened)
            has_alpha = "A" in image.getbands()
            alpha = np.array(image.getchannel("A")).astype(np.float32) / 255.0 if has_alpha else None
            rgb = image.convert("RGB")
            tensor = pil2tensor(rgb)

        if alpha is not None:
            mask = 1.0 - torch.from_numpy(alpha).unsqueeze(0)
        else:
            mask = torch.zeros((1, tensor.shape[1], tensor.shape[2]), dtype=torch.float32)

        logger.info("%s %s -> %s", LOG_PREFIX, target.name, tuple(tensor.shape))
        return IO.NodeOutput(tensor, mask, target.stem)


NODE_CLASS_MAPPINGS = {"TS_BatchLoadImage": TS_BatchLoadImage}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_BatchLoadImage": "TS Batch Load Image"}
