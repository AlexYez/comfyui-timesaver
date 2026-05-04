"""TS Image List to Images — split an IMAGE list into N fixed individual outputs.

Pair node for TS_MultiReference: TS_MultiReference returns `multi_images`
as a list (is_output_list=True), and this node fans the list out into
fixed slots so downstream nodes can consume each reference separately.

Behaviour:
- Always exposes 3 outputs (image_1, image_2, image_3).
- For each output slot, if no real image is available, returns
  ExecutionBlocker so downstream consumers connected to that slot are
  silently skipped (no fake placeholder image that could confuse a
  reference-aware model).
- Empty input list (e.g. text-to-image with no references) means all
  three outputs become ExecutionBlocker — safe pass-through, no error.
- INPUT_IS_LIST=(True,) so a list-output upstream node is delivered as a
  Python list rather than triggering per-item iteration.

V1 API is used here intentionally: V3 IO.Schema does not currently expose
INPUT_IS_LIST semantics, and list-fanout is exactly what V1 INPUT_IS_LIST
was designed for. Public contract (node_id, class name, RETURN_TYPES,
RETURN_NAMES, CATEGORY) is fully stable.

node_id: TS_ImageListToImages
"""

from __future__ import annotations

import torch

from comfy_execution.graph_utils import ExecutionBlocker


_OUTPUT_SLOT_COUNT = 3


def _ensure_bhwc(image):
    """Normalize a single IMAGE tensor to [B, H, W, C], or pass through anything we cannot use."""
    if not isinstance(image, torch.Tensor):
        return None
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        return None
    return image


class TS_ImageListToImages:
    """Split an IMAGE list into a fixed number of individual IMAGE outputs."""

    INPUT_IS_LIST = (True,)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "IMAGE list (e.g. multi_images output from TS_MultiReference). "
                            "Empty/short input is allowed: missing slots become "
                            "ExecutionBlocker so downstream consumers are silently skipped."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",) * _OUTPUT_SLOT_COUNT
    RETURN_NAMES = tuple(f"image_{idx + 1}" for idx in range(_OUTPUT_SLOT_COUNT))
    FUNCTION = "split"
    CATEGORY = "TS/Conditioning"
    DESCRIPTION = (
        "Pairs with TS_MultiReference: receives the multi_images list and "
        f"splits it into {_OUTPUT_SLOT_COUNT} individual IMAGE outputs. "
        "Missing slots emit ExecutionBlocker — downstream consumers are "
        "silently skipped instead of receiving a placeholder image. Empty "
        "or unconnected input is a normal state and does not raise."
    )

    def split(self, images=None):
        # INPUT_IS_LIST wraps the input into a list. When the upstream node
        # produces a list output (is_output_list=True), we get a flat list
        # of tensors. When it produces a single tensor, we get a one-item
        # list. When the input is unconnected (optional), images is None.
        if images is None:
            entries: list = []
        elif isinstance(images, list):
            entries = images
        else:
            entries = [images]

        outputs: list = []
        for index in range(_OUTPUT_SLOT_COUNT):
            if index < len(entries):
                normalized = _ensure_bhwc(entries[index])
                if normalized is not None:
                    outputs.append(normalized)
                    continue
            # No real image for this slot — block downstream silently.
            outputs.append(ExecutionBlocker(None))

        return tuple(outputs)


NODE_CLASS_MAPPINGS = {
    "TS_ImageListToImages": TS_ImageListToImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ImageListToImages": "TS Image List to Images",
}
