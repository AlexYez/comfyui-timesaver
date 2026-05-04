"""TS Image List to Images — split an IMAGE list into N fixed individual outputs.

Pair node for TS_MultiReference: TS_MultiReference returns `multi_images`
as a list (is_output_list=True), and this node fans the list out into
fixed slots so downstream nodes can consume each reference separately.

Behaviour:
- Always exposes 3 outputs (image_1, image_2, image_3).
- If the input list has fewer items than outputs, the missing slots get
  a 1x64x64x3 zero IMAGE so downstream connections never fail with a
  None/empty error.
- Empty input list (e.g. text-to-image with no references) returns three
  zero IMAGEs and does NOT raise.
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


_OUTPUT_SLOT_COUNT = 3
_FALLBACK_HEIGHT = 64
_FALLBACK_WIDTH = 64
_FALLBACK_CHANNELS = 3


def _zero_image() -> torch.Tensor:
    return torch.zeros(
        (1, _FALLBACK_HEIGHT, _FALLBACK_WIDTH, _FALLBACK_CHANNELS),
        dtype=torch.float32,
    )


def _ensure_bhwc(image: torch.Tensor) -> torch.Tensor:
    """Normalize a single IMAGE tensor to [B, H, W, C]."""
    if not isinstance(image, torch.Tensor):
        return _zero_image()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        return _zero_image()
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
                            "Empty input is allowed; missing slots fall back to a zero IMAGE."
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
        f"splits it into {_OUTPUT_SLOT_COUNT} individual IMAGE outputs. Missing "
        "slots return a 1x64x64 zero IMAGE so downstream connections never "
        "fail. Empty / missing input is a normal state and does not raise."
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

        outputs: list[torch.Tensor] = []
        for index in range(_OUTPUT_SLOT_COUNT):
            if index < len(entries):
                outputs.append(_ensure_bhwc(entries[index]))
            else:
                outputs.append(_zero_image())

        return tuple(outputs)


NODE_CLASS_MAPPINGS = {
    "TS_ImageListToImages": TS_ImageListToImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_ImageListToImages": "TS Image List to Images",
}
