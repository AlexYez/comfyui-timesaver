"""Image entry point of a TS Image Studio backend workflow."""
from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._marker_shared import (
    CATEGORY,
    LABEL_TOOLTIP,
    PARAM_TOOLTIP,
    annotated_input_path,
    file_fingerprint,
    load_image_tensor,
)


class TS_StudioInputImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_StudioInputImage",
            display_name="TS Studio Input (Image)",
            category=CATEGORY,
            description=(
                "Marker: an image parameter of a TS Image Studio backend "
                "workflow (source image, reference 1-3...). The studio uploads "
                "the picture through /upload/image and sets its annotated name "
                "here. Loads like LoadImage: IMAGE plus a MASK from alpha."
            ),
            inputs=[
                IO.String.Input(
                    "value", default="", socketless=True,
                    tooltip="Annotated upload name ('sub/file.png [input]') the studio sets at run time.",
                ),
                IO.String.Input("param_name", default="source_image", tooltip=PARAM_TOOLTIP),
                IO.String.Input("label", default="", optional=True, tooltip=LABEL_TOOLTIP),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Mask.Output(display_name="mask"),
            ],
            search_aliases=["studio marker image", "studio reference input"],
        )

    @classmethod
    def execute(cls, value: str, param_name: str, label: str = "") -> IO.NodeOutput:
        image, mask = load_image_tensor(annotated_input_path(value))
        return IO.NodeOutput(image, mask)

    @classmethod
    def fingerprint_inputs(cls, value: str, param_name: str, label: str = "") -> str:
        return file_fingerprint(value)


NODE_CLASS_MAPPINGS = {"TS_StudioInputImage": TS_StudioInputImage}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_StudioInputImage": "TS Studio Input (Image)"}
