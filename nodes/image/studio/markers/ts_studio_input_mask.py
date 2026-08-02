"""Mask entry point of a TS Image Studio backend workflow."""
from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._marker_shared import (
    CATEGORY,
    LABEL_TOOLTIP,
    PARAM_TOOLTIP,
    annotated_input_path,
    file_fingerprint,
    load_mask_tensor,
)


class TS_StudioInputMask(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_StudioInputMask",
            display_name="TS Studio Input (Mask)",
            category=CATEGORY,
            description=(
                "Marker: the inpaint mask parameter of a TS Image Studio "
                "backend workflow. The studio saves the painted mask as a PNG "
                "(white = regenerate) and sets its annotated name here."
            ),
            inputs=[
                IO.String.Input(
                    "value", default="", socketless=True,
                    tooltip="Annotated upload name of the mask PNG the studio sets at run time.",
                ),
                IO.String.Input("param_name", default="mask", tooltip=PARAM_TOOLTIP),
                IO.String.Input("label", default="", optional=True, tooltip=LABEL_TOOLTIP),
            ],
            outputs=[IO.Mask.Output(display_name="mask")],
            search_aliases=["studio marker mask", "studio inpaint mask"],
        )

    @classmethod
    def execute(cls, value: str, param_name: str, label: str = "") -> IO.NodeOutput:
        return IO.NodeOutput(load_mask_tensor(annotated_input_path(value)))

    @classmethod
    def fingerprint_inputs(cls, value: str, param_name: str, label: str = "") -> str:
        return file_fingerprint(value)


NODE_CLASS_MAPPINGS = {"TS_StudioInputMask": TS_StudioInputMask}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_StudioInputMask": "TS Studio Input (Mask)"}
