"""Numeric entry point of a TS Image Studio backend workflow."""
from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._marker_shared import CATEGORY, LABEL_TOOLTIP, PARAM_TOOLTIP


class TS_StudioInputNumber(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_StudioInputNumber",
            display_name="TS Studio Input (Number)",
            category=CATEGORY,
            description=(
                "Marker: a numeric parameter of a TS Image Studio backend "
                "workflow (width, height, steps, cfg, denoise...). Both FLOAT "
                "and INT outputs are provided so it connects to either kind of "
                "input; INT is the rounded value."
            ),
            inputs=[
                IO.Float.Input("value", default=0.0, min=-1e9, max=1e9, step=0.01,
                               tooltip="The number the studio replaces at run time."),
                IO.String.Input("param_name", default="steps", tooltip=PARAM_TOOLTIP),
                IO.String.Input("label", default="", optional=True, tooltip=LABEL_TOOLTIP),
            ],
            outputs=[
                IO.Float.Output(display_name="float"),
                IO.Int.Output(display_name="int"),
            ],
            search_aliases=["studio marker number", "studio int input", "studio float input"],
        )

    @classmethod
    def execute(cls, value: float, param_name: str, label: str = "") -> IO.NodeOutput:
        return IO.NodeOutput(float(value), int(round(value)))


NODE_CLASS_MAPPINGS = {"TS_StudioInputNumber": TS_StudioInputNumber}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_StudioInputNumber": "TS Studio Input (Number)"}
