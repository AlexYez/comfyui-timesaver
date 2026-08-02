"""Seed entry point of a TS Image Studio backend workflow."""
from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._marker_shared import CATEGORY, PARAM_TOOLTIP


class TS_StudioInputSeed(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_StudioInputSeed",
            display_name="TS Studio Input (Seed)",
            category=CATEGORY,
            description=(
                "Marker: the noise seed parameter of a TS Image Studio backend "
                "workflow. Connect its INT output to RandomNoise/KSampler seed. "
                "The studio sets the value on every run (manual or randomized)."
            ),
            inputs=[
                IO.Int.Input("value", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             tooltip="The seed the studio replaces at run time."),
                IO.String.Input("param_name", default="seed", tooltip=PARAM_TOOLTIP),
            ],
            outputs=[IO.Int.Output(display_name="seed")],
            search_aliases=["studio marker seed"],
        )

    @classmethod
    def execute(cls, value: int, param_name: str) -> IO.NodeOutput:
        return IO.NodeOutput(int(value))


NODE_CLASS_MAPPINGS = {"TS_StudioInputSeed": TS_StudioInputSeed}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_StudioInputSeed": "TS Studio Input (Seed)"}
