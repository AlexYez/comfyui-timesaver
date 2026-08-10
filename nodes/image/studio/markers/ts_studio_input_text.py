"""Text entry point of a TS Image Studio backend workflow."""
from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._marker_shared import CATEGORY, LABEL_TOOLTIP, PARAM_TOOLTIP


class TS_StudioInputText(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_StudioInputText",
            display_name="TS Studio Input (Text)",
            category=CATEGORY,
            description=(
                "Marker: a text parameter of a TS Image Studio backend workflow. "
                "The studio finds it by param_name and fills the value (prompt, "
                "negative prompt, edit instruction...). Standalone it is just a "
                "string source."
            ),
            inputs=[
                IO.String.Input("value", default="", multiline=True,
                                tooltip="The text the studio replaces at run time."),
                IO.String.Input("param_name", default="prompt", tooltip=PARAM_TOOLTIP),
                IO.String.Input("label", default="", optional=True, tooltip=LABEL_TOOLTIP),
            ],
            outputs=[IO.String.Output(display_name="text")],
            search_aliases=["studio marker text", "studio prompt input"],
        )

    @classmethod
    def execute(cls, value: str, param_name: str, label: str = "") -> IO.NodeOutput:
        return IO.NodeOutput(value)


NODE_CLASS_MAPPINGS = {"TS_StudioInputText": TS_StudioInputText}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_StudioInputText": "TS Studio Input (Text)"}
