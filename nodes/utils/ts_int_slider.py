"""TS Int Slider — slider widget that returns an INT value.

node_id: TS_Int_Slider
"""

from comfy_api.v0_0_2 import IO

from .._shared import TS_Logger


class TS_Int_Slider(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Int_Slider",
            display_name="TS Int Slider",
            category="TS/Utils",
            description=(
                "Integer slider for positive settings such as steps, seconds or resolution. "
                "A new node spans 0 - 2048; set min, max and step in the node's properties."
            ),
            inputs=[
                IO.Int.Input(
                    "value",
                    default=512,
                    # Hard limits only; the working range lives in node.properties
                    # (js/utils/sliders/_slider_helpers.js, TS_SLIDER_SPECS.limits).
                    min=0,
                    max=100000,
                    step=8,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Integer value emitted by the slider.",
                ),
            ],
            outputs=[
                IO.Int.Output(
                    display_name="int_value",
                    tooltip="The slider's integer value.",
                )
            ],
        )

    @classmethod
    def execute(cls, value: int) -> IO.NodeOutput:
        TS_Logger.log("IntSlider", f"Value: {value}")
        return IO.NodeOutput(int(value))


NODE_CLASS_MAPPINGS = {"TS_Int_Slider": TS_Int_Slider}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Int_Slider": "TS Int Slider"}
