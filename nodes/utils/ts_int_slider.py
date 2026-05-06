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
            description="Int slider (320 - 2048)",
            inputs=[
                IO.Int.Input(
                    "value",
                    default=512,
                    min=-2147483648,
                    max=2147483647,
                    step=8,
                    display_mode=IO.NumberDisplay.slider,
                ),
            ],
            outputs=[IO.Int.Output(display_name="int_value")],
        )

    @classmethod
    def execute(cls, value: int) -> IO.NodeOutput:
        TS_Logger.log("IntSlider", f"Value: {value}")
        return IO.NodeOutput(int(value))


NODE_CLASS_MAPPINGS = {"TS_Int_Slider": TS_Int_Slider}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Int_Slider": "TS Int Slider"}
