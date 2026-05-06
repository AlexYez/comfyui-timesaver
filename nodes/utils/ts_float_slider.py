"""TS Float Slider — slider widget that returns a FLOAT value.

node_id: TS_FloatSlider
"""

from comfy_api.v0_0_2 import IO

from .._shared import TS_Logger


class TS_FloatSlider(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_FloatSlider",
            display_name="TS Float Slider",
            category="TS/Utils",
            description="Float slider (0.0 - 1.0)",
            inputs=[
                IO.Float.Input(
                    "value",
                    default=0.5,
                    min=-1000000000.0,
                    max=1000000000.0,
                    step=0.1,
                    round=0.01,
                    display_mode=IO.NumberDisplay.slider,
                ),
            ],
            outputs=[IO.Float.Output(display_name="float_value")],
        )

    @classmethod
    def execute(cls, value: float) -> IO.NodeOutput:
        TS_Logger.log("FloatSlider", f"Value: {value:.2f}")
        return IO.NodeOutput(float(value))


NODE_CLASS_MAPPINGS = {"TS_FloatSlider": TS_FloatSlider}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_FloatSlider": "TS Float Slider"}
