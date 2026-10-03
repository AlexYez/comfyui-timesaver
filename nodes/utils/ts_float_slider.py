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
            description=(
                "Float slider for positive settings such as fps, megapixels or strength. "
                "A new node spans 0 - 10; set min, max and step in the node's properties."
            ),
            inputs=[
                IO.Float.Input(
                    "value",
                    default=0.5,
                    # Hard limits only; the working range lives in node.properties
                    # (js/utils/sliders/_slider_helpers.js, TS_SLIDER_SPECS.limits).
                    min=0.0,
                    max=10000.0,
                    step=0.1,
                    round=0.01,
                    display_mode=IO.NumberDisplay.slider,
                    tooltip="Float value emitted by the slider.",
                ),
            ],
            outputs=[
                IO.Float.Output(
                    display_name="float_value",
                    tooltip="The slider's float value.",
                )
            ],
        )

    @classmethod
    def execute(cls, value: float) -> IO.NodeOutput:
        TS_Logger.log("FloatSlider", f"Value: {value:.2f}")
        return IO.NodeOutput(float(value))


NODE_CLASS_MAPPINGS = {"TS_FloatSlider": TS_FloatSlider}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_FloatSlider": "TS Float Slider"}
