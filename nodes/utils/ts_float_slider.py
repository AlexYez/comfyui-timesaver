"""TS Float Slider — slider widget that returns a FLOAT value.

node_id: TS_FloatSlider
"""

from .._shared import TS_Logger


class TS_FloatSlider:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.5, "min": -1000000000.0, "max": 1000000000.0, "step": 0.1,
                    "display": "slider", "round": 0.01
                }),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float_value",)
    FUNCTION = "get_value"
    CATEGORY = "TS Tools/Sliders"
    DESCRIPTION = "Float slider (0.0 - 1.0)"

    def get_value(self, value):
        TS_Logger.log("FloatSlider", f"Value: {value:.2f}")
        return (float(value),)


NODE_CLASS_MAPPINGS = {"TS_FloatSlider": TS_FloatSlider}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_FloatSlider": "TS Float Slider"}
