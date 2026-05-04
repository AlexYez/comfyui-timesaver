"""TS Int Slider — slider widget that returns an INT value.

node_id: TS_Int_Slider
"""

from .._shared import TS_Logger


class TS_Int_Slider:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {
                    "default": 512, "min": -2147483648, "max": 2147483647, "step": 8,
                    "display": "slider" 
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int_value",)
    FUNCTION = "get_value"
    CATEGORY = "TS Tools/Sliders"
    DESCRIPTION = "Int slider (320 - 2048)"

    def get_value(self, value):
        TS_Logger.log("IntSlider", f"Value: {value}")
        return (int(value),)


NODE_CLASS_MAPPINGS = {"TS_Int_Slider": TS_Int_Slider}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Int_Slider": "TS Int Slider"}
