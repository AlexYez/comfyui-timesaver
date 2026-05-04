"""TS Smart Switch — type-aware boolean toggle between two graph branches.

node_id: TS_Smart_Switch
"""

from collections.abc import Mapping

import torch

from .._shared import TS_Logger


class TS_Smart_Switch:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_type": (["images", "video", "audio", "mask", "string", "int", "float"],),
                "switch": ("BOOLEAN", {"default": True, "label_on": "Input 1", "label_off": "Input 2"}),
            },
            "optional": {
                "input_1": ("*",), 
                "input_2": ("*",),
            }
        }

    RETURN_TYPES = ("*",) 
    RETURN_NAMES = ("output",)
    FUNCTION = "smart_switch"
    CATEGORY = "TS Tools/Logic"
    DESCRIPTION = "Smart switch for ANY data. Auto-failover if one input is missing."

    def _is_valid_image(self, data):
        return isinstance(data, torch.Tensor) and data.ndim == 4

    def _is_valid_mask(self, data):
        if not isinstance(data, torch.Tensor):
            return False
        return data.ndim == 3 or (data.ndim == 4 and data.shape[-1] == 1)

    def _is_valid_video(self, data):
        if isinstance(data, torch.Tensor):
            return data.ndim == 5
        if isinstance(data, (list, tuple)) and data:
            return all(isinstance(item, torch.Tensor) and item.ndim == 4 for item in data)
        return False

    def _is_valid_audio(self, data):
        if isinstance(data, dict) and "waveform" in data:
            return isinstance(data["waveform"], torch.Tensor)
        return False

    def _is_valid_scalar(self, data, data_type):
        if data_type == "string":
            return isinstance(data, str)
        if data_type == "int":
            return isinstance(data, int) and not isinstance(data, bool)
        if data_type == "float":
            return isinstance(data, float)
        return False

    def _is_valid_by_type(self, data, data_type):
        if data is None:
            return False
        if data_type == "images":
            return self._is_valid_image(data)
        if data_type == "video":
            return self._is_valid_video(data)
        if data_type == "audio":
            return self._is_valid_audio(data)
        if data_type == "mask":
            return self._is_valid_mask(data)
        if data_type in ("string", "int", "float"):
            return self._is_valid_scalar(data, data_type)
        return False

    def smart_switch(self, switch, data_type="images", input_1=None, input_2=None):
        selected_source = "None"
        result = None
        status_msg = ""

        try:
            valid_1 = self._is_valid_by_type(input_1, data_type)
            valid_2 = self._is_valid_by_type(input_2, data_type)

            if input_1 is not None and not valid_1:
                TS_Logger.log("SmartSwitch", f"Input 1 ignored: type mismatch for data_type={data_type}", "yellow")
            if input_2 is not None and not valid_2:
                TS_Logger.log("SmartSwitch", f"Input 2 ignored: type mismatch for data_type={data_type}", "yellow")

            has_input_1 = valid_1
            has_input_2 = valid_2

            if has_input_1 and has_input_2:
                if switch:
                    result = input_1
                    selected_source = "Input 1"
                    status_msg = "(Switch: ON)"
                else:
                    result = input_2
                    selected_source = "Input 2"
                    status_msg = "(Switch: OFF)"
            elif has_input_1:
                result = input_1
                selected_source = "Input 1"
                status_msg = "(Auto-Failover)"
            elif has_input_2:
                result = input_2
                selected_source = "Input 2"
                status_msg = "(Auto-Failover)"
            else:
                TS_Logger.error("SmartSwitch", "Warning: Both inputs are None.")
                return (None,)

            # Log info
            info = "Unknown"
            if hasattr(result, 'shape'): info = f"Tensor {result.shape}"
            elif isinstance(result, (int, float, str)): info = str(result)
            
            TS_Logger.log("SmartSwitch", f"Selected: {selected_source} {status_msg} | {info}")
            return (result,)

        except Exception as e:
            TS_Logger.error("SmartSwitch", f"Error: {str(e)}")
            return (None,)


NODE_CLASS_MAPPINGS = {"TS_Smart_Switch": TS_Smart_Switch}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Smart_Switch": "TS Smart Switch"}
