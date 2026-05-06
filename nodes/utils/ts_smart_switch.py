"""TS Smart Switch — type-aware boolean toggle between two graph branches.

node_id: TS_Smart_Switch
"""

import torch

from comfy_api.latest import IO

from .._shared import TS_Logger


_DATA_TYPES = ["images", "video", "audio", "mask", "string", "int", "float"]


def _is_valid_image(data) -> bool:
    return isinstance(data, torch.Tensor) and data.ndim == 4


def _is_valid_mask(data) -> bool:
    if not isinstance(data, torch.Tensor):
        return False
    return data.ndim == 3 or (data.ndim == 4 and data.shape[-1] == 1)


def _is_valid_video(data) -> bool:
    if isinstance(data, torch.Tensor):
        return data.ndim == 5
    if isinstance(data, (list, tuple)) and data:
        return all(isinstance(item, torch.Tensor) and item.ndim == 4 for item in data)
    return False


def _is_valid_audio(data) -> bool:
    if isinstance(data, dict) and "waveform" in data:
        return isinstance(data["waveform"], torch.Tensor)
    return False


def _is_valid_scalar(data, data_type: str) -> bool:
    if data_type == "string":
        return isinstance(data, str)
    if data_type == "int":
        return isinstance(data, int) and not isinstance(data, bool)
    if data_type == "float":
        return isinstance(data, float)
    return False


def _is_valid_by_type(data, data_type: str) -> bool:
    if data is None:
        return False
    if data_type == "images":
        return _is_valid_image(data)
    if data_type == "video":
        return _is_valid_video(data)
    if data_type == "audio":
        return _is_valid_audio(data)
    if data_type == "mask":
        return _is_valid_mask(data)
    if data_type in ("string", "int", "float"):
        return _is_valid_scalar(data, data_type)
    return False


class TS_Smart_Switch(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Smart_Switch",
            display_name="TS Smart Switch",
            category="TS/Utils",
            description="Smart switch for ANY data. Auto-failover if one input is missing.",
            inputs=[
                IO.Combo.Input("data_type", options=_DATA_TYPES),
                IO.Boolean.Input(
                    "switch",
                    default=True,
                    label_on="Input 1",
                    label_off="Input 2",
                ),
                IO.AnyType.Input("input_1", optional=True),
                IO.AnyType.Input("input_2", optional=True),
            ],
            outputs=[IO.AnyType.Output(display_name="output")],
        )

    @classmethod
    def execute(cls, data_type: str, switch: bool, input_1=None, input_2=None) -> IO.NodeOutput:
        try:
            valid_1 = _is_valid_by_type(input_1, data_type)
            valid_2 = _is_valid_by_type(input_2, data_type)

            if input_1 is not None and not valid_1:
                TS_Logger.warn(
                    "SmartSwitch",
                    f"Input 1 ignored: type mismatch for data_type={data_type}",
                )
            if input_2 is not None and not valid_2:
                TS_Logger.warn(
                    "SmartSwitch",
                    f"Input 2 ignored: type mismatch for data_type={data_type}",
                )

            if valid_1 and valid_2:
                if switch:
                    result = input_1
                    selected_source = "Input 1"
                    status_msg = "(Switch: ON)"
                else:
                    result = input_2
                    selected_source = "Input 2"
                    status_msg = "(Switch: OFF)"
            elif valid_1:
                result = input_1
                selected_source = "Input 1"
                status_msg = "(Auto-Failover)"
            elif valid_2:
                result = input_2
                selected_source = "Input 2"
                status_msg = "(Auto-Failover)"
            else:
                TS_Logger.warn("SmartSwitch", "Both inputs are None.")
                return IO.NodeOutput(None)

            info = "Unknown"
            if hasattr(result, "shape"):
                info = f"Tensor {result.shape}"
            elif isinstance(result, (int, float, str)):
                info = str(result)

            TS_Logger.log(
                "SmartSwitch",
                f"Selected: {selected_source} {status_msg} | {info}",
            )
            return IO.NodeOutput(result)

        except Exception as e:
            TS_Logger.error("SmartSwitch", f"Error: {str(e)}")
            return IO.NodeOutput(None)


NODE_CLASS_MAPPINGS = {"TS_Smart_Switch": TS_Smart_Switch}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Smart_Switch": "TS Smart Switch"}
