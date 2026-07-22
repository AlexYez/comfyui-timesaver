"""TS Model Converter Advanced Direct — convert a connected MODEL directly inside the graph.

node_id: TS_ModelConverterAdvancedDirect
"""

from comfy_api.v0_0_2 import IO

from .ts_model_converter_advanced import TS_ModelConverterAdvancedNode


class TS_ModelConverterAdvancedDirectNode(TS_ModelConverterAdvancedNode):
    """Convert loaded MODEL to FP8 (e4m3fn / e5m2) and save to disk."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ModelConverterAdvancedDirect",
            display_name="TS Model Converter Advanced Direct",
            category="TS/Files",
            inputs=[
                IO.Model.Input("model", tooltip="In-graph MODEL to convert directly to FP8 and save to disk."),
                IO.Combo.Input("fp8_mode", options=["e4m3fn", "e5m2"], default="e5m2", tooltip="Target FP8 format. e4m3fn has more precision, e5m2 a wider range. Pick what your target pipeline expects."),
                IO.Combo.Input("conversion_preset", options=["WAN", "Flux2"], default="WAN", tooltip="Per-architecture rule set deciding which tensors are cast to FP8. Choose the preset matching the model family."),
                IO.String.Input("final_filename", default="converted_model_fp8.safetensors", multiline=False, tooltip="Output safetensors filename, written into ComfyUI's output directory. Path segments are stripped for safety."),
            ],
            outputs=[IO.String.Output(display_name="log", tooltip="Conversion log: target format, preset, counts of converted vs kept tensors, and the output path.")],
        )

    @classmethod
    def execute(cls, model, fp8_mode, conversion_preset, final_filename, shard_subdir=None) -> IO.NodeOutput:
        # `shard_subdir` tolerated for workflows saved with the old widget.
        return IO.NodeOutput(cls._convert_loaded_model(model, fp8_mode, conversion_preset, final_filename))


NODE_CLASS_MAPPINGS = {"TS_ModelConverterAdvancedDirect": TS_ModelConverterAdvancedDirectNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ModelConverterAdvancedDirect": "TS Model Converter Advanced Direct"}
