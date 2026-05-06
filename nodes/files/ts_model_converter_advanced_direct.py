"""TS Model Converter Advanced Direct — convert a connected MODEL directly inside the graph.

node_id: TS_ModelConverterAdvancedDirect
"""

from comfy_api.latest import IO

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
                IO.Model.Input("model"),
                IO.Combo.Input("fp8_mode", options=["e4m3fn", "e5m2"], default="e5m2"),
                IO.Combo.Input("conversion_preset", options=["WAN", "Flux2"], default="WAN"),
                IO.String.Input("shard_subdir", default="fp8_shards", multiline=False),
                IO.String.Input("final_filename", default="converted_model_fp8.safetensors", multiline=False),
            ],
            outputs=[IO.String.Output(display_name="log")],
        )

    @classmethod
    def execute(cls, model, fp8_mode, conversion_preset, shard_subdir, final_filename) -> IO.NodeOutput:
        return IO.NodeOutput(cls._convert_loaded_model(model, fp8_mode, conversion_preset, shard_subdir, final_filename))


NODE_CLASS_MAPPINGS = {"TS_ModelConverterAdvancedDirect": TS_ModelConverterAdvancedDirectNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ModelConverterAdvancedDirect": "TS Model Converter Advanced Direct"}
