"""TS Model Converter Advanced Direct — convert a connected MODEL directly inside the graph.

node_id: TS_ModelConverterAdvancedDirect
"""

from .ts_model_converter_advanced import TS_ModelConverterAdvancedNode


class TS_ModelConverterAdvancedDirectNode(TS_ModelConverterAdvancedNode):
    """
    Convert loaded MODEL to FP8 (e4m3fn / e5m2) and save to disk.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "fp8_mode": (["e4m3fn", "e5m2"], {"default": "e5m2"}),
                "conversion_preset": (["WAN", "Flux2"], {"default": "WAN"}),
                "shard_subdir": ("STRING", {"multiline": False, "default": "fp8_shards"}),
                "final_filename": ("STRING", {"multiline": False, "default": "converted_model_fp8.safetensors"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "convert_model"
    CATEGORY = "TS/Model Conversion"

    def convert_model(self, model, fp8_mode, conversion_preset, shard_subdir, final_filename):
        return self._convert_loaded_model(model, fp8_mode, conversion_preset, shard_subdir, final_filename)

# ==========================
# Model Scanner


NODE_CLASS_MAPPINGS = {"TS_ModelConverterAdvancedDirect": TS_ModelConverterAdvancedDirectNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ModelConverterAdvancedDirect": "TS Model Converter Advanced Direct"}
