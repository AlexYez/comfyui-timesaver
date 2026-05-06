"""TS Model Converter — convert in-memory model to a target precision/format.

node_id: TS_ModelConverter
"""

import logging

import torch

from comfy.model_patcher import ModelPatcher
from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_model_converter")
LOG_PREFIX = "[TS Model Converter]"


class TS_ModelConverterNode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ModelConverter",
            display_name="TS Model Converter",
            category="TS/Files",
            inputs=[IO.Model.Input("model")],
            outputs=[IO.Model.Output(display_name="MODEL")],
        )

    @classmethod
    def execute(cls, model) -> IO.NodeOutput:
        try:
            if hasattr(model, "diffusion_model"):
                model.diffusion_model = model.diffusion_model.to(torch.float8_e4m3fn)
            elif isinstance(model, ModelPatcher):
                model.model = model.model.to(torch.float8_e4m3fn)
            else:
                model = model.to(torch.float8_e4m3fn)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return IO.NodeOutput(model)
        except Exception as e:
            logger.error("%s FP8 Conversion Error: %s", LOG_PREFIX, e)
            return IO.NodeOutput(model)


NODE_CLASS_MAPPINGS = {"TS_ModelConverter": TS_ModelConverterNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ModelConverter": "TS Model Converter"}
