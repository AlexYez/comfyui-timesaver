"""TS Model Converter — convert in-memory model to a target precision/format.

node_id: TS_ModelConverter
"""

import copy
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
        # Convert a deep copy, never the input: ComfyUI caches the incoming
        # MODEL object between runs, so an in-place ``.to(fp8)`` would poison
        # every other consumer of the cached model and re-quantize already
        # quantized weights on each re-run. The transient duplicate is the
        # honest price of a non-destructive conversion. Errors propagate to
        # the pack's runtime guard instead of returning a half-converted model.
        if isinstance(model, ModelPatcher):
            converted = model.clone()
            # clone() shares the underlying torch module — copy it before
            # touching the weights.
            converted.model = copy.deepcopy(model.model)
            converted.model.to(torch.float8_e4m3fn)
        elif hasattr(model, "diffusion_model"):
            converted = copy.deepcopy(model)
            converted.diffusion_model.to(torch.float8_e4m3fn)
        else:
            converted = copy.deepcopy(model)
            converted = converted.to(torch.float8_e4m3fn)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return IO.NodeOutput(converted)


NODE_CLASS_MAPPINGS = {"TS_ModelConverter": TS_ModelConverterNode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ModelConverter": "TS Model Converter"}
