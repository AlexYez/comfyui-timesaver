"""TS Load LoRA, model only (locked) — штатный LoraLoaderModelOnly для `.tsmodel`.

Отдельная нода, а не флаг у соседней: у видеомоделей и дистиллятов CLIP в графе
часто просто нет, и вход, который нечем заполнить, там мешает.
"""

from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._locked import options
from .ts_locked_lora import CATEGORY_FOLDER, apply_locked_lora


class TS_LockedLoraModelOnly(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LockedLoraModelOnly",
            display_name="TS Load LoRA, model only",
            category="TS/Loaders",
            description=(
                "Apply a locked .tsmodel LoRA to a model alone, the way LoraLoaderModelOnly "
                "does — for graphs that carry no CLIP at all."
            ),
            inputs=[
                IO.Model.Input("model", tooltip="The model the LoRA is applied to."),
                IO.Combo.Input(
                    "lora_name",
                    options=options(CATEGORY_FOLDER),
                    tooltip="A .tsmodel file from models/loras.",
                ),
                IO.Float.Input(
                    "strength_model", default=1.0, min=-100.0, max=100.0, step=0.01,
                    tooltip="How strongly the LoRA affects the model. 0 leaves it untouched.",
                ),
            ],
            outputs=[IO.Model.Output(display_name="model")],
            search_aliases=["locked", "tsmodel", "lora loader", "lora model only"],
        )

    @classmethod
    def execute(cls, model, lora_name: str, strength_model: float) -> IO.NodeOutput:
        patched_model, _clip = apply_locked_lora(model, None, lora_name, strength_model, 0)
        return IO.NodeOutput(patched_model)


NODE_CLASS_MAPPINGS = {"TS_LockedLoraModelOnly": TS_LockedLoraModelOnly}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LockedLoraModelOnly": "TS Load LoRA, model only"}
