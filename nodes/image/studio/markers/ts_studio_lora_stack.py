"""LoRA-chain insertion point of a TS Image Studio backend workflow."""
from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._marker_shared import CATEGORY


class TS_StudioLoraStack(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_StudioLoraStack",
            display_name="TS Studio LoRA Stack",
            category=CATEGORY,
            description=(
                "Marker: where the studio inserts the user's LoRA chain. Place "
                "it between the model loader and the sampler. At run time the "
                "studio patcher expands it into a chain of LoraLoaderModelOnly "
                "nodes (one per stacked LoRA, in order, with its strength); an "
                "empty stack collapses to a direct connection. Standalone it is "
                "a plain passthrough."
            ),
            inputs=[IO.Model.Input("model", tooltip="Model to receive the LoRA chain.")],
            outputs=[IO.Model.Output(display_name="model")],
            search_aliases=["studio marker lora", "studio lora chain"],
        )

    @classmethod
    def execute(cls, model) -> IO.NodeOutput:
        return IO.NodeOutput(model)


NODE_CLASS_MAPPINGS = {"TS_StudioLoraStack": TS_StudioLoraStack}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_StudioLoraStack": "TS Studio LoRA Stack"}
