"""Result exit point of a TS Image Studio backend workflow."""
from __future__ import annotations

from comfy_api.v0_0_2 import IO, UI

from ._marker_shared import CATEGORY


class TS_StudioOutput(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_StudioOutput",
            display_name="TS Studio Output",
            category=CATEGORY,
            description=(
                "Marker: THE result of a TS Image Studio backend workflow. "
                "Saves the image under the studio session prefix and reports it "
                "back through the executed event — this is how the gallery "
                "learns a run finished. One per workflow; debug SaveImage nodes "
                "elsewhere are ignored by the studio."
            ),
            inputs=[
                IO.Image.Input("image", tooltip="The final image of the backend workflow."),
                IO.String.Input(
                    "filename_prefix", default="ts_studio/session",
                    tooltip="Save path prefix under output/. The studio sets it per session.",
                ),
            ],
            outputs=[],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            search_aliases=["studio marker output", "studio result"],
        )

    @classmethod
    def execute(cls, image, filename_prefix: str) -> IO.NodeOutput:
        results = UI.ImageSaveHelper.save_images(
            image, filename_prefix=filename_prefix, folder_type=UI.FolderType.output, cls=cls,
        )
        return IO.NodeOutput(ui={"images": results})


NODE_CLASS_MAPPINGS = {"TS_StudioOutput": TS_StudioOutput}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_StudioOutput": "TS Studio Output"}
