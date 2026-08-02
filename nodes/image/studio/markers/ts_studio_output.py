"""Result exit point of a TS Image Studio backend workflow."""
from __future__ import annotations

import json
import logging

from comfy_api.v0_0_2 import IO, UI

from ._marker_shared import CATEGORY

logger = logging.getLogger("comfyui_timesaver.ts_studio_output")
LOG_PREFIX = "[TS Studio Output]"

# tEXt chunk that carries the studio's own run snapshot. ComfyUI writes one
# chunk per extra_pnginfo key, so this rides ALONGSIDE the standard `prompt`
# and `workflow` chunks instead of competing with them: ComfyUI keeps reading
# its metadata, image browsers keep reading theirs, and the studio reads this.
STATE_CHUNK = "ts_studio"


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
                IO.String.Input(
                    "studio_state", default="", optional=True, socketless=True,
                    tooltip=(
                        "JSON snapshot of the studio run, saved into the PNG's "
                        "'ts_studio' text chunk so the studio can recreate the "
                        "exact session from the image. The studio fills it in; "
                        "leave it empty when running the workflow by hand."
                    ),
                ),
            ],
            outputs=[],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            search_aliases=["studio marker output", "studio result"],
        )

    @classmethod
    def execute(cls, image, filename_prefix: str, studio_state: str = "") -> IO.NodeOutput:
        cls._attach_state(studio_state)
        results = UI.ImageSaveHelper.save_images(
            image, filename_prefix=filename_prefix, folder_type=IO.FolderType.output, cls=cls,
        )
        return IO.NodeOutput(ui={"images": results})

    @classmethod
    def _attach_state(cls, studio_state: str) -> None:
        """Put the snapshot in extra_pnginfo so the saver writes its own chunk."""
        text = (studio_state or "").strip()
        if not text:
            return
        info = cls.hidden.extra_pnginfo
        if not isinstance(info, dict):
            # Metadata is disabled for this run (--disable-metadata); nothing
            # to attach to, and the image simply saves without the snapshot.
            logger.info("%s no extra_pnginfo — run snapshot not embedded", LOG_PREFIX)
            return
        try:
            info[STATE_CHUNK] = json.loads(text)
        except ValueError:
            logger.warning("%s studio_state is not valid JSON — stored verbatim", LOG_PREFIX)
            info[STATE_CHUNK] = text


NODE_CLASS_MAPPINGS = {"TS_StudioOutput": TS_StudioOutput}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_StudioOutput": "TS Studio Output"}
