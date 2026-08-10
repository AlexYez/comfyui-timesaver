"""Manifest carrier of a TS Image Studio backend workflow."""
from __future__ import annotations

import json

from comfy_api.v0_0_2 import IO

from ._marker_shared import CATEGORY, LOG_PREFIX


class TS_StudioManifest(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_StudioManifest",
            display_name="TS Studio Manifest",
            category=CATEGORY,
            description=(
                "Marker: the manifest of a TS Image Studio backend workflow — "
                "one per file. JSON describing id, family, mode, controls, "
                "reference slots and dependencies (see the pack's authoring "
                "help). The studio reads it from the exported workflow; at "
                "execution time the node only validates and passes the text on."
            ),
            inputs=[
                IO.String.Input(
                    "manifest", default="{}", multiline=True,
                    tooltip="Backend manifest JSON. Schema: doc/IMAGE_STUDIO_PLAN.md §4.",
                ),
            ],
            outputs=[IO.String.Output(display_name="manifest")],
            search_aliases=["studio manifest", "studio backend descriptor"],
        )

    @classmethod
    def validate_inputs(cls, manifest: str):
        try:
            parsed = json.loads(manifest or "{}")
        except json.JSONDecodeError as exc:
            return f"{LOG_PREFIX} Manifest is not valid JSON: {exc}"
        if not isinstance(parsed, dict):
            return f"{LOG_PREFIX} Manifest must be a JSON object."
        return True

    @classmethod
    def execute(cls, manifest: str) -> IO.NodeOutput:
        return IO.NodeOutput(manifest)


NODE_CLASS_MAPPINGS = {"TS_StudioManifest": TS_StudioManifest}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_StudioManifest": "TS Studio Manifest"}
