"""TS Lama Cleanup - inpaint small defects with the LaMa model.

node_id: TS_LamaCleanup

The actual cleanup happens interactively inside the node UI through aiohttp
routes (see ``_lama_helpers.py``). The ``execute`` method only loads the
current working file (or the source) and emits it as IMAGE for the workflow.
"""

import hashlib
import os

import folder_paths
from comfy_api.latest import IO

from ._lama_helpers import (
    _load_image_tensor,
    _resolve_image_path,
    _session_working_path,
    _working_file_signature,
)


class TS_LamaCleanup(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LamaCleanup",
            display_name="TS Lama Cleanup",
            category="TS/Image",
            description=(
                "Interactive defect removal with the LaMa inpainting model. "
                "Open an image with the in-node Load Image button, paint over "
                "small defects with a brush, and the selected region is "
                "cleaned up automatically on mouse-up. Saves edits to the "
                "ComfyUI temp folder; the IMAGE output emits the current "
                "state on workflow execution."
            ),
            inputs=[
                # All controls are rendered by the JS frontend (ts-lama-cleanup.js).
                # The schema defines only hidden serialized state so workflows
                # can round-trip and ``execute`` can pick up the right file.
                IO.String.Input(
                    "source_path",
                    default="",
                    tooltip="Internal: annotated path of the loaded source image.",
                    socketless=True,
                ),
                IO.Int.Input(
                    "brush_size",
                    default=40,
                    min=1,
                    max=400,
                    step=1,
                    tooltip="Brush diameter in image pixels.",
                    socketless=True,
                ),
                IO.Int.Input(
                    "max_resolution",
                    default=512,
                    min=128,
                    max=2048,
                    step=64,
                    tooltip="Maximum longest-side of the LaMa crop in pixels.",
                    socketless=True,
                ),
                IO.Int.Input(
                    "mask_padding",
                    default=64,
                    min=0,
                    max=512,
                    step=8,
                    tooltip="Context padding around the mask bounding box.",
                    socketless=True,
                ),
                IO.Int.Input(
                    "feather",
                    default=4,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Composite seam feather radius (pixels).",
                    socketless=True,
                ),
                IO.String.Input(
                    "session_id",
                    default="",
                    tooltip="Internal: per-node working session id.",
                    socketless=True,
                ),
                IO.String.Input(
                    "working_path",
                    default="",
                    tooltip="Internal: path to the current temp working file.",
                    socketless=True,
                ),
            ],
            outputs=[IO.Image.Output(display_name="image")],
            search_aliases=["lama", "inpaint", "cleanup", "remove tool", "spot heal"],
        )

    @classmethod
    def validate_inputs(cls, **_kwargs) -> bool:
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        source_path: str = "",
        brush_size: int = 40,
        max_resolution: int = 512,
        mask_padding: int = 64,
        feather: int = 4,
        session_id: str = "",
        working_path: str = "",
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(source_path).encode("utf-8"))
        hasher.update(str(session_id).encode("utf-8"))
        hasher.update(str(working_path).encode("utf-8"))
        hasher.update(f"{int(max_resolution)}|{int(mask_padding)}|{int(feather)}".encode("utf-8"))
        active_path = _select_active_path(source_path, working_path, session_id)
        hasher.update(_working_file_signature(active_path).encode("utf-8"))
        return hasher.hexdigest()

    @classmethod
    def execute(
        cls,
        source_path: str = "",
        brush_size: int = 40,
        max_resolution: int = 512,
        mask_padding: int = 64,
        feather: int = 4,
        session_id: str = "",
        working_path: str = "",
    ) -> IO.NodeOutput:
        active_path = _select_active_path(source_path, working_path, session_id)
        tensor = _load_image_tensor(active_path)
        return IO.NodeOutput(tensor)


def _select_active_path(source_path: str, working_path: str, session_id: str) -> str:
    """Prefer working file, then session-derived working path, then source."""
    if working_path:
        resolved = _resolve_image_path(working_path)
        if resolved and os.path.isfile(resolved):
            return resolved
    if session_id:
        candidate = _session_working_path(session_id)
        if candidate.is_file():
            return str(candidate)
    if source_path:
        resolved = _resolve_image_path(source_path)
        if resolved and os.path.isfile(resolved):
            return resolved
        try:
            annotated = folder_paths.get_annotated_filepath(source_path)
        except Exception:
            annotated = ""
        if annotated and os.path.isfile(annotated):
            return annotated
    return ""


NODE_CLASS_MAPPINGS = {"TS_LamaCleanup": TS_LamaCleanup}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LamaCleanup": "TS Lama Cleanup"}
