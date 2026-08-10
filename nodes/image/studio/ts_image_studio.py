"""TS Image Studio — the bridge node of the fullscreen studio app.

The studio itself is a frontend application (js/image/studio/) that submits
its own prompts through the ComfyUI API; this node is its representative in
the user's graph. It carries the session id, and its IMAGE output emits the
result currently selected in the studio gallery, so the studio composes with
any downstream graph.

Design: doc/IMAGE_STUDIO_PLAN.md. State (session id, selected result path)
lives in hidden widgets mirrored to node.properties by the frontend
(CLAUDE.md §12.5.13), so it survives workflow reloads.
"""
from __future__ import annotations

import logging
import os

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_image_studio")
LOG_PREFIX = "[TS Image Studio]"

try:
    import folder_paths
except ImportError:  # unit tests outside ComfyUI
    folder_paths = None


def _resolve_result_path(result_path: str) -> str:
    """Absolute path of a studio result, confined to the output directory.

    The widget stores a path RELATIVE to output/ ("ts_studio/<session>/x.png").
    A workflow file is shared material — treat the value as untrusted and
    refuse anything that climbs outside output/.
    """
    rel = (result_path or "").strip().replace("\\", "/")
    if not rel or folder_paths is None:
        return ""
    base = os.path.abspath(folder_paths.get_output_directory())
    candidate = os.path.abspath(os.path.join(base, rel))
    try:
        if os.path.commonpath([base, candidate]) != base:
            logger.warning(f"{LOG_PREFIX} result_path '{rel}' escapes the output folder; ignored.")
            return ""
    except ValueError:
        return ""
    return candidate if os.path.isfile(candidate) else ""


class TS_ImageStudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ImageStudio",
            display_name="TS Image Studio",
            category="TS/Image",
            description=(
                "Fullscreen image studio: generate, edit, inpaint and upscale "
                "with local models through native ComfyUI workflows under the "
                "hood. Open the interface, work in the app; the IMAGE output "
                "emits the result selected in the studio gallery."
            ),
            inputs=[
                # No IMAGE input: sources are chosen inside the studio — dropped
                # onto its canvas, picked from the library or the Artius browser.
                # A socket for the same job only invited the question of which
                # one wins.
                IO.String.Input(
                    "session_id", default="", socketless=True,
                    tooltip="Studio session id. Managed by the frontend.",
                ),
                IO.String.Input(
                    "result_path", default="", socketless=True,
                    tooltip="Selected result, relative to output/. Managed by the frontend.",
                ),
            ],
            outputs=[IO.Image.Output(display_name="image")],
            search_aliases=["image studio", "generate", "inpaint app", "ts studio"],
        )

    @classmethod
    def execute(cls, image=None, session_id: str = "", result_path: str = "") -> IO.NodeOutput:
        path = _resolve_result_path(result_path)
        if path:
            from .markers._marker_shared import load_image_tensor

            tensor, _mask = load_image_tensor(path)
            return IO.NodeOutput(tensor)
        if image is not None:
            logger.info(f"{LOG_PREFIX} No studio result selected; passing the input image through.")
            return IO.NodeOutput(image)
        raise RuntimeError(
            f"{LOG_PREFIX} No result selected. Open the interface, generate an image "
            f"and pick it in the gallery (or connect an input image)."
        )

    @classmethod
    def fingerprint_inputs(cls, image=None, session_id: str = "", result_path: str = "") -> str:
        path = _resolve_result_path(result_path)
        if not path:
            return f"none:{result_path}"
        stat = os.stat(path)
        return f"{result_path}:{stat.st_mtime_ns}:{stat.st_size}"


NODE_CLASS_MAPPINGS = {"TS_ImageStudio": TS_ImageStudio}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ImageStudio": "TS Image Studio"}
