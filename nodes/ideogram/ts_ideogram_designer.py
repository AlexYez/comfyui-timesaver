"""TS Ideogram Designer.

node_id: TS_IdeogramDesigner

A visual designer for Ideogram 4 structured-JSON prompts. The interactive
editor (drag/resize text + object blocks on an aspect-correct artboard, pick
font/style presets, place text over an optional reference image) lives in
``js/ideogram/``. The editor serializes its full state into the hidden
``design_json`` STRING input; ``execute`` turns that into a valid Ideogram 4
caption (see ``_ideogram_helpers.build_caption``) and emits it as ``json_prompt``
(STRING), along with the resolved ``width`` and ``height`` (INT) derived from the
design's aspect ratio + megapixels (see ``_ideogram_helpers.dims_from_design``).

The optional ``image`` input is a reference-only underlay aid: when connected,
``execute`` caches its first frame into the input directory so the editor can
trace text over it. It does not affect the emitted caption.
"""

from __future__ import annotations

import hashlib
import logging

from comfy_api.v0_0_2 import IO

from ._ideogram_helpers import (
    build_caption,
    dims_from_design,
    register_routes,
    save_graph_reference,
)

logger = logging.getLogger("comfyui_timesaver.ts_ideogram_designer")
LOG_PREFIX = "[TS Ideogram Designer]"

# Register the /ts_ideogram/* API routes once, at import time.
register_routes()


class TS_IdeogramDesigner(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_IdeogramDesigner",
            display_name="TS Ideogram Designer",
            category="TS/Ideogram",
            description=(
                "Визуальный редактор JSON-промтов для Ideogram 4: расставьте "
                "текстовые/объектные блоки, выберите шрифты и стиль — на выходе "
                "валидный Ideogram-4 капшен (STRING) + размеры width/height (INT), "
                "рассчитанные из соотношения сторон и мегапикселей."
            ),
            inputs=[
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional reference underlay: its first frame is cached so the editor can trace text over it. Does not affect the caption.",
                ),
                IO.String.Input(
                    "design_json",
                    default="",
                    multiline=False,
                    tooltip="Serialized editor state, managed by the node UI. Converted into the Ideogram 4 caption on execution.",
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="json_prompt",
                    tooltip="Ideogram 4 caption built from the design.",
                ),
                IO.Int.Output(
                    display_name="width",
                    tooltip="Output width in pixels, derived from the design's aspect ratio and megapixels.",
                ),
                IO.Int.Output(
                    display_name="height",
                    tooltip="Output height in pixels, derived from the design's aspect ratio and megapixels.",
                ),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, image=None, design_json: str = "") -> IO.NodeOutput:
        if image is not None:
            try:
                node_id = getattr(cls.hidden, "unique_id", None)
                filename = save_graph_reference(image, node_id)
                if filename:
                    logger.info("%s Cached graph reference: %s", LOG_PREFIX, filename)
            except Exception as exc:  # noqa: BLE001 - preview aid must never fail the run
                logger.warning("%s Graph reference caching failed: %s", LOG_PREFIX, exc)

        json_prompt, _aspect = build_caption(design_json or "")
        width, height = dims_from_design(design_json or "")
        return IO.NodeOutput(json_prompt, width, height)

    @classmethod
    def fingerprint_inputs(cls, image=None, design_json: str = "") -> str:
        design_sig = hashlib.blake2b((design_json or "").encode("utf-8"), digest_size=16).hexdigest()
        if image is not None and hasattr(image, "shape"):
            try:
                image_sig = f"{tuple(image.shape)}_{float(image.float().mean()):.6f}"
            except Exception:  # noqa: BLE001
                image_sig = str(getattr(image, "shape", "img"))
        else:
            image_sig = "none"
        return f"{design_sig}_{image_sig}"


NODE_CLASS_MAPPINGS = {"TS_IdeogramDesigner": TS_IdeogramDesigner}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_IdeogramDesigner": "TS Ideogram Designer"}
