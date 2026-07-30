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
import json
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


def _extract_json_object(text: str) -> str:
    """Return the first balanced top-level JSON object found in ``text``.

    LLMs occasionally wrap the caption in prose or code fences despite the
    system prompt; a string-aware brace scan recovers the object without being
    fooled by braces inside string values.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        # Compact separators + literal UTF-8: the serialization Ideogram 4
                        # was trained on (docs/prompting.md).
                        return json.dumps(json.loads(candidate), ensure_ascii=False, separators=(",", ":"))
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return ""



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
                IO.String.Input(
                    "mode",
                    default="designer",
                    tooltip="UI mode, managed by the node: 'designer' builds the caption from the visual editor, 'auto' generates it from auto_prompt with the connected clip.",
                ),
                IO.String.Input(
                    "auto_prompt",
                    default="",
                    multiline=True,
                    tooltip="Plain-text idea for Auto mode. The connected clip's LLM turns it into a structured Ideogram 4 JSON caption.",
                ),
                IO.String.Input(
                    "auto_caption",
                    default="",
                    multiline=True,
                    tooltip="Last caption produced by the Generate Prompt button (via the pack's Qwen engine). Managed by the node UI; used as the output in Auto mode.",
                ),
                IO.Int.Input(
                    "auto_seed",
                    default=0,
                    min=0,
                    max=0x7FFFFFFF,
                    tooltip="Sampling seed for Auto mode. The Generate button bumps it so a fresh caption is produced on the next run.",
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
    def execute(cls, image=None, design_json: str = "", mode: str = "designer",
                auto_prompt: str = "", auto_caption: str = "", auto_seed: int = 0) -> IO.NodeOutput:
        if image is not None:
            try:
                node_id = getattr(cls.hidden, "unique_id", None)
                filename = save_graph_reference(image, node_id)
                if filename:
                    logger.info("%s Cached graph reference: %s", LOG_PREFIX, filename)
            except Exception as exc:  # noqa: BLE001 - preview aid must never fail the run
                logger.warning("%s Graph reference caching failed: %s", LOG_PREFIX, exc)

        width, height = dims_from_design(design_json or "")
        if (mode or "designer").strip().lower() == "auto":
            # The caption is produced interactively by the Generate Prompt
            # button through the SuperPrompt engine (its /enhance route with
            # the 'Ideogram Prompt Enhance' preset) and stored here — queue
            # time does zero model work. The shared contract between the two
            # nodes is pinned by tests/test_ideogram_superprompt_contract.py.
            raw = (auto_caption or "").strip()
            if not raw:
                raise RuntimeError(
                    f"{LOG_PREFIX} Auto mode has no caption yet: type your idea and press "
                    "Generate Prompt in the node, or switch back to Designer mode."
                )
            # Belt: re-extract the JSON object in case the LLM wrapped it in
            # prose, and re-serialize compactly (the format Ideogram 4 expects).
            json_prompt = _extract_json_object(raw)
            if not json_prompt:
                # The raw text still goes downstream — a caption that Ideogram
                # can half-read beats failing the run — but silence here meant a
                # truncated or malformed reply looked exactly like a good one.
                logger.warning(
                    "%s Auto caption is not valid JSON (truncated or wrapped in prose); "
                    "passing the raw text through. Press Generate Prompt again if the "
                    "result looks wrong.",
                    LOG_PREFIX,
                )
                json_prompt = raw
            # Push the fresh caption back to the node UI (the Auto panel shows it).
            return IO.NodeOutput(json_prompt, width, height, ui={"ts_ideo_auto": [json_prompt]})
        json_prompt, _aspect = build_caption(design_json or "")
        return IO.NodeOutput(json_prompt, width, height)

    @classmethod
    def fingerprint_inputs(cls, image=None, design_json: str = "", mode: str = "designer",
                           auto_prompt: str = "", auto_caption: str = "", auto_seed: int = 0) -> str:
        design_sig = hashlib.blake2b((design_json or "").encode("utf-8"), digest_size=16).hexdigest()
        auto_sig = hashlib.blake2b(
            f"{mode}|{auto_prompt}|{auto_caption}|{auto_seed}".encode("utf-8"), digest_size=16
        ).hexdigest()
        if image is not None and hasattr(image, "shape"):
            try:
                image_sig = f"{tuple(image.shape)}_{float(image.float().mean()):.6f}"
            except Exception:  # noqa: BLE001
                image_sig = str(getattr(image, "shape", "img"))
        else:
            image_sig = "none"
        return f"{design_sig}_{image_sig}_{auto_sig}"


NODE_CLASS_MAPPINGS = {"TS_IdeogramDesigner": TS_IdeogramDesigner}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_IdeogramDesigner": "TS Ideogram Designer"}
