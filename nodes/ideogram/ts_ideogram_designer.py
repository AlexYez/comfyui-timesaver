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
from pathlib import Path

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

# Auto mode reuses the "Ideogram Prompt Enhance" preset shipped for the Qwen
# nodes: same system prompt, same sampling parameters — one source of truth for
# how an idea becomes an Ideogram 4 JSON caption.
_AUTO_PRESET_NAME = "Ideogram Prompt Enhance"
_AUTO_PRESET_PATH = Path(__file__).resolve().parents[1] / "qwen_3_vl_presets.json"


def _load_auto_preset() -> tuple[str, dict]:
    try:
        data = json.loads(_AUTO_PRESET_PATH.read_text(encoding="utf-8"))
        preset = data.get(_AUTO_PRESET_NAME) or {}
        system_prompt = str(preset.get("system_prompt") or "").strip()
        gen_params = preset.get("gen_params") or {}
        if system_prompt:
            return system_prompt, gen_params if isinstance(gen_params, dict) else {}
    except Exception as exc:  # noqa: BLE001 - a broken preset must surface, not crash import
        logger.warning("%s Failed to load auto preset: %s", LOG_PREFIX, exc)
    return "", {}


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


def _generate_auto_caption(clip, auto_prompt: str, image, seed: int) -> str:
    """Run the clip's LLM (the image model's text encoder) over the user idea.

    Mirrors the built-in ``TextGenerate`` node's use of the clip API
    (tokenize -> generate -> decode) so any encoder that works with the core
    node works here too. A connected reference image is passed to the LLM —
    the preset knows how to caption from an image.
    """
    prompt_text = (auto_prompt or "").strip()
    if not prompt_text:
        raise RuntimeError(
            f"{LOG_PREFIX} Auto mode: the prompt is empty. Type your idea into the node's text field."
        )
    if clip is None:
        raise RuntimeError(
            f"{LOG_PREFIX} Auto mode has no caption yet: press Generate Prompt in the node, "
            "or connect a clip (LLM text encoder) so the caption can be generated at queue time."
        )
    system_prompt, gen_params = _load_auto_preset()
    if not system_prompt:
        raise RuntimeError(
            f"{LOG_PREFIX} Auto preset '{_AUTO_PRESET_NAME}' is missing from {_AUTO_PRESET_PATH.name}."
        )

    # One user message carrying both the instructions and the idea keeps this
    # model-agnostic: the clip's own default chat template wraps it correctly
    # for whichever encoder family is connected.
    full_prompt = f"{system_prompt}\n\nUser idea: {prompt_text}"
    try:
        tokens = clip.tokenize(
            full_prompt,
            image=image,
            skip_template=False,
            min_length=1,
            thinking=False,
            video=None,
            audio=None,
        )
        generated_ids = clip.generate(
            tokens,
            do_sample=True,
            max_length=int(gen_params.get("max_new_tokens", 1024)),
            temperature=float(gen_params.get("temperature", 0.45)),
            top_k=int(gen_params.get("top_k", 20)),
            top_p=float(gen_params.get("top_p", 0.9)),
            min_p=0.0,
            repetition_penalty=float(gen_params.get("repetition_penalty", 1.05)),
            presence_penalty=0.0,
            seed=int(seed) & 0x7FFFFFFF,
        )
        generated_text = clip.decode(generated_ids)
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface encoder failures with a TS-prefixed message
        raise RuntimeError(
            f"{LOG_PREFIX} Auto mode generation failed: {exc}. "
            "Make sure the connected clip is an LLM-based text encoder supported by the built-in "
            "Generate Text node."
        ) from exc

    caption = _extract_json_object(str(generated_text or ""))
    if caption:
        return caption
    # The model refused to produce JSON — pass its text through rather than
    # failing the whole workflow; downstream encoders accept plain prose too.
    logger.warning("%s Auto mode: no JSON object in the LLM reply, passing raw text through.", LOG_PREFIX)
    return str(generated_text or "").strip()


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
                IO.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip="Text encoder of your image model (an LLM on modern models). Required by Auto mode: it writes the Ideogram JSON caption from your plain-text prompt.",
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
            # Output-node status makes the node a valid partial-execution
            # target: the Auto panel's Generate button queues just this node,
            # and the server only accepts output nodes as targets. Results are
            # cached by fingerprint, so a full queue does not re-run the LLM
            # unless the prompt or seed changed.
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image=None, design_json: str = "", clip=None, mode: str = "designer",
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
            # Primary path: the connected clip (the image model's own LLM text
            # encoder) generates the caption at queue time, exactly like the
            # built-in Generate Text node — with ComfyUI's own node progress.
            # The stored caption is only a fallback for graphs without a clip.
            if clip is not None:
                json_prompt = _generate_auto_caption(clip, auto_prompt or "", image, int(auto_seed or 0))
            else:
                json_prompt = (auto_caption or "").strip()
                if not json_prompt:
                    json_prompt = _generate_auto_caption(clip, auto_prompt or "", image, int(auto_seed or 0))
            # Push the fresh caption back to the node UI (the Auto panel shows it).
            return IO.NodeOutput(json_prompt, width, height, ui={"ts_ideo_auto": [json_prompt]})
        json_prompt, _aspect = build_caption(design_json or "")
        return IO.NodeOutput(json_prompt, width, height)

    @classmethod
    def fingerprint_inputs(cls, image=None, design_json: str = "", clip=None, mode: str = "designer",
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
