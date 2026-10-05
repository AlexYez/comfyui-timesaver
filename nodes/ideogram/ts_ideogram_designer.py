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

from .._caption_json import repair_caption_text
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


#: The Qwen preset that turns an idea into an Ideogram 4 caption. Shared with
#: the editor's Generate button (pinned by test_ideogram_superprompt_contract).
PROMPT_PRESET = "Ideogram Prompt Enhance"


def caption_from_prompt(text: str, seed: int, bigger_model: bool = True) -> str:
    """Turn a plain prompt into an Ideogram 4 JSON caption with Qwen.

    ⚠️ По умолчанию четырёхмиллиардная модель (просьба владельца, 05.10.2026):
    структурный JSON с bbox и стилем 2B пишет заметно хуже, а здесь качество
    капшена — это качество картинки. Галочка `bigger_model` переключает на 2B.

    Falls back to the minimal valid caption (the text verbatim as
    ``high_level_description``) when the model returns no parseable JSON, so a
    run never dies on a malformed answer.
    """
    # Lazy: the Qwen engine pulls in transformers, which the designer mode
    # never needs.
    from ..llm.super_prompt._helpers import resolve_prompt_model  # noqa: PLC0415
    from ..llm.super_prompt._qwen import _generate_with_qwen  # noqa: PLC0415

    raw = _generate_with_qwen(
        text=text,
        system_preset=PROMPT_PRESET,
        operation_id=None,
        seed=int(seed),
        model_id=resolve_prompt_model(bool(bigger_model)),
    )
    # The engine already mends JSON-preset answers; this is the belt for an
    # engine that returned the raw text (and the same repair, so idempotent).
    caption = repair_caption_text(raw or "", text) or _extract_json_object(raw or "")
    if caption:
        return caption
    logger.warning(
        "%s Qwen returned no JSON caption; the prompt is passed on verbatim inside "
        "the minimal Ideogram 4 schema.", LOG_PREFIX,
    )
    return json.dumps({"high_level_description": text}, ensure_ascii=False)


def caption_as_typed(text: str) -> str:
    """The prompt as it is, with no model: for a caption the user already has.

    A JSON caption is only checked and re-serialized compactly (a stray brace
    from hand editing is mended; nothing is added or held to an idea). Plain
    text goes into the minimal valid caption, verbatim — the envelope Ideogram 4
    expects, nothing more.
    """
    if text.lstrip().startswith("{"):
        caption = repair_caption_text(text)
        if caption:
            return caption
    return json.dumps({"high_level_description": text}, ensure_ascii=False)


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
                    tooltip="UI mode, managed by the node: 'designer' builds the caption from the visual editor, 'prompt' turns the plain prompt (auto_prompt) into a caption with Qwen (2B or 4B, see bigger_model) when the graph runs.",
                ),
                IO.String.Input(
                    "auto_prompt",
                    default="",
                    multiline=True,
                    tooltip="Plain prompt for Prompt mode, typed in the node. Qwen turns it into a structured Ideogram 4 JSON caption when the graph runs.",
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
                # ⚠️ Последним: `widgets_values` позиционен. Сохранённый граф
                # приходит без этого поля и получает умолчание — 4B, как просил
                # владелец (у TS Super Prompt умолчание 2B, там другая история).
                IO.Boolean.Input(
                    "bigger_model",
                    default=True,
                    tooltip=(
                        "Which Qwen model writes the caption — in Prompt mode and for the "
                        "editor's Generate and From image buttons. On: the 4B model, better "
                        "structured captions and layouts. Off: the 2B one, faster and half "
                        "the VRAM. The 4B model is fetched on first use."
                    ),
                ),
                # ⚠️ Тоже последним, по той же причине. Умолчание True — так
                # режим «Промпт» работал с самого появления.
                IO.Boolean.Input(
                    "enhance_prompt",
                    default=True,
                    tooltip=(
                        "Prompt mode only. On: Qwen turns the prompt into a structured "
                        "Ideogram 4 JSON caption when the graph runs (15-40 s). Off: the "
                        "prompt goes out as it is — a ready JSON caption only gets its "
                        "structure checked, plain text is wrapped into the minimal caption "
                        "Ideogram 4 expects. No model is loaded."
                    ),
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
                auto_prompt: str = "", auto_caption: str = "", auto_seed: int = 0,
                bigger_model: bool = True, enhance_prompt: bool = True) -> IO.NodeOutput:
        if image is not None:
            try:
                node_id = getattr(cls.hidden, "unique_id", None)
                filename = save_graph_reference(image, node_id)
                if filename:
                    logger.info("%s Cached graph reference: %s", LOG_PREFIX, filename)
            except Exception as exc:  # noqa: BLE001 - preview aid must never fail the run
                logger.warning("%s Graph reference caching failed: %s", LOG_PREFIX, exc)

        width, height = dims_from_design(design_json or "")
        mode_key = (mode or "designer").strip().lower()
        if mode_key == "prompt":
            text = (auto_prompt or "").strip()
            if not text:
                raise RuntimeError(
                    f"{LOG_PREFIX} Prompt mode has no prompt: type it in the node, or "
                    "switch back to Designer mode."
                )
            if enhance_prompt:
                json_prompt = caption_from_prompt(text, auto_seed, bigger_model)
            else:
                json_prompt = caption_as_typed(text)
            return IO.NodeOutput(json_prompt, width, height, ui={"ts_ideo_auto": [json_prompt]})
        if mode_key == "auto":
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
                # Ideogram 4 is trained on structured JSON captions, and its
                # official inference validates prompts against that schema
                # (ComfyUI's own Ideogram4 template says so outright). Free text
                # is accepted but read loosely — that is how "detailed woman
                # head" came back as a poster with invented typography.
                #
                # So the text is not passed through bare any more: it is placed
                # into the smallest valid caption, verbatim and as the only
                # field. Nothing is added to what was written — only the
                # envelope the model expects is put around it.
                json_prompt = json.dumps(
                    {"high_level_description": raw}, ensure_ascii=False)
                logger.info(
                    "%s Caption was plain text; wrapped verbatim into the minimal "
                    "JSON schema Ideogram 4 expects. Press Generate Prompt for a "
                    "full structured caption with style and background.",
                    LOG_PREFIX,
                )
            # Push the fresh caption back to the node UI (the Auto panel shows it).
            return IO.NodeOutput(json_prompt, width, height, ui={"ts_ideo_auto": [json_prompt]})
        json_prompt, _aspect = build_caption(design_json or "")
        return IO.NodeOutput(json_prompt, width, height)

    @classmethod
    def fingerprint_inputs(cls, image=None, design_json: str = "", mode: str = "designer",
                           auto_prompt: str = "", auto_caption: str = "", auto_seed: int = 0,
                           bigger_model: bool = True, enhance_prompt: bool = True) -> str:
        design_sig = hashlib.blake2b((design_json or "").encode("utf-8"), digest_size=16).hexdigest()
        auto_sig = hashlib.blake2b(
            f"{mode}|{auto_prompt}|{auto_caption}|{auto_seed}|{bool(bigger_model)}"
            f"|{bool(enhance_prompt)}".encode(),
            digest_size=16,
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
