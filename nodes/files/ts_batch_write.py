"""TS Batch Write — writes each result as it arrives, not at the end.

node_id: TS_BatchWrite

⚠️ This node must NOT declare ``is_input_list``. A list-input node is called
ONCE, after every item is done — which is exactly the behaviour this node
exists to avoid. Called per item instead, it appends result 47 while item 48
is still being computed, so a run that dies at item 90 leaves 89 results on
disk rather than nothing.

The same asymmetry explains the preview: ComfyUI collects the UI payloads of
all iterations and sends them in a single ``executed`` event after the last one
(``execution.py``, ``get_output_from_returns``), so a hundred pictures appear
in the canvas at once, at the end. A progress preview is not batched, so this
node pushes the current image through the progress bar and the person sees
item 47 while it is item 47.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from comfy.utils import ProgressBar
from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_batch_write")
LOG_PREFIX = "[TS Batch Write]"

MODE_ONE_FILE = "One file, blocks"
MODE_ONE_LINE = "One file, one line per item"
MODE_PER_ITEM = "One .txt per item"

PREVIEW_MAX_SIDE = 512


def _clean_path(path: str) -> Path:
    """The user's path, with the quotes Explorer's "Copy as path" adds removed."""
    return Path(str(path).strip().strip('"'))


def _safe_name(name: str, index: int) -> str:
    """A file name that cannot escape the output folder or collide by accident."""
    cleaned = "".join(ch for ch in str(name).strip() if ch.isalnum() or ch in " ._-").strip()
    return cleaned or f"item_{index:05d}"


def _send_preview(image: torch.Tensor | None, value: int, total: int) -> None:
    """Show the freshly made picture on the progress bar, right now.

    Silent on failure by design: a preview that cannot be built is a cosmetic
    loss, and losing the written result over it would be absurd.
    """
    if image is None or not isinstance(image, torch.Tensor) or image.numel() == 0:
        ProgressBar(total).update_absolute(value, total)
        return
    try:
        from PIL import Image

        frame = image[0] if image.ndim == 4 else image
        array = (frame.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
        pil = Image.fromarray(array)
        # The tuple shape the progress handler expects: (format, image, max side).
        ProgressBar(total).update_absolute(value, total, ("JPEG", pil, PREVIEW_MAX_SIDE))
    except Exception as exc:  # noqa: BLE001 - cosmetic path, never fatal
        logger.debug("%s Preview skipped: %s", LOG_PREFIX, exc)
        ProgressBar(total).update_absolute(value, total)


class TS_BatchWrite(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_BatchWrite",
            display_name="TS Batch Write",
            category="TS/Files",
            description=(
                "Append one batch result to disk per iteration, so results survive a run "
                "that stops halfway. Wire index and total from TS Batch Source; connect an "
                "image to watch results appear while the batch is still running."
            ),
            inputs=[
                IO.String.Input(
                    "text",
                    default="",
                    multiline=True,
                    force_input=True,
                    tooltip="The result for this item — a caption, a prompt, any text.",
                ),
                IO.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=1000000,
                    tooltip=(
                        "Position of this item. Index 0 starts the file from scratch; "
                        "anything else appends, which is what makes a resumed run add to "
                        "the existing file instead of wiping it."
                    ),
                ),
                IO.Int.Input(
                    "total",
                    default=1,
                    min=1,
                    max=1000000,
                    tooltip="How many items the batch has. Used for the progress bar.",
                ),
                IO.Combo.Input(
                    "mode",
                    options=[MODE_ONE_FILE, MODE_ONE_LINE, MODE_PER_ITEM],
                    default=MODE_ONE_FILE,
                    tooltip=(
                        "Blocks separated by a blank line (read back by TS Batch Prompt "
                        "Loader); one line per item (read back by TS Batch Source in "
                        "'Lines in text file' mode — this is what closes the loop from "
                        "captions to generation); or one .txt per item named after it, "
                        "the layout caption datasets expect."
                    ),
                ),
                IO.String.Input(
                    "output_path",
                    default="",
                    tooltip="Target file in 'one file' mode, target folder in 'per item' mode.",
                ),
                IO.String.Input(
                    "name",
                    default="",
                    optional=True,
                    tooltip=(
                        "Name for this item's .txt, without extension. Usually the 'name' "
                        "output of TS Batch Load Image, so the caption sits beside its picture."
                    ),
                ),
                IO.Boolean.Input(
                    "prefix_with_name",
                    default=False,
                    optional=True,
                    tooltip=(
                        "In the one-file modes, put the item's name in front of the result "
                        "— on its own line for blocks, tab-separated for one line."
                    ),
                ),
                IO.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "Optional. Shown on the progress bar as each item finishes — the "
                        "only way to watch results arrive, since ComfyUI holds normal "
                        "previews until the whole batch is done."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="text",
                    tooltip="The same text, passed through so the chain can continue.",
                ),
            ],
            is_output_node=True,
            search_aliases=["batch write", "save text", "append", "caption dataset"],
        )

    @classmethod
    def validate_inputs(
        cls, text, index, total, mode, output_path, name=None,
        prefix_with_name=None, image=None,
    ) -> bool | str:
        if not str(output_path).strip():
            return f"{LOG_PREFIX} No output_path given."
        return True

    @classmethod
    def execute(
        # ⚠️ Умолчание обязано совпадать со схемой (`default=""`), иначе вызов
        # мимо ComfyUI — из теста или из чужого кода — получит не то же самое.
        cls, text, index, total, mode, output_path, name="",
        prefix_with_name=False, image=None,
    ) -> IO.NodeOutput:
        target = _clean_path(output_path)
        if not str(target).strip():
            raise ValueError(f"{LOG_PREFIX} No output_path given.")

        body = str(text or "").strip()
        position = int(index)
        count = max(1, int(total))

        if str(mode) == MODE_PER_ITEM:
            target.mkdir(parents=True, exist_ok=True)
            destination = target / f"{_safe_name(name or '', position)}.txt"
            destination.write_text(body + "\n", encoding="utf-8")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            one_line = str(mode) == MODE_ONE_LINE
            if one_line:
                # ⚠️ The model answers in paragraphs whenever it feels like it.
                # A "one line per item" file carrying a two-line answer is no
                # longer one line per item, and whatever reads it back silently
                # gains an item. Collapsing IS the contract of this mode.
                body = " ".join(body.split())
            label = _safe_name(name or "", position)
            if prefix_with_name and name:
                block = f"{label}\t{body}" if one_line else f"{label}\n{body}"
            else:
                block = body
            separator = "\n" if one_line else "\n\n"
            # Index 0 truncates: a fresh run should not silently continue the
            # previous one's file. Every other index appends, which is what lets
            # a resumed run (start_at > 0) add to what is already there.
            if position == 0:
                target.write_text(block + separator, encoding="utf-8")
            else:
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(block + separator)
            destination = target

        _send_preview(image, position + 1, count)
        logger.info("%s %d/%d -> %s", LOG_PREFIX, position + 1, count, destination.name)
        return IO.NodeOutput(str(text or ""))


NODE_CLASS_MAPPINGS = {"TS_BatchWrite": TS_BatchWrite}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_BatchWrite": "TS Batch Write"}
