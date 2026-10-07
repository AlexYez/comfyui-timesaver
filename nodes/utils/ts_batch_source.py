"""TS Batch Source — turns a job into a list ComfyUI runs one item at a time.

node_id: TS_BatchSource

Why a list and not a loop: ComfyUI already has a batch engine. When a node
receives a LIST, ``execution.py`` calls ``execute`` once per element and the
whole branch below runs that many times, each run independent of the others.
That independence is the point — a captioning model gets a fresh conversation
per picture instead of one ever-growing context.

⚠️ But the list runs BREADTH-FIRST, and that is measured, not assumed: on a
live server the loader logged items 1, 2, 3 and only then the writer logged
1/3, 2/3, 3/3. ``_async_map_node_over_list`` is a loop inside ONE node, so
every copy of a node finishes before the next node starts. Two consequences,
both invisible until they bite:

* results reach disk only after the model has done every item, so nothing can
  be watched arriving;
* a node that edits shared state per item — ``TS_ImagePromptInjector`` writing
  the current prompt into the saved metadata — is overwritten by the last item
  before the save nodes run, and every picture ends up stamped identically.

``one_per_run`` exists for exactly those cases: one job per queued run, the
graph traversed end to end each time.

⚠️ This node deliberately hands out PATHS, not pictures. Loading a hundred 4K
frames to pass them along would hold roughly 10 GB in the output cache before
the first caption is written; ``TS Batch Load Image`` reads them one at a time
instead, so the peak is a single frame.
"""

from __future__ import annotations

import logging
import random
import re
import time
from pathlib import Path

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_batch_source")
LOG_PREFIX = "[TS Batch Source]"


class _Cursor:
    """Where each job stands when it is handed out one item per run.

    ⚠️ Measured, not assumed: with a LIST output ComfyUI runs a node N times in
    a row and only then moves to the next node (`_async_map_node_over_list` is
    a loop INSIDE one node). So the writer only starts after the model has done
    all N, and anything that edits shared state per item — the prompt injector
    stamping metadata — ends up with the last value on every result.

    One item per run avoids all of it: each queued run is a full pass through
    the graph, so a picture is saved, stamped and previewed before the next job
    begins. State lives at module level because a V3 node class is locked.
    """

    def __init__(self) -> None:
        self.positions: dict[str, int] = {}
        # ⚠️ A plain timestamp is NOT unique here: `time.time_ns()` on Windows
        # is quantised to ~15 ms, so two calls in a row return the same number
        # and the run gets served from cache after all. A counter cannot tie.
        self.ticks: int = 0


_cursor = _Cursor()

MODE_IMAGES = "Images in folder"
MODE_LINES = "Lines in text file"
MODE_COUNT = "Count only"
#: One prompt per file. ⚠️ Appended LAST to the combo: the options' order is
#: not stored in graphs, but the default and the existing names must not move.
MODE_TEXT_FILES = "Text files in folder"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
TEXT_SUFFIXES = (".txt", ".md")
#: A prompt is a few kilobytes. Anything this big in the folder is not one (a
#: log, a dump) and would land whole in every model call downstream.
TEXT_FILE_MAX_BYTES = 1_000_000

_DIGITS = re.compile(r"(\d+)")


def _natural_key(name: str) -> list:
    """Sort key that reads digit runs as numbers, so ``2`` precedes ``10``.

    Plain lexicographic order puts ``img10`` before ``img2``, which silently
    scrambles a numbered sequence — the one thing a batch of frames must keep.
    """
    return [int(part) if part.isdigit() else part.lower() for part in _DIGITS.split(name)]


def _folder_files(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Files with these suffixes directly inside ``folder``, in natural name order."""
    found = [
        entry for entry in folder.iterdir()
        if entry.is_file() and entry.suffix.lower() in suffixes
    ]
    found.sort(key=lambda entry: _natural_key(entry.name))
    return found


def _folder_items(folder: Path) -> list[str]:
    """Every image directly inside ``folder``, in natural name order."""
    return [str(entry) for entry in _folder_files(folder, IMAGE_SUFFIXES)]


def _folder_texts(folder: Path) -> list[str]:
    """The text of every prompt file in ``folder``, one job per file.

    The whole file is the prompt — line breaks inside it stay, unlike the
    "Lines in text file" mode, where every line is a job of its own. A BOM
    (Notepad's UTF-8) is dropped, outer whitespace trimmed; empty files and
    files too big to be a prompt are skipped with a word in the log rather than
    becoming blank or enormous jobs.
    """
    texts: list[str] = []
    skipped: list[str] = []
    for entry in _folder_files(folder, TEXT_SUFFIXES):
        if entry.stat().st_size > TEXT_FILE_MAX_BYTES:
            logger.warning("%s Skipped %s: %d bytes is not a prompt.",
                           LOG_PREFIX, entry.name, entry.stat().st_size)
            continue
        text = entry.read_text(encoding="utf-8-sig", errors="replace")
        text = text.replace("\r\n", "\n").strip()
        if text:
            texts.append(text)
        else:
            skipped.append(entry.name)
    if skipped:
        logger.info("%s Skipped %d empty file(s): %s", LOG_PREFIX, len(skipped),
                    ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""))
    return texts


def _file_lines(file_path: Path) -> list[str]:
    """Non-empty lines of a text file, whitespace trimmed."""
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    return [line.strip() for line in raw.replace("\r\n", "\n").split("\n") if line.strip()]


def _clean_path(path: str) -> Path:
    """The user's path, with the quotes Explorer's "Copy as path" adds removed."""
    return Path(str(path).strip().strip('"'))


def _collect(mode: str, path: str, count: int) -> list[str]:
    """The job's items, before ``start_at`` and ``limit`` narrow them down."""
    if mode == MODE_COUNT:
        return [str(i) for i in range(int(count))]

    target = _clean_path(path)
    if not str(target).strip():
        raise ValueError(f"{LOG_PREFIX} Mode '{mode}' needs a path, but none was given.")
    if not target.exists():
        raise FileNotFoundError(f"{LOG_PREFIX} Path does not exist: {target}")

    if mode in (MODE_IMAGES, MODE_TEXT_FILES):
        if not target.is_dir():
            raise NotADirectoryError(f"{LOG_PREFIX} Expected a folder, got a file: {target}")
        return _folder_items(target) if mode == MODE_IMAGES else _folder_texts(target)

    if not target.is_file():
        raise IsADirectoryError(f"{LOG_PREFIX} Expected a text file, got a folder: {target}")
    return _file_lines(target)


class TS_BatchSource(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_BatchSource",
            display_name="TS Batch Source",
            category="TS/Utils",
            description=(
                "Turn a folder of images, a folder of prompt files, a text file or a plain "
                "count into a job list. Everything wired below this node runs once per "
                "item, independently — so a hundred captions are a hundred separate model "
                "calls, not one growing context."
            ),
            inputs=[
                IO.Combo.Input(
                    "mode",
                    options=[MODE_IMAGES, MODE_LINES, MODE_COUNT, MODE_TEXT_FILES],
                    default=MODE_IMAGES,
                    tooltip=(
                        "Where the jobs come from: image files in a folder, non-empty "
                        "lines of a text file, simply N numbered iterations, or prompt "
                        "files in a folder (.txt / .md, one prompt per file, its line "
                        "breaks kept)."
                    ),
                ),
                IO.String.Input(
                    "path",
                    default="",
                    tooltip="Folder or text file. Ignored in 'Count only' mode.",
                ),
                IO.Int.Input(
                    "count",
                    default=10,
                    min=1,
                    max=10000,
                    tooltip="How many iterations to emit in 'Count only' mode.",
                ),
                IO.Int.Input(
                    "start_at",
                    default=0,
                    min=0,
                    max=1000000,
                    tooltip=(
                        "Skip this many items from the front. Use it to resume a run "
                        "that stopped halfway instead of redoing the finished part."
                    ),
                ),
                IO.Int.Input(
                    "limit",
                    default=0,
                    min=0,
                    max=1000000,
                    tooltip="Stop after this many items. 0 means no limit.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                    tooltip=(
                        "Base for the per-item seed output. Every item gets its own "
                        "derived seed, so repeated iterations of the SAME prompt come "
                        "back different instead of identical."
                    ),
                ),
                IO.Boolean.Input(
                    "one_per_run",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Hand out ONE job per queued run instead of the whole list. Set the "
                        "Batch count in ComfyUI's queue to the number of jobs and each run "
                        "becomes a full pass through the graph — the picture is saved and "
                        "its metadata stamped before the next job starts. Needed whenever "
                        "results must appear as they are made, or when a node writes "
                        "per-item metadata (TS Image Prompt Injector). With the list "
                        "instead, ComfyUI finishes every copy of one node before moving to "
                        "the next, so results land only at the very end."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="item",
                    is_output_list=True,
                    tooltip=(
                        "One job per element: an image path, a text line, a number, or "
                        "the whole text of a prompt file."
                    ),
                ),
                IO.Int.Output(
                    display_name="index",
                    is_output_list=True,
                    tooltip="Position of the item in the job list, starting at 0.",
                ),
                IO.Int.Output(
                    display_name="total",
                    tooltip="How many items the job has. The same value for every item.",
                ),
                IO.Int.Output(
                    display_name="seed",
                    is_output_list=True,
                    tooltip=(
                        "A distinct seed per item, derived from the seed widget. Wire it "
                        "into the sampler or LLM seed — a widget alone would hand every "
                        "iteration the same number."
                    ),
                ),
            ],
            search_aliases=["batch", "iterator", "for each", "folder batch", "queue"],
        )

    @classmethod
    def validate_inputs(cls, mode, path, count, start_at, limit, seed, one_per_run=False) -> bool | str:
        if mode != MODE_COUNT and not str(path).strip():
            return f"{LOG_PREFIX} Mode '{mode}' needs a path."
        return True

    @classmethod
    def fingerprint_inputs(cls, mode, path, count, start_at, limit, seed, one_per_run=False) -> str:
        """Change the folder's contents and the node must run again.

        Without this the cached list survives new files landing in the folder,
        and the batch quietly processes yesterday's set.
        """
        if one_per_run:
            # Every queued run must actually advance the cursor, so this node
            # can never be served from cache. Same trick the prompt injector
            # uses for the same reason.
            _cursor.ticks += 1
            return f"one_per_run:{_cursor.ticks}:{time.time_ns()}"
        stamp = ""
        if mode != MODE_COUNT:
            target = _clean_path(path)
            try:
                if target.is_dir():
                    suffixes = TEXT_SUFFIXES if mode == MODE_TEXT_FILES else IMAGE_SUFFIXES
                    entries = sorted(
                        (entry.name, entry.stat().st_mtime_ns, entry.stat().st_size)
                        for entry in target.iterdir()
                        if entry.is_file() and entry.suffix.lower() in suffixes
                    )
                    stamp = str(entries)
                elif target.is_file():
                    info = target.stat()
                    stamp = f"{info.st_mtime_ns}_{info.st_size}"
            except OSError as exc:
                stamp = f"unreadable:{exc}"
        return f"{mode}|{path}|{count}|{start_at}|{limit}|{seed}|{stamp}"

    @classmethod
    def execute(cls, mode, path, count, start_at, limit, seed, one_per_run=False) -> IO.NodeOutput:
        items = _collect(str(mode), str(path), int(count))

        start = max(0, int(start_at))
        items = items[start:]
        if int(limit) > 0:
            items = items[: int(limit)]

        if not items:
            raise ValueError(
                f"{LOG_PREFIX} Nothing to do: the job list is empty after "
                f"start_at={start_at} and limit={limit}."
            )

        total = len(items)
        indices = [start + offset for offset in range(total)]
        # A derived seed per item, not base+i: neighbouring seeds are a poor
        # source of variety, and the whole point of this output is variety.
        seeds = [random.Random(f"{int(seed)}:{i}").getrandbits(32) for i in indices]

        if one_per_run:
            key = f"{mode}|{path}|{count}|{start_at}|{limit}"
            position = _cursor.positions.get(key, 0)
            if position >= total:
                # The job is done; the next queued run starts it over rather
                # than failing, so a Batch count set too high simply loops.
                position = 0
            _cursor.positions[key] = position + 1
            logger.info(
                "%s %s: job %d/%d (one per run) -> %s",
                LOG_PREFIX, mode, position + 1, total, items[position],
            )
            # Still a one-element list on the list outputs: the branch below
            # runs exactly once, which is the entire point of this mode.
            return IO.NodeOutput(
                [items[position]], [indices[position]], total, [seeds[position]],
            )

        logger.info(
            "%s %s: %d item(s), index %d..%d",
            LOG_PREFIX, mode, total, indices[0], indices[-1],
        )
        return IO.NodeOutput(items, indices, total, seeds)


NODE_CLASS_MAPPINGS = {"TS_BatchSource": TS_BatchSource}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_BatchSource": "TS Batch Source"}
