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

import json
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

#: Every file in the folder, whatever its type — the node only hands out
#: paths, and TS Batch Load Image, TS Video Loader and TS Audio Loader all
#: take one on their `path` socket. First in the combo and the default.
MODE_FILES = "Files in folder"
#: ⚠️ LEGACY. The mode was renamed to MODE_FILES on 23.10.2026 and left the
#: combo, but graphs saved before still carry this value. It keeps working —
#: and keeps its old meaning, images only: a folder of pictures with .txt
#: captions beside them would otherwise start feeding text files into an
#: image loader. The core does not check this combo against its options,
#: because ``validate_inputs`` names ``mode`` (execution.py, 2026-10).
MODE_IMAGES = "Images in folder"
MODE_LINES = "Lines in text file"
MODE_COUNT = "Count only"
#: One prompt per file. ⚠️ Appended LAST to the combo: the options' order is
#: not stored in graphs, but the default and the existing names must not move.
MODE_TEXT_FILES = "Text files in folder"
#: The positive prompt each image was generated with, read from its metadata.
#: ⚠️ Appended after MODE_TEXT_FILES for the same reason.
MODE_IMAGE_PROMPTS = "Prompts from images in folder"

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


#: Files the OS drops into folders on its own. Nobody batches them on purpose.
_SYSTEM_FILES = frozenset({"thumbs.db", "desktop.ini", ".ds_store"})

# The 'file_types' choice of 'Files in folder'. Each list is what the loader
# downstream actually opens; tests/test_batch_source_files.py holds them
# against the loaders' own lists so the two cannot drift apart.
FILE_TYPES_ALL = "All"
FILE_TYPES_IMAGES = "Images"
FILE_TYPES_VIDEOS = "Videos"
FILE_TYPES_AUDIO = "Audio"
#: TS Batch Load Image (Pillow): the classic set, plus GIF (its first frame)
#: and AVIF, which the Pillow shipped with ComfyUI reads (11.3, checked).
IMAGE_TYPE_SUFFIXES = IMAGE_SUFFIXES + (".gif", ".avif")
#: TS Video Loader's VIDEO_EXTENSIONS without the animated pictures (.gif,
#: .webp, .apng, .avif): asked for "videos", nobody means a folder's stills.
VIDEO_TYPE_SUFFIXES = (
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg",
    ".m2ts", ".mts", ".ts", ".flv", ".wmv", ".ogv",
    ".mxf", ".qt", ".dv", ".divx",
    ".m1v", ".m2v", ".mpv", ".vob", ".m2p", ".mod", ".tod",
    ".3gp", ".3g2", ".f4v", ".asf", ".rm", ".rmvb", ".ogm",
    ".y4m", ".nut", ".ivf", ".mjpeg", ".mjpg",
)
#: TS Audio Loader's SUPPORTED_AUDIO_EXTENSIONS without .webm, a video
#: container: sound from videos is the 'Videos' choice.
AUDIO_TYPE_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac",
                       ".aif", ".aiff", ".wma")
_FILE_TYPE_SUFFIXES = {
    FILE_TYPES_ALL: None,
    FILE_TYPES_IMAGES: IMAGE_TYPE_SUFFIXES,
    FILE_TYPES_VIDEOS: VIDEO_TYPE_SUFFIXES,
    FILE_TYPES_AUDIO: AUDIO_TYPE_SUFFIXES,
}


def _folder_files(folder: Path, suffixes: tuple[str, ...] | None,
                  recursive: bool = False) -> list[Path]:
    """Files inside ``folder`` (and its subfolders, if asked), in natural order.

    ``suffixes=None`` takes every file. Hidden files and folders (a leading
    dot) and the OS's own droppings are skipped. With subfolders the order is
    natural per path segment: ``a/2.png`` before ``a/10.png`` before ``b/1.png``.
    """
    entries = folder.rglob("*") if recursive else folder.iterdir()
    found = []
    for entry in entries:
        relative = entry.relative_to(folder)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if not entry.is_file() or entry.name.lower() in _SYSTEM_FILES:
            continue
        if suffixes is not None and entry.suffix.lower() not in suffixes:
            continue
        found.append(entry)
    found.sort(key=lambda entry: [_natural_key(part) for part in entry.relative_to(folder).parts])
    return found


def _folder_items(folder: Path, suffixes: tuple[str, ...] | None = IMAGE_SUFFIXES,
                  recursive: bool = False) -> list[str]:
    """Paths of the matching files in ``folder``, in natural order."""
    return [str(entry) for entry in _folder_files(folder, suffixes, recursive)]


def _folder_texts(folder: Path, recursive: bool = False) -> list[str]:
    """The text of every prompt file in ``folder``, one job per file.

    The whole file is the prompt — line breaks inside it stay, unlike the
    "Lines in text file" mode, where every line is a job of its own. A BOM
    (Notepad's UTF-8) is dropped, outer whitespace trimmed; empty files and
    files too big to be a prompt are skipped with a word in the log rather than
    becoming blank or enormous jobs.
    """
    texts: list[str] = []
    skipped: list[str] = []
    for entry in _folder_files(folder, TEXT_SUFFIXES, recursive):
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


# ─────────────────────── prompts read back from images ───────────────────────
#
# Who writes what (checked against real files, 23.10.2026):
#   * A1111 / Forge / civitai.com — PNG text chunk "parameters", or EXIF
#     UserComment in JPEG/WebP: the prompt, then "Negative prompt: …", then a
#     line starting "Steps: ". Written for people, so it is read first.
#   * ComfyUI — PNG text chunk "prompt": the API graph of the run. The prompt
#     is the text of the encoder wired into the sampler's "positive". When it
#     came down a wire (an LLM, a wildcard node), the graph holds only the wire
#     — unless TS Image Prompt Injector wrote the real text into it, which is
#     exactly what that node is for. ComfyUI's WebP/JPEG savers put the same
#     graph into EXIF as "prompt:{…}" (tag Model or Make).
#   * NovelAI — PNG "Description" is the prompt itself.

#: Encoder inputs that hold the prompt, most specific first. FLUX's encoder
#: carries the prompt twice (clip_l and t5xxl); t5xxl is the full one.
_PROMPT_KEYS = ("text", "prompt", "t5xxl", "text_g", "clip_l", "text_l")
#: Nodes that HOLD a string the person typed, and the input it lives in.
#: ⚠️ A whitelist on purpose. Following any "text"/"prompt" input of any node
#: would walk into an LLM node and return its INSTRUCTION as the image prompt —
#: measured on a real Moss graph, where TextGenerate's "prompt" is the idea
#: handed to the model, not the caption it wrote.
_STRING_SOURCES = {
    "PrimitiveString": "value",
    "PrimitiveStringMultiline": "value",
    "String Literal": "string",
    "StringConstant": "string",
    "StringConstantMultiline": "string",
    "Text Multiline": "text",
    "ImpactWildcardProcessor": "populated_text",
}
#: Nodes that pass a string through unchanged, and the input it comes from.
_PASSTHROUGH = {
    "PreviewAny": "source",
    "Reroute": "input",
    "ShowText|pysssss": "text",
}
_STEPS_LINE = re.compile(r"^Steps: ", re.MULTILINE)
_EXIF_USER_COMMENT = 0x9286
_EXIF_IFD = 0x8769
_EXIF_GRAPH_TAGS = (0x0110, 0x010F, 0x010E)  # Model, Make, ImageDescription


def _is_link(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2 and not isinstance(value[0], (dict, list))


def _a1111_positive(parameters: str) -> str:
    """The positive prompt of an A1111-style parameters block, or ''."""
    text = str(parameters or "").replace("\r\n", "\n")
    steps = list(_STEPS_LINE.finditer(text))
    if not steps:
        return ""
    head = text[: steps[-1].start()]
    cut = head.find("\nNegative prompt:")
    if head.startswith("Negative prompt:"):
        cut = 0
    return (head if cut < 0 else head[:cut]).strip()


def _is_encoder(node: dict) -> bool:
    class_type = str(node.get("class_type", ""))
    return class_type.startswith("CLIPTextEncode") or "TextEncode" in class_type


def _style_prompt(style_id) -> str | None:
    """What TS Style Prompt Selector put out for this style, or None if unknown."""
    selected = str(style_id or "").strip()
    if not selected or selected == "None":
        return " "  # the node's own answer for "no style"
    try:
        from ..text.ts_style_prompt_selector import _load_styles
    except Exception:  # noqa: BLE001 - outside ComfyUI the library is not there
        return None
    for style in _load_styles() or []:
        if selected in {str(style.get("id", "")).strip(), str(style.get("name", "")).strip()}:
            return str(style.get("prompt") or "")
    return None


def _string_value(graph: dict, value, depth: int = 0) -> str | None:
    """A literal string, or the string a wired node holds.

    None when the text was made at run time (an LLM, an unknown node) — the
    graph keeps only the wire then, and guessing would hand out the wrong text.
    '' is a real, empty string and is NOT the same thing: a concatenation with
    an empty style is still a known prompt.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]  # ShowText stores its text as a one-element list
    if not _is_link(value) or depth > 12:
        return None
    node = graph.get(str(value[0]))
    if not isinstance(node, dict):
        return None
    class_type = str(node.get("class_type", ""))
    inputs = node.get("inputs") or {}

    def follow(key):
        return _string_value(graph, inputs.get(key), depth + 1)

    switch = inputs.get("switch")
    if isinstance(switch, bool) and "on_true" in inputs and "on_false" in inputs:
        # A switch set by hand: the branch it points at is the text that ran.
        return follow("on_true" if switch else "on_false")
    if class_type == "StringConcatenate":
        first, second = follow("string_a"), follow("string_b")
        delimiter = inputs.get("delimiter", "")
        if first is None or second is None or not isinstance(delimiter, str):
            return None
        return first + delimiter + second
    if class_type == "TS_SuperPrompt":
        # The Enhance button rewrites the typed field, so the field IS the
        # prompt that ran. A wired prompt goes out as it came, unless the node
        # was told to enhance it at run time — that text exists nowhere.
        if _is_link(inputs.get("prompt")):
            return None if inputs.get("enhance_prompt") else follow("prompt")
        return inputs.get("text") if isinstance(inputs.get("text"), str) else None
    if class_type == "TS_StylePromptSelector":
        return _style_prompt(inputs.get("style_id"))
    key = _STRING_SOURCES.get(class_type) or _PASSTHROUGH.get(class_type)
    if key and key in inputs:
        return follow(key)
    return None


def _encoder_text(graph: dict, node: dict) -> str:
    inputs = node.get("inputs") or {}
    for key in _PROMPT_KEYS:
        if key in inputs:
            text = _string_value(graph, inputs[key])
            if text and text.strip():
                return text.strip()
    return ""


def _encoders_below(graph: dict, roots: list) -> list[str]:
    """Encoders feeding these conditioning wires, in the order they are reached."""
    found: list[str] = []
    seen: set[str] = set()
    stack = [str(root) for root in reversed(roots)]
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        node = graph.get(key)
        if not isinstance(node, dict):
            continue
        if _is_encoder(node):
            found.append(key)
            continue  # an encoder's own inputs are CLIP and text, not more conditioning
        inputs = node.get("inputs") or {}
        stack.extend(str(v[0]) for v in reversed(list(inputs.values())) if _is_link(v))
    return found


def _comfy_positive(graph) -> str:
    """The positive prompt of a ComfyUI API graph, or ''."""
    if not isinstance(graph, dict):
        return ""
    graph = {str(k): v for k, v in graph.items() if isinstance(v, dict)}

    def first_pass(item) -> tuple:
        # The sampler that starts from an empty latent is the one the prompt
        # was written for; an upscale pass may reuse or tweak it.
        _, node = item
        latent = (node.get("inputs") or {}).get("latent_image")
        source = graph.get(str(latent[0])) if _is_link(latent) else None
        empty = isinstance(source, dict) and "Empty" in str(source.get("class_type", ""))
        return (0 if empty else 1, int(item[0]) if item[0].isdigit() else 0)

    nodes = sorted(graph.items(), key=first_pass)
    positive = [n["inputs"]["positive"][0] for _, n in nodes
                if _is_link((n.get("inputs") or {}).get("positive"))]
    negative = [n["inputs"]["negative"][0] for _, n in nodes
                if _is_link((n.get("inputs") or {}).get("negative"))]
    if not positive:
        # BasicGuider (FLUX, Qwen) takes one "conditioning" and no negative.
        positive = [n["inputs"]["conditioning"][0] for _, n in nodes
                    if "Guider" in str(n.get("class_type", ""))
                    and _is_link((n.get("inputs") or {}).get("conditioning"))]

    encoders = _encoders_below(graph, positive)
    # A negative made by zeroing the positive (ConditioningZeroOut) reaches the
    # same encoder; dropping it then would leave nothing, so it only filters.
    negatives = set(_encoders_below(graph, negative))
    ordered = [e for e in encoders if e not in negatives] or encoders
    for key in ordered:
        text = _encoder_text(graph, graph[key])
        if text.strip():
            return text
    return ""


def _decode_user_comment(raw) -> str:
    """EXIF UserComment: 8-byte charset header, then the text (A1111 writes UTF-16)."""
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, (bytes, bytearray)):
        return ""
    head, body = bytes(raw[:8]), bytes(raw[8:])
    if head.startswith(b"UNICODE"):
        # Byte order is not declared. A prompt starts with a Latin letter or a
        # bracket, so its high byte is zero: first byte zero = big-endian.
        big = body[:1] == b"\x00"
        return body.decode("utf-16-be" if big else "utf-16-le", errors="replace").rstrip("\x00")
    return body.decode("utf-8", errors="replace").rstrip("\x00")


def _json_graph(text: str):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _image_prompt(path: Path) -> str:
    """The positive prompt stored in one image, or '' when it carries none."""
    from PIL import Image  # Pillow ships with ComfyUI; imported only for this mode

    with Image.open(path) as image:
        info = dict(image.info or {})
        # ComfyUI writes its chunks before the pixels, but other tools put text
        # AFTER them, where Image.open has not read yet. `.text` reads on.
        if not any(key in info for key in ("parameters", "prompt", "Description")):
            try:
                info.update(getattr(image, "text", {}) or {})
            except Exception as exc:  # noqa: BLE001 - a damaged tail must not stop the batch
                logger.debug("%s %s: trailing text chunks unreadable: %s", LOG_PREFIX, path.name, exc)
        exif_texts: list[str] = []
        try:
            exif = image.getexif()
            exif_texts.append(_decode_user_comment(exif.get_ifd(_EXIF_IFD).get(_EXIF_USER_COMMENT)))
            exif_texts.extend(str(exif.get(tag) or "") for tag in _EXIF_GRAPH_TAGS)
        except Exception as exc:  # noqa: BLE001 - EXIF is optional decoration
            logger.debug("%s %s: EXIF unreadable: %s", LOG_PREFIX, path.name, exc)

    candidates = [info.get("parameters")] + [t for t in exif_texts if "Steps: " in t]
    for parameters in candidates:
        text = _a1111_positive(parameters) if isinstance(parameters, str) else ""
        if text:
            return text

    graphs = [info.get("prompt")] + [t.split(":", 1)[1] for t in exif_texts if t.startswith("prompt:")]
    for raw in graphs:
        text = _comfy_positive(_json_graph(raw) if isinstance(raw, str) else raw)
        if text.strip():
            return text.strip()

    description = info.get("Description")
    return description.strip() if isinstance(description, str) else ""


def _folder_image_prompts(folder: Path, recursive: bool = False) -> list[str]:
    """The positive prompt of every image in ``folder``, one job per image.

    Images that carry none — a photo, a screenshot, a picture whose prompt was
    generated on the fly and never written back — are skipped and named in the
    log, the same way empty prompt files are.
    """
    prompts: list[str] = []
    skipped: list[str] = []
    for entry in _folder_files(folder, IMAGE_SUFFIXES, recursive):
        try:
            text = _image_prompt(entry)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not sink the batch
            logger.warning("%s Skipped %s: cannot read it (%s).", LOG_PREFIX, entry.name, exc)
            continue
        if text:
            prompts.append(text)
        else:
            skipped.append(entry.name)
    if skipped:
        logger.info("%s %d image(s) carry no prompt and were skipped: %s", LOG_PREFIX, len(skipped),
                    ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""))
    return prompts


def _file_lines(file_path: Path) -> list[str]:
    """Non-empty lines of a text file, whitespace trimmed."""
    raw = file_path.read_text(encoding="utf-8", errors="replace")
    return [line.strip() for line in raw.replace("\r\n", "\n").split("\n") if line.strip()]


def _clean_path(path: str) -> Path:
    """The user's path, with the quotes Explorer's "Copy as path" adds removed."""
    return Path(str(path).strip().strip('"'))


_FOLDER_MODES = (MODE_FILES, MODE_IMAGES, MODE_TEXT_FILES, MODE_IMAGE_PROMPTS)
_KNOWN_MODES = (MODE_FILES, MODE_LINES, MODE_COUNT, MODE_TEXT_FILES, MODE_IMAGE_PROMPTS, MODE_IMAGES)


def _mode_suffixes(mode: str, file_types: str | None = FILE_TYPES_ALL) -> tuple[str, ...] | None:
    """Which files a folder mode reads: None = every file."""
    if mode == MODE_FILES:
        return _FILE_TYPE_SUFFIXES.get(str(file_types or FILE_TYPES_ALL))
    if mode == MODE_TEXT_FILES:
        return TEXT_SUFFIXES
    return IMAGE_SUFFIXES


def _collect(mode: str, path: str, count: int, recursive: bool = False,
             file_types: str = FILE_TYPES_ALL) -> list[str]:
    """The job's items, before ``start_at`` and ``limit`` narrow them down."""
    if mode == MODE_COUNT:
        return [str(i) for i in range(int(count))]

    target = _clean_path(path)
    if not str(target).strip():
        raise ValueError(f"{LOG_PREFIX} Mode '{mode}' needs a path, but none was given.")
    if not target.exists():
        raise FileNotFoundError(f"{LOG_PREFIX} Path does not exist: {target}")

    if mode in _FOLDER_MODES:
        if not target.is_dir():
            raise NotADirectoryError(f"{LOG_PREFIX} Expected a folder, got a file: {target}")
        if mode == MODE_IMAGE_PROMPTS:
            return _folder_image_prompts(target, recursive)
        if mode == MODE_TEXT_FILES:
            return _folder_texts(target, recursive)
        return _folder_items(target, _mode_suffixes(mode, file_types), recursive)

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
                "Turn a folder of files (images, videos, audio — any type), a folder of "
                "prompt files, the prompts stored in a folder of generated images, a text "
                "file or a plain count into a job list. Everything wired below this node "
                "runs once per "
                "item, independently — so a hundred captions are a hundred separate model "
                "calls, not one growing context."
            ),
            inputs=[
                IO.Combo.Input(
                    "mode",
                    # ⚠️ "Images in folder" left this list on 23.10.2026 and became
                    # "Files in folder". Graphs saved with it keep working (images
                    # only, as before) — see MODE_IMAGES.
                    options=[MODE_FILES, MODE_LINES, MODE_COUNT, MODE_TEXT_FILES,
                             MODE_IMAGE_PROMPTS],
                    default=MODE_FILES,
                    tooltip=(
                        "Where the jobs come from: the paths of the files in a folder "
                        "(any type — feed them to TS Batch Load Image, or to the 'path' "
                        "socket of TS Video Loader or TS Audio Loader), non-empty lines of a "
                        "text file, simply N numbered iterations, prompt files in a folder "
                        "(.txt / .md, one prompt per file, its line breaks kept), or the "
                        "positive prompts of generated images in a folder (ComfyUI, A1111 / "
                        "Forge, Civitai, NovelAI metadata; images without one are skipped)."
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
                # ⚠️ Both appended LAST and optional: widgets_values is positional,
                # and a graph saved before them simply does not carry them.
                IO.Boolean.Input(
                    "include_subfolders",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Also walk every folder inside the folder. The order stays "
                        "natural per path: a/2 before a/10 before b/1. Hidden folders "
                        "(a leading dot) are skipped."
                    ),
                ),
                IO.Combo.Input(
                    "file_types",
                    options=[FILE_TYPES_ALL, FILE_TYPES_IMAGES, FILE_TYPES_VIDEOS, FILE_TYPES_AUDIO],
                    default=FILE_TYPES_ALL,
                    optional=True,
                    tooltip=(
                        "'Files in folder' only: which files to take. All — every file. "
                        "Images — what TS Batch Load Image reads (png, jpg, webp, bmp, tiff, "
                        "gif, avif). Videos — what TS Video Loader reads (mp4, mov, mkv, webm, "
                        "avi, mxf, ProRes and broadcast containers, …). Audio — what TS Audio "
                        "Loader reads (wav, mp3, flac, ogg, opus, m4a, aac, aiff, wma)."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="item",
                    is_output_list=True,
                    tooltip=(
                        "One job per element: a file path, a text line, a number, "
                        "the whole text of a prompt file, or the prompt an image was "
                        "generated with."
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
    def validate_inputs(cls, mode, path, count, start_at, limit, seed, one_per_run=False,
                        include_subfolders=False, file_types=FILE_TYPES_ALL) -> bool | str:
        # ⚠️ `mode` named here means the core does NOT check it against the
        # combo — which is what lets graphs saved with "Images in folder" run.
        # So the check is ours, and it accepts that legacy name.
        if mode not in _KNOWN_MODES:
            return f"{LOG_PREFIX} Unknown mode '{mode}'."
        if file_types is not None and str(file_types) not in _FILE_TYPE_SUFFIXES:
            return f"{LOG_PREFIX} Unknown file_types '{file_types}'."
        if mode != MODE_COUNT and not str(path).strip():
            return f"{LOG_PREFIX} Mode '{mode}' needs a path."
        return True

    @classmethod
    def fingerprint_inputs(cls, mode, path, count, start_at, limit, seed, one_per_run=False,
                           include_subfolders=False, file_types=FILE_TYPES_ALL) -> str:
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
                    # The very listing execute() will use, subfolders included.
                    entries = [
                        (str(entry.relative_to(target)), entry.stat().st_mtime_ns,
                         entry.stat().st_size)
                        for entry in _folder_files(target, _mode_suffixes(mode, file_types),
                                                   bool(include_subfolders))
                    ]
                    stamp = str(entries)
                elif target.is_file():
                    info = target.stat()
                    stamp = f"{info.st_mtime_ns}_{info.st_size}"
            except OSError as exc:
                stamp = f"unreadable:{exc}"
        return (f"{mode}|{path}|{count}|{start_at}|{limit}|{seed}|"
                f"{bool(include_subfolders)}|{file_types or FILE_TYPES_ALL}|{stamp}")

    @classmethod
    def execute(cls, mode, path, count, start_at, limit, seed, one_per_run=False,
                include_subfolders=False, file_types=FILE_TYPES_ALL) -> IO.NodeOutput:
        # Optional inputs of a graph saved before them arrive as None.
        recursive = bool(include_subfolders)
        file_types = str(file_types or FILE_TYPES_ALL)
        items = _collect(str(mode), str(path), int(count), recursive, file_types)

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
            key = f"{mode}|{path}|{count}|{start_at}|{limit}|{recursive}|{file_types}"
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
