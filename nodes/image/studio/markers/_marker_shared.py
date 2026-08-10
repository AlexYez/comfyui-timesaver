"""Shared pieces of the TS Image Studio marker nodes.

Markers are the entry/exit points of a studio backend workflow
(doc/IMAGE_STUDIO_PLAN.md §3). Each is a tiny value source or passthrough;
the studio frontend finds them in the exported API-format JSON by node type,
reads ``param_name`` and replaces ``value`` before submitting the prompt.

Contract note: node ids, input names and their order are referenced by every
backend workflow file ever exported. They are frozen by the contract snapshot
and must never change.
"""
from __future__ import annotations

import logging

CATEGORY = "TS/Studio"
LOG_PREFIX = "[TS Studio]"

logger = logging.getLogger("comfyui_timesaver.ts_image_studio")

PARAM_TOOLTIP = (
    "Name the studio uses to find this marker and set its value "
    "(e.g. 'prompt', 'width', 'source_image'). Must be unique per workflow."
)

LABEL_TOOLTIP = (
    "Optional human-readable label shown in the studio UI instead of param_name."
)


def annotated_input_path(value: str) -> str:
    """Absolute path of an annotated upload name ("sub/file.png [temp]").

    Пометка в скобках называет папку ComfyUI: студия кладёт рабочие файлы во
    временную (`temp`), потому что входную индексируют сторонние браузеры
    ассетов и маски начинали появляться в библиотеке рядом с работами.

    Raises RuntimeError with a TS-prefixed message when the file is missing —
    a backend run with an unset image parameter should fail with words, not a
    PIL traceback.
    """
    import folder_paths

    name = (value or "").strip()
    if not name:
        raise RuntimeError(f"{LOG_PREFIX} Image parameter is empty — the studio did not set it.")
    if not folder_paths.exists_annotated_filepath(name):
        raise RuntimeError(
            f"{LOG_PREFIX} Image '{name}' is gone. Working files live in ComfyUI's "
            "temp folder, which is cleared on restart — drop the picture in again."
        )
    return folder_paths.get_annotated_filepath(name)


def load_image_tensor(path: str):
    """[1, H, W, C] float32 image tensor plus [1, H, W] mask from alpha."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb = torch.from_numpy(arr[..., :3]).unsqueeze(0)
    alpha = torch.from_numpy(arr[..., 3]).unsqueeze(0)
    return rgb, 1.0 - alpha


def load_mask_tensor(path: str):
    """[1, H, W] float32 mask: white = selected, read from luminance."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def file_fingerprint(value: str) -> str:
    """Deterministic stamp of the referenced file: name + mtime + size.

    The studio writes versioned filenames, but a same-name overwrite must
    still invalidate the cache.
    """
    import os

    import folder_paths

    name = (value or "").strip()
    try:
        if name and folder_paths.exists_annotated_filepath(name):
            path = folder_paths.get_annotated_filepath(name)
            stat = os.stat(path)
            return f"{name}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        pass
    return f"{name}:missing"
