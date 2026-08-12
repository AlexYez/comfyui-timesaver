"""Let ComfyUI's own image loaders see the files ComfyUI itself puts in subfolders.

The bug, in the order it happens:

1. Paste an image into a Load Image node (Ctrl+V), or drop one onto it. The
   frontend uploads it with ``subfolder=pasted``, so the file lands in
   ``input/pasted/`` and the widget's value becomes ``pasted/name.png``. The
   frontend pushes that value into the widget's option list itself, so
   everything works for the rest of the session.
2. Save the workflow, reload the page. The option list now comes from
   ``/object_info``, and core's ``LoadImage.INPUT_TYPES`` lists only files
   sitting directly in ``input`` — ``os.listdir`` with an ``isfile`` check.
3. The frontend's missing-media scanner looks for the saved value in that list,
   does not find it, and reports "A required media input has no file selected."
   The file is on disk the whole time; the run would even succeed.

So the fix is one line of behaviour: advertise every image UNDER ``input``,
not only the ones at its root. Nothing about how images are stored changes,
already-saved workflows start resolving again on their own, and no node is
replaced — ``LoadImage`` stays ``LoadImage``.

WHY PATCH CORE AT ALL. This pack does not modify ComfyUI, and this module is
the single exception: the value that fails validation is produced by ComfyUI
and stored by ComfyUI, and the only place that can vouch for it is the schema
core publishes. A node of our own could not fix a workflow that uses theirs.
The patch is idempotent, keeps core's own schema (so tooltips and flags added
by future versions survive), falls back to the untouched schema if the scan
fails, and can be switched off entirely with ``TS_DISABLE_PASTED_MEDIA_FIX=1``.

``LoadImageMask`` needs nothing extra: it subclasses ``LoadImage`` and builds
its schema by calling ``super().INPUT_TYPES()``, so it inherits the wider list.
``LoadImageOutput`` lists the output directory with its own code and is left
alone.
"""

from __future__ import annotations

import logging
import os
import time

LOGGER = logging.getLogger("comfyui_timesaver.pasted_media_fix")
LOG_PREFIX = "[TS PastedMediaFix]"

# Set on the patched class so a second import cannot wrap the wrapper.
PATCH_MARKER = "_ts_recursive_input_listing"
OPT_OUT_ENV = "TS_DISABLE_PASTED_MEDIA_FIX"


def _is_hidden(relative_path: str) -> bool:
    """True for anything inside a dot-folder (or named with a leading dot).

    Packs keep their working files in such folders precisely so they stay out
    of sight — Artius' 3D asset cache is one. Widening the list is meant to
    surface pictures a person put there, not everyone's scratch space.
    """
    return any(segment.startswith(".") for segment in relative_path.split("/") if segment)


# Короткая память о последнем обходе: (момент, каталог, список).
#
# ⚠️ Это КЭШ ПО ВРЕМЕНИ, а не сужение области. Доступ ко всем подпапкам `input`
# сохраняется полностью — ради него патч и написан. Обход же случается на
# КАЖДЫЙ `/object_info`, а его фронтенд спрашивает часто: при открытии графа, у
# каждой панели нод, после любой загрузки файла. На папке в десятки тысяч
# картинок это заметная пауза, причём результат между двумя соседними
# запросами почти всегда один и тот же.
#
# Полсекунды выбраны так, чтобы только что вставленная картинка появлялась в
# списке практически сразу: человек не успевает дойти до ноды быстрее.
_LISTING_TTL_SECONDS = 0.5
_listing_cache: tuple[float, str, list[str]] | None = None


def _list_input_images_uncached(folder_paths) -> list[str]:
    input_dir = folder_paths.get_input_directory()
    files, _ = folder_paths.recursive_search(input_dir)
    images = folder_paths.filter_files_content_types(files, ["image"])
    unique = {str(path).replace("\\", "/") for path in images}
    return sorted((path for path in unique if not _is_hidden(path)), key=str.casefold)


def list_input_images(folder_paths) -> list[str]:
    """Every image under the input directory, as ComfyUI would name it.

    Forward slashes on every platform: that is what the upload API returns and
    what saved workflows contain, and the frontend compares these strings
    literally.
    """
    global _listing_cache

    input_dir = str(folder_paths.get_input_directory())
    now = time.monotonic()
    cached = _listing_cache
    if cached is not None:
        stamped_at, cached_dir, cached_list = cached
        if cached_dir == input_dir and (now - stamped_at) < _LISTING_TTL_SECONDS:
            return list(cached_list)

    listing = _list_input_images_uncached(folder_paths)
    _listing_cache = (now, input_dir, listing)
    return list(listing)


def forget_cached_listing() -> None:
    """Забыть последний обход — для тестов и для явного обновления."""
    global _listing_cache
    _listing_cache = None


def _build_input_types(original, folder_paths):
    """Wrap core's INPUT_TYPES so the image list covers subfolders."""

    def recursive_input_types(cls):
        # Start from the running ComfyUI's own schema: only the list of files is
        # ours to change, and anything the node grows later comes along free.
        schema = original()
        try:
            spec = schema["required"]["image"]
            options = dict(spec[1]) if isinstance(spec, (tuple, list)) and len(spec) > 1 else {}
            schema["required"]["image"] = (list_input_images(folder_paths), options)
        except Exception:
            # An unreadable subdirectory must not take /object_info down with
            # it — that would break the whole editor, which is far worse than
            # the alert this module exists to remove.
            LOGGER.warning(
                "%s Could not list the input directory recursively; "
                "leaving core's own list in place.", LOG_PREFIX, exc_info=True,
            )
        return schema

    return recursive_input_types


def apply_patch() -> bool:
    """Install the wider listing. Returns True when this call did the patching."""
    if os.environ.get(OPT_OUT_ENV, "").strip():
        LOGGER.info("%s Switched off by %s.", LOG_PREFIX, OPT_OUT_ENV)
        return False
    try:
        import folder_paths
        import nodes
    except Exception:
        # Imported outside ComfyUI (tests, tooling) — nothing to patch.
        return False

    target = getattr(nodes, "LoadImage", None)
    original = getattr(target, "INPUT_TYPES", None)
    if target is None or original is None:
        LOGGER.debug("%s No LoadImage to patch in this ComfyUI.", LOG_PREFIX)
        return False
    if getattr(target, PATCH_MARKER, False):
        return False

    target.INPUT_TYPES = classmethod(_build_input_types(original, folder_paths))
    setattr(target, PATCH_MARKER, True)
    # Успешно применённая заплатка — не новость, а норма: сообщать о ней при
    # каждом запуске незачем. Всё, что пошло не так, по-прежнему на warning.
    LOGGER.debug(
        "%s Load Image and Load Image (as Mask) now list images in input subfolders "
        "(pasted, clipspace, and the rest).", LOG_PREFIX,
    )
    return True
