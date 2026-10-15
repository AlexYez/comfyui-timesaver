"""Preview thumbnails for TS Film Emulation's look gallery.

Two GET routes:

* ``/ts_film/looks`` — the catalogue the gallery draws: an "Original" card, every
  built-in stock (family and a one-line note in en/ru), every ``.cube`` in the
  LUT folder. The folder is read on every call, so a LUT dropped in by the user
  shows up without a restart.
* ``/ts_film/preview?preset=…&lut=…&size=small|large`` — a JPEG of the bundled
  reference portrait run through the node's OWN pipeline (``_process_chunk``,
  the node's defaults, no grain — grain is a separate control and would only
  make every card look noisy). What the card shows is what the node does.
  Optional ``strength``, ``contrast``, ``warmth``, ``fade``,
  ``shadow_saturation``, ``highlight_saturation`` (clamped to the schema's
  ranges) render the look WITH the node's sliders — that is the thumbnail in
  the node itself.

⚠️ Names are matched against the catalogue, never joined into a path: the LUT
name comes from the request, and only a file that ``_scan_luts`` listed itself
is ever opened.

Rendering runs on the CPU in a worker thread — a 480 px preview costs tens of
milliseconds, and the event loop stays free. Results are cached by look and by
the LUT file's size and mtime, so an edited LUT renders afresh.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
from collections import OrderedDict

import torch

from ..._shared import make_route_registrars, resolve_prompt_server
from ._film_stocks import FAMILY_LABELS, STOCKS

logger = logging.getLogger("comfyui_timesaver.ts_film_emulation")
LOG_PREFIX = "[TS Film Emulation]"

REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.jpg")
ORIGINAL = ("External LUT", "None")
SIZES = {"small": 480, "large": 960}
_CACHE_LIMIT = 160

_CACHE: "OrderedDict[tuple, bytes]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_REFERENCE: dict[int, torch.Tensor] = {}


def _node():
    # Imported lazily: the node module imports this one to register the routes.
    from .ts_film_emulation import TS_Film_Emulation

    return TS_Film_Emulation


def catalogue() -> dict:
    """Everything the gallery shows, in gallery order."""
    looks = [{
        "id": "original", "kind": "original", "preset": ORIGINAL[0], "lut": ORIGINAL[1],
        "family": "original", "name": "Original",
        "note_en": "No look: the picture as it comes in.",
        "note_ru": "Без обработки: картинка как есть.",
    }]
    for stock in STOCKS:
        looks.append({
            "id": f"stock:{stock.name}", "kind": "stock", "preset": stock.name, "lut": "None",
            "family": stock.family, "name": stock.name,
            "note_en": stock.note_en, "note_ru": stock.note_ru,
        })
    for lut in _node()._scan_luts():
        if lut == "None":
            continue
        looks.append({
            "id": f"lut:{lut}", "kind": "lut", "preset": "External LUT", "lut": lut,
            "family": "lut", "name": os.path.splitext(lut)[0],
            "note_en": f"External LUT file {lut}.", "note_ru": f"Внешний LUT-файл {lut}.",
        })
    families = [{"id": key, "en": en, "ru": ru} for key, (en, ru) in FAMILY_LABELS.items()]
    families.append({"id": "lut", "en": "LUT files", "ru": "LUT-файлы"})
    return {"looks": looks, "families": families}


def _reference(width: int) -> torch.Tensor:
    cached = _REFERENCE.get(width)
    if cached is not None:
        return cached
    from PIL import Image
    import numpy as np

    with Image.open(REFERENCE_PATH) as image:
        image = image.convert("RGB")
        height = round(image.height * width / image.width)
        image = image.resize((width, height), Image.LANCZOS)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
    _REFERENCE[width] = tensor
    return tensor


def _lut_stamp(lut: str) -> tuple:
    if lut == "None":
        return ()
    try:
        stat = os.stat(os.path.join(_node()._resolve_luts_dir(), lut))
    except OSError:
        return ("missing",)
    return (stat.st_size, stat.st_mtime_ns)


#: The node's own adjustments the thumbnail can follow: name -> (neutral, min, max).
#: Grain is left out on purpose — at thumbnail size it is noise, not texture.
ADJUSTMENTS = {
    "strength": (1.0, 0.0, 1.0),
    "contrast": (1.0, 0.0, 3.0),
    "warmth": (0.0, -1.0, 1.0),
    "fade": (0.0, 0.0, 0.5),
    "shadow_saturation": (1.0, 0.0, 2.0),
    "highlight_saturation": (1.0, 0.0, 2.0),
}


def parse_adjustments(query) -> dict:
    """Read the adjustment values of a request, clamped; unknown or broken -> neutral."""
    values = {}
    for name, (neutral, low, high) in ADJUSTMENTS.items():
        try:
            value = float(query.get(name, neutral))
        except (TypeError, ValueError):
            value = neutral
        if value != value:  # NaN
            value = neutral
        values[name] = round(min(high, max(low, value)), 3)
    return values


def render_preview(preset: str, lut: str, size: str = "small", adjust: dict | None = None) -> bytes:
    """JPEG bytes of the reference with this look; raises KeyError for unknown names.

    ``adjust`` (see ADJUSTMENTS) lets the node's thumbnail show the look WITH
    the user's sliders; the gallery cards leave it out and show the look alone.
    """
    node = _node()
    from ._film_stocks import PRESET_OPTIONS

    if preset not in PRESET_OPTIONS:
        raise KeyError(f"unknown preset {preset!r}")
    if lut != "None" and lut not in node._scan_luts():
        raise KeyError(f"unknown LUT {lut!r}")
    width = SIZES.get(size, SIZES["small"])
    adjust = {name: (adjust or {}).get(name, spec[0]) for name, spec in ADJUSTMENTS.items()}
    neutral = all(adjust[name] == spec[0] for name, spec in ADJUSTMENTS.items())
    key = (preset, lut, _lut_stamp(lut), width, tuple(adjust.values()))
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]

    image = _reference(width)
    if (preset, lut) != ORIGINAL or not neutral:
        table, table_size = (None, None)
        if preset == "External LUT" and lut != "None":
            table, table_size = node.load_cube_lut(os.path.join(node._resolve_luts_dir(), lut))
        with torch.no_grad():
            image = node._process_chunk(
                image.clone(), film_preset=preset, lut=table, lut_size=table_size,
                lut_strength=adjust["strength"], gamma_correction=True, film_strength=adjust["strength"],
                contrast_curve=adjust["contrast"], warmth=adjust["warmth"], fade=adjust["fade"],
                shadow_saturation=adjust["shadow_saturation"],
                highlight_saturation=adjust["highlight_saturation"], grain_intensity=0.0, grain_size=0.5,
                grain_speed=1.0, grain_seed=0, frame_offset=0,
            )

    from PIL import Image
    import numpy as np

    array = (image[0].clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=88)
    data = buffer.getvalue()
    with _CACHE_LOCK:
        _CACHE[key] = data
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return data


def _warn_route(message: str) -> None:
    logger.warning("%s %s", LOG_PREFIX, message)


_register_get, _ = make_route_registrars(resolve_prompt_server(_warn_route), _warn_route)


@_register_get("/ts_film/looks")
async def ts_film_looks(request):
    from aiohttp import web

    return web.json_response(catalogue())


@_register_get("/ts_film/preview")
async def ts_film_preview(request):
    from aiohttp import web

    preset = request.query.get("preset", "")
    lut = request.query.get("lut", "None") or "None"
    size = request.query.get("size", "small")
    adjust = parse_adjustments(request.query)
    try:
        data = await asyncio.to_thread(render_preview, preset, lut, size, adjust)
    except KeyError:
        return web.Response(status=404)
    except Exception as exc:  # noqa: BLE001 - a broken LUT must not 500 the gallery
        logger.warning("%s Preview for %s / %s failed: %s", LOG_PREFIX, preset, lut, exc)
        return web.Response(status=500)
    return web.Response(body=data, content_type="image/jpeg",
                        headers={"Cache-Control": "no-cache"})
