"""HTTP-роуты видео-нод: метаданные, миниатюры, пики, отдача файла, каталог форматов.

Регистрируются на импорте модуля, а не при создании ноды: фронтенду они нужны
независимо от того, какая нода появилась на холсте первой (тот же приём в
аудио-лоадере).

⚠️ ВСЯ ТЯЖЁЛАЯ РАБОТА — В ``asyncio.to_thread``. Один ``av.open`` в цикле событий
вешает весь ComfyUI вместе с вебсокетами; эту аварию в паке уже ловили.
Параллелизм ограничен семафором, а одинаковые запросы схлопываются замком по
ключу — иначе три ноды на один файл декодируют его трижды.

⚠️ ПУТИ НЕ ОГРАНИЧЕНЫ ПАПКАМИ ComfyUI — это осознанный выбор владельца пака:
часовые исходники не таскают в ``input``. Взамен действуют белый список
расширений, отсутствие листинга каталогов и безопасный лог.
"""

from __future__ import annotations

import asyncio
import logging
import os

from ..._shared import make_route_registrars, resolve_prompt_server
from ._common import (
    LOG_PREFIX,
    http_path_allowed,
    is_video_path,
    resolve_media_path,
    safe_log_path,
)

logger = logging.getLogger("comfyui_timesaver.ts_video.routes")

ROUTE_BASE = "/ts_video"

# Два одновременных декодирования: пул asyncio.to_thread общий на весь процесс
# ComfyUI, и десяток параллельных проб заморозит чужие задачи.
_GATE = asyncio.Semaphore(2)
_locks: dict[str, asyncio.Lock] = {}


def _log_warning(message: str) -> None:
    logger.warning("%s %s", LOG_PREFIX, message)


_PROMPT_SERVER = resolve_prompt_server(_log_warning)
_register_get, _register_post = make_route_registrars(_PROMPT_SERVER, _log_warning)

try:
    from aiohttp import web
except Exception:                           # noqa: BLE001 - вне ComfyUI и в тестах
    web = None                              # type: ignore[assignment]


def _lock_for(key: str) -> asyncio.Lock:
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    return lock


OUTSIDE_ROOTS_MESSAGE = (
    "This folder is not open to the in-node preview. The node itself reads any path; "
    "the preview is served over HTTP, so it stays inside ComfyUI's input/output/temp/user "
    "folders. Open yours with TS_MEDIA_EXTRA_ROOTS=<folder> (several separated by the OS "
    "path separator), or lift the limit with TS_MEDIA_ALLOW_ANY_PATH=1. Set it on this "
    "machine before starting ComfyUI."
)


def _requested_path(request) -> str | None:
    """Развернуть и проверить путь из запроса.

    Возвращает ``None``, если файл не годится: и «нет такого», и «не видео»
    отвечают одинаково, чтобы по ответу нельзя было прощупывать чужой диск.
    """
    raw = request.query.get("filepath", "")
    path = resolve_media_path(raw)
    if not path or not is_video_path(path) or not os.path.isfile(path):
        logger.debug("%s rejected path %s", LOG_PREFIX, safe_log_path(raw))
        return None
    return path


def _outside_allowed_roots(request) -> bool:
    """Лежит ли запрошенный путь ВНЕ корней, открытых для HTTP.

    ⚠️ Проверяется ДО существования файла и отвечает одинаково для любого пути
    снаружи — существующего и нет. Иначе «403 только на существующие» само по
    себе стало бы способом прощупать чужой диск.

    Нода читает любой путь — это решение владельца пака. Маршруты видит всякий,
    кто дотянулся до порта ComfyUI, поэтому им открыты только разрешённые корни.
    """
    raw = request.query.get("filepath", "")
    path = resolve_media_path(raw)
    if not path:
        return False
    if http_path_allowed(path):
        return False
    logger.info(
        "%s the preview may not read %s: outside the allowed roots. %s",
        LOG_PREFIX, safe_log_path(path), OUTSIDE_ROOTS_MESSAGE,
    )
    return True


def _float(request, name: str, default: float) -> float:
    try:
        return float(request.query.get(name, default))
    except (TypeError, ValueError):
        return default


def _int(request, name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(float(request.query.get(name, default)))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


@_register_get(f"{ROUTE_BASE}/metadata")
async def ts_video_metadata(request):
    """Всё, что нужно таймлайну до первого кадра: длительность, частота, пики."""
    if _outside_allowed_roots(request):
        return web.json_response({"error": OUTSIDE_ROOTS_MESSAGE}, status=403)
    path = _requested_path(request)
    if path is None:
        return web.json_response({"error": "File not found."}, status=404)

    want_peaks = request.query.get("peaks", "1") != "0"
    from ._probe import probe_cached

    async with _GATE:
        async with _lock_for(path):
            try:
                info = await asyncio.to_thread(probe_cached, path, want_peaks=want_peaks)
            except Exception as error:      # noqa: BLE001 - чужой файл может быть любым
                logger.warning("%s probe failed for %s: %s",
                               LOG_PREFIX, safe_log_path(path), error)
                return web.json_response({"error": str(error)}, status=422)

    return web.json_response({
        "schema": 1,
        "filename": info.filename,
        "duration": info.duration,
        "fps": info.fps,
        "fps_exact": list(info.fps_exact),
        "frame_count": info.frame_count,
        "frame_count_estimated": info.frame_count_estimated,
        "width": info.display_width or info.width,
        "height": info.display_height or info.height,
        "rotation": info.rotation,
        "sar": list(info.sar),
        "vfr": info.vfr,
        "codec": info.codec,
        "container": info.container,
        "pix_fmt": info.pix_fmt,
        "bit_depth": info.bit_depth,
        "has_alpha": info.has_alpha,
        "has_audio": info.has_audio,
        "faststart": info.faststart,
        "browser_playable": info.codec in ("h264", "vp8", "vp9", "av1"),
        "audio": ({"codec": info.audio.codec,
                   "sample_rate": info.audio.sample_rate,
                   "channels": info.audio.channels} if info.audio else None),
        "peaks": list(info.peaks) if info.peaks else None,
    })


@_register_get(f"{ROUTE_BASE}/strip")
async def ts_video_strip(request):
    """Спрайт миниатюр: одна лента вместо шестнадцати запросов."""
    if _outside_allowed_roots(request):
        return web.json_response({"error": OUTSIDE_ROOTS_MESSAGE}, status=403)
    path = _requested_path(request)
    if path is None:
        return web.json_response({"error": "File not found."}, status=404)

    step = max(0.001, _float(request, "step", 1.0))
    index = max(0, _int(request, "index", 0, 0, 10_000_000))
    count = _int(request, "count", 16, 1, 64)
    height = _int(request, "height", 54, 16, 256)
    start = index * count * step

    from ._probe import filmstrip_sprite

    async with _GATE:
        key = f"{path}|{step}|{index}|{count}|{height}"
        async with _lock_for(key):
            try:
                payload = await asyncio.to_thread(
                    filmstrip_sprite, path, start=start, step=step,
                    count=count, height=height)
            except Exception as error:      # noqa: BLE001
                logger.warning("%s filmstrip failed for %s: %s",
                               LOG_PREFIX, safe_log_path(path), error)
                return web.json_response({"error": str(error)}, status=422)

    return web.Response(
        body=payload,
        content_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@_register_get(f"{ROUTE_BASE}/peaks")
async def ts_video_peaks(request):
    """Огибающая звука на окне — когда обзорных пиков уже не хватает."""
    if _outside_allowed_roots(request):
        return web.json_response({"error": OUTSIDE_ROOTS_MESSAGE}, status=403)
    path = _requested_path(request)
    if path is None:
        return web.json_response({"error": "File not found."}, status=404)

    start = max(0.0, _float(request, "start", 0.0))
    end = max(start + 0.001, _float(request, "end", start + 1.0))
    bins = _int(request, "bins", 512, 8, 4096)

    from ._probe import peaks_window

    async with _GATE:
        async with _lock_for(f"{path}|peaks|{start}|{end}|{bins}"):
            try:
                data = await asyncio.to_thread(peaks_window, path, start, end, bins)
            except Exception as error:      # noqa: BLE001
                logger.debug("%s peaks failed: %s", LOG_PREFIX, error)
                data = []

    return web.json_response({"start": start, "end": end, "bins": bins, "data": data})


@_register_get(f"{ROUTE_BASE}/view")
async def ts_video_view(request):
    """Отдать файл плееру.

    ``FileResponse`` сам обрабатывает заголовок ``Range``, поэтому браузер умеет
    перематывать, не скачивая всё целиком.
    """
    if _outside_allowed_roots(request):
        return web.json_response({"error": OUTSIDE_ROOTS_MESSAGE}, status=403)
    path = _requested_path(request)
    if path is None:
        return web.json_response({"error": "File not found."}, status=404)
    return web.FileResponse(path)


@_register_get(f"{ROUTE_BASE}/formats")
async def ts_video_formats(request):
    """Каталог форматов сохранения — подписи, уровни качества, доступность."""
    from ._formats import catalog

    check_hardware = request.query.get("hardware", "0") == "1"
    payload = await asyncio.to_thread(catalog, check_hardware=check_hardware)
    return web.json_response(payload)
