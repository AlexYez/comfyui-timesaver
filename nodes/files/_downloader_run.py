"""Запуск загрузчика КНОПКОЙ, без прогона графа.

Зачем отдельный маршрут, когда рядом уже есть ``/ts_downloader/fetch``: тот
принимает по одной модели и с прибитыми настройками — без токенов, без зеркал,
со строгой проверкой «папка обязана быть зарегистрированной». Он писан под
студию. Кнопке на ноде нужно ровно то, что делает сама нода: её список, её
токены, её зеркала, её распаковка.

    POST /ts_downloader/run     {file_list, ...виджеты..., operation_id} -> {ok}
    POST /ts_downloader/run_cancel {operation_id}

События: ``ts_downloader.run_progress`` — те же поля, что копит ``_RunProgress``,
плюс ``operation_id`` и ``stage`` (``started`` / ``progress`` / ``finished`` /
``error``).

⚠️ Прогон один на процесс. Две кнопки, нажатые разом, качали бы одни и те же
файлы в один и тот же ``.part``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any

from .._shared import make_route_registrars, resolve_prompt_server

logger = logging.getLogger("comfyui_timesaver.ts_downloader.run")
LOG_PREFIX = "[TS Files Downloader]"

_PROMPT_SERVER = resolve_prompt_server(lambda message: logger.warning("%s %s", LOG_PREFIX, message))
_register_get, _register_post = make_route_registrars(
    _PROMPT_SERVER, lambda message: logger.warning("%s %s", LOG_PREFIX, message))

EVENT = "ts_downloader.run_progress"
_THROTTLE_S = 0.2

# Состояние единственного прогона: кто его начал и просили ли отменить.
_state: dict[str, Any] = {"operation_id": "", "cancel": False, "last_emit": 0.0}
_guard = threading.Lock()


def _emit(payload: dict, *, force: bool = False) -> None:
    if _PROMPT_SERVER is None:
        return
    now = time.monotonic()
    if not force and now - float(_state.get("last_emit") or 0.0) < _THROTTLE_S:
        return
    _state["last_emit"] = now
    try:
        _PROMPT_SERVER.send_sync(EVENT, payload)
    except Exception:                       # noqa: BLE001 - клиент мог уйти
        pass


class _Cancelled(RuntimeError):
    """Нажата «Отмена» — не ошибка загрузки, а решение человека."""


def _run_blocking(body: dict, operation_id: str) -> dict:
    """Позвать движок ноды ровно с её собственными настройками."""
    from .ts_downloader import TS_DownloadFilesNode as Node

    def sink(payload: dict) -> None:
        if _state.get("cancel") and _state.get("operation_id") == operation_id:
            # Тот же приём, что у студийных задач: отмена живёт в колбэке
            # прогресса, потому что это единственная точка, куда загрузка
            # заглядывает между кусками.
            raise _Cancelled()
        stage = "finished" if payload.get("status") == "done" else "progress"
        _emit({**payload, "operation_id": operation_id, "stage": stage},
              force=stage == "finished")

    def number(name: str, fallback):
        try:
            return type(fallback)(body.get(name, fallback))
        except (TypeError, ValueError):
            return fallback

    Node._execute_blocking(
        str(body.get("file_list") or ""),
        bool(body.get("skip_existing", True)),
        bool(body.get("verify_size", True)),
        number("chunk_size_kb", 4096),
        str(body.get("hf_token") or ""),
        str(body.get("hf_domain") or "huggingface.co, hf-mirror.com"),
        str(body.get("proxy_url") or ""),
        str(body.get("modelscope_token") or ""),
        bool(body.get("unzip_after_download", False)),
        # ⚠️ `enable` НЕ читается: этот выключатель говорит «не качай во время
        # прогона графа», а кнопку человек нажал только что и вручную.
        True,
        str(body.get("integrity_mode") or "hf_sha256_auto"),
        "",                                 # prompt_id: прогона графа здесь нет
        sink,
    )
    return {"ok": True}


async def _run_worker(body: dict, operation_id: str) -> None:
    try:
        await asyncio.to_thread(_run_blocking, body, operation_id)
    except _Cancelled:
        logger.info("%s Run cancelled by the user.", LOG_PREFIX)
        _emit({"operation_id": operation_id, "stage": "finished",
               "status": "cancelled"}, force=True)
    except Exception as exc:                # noqa: BLE001 - показываем причину человеку
        logger.error("%s Run failed: %s", LOG_PREFIX, exc)
        _emit({"operation_id": operation_id, "stage": "error",
               "status": "error", "error": str(exc)}, force=True)
    finally:
        with _guard:
            if _state.get("operation_id") == operation_id:
                _state["operation_id"] = ""
                _state["cancel"] = False


@_register_post("/ts_downloader/run")
async def run_route(request):
    """Скачать список прямо сейчас, не запуская граф."""
    from aiohttp import web

    try:
        body = await request.json()
    except Exception:                       # noqa: BLE001 - кривое тело
        return web.json_response({"error": "Expected a JSON body."}, status=400)

    if not str(body.get("file_list") or "").strip():
        return web.json_response({"error": "The list is empty."}, status=400)

    operation_id = str(body.get("operation_id") or uuid.uuid4().hex)
    with _guard:
        if _state.get("operation_id"):
            return web.json_response(
                {"error": "A download is already running."}, status=409)
        _state["operation_id"] = operation_id
        _state["cancel"] = False
        _state["last_emit"] = 0.0

    _emit({"operation_id": operation_id, "stage": "started", "status": "running"},
          force=True)
    asyncio.get_running_loop().create_task(_run_worker(body, operation_id))
    return web.json_response({"ok": True, "operation_id": operation_id})


@_register_post("/ts_downloader/run_cancel")
async def run_cancel_route(request):
    """Остановить прогон, начатый кнопкой.

    Незавершённый файл остаётся как ``.part`` — движок докачивает его Range-ом,
    поэтому отмена не выбрасывает уже скачанные гигабайты.
    """
    from aiohttp import web

    try:
        body = await request.json()
    except Exception:                       # noqa: BLE001
        body = {}

    with _guard:
        running = _state.get("operation_id")
        if not running:
            return web.json_response({"ok": True, "running": False})
        wanted = str(body.get("operation_id") or "")
        if wanted and wanted != running:
            return web.json_response({"ok": True, "running": True, "matched": False})
        _state["cancel"] = True
    return web.json_response({"ok": True, "running": True, "matched": True})


@_register_get("/ts_downloader/run_status")
async def run_status_route(_request):
    """Идёт ли загрузка — чтобы кнопка пережила перезагрузку страницы."""
    from aiohttp import web

    with _guard:
        return web.json_response({
            "running": bool(_state.get("operation_id")),
            "operation_id": _state.get("operation_id") or "",
        })
