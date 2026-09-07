"""Что из списка уже лежит на диске: маршрут статусов для ноды.

Три состояния, и они не выдуманы — ровно так их различает сам движок загрузки:

    ready    файл на месте, а рядом ``<file>.tsmeta.json`` подтверждает, что он
             докачан до конца и проверен. Ровно это делает список скачанных
             моделей бесплатным: движок не ходит в сеть за тем, что уже есть.
    partial  на месте только ``<file>.part`` — прерванная докачка. Она не мусор:
             следующий запуск продолжит её Range-запросом, а не начнёт заново.
    missing  ни того, ни другого.

⚠️ Маршрут НЕ ходит в сеть. Он отвечает по одному диску, поэтому его можно
звать на каждое открытие ноды и после каждой правки списка — иначе список из
двадцати моделей стоил бы двадцати HEAD-запросов на каждый показ.

Есть и четвёртое состояние — ``unknown``: строку не удалось разобрать (нет
папки, кривой адрес). Молчать о ней нельзя: человек видит строку в списке и
вправе знать, что она не будет скачана.
"""

from __future__ import annotations

import logging
import threading
import os

from .._shared import make_route_registrars, resolve_prompt_server

logger = logging.getLogger("comfyui_timesaver.ts_downloader.status")

# ⚠️ Проверка статусов — это ЧТЕНИЕ ДИСКА, а не прогон. Разбор строки живёт в
# ноде и на кривой папке пишет предупреждение — правильное во время загрузки и
# бессмысленное здесь: у ноды с умолчательным списком-примером (`/path/to/models`)
# каждая перерисовка добавляла в журнал по две строки «Invalid target path».
# Гасим их НА ВРЕМЯ ПРОВЕРКИ И ТОЛЬКО В СВОЁМ ПОТОКЕ: настоящая загрузка может
# идти одновременно в другом, и её предупреждения человеку нужны.
_probe = threading.local()


class _QuietProbe(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(_probe, "quiet", False)


logging.getLogger("comfyui_timesaver.ts_downloader").addFilter(_QuietProbe())


class _quiet_probe:
    """Контекст, в котором разбор списка молчит."""

    def __enter__(self) -> "_quiet_probe":
        _probe.quiet = True
        return self

    def __exit__(self, *exc_info) -> None:
        _probe.quiet = False
LOG_PREFIX = "[TS Files Downloader]"

_PROMPT_SERVER = resolve_prompt_server(lambda message: logger.warning("%s %s", LOG_PREFIX, message))
_register_get, _register_post = make_route_registrars(
    _PROMPT_SERVER, lambda message: logger.warning("%s %s", LOG_PREFIX, message))

# Столько строк принимаем за раз. Список моделей в воркфлоу — это десятки, а не
# тысячи; предел стоит на случай, когда в поле вставили не то.
MAX_LINES = 500


def _entry_status(node, url: str, directory: str, meta_cache: dict) -> dict:
    """Состояние одной строки списка. Ни одного сетевого запроса.

    ⚠️ `directory` приходит УЖЕ разрешённой: её вернул `_parse_file_list`,
    который сам зовёт `_resolve_target_directory`. Разрешать второй раз — это
    прогонять через проверку путей то, что её уже прошло.
    """
    resolved = str(directory or "")
    if not resolved:
        return {"status": "unknown", "reason": "target", "directory": ""}

    name = node._sanitize_filename(
        os.path.basename(url.split("?")[0].split("#")[0].rstrip("/"))
    ) or ""

    # Сначала спрашиваем движок: у него есть запись о проверенной загрузке.
    verified = ""
    try:
        verified = node._verified_local_path(resolved, url, meta_cache)
    except Exception as error:              # noqa: BLE001
        logger.debug("%s verified lookup failed: %s", LOG_PREFIX, error)

    if verified:
        return _found(verified, "ready", name or os.path.basename(verified), resolved)

    # Записи нет — но файл мог быть положен руками, и это тоже «скачан».
    if name:
        final = os.path.join(resolved, name)
        if os.path.isfile(final):
            return _found(final, "ready", name, resolved, unverified=True)
        part = final + ".part"
        if os.path.isfile(part):
            return _found(part, "partial", name, resolved)

    return {"status": "missing", "filename": name, "directory": resolved, "bytes": 0}


def _found(path: str, status: str, name: str, directory: str, *, unverified: bool = False) -> dict:
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    payload = {"status": status, "filename": name, "directory": directory, "bytes": int(size)}
    if unverified:
        # Файл есть, но записи о проверке нет: положили руками или скачали до
        # того, как появились метаданные. Для глаза это «готово», а для отчёта
        # честнее сказать, чем именно это подтверждено.
        payload["unverified"] = True
    return payload


@_register_post("/ts_downloader/status")
async def status_route(request):
    """Статусы строк списка — по одному диску, без сети."""
    from aiohttp import web

    from .ts_downloader import TS_DownloadFilesNode as Node

    try:
        body = await request.json()
    except Exception:                       # noqa: BLE001 - кривое тело
        return web.json_response({"error": "Expected a JSON body."}, status=400)

    text = str(body.get("file_list") or "")
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]

    meta_cache: dict = {}
    results = []
    with _quiet_probe():
        for index, line in enumerate(lines):
            if index >= MAX_LINES:
                break
            if not line or line.startswith("#"):
                # Пустые строки и комментарии тоже возвращаем: фронтенд рисует
                # список по номерам строк, и пропуск сдвинул бы все значки.
                results.append({"line": index, "status": "skip"})
                continue

            parsed = Node._parse_file_list(line)
            if not parsed:
                results.append({"line": index, "status": "unknown", "reason": "parse"})
                continue

            entry = parsed[0]
            state = _entry_status(Node, entry["url"], entry["target_dir"], meta_cache)
            state["line"] = index
            state["url"] = entry["url"]
            results.append(state)

    return web.json_response({"schema": 1, "entries": results})
