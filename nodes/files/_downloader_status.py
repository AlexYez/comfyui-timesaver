"""Что из списка уже лежит на диске: маршрут статусов для ноды.

Три состояния, и они не выдуманы — ровно так их различает сам движок загрузки:

    ready    файл на месте, а рядом ``<file>.tsmeta.json`` подтверждает, что он
             докачан до конца и проверен. Ровно это делает список скачанных
             моделей бесплатным: движок не ходит в сеть за тем, что уже есть.
    partial  на месте только ``<file>.part`` — прерванная докачка. Она не мусор:
             следующий запуск продолжит её Range-запросом, а не начнёт заново.
    missing  ни того, ни другого.

⚠️ «Готово» значит ровно одно: ЗАГРУЗЧИК ИЗ ГРАФА НАЙДЁТ ЭТОТ ФАЙЛ. Поэтому
кроме целевой папки спрашивается сам ComfyUI (`folder_paths.get_full_path`,
тот же вызов, что у загрузчиков): модель в `extra_model_paths.yaml` тоже
«готова», и в ответе появляется ``elsewhere`` — где она лежит. Обратный случай
— ``shadowed``: файл скачан, но загрузчик откроет другой с тем же именем из
папки, которую ComfyUI проверяет раньше.

⚠️ Маршрут НЕ ходит в сеть. Он отвечает по одному диску, поэтому его можно
звать на каждое открытие ноды и после каждой правки списка — иначе список из
двадцати моделей стоил бы двадцати HEAD-запросов на каждый показ.

Есть и четвёртое состояние — ``unknown``: строку не удалось разобрать (нет
папки, кривой адрес). Молчать о ней нельзя: человек видит строку в списке и
вправе знать, что она не будет скачана.
"""

from __future__ import annotations

import logging
import os
import threading
from urllib.parse import unquote

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

    def __enter__(self) -> _quiet_probe:
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
    который здесь зовут с `from_route=True`, то есть строгим разбором для
    сетевых запросов. Разрешать второй раз — это прогонять через проверку
    путей то, что её уже прошло.
    """
    resolved = str(directory or "")
    if not resolved:
        return {"status": "unknown", "reason": "target", "directory": ""}

    # ⚠️ Раскодировать ДО разбора — ровно как движок, который сохраняет файл:
    # для адреса с `%20` проверка искала `my%20model.safetensors`, файл лежал
    # как `my model.safetensors`, и точка оставалась красной навсегда.
    name = node._sanitize_filename(
        os.path.basename(unquote(url.split("?")[0].split("#")[0]).rstrip("/"))
    ) or ""

    # Сначала спрашиваем движок: у него есть запись о проверенной загрузке.
    verified = ""
    try:
        verified = node._verified_local_path(resolved, url, meta_cache)
    except Exception as error:              # noqa: BLE001
        logger.debug("%s verified lookup failed: %s", LOG_PREFIX, error)

    if verified:
        return _shadowed(node, resolved, verified) or _found(
            verified, "ready", name or os.path.basename(verified), resolved)

    # Записи нет — но файл мог быть положен руками, и это тоже «скачан».
    final = os.path.join(resolved, name) if name else ""
    if final and os.path.isfile(final):
        return _shadowed(node, resolved, final) or _found(
            final, "ready", name, resolved, unverified=True)

    # ⚠️ В целевой папке пусто — но загрузчик может найти файл в другой папке
    # той же категории (`models/unet` рядом с `models/diffusion_models`, папки
    # `extra_model_paths.yaml`). Зелёный обязан значить «граф его откроет», и
    # ничего сверх того: файл, лежащий там, куда загрузчик не смотрит, остаётся
    # «не скачан» — это правда.
    #
    # Сначала — запись о НАШЕЙ загрузке в соседней папке, по адресу: только так
    # находится модель, чьё имя дал сервер (ссылка Civitai кончается числом).
    recorded = _ask(node._verified_elsewhere, resolved, url, meta_cache)
    if recorded:
        return _elsewhere(recorded, os.path.basename(recorded), resolved, unverified=False)
    # Потом — по имени, ТЕМ ЖЕ вызовом, которым ищет сам загрузчик.
    loaded = _ask(node._loader_path, resolved, name) if name else ""
    if loaded:
        return _elsewhere(loaded, name, resolved, unverified=True)

    if final and os.path.isfile(final + ".part"):
        return _found(final + ".part", "partial", name, resolved)

    return {"status": "missing", "filename": name, "directory": resolved, "bytes": 0}


def _ask(lookup, *args) -> str:
    """Спросить движок; сбой поиска — это «не нашлось», а не упавший маршрут."""
    try:
        return lookup(*args) or ""
    except Exception as error:              # noqa: BLE001
        logger.debug("%s lookup failed: %s", LOG_PREFIX, error)
        return ""


def _elsewhere(path: str, name: str, directory: str, *, unverified: bool) -> dict:
    """«Скачана», но лежит в другой папке, которую загрузчик тоже читает."""
    payload = _found(path, "ready", name, directory, unverified=unverified)
    payload["elsewhere"] = os.path.dirname(path)
    return payload


def _shadowed(node, directory: str, local: str) -> dict | None:
    """Файл скачан, но загрузчик откроет ДРУГОЙ с тем же именем.

    ComfyUI берёт первую папку категории, где нашлось имя, а `is_default: true`
    в `extra_model_paths.yaml` ставит общую папку впереди `models/`. Тогда
    скачанный файл лежит целым, а граф грузит старый. Зелёная точка в этом
    случае была бы ложью — отдельное состояние, красное.
    """
    try:
        other = node._shadowing_file(directory, local)
    except Exception as error:              # noqa: BLE001
        logger.debug("%s shadow lookup failed: %s", LOG_PREFIX, error)
        return None
    if not other:
        return None
    payload = _found(local, "shadowed", os.path.basename(local), directory)
    payload["loaded"] = other
    return payload


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

            # ⚠️ `from_route=True`: маршрут читает диск по пути из ТЕЛА
            # ЗАПРОСА, а прислать его может любая вкладка браузера. Без
            # строгого разбора этим можно было проверять наличие файла где
            # угодно внутри ComfyUI — тихая, но настоящая разведка. Папка, до
            # которой сетевому запросу дела нет, честно отвечает `unknown`.
            parsed = Node._parse_file_list(line, from_route=True)
            if not parsed:
                results.append({"line": index, "status": "unknown", "reason": "parse"})
                continue

            entry = parsed[0]
            state = _entry_status(Node, entry["url"], entry["target_dir"], meta_cache)
            state["line"] = index
            state["url"] = entry["url"]
            # Под каким именем загрузчики найдут то, что скачает эта строка:
            # `[[категория реестра, подпапка]]`. Сверка с графом сравнивает
            # это с категорией загрузчика — точно, а не по имени папки, которое
            # у корня из `extra_model_paths.yaml` может быть каким угодно.
            state["scopes"] = [list(scope) for scope in
                               (_ask(Node._loader_scopes, entry["target_dir"]) or [])]
            results.append(state)

    return web.json_response({"schema": 1, "entries": results})
