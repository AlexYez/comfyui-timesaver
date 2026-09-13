"""Pack-level shared helpers used by multiple subpackages.

Private module: not registered as a public node by the loader (the
underscore prefix is honored by `_discover_module_entries` in __init__.py).
"""

import logging
from typing import Callable

_logger = logging.getLogger("comfyui_timesaver.ts_shared")


class TS_Logger:
    """Thin facade over stdlib logging for slider/switch/math/animation_preview nodes."""

    @staticmethod
    def log(node_name: str, message: str) -> None:
        _logger.info("[TS %s] %s", node_name, message)

    @staticmethod
    def warn(node_name: str, message: str) -> None:
        _logger.warning("[TS %s] %s", node_name, message)

    @staticmethod
    def error(node_name: str, message: str) -> None:
        _logger.error("[TS %s] %s", node_name, message)


# ---------------------------------------------------------------------------
# Токены даты в пути сохранения
# ---------------------------------------------------------------------------
#
# ⚠️ ЯДРО ЭТОГО НЕ ДЕЛАЕТ. Подсказки ComfyUI обещают `%date:yyyy-MM-dd%` у
# каждой сохраняющей ноды, но `folder_paths.get_save_image_path` знает только
# `%year%`, `%month%`, `%day%`, `%hour%`, `%minute%`, `%second%`, `%width%` и
# `%height%`. Форму с двоеточием подставляет ФРОНТЕНД — перед отправкой графа,
# и только своим нодам. Проверено на живом сервере: тот же префикс, посланный
# в родную `SaveImage` через API, доезжает сырым и падает с
# `OSError: Invalid argument`, потому что двоеточие в имени файла Windows не
# принимает.
#
# Отсюда две вещи. Первая: нашей ноде фронтенд токены не разворачивает, и без
# этой функции человек получал в имени файла буквальное `%date:yyyy-MM-dd%`.
# Вторая: делать надо на сервере — тогда работает и из интерфейса, и из API, и
# из чужого скрипта, а не только там, где повезло.
#
# Формат снят с самого фронтенда (замерено, а не угадано):
#
#     yyyy -> 2026   yy -> 26     MM -> 08   M -> 8
#     dd   -> 12     d  -> 12     hh -> 13   h -> 13   (часы 24-часовые)
#     mm   -> 26     m  -> 26     ss -> 55   s -> 55
#
# Одна буква — без ведущего нуля, две — с ним; пара `M`/`MM` это показывает
# прямо, остальные ведут себя так же.

_DATE_TOKEN = None


def expand_date_tokens(text: str, when=None) -> str:
    """Развернуть `%date:ФОРМАТ%` в готовую строку.

    Всё остальное остаётся нетронутым: `%width%` и прочие токены ядра
    разворачивает сам `folder_paths.get_save_image_path` дальше по пути, и
    съедать их здесь нельзя.

    Args:
        text: префикс имени файла, как его написал человек.
        when: момент времени (для тестов); по умолчанию — сейчас.

    Returns:
        Строку с раскрытыми токенами даты.
    """
    import re
    import time as _time

    global _DATE_TOKEN
    if _DATE_TOKEN is None:
        _DATE_TOKEN = re.compile(r"%date:([^%]*)%", re.IGNORECASE)

    raw = str(text or "")
    if "%date:" not in raw.lower():
        return raw

    moment = when or _time.localtime()
    # Порядок важен: длинные подстановки идут первыми, иначе `yyyy` съедается
    # правилом для `yy` и год превращается в «2626».
    pieces = (
        ("yyyy", f"{moment.tm_year:04d}"),
        ("yy", f"{moment.tm_year % 100:02d}"),
        ("MM", f"{moment.tm_mon:02d}"),
        ("M", str(moment.tm_mon)),
        ("dd", f"{moment.tm_mday:02d}"),
        ("d", str(moment.tm_mday)),
        ("hh", f"{moment.tm_hour:02d}"),
        ("h", str(moment.tm_hour)),
        ("mm", f"{moment.tm_min:02d}"),
        ("m", str(moment.tm_min)),
        ("ss", f"{moment.tm_sec:02d}"),
        ("s", str(moment.tm_sec)),
    )

    def render(match: "re.Match") -> str:
        pattern = match.group(1)
        out = []
        index = 0
        while index < len(pattern):
            for token, value in pieces:
                if pattern.startswith(token, index):
                    out.append(value)
                    index += len(token)
                    break
            else:
                # Не токен — символ разделителя, он идёт как есть.
                out.append(pattern[index])
                index += 1
        return "".join(out)

    return _DATE_TOKEN.sub(render, raw)


# ---------------------------------------------------------------------------
# Отмена прогона
# ---------------------------------------------------------------------------
#
# ⚠️ ComfyUI прерывает выполнение МЕЖДУ нодами само. Внутри ноды это работает
# только там, где её об этом спросили. Из всего пака спрашивали три места, а
# прогресс-бар рисовали десяток: человек жал Cancel на интерполяции пятисот
# кадров и ждал до конца, потому что `ProgressBar.update_absolute` прерывание
# не бросает — он только рисует.
#
# Правило: где есть ProgressBar, там же должен быть и этот вызов.


def raise_if_interrupted() -> None:
    """Прервать работу, если человек нажал Cancel.

    Бросает ``comfy.model_management.InterruptProcessingException`` — она
    наследует ``BaseException`` именно затем, чтобы её не проглотил случайный
    ``except Exception`` по дороге. Вне ComfyUI (в тестах) не делает ничего.
    """
    try:
        import comfy.model_management as mm
    except Exception:                       # noqa: BLE001 - вне ComfyUI
        return
    check = getattr(mm, "throw_exception_if_processing_interrupted", None)
    if callable(check):
        check()


# ---------------------------------------------------------------------------
# aiohttp route registration (shared by the route-owning helper modules)
# ---------------------------------------------------------------------------
# TS_LamaCleanup, TS_SAM_MediaLoader, TS_AudioLoader, TS_SuperPrompt and the
# text nodes each register aiohttp routes on the ComfyUI PromptServer with the
# exact same boilerplate. These two helpers own that boilerplate in one place;
# each caller keeps its own module-level registrar names and warning logger.


def resolve_prompt_server(warn: Callable[[str], None]):
    """Return ``server.PromptServer.instance`` or None (HTTP routes disabled).

    ``warn`` is the caller's own warning logger, so each module's log prefix is
    preserved. ``server`` is imported lazily so this module stays cheap to
    import and free of a hard dependency on the ComfyUI runtime being ready.
    """
    try:
        import server
    except Exception:
        server = None
    if server is None:
        warn("PromptServer unavailable. HTTP routes disabled.")
        return None
    try:
        return server.PromptServer.instance
    except Exception as exc:
        warn(f"PromptServer init failed. HTTP routes disabled: {exc}")
        return None


def guard_cross_site(handler, path: str = ""):
    """Обернуть обработчик отказом для запросов с ЧУЖОЙ страницы.

    ⚠️ Почему это нужно каждому маршруту, а не только тем, что отдают файлы.
    ComfyUI слушает петлю без пароля, и запрос на `127.0.0.1:8188` отправляет
    любая страница, открытая в браузере, пока человек читает новости. Ответ
    она не прочитает — это запретит сам браузер, — но ДЕЙСТВИЕ произойдёт:
    начнётся скачивание на десятки гигабайт, в память уедет модель на 3 ГБ,
    сотрутся рабочие файлы сессии. Реестр называет это
    ``UNAUTHENTICATED_SIDE_EFFECT`` и банил пак за это ещё в 9.31.0, перечислив
    маршруты поимённо.

    Проверка живёт ЗДЕСЬ, в общем регистраторе, ровно потому, что в новом
    маршруте её забудут — и забывали.

    Args:
        handler: асинхронный обработчик aiohttp.
        path: адрес маршрута — только для сообщения в журнал.

    Returns:
        Обработчик, который отвечает 403 на межсайтовый запрос.
    """
    import functools

    @functools.wraps(handler)
    async def _guarded(request, *args, **kwargs):
        if not request_is_same_origin(request):
            _logger.warning(
                "[TS Routes] refused a cross-site request to '%s'.", path or handler.__name__)
            try:
                from aiohttp import web

                return web.json_response(
                    {"error": "This route only answers the ComfyUI page itself."},
                    status=403,
                )
            except ImportError:             # pragma: no cover - вне ComfyUI
                raise PermissionError("cross-site request refused") from None
        return await handler(request, *args, **kwargs)

    return _guarded


def make_route_registrars(prompt_server, warn: Callable[[str], None],
                          *, same_origin_only: bool = True):
    """Return ``(register_get, register_post)`` decorators bound to
    ``prompt_server``.

    Каждый декоратор регистрирует обработчик на своём методе и ВОЗВРАЩАЕТ его
    обёрнутым в ``guard_cross_site`` — тем же объектом, что зарегистрирован,
    чтобы вызов из теста судился по тем же правилам, что и живой запрос.

    Args:
        prompt_server: экземпляр PromptServer, либо None (сервера ещё нет).
        warn: куда сказать, что регистрация не удалась.
        same_origin_only: оставлено выключателем на случай маршрута, который
            обязан отвечать кому угодно. Такого в паке нет, и заводить его
            без явной причины не стоит.
    """

    def _make(method_name: str):
        def register(path: str):
            def decorator(func):
                handler = guard_cross_site(func, path) if same_origin_only else func
                if prompt_server is None:
                    return handler
                try:
                    getattr(prompt_server.routes, method_name)(path)(handler)
                except Exception as exc:
                    warn(f"Failed to register {method_name.upper()} route '{path}': {exc}")
                return handler

            return decorator

        return register

    return _make("get"), _make("post")


# ---------------------------------------------------------------------------
# Что маршруты пака вправе показывать браузеру
# ---------------------------------------------------------------------------
#
# ⚠️ ОДНО правило на весь пак: видео, аудио и загрузчик кадров вели себя
# по-разному, и человек, положивший материал в «Документы», получал превью в
# одной ноде и пустоту в другой. Смысл произвольного пути в том, чтобы НЕ
# копировать часовые исходники в `input`: эта папка иначе растёт без конца.
#
# Черта проходит по тому, кому виден сервер:
#
#   * ComfyUI слушает петлю (обычная локальная работа) — путь любой. «Всякий,
#     кто дотянулся до порта» здесь и есть хозяин машины.
#   * ComfyUI открыт наружу (`--listen 0.0.0.0`, машина в локалке, облако) —
#     только домашняя папка пользователя, каталоги самого ComfyUI и то, что
#     хозяин добавил в `TS_MEDIA_EXTRA_ROOTS`. Иначе знание имени файла
#     означало бы возможность скачать его с чужой машины.
#
# Переменные ставит ХОЗЯИН МАШИНЫ, снаружи workflow: workflow приезжает от кого
# угодно, а переменная окружения — нет.

_MEDIA_EXTRA_ROOTS_ENV = "TS_MEDIA_EXTRA_ROOTS"
_MEDIA_ANY_PATH_ENV = "TS_MEDIA_ALLOW_ANY_PATH"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"}


def server_is_local_only() -> bool:
    """Слушает ли ComfyUI только петлю.

    Не смогли выяснить — считаем, что открыт: осторожная сторона дешевле.
    """
    try:
        from comfy.cli_args import args

        listen = str(getattr(args, "listen", "") or "").strip()
    except Exception as error:              # noqa: BLE001 - вне ComfyUI и в тестах
        _logger.debug("[TS Media] could not read the listen address: %s", error)
        return False
    if not listen:
        return False
    hosts = {part.strip().lower() for part in listen.split(",") if part.strip()}
    return bool(hosts) and hosts <= _LOOPBACK_HOSTS


def media_roots(extra: "list[str] | tuple[str, ...] | None" = None) -> list[str]:
    """Каталоги, открытые маршрутам на ОТКРЫТОМ наружу сервере.

    Args:
        extra: собственные папки вызывающего (записи, кэш) — они всегда свои.
    """
    import os

    roots: list[str] = []
    try:
        import folder_paths

        for getter in ("get_input_directory", "get_output_directory", "get_temp_directory"):
            fn = getattr(folder_paths, getter, None)
            if callable(fn):
                roots.append(os.path.abspath(fn()))
        base = getattr(folder_paths, "base_path", None)
        if base:
            roots.append(os.path.abspath(os.path.join(base, "user")))
    except Exception as error:              # noqa: BLE001 - вне ComfyUI и в тестах
        _logger.debug("[TS Media] folder_paths unavailable: %s", error)

    try:
        home = os.path.abspath(os.path.expanduser("~"))
        # Пустой или корневой «дом» (бывает у служб) открыл бы весь диск.
        if home and os.path.dirname(home) != home:
            roots.append(home)
    except Exception as error:              # noqa: BLE001 - экзотическое окружение
        _logger.debug("[TS Media] home directory unavailable: %s", error)

    for chunk in str(os.environ.get(_MEDIA_EXTRA_ROOTS_ENV, "")).split(os.pathsep):
        chunk = chunk.strip().strip('"')
        if chunk:
            roots.append(os.path.abspath(os.path.expanduser(chunk)))

    for chunk in extra or ():
        if chunk:
            roots.append(os.path.abspath(str(chunk)))
    return [r for r in roots if r]


# ⚠️ Второе правило поверх корней: ЧТО именно отдаётся. Корни отвечают на
# вопрос «откуда», и на локальной петле ответ был «откуда угодно» — то есть
# маршрут превращался в чтение любого файла на диске по имени. Между тем ни
# одному плееру пака не нужен файл, который не является медиа: список ниже —
# объединение того, что читают видео-, аудио- и картиночные ноды. Так
# произвольный путь остаётся (ради него всё и заведено — часовые исходники не
# таскают в `input`), а `~/.ssh/id_rsa` и `cookies.sqlite` перестают быть
# достижимыми.
_MEDIA_EXTENSIONS = frozenset({
    # видео
    ".avi", ".flv", ".m2ts", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg",
    ".mts", ".ts", ".webm", ".ogv", ".wmv", ".mxf", ".y4m",
    # аудио
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".oga", ".ogg", ".opus",
    ".wav", ".wma",
    # изображения и кадры
    ".apng", ".bmp", ".exr", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff",
    ".webp",
    # субтитры: их тоже читает таймлайн
    ".srt", ".vtt",
})


def media_extension_allowed(path) -> bool:
    """Похоже ли это на медиафайл — по расширению.

    Отдельная функция, а не строка внутри ``media_path_allowed``: тесты
    проверяют оба правила по отдельности, и читать их порознь проще.
    """
    import os

    return os.path.splitext(str(path))[1].lower() in _MEDIA_EXTENSIONS


def media_path_allowed(path, extra_roots=None) -> bool:
    """Можно ли отдать ЭТОТ файл по HTTP. Единая политика пака.

    Два условия, и второе не отменяется ничем: файл обязан быть медиа. Корни
    (откуда) настраиваются переменными окружения, расширение (что) — нет.
    """
    import os

    if not media_extension_allowed(path):
        return False
    if str(os.environ.get(_MEDIA_ANY_PATH_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if server_is_local_only():
        return True
    try:
        target = os.path.abspath(str(path))
    except Exception:                       # noqa: BLE001 - мусор вместо пути
        return False
    for root in media_roots(extra_roots):
        try:
            if os.path.commonpath([target, root]) == root:
                return True
        except ValueError:
            continue                        # другой диск — просто не тот корень
    return False


# ---------------------------------------------------------------------------
# Кто прислал запрос: своя страница или чужая вкладка
# ---------------------------------------------------------------------------
#
# ⚠️ У ComfyUI нет защиты от подделки запроса: он слушает петлю без пароля, и
# POST (или <img src=...>) на `127.0.0.1:8188` отправляет ЛЮБАЯ открытая в
# браузере страница, пока человек читает новости. Поэтому «сервер локальный,
# значит по ту сторону хозяин машины» — неверно, и именно на этом допущении
# строилась политика путей выше.
#
# Отличить свою страницу от чужой можно по заголовкам, которые браузер ставит
# сам и которые страница подделать не может:
#
#   * `Sec-Fetch-Site` — современные браузеры шлют его всегда. `same-origin` и
#     `same-site` — свои; `none` — человек сам открыл адрес; `cross-site` —
#     чужая вкладка.
#   * `Origin` / `Referer` — запасной путь для старых браузеров.
#
# Ничего не пришло вовсе — это не браузер (curl, плеер, скрипт), и запрет тут
# не поможет: разрешаем, потому что иначе сломались бы законные сценарии, а
# защиты это не добавит.

_SAME_SITE_FETCH = {"same-origin", "same-site", "none"}

#: Выключатель проверки. Заведён не «на всякий случай»: ComfyUI ставят и в
#: обёртках, где страница живёт на другом адресе, чем сервер (Desktop, чужой
#: прокси, встраивание в свою панель). Там запрос честно межсайтовый, и
#: человеку нужен способ сказать «я знаю, это мой интерфейс», не правя код.
_ORIGIN_GUARD_OFF_ENV = "TS_DISABLE_ORIGIN_GUARD"


def request_is_same_origin(request) -> bool:
    """Пришёл ли запрос со страницы самого ComfyUI."""
    import os

    if str(os.environ.get(_ORIGIN_GUARD_OFF_ENV, "")).strip().lower() in {
            "1", "true", "yes", "on"}:
        return True
    headers = getattr(request, "headers", None) or {}
    fetch_site = str(headers.get("Sec-Fetch-Site", "") or "").strip().lower()
    if fetch_site:
        return fetch_site in _SAME_SITE_FETCH

    host = str(headers.get("Host", "") or "").strip().lower()
    for name in ("Origin", "Referer"):
        raw = str(headers.get(name, "") or "").strip()
        if not raw:
            continue
        try:
            from urllib.parse import urlsplit

            netloc = urlsplit(raw).netloc.lower()
        except Exception:                   # noqa: BLE001 - мусор вместо заголовка
            return False
        # Пустой Host сравнивать не с чем — считаем чужим: перепутать в эту
        # сторону дешевле.
        return bool(netloc) and netloc == host
    return True


# ---------------------------------------------------------------------------
# Куда паку разрешено ходить по запросу ИЗ БРАУЗЕРА
# ---------------------------------------------------------------------------
#
# ⚠️ Это про SSRF, и он не теоретический: у загрузчика есть маршруты, которым
# адрес называет клиент. Без проверки чужая вкладка заставляла бы ComfyUI
# стучаться на `127.0.0.1:<любой порт>`, в домашнюю сеть (`192.168.*`) или на
# `169.254.169.254` — служебный адрес облачных машин, отдающий ключи доступа.
# Сам ComfyUI при этом внутри сети, а браузер атакующего — нет: в этом вся
# ценность такого запроса для него.
#
# ⚠️ Прогон ГРАФА эта проверка не касается: адрес там пишет тот, кто сидит за
# машиной, и NAS в локальной сети — законный источник моделей. Ограничение
# живёт ровно там, где адрес приезжает по сети.

def url_target_is_public(url: str, *, via_proxy: bool = False) -> "tuple[bool, str]":
    """Ведёт ли адрес наружу, а не внутрь этой машины или её сети.

    Args:
        url: адрес целиком, как его прислал клиент.
        via_proxy: запрос пойдёт через прокси — тогда имя разрешает прокси, и
            наш собственный резолв ничего не доказывает. Проверяется только
            то, что хост не назван приватным адресом напрямую.

    Returns:
        ``(можно, причина отказа)``. Причина пуста, когда можно.
    """
    import ipaddress
    import socket
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(str(url or "").strip())
    except Exception:                       # noqa: BLE001 - мусор вместо адреса
        return False, "the address cannot be parsed"
    if parts.scheme.lower() not in {"http", "https"}:
        return False, f"scheme '{parts.scheme}' is not allowed (http/https only)"
    host = (parts.hostname or "").strip()
    if not host:
        return False, "the address has no host"

    def _verdict(raw: str) -> "tuple[bool, str]":
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            return True, ""                 # не IP — решает вызывающий
        if (address.is_private or address.is_loopback or address.is_link_local
                or address.is_reserved or address.is_multicast
                or address.is_unspecified):
            return False, f"{raw} is not a public address"
        return True, ""

    literal_ok, literal_reason = _verdict(host.strip("[]"))
    if not literal_ok:
        return False, literal_reason
    if via_proxy:
        return True, ""

    port = parts.port or (443 if parts.scheme.lower() == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as error:
        return False, f"'{host}' cannot be resolved ({error})"
    for info in infos:
        resolved = str(info[4][0])
        ok, reason = _verdict(resolved)
        if not ok:
            return False, f"'{host}' resolves to {reason}"
    return True, ""
