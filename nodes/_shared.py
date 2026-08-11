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


def make_route_registrars(prompt_server, warn: Callable[[str], None]):
    """Return ``(register_get, register_post)`` decorators bound to
    ``prompt_server``.

    Each decorator registers ``func`` on the given HTTP method and returns it
    unchanged, and is a no-op passthrough when ``prompt_server`` is None (server
    not ready) — matching the standalone helpers these replace exactly.
    """

    def _make(method_name: str):
        def register(path: str):
            def decorator(func):
                if prompt_server is None:
                    return func
                try:
                    getattr(prompt_server.routes, method_name)(path)(func)
                except Exception as exc:
                    warn(f"Failed to register {method_name.upper()} route '{path}': {exc}")
                return func

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


def media_path_allowed(path, extra_roots=None) -> bool:
    """Можно ли отдать ЭТОТ файл по HTTP. Единая политика пака."""
    import os

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
