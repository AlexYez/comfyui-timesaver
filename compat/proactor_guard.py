"""Погасить ложную ConnectionResetError (WinError 10054) из asyncio на Windows.

ЧТО ВИДНО БЕЗ ЭТОГО. Консоль ComfyUI на Windows забита трассировками::

    [ERROR] Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)
    ConnectionResetError: [WinError 10054] An existing connection was forcibly
    closed by the remote host

Замерено на живой сборке: 355 строк из 2852 в логе сессии — 12%. Приходит
пачками по 3-6 штук в секунду (закрытие websocket'а после задачи, перезагрузка
страницы), а не равномерно.

ПРИЧИНА — БАГ CPYTHON, НЕ ComfyUI. В `Lib/asyncio/proactor_events.py`::

    finally:
        if hasattr(self._sock, 'shutdown') and self._sock.fileno() != -1:
            self._sock.shutdown(socket.SHUT_RDWR)   # ← бросает 10054
        self._sock.close()                          # ← НЕ ВЫПОЛНЯЕТСЯ
        self._sock = None
        ...
        self._called_connection_lost = True

Ни `shutdown`, ни `close` не защищены — ни в 3.12.9 (проверено дизассемблированием:
в теле метода нет ссылки на `ConnectionResetError`), ни в ветке `main` CPython.
python/cpython#83191 открыт с 2020 года, свежее подтверждение — #149388.
Обновление Python НЕ помогает.

⚠️ ЭТО НЕ ТОЛЬКО ШУМ. Исключение вылетает ДО `self._sock.close()`, поэтому сокет
остаётся незакрытым и освобождается только сборщиком мусора; не выполняются и
`self._sock = None`, и `_called_connection_lost = True`. Генерации это не ломает,
но «безвредно» — неверное слово.

ЧЕГО ЗДЕСЬ НЕТ И ПОЧЕМУ:

* НЕ переключаем цикл на `WindowsSelectorEventLoopPolicy` — совет, который
  попадается в половине ответов в интернете, и он вредный: SelectorEventLoop на
  Windows не умеет подпроцессы и упирается в 512 сокетов из-за `select()`. Ради
  тишины в логе получить настоящие поломки — плохой размен.
* НЕ правим файлы в `python/`: патч стдлиба на диске снесёт любое обновление, а
  переносимость сборки сломается сразу.
* НЕ копируем тело метода, а ОБОРАЧИВАЕМ его: когда CPython починит, `except`
  просто перестанет срабатывать. Копия заморозила бы реализацию 2026 года.
* НЕ трогаем `server._detach`: у него разная сигнатура (в 3.12 без аргументов, в
  `main` — `_detach(self)`), а влияет он только на счётчик соединений для
  `wait_closed()`, который ComfyUI штатно не зовёт.

Выключатель: `TS_DISABLE_PROACTOR_GUARD=1` в окружении — как у заплатки на
LoadImage (`ts_pasted_media_fix.py`).
"""

from __future__ import annotations

import logging
import os
import sys

LOGGER = logging.getLogger("comfyui_timesaver.proactor_guard")
LOG_PREFIX = "[TS ProactorGuard]"

DISABLE_ENV = "TS_DISABLE_PROACTOR_GUARD"

# ⚠️ Пак латает дважды: на импорте и в `on_load` — так задумано (см.
# `_apply_core_patches`). Установка от этого не страдает, она идемпотентна, а
# вот сообщение о выключении печаталось два раза. Лог этой заплатки как раз про
# то, чтобы в консоли не было лишнего — начинать с себя.
_ANNOUNCED = False

# Коды, при которых уборку доделываем сами.
#
# ⚠️ Список узкий НАМЕРЕННО. Через тот же метод проходит
# `self._protocol.connection_lost(exc)` — то есть код приложения. Широкий
# `except OSError: pass` глушил бы настоящие ошибки протокола, и они исчезали бы
# бесследно. Незнакомый код — пробрасываем.
BENIGN_WINERRORS = frozenset({
    10054,  # WSAECONNRESET — соединение сброшено клиентом
    10053,  # WSAECONNABORTED
    10058,  # WSAESHUTDOWN
    64,     # ERROR_NETNAME_DELETED — упомянут в комментарии самого CPython
    995,    # ERROR_OPERATION_ABORTED
    6,      # ERROR_INVALID_HANDLE — случай cpython#149388
})


def install() -> bool:
    """Поставить обёртку. Возвращает True, только если поставили именно сейчас.

    Идемпотентно: пак может быть импортирован повторно (reload, две копии в
    `custom_nodes`), а двойная обёртка недопустима.
    """
    if sys.platform != "win32":
        # На Linux и macOS `_ProactorBasePipeTransport` вообще нет — импорт упал бы.
        return False
    if os.environ.get(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        global _ANNOUNCED
        if not _ANNOUNCED:
            _ANNOUNCED = True
            LOGGER.info("%s disabled via %s", LOG_PREFIX, DISABLE_ENV)
        return False

    try:
        from asyncio import proactor_events
    except Exception as error:  # pragma: no cover - сборка без asyncio.proactor
        LOGGER.debug("%s asyncio.proactor_events unavailable: %s", LOG_PREFIX, error)
        return False

    transport = getattr(proactor_events, "_ProactorBasePipeTransport", None)
    if transport is None:  # pragma: no cover - будущая перестройка asyncio
        return False

    original = getattr(transport, "_call_connection_lost", None)
    if original is None or getattr(original, "_ts_guarded", False):
        return False

    def _call_connection_lost(self, exc):
        try:
            original(self, exc)
        except OSError as error:
            if getattr(error, "winerror", None) not in BENIGN_WINERRORS:
                raise
            # Оригинал упал в `finally` на `shutdown`, не дойдя до уборки.
            # Доводим её до конца: иначе сокет живёт до сборщика мусора.
            sock = getattr(self, "_sock", None)
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
                self._sock = None
            self._called_connection_lost = True

    _call_connection_lost._ts_guarded = True
    transport._call_connection_lost = _call_connection_lost
    LOGGER.info("%s installed: asyncio socket teardown is finished on WinError "
                "10054 and friends", LOG_PREFIX)
    return True
