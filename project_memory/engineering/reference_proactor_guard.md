---
name: reference_proactor_guard
description: "Заплатка на asyncio Windows — ложная ConnectionResetError 10054 при закрытии websocket"
metadata:
  type: reference
---

Заплатка на asyncio Windows — ложная ConnectionResetError 10054 при закрытии websocket

Исходники темы: [proactor_guard.py](../../compat/proactor_guard.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Сделано 16.08.2026. `compat/proactor_guard.py`, зовётся из `_apply_core_patches`
в корневом `__init__.py` (там же живёт заплатка на LoadImage).

## Что чинится

`_ProactorBasePipeTransport._call_connection_lost` в CPython зовёт
`self._sock.shutdown(...)` в `finally` без защиты. На сокете, который клиент уже
закрыл, это бросает WinError 10054 — и исключение вылетает ДО `self._sock.close()`.
То есть сокет остаётся открытым до сборщика мусора, а в консоль сыплется
трассировка.

Замеры на живой сборке: **402 строки из 1865 (22%)** лога — упоминания
`_call_connection_lost`. Приходят пачками при закрытии websocket'а.

Баг CPython, не ComfyUI: python/cpython#83191 открыт с 2020, свежее
подтверждение — #149388. В ветке `main` код по-прежнему без защиты, обновление
Python НЕ помогает.

## Как сделано и почему именно так

- **Обёртка, а не копия метода.** Починят в CPython — `except` просто перестанет
  срабатывать. Копия заморозила бы реализацию.
- **Фильтр по `winerror`** (10054, 10053, 10058, 64, 995, 6), а не
  `except OSError: pass`: через тот же метод проходит
  `protocol.connection_lost(exc)` — код приложения. Незнакомый код пробрасываем.
- **`server._detach` не трогаем**: разная сигнатура в 3.12 и в `main`, влияет
  только на счётчик для `wait_closed()`, который ComfyUI не зовёт.
- **Гвард `sys.platform != "win32"`** — на Linux/macOS класса нет.
- **Идемпотентность** через `_ts_guarded` на функции: пак импортируется дважды
  (импорт + `on_load`), плюс возможны две копии в `custom_nodes`.
- Выключатель `TS_DISABLE_PROACTOR_GUARD=1`.

⚠️ **НЕ переключать цикл на `WindowsSelectorEventLoopPolicy`** — этот совет
занимает половину ответов в интернете и он вредный: нет подпроцессов, потолок
512 сокетов из-за `select()`.

## Замер A/B (одинаковая нагрузка: 8 перезагрузок страницы + 2 генерации)

| | вхождений `_call_connection_lost` |
|---|---|
| с заплаткой | **0** |
| `TS_DISABLE_PROACTOR_GUARD=1` | 9 |

## Грабли, найденные по дороге

⚠️ **`from asyncio import proactor_events` берёт атрибут пакета, а не
`sys.modules`.** Первая версия теста подставляла фейковый модуль в
`sys.modules["asyncio.proactor_events"]` — и `install()` молча патчил НАСТОЯЩИЙ
asyncio, а проверялся фейк. Подменять надо атрибут
`proactor_events._ProactorBasePipeTransport`.

⚠️ **`tools/verify_live_sync.py` возит только перечисленные папки** (`js`,
`nodes`, `locales`). Новую `compat/` пришлось добавить в список — иначе живая
копия молча пишет «ProactorGuard not installed», а живая проверка проверяет
вчерашнюю сборку.

## Потом

Готовый кандидат в PR к ComfyUI: Comfy-Org/ComfyUI#15163 ровно про это, с тем же
диагнозом. Правка на уровне `main.py` избавила бы от шума всех пользователей
Windows, а не только тех, кто ставит этот пак.

Связано: [reference_pasted_media_fix](reference_pasted_media_fix.md).
