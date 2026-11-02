# Runtime и регистрация

Корневой загрузчик отделяет обнаружение, импорт, сбор схем и исполнение конкретной ноды.

## Вход пака

[__init__.py](../../__init__.py) задаёт WEB_DIRECTORY="./js" и строит список
модулей через _discover_module_entries. Рекурсивный rglob по nodes/ сортируется;
пропускаются __init__.py, файлы без ts_-префикса и любой путь с компонентом,
начинающимся на `_`. Поэтому public entry живёт в ts_*.py, а helpers и vendor
код не регистрируются по случайному наличию класса.

Импорт по файловому пути без package name включает _STANDALONE_IMPORT:
реальный loop импорта/регистрации/отчёта не исполняется. Такой static/stub import
не равен успешной загрузке через ComfyUI.

_load_module импортирует модуль и собирает NODE_CLASS_MAPPINGS. Пустой словарь
даёт ноль нод, но import-side-effect уже выполнен — так работают
[инжектор samplers](../../nodes/utils/ts_samplers_injector.py) и
[инжектор schedulers](../../nodes/utils/ts_schedulers_injector.py).
Они регистрируют имена в существующих списках ComfyUI идемпотентно и не заменяют
чужой уже зарегистрированный обработчик. Строки res_2s, beta57 и bong_tangent
сами по себе не node_id.

## V3 наружу, mappings внутрь

В каждом public module mappings остаются источником классов. Когда доступен
ComfyExtension, корневой модуль снимает список _NODE_CLASSES, определяет
TimesaverExtension/comfy_entrypoint и **удаляет корневые mappings из наружного
модуля**. Иначе загрузчик ComfyUI выбрал бы старую ветку и не дошёл до V3
entrypoint. На старом host без ComfyExtension остаётся путь mappings;
это не означает наличие V1-схем в паке.

TimesaverExtension.get_node_list проверяет GET_SCHEMA каждой ноды отдельно.
Схема, читающая диск, может упасть; тогда исключается одна нода с логом,
а не весь последующий список. Для диагностики спрашивают зарегистрированные
node_id или /object_info, а не считают успешные файлы import-а.

## Ошибки и диагностика импорта

Чужая отсутствующая зависимость с распознанным именем → SKIPPED.
Сломанный внутренний import или ImportError без основания считать его
недостающим optional-пакетом → ERROR. Этот разбор важен: прежний безусловный
SKIPPED скрывал опечатку внутри пакета как нормальную нехватку зависимости.
Остальные исключения импорта также ERROR с traceback.

_scan_external_imports использует AST, а не поиск строк import: текст docstring
больше не превращается в ложную зависимость, а вложенные lazy imports видны.
Это аудит наличия import root, не доказательство успешного inference или
совместимости GPU runtime. Обычный успешный старт печатает короткую строку;
TS_VERBOSE_STARTUP раскрывает полные таблицы. Как читать live-лог —
[диагностика](../operations/troubleshooting.md).

## Guard исполнения

[TSDependencyManager](../../ts_dependency_manager.py) оборачивает classmethod
execute зарегистрированных классов один раз. Для async execute wrapper также
async: синхронная обёртка вернула бы ядру неawaited coroutine. Guard ловит Exception,
сохраняет traceback в логе и бросает нормализованный RuntimeError с TS-префиксом
и install hint для отсутствующей зависимости. Он не изобретает пустой IMAGE
или другой typed fallback. InterruptProcessingException может наследовать
BaseException и должна пройти наружу после cleanup.

import_optional хранит успешный модуль или None на время процесса. Установить
пакет в работающий Python и ждать повторного import на следующем вызове нельзя;
потребуется перезапуск по правилам владельца. Политика зависимостей —
[backend](../engineering/backend.md), установка — [development](../operations/development.md).

## Уже существующие адаптации host

_apply_core_patches вызывается при package startup и на on_load. Каждый адаптер
изолирован своим try; неудача не должна лишить пак собственных нод.

- [Pasted media fix](../../ts_pasted_media_fix.py): LoadImage видит подпапки input,
  включая pasted, с откатом на исходную схему и выключателем.
- [Proactor guard](../../compat/proactor_guard.py): обработка Windows asyncio
  transport при закрытии соединения. Код не меняет Python на диске и не
  переключает весь event loop на Selector ради тишины в логах.

Повторная установка идемпотентна. Наличие этих двух адаптеров не разрешает
добавлять произвольные monkey-patches; новые изменения чужого runtime требуют
отдельной задачи. Флаги и расположение логов — в [operations](../operations/README.md).
