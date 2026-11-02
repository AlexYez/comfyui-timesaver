---
name: reference_optional_xformers_warnings
description: "DINOv2 optional xFormers warnings downgraded to debug; widgetInputs import absent in our pack"
metadata:
  type: reference
---

DINOv2 optional xFormers warnings downgraded to debug; widgetInputs import absent in our pack

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

02.11.2026 владелец прислал жалобы пользователей на deprecated
/extensions/core/widgetInputs.js и два xFormers not available.
В нашем JS (DEV + установленный пак) widgetInputs/старый импорт не найден.
В текущем installed custom_nodes тоже не найден runtime импорт; только исходные
пути TS Nodes Wizard tools/frontend_inventory.py, которые НЕ импортируют JS
и не обращаются к legacy HTTP API. В comfyui_manager JS такого импорта нет.
Middleware ComfyUI server.py deprecation_warning реагирует на GET пути
/extensions/core/* и /scripts/ui* один раз на путь за процесс. Оно не пишет
имя инициатора и срабатывает на HTTP-запрос даже без JS import. Не считать
эту строку доказательством вины нашего пака. Для конкретного пользователя
нужен Network initiator или его установленный список JS-расширений/версий.
Есть upstream пример старого VHS imports (PR666), но VHS на этой установке
не найден; нельзя назначать виновника без проверки.

Наши dinov2_layers/attention.py и block.py действительно писали по WARNING
xFormers not available при ImportError. Это optional ускоритель: MemEffAttention
с обычным Tensor уходит в родительский Attention с SDPA/math, NestedTensorBlock
с Tensor в стандартный Block. Только list/nested tensor путь требует xFormers
и по-прежнему выдаёт AssertionError. Заменили лишь 2 warning на debug с
пояснением fallback, без фильтров логов, установки xFormers или смены математики.
Исходные CRLF из git сохранены; diff --check запускается с cr-at-eol, потому что
vendored файлы уже хранят CRLF в git blob. Узкий diff: по 3 строки на файл.

64 targeted tests прошли: test_video_depth_helpers.py + test_universal_depth.py.
Отдельный CPU smoke с sys.modules[xformers]=None и xformers.ops=None проверил:
нет WARNING, Attention == MemEffAttention, Block == NestedTensorBlock с теми
же весами на Tensor; list input сохраняет зависимость/AssertionError.
compileall helpers прошёл. Другие предупреждения не подавлялись.

Изменены эти два файла и Unreleased в CHANGELOG. DEV/runtime синхронизированы
после проверки runtime == HEAD и backup tests/.cache/quiet-depth-warnings-20261102/runtime-before.zip.
Версия осталась 12.13.4, commit/push НЕ было. Сервер не перезапускался;
он подхватит Python исправление при следующем перезапуске владельца.
ControlNet Aux имеет такую же пару warning в собственной depth_anything_v2:
не трогали; он может продолжать печатать те же сообщения независимо от нас.

## Публикация 12.13.5 — 02.11.2026

По прямой команде владельца сделаны patch и push: коммит
0e1aa5540cb95b8d5d932d4a082ec01d517cd03a в origin/master.
Состав: 2 исправленных DINOv2 файла, CHANGELOG, pyproject и 2 README badges.
64 теста и forced-no-xformers smoke были уже зелёными до релиза;
перед commit повторно прошли compileall helpers и diff --check с cr-at-eol.
Publish action 41418917479 success, включая Publish Custom Node.
https://github.com/AlexYez/comfyui-timesaver/actions/runs/41418917479
Runtime обновлён git pull --ff-only до того же SHA, clean. Все 6 файлов
сверены с DEV с нормализацией EOL. Backup локальных runtime изменений:
tests/.cache/release-12.13.5/runtime-before-pull.zip. Временный stash удалён
после успешной сверки; квитанция runtime-sync.json рядом.
DEV tracked clean, посторонние untracked audit/report и workflow_exports оставлены.
ComfyUI не перезапускался, Python фикс подхватится после перезапуска владельцем.
