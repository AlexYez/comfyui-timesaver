---
name: reference_batch_files_and_wired_path
description: "TS Batch Source «Files in folder» (переименование с псевдонимом) и сокет path у TS Video Loader / TS Audio Loader — как сделано совместимо и чего не трогать"
metadata:
  type: reference
---

TS Batch Source «Files in folder» (переименование с псевдонимом) и сокет path у TS Video Loader / TS Audio Loader — как сделано совместимо и чего не трогать

Исходники темы: [_wired_source.js](../../js/_wired_source.js).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

23.10.2026, просьба владельца: батч по любым файлам, без новой ноды.

**Переименование значения комбо без поломки графов.** «Images in folder» →
«Files in folder». Старое имя убрано из `options`, но работает: ядро НЕ сверяет
комбо со списком, если имя входа явно стоит в сигнатуре `validate_inputs`
(`execution.py`, `validate_function_inputs`). Значит, проверку делаем сами
(`_KNOWN_MODES`) и принимаем псевдоним. ⚠️ Псевдоним сохраняет СТАРЫЙ смысл
(только картинки): иначе в старом батче рядом с кадрами в загрузчик картинок
поехали бы `.txt`-подписи. Пример `09_caption...json` ещё несёт старое имя —
`tools/build_example_workflows.py` поправлен, но без Playwright не пересобран.

**Сокет `path` у загрузок:** `IO.String.Input(optional=True, force_input=True)`
в КОНЦЕ — без виджета, места в `widgets_values` не занимает. ⚠️ Ядро передаёт в
`fingerprint_inputs` ВСЕ входы (`get_input_data` без кэша): у TS Audio Loader
сигнатура явная, и без `path` вызов падал бы → кэш = NaN → пересчёт всегда.
Провод на этапе `validate_inputs` = None → возвращать True. Подрезка/обрезка
с таймлайна для файла с провода НЕ применяется (ставилась на другом файле).
Имя файла уходит в ui `ts_wired_path`; фронтенд — общий `js/_wired_source.js`.

**file_types:** All / Images / Videos / Audio — списки = то, что открывают сами
загрузки; совпадение стережёт `tests/test_batch_source_files.py` (читает
константы загрузок через AST, без импорта).

Связано: [reference_batch_semantics](reference_batch_semantics.md), [reference_image_prompt_extraction](reference_image_prompt_extraction.md).
