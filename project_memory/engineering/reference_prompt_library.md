---
name: reference_prompt_library
description: "TS Prompt Library — библиотека промптов на данных (раздел/модель/collection.json), поля подстановки, Qwen Image 2.1 дословно из авторского MD, FLUX.2 Klein 9B по правилам BFL"
metadata:
  type: reference
---

TS Prompt Library — библиотека промптов на данных (раздел/модель/collection.json), поля подстановки, Qwen Image 2.1 дословно из авторского MD, FLUX.2 Klein 9B по правилам BFL

Исходники темы: [_prompt_library.py](../../nodes/text/_prompt_library.py), [ts_prompt_library.py](../../nodes/text/ts_prompt_library.py), [ts-prompt-library.js](../../js/text/ts-prompt-library.js), [_prompt_library.js](../../js/text/_prompt_library.js).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

# TS Prompt Library (27.09.2026)

Просьба владельца: библиотека промптов, выбор пресета, красивый показ, кнопка
«Копировать», выход STRING, данные в JSON отдельной папкой, задел на превью и
новые разделы (видео и т.д.). Первый раздел — контекстное редактирование для
Qwen Image 2.1 и FLUX.2 Klein 9B.

## Устройство

* Данные: `nodes/text/prompt_library/<раздел>/<коллекция>/collection.json`,
  раздел = папка с `section.json`. Превью — `previews/` внутри коллекции.
  Формат описан в [README библиотеки](../../nodes/text/prompt_library/README.md). Новая модель/раздел = папка.
* `nodes/text/_prompt_library.py` — поиск, проверка, `fill()`, кэш по mtime
  всех JSON, маршруты `GET /ts_prompt_library/catalog` и `/preview?key=`.
* `nodes/text/ts_prompt_library.py` — нода `TS_PromptLibrary`: `preset`
  (Combo ключей, опции строятся из библиотеки) + `fields` (JSON значений полей,
  socketless). Выход `prompt`.
* `js/text/ts-prompt-library.js` + чистый `js/text/_prompt_library.js`.
  Оба штатных виджета спрятаны `hideWidget` — граф работает и без JS.

⚠️ Ключ пресета `раздел/коллекция/ID` живёт в сохранённых графах — папки и id
после публикации не переименовывать.

⚠️ Подстановка в Python и JS обязана совпадать (кнопка «Копировать» = выход
ноды): один проход по исходному тексту, пустое поле остаётся в скобках, bool
не значение (в Python bool — подкласс int). Гард —
`test_copy_and_output_are_the_same_text`.

⚠️ Кривой пресет НЕ подключается (необъявленное поле, пустой промпт, дубль id,
inputs вне 0..8, schema≠1): иначе в модель ушло бы буквальное `{TARGET}`.
Превью — только внутри папки коллекции и только картинка.

⚠️ Полноэкранный режим (27.09.2026): общая `createOpenInterfaceButton` +
`openFullscreenOverlay`; виджет держит ОБОЛОЧКУ `ts-plib-shell`, в окно
переезжает наполнение `root` (приём TS Song Creator), закрытие возвращает его
`shell.appendChild(root)`. Во весь экран видна колонка `.ts-plib__side` —
список модели с поиском (`filterGroups`: код/название/описание на всех языках +
текст промпта, все слова) и миниатюрами. Список рисуется ТОЛЬКО при открытом
окне — компактная нода превью-миниатюры не грузит. ↑/↓ листают (пока фокус не
в поле ввода — это решает общий `onKey`).

⚠️ «Окно открыто» — это класс `is-fullscreen` на `root`, а НЕ
`state.fullscreen?.isOpen()`: `onOpen` вызывается ИЗНУТРИ
`openFullscreenOverlay`, когда её результат ещё не присвоен, и список при
открытии оставался пустым (поймано вживую). Касается любой ноды, где `onOpen`
смотрит на хэндл окна.

⚠️ Новой ноде размер 540×620 ставится явно ДО `addResizableDomWidget` (как у
TS Angle Select): LiteGraph уже посчитал размер по виджетам, хелпер его уважает,
и нода вставала в минимум 400×360.

⚠️ Nodes 2.0: прокрутка карточки вынесена из потока (`position:absolute;
inset:0` внутри `flex:1 1 0` блока) — как у TS Style Prompt, иначе нода
раздувается под высоту содержимого.

## Контент

* Qwen Image 2.1: авторский MD владельца (`Qwen_Image_2_1_Context_Editing_Catalog_RU.md`,
  редакция 27.09.2026) → JSON скриптом, промпты ДОСЛОВНО. 50 записей (34×1
  вход, 16×2), 26 полей. Английские подписи/пояснения/примечания дописаны.
* FLUX.2 Klein 9B: те же 50 кодов, свои промпты по docs.bfl.ml:
  входы словами «image 1/2» (не `<image1>`), до 4 референсов, негативного
  промпта нет → без «Do not/Avoid» (тест это стережёт), явное «Keep …
  unchanged», hex-цвета, текст в кавычках. O12: альфы у Klein нет → белый фон
  + TS Remove Background.
* Коды совпадают между коллекциями — при смене модели интерфейс остаётся на
  той же задаче (`switchKey`).
* `examples` у пресета — свой пример поля там, где общий сбивает (O10: «apples»
  вместо одной кружки).
* Заметка владельцу: в авторском Qwen-каталоге `Add exactly one {OBJECT}` +
  пример с артиклем даёт «exactly one a small …» — в Klein переписано, Qwen не
  трогали (авторский текст).

## 01.10.2026: O13 и поиск в ноде

* **O13** «Удаление фона: главный объект на прозрачном фоне» — без полей,
  объект модель находит сама. Qwen: официальная формула из README
  QwenLM/Qwen-Image-2.1 — «This is an RGBA image with transparency. …
  The image has alpha channel and the background is transparent.». Не из
  авторского MD — сказано в примечании. Klein — белый фон.
* ⚠️ Альфа в ComfyUI 0.38 у Qwen 2.1 — нативная: VAE (`comfy/sd.py`, «Qwen
  Image 2.1 VAE … RGBA», `output_channels` = 4, `pad_channel_value` = 1 для
  RGB-входа) декодирует RGBA, штатный SaveImage пишет 4-канальный PNG. Load
  Image альфу отрывает в MASK (=1−alpha) → обратно Join Image with Alpha.
  TextEncodeQwenImage21 сам кладёт альфу на белый для vision-башни, VAE
  получает все 4 канала.
* **Поиск** — одна строка `searchRow`, переезжает: в ноде над пресетом, во
  весь экран в `.ts-plib__side` (insertBefore на open/close). В ноде при
  запросе найденное рисуется В `scroll` вместо карточки (`state.picking`),
  без миниатюр; ↑/↓ — подсветка, Enter/щелчок — выбор (`choose`), Esc —
  сброс. Во весь экран Esc закрывает окно (общий обработчик в фазе
  перехвата) — так и оставлено. Поиск не смотрит в примечания: там чужие
  коды («возьмите O12»). `neighbourKey(..., query)` листает найденное.
* Пробник с настоящими нажатиями (классика/Vue, ru/en) — scratchpad
  `probe_plib_search.py` сессии 01.10.2026; виджет `preset` спрятан —
  искать через `n._tsHiddenWidgets?.preset`.

Тесты: `tests/test_prompt_library.py` (52).
