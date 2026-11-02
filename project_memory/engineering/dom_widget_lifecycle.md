---
name: dom_widget_lifecycle
description: "DOM-виджеты: backend-сессии, загрузка файлов, full-bleed, рост ноды и слои интерфейса."
metadata:
  type: reference
---

DOM-виджеты: backend-сессии, загрузка файлов, full-bleed, рост ноды и слои интерфейса.

Исходники темы: [_compare.js](../../js/_studio/_compare.js).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

## 9. Backend: per-session asyncio.Lock + cleanup

Для нод с длительными jobs (inpaint, transcoding, etc.):
- `_session_locks: dict[str, asyncio.Lock]` — lazy-create per session. Сериализует параллельные запросы той же сессии.
- Все temp-файлы версионировать через `{session}_{tag}_{nanos:020d}.png` (никогда не overwrite — нужно для undo/redo).
- Cleanup при `/seed`, `/reset`, при превышении history-limit. Чистить только `name.startswith(f"{safe_session}_")` — защита от удаления чужих файлов.

Сессионные locks "утекают" в dict (один на ноду в графе). Это OK — overhead ничтожный.

## 10. Folder registration для моделей

Создавать подпапку в `models/` и регистрировать через `folder_paths.add_model_folder_path("name", path)` на module-level. Делает её видимой для `extra_model_paths.yaml` overrides.

```python
def _register_model_folder() -> None:
    base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(base))

_register_model_folder()  # на module import
```

## 11. Output organization

Сохранять результаты в подпапку и с тегом в имени файла:
- `output/<feature_name>/<source_stem>_<feature_name>_<timestamp>.png`
- В response `{"subfolder": "<feature_name>", "type": "output"}`.

Группирует output для пользователя, легко найти/убрать.

## 12. CSS layering для full-bleed UI

```
canvas (full-bleed inset:0, z-auto)
empty (matches image padded area, z-auto, pointer-events:none)
overlay (full-bleed, z-auto, hidden default, pointer-events:none)
toolbar (top:8 left:8 right:8, z:6)
settings popover (top:50 right:8, z:7, hidden default)
statusbar (bottom:8 left:8 right:8, z:4, pointer-events:none)
hidden file input (position:fixed offscreen)
cursor (position:absolute, dynamic left/top, z:3)
drop hint (matches canvas inset:8, z:8, hidden default)
```

z-order: cursor (3) < statusbar (4) < toolbar (6) < settings (7) < drop hint (8). Cursor under everything visible так что toolbar buttons видимы поверх него.

## 13. Hidden file input: position:fixed offscreen

`position:fixed; left:-9999px; top:-9999px` для скрытого `<input type="file">`. Не использовать `position:absolute width:1px height:1px` — некоторые браузеры блокируют программный `.click()` на элементах с micro-size.

## 14. Drag-and-drop + paste image

Container-level dragenter/dragover/dragleave/drop для drag-and-drop. Document-level paste с проверкой `pointerOverContainer()` чтобы избежать конфликта между несколькими нодами одного типа на графе.

```js
function pointerOverContainer() {
    const rect = container.getBoundingClientRect();
    return state.cursorClientX >= rect.left && state.cursorClientX <= rect.right
        && state.cursorClientY >= rect.top && state.cursorClientY <= rect.bottom;
}
```

Не забыть `document.removeEventListener("paste", ...)` в node._tsCleanup чтобы не утекать listener'ы.

## 15. Nodes 2.0 бесконечно растягивается вниз: media-контент ДОЛЖЕН быть out-of-flow + getMaxHeight ОБЯЗАТЕЛЕН

Самый частый и коварный баг V2 (Vue) DOM-нод. Симптом: при переключении workflow-вкладок / рендерера нода **бесконечно растягивается вниз** (или растёт на ~200px за каждый tab-switch). Ловил на `TS_AudioPreview` / `TS_AudioLoader` (waveform `<canvas>`) и `TS_Animation_Preview` (`<video>`).

**Механизм.** Vue-грид ноды считает высоту слота DOM-виджета из **min-content контейнера**. Если внутри лежит `<canvas>`/`<video>`/`<img>` **в потоке** с `width:100%;height:100%`, у него есть *intrinsic height* (canvas — из выставленного `canvas.height`; video/img — из aspect ratio). В потоке эта высота становится min-content → Vue растит ноду под неё → `syncDomSize`/следующая отрисовка кормит canvas новой (большей) высотой → цикл. `getMaxHeight`, читающий `node.size`, только усиливает: «любая высота — устойчивая точка», нода ползёт до упора (viewport).

**Единое правило для ЛЮБОЙ resizable DOM-ноды с canvas/video/img/большим списком:**

1. **Growable/media-контент — `position:absolute; inset:0`** (не `width:100%;height:100%` в потоке). Заполняет слот, но НЕ даёт min-content. Так уже сделано в `sam_media_loader` (`.ts-sml__canvas`), `ideogram` (`.ts-ideo-node__canvas`), `style` (`.ts-style-scroll`). Аудио/анимация были исключением — теперь приведены к этому.
2. **Задавать И `getMinHeight`, И `getMaxHeight`** (не только min). `getMaxHeight = () => max(floor, node.size[1] - CHROME)` — тогда панель заполняет пользовательский ресайз, но потолок привязан к ноде и не убегает (безопасно ТОЛЬКО вместе с п.1).
3. **Если ВСЕ дети absolute** — контейнер схлопнется в полоску на первом mount (min-content 0). Добавить **скрытый in-flow спейсер**: `width:1px;height:<floor>px;visibility:hidden;pointer-events:none;flex:0 0 auto`. См. `sam`, `ideogram`, теперь `animation`.
4. **Не писать явную `domWidgetEl.style.height` в V2** — Vue сам владеет высотой слота через getMin/getMax; явная высота конфликтует и через `afterResize` кормит цикл. Ставить высоту элемента только в legacy (V1) ветке.

Проверка: Playwright-аудит `create → serialize → clear → loadGraphData` ×3, мерить `node.size[1]`. Стабильная высота = ok; рост >60px за 3 reload = RUNAWAY. Гоняю по всем GUI-нодам (`tests`-скрипт в scratchpad) — цель «RUNAWAY count: 0». Guard-тест: `tests/test_dom_widget_inflow_media.py` статически стережёт п.1.

---

## ⚠️ «Задать размер, если его ещё нет» — не работает НИКОГДА (аудит 11.08.2026)

Расширение получает ноду, которой LiteGraph УЖЕ выдал заводской размер. Поэтому
такая заготовка тихо не делает ничего:

```javascript
if (!node.size || node.size.length < 2) {          // ложь всегда
    node.size = [300, 260];
}
node.size = [Math.max(node.size?.[0] || DEFAULT, MIN), ...];  // тоже: `||` не сработает
```

Итог — нода остаётся заводского размера, а её собственный интерфейс в него не
помещается. Живые последствия, найденные разом: у `TS_Animation_Preview` ряд
управления плеером был обрезан на 20 точек (человек, вытащивший ноду из меню, не
видел ни play, ни звука), у `TS_IdeogramDesigner` — нижний ряд на 8 точек.

**Правило:** не «задать, если пусто», а **дорастить до своего размера**:

```javascript
node.size = [
    Math.max(Number(node.size?.[0]) || 0, DEFAULT_WIDTH),
    Math.max(Number(node.size?.[1]) || 0, DEFAULT_HEIGHT),
];
```

⚠️ И только **после `addDOMWidget`**: он сам пересчитывает размер и затирает всё,
выставленное раньше (замерено на студии — высота возвращалась к 82). Там, где
нужен пересчёт раскладки, звать `node.setSize(...)`, а не присваивать поле.

**Гард на весь пак:** `test_every_node_ui_fits_its_own_box` в
`tests/test_browser_smoke.py` — создаёт каждую ноду и требует, чтобы корень
`.ts-ui` не переполнялся собственными детьми (допуск 2 точки). Прокручиваемые по
замыслу панели перечислены поимённо, а не выключены оптом.

**Закрыто 11.08.2026 — и урок общий.** Запускатель студии обрезался на четыре
точки. Перевод на `addResizableDomWidget` помог только наполовину: сколько ноды
занимает её обвязка, КОНСТАНТОЙ НЕ ОПИСЫВАЕТСЯ. Замерено на живой ноде — сразу
после создания слот получает 24 точки, а после первого изменения размера при той
же высоте ноды 44; прикидка «обвязка 58» давала 28, «78» тоже не спасала.

Работает только самокоррекция по факту: на следующих кадрах измерить, чего не
хватило слоту, и дорастить ноду ровно на столько. Попыток несколько (раскладка
складывается не за один кадр — единственная попытка успевала не всегда), выход
по достижении нужной высоты. Итог: слот 44, переполнение 0, кнопка помещается.

⚠️ Отсюда правило: **не подбирать константу обвязки, а мерить**. Любое число,
подходящее сегодня, завтра разъедется — у другого рендерера, при другом наборе
сокетов, при другом шрифте.

## ⚠️ Абсолютному ребёнку нужен позиционированный хозяин (14.09.2026)

Жалоба владельца: «TS Compare — в Nodes 2.0 полосочка перекрывает верхние
виджеты настроек. На видео всё нормально, заползал именно в режиме фото».

Картиночная шторка `.ts-cmp` (`js/_studio/_compare.js`) лежит ВНЕ ПОТОКА —
`position:absolute; inset:0`, — то есть ищет ближайшего позиционированного
предка. Хозяин обязан его дать, но `.ts-compare__slot` был `static`.

Почему не замечали: **в Nodes 1.0 корень DOM-виджета — `position:fixed`**
(`div.dom-widget.size-full`), и шторка случайно попадала почти туда, куда надо.
В Nodes 2.0 виджет лежит в обычной сетке (`lg-node-widgets`, все предки
`static`), containing block уезжает выше — и шторка встаёт поверх ноды.

Замерено на живом сервере, нода 360×340:

| | шторка | панель | итог |
|---|---|---|---|
| Nodes 2.0, до | 87…343 | 206…364 | накрывала fps / label_a / label_b |
| Nodes 1.0, до | 188…390 | 188…381 | вылезала на 9 px вниз, глазом не видно |
| после, оба | = панели | | |

Лечение — одна строка: `position:relative` у `.ts-compare__slot`. Видео-шторку
(`.ts-cmpv`) это не задевало никогда: её корень лежит в потоке и сам
`position:relative`. Студия тоже была цела — её хост `.ts-istudio__fitbox`
позиционирован.

**Правило:** если в общий модуль отдают элемент с `position:absolute`, хозяин
обязан быть позиционирован — и проверять это надо в Nodes 2.0, потому что
классический режим прячет ошибку своим `fixed`-корнем.
