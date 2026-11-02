# Локализация и resizable DOM

Интерфейс EN/RU имеет один источник для каждого вида текста, а размеры DOM задаёт общий хелпер.

Сверено с [js/_theme.js](../../js/_theme.js), [js/_dom_widget.js](../../js/_dom_widget.js) и [locale JSON](../../locales/ru/nodeDefs.json) 2026-11-02. Упоминания tests/ и tools/ — локальная лаборатория, не содержимое чистого клона.

## 5.5. Локализация: весь видимый текст двуязычный (en/ru)

Кнопки, тултипы, статусы, плейсхолдеры и пустые состояния каждой GUI-ноды
обязаны существовать на английском и русском. Механизм — `pickLocaleStrings()`
из `_theme.js`:

```javascript
import { pickLocaleStrings } from "../_theme.js";

const STRINGS = {
    en: { load: "Load Image", saved: (n) => `Saved to output: ${n}` },
    ru: { load: "Загрузить изображение", saved: (n) => `Сохранено в output: ${n}` },
};

function setupMyNode(node) {
    const L = pickLocaleStrings(STRINGS);   // ← внутри setup, не на module level
    button.textContent = L.load;
    setStatus(L.saved(filename));
}
```

Правила:

- Язык берётся из `Comfy.Locale` (настройка ComfyUI); неизвестная локаль и
  недостающие ключи откатываются на английский **поштучно** — `undefined` в UI
  невозможен.
- Резолвить словарь **внутри setup-функции**: на module level настройки могут
  быть ещё не загружены. Смена локали перезагружает страницу, поэтому живой
  перерисовки не требуется.
- Динамические сообщения — значения-функции (`saved: (n) => ...`), а не
  конкатенация переводов по кускам.
- Текст, пришедший с backend (`payload.error`, `status.message`), не переводим —
  показываем как есть; переводим только свои fallback-литералы вокруг.
- **Логи (`console.*`, Python logging) остаются английскими** — это операционные
  сообщения, не UI.
- Название кнопки-лаунчера не дублировать в словарях — только
  `getOpenInterfaceLabel()` ([fullscreen](overlays.md)).
- Скриншоты для документации снимаются на английском: `tools/screenshot_nodes.py`
  сам временно ставит `Comfy.Locale=en` и возвращает обратно.

Ideogram Designer использует собственную, более широкую i18n-таблицу (`t()` в
`_ideogram_shared.js`) — исторически и осознанно: у него локализуется весь
редактор, включая контентные подписи. Новым нодам достаточно `pickLocaleStrings`.

### Масштабирование DOM-виджета: `addResizableDomWidget()`

Ноды с собственным resizable-интерфейсом **обязаны** монтировать его через
`addResizableDomWidget()` из [`js/_dom_widget.js`](../../js/_dom_widget.js), а не
писать sizing-логику заново. Хелпер один раз кодирует правильную работу в обоих
рендерерах (Nodes 1.0 и Nodes 2.0/Vue) и на любом зуме канваса:

```javascript
import { addResizableDomWidget } from "../_dom_widget.js";

addResizableDomWidget(node, container, {
    name: "ts_my_widget",
    minWidth: 240, minHeight: 280,
    defaultWidth: 250, defaultHeight: 340,
    chromeHeight: 56,        // высота над виджетом (title + штатные виджеты), legacy
    minWidgetHeight: 160,
});
```

Что он делает и почему так:

- **Никакой JS-геометрии.** Высота берётся из `node.size` (это graph-единицы =
  layout-пиксели, не зависят от zoom-трансформа), а не из `getBoundingClientRect()`
  (viewport-пиксели, домноженные на зум). Измерение rect и запись его обратно —
  причина #1 перекошенного UI на зумах ≠ 1 ([геометрия DOM](../engineering/dom-widgets.md#1253-координаты-курсора-viewport-vs-local-css-pixels)).
- **Разные хуки для разных рендереров.** Legacy сайзит виджет через `computeSize`,
  Vue — через `getMinHeight/getMaxHeight` ([layout DOM](../engineering/dom-widgets.md#1251-layout-не-задавать-computesize-для-vue)). Хелпер ставит нужный.
- `hideOnZoom: false` — содержимое не пропадает при отдалении.
- Ставит `node.resizable`, `min_size`, оборачивает `onResize` с клампом размеров.

CSS виджета должен быть flex/absolute-раскладкой, заполняющей 100% высоты — хелпер
раздаёт только слот, внутреннюю геометрию делает CSS. Если внутри есть длинный
скроллируемый блок, держи его **вне потока** (`position:absolute` внутри
flex-контейнера), иначе его натуральная высота раздует ноду в V2 — эталон
обхода в `js/text/ts-style-prompt.js` (`.ts-style-body` / `.ts-style-scroll`).

Эталон применения хелпера — `js/image/ts-resolution-selector.js`.

### Нативные ноды (без своего DOM): `locales/ru/nodeDefs.json`

У нод, чей интерфейс — стандартные виджеты ComfyUI (нет своего DOM), `pickLocaleStrings`
не применяется. Их имена входов/выходов и тултипы локализует **штатный механизм
ComfyUI**:

- **Канон — английский, в Python-схеме**: у каждого входа `IO.*.Input(..., tooltip="...")`
  по-английски. Это то, что видит английский пользователь и что уходит в `/object_info`.
- **Русский — в `locales/ru/nodeDefs.json`**, ключ = `node_id`:
  ```json
  {
    "TS_Color_Match": {
      "inputs":  { "reference": { "name": "референс", "tooltip": "..." } },
      "outputs": { "0":         { "name": "изображение", "tooltip": "..." } }
    }
  }
  ```
- Сервер собирает `locales/<lang>/nodeDefs.json` со всех custom_nodes и отдаёт на
  `/api/i18n`; frontend применяет к нативным нодам. Читается на **старте сервера**
  (`@lru_cache`) — правки требуют перезапуска ComfyUI.
- Имена нод (`display_name`) НЕ локализуем — это идентификаторы для поиска и туториалов.
- Тултипы не входят в contract snapshot → workflow-совместимость не затрагивается.
- `locales/` входит в репозиторий. Отдельного .comfyignore в нынешнем дереве нет.
- **Данные, а не интерфейс** (библиотеки пресетов вроде `nodes/styles/styles.json`)
  локализуются внутри самой записи, парой ключей: **голый ключ — английский,
  суффикс `_ru` — русский** (`name`/`name_ru`, `category`/`category_ru`,
  `description`/`description_ru`). JS выбирает по `getUiLanguage()` и откатывается
  на английский **поштучно**, поэтому неполный перевод не ломает карточку. Новую
  пару нужно объявить и в белом списке полей нормализатора на бэкенде — иначе она
  не доедет до фронтенда.

Guard: `tests/test_locale_nodedefs.py` — паритет схема↔`locales/ru` (каждый видимый
вход имеет русскую запись; нет ссылок на несуществующие входы). Термины — по
единому глоссарию (image→изображение, mask→маска, strength→сила, …).

---
