# TS Theme — единый визуальный стиль пака

Все ноды `comfyui-timesaver`, рисующие собственный интерфейс, используют одну
дизайн-систему: [`js/_theme.js`](../js/_theme.js). Хардкод цветов в новых нодах —
ошибка ревью.

Кредо: **Один акцент. Серая база из ComfyUI. Ноль хардкода.**

---

## 1. Как подключить (шаблон новой ноды)

```javascript
import { app } from "/scripts/app.js";

import { TS_UI_CLASS, ensureThemeStyles } from "../_theme.js";  // путь от твоей папки

const STYLE_ID = "ts-my-node-styles";

function ensureStyles() {
    // Токены и общие компоненты приходят из js/_theme.js; ниже — только layout.
    ensureThemeStyles();
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.ts-my-node{display:flex;flex-direction:column;gap:6px;padding:6px;
  background:var(--ts-bg);border:1px solid var(--ts-border-soft);
  border-radius:var(--ts-radius-lg)}
`;
    document.head.appendChild(style);
}

// Корневой хост обязан нести класс-скоуп токенов:
container.className = `${TS_UI_CLASS} ts-my-node`;
```

Правило: **свой stylesheet — только layout** (расположение, размеры, z-index).
Всё, что касается цвета, шрифта и скруглений, берётся из токенов.

---

## 2. Токены

### Акцент — одна ручка

```javascript
export const TS_ACCENT = "#9a8cc7";   // js/_theme.js
```

Это единственное место, где задан фирменный цвет. Остальная гамма
(`--ts-accent-strong`, `--ts-accent-dim`, `--ts-accent-soft`, `--ts-accent-line`,
`--ts-accent-contrast`) выводится из него через `color-mix()`. Меняешь одну
строку — перекрашивается весь пак.

### Поверхности и текст

| Токен | Роль |
| --- | --- |
| `--ts-bg` | фон панели/тела ноды (из `--comfy-menu-bg`) |
| `--ts-surface` | кнопки, поля (из `--comfy-input-bg`) |
| `--ts-surface-hover` / `--ts-surface-active` | состояния кнопки |
| `--ts-elevated` | поповеры, всплывающие панели |
| `--ts-sunken` | «утопленные» поля ввода, колодцы |
| `--ts-scrim` | затемняющая подложка (**тёмная в любой теме** — это её смысл) |
| `--ts-modal-bg` | фон полноэкранного редактора |
| `--ts-text` / `--ts-muted` / `--ts-faint` | текст: основной / вторичный / третичный |
| `--ts-border` / `--ts-border-soft` / `--ts-border-strong` | линии |
| `--ts-danger` / `--ts-success` / `--ts-warning` | семантика |
| `--ts-radius-sm/-/-lg`, `--ts-font`, `--ts-fs-xs/-sm/-/-lg` | форма и типографика |
| `--ts-shadow`, `--ts-shadow-sm` | тени |
| `--ts-checker` | шахматка прозрачности под изображениями |

### Почему работает светлая тема ComfyUI

Из ComfyUI читаются **только четыре** значения: `--comfy-menu-bg`,
`--comfy-input-bg`, `--input-text`, `--border-color`. Всё остальное
**подмешивается к ним**, а не задано константами:

```css
--ts-surface-hover: color-mix(in srgb, var(--ts-surface) 88%, var(--ts-text));
```

Шаг делается **в сторону цвета текста**, то есть всегда прочь от фона. На
тёмной теме это осветляет, на светлой — затемняет. Одно правило, оба
направления. Так же считаются рамки, приглушённый текст, шахматка и тень.

Семантические красный/зелёный/жёлтый тоже подтягиваются к цвету текста, иначе
пастель, подобранная под тёмный фон, выцвела бы на белом.

`color-mix()` есть в любом Chromium с 2023 года. На старом фронтенде сработает
статический тёмный набор из блока до `@supports` — интерфейс останется рабочим.

---

## 3. Готовые компоненты

Не переизобретай кнопку — используй классы из `_theme.js`:

| Класс | Что это |
| --- | --- |
| `ts-ui-btn` + `--primary` / `--danger` / `--ghost` / `--icon`, `.is-active` | кнопки |
| `ts-ui-slider` | ползунок |
| `ts-ui-toolbar`, `ts-ui-group`, `ts-ui-sep` | панель инструментов |
| `ts-ui-panel`, `ts-ui-title` | поповер настроек |
| `ts-ui-statusbar` + `.is-error` / `.is-success`, `ts-ui-ellipsis`, `ts-ui-meta` | статус-бар |
| `ts-ui-field`, `__row`, `__name`, `__value`, `ts-ui-label` | поля |
| `ts-ui-input`, `ts-ui-select`, `ts-ui-textarea` | ввод |
| `ts-ui-modal`, `ts-ui-keyanchor` | полноэкранный редактор (см. §5) |
| `ts-ui-scrim`, `ts-ui-spinner` | индикатор занятости |
| `ts-ui-drop`, `.is-drag-over` | drag-and-drop |
| `ts-ui-file` | скрытый `<input type=file>` (off-screen, **не** `display:none`) |
| `ts-ui-checker` | шахматка прозрачности |

Эталон применения — [`js/image/lama_cleanup/_lama_helpers.js`](../js/image/lama_cleanup/_lama_helpers.js).

---

## 4. Canvas: цвета через `getThemeColors()`

В `<canvas>` CSS-переменные недоступны. Для waveform, графиков и прочей
отрисовки:

```javascript
import { getThemeColors } from "../_theme.js";

const colors = getThemeColors();   // кешируется, дёргать в draw-цикле безопасно
ctx.fillStyle = colors.accent;
ctx.strokeStyle = colors.border;
```

Доступно: `accent`, `accentStrong`, `accentDim`, `text`, `muted`, `faint`,
`border`, `borderSoft`, `bg`, `surface`, `sunken`, `danger`, `success`.

---

## 5. Полноэкранные редакторы

Оверлей монтируется на `document.body` с классами
`` `${TS_UI_CLASS} ts-ui-modal` ``. **Обязателен** keyAnchor —
скрытая `<textarea class="ts-ui-keyanchor">` с припаркованным фокусом, иначе
Ctrl+Z уйдёт в graph ChangeTracker и снесёт ноду вместе с открытой модалкой.
Подробности и полный паттерн: CLAUDE.md §12.5 и
`project_memory/reference_modal_hotkeys.md`.

---

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
  `getOpenInterfaceLabel()` (см. §5).
- Скриншоты для документации снимаются на английском: `tools/screenshot_nodes.py`
  сам временно ставит `Comfy.Locale=en` и возвращает обратно.

Ideogram Designer использует собственную, более широкую i18n-таблицу (`t()` в
`_ideogram_shared.js`) — исторически и осознанно: у него локализуется весь
редактор, включая контентные подписи. Новым нодам достаточно `pickLocaleStrings`.

### Масштабирование DOM-виджета: `addResizableDomWidget()`

Ноды с собственным resizable-интерфейсом **обязаны** монтировать его через
`addResizableDomWidget()` из [`js/_dom_widget.js`](../js/_dom_widget.js), а не
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
  причина #1 перекошенного UI на зумах ≠ 1 (§12.5.3).
- **Разные хуки для разных рендереров.** Legacy сайзит виджет через `computeSize`,
  Vue — через `getMinHeight/getMaxHeight` (§12.5.1). Хелпер ставит нужный.
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
- `locales/` трекается git и не режется `.comfyignore` — доедет до пользователей.
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

## 6. Когда хардкод цвета допустим

Ровно один случай: **элемент лежит поверх пользовательского контента** и обязан
читаться на любой картинке независимо от темы.

- чёрный letterbox под видео;
- панель управления плеером поверх кадра;
- подложка подписи на миниатюре;
- кольцо кисти (белое с тёмным ореолом);
- артборд Ideogram — там цвета это **данные пользователя**, а не наш chrome.

Каждый такой случай сопровождается CSS-комментарием с объяснением. Всё
остальное — токены.

---

## 7. Проверка

`tests/test_theme_compliance.py` статически стережёт правила: каждый JS-файл,
инжектящий `<style>`, обязан импортировать `_theme.js` и звать
`ensureThemeStyles()`, а количество хардкод-цветов в его stylesheet не должно
превышать зафиксированный бюджет. Добавил новый цвет — тест упадёт; либо
переводи на токен, либо осознанно расширяй бюджет с комментарием.

Визуальная проверка обеих тем — `tools/screenshot_nodes.py` при запущенном
ComfyUI (см. CLAUDE.md §4.5.3).
