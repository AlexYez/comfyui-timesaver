# Тема: подключение и компоненты

Общие токены и готовые контролы заменяют независимые палитры каждой ноды.

Источник API — [js/_theme.js](../../js/_theme.js), сверено 2026-11-02. Упоминания tests/ и tools/ ниже относятся к локальной лаборатории; доступные проверки описаны в [testing](../operations/testing.md).

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

Основные значения из ComfyUI: `--comfy-menu-bg`,
`--comfy-input-bg`, `--input-text`, `--border-color`; статический fallback для muted также читает `--descrip-text`. Производные оттенки
**подмешиваются к ним**, а не задаются отдельной палитрой:

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
| `ts-ui-modal`, `ts-ui-keyanchor` | [полноэкранный редактор](overlays.md) |
| `ts-ui-scrim`, `ts-ui-spinner` | индикатор занятости |
| `ts-ui-drop`, `.is-drag-over` | drag-and-drop |
| `ts-ui-file` | скрытый `<input type=file>` (off-screen, **не** `display:none`) |
| `ts-ui-checker` | шахматка прозрачности |
| `ts-ui-ratios`, `ts-ui-ratio`, `__wrap`, `__frame`, `__label`, `.is-selected` | открытая сетка пропорций — `createRatioCards()` |
| `ts-ui-ratiopick`, `__trigger`, `__value`, `__caret`, `__pop`, `__custom` | компактный выбор пропорций — `createRatioPicker()` |

### Выбор пропорций — один контрол на весь пак

Везде, где спрашивают «какой формы кадр», спрашивают одинаково: сетка карточек в три столбца, в каждой — прямоугольник ровно той пропорции и подпись под ним. Собирается фабрикой `createRatioCards({values, onSelect})` из `_theme.js`; своей разметки и своих стилей заводить нельзя — на этом пак уже разъезжался.

Два вида одного контрола — выбирается по тому, сколько места есть:

```js
import { createRatioCards, createRatioPicker, isRatioList } from "../_theme.js";

// Открытая сетка — там, где тело можно растянуть (нода).
const cards = createRatioCards({
    values: ["1:1", "16:9", "9:16"],
    onSelect: (value) => apply(value),   // клик; select() молчит
});
container.appendChild(cards.element);
cards.select("16:9");                    // восстановление состояния
cards.add("5:4");                        // своя пропорция, в ту же сетку
cards.setDisabled(true);

// Компактный — в панели, где один столбец места.
const picker = createRatioPicker({
    values: ["1:1", "16:9", "9:16"],
    onSelect: (value) => apply(value),
    onCustom: (value) => apply(value),   // включает поле «ш:в» справа
    customPlaceholder: t.aspectCustom,
});
```

У компактного варианта то же API (`select/add/has/values/setDisabled`), плюс
`trigger`, `cards` и `close()`. Триггер показывает выбранную форму тем же
квадратом с рамкой — по нему видно выбор, не читая подпись. ⚠️ Слушатели
документа живут только пока окошко открыто и снимаются при закрытии: панель,
которую пересобирают (так делала студия на каждой смене модели), иначе
оставляет слушатель, переживший её, — это утечка ([DOM cleanup](../engineering/dom-widgets.md#12511-hidden-file-input--drag-drop--paste)).

⚠️ Пропорцию несёт **рамка внутри** карточки, а не сама кнопка: карточки одного размера, поэтому подписи стоят в ряд, а 9:21 не сплющивается.

⚠️⚠️ **Коробка под рамкой обязана быть КВАДРАТНОЙ** — на этом я уже ошибся. Обе стороны рамки заданы в процентах, а процент считается от разных баз: ширина от ширины коробки, высота от её высоты. Коробка шириной во всю карточку и высотой 26 px превращала «100% на 56%» в 80×15 — то есть 16:9 рисовался щелью, и так были сплющены все формы. Обе стороны квадрата берутся из одного токена `--ts-ratio-box`; у ноды квадрат растёт вместе с карточкой через `aspect-ratio: 1/1`.

⚠️ Список значений, состоящий **целиком** из пропорций, — это и есть выбор кадра: `isRatioList()` распознаёт это по данным, и контрол выбора может сам нарисовать карточки вместо строки кнопок (так работал «Новый кадр» в OutPaint студии). Одна не-пропорция в списке — и это обычные кнопки.

Где стоит: [`js/image/ts-resolution-selector.js`](../../js/image/ts-resolution-selector.js) (нода, сетка растянута на высоту виджета). Компактный вариант и `isRatioList()` жили в контролах TS Image Studio (снята 27.09.2026, локальный архив снятой TS Image Studio; для чистого клона — история Git) и остались в теме для следующего интерфейса. Гарды — `tests/test_ratio_cards.py`.

Эталон применения — [`js/image/lama_cleanup/_lama_helpers.js`](../../js/image/lama_cleanup/_lama_helpers.js).

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
ComfyUI (см. [testing](../operations/testing.md)).
