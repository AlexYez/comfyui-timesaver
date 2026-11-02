# Дизайн-система TS

Один акцент, база из темы ComfyUI, общие компоненты и предсказуемый ввод.

| Документ | Содержание |
| --- | --- |
| [Подключение и компоненты](components.md) | Токены, кнопки, выбор пропорций, Canvas, допустимые цвета поверх медиа. |
| [Локализация и размеры](localization.md) | pickLocaleStrings, native nodeDefs, данные EN/RU, addResizableDomWidget. |
| [Fullscreen и playback](overlays.md) | Общий оверлей, клавиатура, курсоры, pause при генерации и уходе в фон. |
| [DOM widgets](../engineering/dom-widgets.md) | Rehydrate, hidden state, DnD, layout и cleanup интерактивной ноды. |

Канонический код темы: [js/_theme.js](../../js/_theme.js). Каждая нода со своим
GUI импортирует TS_UI_CLASS/ensureThemeStyles и использует токены --ts-*.
Собственный CSS описывает layout, не независимую палитру. Это нужно, чтобы
светлая и тёмная темы, размеры и локализация работали одинаково во всём паке.
