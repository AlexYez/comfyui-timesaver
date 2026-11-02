---
name: reference_sidebar_tabs
description: "Галочки вкладок левого сайдбара ComfyUI в настройках пака (js/utils/ts-sidebar-tabs.js) — API, порядок, ловушка схлопывания настроек"
metadata:
  type: reference
---

Галочки вкладок левого сайдбара ComfyUI в настройках пака (js/utils/ts-sidebar-tabs.js) — API, порядок, ловушка схлопывания настроек

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

# Вкладки левого сайдбара (07.10.2026, фронтенд 1.53.10)

Просьба владельца: в «Настройки → TS Timesaver» по галочке на каждый пункт
левого меню. Сделано для ВКЛАДОК: assets, node-library, model-library,
workflows, apps + вкладки других паков (Artius Browser). «Шаблоны» и «Центр
поддержки» (кнопки, не вкладки; метки `.templates-tab-button`,
`[data-testid=help-center-button]`) владелец сказал НЕ трогать; нижние кнопки
(нижняя панель, горячие клавиши, «Настройки») стабильных меток не имеют вовсе.

## Факты, замеренные в браузере

- API: `app.extensionManager.getSidebarTabs()` / `registerSidebarTab(tab)` /
  `unregisterSidebarTab(id)`; хранилище `app.extensionManager.sidebarTab`
  (Pinia: `activeSidebarTabId`, `toggleSidebarTab`, `$subscribe`).
- ⚠️ `registerSidebarTab` дописывает В КОНЕЦ — порядок первого знакомства
  запоминается, при изменении переставляется только «хвост» не на месте.
- ⚠️ Окно настроек строит дерево по ПОЛНОМУ пути `category`: настройки с
  одинаковым путём схлопываются в одну (видна последняя). Пять галочек с общим
  `["TS Timesaver","Interface","Left sidebar items"]` показались ОДНОЙ (Artius).
  Последний элемент категории — уникальный (`Left sidebar: <id>`), он не
  отображается. Порядок в окне — по `sortOrder` (больше — выше).
- Вкладки других паков ловятся `sidebarTab.$subscribe` (не таймером): галочка
  через `app.ui.settings.addSetting`, перевод — `pickLocaleStrings` (файл
  локали их id не знает). `setting.set` того же значения onChange НЕ зовёт.
- Переводы `locales/ru/settings.json` подхватываются только после перезапуска
  ComfyUI.
- Тест `tests/test_sidebar_tabs.py` (настройки изолированы conftest — реальные
  настройки владельца не трогает). `test_group_badge` иногда падает на
  `tsab:health` из пака Artius — чужой шум, не этот модуль.
