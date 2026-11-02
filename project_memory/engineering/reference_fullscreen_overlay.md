---
name: reference_fullscreen_overlay
description: "Общий хелпер полноэкранного редактора js/_fullscreen.js — openFullscreenOverlay; используют Lama Cleanup и SAM Media Loader"
metadata:
  type: reference
---

Общий хелпер полноэкранного редактора js/_fullscreen.js — openFullscreenOverlay; используют Lama Cleanup и SAM Media Loader

Исходники темы: [_fullscreen.js](../../js/_fullscreen.js), [_theme.js](../../js/_theme.js), [_lama_helpers.js](../../js/image/lama_cleanup/_lama_helpers.js), [_sam_media_helpers.js](../../js/image/sam_media_loader/_sam_media_helpers.js), [_ideogram_editor.js](../../js/ideogram/_ideogram_editor.js).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

# Shared fullscreen overlay — `js/_fullscreen.js`

Ноды с тяжёлым in-node UI (Lama Cleanup, SAM Media Loader) прячут редактор в
полноэкранный оверлей. Плюмбинг оверлея вынесен в **`js/_fullscreen.js`** —
не изобретать заново.

## API

```js
import { openFullscreenOverlay } from "<путь>/_fullscreen.js";

const handle = openFullscreenOverlay(container, {
    onOpen:  () => { /* layout live: сбросить кэши, requestRedraw */ },
    onKey:   (event) => { /* editor-scoped хоткеи; Esc УЖЕ обработан хелпером */ },
    onClose: () => { /* teardown: сбросить state, unobserve, redraw шелла */ },
    closeOnBackdrop: false,   // true рискует потерять правки — по умолчанию false
    extraClass: "",
});
handle.close();      // идемпотентно
handle.isOpen();     // bool
handle.parkFocus();  // вернуть фокус на keyAnchor
```

Хелпер сам делает: `.ts-ui-modal` над вьюпортом, скрытый `.ts-ui-keyanchor`
(textarea, парковка фокуса → Comfy capture-phase Ctrl+Z не доходит до графа и не
удаляет ноду, см. [reference_modal_hotkeys](reference_modal_hotkeys.md)), focusin-репарковку (реальные
поля INPUT/TEXTAREA/SELECT/contentEditable фокус держат), window keydown в
capture (Esc → close, остальное → onKey), **унифицированную close-кнопку `×`**
(`.ts-ui-fs-close`, top-right, `showClose`/`closeTitle` опции; одна кнопка в
одном месте во всех редакторах), и на close **detach** контента
(`content.remove()`, НЕ destroy) — canvas/маски/undo-история переживают reopen.

Кнопка `×` живёт в собственной верхней полосе (`.ts-ui-fs-topbar`), содержимое
начинается под ней — налезать нечему. Раньше кнопка была `position:fixed` в углу
и каждый редактор был обязан помнить про инсет `--ts-fs-safe-right`; помнили не
все. Локализуй `closeTitle` у вызывающего (`L.closeEditor` / `L.closeTitle`).

⚠️ **`center: true` — это ДИАЛОГ, и полосы у него нет.** Полоса придумана для
редакторов во весь экран, где она единственная поверхность наверху. У диалога
поверхность своя — его собственная панель, — и полоса вставала НАД ней отдельной
серой плашкой в пустоте: именно её владелец увидел над окном отчёта загрузчика и
назвал битой графикой (2026-08-07). В `js/_theme.js` у центрированного варианта
полоса без фона и без границы, кнопка стоит на углу самой панели, а
`.ts-ui-fs-frame.ts-ui-fs-hastopbar` не резервирует высоту. Диалог держит
заголовок сам, поэтому `label` ему передавать не нужно. Гарды —
`tests/test_theme_compliance.py`.

⚠️ Стили темы — это шаблонная строка JS. Обратная кавычка внутри CSS-комментария
обрывает её и роняет весь модуль; `node --check` такое пропускает, ловится только
импортом (`node --input-type=module -e "await import(...)"`) или тестами.

## Паттерн ноды (shell + re-parent)

1. Компактный `shell` (превью + `createOpenInterfaceButton(openEditor)` +
   статус) монтируется как DOM widget (`addDOMWidget`), НЕ сам редактор.
2. Полный редактор (`container`) живёт в замыкании, в оверлей заезжает по
   `openEditor()`, на close только detach.
3. `openEditor`: выставить `state.editorOpen=true`, `resizeObserver.observe(container)`,
   вызвать `openFullscreenOverlay(container, {...})`, сохранить handle.
4. `closeEditor` = `editorHandle?.close()`. Teardown — в `onClose`.
5. Cleanup ноды (`_ts...Cleanup`) обязан звать `closeEditor()`, иначе оверлей
   утечёт при удалении ноды с открытым редактором.

Превью-canvas шелла — **`position:absolute; inset:0`**, никогда in-flow: in-flow
canvas несёт backing-store высоту как min-content → runaway в Vue (см.
[reference_dom_widget_pitfalls](reference_dom_widget_pitfalls.md) §15). Backing store грузить `w*dpr × h*dpr`,
CSS держит 100%/100%.

## Кто использует

- `js/image/lama_cleanup/_lama_helpers.js` — эталон паттерна (undo/redo, brush [/]).
- `js/image/sam_media_loader/_sam_media_helpers.js` — точки/маска/видео.

**Ideogram** (`js/ideogram/_ideogram_editor.js`) НЕ мигрирован — свой оверлей
(`ts-ideoe-keyanchor`), архитектурно другой (оверлей = постоянный хост, не
detach-on-close); миграция = крупное рискованное переписывание, не трогать без
причины.

Проверено Playwright в обоих рендерерах: оверлей открывается/закрывается,
Ctrl+Z не удаляет ноду, высота ноды стабильна (нет runaway).
