---
name: reference_modal_hotkeys
description: "Modal editors MUST park focus on a hidden textarea — event-order shielding against ComfyUI hotkeys is unwinnable (Ctrl+Z reaches the graph ChangeTracker)"
metadata:
  type: reference
---

Modal editors MUST park focus on a hidden textarea — event-order shielding against ComfyUI hotkeys is unwinnable (Ctrl+Z reaches the graph ChangeTracker)

Исходники темы: [_ideogram_editor.js](../../js/ideogram/_ideogram_editor.js).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Урок из undo/redo для TS_IdeogramDesigner (2026-07-03), применим к ЛЮБОЙ будущей модалке поверх ComfyUI.

**Проблема.** ComfyUI V2 регистрирует свой hotkey-сервис на `window` в CAPTURE-фазе при старте приложения. Любой listener модалки зарегистрирован ПОЗЖЕ → в той же фазе выполняется ПОСЛЕ → `stopPropagation()`/`preventDefault()` НЕ мешают Comfy. Итог: Ctrl+Z внутри модалки уходит в graph ChangeTracker, откатывает последнее действие графа (например, добавление самой ноды) → нода умирает → cleanup закрывает модалку у пользователя под руками. Событийную гонку выиграть НЕЛЬЗЯ.

**Решение (паттерн keyAnchor)** — см. `js/ideogram/_ideogram_editor.js`:
1. Скрытый `<textarea class="...keyanchor" readOnly tabIndex=-1 aria-hidden>` в оверлее (off-screen `left:-9999px`, §12.5.11-стиль). Comfy-сервис пропускает события, чей target — текстовое поле → граф в безопасности.
2. `keyAnchor.focus()` при открытии; `overlay.addEventListener("focusin", ...)` возвращает фокус с кнопок/DIV'ов на якорь (реальные INPUT/TEXTAREA/SELECT/contenteditable — не трогать); `overlay.addEventListener("pointerup", ...)` перепарковывает фокус после кликов по пустым местам (клик по div блюрит поле на body → следующий Ctrl+Z ушёл бы в граф).
3. Свой keydown-хендлер: гейт «не мешать реальным полям» должен ИСКЛЮЧАТЬ якорь: `if ((tag==="TEXTAREA"||...) && active !== keyAnchor) return;`.
4. Обработанные ветки всё равно делают `preventDefault()+stopPropagation()` (экран от поздних слушателей), listener можно держать на `window` capture — но защита обеспечивается ЯКОРЕМ, не порядком.

**ЧАСТЬ 2 — parkFocus (найдено вторым ревью, тем же днём): якорь недостаточен без сети возврата фокуса.** Фокус СБЕГАЕТ на `<body>` (где Comfy-хоткеи снова живы) после: (а) teardown inline-редактора (`ta.remove()`), (б) `window.prompt/confirm/alert`, (в) file-диалогов (`<input type=file>` change), (г) ЛЮБОЙ пересборки DOM, удаляющей сфокусированный элемент (`renderInspector`/`renderBlockPanel`/`renderLayers` — напр., выбор идеи из дропдауна пересобирает инспектор). Решение: `parkFocus()` = «если activeElement вне overlay (или body/null) → keyAnchor.focus(); живые поля ВНУТРИ overlay не трогать» + вызовы во всех точках (а)-(г) + страховка в 500ms-тике поллера (worst-case окно 0.5 с). E2e-регресс: `test_ideogram_editor_e2e.py` — inline commit → Ctrl+Z: граф цел, модалка жива, откатился ТЕКСТ.

**Trade-off:** Tab-навигация по кнопкам модалки жертвуется (focus снапится на якорь) — ок для pointer-driven канвас-инструментов.

**Диагностика такой проблемы:** «блоки исчезли после Ctrl+Z» может быть закрывшейся модалкой: проверяй `app.graph._nodes.length` до/после и `!!document.querySelector('.overlay')` — у нас undo «работал» в дебаге, пока не выяснилось, что это Comfy откатывал ГРАФ (нод стало 10 — вернулся прежний воркфлоу).

Связано: [ts-ideogram-designer-node](ts-ideogram-designer-node.md) (undo/redo реализация), [reference_dom_widget_pitfalls](reference_dom_widget_pitfalls.md).
