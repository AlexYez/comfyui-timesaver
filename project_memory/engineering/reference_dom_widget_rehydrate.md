---
name: reference_dom_widget_rehydrate
description: "Pattern для interactive DOM widgets чтобы state переживал переключение workflow-вкладок ComfyUI без дублирования UI"
metadata:
  type: reference
originSessionId: 8022fd27-bafd-461a-97d9-dc12a4035284
---

Pattern для interactive DOM widgets чтобы state переживал переключение workflow-вкладок ComfyUI без дублирования UI

Исходники темы: [_audio_helpers.js](../../js/audio/loader/_audio_helpers.js).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

При переключении workflow-вкладок ComfyUI порядок хуков:

1. `onNodeCreated` срабатывает с widget values = defaults (восстановление ещё не произошло)
2. ComfyUI применяет widgets_values из workflow JSON
3. `loadedGraphNode` срабатывает

Если в `onNodeCreated` создать DOM widget со state из widget values — state навсегда зависнет на defaults, а виджеты будут перезаписаны при первом `syncWidgets()`.

**Неправильный фикс:** безусловный rebuild в `loadedGraphNode` — V2 Vue `DOMWidgetImpl` регистрирует widget в двух местах (Vue render tree + LiteGraph widgets), `element.remove() + splice` не вычищает обе регистрации → визуальное двоение верхней части ноды.

**Правильный паттерн:**

```javascript
// В setupAudioLoader(node):
node._tsAudioLoaderRehydrate = () => {
    state.mode = String(getWidgetValue(node, INPUT_MODE, state.mode) || state.mode);
    state.cropStart = readPersistedNumber(node, INPUT_CROP_START, state.cropStart);
    // ... rest of state from widgets/properties ...
    if (isPreviewNode) restorePreviewState();
    else if (state.sourcePath) fetchMetadata(state.sourcePath);
    else { updateText(); drawWaveform(); }
};

// В registerExtension:
loadedGraphNode(node) {
    if (!getWidget(node, DOM_WIDGET_NAME)) {
        setupAudioLoader(node);  // первая регистрация
        return;
    }
    node._tsAudioLoaderRehydrate?.();  // обновить state без rebuild
}
```

**Why:** не пересоздаём DOM (избегаем двоения), но синхронизируем JS state из восстановленных widget values после загрузки workflow.

**How to apply:** Любая нода с `addDOMWidget` и persistent state (crop, brushstrokes, etc.) должна экспортировать через `node._tsXRehydrate` функцию, которая читает widget values + node.properties и обновляет state + UI. Эталон — `js/audio/loader/_audio_helpers.js`.

Дополнительно: для hidden widgets (`advanced=True` + `widget.type = "hidden"`) V2 Vue может не сериализовать `widget.value` — используй `setWidgetValue`, который зеркалит value в `node.properties[name]`, и при чтении state используй helper типа `readPersistedNumber(node, name, fallback)` который сначала проверяет widget value, потом properties.
