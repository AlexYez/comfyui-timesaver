# DOM widgets: layout и жизненный цикл

Практические правила собственного интерфейса: геометрия, сохранение состояния, ввод и очистка.

Паттерны перенесены из прежнего CLAUDE.md; соответствующие shared helpers
сверены 2026-11-02. Все упоминания tests/ в этом документе — имена сценариев
**локальной лаборатории**, а не команды, доступные в чистом клоне.
Для нового интерфейса используйте [тему и fullscreen](../theme/README.md),
а для запуска проверок — [testing](../operations/testing.md).


Если делаешь полноценный in-node UI (canvas, кисти, drag-drop и т.п.) через `addDOMWidget` — соблюдай эти правила. Каждое — реальная ошибка из истории. История инцидентов доступна через [индекс памяти](../../project_memory/MEMORY.md).

### 12.5.1. Layout: не задавать computeSize для Vue

Для Vue layout установка widget.computeSize выталкивает DOM widget в фиксированную ветку и может вызвать бесконечный рост высоты. Используйте addResizableDomWidget из [общего хелпера](../../js/_dom_widget.js): он задаёт computeSize только для legacy, а для Vue — getMinHeight/getMaxHeight. Пример ниже показывает низкоуровневую причину, а не заменяет общий хелпер.

```javascript
const widgetOptions = {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => 220,
    getMaxHeight: () => 8192,
    afterResize: () => requestRedraw(),
};
const domWidget = node.addDOMWidget(name, "div", container, widgetOptions);
// НЕ делать domWidget.computeSize = ... !
```

### 12.5.2. Не использовать IMAGEUPLOAD/upload= если есть свой UI

`IO.Combo.Input(..., upload=IO.UploadType.image)` добавляет нежелательные UI-элементы в обоих режимах:
- V1: кнопку "choose file to upload" над нодой.
- V2 Vue: image preview **под** нодой (через `node.imgs`, который Vue render не пропускает через `Object.defineProperty`).

Решение: `IO.String.Input("source_path", default="", socketless=True)` + ручной upload через `/upload/image` из JS. Также убирай `advanced=True` с inputs скрытых JS — иначе V2 показывает "Show advanced inputs" toggle.

### 12.5.3. Координаты курсора: viewport vs local CSS pixels

`event.clientX` и `getBoundingClientRect()` — viewport (post-transform). `cursor.style.left` — local CSS pixels (pre-transform). Parent `transform: scale(s)` (LiteGraph zoom, Vue node scale) разводит эти системы → cursor drifts.

Compensation через ratio offsetWidth:

```javascript
const containerRect = container.getBoundingClientRect();
const layoutWidth = container.offsetWidth || containerRect.width;
const parentScale = layoutWidth > 0 ? containerRect.width / layoutWidth : 1;
const inverseScale = parentScale > 0.001 ? 1 / parentScale : 1;
const xLocal = (clientX - containerRect.left) * inverseScale - (container.clientLeft || 0);
const yLocal = (clientY - containerRect.top) * inverseScale - (container.clientTop || 0);
cursor.style.left = `${xLocal - radius}px`;  // НЕ transform: translate — sub-pixel ошибки
cursor.style.top = `${yLocal - radius}px`;
```

### 12.5.4. Mask compositing: НЕ использовать `source-in` с fillRect

Антипаттерн: `drawImage(image)` → `drawImage(maskCanvas)` → `globalCompositeOperation = "source-in"` → `fillRect(...)`. Это **стирает image** даже когда маска пустая, потому что source-in fillRect полностью покрывает destination.

Правильно: tinted mask — отдельный offscreen canvas с тем же paint operations что и mask, но dark color. Redraw делает простой `drawImage(tintedMaskCanvas, ...)` без compositing трюков.

### 12.5.5. Image padding под floating overlays — через JS scale, НЕ CSS canvas inset

Для in-node UI с floating toolbar/statusbar: НЕ позиционировать canvas через `top:56px;bottom:44px` — `<canvas>` это replaced element, нестабилен на resize.

Канвас держать full-bleed (`inset:0`), padding делать в `state.scale/offsetX/offsetY`:

```javascript
const IMAGE_PAD_TOP = 56;
const IMAGE_PAD_BOTTOM = 44;
const IMAGE_PAD_SIDE = 8;

function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    if (state.imageWidth > 0 && rect.width > 0) {
        const usableWidth = Math.max(1, rect.width - IMAGE_PAD_SIDE * 2);
        const usableHeight = Math.max(1, rect.height - IMAGE_PAD_TOP - IMAGE_PAD_BOTTOM);
        state.scale = Math.min(usableWidth / state.imageWidth, usableHeight / state.imageHeight);
        state.offsetX = IMAGE_PAD_SIDE + (usableWidth - state.imageWidth * state.scale) / 2;
        state.offsetY = IMAGE_PAD_TOP + (usableHeight - state.imageHeight * state.scale) / 2;
    }
}
```

`rebuildImageCache` должен использовать `state.offsetX/Y/scale` (не пересчитывать) — иначе cache и mask blit рассинхронизируются.

### 12.5.6. Performance recipe для big images

Три ключевых паттерна, дающих плавную работу даже на 8K:

1. **Image render cache** — pre-rendered image at display resolution в offscreen canvas, rebuild только на resize/image-change. Каждый redraw — cheap blit.
2. **Incremental tinted mask** — рисовать в обе offscreen canvas (mask + tintedMask) во время `drawSegment`/`drawBrushAt`. Redraw = blit готового tinted без full rebuild.
3. **HTML cursor element** — `<div class="cursor">` с `position:absolute; pointer-events:none`. Обновлять через style.left/top на pointer move. Cursor-only движения **НЕ** вызывают `requestRedraw`.

### 12.5.7. Cursor visibility

`cursor:none` на canvas скрывает native cursor. Если custom HTML cursor показывается только при `state.image` — без image над canvas получается "мёртвая зона". Решение: условный CSS класс:

```css
.ts-lama__canvas{cursor:default}
.ts-lama__canvas.has-image{cursor:none}
```

```javascript
canvas.classList.toggle("has-image", Boolean(state.image));
```

### 12.5.8. Backend для интерактивных нод

- **Per-session asyncio.Lock** для длительных jobs, исполняемых в одном event loop:
  ```python
  _session_locks: dict[str, asyncio.Lock] = {}
  def _get_session_lock(session_id):
      lock = _session_locks.get(session_id)
      if lock is None:
          lock = asyncio.Lock()
          _session_locks[session_id] = lock
      return lock

  async def handler(request):
      lock = _get_session_lock(safe_session_id)
      async with lock:
          # serialised per session
  ```
- **Versioned working files** — `{session}_{tag}_{nanos:020d}.png`. Никогда не overwrite — нужно для undo/redo.
- **Cleanup** на `/seed`, `/reset`, history overflow. Только `path.name.startswith(f"{safe_session}_")` — защита от удаления чужих файлов.

### 12.5.9. Output organization

Сохранять результаты в подпапку с тегом в имени:
- `output/<feature_name>/<source_stem>_<feature_name>_<timestamp>.png`.
- Response `{"subfolder": "<feature_name>", "type": "output"}`.

### 12.5.10. Folder registration для моделей

```python
def _register_model_folder():
    base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(base))

_register_model_folder()  # на module import — поддержка extra_model_paths.yaml
```

### 12.5.11. Hidden file input + drag-drop + paste

- `<input type="file">` скрыть через `position:fixed; left:-9999px; top:-9999px;` (НЕ `width:1px;height:1px` и НЕ `display:none` — некоторые браузеры блокируют программный `.click()` по схлопнутому/неотрисованному инпуту). Дополнить off-screen позиционирование `opacity:0; pointer-events:none` можно; `width/height:1px` поверх off-screen — не нарушение, но и не нужно. Эталон: `_lama_helpers.js`, `_sam_media_helpers.js`; в v9.77 к нему приведены `_audio_helpers.js` (был `width:1px` без off-screen) и `_ideogram_editor.js` (был `display:none`).
- Container-level dragenter/dragover/dragleave/drop для drag-and-drop.
- Document-level paste с проверкой `pointerOverContainer()` чтобы избежать конфликта между несколькими нодами одного типа.
- В cleanup-хуке ноды (с согласованной цепочкой прежних обработчиков) обязательно `document.removeEventListener("paste", ...)` — иначе утечка listener'ов.

### 12.5.12. State rehydration при переключении workflow-вкладок

Порядок хуков ComfyUI при загрузке/переключении workflow-вкладок:

1. `onNodeCreated` — widget values ещё **дефолтные**, restore из workflow JSON ещё не произошёл.
2. ComfyUI применяет `widgets_values` к нодам.
3. `loadedGraphNode` — здесь widget values уже актуальные.

Если в `onNodeCreated` инициализировать JS state из widget values, он навсегда зависнет на defaults; затем `syncWidgets()` затрёт восстановленные значения этими defaults.

**Антипаттерн:** безусловно пересобрать DOM widget в `loadedGraphNode` через cleanup + новый `addDOMWidget`. В V2 Vue `DOMWidgetImpl` регистрирует widget одновременно в Vue render tree и LiteGraph widgets — простой `element.remove() + splice(node.widgets)` не вычищает обе регистрации → **двоение верхней части ноды**.

**Правильный паттерн:** отдельная `rehydrate` функция, которая обновляет state из текущих widget values + `node.properties`, без пересоздания DOM:

```javascript
// В setup<NodeName>(node):
node._ts<Name>Rehydrate = () => {
    state.mode = String(getWidgetValue(node, INPUT_MODE, state.mode) || state.mode);
    state.cropStart = readPersistedNumber(node, INPUT_CROP_START, state.cropStart);
    state.cropEnd = readPersistedNumber(node, INPUT_CROP_END, state.cropEnd);
    // ...
    if (isPreviewNode) restorePreviewState();
    else if (state.sourcePath) fetchMetadata(state.sourcePath);
    else { updateText(); drawWaveform(); }
};

// В registerExtension:
loadedGraphNode(node) {
    if (!getWidget(node, DOM_WIDGET_NAME)) {
        setup<NodeName>(node);   // первая регистрация
        return;
    }
    node._ts<Name>Rehydrate?.();  // sync state без rebuild
}
```

### 12.5.13. Hidden widgets в V2 Vue: используй `node.properties` как fallback

Hidden widgets (`advanced=True` + `widget.type = "hidden"` в JS) в V2 Vue могут не сериализовать `widget.value` через workflow JSON. Для скрытых float/int/string инпутов используй двойной канал хранения:

```javascript
function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (widget) {
        widget.value = value;
        if (typeof widget.callback === "function") widget.callback(value);
    }
    node.properties ||= {};
    node.properties[name] = value;  // зеркало для надёжности
}

function readPersistedNumber(node, name, fallback) {
    const w = getWidget(node, name)?.value;
    if (w !== undefined && w !== null && w !== "") {
        const n = Number(w);
        if (Number.isFinite(n)) return n;
    }
    const p = node?.properties?.[name];
    if (p !== undefined && p !== null && p !== "") {
        const n = Number(p);
        if (Number.isFinite(n)) return n;
    }
    return fallback;
}
```

Также: при `applyMediaPayload` / любой ре-инициализации не сбрасывай user-state (crop, brushes) безусловно — clamp к новой длительности/размеру и сохраняй sentinel-значения (`-1` = "до конца") как валидные. Эталон фикса — `js/audio/loader/_audio_helpers.js` (`applyMediaPayload`).

### 12.5.15. Горячие клавиши против набора текста

Пока фокус в поле ввода, простые клавиши — это ввод, а не команды. Полноэкранный
оверлей раздаёт `keydown` всем подряд, и без проверки буква «e» в промпте
переключает режим, а `[` и `]` меняют кисть (реальная жалоба).

Правило одно на весь пак — [`js/_keys.js`](../../js/_keys.js):

```javascript
import { isTypingTarget, hotkeysAllowed } from "../_keys.js";

// В раздатчике оверлея:
if (!hotkeysAllowed(event)) return;              // в поле — только с Ctrl/Cmd
onKey?.(event, { typing: isTypingTarget(event.target) });

// В обработчике: что можно во время набора, решает он сам.
onKey: (event, { typing = false } = {}) => {
    if (event.ctrlKey && event.key === "Enter") { run(); return; }
    if (typing) return;
    // ...команды холста
}
```

Свой список тегов (`["input","textarea"]`) заводить нельзя — он забудет
`<select>` и `contenteditable`. Исторический локальный гард: tests/test_hotkeys_vs_typing.py (не поставляется в чистом клоне).

### 12.5.16. `DataTransfer` живёт только до первого `await`

После первой уступки управления `event.dataTransfer.getData()` возвращает пустую
строку, а запасные метки на `window` успевает стереть `dragend` (приходит сразу
после `drop`). Асинхронный разбор дропа поэтому работает через раз — тем чаще
ломается, чем больше источников.

Снимок снимается синхронно, первой строкой обработчика, дальше разбирают его:

```javascript
const drop = async (event) => {
    const shot = captureDrop(event.dataTransfer);   // ДО любого await
    if (!sniffDrop(shot)) return;
    const items = await normalizeDrop(shot);
};
```

Эталон — `js/_studio/_dnd.js`; гард — `tests/test_drop_snapshot.py`.

### 12.5.17. Размер ноды и клавиша «R» — два крючка, мимо которых живут свои интерфейсы

**Размер после загрузки workflow не тот, что сохранили.** В файле числа
правильные — их затирает раскладка, пересчитывая высоту по виджетам ПОСЛЕ
`configure`. Замерено: TS Video Saver 520×640 → 520×**480**, TS Video Loader
650 → **670**. Лечится общим `holdSize()` в
[`js/_dom_widget.js`](../../js/_dom_widget.js) — размер ставится заново, когда
раскладка улеглась (сразу, кадр, 120 мс, 400 мс), на `onConfigure` и
`onExecuted`. Своей копии в ноде не заводить. Гард:
`tests/test_node_size_memory.py`.

**Списки по «R».** `app.refreshComboInNodes()` обновляет только СТАНДАРТНЫЕ
combo-виджеты, перебирая `node.widgets`. Нода, рисующая выбор сама, обязана
слушать крючки:

```javascript
// У каждой ноды графа, включая лежащие в подграфах; defs приходят с вызовом.
node.refreshComboInNode = (defs) => { /* обновить свой список */ };
// У расширения, ПОСЛЕ обхода нод: сбросить общий кэш.
app.registerExtension({ refreshComboInNodes(defs) { forgetCache(defs); } });
```

⚠️ **Combo описан ДВУМЯ способами в одной сборке**: V1 — `[[...варианты], {}]`,
V3 — `["COMBO", {options: [...]}]`. Замерено в `/object_info`: 256 виджетов
первой формы, 576 второй. Читать обе — и во фронтенде, и на бэкенде
(`INPUT_TYPES()` V3-ноды тоже может вернуть вторую).

Гард: `tests/test_lora_refresh.py`. Подробности —
[индекс памяти](../../project_memory/MEMORY.md).

### 12.5.14. Эталонная реализация

`nodes/image/lama_cleanup/` + `js/image/lama_cleanup/` — TS_LamaCleanup, наиболее полный пример паттернов из §12.5.1–§12.5.11.

`nodes/audio/loader/` + `js/audio/loader/` — TS_AudioLoader + TS_AudioPreview, эталон для §12.5.12–§12.5.13 (workflow-tab rehydrate, hidden widgets + properties fallback).

---
