---
name: dom_widget_layout
description: "DOM-виджеты: размеры, координаты, маска, курсор и производительность."
metadata:
  type: reference
---

DOM-виджеты: размеры, координаты, маска, курсор и производительность.

Исходники темы: [_lama_helpers.js](../../js/image/lama_cleanup/_lama_helpers.js).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Накопленные правила работы с интерактивными нодами через `addDOMWidget` (см. `js/image/lama_cleanup/_lama_helpers.js`). Каждый пункт — реальная ошибка, которую я делал в этой сессии.

## 1. Layout: НЕ переопределять `widget.computeSize`

ComfyUI core (>=1.34) для DOM widgets использует **`computeLayoutSize()`** в методе `LGraphNode#b`:

```js
if (widget.computeSize) {
    // фиксированный размер: r += widget.computeSize()[1] + 4
} else if (widget.computeLayoutSize) {
    // distributeSpace раздаёт остаток между layout-size widgets
    l.push({minHeight, maxHeight, w: widget});
}
```

`DOMWidgetImpl.computeLayoutSize()` возвращает `{minHeight, maxHeight}` из `getMinHeight()` / `getMaxHeight()`. **Если задать `widget.computeSize`** — widget уходит в первую ветку, фиксируется и не получает остаточное пространство. В сочетании с node.size feedback даёт **infinite layout loop** (особенно в V2 Vue).

Правильно:
```js
const widgetOptions = {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => 220,
    getMaxHeight: () => 8192,
    afterResize: () => { requestRedraw(); },
};
const domWidget = node.addDOMWidget(name, "div", container, widgetOptions);
// НЕ делать domWidget.computeSize = ...
```

## 2. Скрытие IMAGEUPLOAD/Vue input-area: использовать String input

Если в schema стоит `IO.Combo.Input(..., upload=IO.UploadType.image)`, ComfyUI добавляет:
- кнопку "choose file to upload" в V1
- preview изображения **под нодой** в V2 Vue (через `node.imgs`, который мой `Object.defineProperty` override НЕ перехватывает в Vue render)

Решение: для нод с собственным in-node UI заменить на `IO.String.Input("source_path", default="", socketless=True)` и делать upload через `/upload/image` напрямую из JS. Никакой IMAGEUPLOAD = никаких лишних preview, никаких лишних кнопок.

Дополнительно: убрать `advanced=True` с inputs если они скрыты JS — иначе V2 показывает toggle "Show advanced inputs" под нодой.

## 3. Координаты курсора: viewport vs local CSS pixels

CSS `left/top/width/height` интерпретируются в LOCAL (pre-transform) coords. `event.clientX` и `getBoundingClientRect()` — в VIEWPORT (post-transform). Когда родитель имеет `transform: scale(s)` (LiteGraph zoom, Vue node scale), эти системы расходятся.

Compensation:
```js
const containerRect = container.getBoundingClientRect();
const layoutWidth = container.offsetWidth || containerRect.width;
const parentScale = layoutWidth > 0 ? containerRect.width / layoutWidth : 1;
const inverseScale = parentScale > 0.001 ? 1 / parentScale : 1;
const xLocal = (clientX - containerRect.left) * inverseScale - (container.clientLeft || 0);
const yLocal = (clientY - containerRect.top) * inverseScale - (container.clientTop || 0);
```

`offsetWidth / getBoundingClientRect().width` — автоматически детектит **любой** parent transform (не только LiteGraph zoom), включая Vue node scale.

Размер кисти/курсора в local pixels: `displaySize = state.brushSize * (state.scale / parentScale)`.

## 4. Не использовать `transform: translate()` для динамической позиции

Композиция CSS-трансформ родителя и `cursor.style.transform` даёт sub-pixel rounding errors, накапливающиеся пропорционально расстоянию от viewport-origin. Это видимо при очень большом изображении или сильном zoom.

Использовать `cursor.style.left/top` напрямую — гарантированно совпадает с системой координат `getBoundingClientRect`.

## 5. Mask compositing с `source-in` стирает изображение!

**Антипаттерн** который я делал в первой версии:
```js
ctx.drawImage(state.image, ...);
ctx.drawImage(maskCanvas, ...);                    // mask поверх image
ctx.globalCompositeOperation = "source-in";
ctx.fillRect(0, 0, imageWidth, imageHeight);       // ← стирает image!
```

`source-in` сохраняет пиксели только там где **И** source **И** destination непрозрачные. После drawImage(image), destination — это всё image. fillRect полным dark color заполняет всю область. Результат: image полностью заменён dark color **независимо от того, нарисовал ли user маску**. Картинка пропадает сразу.

**Правильно**: tinted mask — отдельный offscreen canvas с тем же paint operation что и mask, но другим цветом. В redraw — простой `drawImage(tintedMaskCanvas, ...)` без compositing трюков:

```js
function drawSegment(fromX, fromY, toX, toY, radius) {
    // mask: white для backend payload
    maskCtx.strokeStyle = "rgba(255,255,255,1)";
    maskCtx.beginPath(); maskCtx.moveTo(...); maskCtx.lineTo(...); maskCtx.stroke();
    // tinted mask: dark для display
    tintedMaskCtx.strokeStyle = "rgba(8,12,18,1)";
    tintedMaskCtx.beginPath(); tintedMaskCtx.moveTo(...); tintedMaskCtx.lineTo(...); tintedMaskCtx.stroke();
}

function redraw() {
    ctx.drawImage(imageCacheCanvas, 0, 0);
    ctx.globalAlpha = 0.72;
    ctx.drawImage(tintedMaskCanvas, state.offsetX, state.offsetY, drawWidth, drawHeight);
}
```

## 6. Image padding для floating toolbar/statusbar — через JS scale, НЕ через CSS canvas inset

**Антипаттерн**: позиционировать canvas через `top:56px; bottom:44px; left:8px; right:8px` чтобы избежать перекрытия toolbar/statusbar.

Не работает: `<canvas>` — replaced element, при `position:absolute` без явных width/height ведёт себя нестабильно при resize ноды (image «улетает», getBoundingClientRect возвращает аномальные значения).

**Правильно**: оставить canvas full-bleed (`inset: 0`), а **image padding** делать в JS при вычислении scale/offset:

```js
const IMAGE_PAD_TOP = 56;
const IMAGE_PAD_BOTTOM = 44;
const IMAGE_PAD_SIDE = 8;

function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();  // full container
    if (state.imageWidth > 0 && rect.width > 0) {
        const usableWidth = Math.max(1, rect.width - IMAGE_PAD_SIDE * 2);
        const usableHeight = Math.max(1, rect.height - IMAGE_PAD_TOP - IMAGE_PAD_BOTTOM);
        state.scale = Math.min(usableWidth / state.imageWidth, usableHeight / state.imageHeight);
        state.offsetX = IMAGE_PAD_SIDE + (usableWidth - state.imageWidth * state.scale) / 2;
        state.offsetY = IMAGE_PAD_TOP + (usableHeight - state.imageHeight * state.scale) / 2;
    }
}
```

Image fit-letterbox внутри usable area (canvas минус padding). Pointer→image math автоматически работает корректно (mouse в области toolbar/statusbar даёт imageX < 0 или > imageWidth → withinImage=false → cursor скрывается). Toolbar/statusbar overlap визуально canvas, но не image.

`rebuildImageCache` должен использовать `state.offsetX/Y/scale` (не пересчитывать из rect) — иначе image cache и mask blit рассинхронизируются.

## 7. Cursor visibility: условный `cursor:none`

`cursor:none` на canvas скрывает нативный курсор всегда. Если кастомный HTML cursor показывается ТОЛЬКО при `state.image` — то без картинки пользователь видит "мёртвую зону" над canvas (нет ни native cursor, ни custom cursor).

```css
.ts-lama__canvas{cursor:default}                /* default arrow без картинки */
.ts-lama__canvas.has-image{cursor:none}         /* hide native только когда есть image */
```

```js
canvas.classList.toggle("has-image", Boolean(state.image));  // в updateMeta
```

## 8. Performance recipe для big images (4K+)

Три ключевых паттерна, дающие плавную работу даже на 8K:

### 8.1. Image render cache

Предрендерить изображение в offscreen canvas at **display resolution**. Invalidate только при resize или image change. Каждый redraw — cheap blit вместо downscaling источника:

```js
function rebuildImageCache(rectWidth, rectHeight, dpr) {
    imageCacheCanvas.width = rectWidth * dpr;
    imageCacheCanvas.height = rectHeight * dpr;
    imageCacheCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    imageCacheCtx.drawImage(state.image, state.offsetX, state.offsetY,
                             state.imageWidth * state.scale,
                             state.imageHeight * state.scale);
    imageCacheValid = true;
}

function redraw() {
    if (!imageCacheValid) rebuildImageCache(...);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(imageCacheCanvas, 0, 0);  // cheap blit
}
```

### 8.2. Incremental tinted mask

Вместо rebuild tinted mask с нуля каждый frame (clearRect + fillRect + destination-in на full image resolution = ~48M ops для 4K), рисовать тонированную копию **в обе offscreen canvas** во время `drawSegment`/`drawBrushAt`:
- `maskCanvas` — белые пиксели (для backend payload)
- `tintedMaskCanvas` — тёмные пиксели (для display, identical paint operations)

Redraw делает только blit готового tinted.

### 8.3. HTML cursor element (не canvas drawing)

`<div class="cursor"/>` с `position:absolute; pointer-events:none`. Обновление через style.left/top на pointer move без `requestRedraw`. Cursor-only движения не вызывают canvas paint.

```js
function onPointerMove(event) {
    state.cursorClientX = event.clientX;
    state.cursorClientY = event.clientY;
    if (state.isDrawing) {
        drawSegment(...);                      // mask меняется
        requestRedraw();                       // нужен полный redraw
    } else {
        updateCursorElement();                 // только CSS left/top, без canvas
    }
}
```
