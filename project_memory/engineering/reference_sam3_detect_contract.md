---
name: reference_sam3_detect_contract
description: "Родная ComfyUI SAM3_Detect ожидает positive_coords/negative_coords как STRING JSON в пикселях, не custom-type с нормализацией"
metadata:
  type: reference
originSessionId: bc8042bf-eee4-4edc-90d7-f0e534e58fbe
---

Родная ComfyUI SAM3_Detect ожидает positive_coords/negative_coords как STRING JSON в пикселях, не custom-type с нормализацией

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Родная ComfyUI-нода `SAM3_Detect` (`comfy_extras/nodes_sam3.py`, node_id `SAM3_Detect`) принимает точечные подсказки как:

- `positive_coords: STRING` (force_input, optional) — JSON `[{"x": int, "y": int}, ...]` в **пиксельных** координатах исходного `image` входа.
- `negative_coords: STRING` (force_input, optional) — то же самое.

Парсинг внутри: `json.loads(positive_coords)` → `pos_pts` → `p["x"] / W * 1008`, label=1 (positive) или 0 (negative).

**Не путать** со сторонним паком `comfyui-sam3` (`SAM3PointCollector` / `SAM3Segmentation`), который использует **другой** контракт:
- Custom тип `SAM3_POINTS_PROMPT`.
- Формат `{"points": [[x_norm, y_norm], ...], "labels": [...]}` в **нормализованных** [0,1] координатах.

**Why:** при первом походе по запросу пользователя я ошибочно изучил сторонний `comfyui-sam3` пак и сделал output как `IO.Custom("SAM3_POINTS_PROMPT")` с нормализацией — координаты не подключались к родной SAM3 Detect.

**How to apply:** если задача упоминает "родная нода ComfyUI SAM3 Detect" — сразу смотреть `comfy_extras/nodes_sam3.py` в корне ComfyUI, не custom-nodes-пак. Output для совместимости — `IO.String.Output(display_name="positive_coords"/"negative_coords")` с JSON-строкой `[{"x":int,"y":int},...]` в пикселях исходного image (для видео — размер первого/всех кадров совпадает).

**Verified 2026-07-24 (v9.82):** формат подтверждён сквозным тестом — `_serialize_sam3_coords` (execute) даёт `[{"x":int,"y":int}]` (int, клампится в границы, пустой `[]`), `graphToPrompt` отдаёт STRING, воспроизведение родного парсинга `json.loads`+`p["x"]/W*1008`+labels `[1..]/[0..]` проходит без ошибок. Прошлая ошибка «родные ноды не понимали» = именно тот custom-type `SAM3_POINTS_PROMPT`, уже исправлено.

**Mask postprocess (v9.83, Meta-faithful):** `_postprocess_mask` в `_sam_media_helpers.py` = **точный** алгоритм Meta `remove_small_regions` (segment_anything/utils/amg.py) через хелпер `_remove_small_regions`: два size-aware прохода `cv2.connectedComponentsWithStats(8)` — сначала `mode="holes"` (закрыть дыры **меньше** порога, крупные легитимные дыры бублика/просвета сохраняются), потом `mode="islands"` (убрать острова меньше порога, крупнейший всегда остаётся). Порог `area = max(32, 0.0004*H*W)`. **БЕЗ морфологии/median-blur** — граница уже гладкая: SAM3_Detect билинейно апскейлит логиты до полного разрешения ДО порога (`F.interpolate(...,'bilinear')` → `>0`), морфология только съедает тонкие структуры (Meta её сознательно не делает; исследования: plain closing вредит чётким краям). `refine_sam_mask_tensor` применяет ту же чистку к **выходной** `initial_mask` (раньше сырая) — превью == выход, обе строго бинарны (валидный seed для Video Track). **Осторожно:** порог должен быть << площади фона, иначе holes-проход зальёт весь кадр (сам фон посчитается «дырой» < порога) — реальные пороги маленькие, но не давай пользователю выкручивать. Тесты: `tests/test_sam_mask_postprocess.py`. Отвергнутый ранее v9.82-подход (contour-fill ВСЕХ дыр + median) — не воскрешать: заливал крупные легитимные дыры и скруглял реальные углы. Guided-filter/матирование (софт-край) — НЕ делали: маска перестаёт быть бинарной, плоха как seed; отдельной опцией на будущее. Детали ресёрча — сессия 2026-07-24.

**Точность координат — доказана строго (2026-07-25, v9.83), не перепроверять без причины.** Точки ставятся только в полноэкранном оверлее (внутреннего зума/пана у SAM-редактора НЕТ, только letterbox-фит; CSS-zoom ноды не участвует), `pointerToImageCoords` зовёт `resizeCanvas()` на каждом pointer-event → scale/offset всегда из живого rect, дрейфа при резайзе нет. Два независимых Playwright-теста:
1. **WYSIWYG** (40 кликов, viewport 1600×900/1280×720/1024×768/1920×1080 × dpr 1 и 2): зелёная точка находится поиском по пикселям canvas — рисуется под курсором с ошибкой **0.35–0.8px**, 40/40.
2. **Ground truth** (15 кликов, включая нестандартные пропорции 1100×900, 1920×700): в картинку вшиты красные маркеры на известных пикселях, их экранная позиция ищется **по пикселям** (без математики ноды), клик ровно туда → сохранённая координата совпадает с маркером, ошибка **0.22–0.83px** (после int-округления в `_serialize_sam3_coords` — точное попадание), 15/15.

⚠️ Артефакт тестирования: повторный клик по существующей точке **удаляет** её (штатный toggle) — в тестах чистить точки между прогонами, иначе выглядит как «точка не нарисовалась». См. [reference_fullscreen_overlay](reference_fullscreen_overlay.md).
