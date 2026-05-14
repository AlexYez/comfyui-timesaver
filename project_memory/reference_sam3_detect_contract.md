---
name: SAM3 Detect (core) coords contract
description: Родная ComfyUI SAM3_Detect ожидает positive_coords/negative_coords как STRING JSON в пикселях, не custom-type с нормализацией
type: reference
originSessionId: bc8042bf-eee4-4edc-90d7-f0e534e58fbe
---
Родная ComfyUI-нода `SAM3_Detect` (`comfy_extras/nodes_sam3.py`, node_id `SAM3_Detect`) принимает точечные подсказки как:

- `positive_coords: STRING` (force_input, optional) — JSON `[{"x": int, "y": int}, ...]` в **пиксельных** координатах исходного `image` входа.
- `negative_coords: STRING` (force_input, optional) — то же самое.

Парсинг внутри: `json.loads(positive_coords)` → `pos_pts` → `p["x"] / W * 1008`, label=1 (positive) или 0 (negative).

**Не путать** со сторонним паком `comfyui-sam3` (`SAM3PointCollector` / `SAM3Segmentation`), который использует **другой** контракт:
- Custom тип `SAM3_POINTS_PROMPT`.
- Формат `{"points": [[x_norm, y_norm], ...], "labels": [...]}` в **нормализованных** [0,1] координатах.

**Why:** при первом походе по запросу пользователя я ошибочно изучил сторонний `comfyui-sam3` пак и сделал output как `IO.Custom("SAM3_POINTS_PROMPT")` с нормализацией — координаты не подключались к родной SAM3 Detect.

**How to apply:** если задача упоминает "родная нода ComfyUI SAM3 Detect" — сразу смотреть `comfy_extras/nodes_sam3.py` в корне ComfyUI, не custom-nodes-пак. Output для совместимости — `IO.String.Output(display_name="positive_coords"/"negative_coords")` с JSON-строкой `[{"x":int,"y":int},...]` в пикселях исходного image (для видео — размер первого/всех кадров совпадает).
