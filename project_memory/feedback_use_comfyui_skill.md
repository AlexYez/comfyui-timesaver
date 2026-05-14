---
name: Use comfyui-custom-nodes skill
description: Перед нетривиальной работой над ComfyUI нодой подключать skill /comfyui-custom-nodes (frontend, basics, lifecycle, etc.)
type: feedback
originSessionId: bf21541d-3471-465e-95ae-3e55c9033923
---
Перед написанием/изменением ComfyUI ноды (особенно frontend JS, DOM widgets, V2 Vue layout, IMAGEUPLOAD/preview suppression, advanced inputs) **обязательно** инвокать skill `/comfyui-custom-nodes` или его подскилы:

- `comfyui-custom-nodes:comfyui-node-basics` — V3 структура ноды.
- `comfyui-custom-nodes:comfyui-node-frontend` — JS extensions, DOM widgets (`addDOMWidget` + `getMinHeight`/`getHeight`/`computeSize`), V2 Vue layout, suppression стандартных preview.
- `comfyui-custom-nodes:comfyui-node-inputs` / `outputs` / `datatypes` / `lifecycle` / `advanced` / `packaging` / `migration` — по назначению.

**Why:** база знаний skill'а содержит точные API и паттерны (например `widget.computeSize`, `widgetOptions.getMinHeight`/`getHeight`, отличия V1 LiteGraph vs V2 Vue render). Без skill'а легко наугад выбрать неработающий подход — пользователю пришлось дважды переделывать UI TS_LamaCleanup потому что я начинал кодить без skill'а.

**How to apply:** при первом упоминании задачи на ноду подключи нужный sub-skill **до** написания кода. Особенно критично для frontend и V2 (Vue) issues.
