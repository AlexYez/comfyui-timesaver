---
name: Frontend extension IDs
description: Стабильные JS extension IDs пака — менять нельзя, на них завязаны сохранённые workflow (актуально на v9.5)
type: reference
originSessionId: 8022fd27-bafd-461a-97d9-dc12a4035284
---
JS-расширения пака (папка `js/`, `WEB_DIRECTORY = "./js"`):

| File | Extension ID | Привязанные ноды |
| --- | --- | --- |
| `js/utils/ts-bookmark.js` | `ts.bookmark` | `TS_Bookmark` (virtual node) |
| `js/image/ts-resolution-selector.js` | `ts.resolutionselector` | `TS_ResolutionSelector` |
| `js/audio/loader/ts-audio-loader.js` | `ts.audioLoader` | `TS_AudioLoader` |
| `js/audio/loader/ts-audio-preview.js` | `ts.audioPreview` | `TS_AudioPreview` |
| `js/image/lama_cleanup/ts-lama-cleanup.js` | `ts.lamaCleanup` | `TS_LamaCleanup` |
| `js/image/sam_media_loader/ts-sam-media-loader.js` | `ts.samMediaLoader` | `TS_SAM_MediaLoader` |
| `js/video/ts-animation-preview.js` | `ts.animationpreview` | `TS_Animation_Preview` |
| `js/text/ts-prompt-builder.js` | `ts.prompt_builder` | `TS_PromptBuilder` |
| `js/text/ts-style-prompt.js` | `ts_suite.style_prompt_selector` | `TS_StylePromptSelector` |
| `js/llm/ts-super-prompt.js` | `ts.superPrompt` | `TS_SuperPrompt` |
| `js/utils/sliders/ts-float-slider.js` | `ts.float-slider` | `TS_FloatSlider` |
| `js/utils/sliders/ts-int-slider.js` | `ts.int-slider` | `TS_Int_Slider` |
| `js/utils/sliders/_slider_helpers.js` | (private ES module) | shared helpers для slider extensions |
| `js/audio/loader/_audio_helpers.js` | (private ES module) | shared logic для TS_AudioLoader + TS_AudioPreview |
| `js/image/lama_cleanup/_lama_helpers.js` | (private ES module) | DOM widget logic для TS_LamaCleanup |
| `js/image/sam_media_loader/_sam_media_helpers.js` | (private ES module) | DOM widget logic для TS_SAM_MediaLoader |

Заметки про эволюцию:
- В релизе 8.8 общий `ts.slider-settings` разделён на `ts.float-slider` + `ts.int-slider` (через shared `_slider_helpers.js`). Property-значения (`min`/`max`/`step`/`default`) сохраняются на самих нодах.
- В v9.5 split-структура `js/audio/loader/` имеет два EXTENSION_ID (один на ноду); `_audio_helpers.js` — общий ES module (loader detect по `node.type === "TS_AudioPreview"`).
- В v9.4 появился `ts.samMediaLoader` (TS_SAM_MediaLoader) — клик-пикер позитивных/негативных точек для SAM3.
- Слайдеры переехали в `js/utils/sliders/` (с v9.4-v9.5). Extension ID не менялись.

Where:
- Все JS подключаются автоматически через `WEB_DIRECTORY` в `__init__.py`.
- Регистрируются через `app.registerExtension({ name: EXTENSION_ID, ... })`.

Important:
- Эти ID НЕ менять — на них завязаны сохранённые workflow и persistent UI state.
- При создании новой ноды с фронтендом использовать новый стабильный ID, заранее согласованный с пользователем.
- `ts.bookmark` использует legacy LiteGraph (`registerCustomNodes` + `LGraphNode`), это ожидаемое исключение.
- Для DOM widget нод (lama_cleanup, sam_media_loader, audio loader/preview) см. CLAUDE.md §12.5.1–§12.5.13 (layout, parent CSS scale, rehydrate, hidden widgets fallback).
