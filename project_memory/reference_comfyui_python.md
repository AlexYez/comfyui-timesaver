---
name: ComfyUI Python для GPU тестов
description: Использовать D:/AiApps/ComfyUI/comfyui/python/python.exe для тестов с numpy/torch/PIL и регенерации contract snapshot
type: reference
originSessionId: bf21541d-3471-465e-95ae-3e55c9033923
---
Тестовый Python (`C:/Users/Sanchez/Documents/Apps/EnvPortable/python/python.exe`) **не** содержит numpy/torch/PIL — тесты с `pytest.importorskip("numpy")` под ним скипаются.

Для настоящей верификации используй ComfyUI-овский portable Python:

```
D:/AiApps/ComfyUI/comfyui/python/python.exe
```

Содержит `numpy`, `torch` (CUDA), `PIL`, `comfy_api`, `folder_paths`, весь ComfyUI runtime.

Команды:

```bash
D:/AiApps/ComfyUI/comfyui/python/python.exe -m compileall .
D:/AiApps/ComfyUI/comfyui/python/python.exe -m pytest tests/ -v
D:/AiApps/ComfyUI/comfyui/python/python.exe tools/build_node_contracts.py
```

В verification summary всегда указывай, под каким Python запускались проверки (тесты под обычным Python со скипом — это не настоящая проверка).

См. также CLAUDE.md §4.5.2 и §10.
