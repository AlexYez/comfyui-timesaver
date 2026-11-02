---
name: reference_comfyui_python
description: "Для проверок нужен Python окружения ComfyUI; абсолютный D:-путь — пример установки владельца."
metadata:
  type: reference
originSessionId: bf21541d-3471-465e-95ae-3e55c9033923
---

Для проверок нужен Python окружения ComfyUI; абсолютный D:-путь — пример установки владельца.

**Текущая опора (2026-11-02):** Используется Python того окружения, где запущен ComfyUI. Абсолютный D:-путь ниже — пример машины владельца, а не переносимая зависимость. Канонические команды — [проверки](../../doc/operations/testing.md).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

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

Актуальные команды: [проверки](../../doc/operations/testing.md), конкретные пути интерпретатора: [окружение](../../doc/operations/environment.md).
