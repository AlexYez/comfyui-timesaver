---
name: Project overview
description: comfyui-timesaver — пак из 59 ComfyUI-нод (все на V3 API), v9.11, репо AlexYez/comfyui-timesaver
type: project
originSessionId: 8022fd27-bafd-461a-97d9-dc12a4035284
---
Пак `comfyui-timesaver` — production-quality custom nodes и frontend extensions для ComfyUI.

Версия: `9.11` (см. pyproject.toml). Структура one-node-one-file с категориальным разбиением (`nodes/image/`, `nodes/video/`, `nodes/audio/`, `nodes/llm/`, `nodes/text/`, `nodes/files/`, `nodes/utils/`, `nodes/conditioning/`). Все 59 нод на V3 API.

Репозиторий: https://github.com/AlexYez/comfyui-timesaver
Корень: `ComfyUI/custom_nodes/comfyui-timesaver/`.

Распределение нод (по `tests/contracts/node_contracts.json`, всего 59):
- TS/Image: 28
- TS/Files: 8
- TS/Video: 7
- TS/Audio: 5
- TS/Text: 4
- TS/Utils: 4
- TS/LLM: 2
- TS/Conditioning: 1

Split-нодки (подпапки без `_`-префикса, внутри один публичный `ts_<name>.py` + приватные `_<name>.py`):
- `nodes/audio/loader/` — TS_AudioLoader, TS_AudioPreview
- `nodes/image/keying/` — TS_Despill, TS_Keyer
- `nodes/image/lama_cleanup/` — TS_LamaCleanup
- `nodes/image/sam_media_loader/` — TS_SAMMediaLoader
- `nodes/llm/super_prompt/` — TS_SuperPrompt (с backward-compat shim в `nodes/llm/ts_super_prompt.py`, у которого `NODE_CLASS_MAPPINGS = {}`, чтобы избежать двойной регистрации)

Why: Пак позиционируется как набор time-saving утилит для рабочих графов изображения/видео/аудио/LLM, ориентирован на пользователей с разной квалификацией (включая новичков ComfyUI).

How to apply: При работе сохранять стабильность публичных контрактов (workflow compatibility — приоритет №1), правильно классифицировать ноду по категории, использовать ts_-префикс файлов и TS_ префикс классов. Точное количество нод и схемы — всегда из `tests/contracts/node_contracts.json`, не из памяти.
