---
name: project_api_mix
description: "V3 API и module-level состояние; причина запрета записи в заблокированный класс ноды."
metadata:
  type: project
originSessionId: 166438cb-713b-4ab0-b416-6d7d7ebdb2ff
---

V3 API и module-level состояние; причина запрета записи в заблокированный класс ноды.

**Текущая опора (2026-11-02):** Ноды используют закреплённое пространство comfy_api.v0_0_2.IO. Изменяемое состояние хранится в module-level singleton, не присваивается классу заблокированной V3-ноды. Снимок [контрактов](../../tests/contracts/node_contracts.json) содержит 88 нод.

Исходники темы: [ts_bgrm_birefnet.py](../../nodes/image/ts_bgrm_birefnet.py), [ts_whisper.py](../../nodes/audio/ts_whisper.py), [ts_video_depth.py](../../nodes/video/ts_video_depth.py), [ts_qwen3_vl.py](../../nodes/llm/ts_qwen3_vl.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

С релиза `8.9` весь `comfyui-timesaver` переведён на ComfyUI V3 API. До этого был смешанным (часть V1, часть V3).

**Why:** в 8.9 пользователь явно попросил мигрировать всё; миграция выполнена 50 нод за 6 этапов (utils → text → audio → files → video → image) + финальный аудит для TS_Qwen3_VL_V3 / TS_SileroStress. Workflow compatibility сохранена побитно (`/object_info` сравнение до/после показало только V3 wire-format косметику).

**How to apply:**
- Все новые ноды пишем строго на V3: `class X(IO.ComfyNode)`, `define_schema()`, `execute()` как `@classmethod`, helpers тоже classmethod/staticmethod, без `__init__`.
- V1 можно встретить в истории или чужих legacy-плагинах; текущие правила — [engineering](../../doc/engineering/README.md), новые ноды этого пака пишутся на V3.
- Runtime state (model cache, resamplers) — module-level singleton; classmethod-getter может только обращаться к нему, не менять заблокированный класс ноды.
- `IS_CHANGED` → `fingerprint_inputs`, `VALIDATE_INPUTS` → `validate_inputs`, `OUTPUT_NODE` → `is_output_node`, `INPUT_IS_LIST` → `is_input_list`, `OUTPUT_IS_LIST` → `IO.X.Output(is_output_list=True)`, hidden `"PROMPT"` → `IO.Hidden.prompt` + `cls.hidden.prompt`.
- Wildcard `*` → `IO.AnyType.Input/Output`. Custom типы (TILE_INFO, CROP_DATA) → `IO.Custom("...")`.
- TS_Qwen3_VL_V3: суффикс _V3 относится к версии модели Qwen3-VL, а не к ComfyUI Node API (нода всё равно V3).
- Runtime state на класс ноды НЕЛЬЗЯ присваивать через `cls.X = ...` — V3 регистрация создаёт `<Name>Clone` через `comfy_api.internal.lock_class()` (V1-нод сейчас нет). Использовать module-level `class _NodeState: ...` + singleton инстанс. Эталоны: `nodes/image/ts_bgrm_birefnet.py`, `nodes/audio/ts_whisper.py`, `nodes/video/ts_video_depth.py`, `nodes/llm/ts_qwen3_vl.py`.
- Импорт: `from comfy_api.v0_0_2 import IO`. Pinned namespace, не `comfy_api.latest`.
- Историческая проверка v9.5 дала 59 нод. Текущее количество берётся из снимка контрактов, а не из числа файлов; строка `RETURN_TYPES` сама по себе не является надёжной проверкой API.
