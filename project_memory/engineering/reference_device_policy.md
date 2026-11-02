---
name: reference_device_policy
description: "Пак не хардкодит CUDA — устройство через mm.get_torch_device(); регрессия охраняется test_no_new_cuda_hardcodes (frozen-список пуст с 2026-06-23)"
metadata:
  type: reference
---

Пак не хардкодит CUDA — устройство через mm.get_torch_device(); регрессия охраняется test_no_new_cuda_hardcodes (frozen-список пуст с 2026-06-23)

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Выбор устройства в паке — **только** через `comfy.model_management.get_torch_device()` (эталон: `nodes/llm/_qwen_engine.py::get_device()`). Прямой `.cuda()` / `torch.device("cuda")` / `device = "cuda" if ... else "cpu"` запрещён ([правила backend](../../doc/engineering/backend.md)): ломает CPU-only / ROCm / Apple MPS и игнорирует `--cpu` / `--cuda-device`.

## Guard
`tests/test_static_invariants.py::test_no_new_cuda_hardcodes` — regex-скан `nodes/` на `\.cuda\(\)` и `torch.device("cuda")`. Frozen-список `KNOWN_CUDA_HARDCODES` теперь **пуст** (2026-06-23) — в паке ноль хардкодов, тест PASSED (раньше был SKIPPED из-за устаревших записей `ts_whisper.py:590/594`, которых уже нет: файл отрефакторен в `_whisper_engine.py`). Любая новая запись = регрессия, чинить, а не grandfather'ить.

Нюанс теста: если в `KNOWN_*` есть запись, которой больше нет в коде → `pytest.skip()` (soft nudge почистить список), НЕ fail. Поэтому «удалил хардкод → почисти frozen-список», иначе тест молча скипается.

## Легитимные паттерны (НЕ хардкод, не трогать)
`device.type == "cuda"`, `torch.cuda.is_available()`, `torch.cuda.empty_cache()`, `torch.autocast(device_type="cuda")` под guard, `torch.cuda.device(idx)`, разбор `hf_device_map`, OOM-детект по строке. Это детект возможностей, а не предположение о наличии GPU.

## Whisper — особый случай
`nodes/_whisper_engine.py::resolve_device(device)` — единственный источник истины для Whisper (cuda/cpu, **без MPS** — openai-whisper его не ест надёжно). `load_model()` и `comfy_audio_to_mono16k()` оба резолвят через неё, чтобы аудио-тензор и модель были на одном устройстве (иначе device mismatch на инференсе). НЕ переводить whisper на `mm.get_torch_device()` напрямую — вернёт mps и сломает. Правки device-логики Whisper делай в `resolve_device`, не размазывай.

## Фикс 2026-06-23 (что было приведено к политике)
- `nodes/files/ts_model_converter_advanced.py` — 2× `device = "cuda" if ...` → `mm.get_torch_device()`; `device == "cuda"` → `device.type == "cuda"` (на mps/cpu падает в CPU-ветку конвертации, как и раньше; FP8-cast только на CUDA).
- `nodes/_whisper_engine.py::comfy_audio_to_mono16k` — ручной `torch.device("cuda")`-резолв → `torch.device(resolve_device(device))`.
- `tests/_engine_stubs.py` — стаб `_qwen_engine` теперь экспортит `QWEN_MODEL_LOCK` (см. [reference_voice_pipeline](reference_voice_pipeline.md)).
