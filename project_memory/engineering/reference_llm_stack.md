---
name: reference_llm_stack
description: "TS_Qwen3_VL_V3 и TS_SuperPrompt используют общий nodes/llm/_qwen_engine.py (с v9.5); это где живёт пайплайн Qwen, model loader, presets — не дублируй"
metadata:
  type: reference
originSessionId: 8022fd27-bafd-461a-97d9-dc12a4035284
---

TS_Qwen3_VL_V3 и TS_SuperPrompt используют общий nodes/llm/_qwen_engine.py (с v9.5); это где живёт пайплайн Qwen, model loader, presets — не дублируй

Исходники темы: [_qwen_engine.py](../../nodes/llm/_qwen_engine.py), [ts_qwen3_vl.py](../../nodes/llm/ts_qwen3_vl.py), [_qwen.py](../../nodes/llm/super_prompt/_qwen.py), [qwen_3_vl_presets.json](../../nodes/qwen_3_vl_presets.json), [ts_super_prompt.py](../../nodes/llm/super_prompt/ts_super_prompt.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

С релиза `v9.5` (commit `20d84ff` — "LLM stack refactor") Qwen-пайплайн вынесен в shared модуль `nodes/llm/_qwen_engine.py` (~1250 строк). Он экспонирует loader, chat template, generation, presets и другие primitives.

Кто его использует:
- `nodes/llm/ts_qwen3_vl.py` — `TS_Qwen3_VL_V3` стал тонкой обёрткой над `_qwen_engine` (раньше был ~2300 строк, теперь ~600). Schema, UI и presets — здесь, тяжёлая логика — в engine.
- `nodes/llm/super_prompt/_qwen.py` — тонкий адаптер для `TS_SuperPrompt` (voice → text → prompt enhancement). Подаёт другие presets, но дёргает тот же engine.

Why: модели и пайплайны загрузки Qwen3-VL дублировались между двумя нодами; обновление чего-либо требовало править два файла, легко расходились (preset format, quantisation, attention impl).

How to apply:
- Перед изменением Qwen-логики (квантизация, FlashAttention, chat template, генерация) — правь `_qwen_engine.py`, а не одну из нод.
- Тестирование: `tests/test_qwen_engine.py` + `tests/_engine_stubs.py` (shared fixture). Не дублируй стабы по нодам.
- Presets: `nodes/qwen_3_vl_presets.json` — единый файл.
- Lazy import: всё heavy (`torch`, `transformers`, `bitsandbytes`) грузится только в момент исполнения `execute`, не на module-import.
- Module-level state класса ноды (V3 lock_class) — через `class _NodeState: ...` + singleton; в engine своя система кеша.

Эталоны архитектуры:
- Loader cache: `nodes/llm/_qwen_engine.py` (модели, tokenizer).
- Слой ноды: `nodes/llm/ts_qwen3_vl.py` — schema + execute → `engine.run(...)`.
- Слой ноды с UI: `nodes/llm/super_prompt/ts_super_prompt.py` — schema + execute + aiohttp routes; `_qwen.py` — обёртка.
