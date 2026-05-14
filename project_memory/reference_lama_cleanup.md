---
name: TS_LamaCleanup — pure PyTorch + safetensors loader (v9.3)
description: Lama-нода переехала с pickled .ckpt на safetensors + чистый PyTorch код вместо упакованного оригинала; модели в models/lama/, формат .safetensors
type: reference
originSessionId: 8022fd27-bafd-461a-97d9-dc12a4035284
---
С релиза `v9.3` (commit `e805a3d` — "TS_LamaCleanup pure-PyTorch + safetensors loader") TS_LamaCleanup использует:
- Чистый PyTorch код архитектуры в `nodes/image/lama_cleanup/_lama_arch.py` (отказ от внешней зависимости `lama-cleaner` или upstream lama pickle).
- Loader для `.safetensors` (а не `pickle` `.ckpt`) → безопаснее, проверяется `pickle.load`-aware sec rules в CLAUDE.md.
- Модели лежат в `models/lama/` (зарегистрировано через `folder_paths.add_model_folder_path`).

Why: pickled checkpoints — security risk и неудобство (зависели от точной структуры внешнего пакета). Safetensors + локальная архитектура снимают оба.

How to apply:
- Не путать с "lama-cleaner" python package — пак НЕ зависит от него.
- При обновлении весов: формат `.safetensors`, ключи матчатся локальной архитектурой в `_lama_arch.py`.
- При тестировании: `tests/test_lama_cleanup_contract.py` покрывает schema + helpers + save/cleanup/lock; тяжёлая inference часть скипается без torch.
- Эталонная нода для интерактивных DOM widgets — см. CLAUDE.md §12.5.1–§12.5.11.

Дополнительно (v9.5 fixes ранее в v9.3 батче):
- ffmpeg вызовы с таймаутами (без них процесс мог зависнуть на коррапнутом аудио).
- Upload size caps на HTTP routes.
- Memory leak fix в long-running session (per-session asyncio.Lock + versioned working files).
