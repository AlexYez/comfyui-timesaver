---
name: reference_lama_cleanup
description: "Lama-нода переехала с pickled .ckpt на safetensors + чистый PyTorch код вместо упакованного оригинала; модели в models/lama/, формат .safetensors"
metadata:
  type: reference
originSessionId: 8022fd27-bafd-461a-97d9-dc12a4035284
---

Lama-нода переехала с pickled .ckpt на safetensors + чистый PyTorch код вместо упакованного оригинала; модели в models/lama/, формат .safetensors

Исходники темы: [_lama_arch.py](../../nodes/image/lama_cleanup/_lama_arch.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

С релиза `v9.3` (commit `e805a3d` — "TS_LamaCleanup pure-PyTorch + safetensors loader") TS_LamaCleanup использует:
- Чистый PyTorch код архитектуры в `nodes/image/lama_cleanup/_lama_arch.py` (отказ от внешней зависимости `lama-cleaner` или upstream lama pickle).
- Loader для `.safetensors` (а не `pickle` `.ckpt`) → безопаснее, проверяется `pickle.load`-aware sec rules в CLAUDE.md.
- Модели лежат в `models/lama/` (зарегистрировано через `folder_paths.add_model_folder_path`).

Why: pickled checkpoints — security risk и неудобство (зависели от точной структуры внешнего пакета). Safetensors + локальная архитектура снимают оба.

How to apply:
- Не путать с "lama-cleaner" python package — пак НЕ зависит от него.
- При обновлении весов: формат `.safetensors`, ключи матчатся локальной архитектурой в `_lama_arch.py`.
- При тестировании: `tests/test_lama_cleanup_contract.py` покрывает schema + helpers + save/cleanup/lock; тяжёлая inference часть скипается без torch.
- Эталонная нода для интерактивных DOM widgets — см. [правила DOM](../../doc/engineering/dom-widgets.md).

UI-архитектура (2026-07-21, commit `04c5b5b`): нода — компактный shell (превью + кнопка "Edit Image" + зеркало статуса); весь painting-UI (canvas/toolbar/settings/statusbar) монтируется в fullscreen-оверлей на `document.body` по кнопке. Контейнер редактора НЕ пересоздаётся при open/close — только re-parent, поэтому маска/история/зум переживают закрытие. Хоткеи редактора (Esc, Ctrl+Z/Y, `[`/`]` для кисти) защищены keyAnchor-паттерном ([reference_modal_hotkeys](reference_modal_hotkeys.md)). `requestRedraw` no-op при закрытом редакторе (detached container = 0×0 rect). Drag-drop и paste работают и на shell, и в оверлее; paste-роутинг: открытый оверлей владеет клипбордом, иначе — hover над shell.

Дополнительно (v9.5 fixes ранее в v9.3 батче):
- ffmpeg вызовы с таймаутами (без них процесс мог зависнуть на коррапнутом аудио).
- Upload size caps на HTTP routes.
- Memory leak fix в long-running session (per-session asyncio.Lock + versioned working files).
