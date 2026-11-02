---
name: reference_macos_compat
description: "Совместимость пака с macOS — что измерено на раннере, что починено и что осталось известным ограничением"
metadata:
  type: reference
---

Совместимость пака с macOS — что измерено на раннере, что починено и что осталось известным ограничением

Исходники темы: [ts_color_match.py](../../nodes/image/ts_color_match.py), [_audio_helpers.js](../../js/audio/loader/_audio_helpers.js), [_image_utils.py](../../nodes/image/_image_utils.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

Разбор 2026-08-08 по просьбе владельца. Инструмент —
`.github/workflows/macos_smoke.yml` (запуск только по кнопке, см.
[feedback_ci_disabled](../owner/feedback_ci_disabled.md)).

## Измерено, а не предположено

Прогон 31249806770, шаг «Probe the tensor ops the pack runs on the GPU» —
одиннадцать операторов, которые пак реально гоняет на устройстве:

- **macos-15: 9 из 11 работают.** Не реализованы на MPS ровно два:
  `aten::_linalg_eigh` и `aten::linalg_lstsq`. FFT (несущая для архитектуры
  LaMa) работает.
- **macos-14: все одиннадцать упали** с `MPS backend out of memory (MPS
  allocated: 0 bytes)`. ⚠️ Это свойство ВИРТУАЛКИ раннера, а не Apple Silicon:
  `torch.backends.mps.is_available()` там `True`, но выделить не удаётся
  ничего. Не переносить этот результат на живые маки; судить о покрытии
  операторов только по macos-15.

⚠️ Прошлое утверждение «у раннеров нет Metal» неверно: `mps available: True` на
обеих версиях.

## Починено

- **`TS_Color_Match`** (`_eigh` в `nodes/image/ts_color_match.py`): устройство
  спрашивается первым, при `NotImplementedError` разложение считается на CPU во
  float32 и возвращается обратно. Матрица 3×3 (ковариация каналов) — цена
  нулевая. `linalg.lstsq` уже был обёрнут в try/except с псевдообратной
  матрицей, а `pinv` на MPS работает.
- **Зум осциллограммы** (`js/audio/loader/_audio_helpers.js`): принимает Cmd
  наравне с Ctrl. ⚠️ На маке ctrl+колесо забирает системное увеличение экрана,
  поэтому проверка только на `ctrlKey` делала контрол недоступным. Остальной
  пак (`_lama_helpers`, `_ideogram_editor`, студия) давно считает их одним
  модификатором — это был единственный отставший.

- **`TS_CropToMask` / `TS_RestoreFromCrop`**: было `force_gpu and
  torch.cuda.is_available()` — галка молча не делала ничего на маке. Условие
  убрано: `get_torch_device()` сам отвечает `cpu`, когда ускорителя нет, и знает
  про Metal. Операции внутри — `F.interpolate`, `F.pad`, `nonzero`, `stack` — на
  MPS есть все.
- **`TS_Music_Stems`**: `"cuda"` ОСТАВЛЕН в списке (он лежит в сохранённых
  workflow как значение виджета, выкинуть = сломать их), но на машине без CUDA
  подменяется на `get_torch_device()` с предупреждением вместо `Torch not
  compiled with CUDA enabled`.
- **OOM-откаты**: `is_out_of_memory_error()` в `nodes/image/_image_utils.py` —
  общий распознаватель нехватки памяти для ЛЮБОГО бэкенда. `TS_BGRM_BiRefNet` и
  `TS_Matting_ViTMatte` ловили только `torch.cuda.OutOfMemoryError`, а Metal
  сообщает то же самое обычным `RuntimeError` («MPS backend out of memory») —
  тайловые откаты на маке не включались вовсе. ⚠️ Проверка по тексту нарочно
  узкая: голый `except RuntimeError` глотал бы настоящие ошибки и бесконечно их
  ретраил. Спеллинги CUDAMallocAsync взяты из `ts_video_depth.py`, где с ними
  столкнулись первыми.

## Известные ограничения, оставленные как есть

- **`TS_Whisper`**: `resolve_device` сознательно не знает MPS
  (`"cuda" if torch.cuda.is_available() else "cpu"`), см. [reference_device_policy](reference_device_policy.md).
  На маке всегда CPU. Не чинить вслепую: openai-whisper на MPS не идёт надёжно,
  а проверить «стало лучше» без живого мака нельзя. Цена вопроса — скорость, а
  риск — молча неверная расшифровка.
- **bitsandbytes** (int4/int8 у Qwen) не имеет колёс под arm64 macOS; движок это
  проверяет и откатывается на fp16/fp32 с предупреждением — работает как надо.
- Пути, шрифты, `float64` на устройстве, `.cuda()`, Windows-специфика в
  `nodes/` — проверено, ничего нет. `torch.load` везде с `map_location`.
  Вендорный `nodes/video_depth_anything/` имеет `device='cuda'` в сигнатурах по
  умолчанию, но нода передаёт устройство явно.

## Что раннер проверить не может

Metal-вычисления на реальном железе, скачивание моделей и собственный набор
тестов пака (`tests/` не в гите). Всё это — только на живом маке.
