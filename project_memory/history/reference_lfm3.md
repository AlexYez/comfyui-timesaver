---
name: reference_lfm3
description: "История: LFM3 VL and Omni retired at owner request; VL weights recycled on 2026-11-01"
metadata:
  type: reference
  status: historical
---

История: LFM3 VL and Omni retired at owner request; VL weights recycled on 2026-11-01

Итог последующих событий: 12.13.4 уже опубликована [коммитом 212dcb8](https://github.com/AlexYez/comfyui-timesaver/commit/212dcb8), а затем выпущена 12.13.6. Фраза ниже о приостановленном push относится к промежуточному состоянию 2026-11-01. Снятая экспериментальная LFM3 в Git не публиковалась; её локальные zip не входят в переносимый репозиторий. Сохранено решение об отказе и причины, а не обещание восстановить экспериментальный исходник из клона.

**Историческая запись, не описание текущей реализации и не действующее разрешение.** Даты и результаты ниже относятся к указанным сессиям. Пути `tests/`, `tools/`, `dist/` и `archive/` с артефактами — локальная лаборатория; они не гарантированы в чистом клоне.

## Текущее состояние — LFM3 VL снята 01.11.2026

Владелец отказался и от VL после сравнения. Удалены из DEV и установленного
пака пять файлов `nodes/llm/*lfm3*.py`, справка `TS_LFM3_VL`, локаль,
разделы README и неопубликованного changelog. Контрактов теперь 88, LLM-нод 4.
Исходники и снятый `tests/test_lfm3.py` сохранены вне автозагрузки:
`tests/.cache/retired-lfm3-vl-20261101/{dev,installed}.zip`.
В общем `test_free_memory.py` убрана регистрация уже снятого движка.

Только VL-веса `models/LLM/LFM3/saidutta69--LFM3-VL-2B-heretic-467d0ee4`
отправлены в корзину Windows через IFileOperation с RECYCLEONDELETE и ADDUNDORECORD:
54 файла, 3 854 519 408 байт. Проверены отсутствие исходной папки и полное
совпадение размера файлов в возвращённой Windows папке корзины.
Квитанция: `tests/.cache/retired-lfm3-vl-20261101/recycle-receipt.json`.
Omni-веса и отдельный `.ts_lfm3_runtime` не тронуты; результаты сравнений сохранены.

42 целевых теста прошли: free_memory, node_contracts, widget_order_contract,
pack_imports, locale_nodedefs. Удаление из исходников обеих копий проверено.
15 пресетов сохранены, Erotism совпадает с HEAD. Активного LFM worker не было.
Сервер НЕ перезапускался: нода исчезнет из текущего меню при следующем
перезапуске владельцем. Патч/push по слову «стой» остаются остановленными;
локальная версия 12.13.4 не опубликована, HEAD 2a571c51120131d7ffac94134efbd9d3a76eeca8.

## Историческое состояние после уточнения владельца, 31.10.2026

Оставлена только `TS_LFM3_VL`. Владелец отказался от Omni для этой задачи:
публичная нода, документация, locale и аудио-ветки общего worker удалены
из исходного и установленного пака. Веса Omni с диска не удалялись.
Старые файлы сохранены в `tests/.cache/retired-lfm3-omni/`, вне автозагрузки.
Контрактов теперь 89, после удаления прошли 48 целевых тестов, Ruff и compileall.
В рамках ранее прямо разрешённого перезапуска сервер обновлён; API VL есть,
Omni не зарегистрирована. Других перезапусков без команды не делать.

Для последующего сравнения пользователь явно запретил сжатие LFM3 VL и Qwen:
обе проверены в BF16. Публичный выбор int4/int8 в ноде остаётся доступным,
но для этого сравнения не использовался. См. [результаты](reference_lfm3_vl_peer_comparison.md).

## История первоначальной реализации (Omni ниже — уже удалённая версия)

31.10.2026 пользователь попросил две мультимодальные uncensored-ноды по образцу
Qwen/Gemma, с точностью и выгрузкой. Текстовую LFM3-1.7B он ранее отменил;
она НЕ нужна, вместо неё сделаны `TS_LFM3_VL` и `TS_LFM3_Omni` (`TS/LLM`).

- VL: `saidutta69/LFM3-VL-2B-heretic`, sha `467d0ee442642452755b3c3a6bfbc64e437d5b0d`.
- Omni: `OpenIntelligenceNet/LFM3-Omni-2B-Uncensored`, sha `23b95a548729b23363732625d2f40f05f5c9a55f`.
- Веса лежат в `models/LLM/LFM3/<repo-with-double-hyphens>-<sha8>/`.
  Обе модели скачаны на машине владельца, Omni включает codec (~147 MB).
- Выход обеих нод — STRING. Omni напрямую принимает AUDIO, но генерацию
  речи не экспонирует. VIDEO-вход здесь IMAGE batch + исходный fps, как Qwen.
  До 8 картинок; видео ограничивается равномерной выборкой, звук — началом
  клипа с явным `audio_max_seconds`. Видео и звук — отдельные media spans,
  не синхронная interleaved-дорожка.

Архитектура:

- Публичные схемы в `ts_lfm3_vl.py`, `ts_lfm3_omni.py` — явные IO.Schema,
  чтобы AST-снимок видел порядок/defaults. Не уносить их в общий конструктор.
- `_lfm3_config.py`: pin-модели и инструкции. `_lfm3_node.py`: адаптер ComfyUI.
- `_lfm3_runtime.py`: один процесс и блокировка на обе ноды. При смене модели,
  точности или устройства процесс закрывается. По умолчанию выгрузка включена.
  Зарегистрирован `register_memory_release("lfm3", ...)` для общей кнопки.
- `_lfm3_worker.py`: отдельный интерпретатор, JSON + PNG/NPY IPC (не pickle).
  Личные transformers 5.21.0 / hub 1.31.0 / tokenizers 0.23.1 / safetensors 0.8.0
  ставятся pip --target --no-deps в `models/LLM/.ts_lfm3_runtime/py312-tf521/`.
  PyTorch и остальные зависимости — хостовые. Main ComfyUI Transformers остался 5.4.0.
- Для embedded Python PYTHONPATH игнорируется, поэтому private sys.path
  выставляется bootstrap-кодом в ДОЧЕРНЕМ процессе, до любого HF import.
  Для разработки можно задать `TS_LFM3_RUNTIME` с абсолютным private-path.

Реально найденные грабли:

1. Старая host PEFT 0.17.0 импортирует удалённый `HybridCache`, ломая TF 5.21.
   В worker main `sys.modules["peft"] = None`: полные веса не требуют PEFT.
   Родительский процесс не затрагивается.
2. Новый hub несовместим со старым hf_xet. Worker задаёт
   `HF_HUB_DISABLE_XET=1`, обычный HTTPS работает. Не обновлять hf_xet глобально.
3. **Квантизация: исключения должны быть `model.vision_tower`,
   `model.multi_modal_projector`, `model.audio_encoder`, `model.audio_projector`,
   `model.depth_decoder`, а не просто vision_tower!** Новый BNB-фильтр не
   исключает потомков по голому имени. С неправильным фильтром VL int8 видел
   красную картинку фиолетовой; после исправления видит красной.
   Есть runtime-проверка: все bnb Linear обязаны быть внутри `model.language_model.`.
4. `video_metadata` нужен процессору для timestamp-плейсхолдеров; в generate
   Omni его передавать нельзя — строгая проверка model_kwargs. После обработки
   metadata удаляется, video_grid_thw/pixel_values_videos остаются.

Проверено под ComfyUI Python 3.12.9 на RTX 3080 Ti Laptop 16GB:

- 59 targeted tests: `test_lfm3`, free_memory, node_contracts,
  widget_order_contract, locale_nodedefs, pack_imports. Без полного pytest.
- Реальная генерация VL BF16/FP16/FP32/int4/int8; картинки и видео.
- Реальная генерация Omni BF16/int4/int8; картинки, видео, звук 16 kHz,
  совместный вход картинки и стерео 44.1 kHz с resample.
- auto VL на CPU ответил READY; два auto CUDA-запроса использовали один PID,
  второй ответ на новую картинку вышел за ~0.34s (синтетический smoke, НЕ benchmark).
- Общая release-функция закрывает PID; GPU вернулась к baseline 169 MiB.
- Проверки синтетические, не доказывают качество реальных промптов,
  русского аудио или временного анализа видео. MPS не проверен.
- Полный прогон через живой граф после рестарта не делался: сервер НЕЛЬЗЯ
  перезапускать без прямой команды владельца (`feedback_never_restart_comfyui.md`).

Результаты live-проверок — локально `tests/.cache/lfm3_live_results*.json`,
`lfm3_lifecycle_results.json`. Общий locale-guard нашёл два старых отсутствующих
перевода у H3FastVAEDecode (`latent`, `vae_name`); добавлены только подписи.
