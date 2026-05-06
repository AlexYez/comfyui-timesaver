# TECH_DEBT_AUDIT.md

Аудит технического долга пакета `comfyui-timesaver` (v9.1, 57 нод, всё на ComfyUI V3 API).
Read-only режим — единственный артефакт это сам файл.

## 1. Mental model

`comfyui-timesaver` — это «утилитарный комбайн» для ComfyUI: 57 нод, охватывающих
ресайз/цветокор/кеинг/тайлинг изображений, видео-апскейл и интерполяцию, аудио-загрузку
и Whisper-транскрипцию, локальный Qwen 3 VL для multimodal-промптинга, FP8-конверсию
моделей и пр. Регистрация — через `__init__.py`, который рекурсивно сканирует
`nodes/<категория>/ts_*.py`, оборачивает каждую ноду в `TSDependencyManager.wrap_node_runtime`
и печатает startup-таблицу с отчётом по загрузке и внешним импортам. Frontend —
`WEB_DIRECTORY = "./js"`, ES-модули, стабильные `app.registerExtension` ID
(`ts.bookmark`, `ts.audioLoader`, `ts.lamaCleanup`, `ts.superPrompt`, …).
Большая часть кода уважает соглашения ComfyUI (`folder_paths`, `comfy.model_management`,
`comfy.utils.load_torch_file`), но есть набор реальных проблем — сломанный CI,
heavy-imports на module-level, утечки во фронтенде, дыры в загрузках и пр.

Категория зрелости — продакшн пак с нормальной дисциплиной (V3 миграция целиком завершена,
`fingerprint_inputs` есть в 17 нодах, есть guard-обёртка вокруг runtime), но релиз 8.8/8.9
оставил несколько хвостов: гитигнор скрыл `tests/` и `tools/`, под которые рассчитан CI.

## 2. Excluded paths

- `nodes/luts/*.cube` — данные LUT (~16k строк), не код.
- `nodes/styles/img/*.png` — иконки превью стилей.
- `doc/screenshots/*.png` — скриншоты для README.
- `nodes/video_depth_anything/`, `nodes/frame_interpolation_models/` — vendored модельные
  определения (DiNoV2, IFNet, FILMNet) из upstream-репозиториев. Линтерные предупреждения по
  ним игнорируются — это чужой код.
- `__pycache__/`, `.cache/`, `.codex_tmp/`, `.idea/`.
- Файлы в `.gitignore`: `tests/`, `tools/`, `AGENTS.md`, `CLAUDE.md`, `doc/migration.md`,
  `doc/TS_DEPENDENCY_POLICY.md` — присутствуют локально, но не публикуются.

## 3. Top-5 takeaways

- CI и snapshot-валидация контрактов сломаны: workflow требует `tests/` и `tools/`,
  которые теперь `.gitignore`d (см. #1).
- Heavy module-level импорты (`transformers`, `torchaudio`, `cv2`, `matplotlib`,
  `huggingface_hub`) грузят сотни мегабайт на каждый старт ComfyUI, даже если
  юзер не использует соответствующие ноды (см. #2, #3).
- `requirements.txt` форсирует тяжёлые/платформо-специфичные пакеты (`bitsandbytes`,
  `demucs`, `pykeops`, `silero-stress`, `openai-whisper`) как hard requirements — установка
  ломается на не-CUDA / Apple Silicon / Windows-portable (см. #4).
- TS_LamaCleanup забыл подключить cleanup-функцию к `node.onRemoved` — таймеры и
  `paste`-listener утекают при удалении ноды из графа (см. #6).
- Routes `/ts_voice_recognition/transcribe` и `/ts_audio_loader/upload_recording`
  принимают неограниченные аплоады (см. #5).

---

## Findings

1. **[Critical · S] [RESOLVED in d0977ef] CI workflow ссылается на пути из `.gitignore` — каждая сборка падает**

   Files: `.github/workflows/ci.yml:39`
          `.github/workflows/ci.yml:42`
          `.gitignore:14`
          `.gitignore:17`
   What's wrong: `ci.yml` запускает `python -m pytest tests -q` и
   `python tools/build_node_contracts.py --check`, но коммитом 7fd7b58 каталог `tests/`
   стал untracked, а коммитом 21b7ff9 — каталог `tools/`. После `actions/checkout@v4`
   на runner-е этих директорий нет — pytest падает с "no tests ran" или "file not found",
   contract-check падает с `ModuleNotFoundError: tools`.
   Why it matters here: CI после `release: v9.1` (ed83db9) обязан падать на каждом push.
   Никакой гарантии, что код вообще импортируется на чистом окружении. Smoke-тесты
   `test_pack_imports.py` и contract-snapshot — главные защитные сети пакета — не запускаются.
   Recommendation: либо вернуть `tests/` и `tools/` в репозиторий (отозвать 7fd7b58 и 21b7ff9
   полностью или по `tests/` без `.cache/`), либо переписать `ci.yml` так, чтобы тесты
   жили в отдельном приватном репо и подкачивались через secret git-clone. Минимальное
   изменение: вернуть `tests/` (smoke-тесты не содержат секретов, они полезны юзерам
   для отладки своей установки).

2. **[High · M] [RESOLVED] Heavy import-on-startup в ноде Whisper грузит `transformers` + `torchaudio` при каждом запуске ComfyUI**

   Files: `nodes/audio/ts_whisper.py:9`
          `nodes/audio/ts_whisper.py:10`
          `nodes/audio/ts_music_stems.py:9`
          `nodes/llm/ts_qwen3_vl.py:16`
          `nodes/image/ts_bgrm_birefnet.py:21`
          `nodes/image/ts_bgrm_birefnet.py:23`
          `nodes/image/ts_bgrm_birefnet.py:13`
   What's wrong: `from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline`
   и `from torchaudio.transforms import Resample` стоят на module-level. Каждый старт
   ComfyUI триггерит загрузку всего `transformers` (Rust-tokenizers .so, ~50 MB кода Python),
   `torchaudio` (sox/ffmpeg adapter), `huggingface_hub` (HTTP клиент), `safetensors`,
   `cv2`. Аналогично `cv2`/`matplotlib.cm` в `ts_video_depth.py:5-6`.
   Why it matters here: CLAUDE.md §13 прямо запрещает «side effects на module-level (загрузка
   моделей, открытие файлов и т.п.)» и §14 предписывает использовать
   `TSDependencyManager.import_optional`. На machines с медленным IO/малой RAM это даёт
   1-3 секунды на голый старт ComfyUI и риск OOM-на-импорте у пользователя, который
   никогда даже не открыл ноду TSWhisper.
   Recommendation: переместить тяжёлые `from transformers import ...` внутрь `execute()`
   или внутрь lazy-helper по образцу `_load_whisper_runtime()` в
   [ts_super_prompt.py:1530](nodes/llm/ts_super_prompt.py:1530), который импортит torch
   локально. Аналогично — `cv2` и `matplotlib.cm` в `ts_video_depth.py` поднять в функции
   `preprocess_vda_internal` / `postprocess_vda_colormap_internal`.

3. **[High · S] [RESOLVED] Утечка polling intervals и paste-listener в TS_LamaCleanup при удалении ноды**

   Files: `js/image/lama_cleanup/_lama_helpers.js:1246`
          `js/image/lama_cleanup/_lama_helpers.js:218`
          `js/image/lama_cleanup/ts-lama-cleanup.js:14`
   What's wrong: Cleanup-функция `node._tsLamaCleanupCleanup` определена и вызывается
   только из `setupLamaCleanup` при повторном setup. Нигде не оборачивается
   `node.onRemoved` — при удалении ноды из графа `state.sourcePollHandle` (300ms),
   `state.modelStatusPollHandle` (1500ms), `resizeObserver` и
   `document.removeEventListener("paste", onDocumentPaste)` остаются висеть. Сравните
   с [ts-super-prompt.js:398](js/llm/ts-super-prompt.js:398) и
   [ts-style-prompt.js:508](js/text/ts-style-prompt.js:508), где `onRemoved` корректно оборачивается.
   Why it matters here: эталонная реализация интерактивной ноды, на которую ссылается
   CLAUDE.md §12.5.12, имеет реальную утечку. Каждое удаление ноды добавляет два таймера и
   один глобальный listener — за сессию редактирования это десятки утечек.
   Recommendation: в `setupLamaCleanup` обернуть `node.onRemoved` так же, как в
   `ts-super-prompt.js`:
   ```js
   const prevOnRemoved = node.onRemoved;
   node.onRemoved = function () {
       try { node._tsLamaCleanupCleanup?.(); } catch {}
       return prevOnRemoved?.apply(this, arguments);
   };
   ```

4. **[High · M] [RESOLVED] `requirements.txt` форсирует платформо-специфичные пакеты как hard deps**

   Files: `requirements.txt:2`
          `requirements.txt:7`
          `requirements.txt:16`
          `requirements.txt:17`
          `requirements.txt:18`
          `requirements.txt:19`
          `requirements.txt:20`
   What's wrong: `bitsandbytes>=0.40.0` (требует CUDA, ломает Apple Silicon / CPU-only /
   AMD), `pykeops` (компилируется через C++ runtime), `demucs`, `silero`, `silero-stress`,
   `openai-whisper` — все hard requirements. README §«Optional dependencies» (line 85-96)
   честно перечисляет их как «extras», но `pip install -r requirements.txt` всё равно
   попробует поставить.
   Why it matters here: `bitsandbytes` нужен только пользователям int4/int8 квантизации в
   `TS_Qwen3_VL_V3` — для всех остальных это лишняя ошибка установки. У пакета даже есть
   `TSDependencyManager.import_optional()`, рассчитанный ровно на этот сценарий.
   Recommendation: либо разнести `requirements.txt` на core + optional (через extras в
   `pyproject.toml`: `[project.optional-dependencies]` блоки `audio`, `llm-quant`, `video-depth`),
   либо пометить «тяжёлые» зависимости комментарием и вытащить из обязательного
   списка. Минимум — снять `bitsandbytes`, `demucs`, `silero*`, `pykeops`, `geomloss`,
   `openai-whisper`. README уже честно говорит, что эти пакеты опциональны.

5. **[High · S] [RESOLVED] Routes `/ts_voice_recognition/transcribe` и `/ts_audio_loader/upload_recording` принимают неограниченный размер аплоада**

   Files: `nodes/llm/ts_super_prompt.py:981`
          `nodes/llm/ts_super_prompt.py:1014`
          `nodes/audio/loader/_audio_helpers.py:557`
          `nodes/audio/loader/_audio_helpers.py:572`
   What's wrong: `_read_audio_upload` буферизует все chunk-и в `chunks: list[bytes]`,
   потом `b"".join(chunks)` — клиент в LAN может загнать сервер в OOM, отправив 10 GB.
   `ts_audio_loader_upload_recording` пишет на диск без капа размера и принимает
   произвольный suffix как часть имени файла (`Path(original_name).suffix.lower()`).
   Why it matters here: ComfyUI по умолчанию слушает `0.0.0.0` без auth. Два кейса —
   memory exhaustion и disk fill. Не secret leak, но реальный DoS-вектор. Также suffix
   `.exe`/`.py` пройдут — единственная проверка `len(suffix) <= 10`.
   Recommendation: добавить cap на размер (`MAX_UPLOAD_BYTES = 50 * 1024 * 1024`,
   считать `total_bytes` и возвращать 413 при превышении). Для `_audio_loader` ограничить
   suffix whitelist-ом (`SUPPORTED_AUDIO_EXTENSIONS` уже определён на line 66 — он же и
   должен валидировать suffix).

6. **[High · S] [RESOLVED] `unreachable code` в `_get_vram_gb` — `return 0.0` после if/else никогда не выполняется**

   Files: `nodes/llm/ts_qwen3_vl.py:816`
   What's wrong: после блока `if device.type == "cuda": return ...; else: return 0.0`
   стоит ещё один `return 0.0`. Vulture флагнул как 100% confidence. Скорее всего
   копипаст: автор имел в виду «default 0.0 если выпали из try», но фактически
   следующий `return 0.0` идёт после `except Exception`.
   Why it matters here: безвредно сейчас, но это сигнал, что логика VRAM-probe была
   переделана в спешке и кто-то может в неё поверить как в дефолт.
   Recommendation: удалить лишний `return 0.0` на line 816 — оставшийся
   `return 0.0` на line 819 (в конце функции) уже покрывает все exit-пути.

7. **[High · S] [RESOLVED] Subprocess вызовы ffmpeg без `timeout=`**

   Files: `nodes/audio/loader/_audio_helpers.py:244`
          `nodes/audio/loader/_audio_helpers.py:303`
          `nodes/audio/loader/_audio_helpers.py:385`
          `nodes/llm/ts_super_prompt.py:658`
          `nodes/video/ts_animation_preview.py:339`
   What's wrong: `subprocess.run([ffmpeg_exe, ...], capture_output=True, check=False)`
   без `timeout=`. Зависший / повреждённый ffmpeg-процесс блокирует весь worker-thread
   asyncio навсегда. CLAUDE.md §13 прямо запрещает «`subprocess` с пользовательским
   вводом без timeout».
   Why it matters here: пользователь грузит битый медиа-файл → ffmpeg зацикливается
   на парсинге → нода висит без шанса прервать. В случае `_audio_helpers.py:303`
   subprocess.Popen, чтение бесконечного pipe — ещё хуже.
   Recommendation: добавить `timeout=300` (5 минут — достаточно даже для
   часовых видео). Для Popen на 303 — обернуть в `try/finally` с `process.kill()`
   и `process.wait(timeout=10)`.

8. **[High · S] [RESOLVED] God-file `ts_super_prompt.py` (1731 строка) объединяет AI prompt и Whisper voice-pipeline**

   Files: `nodes/llm/ts_super_prompt.py:1`
          `nodes/llm/ts_super_prompt.py:1731`
   What's wrong: один файл содержит: (а) Whisper модель cache + voice transcription pipeline
   (line ~400-1000), (б) audio decoding + VAD (line ~640-980), (в) Qwen prompt enhancement
   (line ~1100-1600), (г) три HTTP route handler-а, (д) сам класс `TS_SuperPrompt` (line 1634-1727).
   Это вторая по размеру нода в репо.
   Why it matters here: CLAUDE.md §7 разрешает один файл = одна публичная нода (это сюда),
   но §7 также явно разрешает выносить shared логику в `_<name>.py` в той же категории. Voice-pipeline
   независим от prompt-enhancement, занимает >50% файла, и единственная причина быть в одном
   файле — что обе фичи собраны под одной UI-нодой. Тесты сложнее писать, навигация — больно.
   Recommendation: вынести Whisper/voice часть в `nodes/llm/_super_prompt_voice.py` (приватный
   модуль, по правилу `_`-prefix). Оставить в `ts_super_prompt.py`: класс, schema,
   `_generate_with_qwen`, и enhance route. Без миграции public API.

9. **[Medium · S] [RESOLVED] Frontend wildcard import + ruff F405 в audio loader**

   Files: `nodes/audio/loader/ts_audio_loader.py:24`
          `nodes/audio/loader/ts_audio_preview.py:11`
          `nodes/image/lama_cleanup/ts_lama_cleanup.py:16`
   What's wrong: `from ._audio_helpers import *` и `from ._lama_helpers import *` подтягивают
   private API через `__all__`, но конкретные используемые имена (`_normalize_selected_path`,
   `_empty_audio`, `_log_warning`, …) видимы как «may be undefined» (ruff F405) —
   статический анализ не видит откуда они приходят.
   Why it matters here: рефакторинг `_audio_helpers.py` (например, переименование
   `_log_warning`) не выдаст ошибку на месте использования — оно молча сломается в runtime.
   В `_lama_helpers.py` дополнительно перечислены явные импорты (line 17-22), но
   звёздочка остаётся.
   Recommendation: убрать `from .X import *`, перечислить явно используемые имена
   (`from ._audio_helpers import _normalize_selected_path, _empty_audio, ...`).
   ts_lama_cleanup.py уже наполовину сделал это — добить.

10. **[Medium · S] [RESOLVED] Dead function `_temporal_smooth_transforms` в `ts_color_match.py`**

    Files: `nodes/image/ts_color_match.py:264`
    What's wrong: функция `_temporal_smooth_transforms(a_list, b_list, alpha)` определена
    и нигде не вызывается. Реальное темпоральное сглаживание делается inline на line 538-540
    (`A = prev_a * _TEMPORAL_EMA + A * (1.0 - _TEMPORAL_EMA)`).
    Why it matters here: не критично, но мёртвый код вводит читателя в заблуждение
    («вот же функция для сглаживания, наверное её надо использовать») и засоряет
    самый большой не-LLM файл (670 строк).
    Recommendation: удалить функцию `_temporal_smooth_transforms` (lines 264-275).

11. **[Medium · S] [RESOLVED] Unused local `channels` в `_sinkhorn_prepare_reference`**

    Files: `nodes/image/ts_color_match.py:323`
    What's wrong: `channels = ref_img.shape[-1]` присваивается, не используется (ruff F841).
    Возможно, артефакт ранней версии или планировался валидатор «3 канала».
    Recommendation: удалить, либо превратить в проверку `assert ref_img.shape[-1] == 3,
    "Reference must be RGB"`.

12. **[Medium · S] [RESOLVED] Unused imports в `ts_silero_tts.py`**

    Files: `nodes/audio/ts_silero_tts.py:6`
           `nodes/audio/ts_silero_tts.py:7`
           `nodes/audio/ts_silero_tts.py:8`
    What's wrong: `import importlib`, `import inspect`, `import shutil` — все три unused
    (ruff F401). И каждый из них немного добавляет startup latency.
    Recommendation: удалить три строки.

13. **[Medium · M] [RESOLVED] Closure over loop variable в `_upscale_batch_load_unload`**

    Files: `nodes/video/ts_video_upscale_with_model.py:171`
    What's wrong: `lambda a: current_model(a)` замыкается на loop-переменную `current_model`,
    которая переприсваивается на line 188 (`current_model = current_model.to("cpu")`)
    и `del`-ается. Текущий вызов `comfy.utils.tiled_scale(...)` (line 169-176) синхронный
    и завершает lambda до конца итерации — поэтому в проде работает. Но это footgun:
    любая будущая попытка сделать tiled_scale асинхронным или вынести lambda за loop
    немедленно сломает upscale.
    Why it matters here: ruff B023+F821 одновременно. F821 здесь false positive (на момент
    вызова lambda переменная определена), но B023 — реальное предупреждение.
    Recommendation: сохранять locked-binding явно:
    ```python
    model_ref = current_model
    s = comfy.utils.tiled_scale(in_img, lambda a, m=model_ref: m(a), ...)
    ```
    или, проще, передать модель напрямую без лямбды через `partial`/именованную функцию.

14. **[Medium · S] [RESOLVED] `_logger.addHandler(StreamHandler)` в библиотечной ноде**

    Files: `nodes/audio/ts_whisper.py:24`
    What's wrong: модуль на module-level прикрепляет собственный `StreamHandler` к
    `comfyui_ts_whisper`-логгеру: `if not _logger.handlers: _logger.addHandler(...)`.
    Other ts-ноды (≥40) аккуратно делают только `logging.getLogger(__name__)` и оставляют
    handler ComfyUI core.
    Why it matters here: дублированный output (handler ноды + handler ComfyUI), и нет
    единой точки настройки уровня (`_logger.setLevel(logging.INFO)` на line 30 жёстко
    задано — пользователь не может переопределить через `--verbose`).
    Recommendation: удалить блок line 23-30, оставить только `_logger = logging.getLogger(_LOG_NAME)`.

15. **[Medium · S] Hardcoded `cuda` device strings обходят `comfy.model_management`**

    Files: `nodes/audio/ts_whisper.py:567`
           `nodes/audio/ts_whisper.py:571`
           `nodes/image/ts_bgrm_birefnet.py:176`
           `nodes/image/lama_cleanup/_lama_helpers.py:422`
    What's wrong: вместо `comfy.model_management.get_torch_device()` — прямые
    `torch.device("cuda")` или `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
    `_lama_helpers.py:422` хотя бы оборачивает в try/except вокруг `model_management`.
    `ts_bgrm_birefnet.py:174-176` и вовсе принудительно поднимает CUDA, если ComfyUI
    предложил CPU («force GPU even if mm said cpu»).
    Why it matters here: CLAUDE.md §8 прямо требует использовать `mm.get_torch_device()`.
    На AMD/Apple Silicon `torch.cuda.is_available() == False`, на ComfyUI с `--cpu` mm
    сознательно отдаёт CPU — переопределение его на cuda сломает в обоих случаях.
    Recommendation: оставить только `mm.get_torch_device()` без override. Если очень
    нужно прибить нод к GPU — добавить input `device: Combo("auto"/"cpu"/"cuda")` как
    у `ts_color_match.py`, не делать тихий бэйлоут.

16. **[Medium · S] `assert` для контрольной логики (Popen pipes, schema-валидация)**

    Files: `nodes/audio/loader/_audio_helpers.py:304`
           `nodes/audio/loader/_audio_helpers.py:305`
           `nodes/image/ts_qwen_safe_resize.py:50`
           `nodes/image/ts_wan_safe_resize.py:79`
    What's wrong: `assert process.stdout is not None` и `assert process.stderr is not None`
    после `subprocess.Popen(..., stdout=PIPE, stderr=PIPE)`. Эти assert-ы (а) ничего не
    делают, потому что Popen с PIPE гарантированно возвращает не-None handle, (б)
    стираются под `python -O`. CLAUDE.md §13 явно разрешает только assert для type-narrow
    в коде, но S101 в ruff цепляется по всему пакету.
    Why it matters here: assert как guard-проверка инпута юзера снимается под `-O`. Здесь
    assert-ы внутренние и безопасные, но `_qwen_safe_resize.py:50` и `_wan_safe_resize.py:79`
    — assert-ы на размеры тензоров — тут уже опаснее.
    Recommendation: assert-ы внутренние на Popen pipes удалить (line 304-305 — заменить
    на ничего, mypy и так знает что pipes не None). assert на тензоры в
    `*_safe_resize.py` — превратить в `raise ValueError(...)`.

17. **[Medium · M] [RESOLVED] Inconsistent error handling: `handle_model_error` теряет original traceback**

    Files: `nodes/image/ts_bgrm_birefnet.py:279`
           `nodes/image/ts_bgrm_birefnet.py:692`
    What's wrong: `handle_model_error(message)` на 279 делает `raise RuntimeError(message)`,
    в `execute()` (line 692) ловится generic `except Exception as e`, и
    `handle_model_error(f"Error in image processing: {str(e)}")` теряет traceback оригинала
    (нет `from e`).
    Why it matters here: when BiRefNet сломается на CUDA OOM или на повреждённом
    safetensors — пользователь видит "Error in image processing: …" без stack trace из
    PyTorch. Дебажить вслепую.
    Recommendation: `raise RuntimeError(message) from exc` в `handle_model_error`,
    или просто `logger.exception(...)` перед raise.

18. **[Medium · S] [RESOLVED] `try/except: continue` без логирования в loader/cleanup-логике**

    Files: `__init__.py:177`
           `nodes/files/ts_downloader.py:124`
           `nodes/image/lama_cleanup/_lama_helpers.py:269`
           `nodes/image/lama_cleanup/_lama_helpers.py:310`
           `nodes/image/lama_cleanup/_lama_helpers.py:692`
           `nodes/image/ts_film_emulation.py:101`
    What's wrong: ruff S112 — `try / except / continue` глушит любую ошибку. Хуже всего
    `__init__.py:177` (импорт-аудит) и `_lama_helpers.py:692` (cv2 GaussianBlur fallthrough).
    `_lama_helpers.py:692` принципиально плох — если cv2 импорт сломан, soft_mask остаётся
    жёсткой, и `feather` параметр игнорируется без сообщения.
    Why it matters here: тихие фейлы делают баги «работает, но плохо», что хуже громких
    crash-ей. Особенно `__init__.py:177` — если файл с битой кодировкой попадёт в `nodes/`,
    он silently не попадёт в audit.
    Recommendation: заменить `except Exception: continue` на
    `except Exception as exc: logger.debug(...); continue`. Минимум — debug-уровень.

19. **[Medium · S] [PARTIAL — size cap added in 1d73fe8 follow-up; full streaming refactor still pending] Unstreamed аплоад в ts_super_prompt: list[bytes] вместо stream-в-файл**

    Files: `nodes/llm/ts_super_prompt.py:981`
           `nodes/llm/ts_super_prompt.py:997`
    What's wrong: `_read_audio_upload` собирает `chunks: list[bytes]` и потом
    `b"".join(chunks)` — двойная буферизация всего payload в памяти. `_audio_loader`
    делает правильно (line 571-576 — поток на диск), `ts_super_prompt` — нет.
    Why it matters here: типичная voice-запись — 100 KB, без проблем. Но в комбинации с
    finding #5 (нет cap) — атакующий может выслать 1 GB и удвоить память Python-процесса.
    Recommendation: вместо `chunks = []` сразу `with output_path.open("wb") as handle: ...
    handle.write(chunk)`, потом read из файла когда нужно.

20. **[Medium · S] [PARTIAL — `__init__.py` getattr fixed; vendored video_depth_anything mutable defaults left as upstream] `B009`/`B005` getattr с константой и mutable defaults в video_depth_anything**

    Files: `__init__.py:209`
           `__init__.py:216`
           `nodes/video_depth_anything/dpt.py:53`
           `nodes/video_depth_anything/dpt_temporal.py:27`
           `nodes/video_depth_anything/video_depth.py:40`
    What's wrong: `getattr(module, "NODE_CLASS_MAPPINGS")` (line 209) — константный
    атрибут, надёжнее `module.NODE_CLASS_MAPPINGS` (исключение поднимется само,
    тратится один лишний вызов). И mutable default `[256, 512, 1024, 1024]` в DPT
    (vendored код — менять не надо, vendor upstream).
    Why it matters here: первое — мелочь стиля; видеодетекшн mutable defaults — vendored из
    `Depth-Anything-Video`, упомянуть как known issue. Менять не нужно.
    Recommendation: для `__init__.py` — `module.NODE_CLASS_MAPPINGS`. Для vendored
    директорий (`video_depth_anything/`) — оставить как upstream.

21. **[Medium · M] Inconsistent node_id naming style — снимать нельзя**

    Files: `nodes/files/ts_downloader.py:48` (`"TS Files Downloader"` с пробелами)
           `nodes/files/ts_edl_chapters.py:15` (`"TS Youtube Chapters"`)
           `nodes/image/ts_cube_to_equirect.py:15` (`"TS Cube to Equirectangular"`)
           `nodes/image/ts_equirect_to_cube.py:15` (`"TS Equirectangular to Cube"`)
           `nodes/audio/ts_whisper.py:734` (`"TSWhisper"` без подчёркивания)
           `nodes/image/ts_crop_to_mask.py:26` (`"TSCropToMask"`)
    What's wrong: 4 ноды используют `node_id` с пробелами (ломает `gh search`,
    автокомплит, JSON-инструменты), 4 используют camelCase без подчёркивания, остальные
    PascalCase с подчёркиванием. Идентификатор закреплён в saved workflows.
    Why it matters here: CLAUDE.md §6 явно перечисляет эти ноды как «existing исключения,
    не переименовываем». Это compatibility-debt. Любая нормализация ломает workflow,
    которые юзеры успели сохранить.
    Recommendation: документировать как known issue в `doc/AGENTS.md` или README,
    чтобы новые ноды не повторяли паттерн. Не трогать существующие. Если когда-то
    решите выровнять — нужен alias через `search_aliases` + `NodeReplace`-механизм V3.

22. **[Medium · S] [RESOLVED] BGRM module-level side effect: `folder_paths.add_model_folder_path` при импорте**

    Files: `nodes/image/ts_bgrm_birefnet.py:29`
           `nodes/image/lama_cleanup/_lama_helpers.py:75`
           `nodes/image/lama_cleanup/_lama_helpers.py:87`
    What's wrong: `folder_paths.add_model_folder_path("birefnet", os.path.join(folder_paths.models_dir, "BiRefNet"))`
    выполняется при `import nodes.image.ts_bgrm_birefnet`. Аналогично `_register_model_folder()`
    в lama_cleanup. CLAUDE.md §13 запрещает «side effects на module-level».
    Why it matters here: в lama_cleanup это нужно для интеграции `extra_model_paths.yaml`
    и сделано в private helper-е c try/except — относительно безопасно. В bgrm_birefnet
    — голый вызов без try/except, любая поломка `folder_paths` (например, monkey-patch от
    другого custom_node) уронит загрузку всего пакета.
    Recommendation: обернуть `add_model_folder_path` в try/except как в `_register_model_folder()`
    из lama_cleanup. Сам side-effect оставить — это accepted convention для интеграции
    `extra_model_paths.yaml`.

23. **[Medium · S] [RESOLVED] `models/` имя совпадает с core: `add_model_folder_path("birefnet", ...)` — ок, но `models_dir/BiRefNet` хардкод**

    Files: `nodes/image/ts_bgrm_birefnet.py:29`
    What's wrong: путь жёстко `os.path.join(folder_paths.models_dir, "BiRefNet")`.
    Если у пользователя ComfyUI запущен с `--models-dir /alt/path/`, оно подхватится через
    `folder_paths.models_dir`, но `extra_model_paths.yaml`-конфигурация отдельной папки для
    `birefnet` будет проигнорирована — пакет всегда добавит SCP-default. lama_cleanup сделан
    лучше: использует `folder_paths.get_folder_paths(MODEL_FOLDER_NAME)` для уже зарегистрированных.
    Recommendation: следовать паттерну `_lama_helpers.py:443-450` — сначала
    `get_folder_paths`, только если пусто — добавить дефолт.

24. **[Medium · S] Качество тестов: тесты содержат stub-ы V3 API, но CI не запускает их**

    Files: `tests/test_super_prompt_contract.py`
           `tests/test_pack_imports.py`
           `.github/workflows/ci.yml:39`
    What's wrong: тесты разработаны корректно — есть paste-stub `comfy_api.latest`,
    `folder_paths`, `aiohttp`, есть smoke-тест на дискавери модулей. Но (а) каталог
    `tests/` теперь `.gitignore`d → CI на чистом checkout-е тестов не видит,
    (б) `test_pack_imports.py` единственный тест, не требующий torch/numpy, но
    остальные 14 тестов скипаются под обычным Python без ML-стека.
    Why it matters here: пакет растёт, V3 миграция была масштабной, все 57 нод
    могут регрессировать без шума. Сейчас единственная защита — что нода падает в
    runtime у первого пользователя.
    Recommendation: вернуть `tests/` в репо (см. #1) и в `ci.yml` гарантировать
    запуск как минимум `test_pack_imports.py` (он не требует heavy deps).

25. **[Medium · M] [RESOLVED] Класс `TS_DependencyManager.fallback_value_for_type` — dead-path для V3**

    Files: `ts_dependency_manager.py:56`
           `ts_dependency_manager.py:88`
           `ts_dependency_manager.py:96`
           `ts_dependency_manager.py:115`
    What's wrong: `build_v1_fallback_output` использует `RETURN_TYPES`, которого нет ни
    у одной V3 ноды. Метод вызывается из `_wrap_plain_method` и `_wrap_class_method`,
    которые выбираются если `getattr(node_cls, "FUNCTION", None)` truthy (V1). После
    того как все 57 нод стали V3 в 8.9 — этот код мёртв.
    Why it matters here: 60% `TSDependencyManager` (V1 fallback path) больше не выполняется.
    Не баг, но мёртвый код важной инфраструктуры.
    Recommendation: либо удалить V1 fallback функции (`fallback_value_for_type`,
    `build_v1_fallback_output`, `_wrap_plain_method`, `_wrap_class_method`), оставив
    только `_wrap_execute_classmethod`. Либо явно задокументировать «kept for legacy
    custom-node плагинов, которые могут регистрироваться через TSDependencyManager».
    CLAUDE.md уже говорит "V1 шаблон оставлен только как reference" — следовать этой строке.

26. **[Low · S] [RESOLVED] Кодстайл: trailing whitespace, blank-line whitespace, f-string без placeholder, разное**

    Files: `nodes/image/ts_bgrm_birefnet.py:287` и далее (W293)
           `nodes/llm/ts_qwen3_vl.py:229` (F541 — f-string без placeholder)
           `nodes/llm/ts_qwen3_vl.py:764-778` (E701 — несколько statement-ов в одной строке)
           `nodes/audio/loader/_audio_helpers.py:46` (F401 — `IO` импортится не используется)
    What's wrong: косметика, но накопилась. ruff `--fix` решит большую часть автоматически.
    Why it matters here: CI должен ловить это (ruff в `.github/workflows/ci.yml:48` стоит
    `continue-on-error: true`). Сейчас баг и стиль плывут вместе.
    Recommendation: один проход `python -m ruff check . --fix --unsafe-fixes`, потом
    снять `continue-on-error: true` в CI. Не делать в одном PR с другими изменениями.

27. **[Low · S] `bare except` глушит ошибки в стартап-аудите**

    Files: `__init__.py:177`
    What's wrong: `try: content = py_file.read_text(...) except Exception: continue`. Файл
    с битой кодировкой пропустится без следа. Сейчас не воспроизводимо, но если что-то
    битое попадёт в `nodes/` — диагностировать будет сложно.
    Recommendation: ловить узкие `(OSError, UnicodeDecodeError)` и логировать `logger.warning`.
    См. #18.

28. **[Low · S] `random.random` в `ts_prompt_builder.py:324` для рандомизации блоков**

    Files: `nodes/text/ts_prompt_builder.py:324`
    What's wrong: ruff S311 «non-cryptographic random». Здесь это false positive —
    рандом нужен для перетасовки prompt-блоков, не для секретов.
    Recommendation: добавить `# noqa: S311 — not cryptographic` или, чище, использовать
    `random.Random(seed)` явно (контролируемый seed для воспроизводимости workflow).

29. **[Low · S] Wildcard import + `__all__` в `_lama_helpers.py` дублирует public surface**

    Files: `nodes/image/lama_cleanup/_lama_helpers.py:10-22`
           `nodes/image/lama_cleanup/ts_lama_cleanup.py:16`
    What's wrong: `__all__` перечисляет 12 имён, `ts_lama_cleanup.py` импортит звёздочкой
    + дополнительно явно перечисляет 4 (`_load_image_tensor` и т.д.). Дублирование
    делает рефакторинг рискованнее (придётся править оба места).
    Recommendation: убрать `from ._lama_helpers import *`, оставить только явный
    `from ._lama_helpers import (...)` — тогда `__all__` тоже можно удалить (ничто
    его не читает). См. #9.

30. **[Low · S] `xformers.ops.fmha` импортится но не используется в vendored DiNoV2 attention**

    Files: `nodes/video_depth_anything/dinov2_layers/attention.py:21`
           `nodes/video_depth_anything/motion_module/motion_module.py:17`
    What's wrong: ruff F401 для `xformers.ops` импорта без использования. Это
    upstream-код, помеченный «keep for compatibility».
    Recommendation: оставить как есть — vendored код. Не править. Документировать
    в `doc/AGENTS.md` исключения для `nodes/video_depth_anything/`.

31. **[Low · S] [RESOLVED] `print()` для startup-таблицы вместо logger.info**

    Files: `__init__.py:301-332`
    What's wrong: 17 `print()` вызовов в `_print_startup_report`. CLAUDE.md §13 запрещает
    `print()` для логирования.
    Why it matters here: ComfyUI core пишет startup-сообщения через logger; смешивать
    print + logger даёт неконсистентный output (порядок строк может быть перепутан при
    буферизации stdout).
    Recommendation: переключить на `logger.info(...)` (logger уже создан на line 15).
    `_render_table` оставить как есть — он вернёт строку. Просто `logger.info(table_string)`.

32. **[Low · S] [RESOLVED — moved to llm-quant extra] `bitsandbytes` упомянут в `pyproject.toml` deps, но условно загружается через `_is_bitsandbytes_available`**

    Files: `pyproject.toml:8`
           `nodes/llm/ts_qwen3_vl.py:70`
    What's wrong: hard dep `bitsandbytes>=0.40.0`, но `_is_bitsandbytes_available()` сам
    делает gracefully fallback. Если bitsandbytes не установился (а на macOS он сразу
    fail-ит), нода всё равно работает в `auto`/`fp16`/`bf16`, только без int4/int8.
    Why it matters here: это финдинг #4 в другом ракурсе — pyproject.toml форсит
    установку того, что нода умеет обходить. Удвоенный финдинг.
    Recommendation: см. #4 — перенести в `[project.optional-dependencies] llm-quant = ["bitsandbytes>=0.40.0"]`.

33. **[Low · S] Hardcoded modelnames в `ts_qwen3_vl.py._MODEL_LIST`**

    Files: `nodes/llm/ts_qwen3_vl.py:31-42`
           `nodes/llm/ts_qwen3_vl.py:43-53`
    What's wrong: 9 моделей-списков (Qwen3-VL и форки huihui-ai/prithivMLmods) пинятся
    в код. Любое появление новой модели требует выпуска новой версии пакета. Default
    `hfmaster/Qwen3-VL-2B` — не upstream Qwen, а форк (что неочевидно из README,
    который говорит «Qwen 3 VL»).
    Why it matters here: dropdown устаревает быстро. Юзер вынужден писать в
    `custom_model_id`, но default ведёт его к стороннему форку.
    Recommendation: либо вынести список в `qwen_3_vl_models.json` рядом с
    `qwen_3_vl_presets.json` (чтобы юзер мог дополнять не пересобирая пакет), либо
    хотя бы документировать в README, что `hfmaster/Qwen3-VL-2B` — community fork,
    не official Qwen.

34. **[Low · S] `test_pack_imports.py` использует `exec(compile(source, ...))` для namespace-injection**

    Files: `tests/test_pack_imports.py:36`
    What's wrong: вместо нормального `importlib.import_module` тест читает source файл и
    исполняет в синтетическом namespace. Это работает, но (а) ruff S102 на `exec`,
    (б) непереносимо на pytest-collect параллельно.
    Why it matters here: видимо, единственный способ обойти auto-load в `__init__.py`,
    который импортит тяжёлые ноды на module-level. Если применить #2 (lazy imports) —
    `import_module` станет безопасным.
    Recommendation: после lazy-import рефакторинга по #2 — переписать тест на
    `importlib.import_module("comfyui_timesaver")` без exec.

35. **[Low · S] Wildcard re-export в `nodes/audio/loader/__init__.py` и аналогах**

    Files: `nodes/audio/loader/__init__.py`
           `nodes/image/lama_cleanup/__init__.py`
           `nodes/image/keying/__init__.py`
    What's wrong: эти `__init__.py` в подкатегориях — пустые или с `from .X import *`. Loader
    `__init__.py:33` явно пропускает `__init__.py`, так что они никак не используются.
    Recommendation: оставить пустыми, либо удалить вообще (Python 3.10 поддерживает
    namespace packages). Не критично.

36. **[Low · S] `nodes/_shared.py` использует `_logger = logging.getLogger("comfyui_timesaver.ts_shared")`, остальные ноды — `logging.getLogger(__name__)`**

    Files: `nodes/_shared.py:9`
           ~30 `ts_*.py` файлов с `logging.getLogger("comfyui_timesaver.ts_<name>")`
           ~8 файлов с `logging.getLogger(__name__)`
    What's wrong: ~38 нод используют hardcoded имя логгера, ~8 — `__name__`. Несовместимо
    с настройкой `logging.config` через словарь, где обычно матчится по dotted path
    `nodes.image.ts_color_match`.
    Recommendation: унифицировать на `logging.getLogger(__name__)` — это даст естественный
    префикс `comfyui_timesaver.nodes.image.ts_color_match` без хардкода. Без
    миграции public API.

37. **[Low · S] Test fixture `ts_tmp_path` дублирует встроенный `tmp_path`**

    Files: `tests/conftest.py:21-31`
    What's wrong: документировано как обход OS-policy, что блокирует tmp под
    `%LOCALAPPDATA%\Temp`. Это специфика конкретной машины, а не репо.
    Recommendation: если tmp_path не работает — это ОС-настройка тестового хоста.
    Возможно, лучше задокументировать в README для разработчиков, а не дублировать
    fixture для каждого юзера. Не блокер.

38. **[Low · S] Несколько похожих `_resolve_prompt_server` в трёх разных файлах**

    Files: `nodes/audio/loader/_audio_helpers.py:91`
           `nodes/image/lama_cleanup/_lama_helpers.py:102`
           `nodes/llm/ts_super_prompt.py:175`
    What's wrong: одна и та же функция `_resolve_prompt_server`, идентичные
    `_register_get` / `_register_post` декораторы — три копии.
    Why it matters here: minor — рутинный boilerplate. Если PromptServer API однажды
    изменится, нужно править 3 места.
    Recommendation: вынести в `nodes/_shared.py` как `register_aiohttp_routes(prefix)`
    helper. Файлы стали бы на ~30 строк короче каждый. Не блокер, но
    разумный «refactor opportunity» если кто-то всё равно туда лезет.

39. **[Low · S] README не упоминает `models/lama/` и BiRefNet репозиторий — ссылка ведёт на `1038lab/BiRefNet`, не upstream**

    Files: `README.md:712-722`
           `nodes/image/ts_bgrm_birefnet.py:34`
    What's wrong: README §«Where do model files live?» говорит `models/birefnet/` (?
    проверить), но MODEL_CONFIG в коде использует `repo_id="1038lab/BiRefNet"` —
    это форк/зеркало, не оригинальный `ZhengPeng7/BiRefNet`. Не баг, но непрозрачно.
    Recommendation: задокументировать в README, что BiRefNet веса берутся из зеркала
    `1038lab/BiRefNet`. README говорит про upstream `BiRefNet` (ZhengPeng7) — пользователь
    может ожидать что веса тянутся оттуда.

40. **[Low · S] Внутренние `logger = logging.getLogger("comfyui_timesaver.ts_*")` без `LOG_PREFIX` константы**

    Files: разное (несколько ts-нод)
    What's wrong: некоторые ноды (`ts_color_match`, `ts_video_depth`) определяют
    `_LOG_PREFIX = "[TS X]"` и используют его в каждом log-вызове. Другие
    (`ts_color_grade`, `ts_film_emulation`) этого не делают, и логи выглядят
    `INFO comfyui_timesaver.ts_color_grade: started` без `[TS Color Grade]` префикса.
    Recommendation: соблюсти CLAUDE.md §15 единообразно. Шаблон есть в `_shared.py`
    через `TS_Logger.log(node_name, message)` — можно использовать его. Не критично.

---

## Quick wins

Низкий effort, средняя/высокая важность:

- [ ] #6 — удалить лишний `return 0.0` в `ts_qwen3_vl.py:816`.
- [ ] #10 — удалить `_temporal_smooth_transforms` (~12 строк мёртвого кода).
- [ ] #11 — удалить unused `channels` local в `ts_color_match.py:323`.
- [ ] #12 — удалить три unused import в `ts_silero_tts.py:6-8`.
- [ ] #14 — удалить дополнительный `StreamHandler` в `ts_whisper.py:24-30`.
- [ ] #18 — заменить `try/except: continue` на `try/except: log.debug; continue` (5 мест).
- [ ] #22 — обернуть `add_model_folder_path("birefnet", ...)` в try/except.
- [ ] #26 — `ruff check --fix` для тривиальных стилевых ошибок.
- [ ] #31 — заменить `print()` на `logger.info()` в `__init__.py:301-332`.

## Things that look bad but are actually fine

- **`fingerprint_inputs` возвращает `(mtime,)` в `ts_qwen3_vl.py:114-117`** — выглядит как
  «кеш не инвалидируется при смене prompt/seed/image», но ComfyUI core combine-ит
  fingerprint c hash-ом всех input-аргументов в `comfy_execution/caching.py:116`
  (`signature.append((key, inputs[key]))`). Так что mtime — additional state, а не
  единственный ключ. Менять не нужно.
- **`fingerprint_inputs` возвращает `float("nan")` в `ts_resolution_selector.py:51`** — это
  правильный идиом «всегда пересчитывать» для случая когда есть image-input. NaN ≠ NaN
  по семантике IEEE 754, поэтому signature всегда новый.
- **`_register_model_folder()` на module-level в `_lama_helpers.py:87`** — выглядит как
  «side effect на import», но это accepted паттерн ComfyUI для интеграции с
  `extra_model_paths.yaml`. Без этого юзер не сможет переопределить путь к LaMa-модели.
- **closure `lambda a: current_model(a)` в `ts_video_upscale_with_model.py:171`** —
  ruff F821+B023 ругается, но в текущем синхронном паттерне через `comfy.utils.tiled_scale`
  всё работает корректно (lambda вызывается до рекуссии iteration). Однако сделал finding #13
  потому что это footgun на будущее.
- **wildcard `from ._lama_helpers import *` в `ts_lama_cleanup.py:16`** — модуль определяет
  `__all__` (line 10-22), что ограничивает экспорт. Технически не «голый» wildcard. Но #9
  всё равно стоит — для удобства рефакторинга.
- **`folder_paths.add_model_folder_path("birefnet", ...)` на module-level в
  `ts_bgrm_birefnet.py:29`** — да, side effect, но это требуется для интеграции с
  `extra_model_paths.yaml` (см. #22 для меньшей фиксации — обернуть в try/except).
- **Тяжёлые requirements типа `transformers>=4.57.0`, `accelerate>=0.21.0`** — CLAUDE.md
  не запрещает их, и они нужны TSWhisper / TS_Qwen3_VL_V3. Это не «лишние пины», а реальные
  зависимости. Финдинг #4 — про **bitsandbytes**, **demucs**, **silero**, **pykeops** —
  единственные, что должны быть optional. transformers пусть остаётся обязательной.
- **`folder_paths.models_dir` хардкод в `ts_bgrm_birefnet.py:29`** — не баг сам по себе:
  ComfyUI старается чтобы `folder_paths.models_dir` был source-of-truth (см. #23).
- **Большой файл `nodes/audio/ts_whisper.py` (1052 строки)** — приемлемо: одна нода, много
  логики (preprocessing, VAD, generation, SRT/TTML output, retry-on-OOM). Не god-file,
  это специализированный compoundный transcription pipeline, корректно держится в одном
  модуле по правилу #7 CLAUDE.md.
- **PRINT() в `__init__.py` — startup table** — финдинг #31 предлагает logger, но `print`
  это сознательный выбор автора (CLAUDE.md упоминает «startup-таблицу»). Мнение —
  заменить на logger всё равно. Но это не баг.

## Open questions

- Был ли `tests/` исключён из репо намеренно (чтобы ComfyUI Manager не качал лишнего юзерам)
  или это случайность? Если намеренно — почему CI не отключили тогда же? Похоже на «забыли
  убрать шаги CI после `chore: untrack tests/ folder` (7fd7b58)».
- Why `node_id="TSWhisper"` без подчёркивания, тогда как 50 других нод используют
  `TS_*`? CLAUDE.md упоминает это как «existing исключение» — это исторический долг или
  есть deeper reason? Просто хочется убедиться, что переименовывать нельзя (saved
  workflows users).
- Почему default Qwen 3 VL модель — это форк `hfmaster/Qwen3-VL-2B`, а не upstream
  `Qwen/Qwen3.5-2B`? Может быть размер отличается (что критично для пользователей с
  ограниченной VRAM).
- `nodes/llm/ts_qwen3_vl.py:114-117` — `fingerprint_inputs` возвращает `(mtime,)`. Авторы
  знали, что ComfyUI core combine-ит это с input-args для cache key, или это unintended
  behavior? Если последнее — нужны более полные fingerprint в других нодах.
- Поддерживается ли `extra_model_paths.yaml` для `birefnet` папки, или ComfyUI использует
  только захардкоженный `models_dir/BiRefNet`? (См. #23.)
- Тесты CI должны иметь shape-asserts на критичных tensor-операциях (color_match, keyer,
  bgrm). Сейчас защиты нет. Это намеренная инвестиция «потом» или просто оставлено?
