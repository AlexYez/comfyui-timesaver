---
name: reference_voice_pipeline
description: "Поведение Whisper-пайплайна в TS_SuperPrompt — hysteresis VAD, anti-hallucination фильтр, async record-button; всё настраивается константами в _helpers.py"
metadata:
  type: reference
originSessionId: acf91186-c754-4d51-9f22-9e2427ec1fd3
---

Поведение Whisper-пайплайна в TS_SuperPrompt — hysteresis VAD, anti-hallucination фильтр, async record-button; всё настраивается константами в _helpers.py

Исходники темы: [_voice.py](../../nodes/llm/super_prompt/_voice.py), [ts-super-prompt.js](../../js/llm/ts-super-prompt.js), [_helpers.py](../../nodes/llm/super_prompt/_helpers.py), [ts_super_prompt.py](../../nodes/llm/ts_super_prompt.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

TS_SuperPrompt voice recognition (Whisper) — `nodes/llm/super_prompt/_voice.py` + `_helpers.py` + JS `js/llm/ts-super-prompt.js`. С v9.7 пайплайн содержит три не-дефолтных Whisper-фикса.

## Audio preprocessing pipeline

С v9.11 **boundary trim отключён** (`AUDIO_TRIM_ENABLED = False`) — full clip уходит в Whisper. Причина: даже с hysteresis VAD + padding 0.40s edge cuts периодически глотали первый/последний слог. Whisper сам корректно работает с leading/trailing silence для коротких prompt-style фраз; trim приносил больше вреда чем пользы.

Что **остаётся активным**:
- `AUDIO_VAD_ENABLED = True` — VAD detection всё ещё гоняется, но используется только для skip пустых записей (быстрый return пустой строки без waste GPU). Boundaries вычисляются и кладутся в metadata, но не применяются для нарезки.
- `AUDIO_NORMALIZE_ENABLED = True` — peak normalization до `AUDIO_NORMALIZE_TARGET_PEAK (0.92)` с cap `AUDIO_NORMALIZE_MAX_GAIN_DB (12)`. Помогает Whisper с тихими записями.
- Edge fade применяется к самому началу/концу clip'а (6ms) — на случай резкого старта/стопа micrecorder'а.

## Hysteresis VAD (используется в detection, но не для trim)

`_detect_speech_bounds` использует **два порога** вместо одного:
- `high = noise_floor × AUDIO_VAD_ADAPTIVE_MULTIPLIER (2.2)` — детекция core speech.
- `low = max(noise_floor × AUDIO_VAD_LOW_MULTIPLIER (1.3), AUDIO_VAD_RMS_THRESHOLD × 0.55)` — расширение границ наружу через тихие consonants (тихие предлоги «с», «в», «к», «у», окончания слов).
- low всегда clamp ≤ high.
- После hysteresis-расширения добавляется `AUDIO_VAD_PADDING_SEC (0.40)` safety margin.
- Edge fade `AUDIO_EDGE_FADE_MS (6)` — снижен с 12 чтобы не «подъесть» короткий consonant если trim когда-то снова включат.

`_adaptive_vad_thresholds(rms) -> (high, low)` — единая точка для расчёта порогов. Если меняешь логику — правь там, не размазывай.

**Чтобы вернуть trim:** `AUDIO_TRIM_ENABLED = True` в `_helpers.py`. Тест `test_audio_preprocess_trim_when_explicitly_enabled` подтверждает что код-путь рабочий.

## Hallucination filter (post-transcription)

Whisper обучался на YouTube-субтитрах и галлюцинирует «outro»-фразы при затихании аудио. `_strip_whisper_hallucinations` вырезает такие фразы **только в конце текста** (anchored `$`), терпит `.`/`…`/`!`/`?`/`,`/тире.

- `WHISPER_HALLUCINATION_FILTER_ENABLED = True` (можно отключить для отладки).
- `WHISPER_HALLUCINATION_PATTERNS = (r"продолжение\s+следует",)` — список регэкспов; пополняй по мере обнаружения новых галлюцинаций (типичные кандидаты: «спасибо за просмотр», «подписывайтесь», «thanks for watching»).
- Применяется ПОСЛЕ `_collapse_repeated_phrases` в `_clean_transcription_text`.

## Near-duplicate sentence collapse (Whisper repeats phrase 2x with inflection)

Симптом пользователя: Whisper на короткой фразе иногда выдаёт её **дважды**, но не байт-в-байт — со сменой падежа/окончания:
`"Замени плюшевую мишку на плюшевую обезьянку. Замени плюшевого мишку на плюшевую обезьянку."`
`_collapse_repeated_phrases` это пропускает (он сворачивает только ТОЧНЫЕ повторы n-gram), `compression_ratio_threshold=2.0` на двух коротких почти-одинаковых предложениях не дотягивает.

`_collapse_near_duplicate_sentences` (в `_voice.py`, запускается в `_clean_transcription_text` ПОСЛЕ exact-collapse, ПЕРЕД hallucination-strip):
- Режет текст по терминаторам `.!?…`, сравнивает только СОСЕДНИЕ предложения (loop у Whisper всегда подряд), выбрасывает почти-дубликат предыдущего. Между предложениями rejoin через один пробел.
- `_sentence_similarity` = word-level через `difflib.SequenceMatcher`; «replace»-слово засчитывается за совпадение, если это морфо-вариант (`_is_inflection_variant`).
- `_is_inflection_variant(a, b)`: общий префикс ≥4 символа И отличается хвост ≤3 символа с каждой стороны. Так `плюшевую`/`плюшевого`, `мишку`/`мишка`, `стол`/`столе` = варианты, а `десять`/`двадцать`, `красный`/`красивый` = разные слова. Слова <4 символов не судим.
- Порог `WHISPER_NEAR_DUP_SENTENCE_THRESHOLD = 0.85` подобран так, чтобы кейс пользователя (ratio 1.0) сворачивался, а легитимные разные команды с одним отличающимся смысловым словом (`…на десять процентов.`/`…на двадцать процентов.` → ratio 0.80) — НЕ сворачивались. Если Whisper мисхёрит контентное слово на повторе (ratio <0.85) — оставляем обе фразы (безопаснее).
- Escape hatch: `WHISPER_NEAR_DUP_SENTENCE_ENABLED` в `_helpers.py`.
- Scope: только sentence-level (по `.!?…`); запятая-разделённые повторы НЕ ловятся (запятая легитимна внутри предложения). Если всплывут — расширять осторожно.

## Multilingual hallucination guard (v9.10)

Whisper иногда уплывает в temperature-fallback chaos и эмитит текст с Cyrillic + Greek/CJK/Hangul/Italian/Spanish (типа «продолжение следует, перек sodium quindi Dáit свечку시 примms marc»). Такой выход дропаем **полностью** (text="") вместо вставки мусора в textarea.

`_looks_like_multilingual_hallucination(text, language)` — два независимых сигнала, любой триггерит drop:
1. **Tier 1 — экзотические алфавиты**: letter from Greek/CJK/Hangul/Devanagari/Arabic. Triggers when `other >= WHISPER_SCRIPT_OTHER_MIN_CHARS (5)` AND `other/total > WHISPER_SCRIPT_OTHER_MAX_RATIO (0.10)`.
2. **Tier 2 — mixed-script words**: токены с Cyrillic+Latin одновременно («примms», «увидired», «светлухаgeryный»). Triggers when count `>= WHISPER_SCRIPT_MIXED_WORD_THRESHOLD (2)`. Tier 2 — главный сигнал; реальный пример пользователя ловится именно им.

- Активна **только при language="ru"** (для других языков script mix может быть валиден).
- `WHISPER_SCRIPT_VALIDATION_ENABLED = True` (escape hatch для отладки).
- Логирует WARNING с превью отброшенного текста.

## Whisper decoding thresholds (v9.10 tightening)

Дефолтные параметры Whisper (compression 2.4, logprob -1.0, no_speech 0.6) пропускали многоязычные mush. Сейчас затянуто:
- `WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.0` (было 2.4) — типичная Russian/English речь имеет ratio 1.6-2.1, мусор 2.5+.
- `WHISPER_LOGPROB_THRESHOLD = -0.7` (было -1.0).
- `WHISPER_NO_SPEECH_THRESHOLD = 0.7` (было 0.6) — агрессивнее отсекать silence/noise tail (главный триггер галлюцинаций).
- `WHISPER_TEMPERATURE_FALLBACK = (0.0, 0.2, 0.4)` (было до 1.0) — выше 0.5 Whisper эффективно случайный, лучше пустота, чем мусор.

## initial_prompt MUST stay short and Russian-led (фикс «Посадим, ясмарк, я»)

Симптом: на turbo чистая фраза «посади девушку с референсного изображения на кресло» распозналась как «Посадим, ясмарк, я» — ранний end-of-transcript + замена слов. Не препроцессинг (trim off, полный клип) и не размер модели.

Причина — `INITIAL_PROMPT` в `_helpers.py` был раздут (~230-280 токенов, почти весь английский словарь камер/стилей). Два факта про Whisper `initial_prompt`:
1. Он используется ТОЛЬКО как vocabulary/style/spelling-хинт. Whisper **не выполняет инструкции** из него — предложения вида «Сохраняй смешанный русский и английский…» бесполезны.
2. Whisper оставляет лишь последние `n_text_ctx // 2 - 1 = 223` токена промпта. Длинный English-heavy промпт → русские строки в начале обрезаются, декодер заходит в русскую речь, заряженный чистым английским → биас, мис-декод, ранний EOT. Дальше каскад: низкий logprob → строгий `logprob_threshold=-0.7` → temperature fallback → совсем мусор.

Фикс (v9.68+): `INITIAL_PROMPT` сокращён до компактного русско-первого образца (~40 токенов) с горсткой самых частых mixed RU/EN терминов, без инструкций. Guard в `test_initial_prompt_is_prompt_dictation_context`: `len(prompt) < 400` — падает, если промпт снова раздуют. **НЕ возвращать exhaustive vocab list и инструкции.** Если мусор останется — следующий рычаг это пороги (logprob/compression/no_speech), но их трогать осторожно (тюнились против multilingual-галлюцинаций).

## Async record button (frontend, js/llm/ts-super-prompt.js)

Раньше первый клик на холодной ноде запускал preload, а не запись. Теперь:
- `state.isVoiceBusy` — только transcribe step.
- `state.isModelLoading` — отдельный флаг preload.
- `state.modelReadyPromise` — handle, который ждётся в `mediaRecorder.onstop` перед `sendAudioToServer`.
- `onRecordClick`: запись стартует **сразу**, загрузка модели запускается параллельно в фоне (если ещё не готова).
- `onstop`: если модель не готова — статус «Waiting for voice model...» + await promise + transcribe.
- WS-handlers (`onVoiceStatus`/`onVoiceDone`/`onVoiceError`) НЕ трогают флаги (флаги принадлежат HTTP-промисам), не топчут «Recording...» статус.

## Тесты

**Стаб-гейт:** `tests/_engine_stubs.py::install_engine_module_stub` ДОЛЖЕН экспортировать `QWEN_MODEL_LOCK` (его `_helpers.py` импортирует на module-load с коммита e8a357e). Без него весь `test_voice_recognition_audio.py` падает на import-time с `ImportError: cannot import name 'QWEN_MODEL_LOCK'`. Правило: добавил публичный символ в `_qwen_engine.py`/`_whisper_engine.py`, который тянет `_helpers.py` → синхронизируй стаб.

`tests/test_voice_recognition_audio.py` (28 тестов):
- `test_transcription_cleanup_collapses_near_duplicate_sentence` — кейс пользователя (плюшевую/плюшевого) + 2x exact + 3x с инфлексией.
- `test_transcription_cleanup_keeps_distinct_similar_sentences` — десять/двадцать, влево/вправо, красным/синим НЕ сворачиваются.
- `test_inflection_variant_distinguishes_endings_from_distinct_words` — unit на `_is_inflection_variant`.
- `test_near_duplicate_sentence_collapse_can_be_disabled` — escape hatch.
- `test_adaptive_vad_thresholds_low_never_above_high` — invariant low ≤ high.
- `test_detect_speech_bounds_hysteresis_recovers_quiet_consonants` — синтетическое аудио с тихим pre-roll.
- `test_detect_speech_bounds_hysteresis_stops_at_silence` — anti-regression, hysteresis не сжирает тишину.
- `test_transcription_cleanup_strips_youtube_outro_hallucination` — 4 кейса.
- `test_transcription_cleanup_filter_can_be_disabled` — флаг отключения работает.
- `test_multilingual_hallucination_drops_mixed_script_garbage` — реальный пример пользователя (Cyrillic + Korean + Spanish + Italian + German + Greek).
- `test_multilingual_hallucination_keeps_pure_russian` / `_keeps_russian_with_english_terms` — anti-false-positive.
- `test_multilingual_hallucination_only_runs_for_russian` — для en/auto/None не срабатывает.
- `test_multilingual_hallucination_short_text_is_not_judged` — `total < 10 letters` skip.
- `test_multilingual_hallucination_can_be_disabled` — escape hatch работает.

## Где править что

- VAD пороги/паддинг/fade — `nodes/llm/super_prompt/_helpers.py` (константы `AUDIO_VAD_*`, `AUDIO_EDGE_FADE_MS`).
- Whisper параметры (temperature, compression_ratio_threshold, beam_size, condition_on_previous_text) — `nodes/llm/super_prompt/_voice.py:transcribe_audio`. Уже схема fallback temperature `(0.0, 0.2, ...)` чтобы выходить из decoding loops.
- Hallucination список — `WHISPER_HALLUCINATION_PATTERNS` в `_helpers.py`.
- Поведение record button — `js/llm/ts-super-prompt.js`: `onRecordClick`, `downloadVoiceModel`, `mediaRecorder.onstop`, WS-handlers.
- Backward-compat shim `nodes/llm/ts_super_prompt.py` re-exports `_adaptive_vad_thresholds`, `AUDIO_VAD_LOW_MULTIPLIER` и др. — при добавлении новых публичных констант _helpers.py не забудь добавить их в импорты shim'а.
