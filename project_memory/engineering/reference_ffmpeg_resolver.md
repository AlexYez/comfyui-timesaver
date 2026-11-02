---
name: reference_ffmpeg_resolver
description: "Где пак берёт ffmpeg (свой бинарник из imageio-ffmpeg, не PATH) и почему фоллбэка на PyTorch быть не может"
metadata:
  type: reference
---

Где пак берёт ffmpeg (свой бинарник из imageio-ffmpeg, не PATH) и почему фоллбэка на PyTorch быть не может

Исходники темы: [_ffmpeg.py](../../nodes/_ffmpeg.py), [_audio_helpers.py](../../nodes/audio/loader/_audio_helpers.py), [_whisper_engine.py](../../nodes/_whisper_engine.py), [_voice.py](../../nodes/llm/super_prompt/_voice.py), [ts_animation_preview.py](../../nodes/video/ts_animation_preview.py).

**Граница записи:** ниже сохранены исходные наблюдения и причины решений. Замеры, версии окружения и слова «сейчас» относятся к дате соответствующего наблюдения; это не повторная проверка на новом окружении. Порядок чтения и ограничения: [инженерная память](README.md#как-читать-записи).

`nodes/_ffmpeg.py` (2026-08-08). Общий на весь пак; импортируется как
`from ._ffmpeg import` / `from .._ffmpeg import` / `from ..._ffmpeg import` —
глубина по месту, см. [reference_node_import_paths](reference_node_import_paths.md).

## Кто зовёт

Четыре места: `nodes/audio/loader/_audio_helpers.py`, `nodes/_whisper_engine.py`,
`nodes/llm/super_prompt/_voice.py`, `nodes/video/ts_animation_preview.py`.
(`ts_sam_media_loader` ходит через хелперы аудио, своего вызова не имеет.)

⚠️ **Имена функций на местах вызова СОХРАНЕНЫ** (`_get_ffmpeg_executable`,
`get_ffmpeg_executable`, `_get_ffmpeg_exe`) — они тонкие обёртки над общим
резолвером. `nodes/llm/ts_super_prompt.py` реэкспортирует
`_get_ffmpeg_executable`, переименование сломало бы shim.

## Порядок поиска

1. **Свой бинарник**: `imageio_ffmpeg.get_ffmpeg_exe()` — плюс проверка
   `os.path.isfile`. ⚠️ Функция может назвать путь, которого нет (пакет стоит,
   бинарник вычистили) — проверять надо файл, а не факт ответа.
2. `shutil.which("ffmpeg")`.
3. Ничего → `ffmpeg_executable()` всё равно вернёт строку `"ffmpeg"` (вызывающие
   уже ловят падение процесса), а `require_ffmpeg()` бросит `MISSING_MESSAGE` —
   одно на весь пак, с действием, а не констатацией.

Результат кешируется (`_state` на уровне модуля): поиск — обращение к ФС, а
зовут его на каждый кадр. `reset_cache()` — для тестов и для случая «поставил
ffmpeg, не перезапуская ComfyUI».

## ⚠️ Мнимый фоллбэк, который был

`_whisper_engine.decode_audio_file` при `FileNotFoundError` откатывался на
`whisper.load_audio` — а тот **сам зовёт ffmpeg**. То есть при отсутствующем
бинарнике падал так же, только с невнятной ошибкой. Теперь этот случай сразу
даёт `MISSING_FFMPEG_MESSAGE`; для таймаута и сбоя декодирования фоллбэк на
whisper сохранён.

## Почему нет фоллбэка на средства PyTorch

Спрашивал владелец (2026-08-08). Короткий ответ: **не на что откатываться.**

- ffmpeg здесь разбирает контейнеры (mp4/mkv/webm/mov/m4a), декодирует сжатый
  звук, ресемплит в 16 кГц моно и КОДИРУЕТ видео в превью анимации. У торча
  кодеков нет вообще.
- Собственное декодирование `torchaudio` сворачивается в пользу TorchCodec,
  который линкуется с библиотеками FFmpeg; `torchvision.io.read_video` признан
  устаревшим и тоже построен на FFmpeg. «Взять средства торча» = тот же FFmpeg
  под другим именем плюс новая зависимость.
- Единственный настоящий не-ffmpeg путь — `soundfile`/libsndfile: wav, flac,
  ogg. Ни mp3, ни звука из видео он не берёт, то есть закрывает ровно те
  случаи, которые и не были проблемой. Как ускорение (не поднимать процесс) —
  осмысленно, как устойчивость — нет.

**Настоящая гарантия другая:** пак не зависит от ffmpeg в системе.
`imageio-ffmpeg` — обязательная зависимость в `requirements.txt`, она привозит
статический бинарник под все платформы, включая macOS arm64. PATH — запасной
путь, а не план.

Тесты: `tests/test_ffmpeg_resolver.py` (12). ⚠️ Штатный `tmp_path` в этом
окружении недоступен (PermissionError на `%TEMP%\pytest-of-*`) — брать
`ts_tmp_path` из `tests/conftest.py`.
