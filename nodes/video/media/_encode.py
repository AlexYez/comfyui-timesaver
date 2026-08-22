"""Запись видео: контейнер, видеодорожка и звук — за один проход.

⚠️ ОДИН ПРОХОД, ОДИН ФАЙЛ. VideoHelperSuite сначала пишет видео без звука, а
потом отдельным запуском ffmpeg примуксывает дорожку во ВТОРОЙ файл; наш
``TS_Animation_Preview`` вдобавок кладёт временный WAV. Здесь кадры и звук
чередуются по времени в одном контейнере: ни временных файлов, ни второго
процесса, ни лишней записи гигабайтов на диск.

Кодирование идёт через PyAV. Кодеки проверены реальным энкодом: ``libx264`` и
``prores_ks`` (включая 4444 с альфой) на месте.
"""

from __future__ import annotations

import json
import logging
import os
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from ._common import LOG_PREFIX, safe_log_path
from ._formats import Format, Quality, get_format, get_quality, pick_hardware_codec

logger = logging.getLogger("comfyui_timesaver.ts_video.encode")

# Частота кадров уезжает в контейнер дробью. Знаменатель ограничиваем, иначе
# 23.976 превращается в чудовище вида 11988/500 и ломает совместимость.
_RATE_DENOMINATOR = 90000

_CHANNEL_LAYOUTS = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}


def _av():
    from ._probe import _av as resolve

    return resolve()


def _frames_from_tensor(images) -> Iterator:
    """Тензор ComfyUI ``[B,H,W,C]`` 0..1 → кадры uint8 по одному.

    Генератором, а не списком: у сейвера на входе может лежать тысяча кадров 4K,
    и вторая их копия в памяти никому не нужна.
    """
    import numpy as np

    count = int(images.shape[0])
    for index in range(count):
        frame = images[index]
        # ⚠️ Приводим к float32 ЯВНО: превью HDR-декодера приходит в float16, а
        # у него шаг возле 255 равен 0.25 — округление до байта поехало бы.
        array = (frame.detach().float().cpu().numpy() if hasattr(frame, "detach")
                 else np.asarray(frame, dtype=np.float32))
        array = np.clip(array, 0.0, 1.0)
        yield np.ascontiguousarray((array * 255.0 + 0.5).astype(np.uint8))


def _audio_layout(channels: int) -> str:
    return _CHANNEL_LAYOUTS.get(int(channels), "stereo")


def _prepare_audio(audio: Mapping | None, frame_count: int, fps: float):
    """Привести звук к длительности ролика.

    Короче — добиваем тишиной, длиннее — режем. Иначе последний кадр окажется
    без звука или звук переживёт картинку, и оба случая выглядят как брак.

    Returns:
        ``(numpy [C,T] float32, sample_rate)`` или ``(None, 0)``.
    """
    import numpy as np

    if not audio or fps <= 0 or frame_count <= 0:
        return None, 0

    waveform = audio.get("waveform")
    rate = int(audio.get("sample_rate") or 0)
    if waveform is None or rate <= 0:
        return None, 0

    array = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    array = np.ascontiguousarray(array.astype(np.float32))

    wanted = int(round(frame_count / fps * rate))
    if array.shape[1] < wanted:
        pad = np.zeros((array.shape[0], wanted - array.shape[1]), dtype=np.float32)
        array = np.concatenate([array, pad], axis=1)
    elif array.shape[1] > wanted:
        array = array[:, :wanted]
    return array, rate


def _open_output(path: str | os.PathLike[str] | None, fmt: Format, to_memory):
    av = _av()
    options = dict(fmt.mux_options if to_memory else (fmt.file_mux_options or fmt.mux_options))
    target = to_memory if to_memory is not None else str(path)
    return av.open(target, mode="w", format=fmt.container, options=options)


def _apply_metadata(container, metadata: Mapping | None) -> None:
    """Вшить prompt и workflow в контейнер.

    Читается обратно теми же средствами, поэтому воркфлоу можно вытащить из
    самого видеофайла — как из PNG.
    """
    if not metadata:
        return
    for key, value in metadata.items():
        if value is None:
            continue
        try:
            container.metadata[key] = value if isinstance(value, str) else json.dumps(value)
        except Exception as error:          # noqa: BLE001 - контейнер без тегов
            logger.debug("%s metadata key %r skipped: %s", LOG_PREFIX, key, error)


def write_video(
    frames: Iterable,
    *,
    path: str | os.PathLike[str] | None,
    format_key: str,
    quality_key: str = "",
    profile: str = "",
    ten_bit: bool = False,
    fps: float = 24.0,
    audio: Mapping | None = None,
    frame_count: int = 0,
    metadata: Mapping | None = None,
    use_hardware: bool = False,
    to_memory=None,
    on_frame=None,
) -> dict:
    """Записать кадры в видеофайл.

    Args:
        frames: последовательность кадров ``[H,W,3]`` uint8.
        path: куда писать; игнорируется, если задан ``to_memory``.
        format_key: ключ из реестра (``"H.264 / MP4"``).
        quality_key: уровень качества для форматов, где он есть.
        profile: профиль для ProRes.
        ten_bit: писать десятью битами там, где формат это умеет.
        fps: частота записи.
        audio: словарь AUDIO ComfyUI или ``None``.
        frame_count: сколько кадров ожидается (нужно для подгонки звука).
        metadata: что вшить в контейнер.
        use_hardware: разрешить аппаратный кодировщик, если он откроется.
        to_memory: ``io.BytesIO`` вместо файла (для тестов).
        on_frame: зовётся после каждого записанного кадра — чтобы нода могла
            показать полосу выполнения. Кодирование длинного ролика идёт
            минутами, и молчащая нода в это время выглядит зависшей.

    Returns:
        Словарь с фактическими параметрами записи.
    """
    av = _av()
    fmt = get_format(format_key)
    quality = get_quality(fmt, quality_key)

    pix_fmt = quality.pix_fmt or fmt.pix_fmt
    if ten_bit and fmt.ten_bit_pix_fmt:
        pix_fmt = fmt.ten_bit_pix_fmt
    codec = fmt.codec
    options = dict(quality.options)

    if fmt.profiles:
        chosen = profile if profile in fmt.profiles else next(iter(fmt.profiles))
        options["profile"] = fmt.profiles[chosen]
        options.setdefault("vendor", "apl0")
        pix_fmt = fmt.profile_pix_fmt.get(chosen, pix_fmt)

    hardware_used = None
    if use_hardware:
        picked = pick_hardware_codec(fmt, quality)
        if picked is not None:
            codec, hw_options = picked
            options = dict(hw_options)
            hardware_used = codec
            logger.info("%s using hardware encoder %s", LOG_PREFIX, codec)
        else:
            logger.info("%s no hardware encoder available, writing in software", LOG_PREFIX)

    rate = Fraction(float(fps)).limit_denominator(_RATE_DENOMINATOR)
    samples, sample_rate = _prepare_audio(audio, frame_count, float(fps))

    written = 0
    width = height = 0

    with _open_output(path, fmt, to_memory) as container:
        _apply_metadata(container, metadata)

        video_stream = None
        audio_stream = None
        audio_cursor = 0

        for array in frames:
            if video_stream is None:
                height, width = int(array.shape[0]), int(array.shape[1])
                video_stream = container.add_stream(codec, rate=rate)
                video_stream.width = width
                video_stream.height = height
                video_stream.pix_fmt = pix_fmt
                # Шкала времени — обратная частота целиком: метки кадров идут
                # 0,1,2…, значит один шаг обязан равняться одному кадру. Мукс mp4
                # сегодня всё равно пересчитывает метки по объявленной частоте
                # (проверено: 23.976 и 29.97 выходят верными и без этой строки),
                # но полагаться на это незачем — у другого контейнера своя воля.
                video_stream.time_base = Fraction(rate.denominator, rate.numerator)
                if options:
                    video_stream.options = dict(options)
                if fmt.codec_tag:
                    video_stream.codec_tag = fmt.codec_tag

                if samples is not None and fmt.audio_codec:
                    audio_stream = container.add_stream(fmt.audio_codec, rate=sample_rate)
                    layout = _audio_layout(samples.shape[0])
                    try:
                        audio_stream.layout = layout
                    except Exception:       # noqa: BLE001 - старые сборки PyAV
                        pass
                    if fmt.audio_options:
                        audio_stream.options = dict(fmt.audio_options)

            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = written
            for packet in video_stream.encode(frame):
                container.mux(packet)
            written += 1
            if on_frame is not None:
                on_frame(written)

            # Звук доливается ровно до конца уже записанного видео: так дорожки
            # остаются синхронными без отдельного прохода ремукса.
            if audio_stream is not None:
                until = int(round(written / float(fps) * sample_rate))
                audio_cursor = _push_audio(container, audio_stream, samples,
                                           audio_cursor, until, sample_rate)

        if video_stream is None:
            raise RuntimeError(f"{LOG_PREFIX} Nothing to save: no frames were produced.")

        if audio_stream is not None and samples is not None:
            audio_cursor = _push_audio(container, audio_stream, samples,
                                       audio_cursor, samples.shape[1], sample_rate)
            for packet in audio_stream.encode(None):
                container.mux(packet)

        for packet in video_stream.encode(None):
            container.mux(packet)

    size = 0
    if to_memory is None and path is not None:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        logger.info("%s wrote %d frames to %s", LOG_PREFIX, written, safe_log_path(path))

    return {
        "frames": written,
        "width": width,
        "height": height,
        "fps": float(fps),
        "format": fmt.key,
        "codec": codec,
        "extension": fmt.extension,
        "hardware": hardware_used,
        "has_audio": samples is not None and bool(fmt.audio_codec),
        "browser_playable": fmt.browser_playable,
        "size_bytes": size,
    }


def _push_audio(container, stream, samples, cursor: int, until: int, rate: int) -> int:
    """Отправить звук до отметки ``until`` (в сэмплах).

    Кодеки просят кадры своего размера (у AAC это ровно 1024 сэмпла), поэтому
    режем ровно так, как просит кодировщик.
    """
    import numpy as np

    av = _av()
    if samples is None or until <= cursor:
        return cursor

    chunk_size = int(getattr(stream.codec_context, "frame_size", 0) or 1024)
    layout = _audio_layout(samples.shape[0])

    while cursor < until:
        end = min(until, cursor + chunk_size, samples.shape[1])
        if end <= cursor:
            break
        block = np.ascontiguousarray(samples[:, cursor:end])
        frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=layout)
        frame.sample_rate = rate
        frame.pts = cursor
        frame.time_base = Fraction(1, rate)
        for packet in stream.encode(frame):
            container.mux(packet)
        cursor = end
    return cursor


def write_proxy(
    frames: Iterable,
    *,
    path: str | os.PathLike[str],
    fps: float,
    audio: Mapping | None = None,
    frame_count: int = 0,
    on_frame=None,
) -> dict:
    """Маленький H.264 для плеера в ноде.

    Нужен, когда сохранённый формат браузер не проигрывает (ProRes). Дешёвый по
    определению: ``veryfast`` и ширина не больше 1280.
    """
    return write_video(
        frames,
        path=path,
        format_key="H.264 / MP4",
        quality_key="draft",
        fps=fps,
        audio=audio,
        frame_count=frame_count,
        metadata=None,
        use_hardware=False,
        on_frame=on_frame,
    )


def downscale_frames(frames: Iterable, max_width: int = 1280) -> Iterator:
    """Ужать кадры под превью, не трогая исходные.

    Ресайз тут делает PIL, а не граф фильтров: кадры уже в памяти, поток
    короткий, а тянуть ради превью второй декодер незачем.
    """
    import numpy as np
    from PIL import Image

    for array in frames:
        height, width = array.shape[0], array.shape[1]
        if width <= max_width:
            yield array
            continue
        new_w = max_width - (max_width % 2)
        new_h = int(round(height * new_w / width))
        new_h -= new_h % 2
        image = Image.fromarray(array).resize((new_w, max(2, new_h)), Image.BILINEAR)
        yield np.ascontiguousarray(np.asarray(image))


# ────────────────────────── секвенция EXR ──────────────────────────────
#
# Кадры HDR-мастера — не картинки в привычном смысле: значения в них уходят
# далеко за единицу, и весь смысл именно в этом. Поэтому здесь не переиспользуется
# ни `_frames_from_tensor` (он режет в байты), ни `write_video` (у секвенции нет
# контейнера). Запись EXR живёт в `video/hdr/_exr_io.py` — единственном месте
# пака, которое знает этот формат; направление зависимости всегда одно:
# `video/media` импортирует из `video/hdr`, обратно никогда.

def _linear_frames(images) -> Iterator:
    """Тензор ``[B,H,W,3]`` → кадры torch ``[H,W,3]`` float32, без зажима.

    Генератором: ролик 129×1920×1088 в float32 весит 3 ГиБ, и второй его копии
    в памяти быть не должно.
    """
    import numpy as np
    import torch

    for index in range(int(images.shape[0])):
        frame = images[index]
        if hasattr(frame, "detach"):
            yield frame.detach().to(torch.float32)
        else:
            yield torch.from_numpy(np.asarray(frame, dtype=np.float32))


def _uint8_to_linear(array):
    """Кадр uint8 из видеофайла → float32 ``[0, 1]``."""
    import torch

    return torch.from_numpy(array.astype("float32") / 255.0)


def exr_sequence_pass(
    frames: Iterable,
    *,
    folder: Path,
    stem: str,
    half: bool = False,
    tonemap: str = "reinhard_luma",
    exposure_ev: float = 0.0,
    result: dict,
    on_frame=None,
) -> Iterator:
    """Записать секвенцию EXR и попутно отдать кадры для превью.

    Генератор: на каждый кадр пишется файл ``<stem>.<номер>.exr`` и отдаётся
    тот же кадр, приведённый к экрану. Один проход по источнику — а он может
    быть потоком из файла, который второй раз не перемотать.

    Args:
        frames: кадры ``[H,W,3]`` — torch float32 (линейный свет) или numpy
            uint8 (когда пересохраняется обычное видео).
        folder: куда класть файлы; создаётся при необходимости.
        stem: основа имени файла.
        half: писать 16-битными числами.
        tonemap: оператор для превью; на сами EXR не влияет.
        exposure_ev: экспозиция превью; на сами EXR не влияет.
        result: словарь, который заполняется по ходу записи.
        on_frame: обратный вызов для полосы выполнения.

    Yields:
        Кадры ``[H,W,3]`` uint8 для превью.
    """
    import numpy as np

    from ...video.hdr._exr_io import write_exr
    from ...video.hdr._tonemap import make_sdr_preview

    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    total_bytes = 0
    width = height = 0

    for array in frames:
        frame = _uint8_to_linear(array) if isinstance(array, np.ndarray) else array
        if height == 0:
            height, width = int(frame.shape[0]), int(frame.shape[1])
        written += 1
        total_bytes += write_exr(folder / f"{stem}.{written:06d}.exr", frame, half=half)

        preview = make_sdr_preview(frame.unsqueeze(0), exposure_ev=exposure_ev,
                                   operator=tonemap, output_dtype=frame.dtype)
        yield np.ascontiguousarray(
            (np.clip(preview[0].cpu().numpy(), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))

        if on_frame is not None:
            on_frame(written)

    if written == 0:
        raise RuntimeError(f"{LOG_PREFIX} Nothing to save: no frames were produced.")

    result.update({
        "frames": written,
        "width": width,
        "height": height,
        "size_bytes": total_bytes,
        "codec": "exr",
        "has_audio": False,
        "browser_playable": False,
    })
    logger.info("%s wrote %d EXR frames (%s) to %s", LOG_PREFIX, written,
                "16-bit half" if half else "32-bit float", safe_log_path(folder))


def output_sequence_path(prefix: str) -> tuple[Path, str, str]:
    """Папка под секвенцию: ``<output>/<подпапка>/<имя>/<имя>.000001.exr``.

    Отдельная папка на прогон — иначе тысяча файлов ложится вперемешку с
    чужими, и разобрать, где чья секвенция, потом невозможно.

    Returns:
        ``(папка, основа имени, подпапка относительно output)``.
    """
    target, _, subfolder = output_path(prefix, "exr")
    stem = target.stem
    folder = target.parent / stem
    return folder, stem, f"{subfolder}/{stem}" if subfolder else stem


def output_path(prefix: str, extension: str) -> tuple[Path, str, str]:
    """Куда сохранять результат.

    Дальше работает штатный ``folder_paths.get_save_image_path``: подпапки в
    префиксе, его собственные токены (``%year%``, ``%width%`` и прочие) и
    нумерация без затирания чужих файлов.

    ⚠️ Но форму ``%date:yyyy-MM-dd%`` ядро НЕ разворачивает — её подставляет
    фронтенд, и только своим нодам. Поэтому она раскрывается здесь, до вызова
    ядра: иначе двоеточие уезжало в имя файла, и Windows отвечал
    ``OSError: Invalid argument``. Правило одно на пак — ``nodes/_shared.py``.

    Returns:
        ``(полный путь, имя файла, подпапка)``.
    """
    import folder_paths

    from ..._shared import expand_date_tokens

    base = folder_paths.get_output_directory()
    full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        expand_date_tokens(prefix), base)
    name = f"{filename}_{counter:05}_.{extension}"
    return Path(full_folder) / name, name, subfolder
