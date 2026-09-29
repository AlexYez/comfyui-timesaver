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

# Профили ProRes, которые несут альфу. Остальные профили 4:2:2 — прозрачности
# в них нет, и альфа там отбрасывается.
_ALPHA_PROFILES = frozenset({"4444", "4444 XQ"})


def _av():
    from ._probe import _av as resolve

    return resolve()


def _split_channels(array):
    """Кадр ``[H,W]`` / ``[H,W,C]`` → ``(rgb [H,W,3], alpha [H,W,1] или None)``.

    ⚠️ IMAGE в ComfyUI бывает не только трёхканальным: Join Image with Alpha
    отдаёт RGBA, маска, пропущенная через Convert Mask to Image, — серый.
    ``VideoFrame.from_ndarray(..., "rgb24")`` на четырёх каналах падает с
    ``Unexpected numpy array shape (H, W, 4)`` — и падал, уже после минут
    генерации. Здесь любая разумная раскладка сводится к RGB (+ альфа).

    Ничего не пишет во вход: срезы — это представления, повтор канала — копия.
    """
    import numpy as np

    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise RuntimeError(
            f"{LOG_PREFIX} Expected a frame shaped [H, W, C], got {tuple(array.shape)}.")
    channels = int(array.shape[2])
    if channels == 1:
        return np.repeat(array, 3, axis=2), None
    if channels == 2:                       # серый + альфа
        return np.repeat(array[..., :1], 3, axis=2), array[..., 1:2]
    if channels == 3:
        return array, None
    if channels == 4:
        return array[..., :3], array[..., 3:4]
    raise RuntimeError(
        f"{LOG_PREFIX} Unsupported channel count {channels}: expected 1 (gray), "
        f"2 (gray + alpha), 3 (RGB) or 4 (RGBA).")


def _fit_channels(array, alpha: bool):
    """Кадр uint8 → ровно RGB (``alpha=False``) или ровно RGBA (``alpha=True``).

    Альфа отбрасывается без смешивания с фоном — так же поступают штатные
    сохранятели ComfyUI с форматами, где альфы нет. Если альфа нужна, а в
    кадре её нет, кадр считается непрозрачным.
    """
    import numpy as np

    if array.ndim == 3 and int(array.shape[2]) == (4 if alpha else 3):
        return array
    rgb, opacity = _split_channels(array)
    if not alpha:
        return np.ascontiguousarray(rgb)
    if opacity is None:
        opacity = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate([rgb, opacity], axis=2))


def _frames_from_tensor(images) -> Iterator:
    """Тензор ComfyUI ``[B,H,W,C]`` 0..1 → кадры uint8 по одному.

    Генератором, а не списком: у сейвера на входе может лежать тысяча кадров 4K,
    и вторая их копия в памяти никому не нужна.

    Кадры выходят RGB или RGBA: серый размножается в три канала, альфа
    сохраняется — выбросить её или записать решает ``write_video``, потому что
    только он знает, умеет ли выбранный формат прозрачность.
    """
    import numpy as np

    count = int(images.shape[0])
    for index in range(count):
        frame = images[index]
        # ⚠️ Приводим к float32 ЯВНО: превью HDR-декодера приходит в float16, а
        # у него шаг возле 255 равен 0.25 — округление до байта поехало бы.
        # ⚠️ У float32-тензора на CPU `.numpy()` ДЕЛИТ память со входом, поэтому
        # ниже только операции, создающие новый массив (clip, repeat, concat).
        array = (frame.detach().float().cpu().numpy() if hasattr(frame, "detach")
                 else np.asarray(frame, dtype=np.float32))
        if not (array.ndim == 3 and int(array.shape[2]) in (3, 4)):
            # Серый -> RGB, серый + альфа -> RGBA. RGB и RGBA идут как есть.
            rgb, alpha = _split_channels(array)
            array = rgb if alpha is None else np.concatenate([rgb, alpha], axis=2)
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

    # Пустая дорожка — это ОТСУТСТВИЕ звука, а не тишина: дополнив её нулями,
    # мы вшили бы в файл немую дорожку, которой в источнике не было.
    if array.shape[1] == 0:
        return None, 0

    # ⚠️ Раскладка канала должна СОВПАДАТЬ с числом строк, иначе PyAV роняет всю
    # запись: "Expected planar array.shape[0] to equal 2 but got 3" — проверено
    # на трёхканальном звуке. Число каналов приходит из чужого AUDIO, поэтому
    # непредусмотренное сводится к стерео, а не обрушивает сохранение готового
    # ролика.
    channels = int(array.shape[0])
    if channels not in _CHANNEL_LAYOUTS:
        logger.warning("%s Unusual channel count (%d); mixing down to stereo.",
                       LOG_PREFIX, channels)
        if channels > 2:
            array = np.ascontiguousarray(array[:2])
        else:
            array = np.ascontiguousarray(np.repeat(array[:1], 2, axis=0))

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


def _discard_partial(path) -> None:
    """Удалить недописанный файл. Никогда не бросает.

    Зовётся только для файла, которого до записи НЕ было: чужой файл, даже
    совпавший по имени, не трогается никогда.
    """
    try:
        Path(path).unlink(missing_ok=True)
        logger.info("%s removed the unfinished file %s", LOG_PREFIX, safe_log_path(path))
    except OSError as error:
        logger.warning("%s could not remove the unfinished file %s: %s",
                       LOG_PREFIX, safe_log_path(path), error)


def _discard_sequence(files: list[Path], folder: Path | None) -> None:
    """Убрать кадры недописанной секвенции. Никогда не бросает.

    Args:
        files: файлы, созданные ЭТОЙ записью; чужие сюда не попадают.
        folder: папка, если её создала эта запись. Удаляется, только когда
            после уборки в ней ничего не осталось.
    """
    removed = 0
    for path in files:
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError as error:
            logger.warning("%s could not remove the unfinished frame %s: %s",
                           LOG_PREFIX, safe_log_path(path), error)
    if folder is not None:
        try:
            folder.rmdir()                  # только пустую — rmdir иначе откажет
        except OSError:
            pass
    if removed:
        logger.info("%s removed %d frames of the unfinished EXR sequence", LOG_PREFIX, removed)


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
        frames: последовательность кадров ``[H,W,C]`` uint8, ``C`` от 1 до 4.
            Альфа пишется, если её несёт выбранный формат (ProRes 4444 и
            4444 XQ), иначе отбрасывается; серый размножается в RGB.
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

    # Формат с прозрачностью, если выбранный вариант её действительно несёт:
    # у ProRes это только 4444 и 4444 XQ, у форматов без профилей — наличие
    # `alpha_pix_fmt` в реестре.
    alpha_pix_fmt = fmt.alpha_pix_fmt
    if fmt.profiles:
        chosen = profile if profile in fmt.profiles else next(iter(fmt.profiles))
        options["profile"] = fmt.profiles[chosen]
        options.setdefault("vendor", "apl0")
        pix_fmt = fmt.profile_pix_fmt.get(chosen, pix_fmt)
        if chosen not in _ALPHA_PROFILES:
            alpha_pix_fmt = None

    hardware_used = None
    if use_hardware:
        picked = pick_hardware_codec(fmt, quality)
        if picked is not None:
            codec, hw_options = picked
            options = dict(hw_options)
            hardware_used = codec
            alpha_pix_fmt = None            # аппаратные кодировщики альфу не пишут
            logger.info("%s using hardware encoder %s", LOG_PREFIX, codec)
        else:
            logger.info("%s no hardware encoder available, writing in software", LOG_PREFIX)

    rate = Fraction(float(fps)).limit_denominator(_RATE_DENOMINATOR)
    samples, sample_rate = _prepare_audio(audio, frame_count, float(fps))

    written = 0
    width = height = 0
    keep_alpha = False

    # ⚠️ Недописанный файл убирается — но ТОЛЬКО тот, которого до записи не
    # было. Имя выдаёт `output_path` со свежим номером, так что на деле файл
    # всегда новый; проверка страхует от совпадения, при котором удалился бы
    # чужой результат.
    owns_file = to_memory is None and path is not None and not os.path.exists(path)

    # ⚠️ BaseException, а не Exception: «Отмена» в ComfyUI — это
    # InterruptProcessingException, и она наследует BaseException. Ловя одно
    # Exception, мы оставляли бы половину ролика в output после каждой отмены.
    # Удаление идёт ПОСЛЕ выхода из `with`: на Windows открытый контейнером
    # файл не удаляется, пока его не закрыли.
    try:
        with _open_output(path, fmt, to_memory) as container:
            _apply_metadata(container, metadata)

            video_stream = None
            audio_stream = None
            audio_cursor = 0

            for array in frames:
                if video_stream is None:
                    height, width = int(array.shape[0]), int(array.shape[1])
                    has_alpha = array.ndim == 3 and int(array.shape[2]) in (2, 4)
                    keep_alpha = has_alpha and alpha_pix_fmt is not None
                    if keep_alpha:
                        pix_fmt = alpha_pix_fmt
                    elif has_alpha:
                        logger.info("%s %s keeps no alpha channel; transparency is "
                                    "dropped.", LOG_PREFIX, fmt.key)
                    video_stream = container.add_stream(codec, rate=rate)
                    video_stream.width = width
                    video_stream.height = height
                    video_stream.pix_fmt = pix_fmt
                    # Шкала времени — обратная частота целиком: метки кадров идут
                    # 0,1,2…, значит один шаг обязан равняться одному кадру. Мукс
                    # mp4 сегодня всё равно пересчитывает метки по объявленной
                    # частоте (проверено: 23.976 и 29.97 выходят верными и без
                    # этой строки), но полагаться на это незачем — у другого
                    # контейнера своя воля.
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
                        except Exception:   # noqa: BLE001 - старые сборки PyAV
                            pass
                        if fmt.audio_options:
                            audio_stream.options = dict(fmt.audio_options)

                frame = av.VideoFrame.from_ndarray(
                    _fit_channels(array, keep_alpha), format="rgba" if keep_alpha else "rgb24")
                frame.pts = written
                for packet in video_stream.encode(frame):
                    container.mux(packet)
                written += 1
                if on_frame is not None:
                    on_frame(written)

                # Звук доливается ровно до конца уже записанного видео: так
                # дорожки остаются синхронными без отдельного прохода ремукса.
                if audio_stream is not None:
                    until = int(round(written / float(fps) * sample_rate))
                    audio_cursor = _push_audio(container, audio_stream, samples,
                                               audio_cursor, until, sample_rate)

            if video_stream is None:
                raise RuntimeError(f"{LOG_PREFIX} Nothing to save: no frames were produced.")

            if audio_stream is not None and samples is not None:
                # ⚠️ Длина звука подгоняется под ФАКТИЧЕСКИ записанные кадры, а
                # не под заявленные. `frame_count` для видео-источника — это
                # оценка (`duration * fps`), и совпадать с реальностью она не
                # обязана: у VFR, у контейнера без длительности, просто на
                # округлении.
                #
                # Замерено на расхождении в 20 кадров: заявили 50, записали 30 —
                # звук выходил на 0,8 с ДЛИННЕЕ картинки; заявили 30, записали
                # 50 — на 0,8 с короче. Оба случая выглядят как брак, и оба
                # лечатся тем, что мерка берётся с уже написанного видео.
                wanted = (int(round(written / float(fps) * sample_rate)) if fps > 0
                          else samples.shape[1])
                if samples.shape[1] < wanted:
                    import numpy as np

                    pad = np.zeros((samples.shape[0], wanted - samples.shape[1]),
                                   dtype=np.float32)
                    samples = np.concatenate([samples, pad], axis=1)
                audio_cursor = _push_audio(container, audio_stream, samples, audio_cursor,
                                           min(wanted, samples.shape[1]), sample_rate)
                for packet in audio_stream.encode(None):
                    container.mux(packet)

            for packet in video_stream.encode(None):
                container.mux(packet)
    except BaseException:
        if owns_file:
            _discard_partial(path)
        raise

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
    """Тензор ``[B,H,W,C]`` → кадры torch ``[H,W,3]`` float32, без зажима.

    Генератором: ролик 129×1920×1088 в float32 весит 3 ГиБ, и второй его копии
    в памяти быть не должно.

    Запись EXR в паке трёхканальная, поэтому альфа отбрасывается, а серый
    размножается в RGB — раньше RGBA на этом входе ронял запись на первом же
    кадре.
    """
    import numpy as np
    import torch

    warned = False
    for index in range(int(images.shape[0])):
        frame = images[index]
        if hasattr(frame, "detach"):
            frame = frame.detach().to(torch.float32)
        else:
            frame = torch.from_numpy(np.asarray(frame, dtype=np.float32))
        if frame.ndim == 2:
            frame = frame.unsqueeze(-1)
        channels = int(frame.shape[-1]) if frame.ndim == 3 else 0
        if channels in (2, 4) and not warned:
            logger.info("%s The EXR sequence is written as RGB; the alpha channel "
                        "is dropped.", LOG_PREFIX)
            warned = True
        if channels in (1, 2):
            # repeat — копия одного кадра, а не представление: вход не трогаем.
            frame = frame[..., :1].repeat(1, 1, 3)
        elif channels == 4:
            frame = frame[..., :3]
        elif channels != 3:
            raise RuntimeError(
                f"{LOG_PREFIX} Unsupported frame shape {tuple(frame.shape)} for EXR: "
                "expected [H, W, C] with C from 1 to 4.")
        yield frame


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

    folder_is_new = not folder.exists()
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    total_bytes = 0
    width = height = 0
    # Только файлы, которых до этой записи НЕ было: их и только их убираем,
    # если секвенция не дописалась.
    created: list[Path] = []

    # ⚠️ BaseException: отмена в ComfyUI (InterruptProcessingException) —
    # не Exception, и половина секвенции оставалась бы в output. Сюда же
    # попадает GeneratorExit — генератор, брошенный на полпути, секвенцию не
    # дописал.
    try:
        for array in frames:
            # EXR пишется трёхканальным: кадр из файла приводится к RGB (для
            # обычного RGB это тот же массив, без копии).
            frame = (_uint8_to_linear(_fit_channels(array, False))
                     if isinstance(array, np.ndarray) else array)
            if height == 0:
                height, width = int(frame.shape[0]), int(frame.shape[1])
            written += 1
            target = folder / f"{stem}.{written:06d}.exr"
            if not target.exists():
                created.append(target)
            total_bytes += write_exr(target, frame, half=half)

            preview = make_sdr_preview(frame.unsqueeze(0), exposure_ev=exposure_ev,
                                       operator=tonemap, output_dtype=frame.dtype)
            yield np.ascontiguousarray(
                (np.clip(preview[0].cpu().numpy(), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))

            if on_frame is not None:
                on_frame(written)

        if written == 0:
            raise RuntimeError(f"{LOG_PREFIX} Nothing to save: no frames were produced.")
    except BaseException as error:
        _discard_sequence(created, folder if folder_is_new else None)
        if isinstance(error, Exception):
            # Потребитель превью (запись прокси) глотает обычные ошибки как
            # «превью не вышло». Кладём её сюда, чтобы сохранятель поднял
            # настоящую причину, а не отчитался о нуле кадров.
            result["failed"] = error
        raise

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
