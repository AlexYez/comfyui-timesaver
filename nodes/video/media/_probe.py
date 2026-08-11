"""Что внутри видеофайла: метаданные, пики звука и полоса миниатюр.

Пробуем PyAV, а не ``ffprobe`` и не разбор вывода ``ffmpeg -i`` регулярками.
Причины две, обе замерены: 34 мс против 56 мс плюс порождение процесса, и —
главное — результат приходит структурой, а не текстом. Строку вида
``Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709), 4096x2160
[SAR 1:1 DAR 256:135], 8129 kb/s, 30 fps`` разбирать регулярками нельзя: это
ловушка, в которую уже попал аудио-лоадер и весь VideoHelperSuite.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from ._common import LOG_PREFIX, cache_dir, file_identity, safe_log_path

logger = logging.getLogger("comfyui_timesaver.ts_video.probe")

CACHE_TTL_SECONDS = 7 * 24 * 3600
PEAK_BINS = 1024

# Версия формата ленты миниатюр. Поднимается, когда меняется САМА картинка
# ленты, — тогда уже сохранённые на диске ленты перестают находиться по ключу и
# пересчитываются. Без этого исправление пропорций увидел бы только тот, у кого
# кэш пуст. 2: ячеек ровно столько, сколько запрошено, поворот учитывается,
# ширина не округляется до чётной.
STRIP_VERSION = 2

# Кэш проб в памяти процесса: ключ файла -> MediaInfo. Модульный словарь, а не
# атрибут класса ноды (§5 CLAUDE.md: у V3-нод класс заперт).
_memory_cache: dict[str, "MediaInfo"] = {}
_swept = {"done": False}


@dataclass(frozen=True)
class AudioInfo:
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0


@dataclass(frozen=True)
class MediaInfo:
    """Всё, что известно о файле до декодирования."""

    filename: str = ""
    duration: float = 0.0
    fps: float = 0.0
    fps_exact: tuple[int, int] = (0, 1)
    frame_count: int = 0
    frame_count_estimated: bool = False
    width: int = 0
    height: int = 0
    display_width: int = 0
    display_height: int = 0
    codec: str = ""
    container: str = ""
    pix_fmt: str = ""
    bit_depth: int = 8
    rotation: int = 0
    sar: tuple[int, int] = (1, 1)
    vfr: bool = False
    has_alpha: bool = False
    faststart: bool = False
    audio: AudioInfo | None = None
    peaks: tuple[float, ...] = field(default=())

    @property
    def has_audio(self) -> bool:
        return self.audio is not None


def _av():
    """PyAV — лениво и с внятным отказом.

    Модуль идёт вместе с ComfyUI, но пак обязан читаться и без него (§14).
    """
    try:
        import av
    except Exception as error:              # noqa: BLE001 - зависимость необязательна вне ComfyUI
        raise RuntimeError(
            f"{LOG_PREFIX} PyAV is required to read video files. It normally ships with "
            f"ComfyUI; install it with `pip install av`. ({error})"
        ) from error
    return av


def _normalise_rotation(value: object) -> int:
    """Поворот из метаданных — к диапазону 0/90/180/270."""
    try:
        angle = int(round(float(value or 0)))
    except (TypeError, ValueError):
        return 0
    angle %= 360
    return angle if angle in (0, 90, 180, 270) else 0


def _stream_rotation(stream) -> int:
    """Угол поворота потока.

    PyAV отдаёт его по-разному в зависимости от версии и контейнера, поэтому
    спрашиваем и у потока, и у его метаданных.
    """
    for candidate in (getattr(stream, "rotation", None),
                      (stream.metadata or {}).get("rotate")):
        angle = _normalise_rotation(candidate)
        if angle:
            return angle
    return 0


def _bit_depth(pix_fmt: str) -> int:
    """Глубина по имени пиксельного формата — этого достаточно для решения о плане."""
    for depth in (16, 14, 12, 10):
        if f"{depth}le" in pix_fmt or f"{depth}be" in pix_fmt or f"p{depth}" in pix_fmt:
            return depth
    return 8


def _has_alpha(fmt) -> bool:
    try:
        if fmt.name == "pal8":
            return True
        return any(getattr(component, "is_alpha", False) for component in fmt.components)
    except Exception:                       # noqa: BLE001 - экзотический формат
        return False


def _probe_faststart(path: str) -> bool:
    """Лежит ли индекс mp4/mov в начале файла.

    Нужно фронтенду: без faststart браузер ради перемотки утягивает весь файл,
    поэтому плеер стартует с ``preload="none"``.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(256 * 1024)
    except OSError:
        return False
    moov = head.find(b"moov")
    mdat = head.find(b"mdat")
    if moov < 0:
        return False
    return mdat < 0 or moov < mdat


def _read_peaks(container, audio_stream, duration: float, bins: int) -> tuple[float, ...]:
    """Огибающая звука для таймлайна: ``bins`` значений 0..1.

    Считается из тех же аудиокадров, что уже декодируются, а не отдельным
    прогоном ffmpeg (так делает аудио-лоадер — файл читается дважды).
    """
    import numpy as np

    if duration <= 0:
        return ()
    peaks = np.zeros(bins, dtype=np.float32)
    seconds_per_bin = duration / bins

    try:
        for frame in container.decode(audio_stream):
            samples = frame.to_ndarray()
            if samples.size == 0:
                continue
            level = float(np.abs(samples).max())
            start = frame.time if frame.time is not None else 0.0
            index = int(start / seconds_per_bin) if seconds_per_bin > 0 else 0
            if 0 <= index < bins and level > peaks[index]:
                peaks[index] = level
    except Exception as error:              # noqa: BLE001 - битая дорожка не повод падать
        logger.debug("%s peaks stopped early: %s", LOG_PREFIX, error)

    # Дырки между заполненными столбиками (кадры длиннее бина) закрываем
    # соседями — иначе дорожка выглядит гребёнкой.
    last = 0.0
    for index in range(bins):
        if peaks[index] > 0:
            last = float(peaks[index])
        elif last > 0:
            peaks[index] = last * 0.85
    return tuple(round(float(value), 4) for value in peaks)


def probe(path: str, *, want_peaks: bool = True) -> MediaInfo:
    """Прочитать метаданные файла (и, если просят, огибающую звука).

    Args:
        path: абсолютный путь к видеофайлу.
        want_peaks: считать ли пики. Загрузчику они не нужны, фронтенду нужны.

    Raises:
        RuntimeError: файла нет, он не читается или в нём нет видеодорожки.
    """
    av = _av()
    if not os.path.isfile(path):
        raise RuntimeError(f"{LOG_PREFIX} File not found: {safe_log_path(path)}")

    try:
        container = av.open(path)
    except Exception as error:              # noqa: BLE001 - чужой/битый файл
        raise RuntimeError(
            f"{LOG_PREFIX} Could not open {safe_log_path(path)}: {error}"
        ) from error

    with container:
        video_streams = container.streams.video
        if not video_streams:
            raise RuntimeError(
                f"{LOG_PREFIX} No video stream in {safe_log_path(path)}."
            )
        stream = video_streams[0]

        rate = stream.average_rate or stream.guessed_rate or Fraction(0, 1)
        fps = float(rate) if rate else 0.0
        duration = 0.0
        if stream.duration is not None and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        if duration <= 0 and container.duration:
            duration = float(container.duration) / av.time_base

        frame_count = int(stream.frames or 0)
        estimated = False
        if frame_count <= 0 and fps > 0 and duration > 0:
            frame_count = int(round(duration * fps))
            estimated = True
        if duration <= 0 and fps > 0 and frame_count > 0:
            duration = frame_count / fps

        fmt = stream.format
        pix_fmt = fmt.name if fmt else ""
        sar = stream.sample_aspect_ratio or Fraction(1, 1)
        rotation = _stream_rotation(stream)

        width = int(stream.codec_context.width or 0)
        height = int(stream.codec_context.height or 0)
        display_w = int(round(width * float(sar))) if sar else width
        display_h = height
        if rotation in (90, 270):
            display_w, display_h = display_h, display_w

        # Переменная частота: заявленное число кадров заметно расходится с
        # длительностью на средней частоте. На VFR прореживание по счётчику
        # кадров (так делает VHS) врёт, а наш сэмплер по меткам — нет.
        vfr = False
        if frame_count and fps > 0 and duration > 0 and not estimated:
            expected = duration * fps
            vfr = abs(frame_count - expected) > max(2.0, 0.02 * frame_count)

        audio_info = None
        peaks: tuple[float, ...] = ()
        audio_streams = container.streams.audio
        if audio_streams:
            audio_stream = audio_streams[-1]
            audio_info = AudioInfo(
                codec=getattr(audio_stream.codec_context, "name", "") or "",
                sample_rate=int(audio_stream.codec_context.sample_rate or 0),
                channels=int(getattr(audio_stream.codec_context, "channels", 0) or 0),
            )
            if want_peaks:
                peaks = _read_peaks(container, audio_stream, duration, PEAK_BINS)

        return MediaInfo(
            filename=os.path.basename(path),
            duration=max(0.0, duration),
            fps=fps,
            fps_exact=(rate.numerator, rate.denominator) if rate else (0, 1),
            frame_count=max(0, frame_count),
            frame_count_estimated=estimated,
            width=width,
            height=height,
            display_width=display_w,
            display_height=display_h,
            codec=getattr(stream.codec_context, "name", "") or "",
            container=container.format.name if container.format else "",
            pix_fmt=pix_fmt,
            bit_depth=_bit_depth(pix_fmt),
            rotation=rotation,
            sar=(sar.numerator, sar.denominator) if sar else (1, 1),
            vfr=vfr,
            has_alpha=_has_alpha(fmt) if fmt else False,
            faststart=_probe_faststart(path),
            audio=audio_info,
            peaks=peaks,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Кэш на диске
# ──────────────────────────────────────────────────────────────────────────────

def _sweep_cache_once() -> None:
    """Убрать протухшее — лениво, при первом обращении.

    ⚠️ Не на импорте: удаление файлов побочным эффектом ``import`` запрещено
    (§13 CLAUDE.md).
    """
    if _swept["done"]:
        return
    _swept["done"] = True
    root = cache_dir()
    if not root.is_dir():
        return
    deadline = time.time() - CACHE_TTL_SECONDS
    for entry in root.iterdir():
        try:
            if entry.is_file() and entry.stat().st_mtime < deadline:
                entry.unlink()
        except OSError:
            continue


def _write_atomic(path: Path, payload: bytes) -> None:
    """Записать так, чтобы параллельный читатель не увидел половину файла."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def probe_cached(path: str, *, want_peaks: bool = True) -> MediaInfo:
    """Проба с кэшем в памяти и на диске.

    Ключ — путь вместе с размером и временем правки, поэтому перезапись файла
    тем же именем честно инвалидирует запись.
    """
    key = file_identity(path)
    cached = _memory_cache.get(key)
    if cached is not None and (cached.peaks or not want_peaks):
        return cached

    _sweep_cache_once()
    disk = cache_dir() / f"{key}.json"
    if disk.is_file():
        try:
            payload = json.loads(disk.read_text(encoding="utf-8"))
            info = _info_from_payload(payload)
            if info is not None and (info.peaks or not want_peaks):
                _memory_cache[key] = info
                return info
        except Exception as error:          # noqa: BLE001 - битый кэш просто перечитаем
            logger.debug("%s cache entry unreadable: %s", LOG_PREFIX, error)

    info = probe(path, want_peaks=want_peaks)
    _memory_cache[key] = info
    try:
        _write_atomic(disk, json.dumps(_payload_from_info(info)).encode("utf-8"))
    except OSError as error:
        logger.debug("%s could not write cache: %s", LOG_PREFIX, error)
    return info


def _payload_from_info(info: MediaInfo) -> dict:
    payload = asdict(info)
    payload["cache_schema"] = 1
    return payload


def _info_from_payload(payload: dict) -> MediaInfo | None:
    if payload.get("cache_schema") != 1:
        return None
    data = dict(payload)
    data.pop("cache_schema", None)
    audio = data.pop("audio", None)
    peaks = data.pop("peaks", ()) or ()
    known = {f for f in MediaInfo.__dataclass_fields__}
    data = {k: v for k, v in data.items() if k in known}
    for key in ("fps_exact", "sar"):
        if key in data and data[key] is not None:
            data[key] = tuple(data[key])
    return MediaInfo(
        **data,
        audio=AudioInfo(**audio) if audio else None,
        peaks=tuple(peaks),
    )


def peaks_window(path: str, start: float, end: float, bins: int) -> list[float]:
    """Огибающая звука на отрезке — для зума таймлайна.

    Обзорные пики из ``probe`` слишком грубы, когда на экране две секунды из
    часа; здесь считается ровно запрошенное окно.
    """
    import numpy as np

    av = _av()
    bins = max(1, min(4096, int(bins)))
    span = max(1e-6, float(end) - float(start))
    values = np.zeros(bins, dtype=np.float32)

    with av.open(path) as container:
        streams = container.streams.audio
        if not streams:
            return [0.0] * bins
        stream = streams[-1]
        if start > 0 and stream.time_base:
            container.seek(int(start / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            when = frame.time if frame.time is not None else 0.0
            if when < start:
                continue
            if when >= end:
                break
            samples = frame.to_ndarray()
            if samples.size == 0:
                continue
            index = min(bins - 1, int((when - start) / span * bins))
            level = float(np.abs(samples).max())
            if level > values[index]:
                values[index] = level

    last = 0.0
    for index in range(bins):
        if values[index] > 0:
            last = float(values[index])
        elif last > 0:
            values[index] = last * 0.85
    return [round(float(value), 4) for value in values]


def _sanitise_span(count: int, height: int) -> tuple[int, int]:
    """Зажать параметры спрайта.

    Без этого запрос на сто тысяч миниатюр укладывает сервер — и для этого
    достаточно поправить строку в адресе.
    """
    return max(1, min(64, int(count))), max(16, min(256, int(height)))


def filmstrip_sprite(path: str, *, start: float, step: float, count: int, height: int) -> bytes:
    """Спрайт из ``count`` миниатюр, снятых с шагом ``step`` секунд.

    Один запрос вместо шестнадцати. Кадры берутся тем же графом фильтров, что и
    обычная загрузка, поэтому масштабирование делает libswscale, а не Python.

    Returns:
        JPEG одной строкой миниатюр.
    """
    from ._decode import iter_thumbnails

    count, height = _sanitise_span(count, height)
    key = (f"{file_identity(path)}|strip{STRIP_VERSION}"
           f"|{start:.3f}|{step:.4f}|{count}|{height}")
    digest = __import__("hashlib").sha256(key.encode("utf-8")).hexdigest()
    disk = cache_dir() / f"{digest}.jpg"
    if disk.is_file():
        try:
            return disk.read_bytes()
        except OSError:
            pass

    _sweep_cache_once()
    frames = iter_thumbnails(path, start=start, step=step, count=count, height=height)
    payload = _join_sprite(frames, height=height, cells=count)
    try:
        _write_atomic(disk, payload)
    except OSError as error:
        logger.debug("%s could not cache sprite: %s", LOG_PREFIX, error)
    return payload


def _join_sprite(frames: list, height: int, cells: int) -> bytes:
    """Склеить кадры в одну горизонтальную ленту JPEG.

    ⚠️ Ячеек в ленте РОВНО ``cells``, даже если кадров вышло меньше (конец
    ролика). Клиент считает ширину ячейки делением ширины ленты на их число, и
    короткая лента давала ему завышенную ширину: он брал из неё срез чужого
    размера и растягивал — на экране это выглядело как искажённые пропорции у
    последних миниатюр.
    """
    import io

    import numpy as np
    from PIL import Image

    background = (24, 24, 28)
    slots = max(1, int(cells))
    if not frames:
        blank = Image.new("RGB", (max(1, height * 16 // 9) * slots, height), background)
        buffer = io.BytesIO()
        blank.save(buffer, format="JPEG", quality=70)
        return buffer.getvalue()

    images = [Image.fromarray(np.ascontiguousarray(frame)) for frame in frames]
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    sheet = Image.new("RGB", (cell_w * slots, cell_h), background)
    for index, image in enumerate(images[:slots]):
        sheet.paste(image, (index * cell_w, 0))

    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=78, optimize=True)
    return buffer.getvalue()


def frame_count_exact(path: str) -> int:
    """Пересчитать кадры честным декодированием.

    Дорого — это полный проход по файлу, — поэтому вызывается только по явной
    просьбе: у mkv и webm ``stream.frames`` часто ноль, а оценка по длительности
    на переменной частоте врёт. Кадры именно ДЕКОДИРУЮТСЯ: считать пакеты
    нельзя, один пакет не равен одному кадру.
    """
    av = _av()
    total = 0
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for packet in container.demux(stream):
            try:
                total += len(packet.decode())
            except av.error.InvalidDataError:
                continue
    return total


def estimate_bytes(frames: int, width: int, height: int, channels: int = 3) -> int:
    """Сколько памяти займёт тензор кадров."""
    return int(max(0, frames) * max(0, width) * max(0, height) * channels * 4)


def human_bytes(value: int) -> str:
    """Байты в удобочитаемый вид — для сообщений об ошибке."""
    if value <= 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    index = min(len(units) - 1, int(math.log(value, 1024)))
    return f"{value / (1024 ** index):.1f} {units[index]}"
