"""Декодирование видео: кадры, звук, миниатюры.

⚠️ ВСЯ РАБОТА С ПИКСЕЛЯМИ ИДЁТ В ГРАФЕ ФИЛЬТРОВ PyAV, а не в Python. Замерено на
4K-ролике: два секундных окна «наивным» путём (полное разрешение → torch.stack →
ресайз torch) — 5.9 с и 6.4 ГБ; тот же кусок через ``av.filter.Graph`` со
``scale`` внутри — 0.43 с. Отсюда все решения ниже.

⚠️ АППАРАТНОЕ ДЕКОДИРОВАНИЕ НЕ ИСПОЛЬЗУЕТСЯ СОЗНАТЕЛЬНО. Замерено 2026-08-11 на
RTX 3080 Ti, 300 кадров 4K → 1024×540: software+граф 0.89 с, ``hwaccel=cuda``
2.78 с, ``d3d11va`` 4.93 с. Передача кадров из видеопамяти съедает весь выигрыш.
Не «чинить» это из лучших побуждений — сначала перемерить.
"""

from __future__ import annotations

import logging
import math
import os
import pathlib
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Iterator

from ._common import LOG_PREFIX, safe_log_path
from ._probe import MediaInfo, estimate_bytes, human_bytes, probe_cached

logger = logging.getLogger("comfyui_timesaver.ts_video.decode")

# Потолок памяти под кадры. Нужен: без него часовой 4K-ролик просто уводит
# машину в своп. Но ЧИСЛОМ его задавать нельзя — 8 ГиБ на машине с 64 ГБ ОЗУ
# отказывали 13-секундному 4K-ролику, которому хватало памяти с запасом
# (реальная жалоба). Потолок считается от того, сколько ОЗУ у машины на самом
# деле; переменная окружения по-прежнему главнее всего.
_MAX_BYTES_ENV = "TS_VIDEO_MAX_BYTES"
_FALLBACK_MAX_BYTES = 8 * 1024 ** 3     # если про машину узнать не удалось
_TOTAL_RAM_SHARE = 0.6                  # доля ОЗУ, которую готовы отдать кадрам

# Ниже этого порога перемотка не окупается: seek садится на ключевой кадр
# раньше цели (замерено — на 0.77–0.9 с), и проще прочитать с начала.
_SEEK_THRESHOLD = 0.5

_RESIZE_FLAGS = {
    "bicubic": "bicubic",
    "bilinear": "bilinear",
    "lanczos": "lanczos",
    "area": "area",
    "neighbor": "neighbor",
}

_ROTATION_FILTERS = {
    90: (("transpose", "1"),),
    180: (("hflip", ""), ("vflip", "")),
    270: (("transpose", "2"),),
}


def _av():
    from ._probe import _av as resolve

    return resolve()


@dataclass
class DecodeRequest:
    """Что именно просят декодировать."""

    path: str
    start_time: float = 0.0
    end_time: float = -1.0
    frame_rate: float = 0.0
    max_frames: int = 0
    longer_side: int = 0
    shorter_side: int = 0
    divisible_by: int = 1
    frame_step: int = 1
    resize_filter: str = "area"
    want_audio: bool = True
    want_alpha: bool = False
    allow_disk: bool = False                # кадры не влезли в ОЗУ — держать на диске


@dataclass
class DecodeResult:
    """Что получилось."""

    images: object = None                   # torch.Tensor [B,H,W,C]
    alpha: object = None                    # torch.Tensor [B,H,W] или None
    audio: dict | None = None
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    truncated: bool = False
    media: MediaInfo = field(default_factory=MediaInfo)


class FrameSampler:
    """Сколько раз выдать кадр, снятый в момент ``t``.

    Возвращает 0 (выбросить), 1 (обычный случай) или больше (дублировать при
    повышении частоты).

    ⚠️ Считает по ВРЕМЕННЫМ МЕТКАМ, а не по номеру кадра. Прореживание по
    счётчику — то, как это делает VideoHelperSuite, — врёт на видео с переменной
    частотой: кадры пойдут неравномерно, и звук уедет. И не фильтром ``fps``:
    он теряет кадр на границе окна (проверено — 2 с при 30 fps дают 59 кадров).
    """

    def __init__(
        self,
        target_fps: float,
        start_time: float,
        frame_step: int,
        end_time: float = 0.0,
    ) -> None:
        self._period = (1.0 / target_fps) if target_fps > 0 else 0.0
        self._next = float(start_time)
        self._step = max(1, int(frame_step))
        self._end = float(end_time)
        self._raw = 0

    def take(self, when: float) -> int:
        if self._period <= 0.0:
            emitted = 1
        else:
            emitted = 0
            while self._next <= when + self._period * 0.5:
                # Окно ПОЛУОТКРЫТОЕ: момент `end` уже не наш. Иначе двухсекундный
                # кусок при 12 fps даёт 25 кадров вместо 24, и сохранённый ролик
                # оказывается длиннее выбранного на один кадр.
                if self._end > 0 and self._next >= self._end - 1e-9:
                    break
                emitted += 1
                self._next += self._period

        # ⚠️ Шаг прореживает ВЫХОДНОЙ поток, а не входной, — так он и обещан в
        # интерфейсе, и так же считается заявленная частота (`fps / step`).
        # Раньше счётчик двигался по ИСХОДНЫМ кадрам и дубликаты схлопывались в
        # один: при 30→60 и шаге 2 на выходе получалось около 15 кадров в
        # секунду вместо обещанных 30, причём метаданные сообщали 30.
        kept = 0
        for _ in range(emitted):
            if (self._raw % self._step) == 0:
                kept += 1
            self._raw += 1
        return kept


def target_size(
    media: MediaInfo,
    longer_side: int,
    shorter_side: int,
    divisible_by: int,
) -> tuple[int, int]:
    """Итоговый размер кадра.

    ⚠️ Задаются СТОРОНЫ, а не ширина с высотой: материал приходит и альбомный, и
    портретный, и «ширина 1024» значит на них разное, а «длинная сторона 1024» —
    одно и то же. Какая сторона длиннее, решает сам источник.

    ``0`` значит «вывести из второй, сохранив пропорции»; оба нуля — оставить как
    в источнике. Считается от ЭКРАННОГО размера, поэтому анаморфный источник
    (SAR ≠ 1) не выходит сплющенным.
    """
    source_w = media.display_width or media.width
    source_h = media.display_height or media.height
    if source_w <= 0 or source_h <= 0:
        return max(0, longer_side), max(0, shorter_side)

    landscape = source_w >= source_h
    out_w = int(longer_side or 0) if landscape else int(shorter_side or 0)
    out_h = int(shorter_side or 0) if landscape else int(longer_side or 0)

    if out_w <= 0 and out_h <= 0:
        out_w, out_h = source_w, source_h
    elif out_w <= 0:
        out_w = int(round(out_h * source_w / source_h))
    elif out_h <= 0:
        out_h = int(round(out_w * source_h / source_w))

    step = max(1, int(divisible_by))
    if step > 1:
        out_w = (out_w // step) * step
        out_h = (out_h // step) * step

    # Ниже двух пикселей ни один кодек и ни один фильтр не работает.
    return max(2, out_w), max(2, out_h)


def _pixel_plan(media: MediaInfo, want_alpha: bool) -> tuple[str, int]:
    """Пиксельный формат на выходе графа и число каналов.

    Решается ОДИН раз по источнику, а не на каждом кадре. VideoHelperSuite
    всегда тянет ``rgba64le`` — 8 байт на пиксель и промежуточный float64, даже
    для восьмибитного видео без альфы.
    """
    alpha = want_alpha and media.has_alpha
    if media.bit_depth > 8:
        return ("gbrapf32le", 4) if alpha else ("gbrpf32le", 3)
    return ("rgba", 4) if alpha else ("rgb24", 3)


def _build_graph(stream, plan_fmt: str, size: tuple[int, int], rotation: int, flags: str):
    """Собрать граф: поворот → масштаб → формат.

    Поворот делается фильтром ДО масштабирования. Ядро ComfyUI вместо этого
    зовёт ``np.rot90`` на каждый кадр в Python — лишняя полная копия кадра.
    """
    av = _av()
    graph = av.filter.Graph()
    chain = [graph.add_buffer(template=stream)]

    for name, args in _ROTATION_FILTERS.get(rotation, ()):
        chain.append(graph.add(name, args) if args else graph.add(name))

    if size:
        chain.append(graph.add("scale", f"{size[0]}:{size[1]}:flags={flags}"))
        chain.append(graph.add("setsar", "1"))

    chain.append(graph.add("format", plan_fmt))
    sink = graph.add("buffersink")
    chain.append(sink)

    for first, second in zip(chain, chain[1:]):
        first.link_to(second)
    graph.configure()
    return graph, chain[0], sink


def _frame_to_array(frame, channels: int):
    """Кадр графа → numpy в диапазоне 0..1."""
    import numpy as np

    array = frame.to_ndarray()
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[0] != channels:
        array = np.transpose(array, (1, 2, 0))
    return np.clip(array.astype(np.float32), 0.0, 1.0)


def _planar_to_hwc(frame, channels: int):
    """gbrp/gbrap приходят планарно (G,B,R[,A]) — вернуть в порядок RGB[A]."""
    import numpy as np

    array = frame.to_ndarray()
    if array.ndim == 3 and array.shape[0] in (3, 4):
        planes = np.transpose(array, (1, 2, 0))
        order = [2, 0, 1] + ([3] if planes.shape[2] == 4 else [])
        planes = planes[:, :, order]
        return np.clip(planes.astype(np.float32), 0.0, 1.0)[:, :, :channels]
    return _frame_to_array(frame, channels)


def _total_ram() -> int:
    """Сколько ОЗУ у машины, или 0, если узнать не вышло."""
    try:
        import psutil
    except ImportError:                     # pragma: no cover - psutil есть в ComfyUI
        return 0
    try:
        return int(psutil.virtual_memory().total)
    except Exception:                       # noqa: BLE001 - счётчик не должен ронять декод
        return 0


def _available_ram() -> int:
    """Сколько ОЗУ свободно ПРЯМО СЕЙЧАС, или 0. Только для предупреждения.

    ⚠️ Отказывать по этому числу нельзя: оно пляшет от того, какие модели
    сейчас держит ComfyUI, и один и тот же граф то проходил бы, то нет.
    """
    try:
        import psutil
    except ImportError:                     # pragma: no cover
        return 0
    try:
        return int(psutil.virtual_memory().available)
    except Exception:                       # noqa: BLE001
        return 0


def _max_bytes() -> int:
    """Потолок под кадры: переменная окружения → доля ОЗУ → запасное число."""
    raw = os.environ.get(_MAX_BYTES_ENV, "")
    try:
        value = int(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    total = _total_ram()
    if total > 0:
        # Запасное число — именно нижняя граница, а не альтернатива: на машине
        # с 8 ГБ доля дала бы 4.8 ГиБ, то есть отказ там, где раньше работало.
        return max(_FALLBACK_MAX_BYTES, int(total * _TOTAL_RAM_SHARE))
    return _FALLBACK_MAX_BYTES


# Куда кладём кадры, когда их не удержать в ОЗУ. Файл живёт в temp-папке
# ComfyUI: она чистится при старте, так что забытый кусок не копится вечно.
_DISK_PREFIX = "ts_video_frames_"


def _disk_dir() -> pathlib.Path:
    try:
        import folder_paths

        base = pathlib.Path(folder_paths.get_temp_directory())
    except Exception:                       # noqa: BLE001 - вне ComfyUI берём системную
        base = pathlib.Path(tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    return base


def _sweep_disk_frames(directory: pathlib.Path) -> None:
    """Убрать куски от прошлых прогонов.

    ⚠️ Файл, который ещё отображён в память, Windows удалить не даст — и это
    правильно: значит, тензор из него кто-то держит. Такой просто пропускаем.
    """
    for stale in directory.glob(f"{_DISK_PREFIX}*.bin"):
        try:
            stale.unlink()
        except OSError:
            continue


def _allocate_frames(shape: tuple[int, ...], on_disk: bool, tag: str):
    """Тензор под кадры: обычный в ОЗУ или отображённый на файл.

    Отображённый — это НЕ «другой тип данных»: наружу уходит самый обычный
    float32-тензор нужной формы, просто страницы за ним лежат на диске и
    подтягиваются по мере обращения. Замерено: 19.8 ГБ выделяется за 0.00 с и
    занимает в ОЗУ ровно столько, сколько кадров тронули.
    """
    import torch

    numel = 1
    for dim in shape:
        numel *= int(dim)
    if not on_disk:
        return torch.empty(shape, dtype=torch.float32)
    directory = _disk_dir()
    _sweep_disk_frames(directory)
    path = directory / f"{_DISK_PREFIX}{tag}_{os.getpid()}.bin"
    flat = torch.from_file(str(path), shared=True, size=numel, dtype=torch.float32)
    return flat.view(shape)


def _window(media: MediaInfo, req: DecodeRequest) -> tuple[float, float]:
    """Начало и конец куска в секундах. ``end_time = -1`` значит «до конца»."""
    duration = media.duration if media.duration > 0 else 0.0
    start = max(0.0, float(req.start_time))
    end = float(req.end_time)
    if end is None or end <= 0:
        end = duration if duration > 0 else 0.0
    if duration > 0:
        end = min(end, duration)
        start = min(start, max(0.0, end - 1e-3))
    return start, end


def _estimate_frames(media: MediaInfo, req: DecodeRequest, start: float, end: float) -> int:
    fps = req.frame_rate if req.frame_rate > 0 else (media.fps or 25.0)
    window = max(0.0, end - start)
    if window <= 0:
        count = media.frame_count or 1
    else:
        # Полуоткрытое окно: ровно столько кадров, сколько влезает целиком.
        count = int(round(window * fps))
    step = max(1, int(req.frame_step))
    count = (count + step - 1) // step
    if req.max_frames > 0:
        count = min(count, int(req.max_frames))
    return max(1, count)


def _audio_stream(container):
    """Последняя декодируемая аудиодорожка.

    Именно последняя и именно с проверкой: у потока без кодека декодирование
    роняет процесс целиком (тот же приём в ядре ComfyUI).
    """
    for stream in reversed(container.streams.audio):
        if getattr(stream, "codec_context", None) is not None:
            return stream
    return None


def decode(req: DecodeRequest, *, progress: bool = True) -> DecodeResult:
    """Прочитать кусок видео в тензоры.

    Args:
        req: что декодировать.
        progress: показывать ли полосу прогресса ComfyUI.

    Returns:
        Кадры, альфа, звук и фактические параметры куска.

    Raises:
        RuntimeError: файла нет, он не читается, или кусок не влезает в память.
    """
    import numpy as np
    import torch

    av = _av()
    media = probe_cached(req.path, want_peaks=False)
    start, end = _window(media, req)
    size = target_size(media, req.longer_side, req.shorter_side, req.divisible_by)
    plan_fmt, channels = _pixel_plan(media, req.want_alpha)
    estimated = _estimate_frames(media, req, start, end)

    needed = estimate_bytes(estimated, size[0], size[1], channels)
    ceiling = _max_bytes()
    on_disk = False
    if needed > ceiling:
        if not req.allow_disk:
            raise RuntimeError(
                f"{LOG_PREFIX} This clip needs about {human_bytes(needed)} of RAM "
                f"({estimated} frames of {size[0]}x{size[1]}), more than the "
                f"{human_bytes(ceiling)} ceiling. Trim the range on the timeline, set "
                f"width/height, or set max_frames; switch 'when_too_large' to "
                f"'use disk' to keep the frames on disk instead; or raise "
                f"{_MAX_BYTES_ENV} if you know the machine can take it."
            )
        on_disk = True
        logger.info(
            "%s Clip needs %s, over the %s ceiling — keeping the frames on disk.",
            LOG_PREFIX, human_bytes(needed), human_bytes(ceiling),
        )

    # ⚠️ Предупреждаем, но НЕ отказываем: свободная память пляшет от того, что
    # сейчас держит ComfyUI, и отказ по ней сделал бы один и тот же граф то
    # проходящим, то нет. Своп медленный, но это выбор пользователя.
    available = _available_ram()
    if 0 < available < needed:
        logger.warning(
            "%s Clip needs %s but only %s is free right now; the system will page. "
            "Lower width/height or max_frames if it crawls.",
            LOG_PREFIX, human_bytes(needed), human_bytes(available),
        )

    try:
        images = _allocate_frames((estimated, size[1], size[0], 3), on_disk, "rgb")
        alpha = (_allocate_frames((estimated, size[1], size[0]), on_disk, "alpha")
                 if channels == 4 else None)
    except (MemoryError, RuntimeError, OSError) as exc:
        # ⚠️ Настоящий отказ выделения обязан читаться так же, как и наш
        # предварительный: иначе пользователь получает голое MemoryError без
        # единой подсказки, что с этим делать.
        raise RuntimeError(
            f"{LOG_PREFIX} Could not reserve {human_bytes(needed)} for "
            f"{estimated} frames of {size[0]}x{size[1]}. Trim the range on the timeline, "
            f"set width/height, set max_frames, or switch 'when_too_large' to 'use disk'."
        ) from exc

    pbar = None
    interrupt = None
    if progress:
        try:
            from comfy.utils import ProgressBar

            pbar = ProgressBar(estimated)
        except Exception:                   # noqa: BLE001 - вне ComfyUI прогресса нет
            pbar = None
        try:
            from comfy import model_management

            interrupt = model_management.throw_exception_if_processing_interrupted
        except Exception:                   # noqa: BLE001
            interrupt = None

    written = 0
    truncated = False
    audio_payload = None

    with av.open(req.path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = 0             # ×3.7 к скорости — замерено

        time_base = stream.time_base or Fraction(1, 1000)
        start_pts = int(start / time_base)
        end_pts = int(end / time_base) if end > 0 else None

        streams = [stream]
        audio = _audio_stream(container) if req.want_audio else None
        if audio is not None:
            streams.append(audio)

        if start > _SEEK_THRESHOLD:
            container.seek(start_pts, stream=stream)

        graph = source = sink = None
        sampler = FrameSampler(req.frame_rate, start, req.frame_step, end)
        flags = _RESIZE_FLAGS.get(req.resize_filter, "bicubic")
        # ⚠️ Сравнивать надо с КОДИРОВАННЫМ размером, а не с экранным.
        #
        # Без фильтра масштаба граф отдаёт кадр ровно таким, каким он лежит в
        # файле, — то есть `media.width x media.height`. Экранный размер
        # отличается у анаморфного материала (SAR ≠ 1), и там это ломалось
        # насмерть: `target_size` при нулевых сторонах возвращает ЭКРАННЫЙ
        # размер, сравнение с ним давало «масштабировать не нужно», буфер
        # выделялся под экранный размер, а кадры приезжали кодированного —
        # и загрузчик падал на несовпадении формы.
        needs_scale = (size[0], size[1]) != (media.width, media.height)

        audio_chunks: list = []
        audio_rate = 0
        audio_channels = 0
        video_done = False

        for packet in container.demux(*streams):
            if packet.stream.type == "audio":
                if video_done:
                    continue
                for frame in _safe_decode(packet, av):
                    when = float(frame.time or 0.0)
                    if when + float(frame.samples) / max(1, frame.sample_rate) < start:
                        continue
                    if end > 0 and when >= end:
                        continue
                    chunk = _audio_frame_to_float(frame, np)
                    audio_chunks.append((when, chunk))
                    audio_rate = frame.sample_rate or audio_rate
                    audio_channels = max(audio_channels, chunk.shape[0])
                continue

            if video_done:
                break

            for frame in _safe_decode(packet, av):
                if frame.pts is not None and frame.pts < start_pts:
                    # Разгонные кадры после seek декодировать обязаны (межкадровые
                    # зависимости), а масштабировать — нет: в граф не пускаем.
                    continue
                if end_pts is not None and frame.pts is not None and frame.pts >= end_pts:
                    video_done = True
                    break

                when = float(frame.pts * time_base) if frame.pts is not None else start
                repeats = sampler.take(when)
                if repeats <= 0:
                    continue

                if graph is None:
                    graph, source, sink = _build_graph(
                        stream, plan_fmt, size if needs_scale else None,
                        media.rotation, flags)

                source.push(frame)
                while True:
                    try:
                        out_frame = sink.pull()
                    except (av.error.BlockingIOError, av.error.EOFError):
                        break
                    array = (_planar_to_hwc(out_frame, channels)
                             if plan_fmt.startswith("gbr")
                             else _frame_to_array(out_frame, channels))

                    for _ in range(repeats):
                        if written >= estimated:
                            if req.max_frames > 0:
                                truncated = True
                                video_done = True
                                break
                            # ⚠️ Потолок проверяется и ЗДЕСЬ, а не только перед
                            # выделением. Оценка числа кадров берётся из
                            # метаданных; когда они врут (переменная частота,
                            # битый индекс), настоящих кадров оказывается
                            # больше — и рост шёл мимо ограничения, съедая
                            # сколько получится.
                            #
                            # Рост геометрический, а не по 16 кадров: каждое
                            # расширение копирует ВЕСЬ тензор, и постоянный шаг
                            # давал квадратичное копирование — на длинном ролике
                            # это заметнее самой распаковки.
                            grow_by = max(16, images.shape[0] // 4)
                            after = images.shape[0] + grow_by
                            needed_now = estimate_bytes(after, size[0], size[1], channels)
                            if needed_now > ceiling:
                                raise RuntimeError(
                                    f"{LOG_PREFIX} This clip has more frames than its metadata "
                                    f"said and already needs about {human_bytes(needed_now)} of "
                                    f"RAM. Trim the range on the timeline, set the output size, "
                                    f"or set max_frames."
                                )
                            images = torch.cat(
                                [images,
                                 torch.empty_like(images[:1]).repeat(grow_by, 1, 1, 1)], dim=0)
                            if alpha is not None:
                                alpha = torch.cat(
                                    [alpha,
                                     torch.empty_like(alpha[:1]).repeat(grow_by, 1, 1)], dim=0)
                            estimated = images.shape[0]

                        images[written].copy_(torch.from_numpy(
                            np.ascontiguousarray(array[:, :, :3])))
                        if alpha is not None and array.shape[2] == 4:
                            alpha[written].copy_(torch.from_numpy(
                                np.ascontiguousarray(array[:, :, 3])))
                        written += 1
                        if pbar is not None:
                            pbar.update(1)

                    if video_done:
                        break

                if not video_done and req.max_frames > 0 and written >= req.max_frames:
                    truncated = True
                    video_done = True

                # Флаг прерывания одноразовый и недешёвый — раз в 16 кадров хватает.
                if interrupt is not None and (written & 15) == 0:
                    interrupt()

                if video_done:
                    break

        if audio_chunks and audio_rate > 0:
            audio_payload = _assemble_audio(audio_chunks, audio_rate, audio_channels, start, end)

    images = images[:written]
    if alpha is not None:
        alpha = alpha[:written]

    effective_fps = req.frame_rate if req.frame_rate > 0 else (media.fps or 0.0)
    if req.frame_step > 1 and effective_fps > 0:
        effective_fps = effective_fps / req.frame_step

    if media.vfr and req.frame_rate <= 0:
        logger.warning(
            "%s %s has a variable frame rate; set frame_rate to get evenly spaced frames.",
            LOG_PREFIX, safe_log_path(req.path))

    return DecodeResult(
        images=images,
        alpha=alpha,
        audio=audio_payload,
        fps=effective_fps,
        frame_count=written,
        width=size[0],
        height=size[1],
        start_time=start,
        end_time=end,
        truncated=truncated,
        media=media,
    )


def _safe_decode(packet, av):
    """Декодировать пакет, пережив битый.

    Один испорченный пакет посреди файла не должен убивать всю загрузку — тот же
    подход в ядре ComfyUI.
    """
    try:
        return packet.decode()
    except av.error.InvalidDataError:
        return ()
    except Exception as error:              # noqa: BLE001 - редкие ошибки декодера
        logger.debug("%s frame dropped: %s", LOG_PREFIX, error)
        return ()


# Во сколько раз делить целочисленный отсчёт, чтобы он лёг в [-1, 1].
# Ключ — имя формата PyAV без суффикса планарности.
_AUDIO_INT_SCALE = {
    "u8": 128.0,        # беззнаковый: сначала сдвигаем на 128, потом делим
    "s16": 32768.0,
    "s32": 2147483648.0,
    "s64": 9223372036854775808.0,
}


def _audio_frame_to_float(frame, np):
    """Кадр звука → float32 [C, T] в диапазоне [-1, 1].

    ⚠️ ``frame.to_ndarray()`` отдаёт СЫРОЙ формат кадра. У распространённого
    ``s16`` это целые ±32767, и прежний ``astype(np.float32)`` просто расширял
    целое: конвенция ComfyUI AUDIO (float в [-1, 1]) нарушалась, а всё, что
    дальше считало громкость или писало файл, получало клиппинг.

    Второе: у ПАКОВАННЫХ форматов (без ``p`` на конце) PyAV кладёт каналы
    вперемешку в одну строку, поэтому стерео выглядело как одна дорожка вдвое
    длиннее — «искажённое моно». Разворачиваем по числу каналов.
    """
    data = frame.to_ndarray()
    fmt = str(getattr(getattr(frame, "format", None), "name", "") or "")
    planar = fmt.endswith("p")
    base = fmt[:-1] if planar else fmt

    try:
        channels = int(getattr(frame.layout, "nb_channels", 0) or 0)
    except Exception:                       # noqa: BLE001 - старые сборки PyAV
        channels = 0
    if channels <= 0:
        channels = int(data.shape[0]) if data.ndim > 1 else 1

    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Паковано: одна строка, каналы чередуются отсчёт за отсчётом.
    if not planar and data.shape[0] == 1 and channels > 1:
        total = data.shape[1]
        if total % channels == 0:
            data = data.reshape(total // channels, channels).T

    scale = _AUDIO_INT_SCALE.get(base)
    if scale is None:
        # flt/dbl уже в [-1, 1] — только приводим тип.
        return np.ascontiguousarray(data, dtype=np.float32)

    out = data.astype(np.float32)
    if base == "u8":
        out -= 128.0
    return np.ascontiguousarray(out / scale, dtype=np.float32)


def _assemble_audio(chunks, rate: int, channels: int, start: float, end: float) -> dict:
    """Склеить аудиокадры в тензор ComfyUI ``{"waveform": [1,C,T], "sample_rate": int}``."""
    import numpy as np
    import torch

    channels = max(1, channels)
    pieces = []
    for when, chunk in chunks:
        data = chunk
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] < channels:
            data = np.repeat(data[:1], channels, axis=0)
        pieces.append(data[:channels])

    if not pieces:
        return {"waveform": torch.zeros((1, channels, 1), dtype=torch.float32),
                "sample_rate": rate}

    waveform = np.concatenate(pieces, axis=1)

    # Первый кадр после seek почти всегда начинается раньше запрошенного места —
    # подрезаем по смещению, иначе звук поедет относительно картинки.
    first_when = chunks[0][0]
    if first_when < start:
        skip = int(round((start - first_when) * rate))
        waveform = waveform[:, max(0, skip):]

    if end > start:
        limit = int(round((end - start) * rate))
        waveform = waveform[:, :limit]

    tensor = torch.from_numpy(np.ascontiguousarray(waveform.astype(np.float32)))
    return {"waveform": tensor.unsqueeze(0).contiguous(), "sample_rate": int(rate)}


def iter_frames(
    path: str,
    *,
    start: float = 0.0,
    end: float = -1.0,
    fps: float = 0.0,
    size: tuple[int, int] | None = None,
    resize_filter: str = "bicubic",
    limit: int = 0,
) -> Iterator["object"]:
    """Кадры по одному, без накопления тензора.

    Нужен там, где весь ролик в память класть незачем: пересохранение файл→файл
    в сейвере, спрайты миниатюр и прокси для плеера.

    Yields:
        ``numpy.ndarray`` формы ``[H,W,3]``, uint8.
    """
    import numpy as np

    av = _av()
    media = probe_cached(path, want_peaks=False)
    request = DecodeRequest(path=path, start_time=start, end_time=end, frame_rate=fps)
    window_start, window_end = _window(media, request)
    out_size = size or (media.display_width or media.width, media.display_height or media.height)

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = 0
        time_base = stream.time_base or Fraction(1, 1000)
        start_pts = int(window_start / time_base)
        end_pts = int(window_end / time_base) if window_end > 0 else None

        if window_start > _SEEK_THRESHOLD:
            container.seek(start_pts, stream=stream)

        graph = source = sink = None
        sampler = FrameSampler(fps, window_start, 1, window_end)
        produced = 0

        for packet in container.demux(stream):
            for frame in _safe_decode(packet, av):
                if frame.pts is not None and frame.pts < start_pts:
                    continue
                if end_pts is not None and frame.pts is not None and frame.pts >= end_pts:
                    return
                when = float(frame.pts * time_base) if frame.pts is not None else window_start
                # ⚠️ Сэмплер отвечает СКОЛЬКО РАЗ выдать кадр, а не «да/нет».
                # Здесь ответ приводили к «больше нуля» и выдавали ровно один
                # раз, поэтому повышение частоты не дублировало кадры: часовой
                # ролик, пересохранённый с 30 на 60, выходил вдвое короче.
                repeats = sampler.take(when)
                if repeats <= 0:
                    continue

                if graph is None:
                    graph, source, sink = _build_graph(
                        stream, "rgb24", out_size, media.rotation,
                        _RESIZE_FLAGS.get(resize_filter, "bicubic"))
                source.push(frame)
                while True:
                    try:
                        out_frame = sink.pull()
                    except (av.error.BlockingIOError, av.error.EOFError):
                        break
                    array = np.ascontiguousarray(out_frame.to_ndarray(format="rgb24"))
                    for _ in range(repeats):
                        yield array
                        produced += 1
                        if limit and produced >= limit:
                            return


def iter_thumbnails(path: str, *, start: float, step: float, count: int, height: int) -> list:
    """Снять ``count`` миниатюр начиная с ``start`` с шагом ``step`` секунд.

    Каждая миниатюра берётся отдельной перемоткой: при шаге в минуту читать
    подряд бессмысленно, а seek стоит миллисекунды.
    """
    import numpy as np

    av = _av()
    media = probe_cached(path, want_peaks=False)
    source_w = media.display_width or media.width or 16
    source_h = media.display_height or media.height or 9
    # Ширину НЕ округляем до чётной: миниатюра идёт в JPEG без цветовой
    # субдискретизации, а лишний пиксель — это уже заметный перекос пропорций
    # на карточке высотой в семь десятков точек.
    width = max(2, int(round(height * source_w / max(1, source_h))))

    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base or Fraction(1, 1000)
        graph = source = sink = None

        for index in range(count):
            when = start + index * step
            if media.duration > 0 and when > media.duration:
                break
            try:
                container.seek(int(max(0.0, when) / time_base), stream=stream)
            except Exception as error:      # noqa: BLE001 - непозиционируемый контейнер
                logger.debug("%s seek failed at %.3f: %s", LOG_PREFIX, when, error)
                break

            picked = None
            for packet in container.demux(stream):
                for frame in _safe_decode(packet, av):
                    picked = frame
                    break
                if picked is not None:
                    break
            if picked is None:
                break

            # ⚠️ Через ТОТ ЖЕ граф, что и обычное декодирование: `reformat`
            # ничего не знает ни про поворот, ни про неквадратный пиксель, и
            # снятая с телефона вертикальная съёмка ложилась в горизонтальную
            # ячейку сплющенной.
            if graph is None:
                graph, source, sink = _build_graph(
                    stream, "rgb24", (width, height), media.rotation, "area")
            source.push(picked)
            while True:
                try:
                    out_frame = sink.pull()
                except (av.error.BlockingIOError, av.error.EOFError):
                    break
                frames.append(np.ascontiguousarray(out_frame.to_ndarray(format="rgb24")))
                break

    return frames


def resolve_progress_callback(callback: Callable[[int, int], None] | None):
    """Небольшой переходник: наш прогресс наружу без завязки на ComfyUI."""
    if callback is None:
        return lambda _done, _total: None
    return callback
