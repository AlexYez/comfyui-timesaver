"""Поиск склеек: где план сменился другим планом.

Метрика — расхождение ЯРКОСТНЫХ ГИСТОГРАММ соседних кадров. Выбрана замером на
живом материале (диалоговая сцена, 78 с, 1964 сравнения):

* попиксельная разность (MAD) не годится вовсе — её максимум на настоящей
  склейке оказался 0,198 при среднем 0,005, то есть ни один абсолютный порог не
  отделяет склейку от движения в кадре;
* гистограммная разность на тех же склейках даёт 0,13…0,54 при среднем 0,016 —
  разрыв на порядок, и порог работает.

Все найденные точки проверены ГЛАЗАМИ по кадрам до и после: восемь оказались
настоящей сменой плана, а самая заметная не-склейка (тот же кадр, ничего не
поменялось) осталась на 0,051. Порог по умолчанию стоит в этом разрыве.

⚠️ Кэшируется СЫРАЯ метрика, а не список склеек. Иначе смена порога стоила бы
нового прохода по файлу (4,3 с на 78 секундах SD, и минуты на длинном 4K), хотя
считать заново нечего — порог применяется к уже снятым числам.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ._common import LOG_PREFIX, cache_dir, file_identity, safe_log_path

logger = logging.getLogger("comfyui_timesaver.ts_video.scenes")

# Версия снимка метрики. Поднимать при ЛЮБОЙ правке расчёта: записи на диске
# переживают обновление пака и иначе вернут числа, посчитанные старым кодом.
SCENES_SCHEMA = 1

ANALYSIS_WIDTH = 64          # во что ужимается кадр перед сравнением
HISTOGRAM_BINS = 32
# Ниже этого значения хранить нечего: на замеренном материале 95% кадров дают
# меньше 0,03, и держать их в кэше — это мегабайты ради шума.
MIN_KEEP = 0.04

# ⚠️ Порог выбран по КАДРАМ, а не по красоте числа. На проверенном материале
# восемь настоящих склеек дали 0,540 0,504 0,474 0,473 0,458 0,393 0,198 и
# 0,132, а самая заметная НЕ-склейка (тот же план, ничего не изменилось) —
# 0,051. Порог 0,2, который просился по верхней группе, молча терял две
# последние настоящие; 0,10 стоит в разрыве и держит запас с обеих сторон.
DEFAULT_THRESHOLD = 0.10
DEFAULT_MIN_GAP = 0.40       # секунды; две склейки ближе — это одна склейка
MIN_THRESHOLD = MIN_KEEP


def _av():
    try:
        import av
    except Exception as error:              # noqa: BLE001 - зависимость необязательна вне ComfyUI
        raise RuntimeError(
            f"{LOG_PREFIX} PyAV is required to look for cuts. It normally ships with "
            f"ComfyUI; install it with `pip install av`. ({error})"
        ) from error
    return av


def _grey_graph(av, stream, width: int):
    """Граф фильтров: кадр -> уменьшенный градиент серого.

    Масштабирование делает PyAV, а не numpy: ресайз в графе фильтров уже
    измерялся на 4K (0,43 с против 5,9 с) и остаётся самым дешёвым местом.
    """
    source_width = int(stream.codec_context.width or width)
    source_height = int(stream.codec_context.height or width)
    height = max(8, int(round(width * source_height / max(1, source_width))))

    graph = av.filter.Graph()
    buffer = graph.add_buffer(template=stream)
    scale = graph.add("scale", f"{width}:{height}")
    grey = graph.add("format", "gray")
    sink = graph.add("buffersink")
    buffer.link_to(scale)
    scale.link_to(grey)
    grey.link_to(sink)
    graph.configure()
    return graph


def scan_activity(path: str, *, width: int = ANALYSIS_WIDTH, on_progress=None) -> dict:
    """Пройти файл и снять, насколько каждый кадр не похож на предыдущий.

    Args:
        path: абсолютный путь к видеофайлу.
        width: ширина, до которой ужимается кадр перед сравнением.
        on_progress: необязательный ``callable(done_seconds, duration)``.

    Returns:
        Словарь снимка: только заметные точки (``value >= MIN_KEEP``), общее
        число сравнений и длительность.
    """
    import numpy as np

    av = _av()
    times: list[float] = []
    values: list[float] = []
    compared = 0
    last_time = 0.0

    with av.open(path) as container:
        streams = container.streams.video
        if not streams:
            raise RuntimeError(f"{LOG_PREFIX} No video stream in {safe_log_path(path)}.")
        stream = streams[0]
        stream.thread_type = "AUTO"
        graph = _grey_graph(av, stream, width)

        previous = None
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except Exception:               # noqa: BLE001 - битый пакет не повод бросать файл
                continue
            for frame in frames:
                graph.push(frame)
                while True:
                    try:
                        small = graph.pull()
                    except Exception:       # noqa: BLE001 - EOF/EAGAIN у графа
                        break
                    grey = small.to_ndarray().astype(np.float32) / 255.0
                    histogram = np.histogram(grey, bins=HISTOGRAM_BINS, range=(0.0, 1.0))[0]
                    histogram = histogram.astype(np.float32)
                    total = float(histogram.sum())
                    if total > 0:
                        histogram /= total
                    moment = float(small.time or 0.0)
                    if previous is not None:
                        compared += 1
                        # Половина L1-расстояния: 0 — гистограммы совпали,
                        # 1 — не пересекаются нигде.
                        value = float(np.abs(histogram - previous).sum() / 2.0)
                        if value >= MIN_KEEP:
                            times.append(round(moment, 3))
                            values.append(round(value, 4))
                    previous = histogram
                    last_time = max(last_time, moment)
                    if on_progress is not None and compared % 120 == 0:
                        on_progress(last_time)

    logger.info("%s scanned %s: %d comparison(s), %d notable",
                LOG_PREFIX, safe_log_path(path), compared, len(times))
    return {
        "schema": SCENES_SCHEMA,
        "times": times,
        "values": values,
        "compared": compared,
        "duration": round(last_time, 3),
        "min_keep": MIN_KEEP,
    }


def cuts_from_activity(activity: dict, *, threshold: float, min_gap: float) -> list[float]:
    """Применить порог к снятому снимку.

    Соседние всплески ближе ``min_gap`` — одна склейка: победитель тот, у кого
    значение больше. Без этого один переход, размазанный на пару кадров, давал
    бы два маркера в паре сотых секунды друг от друга.
    """
    times = list(activity.get("times") or ())
    values = list(activity.get("values") or ())
    level = max(MIN_THRESHOLD, float(threshold))
    gap = max(0.0, float(min_gap))

    hits = [(t, v) for t, v in zip(times, values) if v >= level]
    if not hits:
        return []

    merged: list[list[float]] = []
    for moment, value in hits:
        if merged and moment - merged[-1][0] <= gap:
            if value > merged[-1][1]:
                merged[-1] = [moment, value]
            continue
        merged.append([moment, value])
    return [round(float(moment), 3) for moment, _ in merged]


def _cache_file(path: str) -> Path:
    return cache_dir() / f"{file_identity(path)}_scenes.json"


def activity_cached(path: str, *, on_progress=None) -> dict:
    """Снимок метрики для файла: с диска, иначе посчитать и сохранить."""
    target = _cache_file(path)
    if target.is_file():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            # Снимок, снятый с более грубым отбором, чем нужен сейчас, не
            # годится: в нём просто нет точек, которые теперь интересны.
            kept = float(payload.get("min_keep", 1.0))
            if payload.get("schema") == SCENES_SCHEMA and kept <= MIN_KEEP + 1e-9:
                return payload
        except Exception as error:          # noqa: BLE001 - битый кэш просто перечитаем
            logger.debug("%s scene cache unreadable: %s", LOG_PREFIX, error)

    activity = scan_activity(path, on_progress=on_progress)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(activity), encoding="utf-8")
    except OSError as error:
        logger.debug("%s could not write scene cache: %s", LOG_PREFIX, error)
    return activity


def detect_cuts(
    path: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_gap: float = DEFAULT_MIN_GAP,
    on_progress=None,
) -> dict:
    """Склейки файла и то, из чего они получены.

    Returns:
        ``{"cuts": [...], "threshold": float, "min_gap": float,
        "duration": float, "compared": int}``.
    """
    activity = activity_cached(path, on_progress=on_progress)
    cuts = cuts_from_activity(activity, threshold=threshold, min_gap=min_gap)
    return {
        "cuts": cuts,
        "threshold": max(MIN_THRESHOLD, float(threshold)),
        "min_gap": max(0.0, float(min_gap)),
        "duration": float(activity.get("duration") or 0.0),
        "compared": int(activity.get("compared") or 0),
    }
