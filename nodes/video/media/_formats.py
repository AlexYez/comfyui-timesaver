"""Реестр форматов сохранения и проба кодировщиков.

Таблица, а не JSON-файлы: чтение файла на импорте запрещено (§13 CLAUDE.md), а
таблицу в Python видит mypy и импортирует тест без единого обращения к диску.
Фронтенду реестр отдаётся роутом — источник истины при этом остаётся один.

Добавить формат = добавить одну запись в ``FORMATS`` и одну ``Option`` в схему
сейвера. Ничего больше трогать не нужно: кодировщик читает эту же таблицу.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping

from ._common import LOG_PREFIX

logger = logging.getLogger("comfyui_timesaver.ts_video.formats")

_EMPTY: Mapping[str, str] = MappingProxyType({})

# ⚠️ Пробный кадр НЕ МЕНЬШЕ 256×256. Замерено: h264_nvenc на 128×128 отвечает
# «Frame Dimension less than the minimum supported value», то есть проба меньшим
# кадром объявила бы рабочий кодировщик недоступным.
_PROBE_SIZE = 256

# Результаты пробы за время жизни процесса не меняются. Модульный словарь —
# санкционированный способ хранить такой кэш (§5 CLAUDE.md).
_probe_cache: dict[tuple, bool] = {}


@dataclass(frozen=True)
class Quality:
    """Уровень качества, названный словом, а не флагами кодировщика."""

    key: str
    label: str
    options: Mapping[str, str] = _EMPTY
    pix_fmt: str | None = None
    hardware: Mapping[str, Mapping[str, str]] = field(default_factory=lambda: _EMPTY)


@dataclass(frozen=True)
class Format:
    """Контейнер + кодек + набор уровней качества."""

    key: str                                # публичный контракт: уезжает в workflow
    container: str                          # имя формата для av.open
    extension: str
    codec: str
    pix_fmt: str
    label: str
    hint: str = ""
    alpha_pix_fmt: str | None = None
    audio_codec: str | None = None
    audio_options: Mapping[str, str] = _EMPTY
    mux_options: Mapping[str, str] = _EMPTY
    file_mux_options: Mapping[str, str] = _EMPTY
    hw_codecs: tuple[str, ...] = ()
    qualities: tuple[Quality, ...] = ()
    profiles: Mapping[str, str] = _EMPTY
    profile_pix_fmt: Mapping[str, str] = _EMPTY
    # Формат для десятибитной записи, если формат её умеет. Десять бит держат
    # плавные градиенты без полос, но читают такой файл не все программы —
    # поэтому это отдельный выбор, а не умолчание.
    ten_bit_pix_fmt: str | None = None
    browser_playable: bool = False
    supports_metadata: bool = True
    codec_tag: str | None = None
    external_lib: bool = False


_H264_QUALITIES = (
    Quality("draft", "Draft",
            MappingProxyType({"crf": "28", "preset": "veryfast"}),
            hardware=MappingProxyType({
                "h264_nvenc": MappingProxyType({"preset": "p1", "rc": "vbr", "cq": "32"}),
                "h264_qsv": MappingProxyType({"global_quality": "30"}),
                "h264_amf": MappingProxyType({"quality": "speed", "rc": "cqp", "qp_i": "32"}),
            })),
    Quality("good", "Good",
            MappingProxyType({"crf": "23", "preset": "medium"}),
            hardware=MappingProxyType({
                "h264_nvenc": MappingProxyType({"preset": "p4", "rc": "vbr", "cq": "26"}),
                "h264_qsv": MappingProxyType({"global_quality": "25"}),
                "h264_amf": MappingProxyType({"quality": "balanced", "rc": "cqp", "qp_i": "26"}),
            })),
    Quality("high", "High",
            MappingProxyType({"crf": "18", "preset": "slow"}),
            hardware=MappingProxyType({
                "h264_nvenc": MappingProxyType({"preset": "p5", "rc": "vbr", "cq": "22"}),
                "h264_qsv": MappingProxyType({"global_quality": "22"}),
                "h264_amf": MappingProxyType({"quality": "quality", "rc": "cqp", "qp_i": "22"}),
            })),
    Quality("visually lossless", "Visually lossless",
            MappingProxyType({"crf": "15", "preset": "slow"}),
            hardware=MappingProxyType({
                "h264_nvenc": MappingProxyType({"preset": "p6", "rc": "vbr", "cq": "19"}),
                "h264_qsv": MappingProxyType({"global_quality": "20"}),
            })),
    # Без потерь требует 4:4:4 — на 4:2:0 «lossless» было бы враньём: цветовая
    # разность уже прорежена вдвое до кодировщика.
    Quality("lossless", "Lossless",
            MappingProxyType({"crf": "0", "preset": "medium"}),
            pix_fmt="yuv444p"),
)

_HEVC_QUALITIES = (
    Quality("draft", "Draft",
            MappingProxyType({"crf": "30", "preset": "veryfast",
                              "x265-params": "log-level=error"}),
            hardware=MappingProxyType({
                "hevc_nvenc": MappingProxyType({"preset": "p1", "rc": "vbr", "cq": "34"}),
                "hevc_qsv": MappingProxyType({"global_quality": "32"}),
            })),
    Quality("good", "Good",
            MappingProxyType({"crf": "26", "preset": "medium",
                              "x265-params": "log-level=error"}),
            hardware=MappingProxyType({
                "hevc_nvenc": MappingProxyType({"preset": "p4", "rc": "vbr", "cq": "28"}),
                "hevc_qsv": MappingProxyType({"global_quality": "27"}),
            })),
    Quality("high", "High",
            MappingProxyType({"crf": "22", "preset": "slow",
                              "x265-params": "log-level=error"}),
            hardware=MappingProxyType({
                "hevc_nvenc": MappingProxyType({"preset": "p5", "rc": "vbr", "cq": "24"}),
                "hevc_qsv": MappingProxyType({"global_quality": "24"}),
            })),
    Quality("visually lossless", "Visually lossless",
            MappingProxyType({"crf": "18", "preset": "slow",
                              "x265-params": "log-level=error"}),
            hardware=MappingProxyType({
                "hevc_nvenc": MappingProxyType({"preset": "p6", "rc": "vbr", "cq": "20"}),
            })),
)

# Профили ProRes: имя для человека -> номер профиля кодировщика prores_ks.
_PRORES_PROFILES = MappingProxyType({
    "Proxy": "0",
    "LT": "1",
    "422": "2",
    "422 HQ": "3",
    "4444": "4",
    "4444 XQ": "5",
})

# 4444 и 4444 XQ несут альфу — им нужен формат с альфа-каналом.
_PRORES_PIX = MappingProxyType({
    "Proxy": "yuv422p10le",
    "LT": "yuv422p10le",
    "422": "yuv422p10le",
    "422 HQ": "yuv422p10le",
    "4444": "yuv444p10le",
    "4444 XQ": "yuv444p10le",
})


FORMATS: tuple[Format, ...] = (
    Format(
        key="H.264 / MP4",
        container="mp4",
        extension="mp4",
        codec="libx264",
        pix_fmt="yuv420p",
        label="MP4",
        hint="Plays everywhere",
        audio_codec="aac",
        audio_options=MappingProxyType({"b": "192000"}),
        # faststart можно просить только у файла на диске: при записи в память
        # av_write_trailer на нём падает (проверено).
        file_mux_options=MappingProxyType({"movflags": "use_metadata_tags+faststart"}),
        mux_options=MappingProxyType({"movflags": "use_metadata_tags"}),
        hw_codecs=("h264_nvenc", "h264_qsv", "h264_amf"),
        qualities=_H264_QUALITIES,
        browser_playable=True,
    ),
    Format(
        key="H.265 (HEVC) / MP4",
        container="mp4",
        extension="mp4",
        codec="libx265",
        pix_fmt="yuv420p",
        ten_bit_pix_fmt="yuv420p10le",
        label="MP4",
        hint="Half the size of H.264",
        audio_codec="aac",
        audio_options=MappingProxyType({"b": "192000"}),
        file_mux_options=MappingProxyType({"movflags": "use_metadata_tags+faststart"}),
        mux_options=MappingProxyType({"movflags": "use_metadata_tags"}),
        # ⚠️ Без этой метки QuickTime и монтажные программы файл не открывают:
        # они узнают HEVC именно по `hvc1`, а не по `hev1`, который ставится по
        # умолчанию.
        codec_tag="hvc1",
        hw_codecs=("hevc_nvenc", "hevc_qsv", "hevc_amf"),
        qualities=_HEVC_QUALITIES,
        # Браузеры проигрывают HEVC далеко не везде, поэтому плеер в ноде
        # показывает его через ту же маленькую копию, что и ProRes.
        browser_playable=False,
        external_lib=True,
    ),
    Format(
        key="ProRes / MOV",
        container="mov",
        extension="mov",
        codec="prores_ks",
        pix_fmt="yuv422p10le",
        alpha_pix_fmt="yuva444p10le",
        label="MOV",
        hint="Editing codec",
        # Монтажные программы ждут в MOV несжатый звук, а не AAC.
        audio_codec="pcm_s16le",
        file_mux_options=MappingProxyType({"movflags": "use_metadata_tags+faststart"}),
        mux_options=MappingProxyType({"movflags": "use_metadata_tags"}),
        profiles=_PRORES_PROFILES,
        profile_pix_fmt=_PRORES_PIX,
        browser_playable=False,
    ),
)

FORMATS_BY_KEY = MappingProxyType({fmt.key: fmt for fmt in FORMATS})
DEFAULT_FORMAT = FORMATS[0].key


def get_format(key: str) -> Format:
    """Найти формат по ключу; неизвестный ключ — внятная ошибка, а не KeyError."""
    fmt = FORMATS_BY_KEY.get(str(key))
    if fmt is None:
        known = ", ".join(FORMATS_BY_KEY)
        raise RuntimeError(f"{LOG_PREFIX} Unknown format '{key}'. Available: {known}.")
    return fmt


def get_quality(fmt: Format, key: str) -> Quality:
    """Уровень качества формата; неизвестный откатывается к первому."""
    for quality in fmt.qualities:
        if quality.key == key:
            return quality
    if fmt.qualities:
        return fmt.qualities[0]
    return Quality("default", "Default")


def probe_encoder(
    codec: str,
    pix_fmt: str,
    options: Mapping[str, str] | None = None,
    container_format: str = "mp4",
) -> bool:
    """Открывается ли такой кодировщик на самом деле.

    ⚠️ Наличие имени в ``av.codecs_available`` не значит НИЧЕГО: на этой машине
    ``av1_nvenc`` в списке есть, а открыть его нельзя (нужна другая карта).
    Единственная надёжная проверка — закодировать один настоящий кадр.

    ⚠️ Контейнер обязателен и обязан быть СВОИМ: ProRes в mp4 не принимается, и
    проба в mp4 объявляла его недоступным, хотя запись в mov работает.

    Никогда не бросает: не открылось — значит недоступно.
    """
    key = (codec, pix_fmt, container_format, tuple(sorted((options or {}).items())))
    cached = _probe_cache.get(key)
    if cached is not None:
        return cached

    ok = False
    try:
        import av

        with av.open(io.BytesIO(), mode="w", format=container_format) as container:
            stream = container.add_stream(codec, rate=Fraction(24, 1))
            stream.width = _PROBE_SIZE
            stream.height = _PROBE_SIZE
            stream.pix_fmt = pix_fmt
            if options:
                stream.options = dict(options)
            frame = av.VideoFrame(_PROBE_SIZE, _PROBE_SIZE, pix_fmt)
            for _ in stream.encode(frame):
                pass
        ok = True
    except Exception as error:              # noqa: BLE001 - недоступность это норма
        logger.debug("%s encoder %s/%s in %s unavailable: %s",
                     LOG_PREFIX, codec, pix_fmt, container_format, error)

    _probe_cache[key] = ok
    return ok


def pick_hardware_codec(fmt: Format, quality: Quality) -> tuple[str, Mapping[str, str]] | None:
    """Выбрать доступный аппаратный кодировщик — по порядку предпочтения.

    Возвращает ``None``, если ни один не открылся: тогда пишем программным, а не
    роняем прогон. Аппаратное кодирование быстрее в разы, но при равном размере
    заметно хуже по качеству, поэтому включается только по просьбе человека.
    """
    for codec in fmt.hw_codecs:
        options = quality.hardware.get(codec)
        if options is None:
            continue
        if probe_encoder(codec, fmt.pix_fmt, options, fmt.container):
            return codec, options
    return None


def catalog(*, check_hardware: bool = False) -> dict:
    """Реестр в виде, пригодном для отправки фронтенду.

    Args:
        check_hardware: пробовать ли открывать аппаратные кодировщики. Проба
            занимает десятки миллисекунд на кандидата, поэтому по умолчанию
            выключена.
    """
    formats = []
    for fmt in FORMATS:
        entry = {
            "key": fmt.key,
            "label": fmt.label,
            "hint": fmt.hint,
            "extension": fmt.extension,
            "browser_playable": fmt.browser_playable,
            "supports_audio": bool(fmt.audio_codec),
            "supports_alpha": bool(fmt.alpha_pix_fmt),
            "supports_metadata": fmt.supports_metadata,
            # ⚠️ Пробуем С НАСТРОЙКАМИ первого качества, а не голым кодеком:
            # x265 без `log-level` вываливает в консоль ComfyUI два десятка
            # строк про свою сборку при каждой пробе.
            "available": probe_encoder(
                fmt.codec, fmt.pix_fmt,
                fmt.qualities[0].options if fmt.qualities else None, fmt.container),
            "qualities": [{"key": q.key, "label": q.label} for q in fmt.qualities],
            "profiles": list(fmt.profiles),
            "hardware": [],
        }
        if check_hardware:
            for codec in fmt.hw_codecs:
                options = fmt.qualities[0].hardware.get(codec) if fmt.qualities else None
                entry["hardware"].append({
                    "codec": codec,
                    "available": probe_encoder(codec, fmt.pix_fmt, options, fmt.container),
                })
        formats.append(entry)
    return {"schema": 1, "default": DEFAULT_FORMAT, "formats": formats}
