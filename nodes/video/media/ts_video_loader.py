"""TS Video Loader — кадры, звук и метаданные одного куска видео.

Кусок выбирается ВИЗУАЛЬНО, на таймлайне в теле ноды: там же плеер, зум и
зацикленный проигрыш выделенного. Числа ``start_seconds``/``end_seconds`` — это
просто то, что записал редактор.

Почему нода быстрая: весь ресайз, поворот и конверсию формата делает граф
фильтров PyAV внутри цикла декодирования, а перемотка обрывает чтение по концу
куска. Двухсекундный фрагмент часового 4K-ролика читается за время перемотки
плюс две секунды, а не за час. Подробности и замеры — в ``_decode.py``.

Выход ``video`` — это ТОТ ЖЕ файл с той же подрезкой, но без ресайза и без смены
частоты: ленивая ссылка для нод, работающих с ядровым типом VIDEO. Стоит ноль,
потому что ``VideoFromFile`` ничего не декодирует в конструкторе.
"""

from __future__ import annotations

import hashlib
import logging
import os

from comfy_api.v0_0_2 import IO

from ._common import (
    LOG_PREFIX,
    VIDEO_INFO_TYPE,
    build_video_info,
    is_video_path,
    resolve_media_path,
    safe_log_path,
)

logger = logging.getLogger("comfyui_timesaver.ts_video.loader")

# Роуты нужны фронтенду независимо от того, какая нода создана первой, поэтому
# регистрируются импортом модуля.
from . import _routes  # noqa: E402,F401  (registers HTTP routes at import)


class TS_VideoLoader(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_VideoLoader",
            display_name="TS Video Loader",
            category="TS/Video",
            description=(
                "Load a video: frames, audio and a compact video_info bundle. Trim the "
                "clip visually on the timeline inside the node, resample the frame rate "
                "and resize while decoding."
            ),
            search_aliases=["load video", "video loader", "import video", "trim video",
                            "vhs load video"],
            inputs=[
                IO.String.Input(
                    "source_path",
                    default="",
                    socketless=True,
                    tooltip=(
                        "Video file. Pick or drop one in the node's own interface; a "
                        "path to a file on the ComfyUI machine works too."
                    ),
                ),
                IO.Float.Input(
                    "start_seconds",
                    default=0.0,
                    min=0.0,
                    max=1_000_000.0,
                    step=0.001,
                    socketless=True,
                    tooltip="Start of the clip. Drag the left handle on the timeline.",
                ),
                IO.Float.Input(
                    "end_seconds",
                    default=-1.0,
                    min=-1.0,
                    max=1_000_000.0,
                    step=0.001,
                    socketless=True,
                    tooltip=(
                        "End of the clip. -1 means \"until the end of the file\". Drag "
                        "the right handle on the timeline."
                    ),
                ),
                IO.Float.Input(
                    "frame_rate",
                    default=0.0,
                    min=0.0,
                    max=240.0,
                    step=0.001,
                    tooltip=(
                        "Output frame rate. 0 keeps the source rate. Fractional rates "
                        "such as 23.976 are allowed."
                    ),
                ),
                IO.Int.Input(
                    "max_frames",
                    default=0,
                    min=0,
                    max=1_000_000,
                    tooltip=(
                        "Hard cap on the number of loaded frames. 0 means no cap. It "
                        "trims the tail of whatever the timeline selected, after the "
                        "frame rate and the frame step have been applied — the node's "
                        "status line always shows how many frames actually come out."
                    ),
                ),
                # ⚠️ СТОРОНЫ, А НЕ ШИРИНА С ВЫСОТОЙ. Материал приходит и
                # горизонтальный, и вертикальный, и «ширина 1024» на портретном
                # ролике значит совсем не то, что на альбомном. «Длинная сторона
                # 1024» одинаково верна для обоих, и один и тот же граф работает
                # со съёмкой любой ориентации.
                IO.Int.Input(
                    "longer_side",
                    default=0,
                    min=0,
                    max=16384,
                    step=2,
                    advanced=True,
                    tooltip=(
                        "Length of the longer side of the frame. 0 derives it from the "
                        "shorter side, keeping the aspect ratio. Works the same on "
                        "landscape and portrait footage. Resizing happens inside the "
                        "decoder, not afterwards."
                    ),
                ),
                IO.Int.Input(
                    "shorter_side",
                    default=0,
                    min=0,
                    max=16384,
                    step=2,
                    advanced=True,
                    tooltip="Length of the shorter side. 0 derives it from the longer one.",
                ),
                IO.Int.Input(
                    "divisible_by",
                    default=1,
                    min=1,
                    max=64,
                    advanced=True,
                    tooltip=(
                        "Round the output size down to a multiple of this value. Video "
                        "models usually want 8, 16 or 32."
                    ),
                ),
                IO.Int.Input(
                    "frame_step",
                    default=1,
                    min=1,
                    max=1000,
                    advanced=True,
                    tooltip="Keep every Nth frame after the frame-rate stage. 1 keeps them all.",
                ),
                IO.Combo.Input(
                    "resize_filter",
                    options=["bicubic", "bilinear", "lanczos", "area", "neighbor"],
                    # area по умолчанию: кадры почти всегда УМЕНЬШАЮТ, а при
                    # уменьшении усреднение по площади даёт меньше муара и шума,
                    # чем интерполяция.
                    default="area",
                    advanced=True,
                    tooltip=(
                        "Scaling filter. area suits strong downscaling, lanczos keeps "
                        "edges sharp, neighbor is for pixel art."
                    ),
                ),
                # ⚠️ Дописан В КОНЕЦ и optional: widgets_values позиционный, а
                # граф в API-формате, сохранённый до этого входа, его не несёт.
                IO.Combo.Input(
                    "when_too_large",
                    options=["stop", "use disk"],
                    default="stop",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "What to do when the frames will not fit in RAM. The ceiling is "
                        "60% of the machine's memory (at least 8 GB), or whatever "
                        "TS_VIDEO_MAX_BYTES says.\n"
                        "• stop (default) — refuse with a message naming the size, so "
                        "nothing quietly starts paging.\n"
                        "• use disk — put the frames in a memory-mapped file in the "
                        "ComfyUI temp folder. What comes out is an ordinary IMAGE "
                        "tensor, and the allocation cannot fail outright however big "
                        "the clip is.\n"
                        "Be honest about the cost: decoding writes every frame once, so "
                        "memory still climbs towards the full size while it runs "
                        "(measured: a 31.9 GB clip took 92 s on disk against 51 s in "
                        "RAM). What you gain is that those pages are backed by a real "
                        "file, so the system can drop them instead of failing, and "
                        "downstream only the frames a node touches are read back. Make "
                        "sure the temp drive has room; the file goes away when the run "
                        "releases the tensor, and leftovers are swept on the next decode."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="images",
                    tooltip="Decoded frames [B,H,W,C], float32 in 0..1.",
                ),
                IO.Audio.Output(
                    display_name="audio",
                    tooltip=(
                        "Audio for the same time range. Nothing at all when the file has "
                        "no audio track — silence would quietly overwrite a real track "
                        "further down the graph."
                    ),
                ),
                VIDEO_INFO_TYPE.Output(
                    display_name="video_info",
                    tooltip="Metadata bundle. Feed it into TS Video Info to get the numbers.",
                ),
                IO.Video.Output(
                    display_name="video",
                    tooltip=(
                        "The trimmed source file itself, without resizing or frame-rate "
                        "change. A cheap reference for nodes that take VIDEO."
                    ),
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        source_path: str = "",
        start_seconds: float = 0.0,
        end_seconds: float = -1.0,
        **_kwargs,
    ) -> bool | str:
        path = resolve_media_path(source_path)
        if not path:
            return "Pick a video file first."
        if not is_video_path(path):
            return f"Not a video file: {safe_log_path(path)}"
        if not os.path.isfile(path):
            return f"File not found: {safe_log_path(path)}"
        if start_seconds < 0:
            return "start_seconds cannot be negative."
        if end_seconds != -1.0 and end_seconds <= start_seconds:
            return "end_seconds must be greater than start_seconds (or -1 for the whole file)."
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> str:
        """Пересчитывать граф, когда изменился файл или любой параметр.

        В отпечаток входят размер и время правки файла: перезапись исходника тем
        же именем обязана инвалидировать кэш, иначе человек правит видео и не
        понимает, почему ничего не поменялось.
        """
        path = resolve_media_path(kwargs.get("source_path", ""))
        parts = [os.path.normcase(path)]
        try:
            stat = os.stat(path)
            parts += [str(stat.st_size), str(stat.st_mtime_ns)]
        except OSError:
            parts.append("missing")
        for key in sorted(kwargs):
            if key != "source_path":
                parts.append(f"{key}={kwargs[key]}")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    @classmethod
    def execute(
        cls,
        source_path: str = "",
        start_seconds: float = 0.0,
        end_seconds: float = -1.0,
        frame_rate: float = 0.0,
        max_frames: int = 0,
        longer_side: int = 0,
        shorter_side: int = 0,
        divisible_by: int = 1,
        frame_step: int = 1,
        resize_filter: str = "area",
        when_too_large: str = "stop",
    ) -> IO.NodeOutput:
        from ._decode import DecodeRequest, decode

        path = resolve_media_path(source_path)
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"{LOG_PREFIX} No video file selected.")

        result = decode(DecodeRequest(
            path=path,
            start_time=float(start_seconds),
            end_time=float(end_seconds),
            frame_rate=float(frame_rate),
            max_frames=int(max_frames),
            longer_side=int(longer_side),
            shorter_side=int(shorter_side),
            divisible_by=int(divisible_by),
            frame_step=int(frame_step),
            resize_filter=str(resize_filter),
            # Необязательный вход из старого графа приходит None, а не пропуском.
            allow_disk=(when_too_large or "stop") == "use disk",
        ))

        media = result.media
        info = build_video_info(
            producer="TS_VideoLoader",
            source={
                "fps": media.fps,
                "fps_exact": list(media.fps_exact),
                "frame_count": media.frame_count,
                "frame_count_estimated": media.frame_count_estimated,
                "duration": media.duration,
                "width": media.display_width or media.width,
                "height": media.display_height or media.height,
                "codec": media.codec,
                "container": media.container,
                "pix_fmt": media.pix_fmt,
                "bit_depth": media.bit_depth,
                "rotation": media.rotation,
                "sar": list(media.sar),
                "vfr": media.vfr,
                "has_audio": media.has_audio,
                "has_alpha": media.has_alpha,
            },
            loaded={
                "fps": result.fps,
                "frame_count": result.frame_count,
                "duration": (result.frame_count / result.fps) if result.fps > 0 else 0.0,
                "width": result.width,
                "height": result.height,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "frame_step": int(frame_step),
                "resize_filter": str(resize_filter),
                "truncated_by_max_frames": result.truncated,
            },
            audio=({"sample_rate": media.audio.sample_rate,
                    "channels": media.audio.channels,
                    "codec": media.audio.codec} if media.audio else None),
            filename=media.filename,
            annotated=str(source_path),
        )

        if result.truncated:
            logger.info("%s stopped at max_frames=%d for %s",
                        LOG_PREFIX, max_frames, safe_log_path(path))

        return IO.NodeOutput(result.images, result.audio, info,
                             cls._as_video(path, result.start_time, result.end_time))

    @staticmethod
    def _as_video(path: str, start: float, end: float):
        """Ленивая ссылка на подрезанный исходник для ядрового типа VIDEO.

        Ничего не декодирует: ``VideoFromFile`` — это обёртка над путём. Если
        ядро вдруг окажется старым и типа не будет, выход просто станет пустым, а
        остальные три продолжат работать.
        """
        try:
            from comfy_api.input_impl import VideoFromFile

            duration = max(0.0, end - start) if end > start else 0.0
            return VideoFromFile(path, start_time=float(start), duration=float(duration))
        except Exception as error:          # noqa: BLE001 - ядро без VIDEO
            logger.debug("%s VIDEO output unavailable: %s", LOG_PREFIX, error)
            return None


NODE_CLASS_MAPPINGS = {"TS_VideoLoader": TS_VideoLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_VideoLoader": "TS Video Loader"}
