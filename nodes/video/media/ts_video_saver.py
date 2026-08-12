"""TS Video Saver — сохранить кадры видеофайлом, с плеером прямо в ноде.

Формат и качество выбираются словами, а не флагами кодировщика: «MP4 → хорошее»
вместо ``-crf 23 -preset medium``. За это отвечает ``DynamicCombo``: подпараметры
выбранного формата ComfyUI показывает сам, и в классическом режиме нод, и в
Nodes 2.0 (проверено).

Звук муксится в тот же проход, в тот же файл. Ни временного WAV, ни второго
запуска ffmpeg, ни файла-дубля «-audio» — см. ``_encode.py``.

На вход можно подать либо кадры, либо ядровой тип VIDEO. Во втором случае кадры
идут из файла потоком и тензор не строится вовсе — пересохранение часового
ролика в ProRes не требует держать его в памяти.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from comfy_api.v0_0_2 import IO

from ._common import LOG_PREFIX, safe_log_path
from ._formats import DEFAULT_FORMAT

logger = logging.getLogger("comfyui_timesaver.ts_video.saver")

PREVIEW_UI_KEY = "ts_video_saver"

# Превью в браузере: маленький H.264 рядом с настоящим файлом, если настоящий
# браузер не проигрывает (ProRes). Ширина ограничена — превью не должно стоить
# дороже самой записи.
_PROXY_MAX_WIDTH = 1280


class TS_VideoSaver(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_VideoSaver",
            display_name="TS Video Saver",
            category="TS/Video",
            description=(
                "Save frames as a video file. H.264/MP4 for sharing, ProRes/MOV for "
                "editing. Audio is muxed in the same pass, and a player shows the "
                "result right in the node."
            ),
            search_aliases=["save video", "export video", "prores", "video combine",
                            "mp4", "mov"],
            is_output_node=True,
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            inputs=[
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Frames to save. Leave empty when a video is connected instead.",
                ),
                IO.Float.Input(
                    "fps",
                    default=24.0,
                    min=0.01,
                    max=240.0,
                    step=0.01,
                    tooltip="Frame rate of the saved file.",
                ),
                IO.String.Input(
                    "filename_prefix",
                    default="video/TS",
                    tooltip=(
                        "Where to save, relative to the output folder. Subfolders are "
                        "created for you. Date tokens work anywhere in the path — "
                        "%date:yyyy-MM-dd% and %date:hhmmss% (yyyy yy MM M dd d hh h "
                        "mm m ss s), as do ComfyUI's own %year% %month% %day% %hour% "
                        "%minute% %second% %width% %height%. Example: "
                        "videos/my-run/my-run-%date:yyyy-MM-dd%_%date:hhmmss%"
                    ),
                ),
                # Умолчание задаётся ПОРЯДКОМ: первый вариант и есть выбранный по
                # умолчанию — отдельного `default` у DynamicCombo нет.
                IO.DynamicCombo.Input(
                    "format",
                    tooltip="Container and codec. Quality is written in words, not encoder flags.",
                    options=[
                        IO.DynamicCombo.Option("H.264 / MP4", [
                            IO.Combo.Input(
                                "quality",
                                options=["draft", "good", "high",
                                         "visually lossless", "lossless"],
                                default="good",
                                tooltip="Higher quality means a bigger file.",
                            ),
                        ]),
                        IO.DynamicCombo.Option("H.265 (HEVC) / MP4", [
                            IO.Combo.Input(
                                "hevc_quality",
                                options=["draft", "good", "high", "visually lossless"],
                                default="good",
                                tooltip=(
                                    "H.265 gives roughly half the size of H.264 at the "
                                    "same quality, and takes longer to encode."
                                ),
                            ),
                            IO.Boolean.Input(
                                "ten_bit",
                                default=False,
                                tooltip=(
                                    "10-bit keeps gradients smooth without banding. Some "
                                    "players and editors cannot read it."
                                ),
                            ),
                        ]),
                        IO.DynamicCombo.Option("ProRes / MOV", [
                            IO.Combo.Input(
                                "profile",
                                options=["Proxy", "LT", "422", "422 HQ", "4444", "4444 XQ"],
                                default="422 HQ",
                                tooltip=(
                                    "Editing codec profile. Proxy and LT are light, "
                                    "422 HQ is the usual choice, 4444 is the heaviest."
                                ),
                            ),
                        ]),
                    ],
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Audio track. Trimmed or padded to the length of the video.",
                ),
                IO.Video.Input(
                    "video",
                    optional=True,
                    tooltip=(
                        "Re-save an existing video instead of frames. Streams straight "
                        "from the file, so nothing is held in memory."
                    ),
                ),
                IO.Combo.Input(
                    "encoder",
                    options=["software", "hardware if available"],
                    default="software",
                    advanced=True,
                    tooltip=(
                        "Hardware encoding is much faster but noticeably worse at the "
                        "same file size, so it is never chosen for you."
                    ),
                ),
                IO.Boolean.Input(
                    "save_metadata",
                    default=True,
                    advanced=True,
                    tooltip="Embed the prompt and workflow into the file, the way PNGs carry them.",
                ),
                IO.Combo.Input(
                    "preview",
                    options=["auto", "off"],
                    default="auto",
                    advanced=True,
                    tooltip=(
                        "auto shows the result in the node, making a small H.264 proxy "
                        "when the saved format is not playable in a browser."
                    ),
                ),
            ],
            outputs=[
                IO.Video.Output(display_name="video", tooltip="The saved file, as a VIDEO reference."),
                IO.String.Output(display_name="path", tooltip="Full path of the saved file."),
            ],
        )

    @classmethod
    def execute(
        cls,
        images=None,
        fps: float = 24.0,
        filename_prefix: str = "video/TS",
        format=None,                        # noqa: A002 - имя входа задано схемой
        audio=None,
        video=None,
        encoder: str = "software",
        save_metadata: bool = True,
        preview: str = "auto",
    ) -> IO.NodeOutput:
        from ._encode import output_path, write_video
        from ._formats import get_format

        choice = format if isinstance(format, dict) else {"format": str(format or DEFAULT_FORMAT)}
        format_key = str(choice.get("format") or DEFAULT_FORMAT)
        # У каждого варианта свой ключ качества: DynamicCombo не разрешает двум
        # вариантам делить одно имя подвиджета.
        quality_key = str(choice.get("quality") or choice.get("hevc_quality") or "")
        profile = str(choice.get("profile") or "")
        ten_bit = bool(choice.get("ten_bit"))
        fmt = get_format(format_key)

        frames, frame_count = cls._source_frames(images, video, float(fps))
        if frames is None:
            raise RuntimeError(
                f"{LOG_PREFIX} Nothing to save: connect either images or a video.")

        target, filename, subfolder = output_path(filename_prefix, fmt.extension)
        metadata = cls._metadata() if save_metadata else None
        wants_proxy = preview != "off" and not fmt.browser_playable
        progress = cls._progress(frame_count, with_proxy=wants_proxy)

        result = write_video(
            frames,
            path=target,
            format_key=format_key,
            quality_key=quality_key,
            profile=profile,
            ten_bit=ten_bit,
            fps=float(fps),
            audio=audio,
            frame_count=frame_count,
            metadata=metadata,
            use_hardware=(encoder == "hardware if available"),
            on_frame=progress,
        )

        # ⚠️ Два адреса в одной посылке, и путать их нельзя. Верхний уровень —
        # то, что ОТКРЫВАЕТ ПЛЕЕР (для ProRes это маленькая копия); ключи
        # saved_* — то, что действительно сохранено, и именно их берут кнопки
        # «Скачать» и «Копировать путь».
        payload = {
            "filename": filename,
            "subfolder": subfolder,
            "type": "output",
            "format": f"video/{fmt.extension}",
            "width": result["width"],
            "height": result["height"],
            "frames": result["frames"],
            "fps": result["fps"],
            "size_bytes": result["size_bytes"],
            "format_key": fmt.key,
            "codec": result["codec"],
            "has_audio": result["has_audio"],
            "is_proxy": False,
            "saved_filename": filename,
            "saved_subfolder": subfolder,
            "saved_type": "output",
            "saved_path": str(target),
        }

        if wants_proxy:
            proxy = cls._write_proxy(target, float(fps), audio, result["frames"], progress)
            if proxy is not None:
                payload.update(proxy)
                payload["is_proxy"] = True

        logger.info("%s saved %s (%s, %d frames)",
                    LOG_PREFIX, safe_log_path(target), fmt.key, result["frames"])

        return IO.NodeOutput(
            cls._as_video(target),
            str(target),
            ui={PREVIEW_UI_KEY: [payload]},
        )

    # ──────────────────────────────────────────────────────────────────────
    @classmethod
    def _source_frames(cls, images, video, fps: float):
        """Откуда брать кадры: из тензора или потоком из файла.

        Возвращает ``(итератор, ожидаемое число кадров)``. Тензор имеет
        приоритет: если подключены оба входа, человек явно собрал кадры сам.
        """
        from ._decode import iter_frames
        from ._encode import _frames_from_tensor

        if images is not None and getattr(images, "shape", None) is not None \
                and int(images.shape[0]) > 0:
            return _frames_from_tensor(images), int(images.shape[0])

        if video is not None:
            source = cls._video_path(video)
            if source:
                start, duration = cls._video_window(video)
                end = (start + duration) if duration > 0 else -1.0
                count = 0
                try:
                    count = int(video.get_frame_count())
                except Exception:           # noqa: BLE001 - не все реализации умеют
                    count = 0
                # ⚠️ Кадры пересчитываются под ЗАПРОШЕННУЮ частоту, а не берутся
                # как есть. Иначе они писались в файл со скоростью `fps`, а
                # сняты были с другой: 300 кадров тридцатикадрового источника,
                # записанные по 24, превращали десять секунд в двенадцать с
                # половиной — и звук, обрезанный по числу кадров, разъезжался с
                # картинкой.
                if fps > 0:
                    if duration <= 0:
                        try:
                            duration = max(0.0, float(video.get_duration()) - start)
                        except Exception:   # noqa: BLE001 - длительность знают не все
                            duration = 0.0
                    if duration > 0:
                        count = int(round(duration * fps))
                    elif count > 0:
                        source_fps = 0.0
                        try:
                            source_fps = float(video.get_frame_rate())
                        except Exception:   # noqa: BLE001
                            source_fps = 0.0
                        if source_fps > 0:
                            count = int(round(count * fps / source_fps))
                return iter_frames(source, start=start, end=end, fps=fps), count

        return None, 0

    @staticmethod
    def _video_path(video) -> str:
        """Путь к файлу за объектом VIDEO, если он вообще файловый."""
        try:
            source = video.get_stream_source()
        except Exception:                   # noqa: BLE001
            return ""
        return source if isinstance(source, str) and os.path.isfile(source) else ""

    @staticmethod
    def _video_window(video) -> tuple[float, float]:
        try:
            start, duration = video.get_active_trim_window()
            return float(start or 0.0), float(duration or 0.0)
        except Exception:                   # noqa: BLE001 - старое ядро без подрезки
            return 0.0, 0.0

    @classmethod
    def _metadata(cls) -> dict | None:
        """Prompt и workflow для контейнера.

        Молчит, когда ComfyUI запущен с ``--disable-metadata``: это явная просьба
        человека не вшивать ничего в файлы.
        """
        try:
            from comfy.cli_args import args

            if getattr(args, "disable_metadata", False):
                return None
        except Exception:                   # noqa: BLE001 - вне ComfyUI
            pass

        payload: dict = {}
        prompt = getattr(cls.hidden, "prompt", None)
        if prompt is not None:
            payload["prompt"] = prompt
        extra = getattr(cls.hidden, "extra_pnginfo", None) or {}
        for key, value in extra.items():
            payload[key] = value
        return payload or None

    @classmethod
    def _progress(cls, frame_count: int, *, with_proxy: bool):
        """Полоса выполнения ComfyUI на время записи.

        Кодирование длинного ролика идёт минутами, и без неё нода выглядит
        зависшей. Когда следом пишется копия для просмотра, она считается второй
        половиной работы — иначе полоса дошла бы до конца и замерла.

        Returns:
            Функция ``(записано) -> None`` или ``None``, если считать нечего.
        """
        total = int(frame_count) * (2 if with_proxy else 1)
        if total <= 0:
            return None
        try:
            from comfy.utils import ProgressBar

            bar = ProgressBar(total)
        except Exception as error:          # noqa: BLE001 - вне ComfyUI прогресса нет
            logger.debug("%s progress unavailable: %s", LOG_PREFIX, error)
            return None

        state = {"done": 0}

        def report(written: int) -> None:
            # Второй проход (прокси) начинает счёт заново — сдвигаем его.
            value = written if written > state["done"] else state["done"] + 1
            state["done"] = min(total, value)
            bar.update_absolute(state["done"], total)

        return report

    @classmethod
    def _write_proxy(cls, target: Path, fps: float, audio, frame_count: int,
                     progress=None) -> dict | None:
        """Маленький H.264 рядом, чтобы плеер в ноде мог что-то показать.

        Нужен для ProRes: браузеры его не проигрывают. Пишется в temp, поэтому
        библиотеку результатов не засоряет.
        """
        from ._decode import iter_frames
        from ._encode import downscale_frames, write_proxy

        try:
            import folder_paths

            temp = Path(folder_paths.get_temp_directory())
            temp.mkdir(parents=True, exist_ok=True)
            name = f"ts_video_preview_{os.getpid()}_{target.stem}.mp4"
            proxy_path = temp / name
            frames = downscale_frames(iter_frames(str(target)), _PROXY_MAX_WIDTH)
            info = write_proxy(frames, path=proxy_path, fps=fps, audio=audio,
                               frame_count=frame_count, on_frame=progress)
            return {
                "filename": name,
                "subfolder": "",
                "type": "temp",
                "format": "video/mp4",
                "width": info["width"],
                "height": info["height"],
            }
        except Exception as error:          # noqa: BLE001 - превью не стоит прогона
            logger.warning("%s could not build a preview: %s", LOG_PREFIX, error)
            return None

    @staticmethod
    def _as_video(path: Path):
        try:
            from comfy_api.input_impl import VideoFromFile

            return VideoFromFile(str(path))
        except Exception as error:          # noqa: BLE001 - ядро без VIDEO
            logger.debug("%s VIDEO output unavailable: %s", LOG_PREFIX, error)
            return None


NODE_CLASS_MAPPINGS = {"TS_VideoSaver": TS_VideoSaver}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_VideoSaver": "TS Video Saver"}
