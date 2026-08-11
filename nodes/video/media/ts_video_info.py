"""TS Video Info — разложить video_info на обычные числа.

Зачем отдельная нода: если вывесить все эти величины на самом загрузчике, у него
станет пятнадцать выходов, из которых в конкретном графе нужны один-два. Один
провод «video_info» и маленький посредник рядом — то же самое, но читается.

Нода принимает и чужой ``VHS_VIDEOINFO`` от VideoHelperSuite: провод от его
загрузчика сюда подключается, словарь переводится в нашу форму. Обратного моста
нет намеренно — назвать свой выход чужим типом значит захватить чужой контракт.
"""

from __future__ import annotations

from comfy_api.v0_0_2 import IO

from ._common import (
    VHS_VIDEO_INFO_TYPE,
    VIDEO_INFO_TYPE,
    coerce_video_info,
    format_summary,
    info_get,
)


class TS_VideoInfo(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_VideoInfo",
            display_name="TS Video Info",
            category="TS/Video",
            description=(
                "Unpack a video_info bundle into plain numbers: what was actually "
                "loaded, and what the source file was. Accepts VHS_VIDEOINFO too."
            ),
            search_aliases=["video info", "fps", "frame count", "duration", "resolution"],
            inputs=[
                IO.MultiType.Input(
                    "video_info",
                    types=[VIDEO_INFO_TYPE, VHS_VIDEO_INFO_TYPE],
                    tooltip=(
                        "Bundle from TS Video Loader. A VHS_VIDEOINFO bundle from Video "
                        "Helper Suite works as well."
                    ),
                ),
            ],
            # ⚠️ ПОРЯДОК ВЫХОДОВ ЗАМОРОЖЕН. Загруженные значения идут первыми —
            # их берут в девяти случаях из десяти. Новое дописывается только в
            # конец, иначе разъедутся связи в сохранённых workflow.
            outputs=[
                IO.Float.Output(display_name="fps",
                                tooltip="Frame rate of the loaded clip."),
                IO.Int.Output(display_name="frame_count",
                              tooltip="How many frames were actually loaded."),
                IO.Float.Output(display_name="duration",
                                tooltip="Length of the loaded clip, in seconds."),
                IO.Int.Output(display_name="width", tooltip="Width after resizing."),
                IO.Int.Output(display_name="height", tooltip="Height after resizing."),
                IO.Float.Output(display_name="source_fps",
                                tooltip="Frame rate as stored in the file."),
                IO.Int.Output(display_name="source_frame_count",
                              tooltip="Frames in the whole file."),
                IO.Float.Output(display_name="source_duration",
                                tooltip="Length of the whole file, in seconds."),
                IO.Int.Output(display_name="source_width", tooltip="Width in the file."),
                IO.Int.Output(display_name="source_height", tooltip="Height in the file."),
                IO.Boolean.Output(display_name="has_audio",
                                  tooltip="Whether the file carries an audio track."),
                IO.Boolean.Output(display_name="has_alpha",
                                  tooltip="Whether the file carries transparency."),
                IO.String.Output(display_name="summary",
                                 tooltip="One line describing the clip, for notes and debugging."),
            ],
        )

    @classmethod
    def execute(cls, video_info=None) -> IO.NodeOutput:
        # Ничего не вычисляем: только достаём с дефолтами. Поэтому нода не падает
        # ни на словаре будущей версии, ни на обрывке, ни на чужом формате.
        info = coerce_video_info(video_info)
        return IO.NodeOutput(
            float(info_get(info, "loaded.fps", 0.0) or 0.0),
            int(info_get(info, "loaded.frame_count", 0) or 0),
            float(info_get(info, "loaded.duration", 0.0) or 0.0),
            int(info_get(info, "loaded.width", 0) or 0),
            int(info_get(info, "loaded.height", 0) or 0),
            float(info_get(info, "source.fps", 0.0) or 0.0),
            int(info_get(info, "source.frame_count", 0) or 0),
            float(info_get(info, "source.duration", 0.0) or 0.0),
            int(info_get(info, "source.width", 0) or 0),
            int(info_get(info, "source.height", 0) or 0),
            bool(info_get(info, "source.has_audio", False)),
            bool(info_get(info, "source.has_alpha", False)),
            format_summary(info),
        )


NODE_CLASS_MAPPINGS = {"TS_VideoInfo": TS_VideoInfo}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_VideoInfo": "TS Video Info"}
