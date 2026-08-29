"""TS Video Cut — trim frames off both ends of a clip, taking the audio with them.

Trimming a clip in ComfyUI usually means two nodes that know nothing about each
other: one slices the IMAGE batch, another slices the AUDIO. They then have to
be told the same numbers in two different units — frames here, seconds or
samples there — and the moment the two disagree the sound drifts away from the
picture. This node does both cuts from ONE pair of numbers, in frames.

⚠️ THE PICTURE IS THE MASTER, and that is the whole point of the design. The cut
is expressed in frames, and the audio boundary is derived from the frame
boundary through ``fps`` — never the other way round. Audio arriving slightly
longer or shorter than the video (a very common thing, since encoders round
differently) therefore cannot shift the cut: it is clamped to what exists.

node_id: TS_VideoCut
"""

from __future__ import annotations

import logging

import torch
from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_video_cut")
LOG_PREFIX = "[TS Video Cut]"

#: Sample rate for the silence produced when no audio is connected. 44.1 kHz is
#: what ComfyUI's own audio nodes default to, so the silence mixes with anything.
_SILENT_RATE = 44100


class TS_VideoCut(IO.ComfyNode):
    """Cut N frames off the start and M off the end, audio included."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_VideoCut",
            display_name="TS Video Cut",
            category="TS/Video",
            description=(
                "Trim frames off the start and the end of a clip and cut the audio to "
                "match, from one pair of numbers given in frames. The frame boundary is "
                "the master and the audio boundary is derived from it through fps, so "
                "the sound cannot drift away from the picture."
            ),
            search_aliases=[
                "video cut", "trim video", "cut frames", "trim clip",
                "cut audio and video", "обрезка видео",
            ],
            inputs=[
                IO.Image.Input(
                    "images",
                    tooltip="Frames to trim. The batch is treated as a clip in playback order.",
                ),
                IO.Float.Input(
                    "fps",
                    default=24.0, min=0.01, max=1000.0, step=0.001,
                    tooltip=(
                        "Frame rate the clip plays at. This is what turns a cut in frames "
                        "into a cut in samples, so it must match the real rate — 23.976 or "
                        "29.97 are written out in full, not rounded to 24 or 30."
                    ),
                ),
                IO.Int.Input(
                    "start_cut",
                    default=0, min=0, max=1000000,
                    tooltip="How many frames to remove from the START of the clip.",
                ),
                IO.Int.Input(
                    "end_cut",
                    default=0, min=0, max=1000000,
                    tooltip="How many frames to remove from the END of the clip.",
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip=(
                        "Soundtrack to cut alongside the frames. Optional: with nothing "
                        "connected the node outputs silence of exactly the trimmed length, "
                        "so a downstream saver still gets a valid audio track."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="images", tooltip="The trimmed frames."),
                IO.Audio.Output(display_name="audio", tooltip="Audio cut to the same span."),
            ],
        )

    @classmethod
    def validate_inputs(cls, fps=24.0, start_cut=0, end_cut=0, **_kwargs) -> bool | str:
        # ⚠️ Число кадров здесь неизвестно (картинки приходят только в execute),
        # поэтому проверяем то, что можно: сами по себе бессмысленные значения.
        if float(fps) <= 0:
            return "fps must be greater than zero."
        if int(start_cut) < 0 or int(end_cut) < 0:
            return "start_cut and end_cut cannot be negative."
        return True

    @classmethod
    def execute(cls, images, fps=24.0, start_cut=0, end_cut=0, audio=None) -> IO.NodeOutput:
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(
                f"{LOG_PREFIX} images must be an IMAGE batch [B, H, W, C]; "
                f"got {type(images).__name__} {tuple(getattr(images, 'shape', ()))}."
            )

        total = int(images.shape[0])
        start = int(start_cut)
        end = int(end_cut)
        fps_value = float(fps)

        if start + end >= total:
            raise ValueError(
                f"{LOG_PREFIX} Nothing would be left: the clip has {total} frame(s) and "
                f"the cuts remove {start} + {end}. Reduce start_cut or end_cut."
            )

        first = start
        last = total - end                      # верхняя граница, не включая
        trimmed = images[first:last]

        cut_audio = cls._cut_audio(audio, first, last, total, fps_value)

        logger.info(
            "%s %d frame(s) -> %d (cut %d from the start, %d from the end) at %.3f fps",
            LOG_PREFIX, total, int(trimmed.shape[0]), start, end, fps_value,
        )
        return IO.NodeOutput(trimmed, cut_audio)

    # ------------------------------------------------------------------
    @classmethod
    def _cut_audio(cls, audio, first_frame, last_frame, total_frames, fps):
        """Звук, обрезанный по границам КАДРОВ.

        Границы считаются из номеров кадров через fps, а не из долей длины: доля
        соврала бы ровно тогда, когда звук и видео разной длительности — а это
        обычное дело, кодировщики округляют по-разному.
        """
        seconds_from = first_frame / fps
        seconds_to = last_frame / fps
        kept_seconds = seconds_to - seconds_from

        waveform, rate = cls._unpack(audio)
        if waveform is None:
            return cls._silence(kept_seconds)

        samples = int(waveform.shape[-1])
        begin = int(round(seconds_from * rate))
        finish = int(round(seconds_to * rate))

        # ⚠️ Зажимаем в то, что реально есть. Аудио, пришедшее короче видео,
        # иначе дало бы пустой срез, а пришедшее длиннее — хвост от соседнего
        # кадра; и то и другое выглядит как рассинхрон, хотя причина в исходнике.
        begin = max(0, min(begin, samples))
        finish = max(begin, min(finish, samples))

        expected = int(round(kept_seconds * rate))
        got = finish - begin
        if abs(got - expected) > rate * 0.05:      # расхождение больше 50 мс
            logger.warning(
                "%s Audio is %.2f s where the frames span %.2f s; the cut was clamped to "
                "the audio that exists. Check that fps (%.3f) matches the clip.",
                LOG_PREFIX, samples / rate, total_frames / fps, fps,
            )

        cut = waveform[..., begin:finish]
        if cut.ndim == 2:                          # [C, T] -> [B, C, T]
            cut = cut.unsqueeze(0)
        return {"waveform": cut.contiguous(), "sample_rate": int(rate)}

    @staticmethod
    def _unpack(audio):
        """``(waveform, sample_rate)`` или ``(None, None)``, если звука нет."""
        if not isinstance(audio, dict):
            return None, None
        waveform = audio.get("waveform")
        rate = audio.get("sample_rate")
        if not isinstance(waveform, torch.Tensor) or not rate:
            return None, None
        if waveform.ndim == 1:                     # [T] -> [B, C, T]
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:                   # [C, T] -> [B, C, T]
            waveform = waveform.unsqueeze(0)
        return waveform, int(rate)

    @staticmethod
    def _silence(seconds):
        """Тишина нужной длины — чтобы выход AUDIO оставался валидным.

        Отдавать `None` нельзя: сохранятель ниже по графу ждёт словарь и упал бы
        не там, где настоящая причина.
        """
        length = max(1, int(round(max(0.0, seconds) * _SILENT_RATE)))
        return {"waveform": torch.zeros(1, 1, length), "sample_rate": _SILENT_RATE}


NODE_CLASS_MAPPINGS = {"TS_VideoCut": TS_VideoCut}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_VideoCut": "TS Video Cut"}
