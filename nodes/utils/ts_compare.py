"""TS Compare — шторка «до и после» для картинок и для видео.

Принимает две стороны, `IMAGE` каждая: одиночный кадр или пачку. Что делать
дальше, решает содержимое, и решения тут два разных, потому что материал разный.

**Пара одиночных кадров — две PNG.** Сравнивают обычно детали: резкость,
артефакты, кожу после ретуши. Прогнать такое через H.264 значило бы уничтожить
ровно то, на что человек смотрит.

**Хотя бы одна пачка — ОДИН файл, где A лежит над B.** Не два файла и не два
плеера, и это не экономия ради экономии:

* два `<video>` расходятся на кадр-два на быстром движении, и сравнение начинает
  тихо врать — отказ, которого не видно;
* браузер декодирует видео ТОЙ ЖЕ картой, на которой считает ComfyUI, и второй
  декодер бьёт туда же, куда била жалоба, ради которой в 12.5.0 появился сторож
  воспроизведения.

Сложенные в один кадр стороны рассинхронизироваться не могут физически, и
декодер работает один. Цена — обе стороны приводятся к одному размеру, но
сравнению это и так необходимо: иначе шторка не совпадает сама с собой.

⚠️ Превью видео СЖАТОЕ (H.264, черновое качество, ширина до 1280). Это плеер в
ноде, а не мастер: судить по нему о шуме и градиентах нельзя. Для одиночных
кадров такого ограничения нет — там PNG.
"""

from __future__ import annotations

import logging
import time

from comfy_api.v0_0_2 import IO

# ⚠️ Зависимость от media НАМЕРЕННАЯ и односторонняя: сборка кадров в файл живёт
# там, второй копии кодировщика в паке быть не должно. Обратной зависимости
# (media -> utils) нет и заводить её нельзя.
from ..video.media._encode import downscale_frames, write_proxy

logger = logging.getLogger("comfyui_timesaver.ts_compare")
LOG_PREFIX = "[TS Compare]"

PREVIEW_UI_KEY = "ts_compare"
#: Ширина превью. Столько же берёт прокси сейвера — плееру в ноде больше незачем.
MAX_WIDTH = 1280


class TS_Compare(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Compare",
            display_name="TS Compare",
            category="TS/Utils",
            description=(
                "Compare two images or two clips behind a wipe. Stills are kept as PNG; "
                "batches are assembled into a single file with one side above the other, "
                "so the two halves cannot drift apart and only one decoder runs."
            ),
            inputs=[
                IO.Image.Input(
                    "image_a",
                    tooltip="Left of the wipe — usually the original.",
                ),
                IO.Image.Input(
                    "image_b",
                    tooltip=(
                        "Right of the wipe — usually the result. Resized to A's frame, "
                        "because a wipe over two different sizes compares nothing."
                    ),
                ),
                IO.Float.Input(
                    "fps",
                    default=24.0, min=1.0, max=240.0, step=0.01,
                    tooltip="Playback rate for the assembled clip. Ignored for a pair of stills.",
                ),
                IO.String.Input(
                    "label_a",
                    default="before",
                    tooltip="Caption on the left half.",
                ),
                IO.String.Input(
                    "label_b",
                    default="after",
                    tooltip="Caption on the right half.",
                ),
            ],
            outputs=[],
            # Выходов нет, но выполниться нода обязана — иначе сравнивать нечего.
            is_output_node=True,
            search_aliases=["compare", "before after", "wipe", "сравнение", "шторка"],
        )

    # ------------------------------------------------------------------ кадры
    @classmethod
    def _to_uint8(cls, images):
        """IMAGE -> список кадров ``[H,W,3]`` uint8, не трогая вход."""
        import numpy as np

        array = images.detach().to("cpu").numpy()
        array = array[..., :3]
        array = np.clip(array, 0.0, 1.0) * 255.0
        return [np.ascontiguousarray(frame.astype(np.uint8)) for frame in np.rint(array)]

    @classmethod
    def _resize(cls, frame, width: int, height: int):
        """Привести кадр к чужому размеру. Растяжением, не вписыванием.

        ⚠️ Именно растяжением: стороны сравнения — это один и тот же кадр в двух
        обработках, а не два разных снимка. Поля по краям сдвинули бы картинку
        относительно второй половины, и шторка перестала бы совпадать.
        """
        import numpy as np
        from PIL import Image

        if frame.shape[1] == width and frame.shape[0] == height:
            return frame
        resized = Image.fromarray(frame).resize((width, height), Image.LANCZOS)
        return np.ascontiguousarray(np.asarray(resized))

    @classmethod
    def _fit_length(cls, frames, count: int):
        """Дотянуть сторону до нужной длины, придержав последний кадр."""
        if not frames:
            return frames
        if len(frames) >= count:
            return frames[:count]
        return frames + [frames[-1]] * (count - len(frames))

    # ---------------------------------------------------------------- выдача
    @classmethod
    def _temp_target(cls, suffix: str):
        import folder_paths

        from pathlib import Path

        base = Path(folder_paths.get_temp_directory())
        base.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.time_ns():020d}"
        return base / f"ts_compare_{stamp}{suffix}"

    @classmethod
    def _progress(cls, total: int):
        """Полоса ComfyUI на время сборки.

        ⚠️ Нода выходная и считает внутри прогона, поэтому штатная полоса здесь
        работает — в отличие от кнопки загрузчика, где её вообще нет к чему
        прицепить. Сборка сотни кадров занимает секунды, и без полосы нода
        выглядит зависшей.
        """
        try:
            import comfy.utils  # noqa: PLC0415

            bar = comfy.utils.ProgressBar(max(1, int(total)))
        except Exception:                   # noqa: BLE001 - нет сервера, нет полосы
            return None

        def report(written: int) -> None:
            try:
                bar.update_absolute(int(written), max(1, int(total)))
            except Exception:               # noqa: BLE001 - интерфейс не роняет сборку
                pass

        return report

    @classmethod
    def _save_png(cls, frame, suffix: str) -> str:
        from PIL import Image

        target = cls._temp_target(f"_{suffix}.png")
        Image.fromarray(frame).save(target, format="PNG", compress_level=4)
        return target.name

    @classmethod
    def execute(
        cls,
        image_a,
        image_b,
        fps: float = 24.0,
        label_a: str = "before",
        label_b: str = "after",
    ) -> IO.NodeOutput:
        import numpy as np

        for name, value in (("image_a", image_a), ("image_b", image_b)):
            if value is None or value.ndim != 4:
                raise ValueError(f"{LOG_PREFIX} '{name}' must be a batch shaped [B,H,W,C].")

        left = cls._to_uint8(image_a)
        right = cls._to_uint8(image_b)
        height, width = left[0].shape[0], left[0].shape[1]

        # Ширина превью: уменьшаем ДО складывания, иначе в файл уедет двойная
        # высота полного разрешения.
        scale = min(1.0, MAX_WIDTH / float(width))
        view_w = max(2, int(round(width * scale)) - (int(round(width * scale)) % 2))
        view_h = max(2, int(round(height * scale)) - (int(round(height * scale)) % 2))

        still = len(left) == 1 and len(right) == 1
        payload = {
            "mode": "image" if still else "video",
            "label_a": str(label_a or "before"),
            "label_b": str(label_b or "after"),
            "width": view_w,
            "height": view_h,
            "type": "temp",
            "subfolder": "",
        }

        if still:
            # ⚠️ PNG, а не кадр видео: сравнивают детали, а H.264 уничтожил бы
            # ровно то, на что человек смотрит.
            payload["filename_a"] = cls._save_png(cls._resize(left[0], view_w, view_h), "a")
            payload["filename_b"] = cls._save_png(cls._resize(right[0], view_w, view_h), "b")
            logger.info("%s two stills %d×%d.", LOG_PREFIX, view_w, view_h)
        else:
            count = max(len(left), len(right))
            if len(left) != len(right):
                logger.info(
                    "%s sides are %d and %d frames; the shorter holds its last frame.",
                    LOG_PREFIX, len(left), len(right),
                )
            left = cls._fit_length(left, count)
            right = cls._fit_length(right, count)

            def stacked():
                for a_frame, b_frame in zip(left, right):
                    yield np.concatenate(
                        (
                            cls._resize(a_frame, view_w, view_h),
                            cls._resize(b_frame, view_w, view_h),
                        ),
                        axis=0,
                    )

            target = cls._temp_target(".mp4")
            write_proxy(
                downscale_frames(stacked(), max_width=view_w),
                path=target,
                fps=float(fps),
                frame_count=count,
                on_frame=cls._progress(count),
            )
            payload["filename"] = target.name
            payload["frames"] = count
            payload["fps"] = float(fps)
            logger.info("%s %d frame(s), %d×%d per side.",
                        LOG_PREFIX, count, view_w, view_h)

        return IO.NodeOutput(ui={PREVIEW_UI_KEY: [payload]})


NODE_CLASS_MAPPINGS = {"TS_Compare": TS_Compare}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Compare": "TS Compare"}
