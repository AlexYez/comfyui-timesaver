"""TS LTX Load HDR EXR — прочитать кадр так, чтобы солнце осталось солнцем.

Штатный ``Load Image`` для EXR не годится: он приводит кадр к диапазону
``[0, 1]``, и всё, ради чего EXR существует, теряется молча. Здесь файл
читается как float32 без нормализации и без зажима сверху, а рядом выдаётся
отчёт — сколько сэмплов оказалось ярче единицы. Если там ноль, значит либо
сцена правда тусклая, либо HDR потеряли ещё до нас.

Файл берётся либо из папки ``input`` списком, либо по прямому пути: рендеры
обычно лежат не там, куда ComfyUI кладёт загруженное.
"""

from __future__ import annotations

import logging
from pathlib import Path

from comfy_api.v0_0_2 import IO

from ._exr_io import BACKENDS, DEFAULT_MAX_PIXELS, ExrError, load_exr
from ._hdr_types import LOG_PREFIX, HdrImage, as_config
from ._schema import CATEGORY, HdrConfigIO, HdrImageIO

logger = logging.getLogger("comfyui_timesaver.ts_ltx_hdr.load_exr")

# Показывается, когда в папке input нет ни одного EXR: пустой список ComfyUI
# рисовать не умеет, а внятная строка сразу объясняет, что делать.
NO_FILES = "(no .exr in the input folder)"

_EXR_SUFFIXES = (".exr", ".EXR")


def _input_files() -> list[str]:
    """Все EXR в папке ``input``, включая подпапки."""
    try:
        import folder_paths

        root = Path(folder_paths.get_input_directory())
    except Exception:                                    # noqa: BLE001 - вне ComfyUI
        return []
    if not root.is_dir():
        return []
    found = []
    for suffix in _EXR_SUFFIXES:
        for path in root.rglob(f"*{suffix}"):
            if path.is_file():
                found.append(path.relative_to(root).as_posix())
    return sorted(set(found))


def _resolve(exr_file: str, path_override: str) -> Path:
    """Что именно читать: прямой путь важнее выбора из списка."""
    override = str(path_override or "").strip().strip('"')
    if override:
        return Path(override).expanduser()

    name = str(exr_file or "").strip()
    if not name or name == NO_FILES:
        raise ExrError(
            f"{LOG_PREFIX} No EXR chosen. Either drop one into ComfyUI's input folder "
            "and pick it from the list, or type a full path into path_override."
        )
    try:
        import folder_paths

        return Path(folder_paths.get_input_directory()) / name
    except Exception as error:                           # noqa: BLE001
        raise ExrError(f"{LOG_PREFIX} Cannot resolve the input folder: {error}") from error


class TS_LTXHDRLoadEXR(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        files = _input_files() or [NO_FILES]
        return IO.Schema(
            node_id="TS_LTXHDRLoadEXR",
            display_name="TS LTX Load HDR EXR",
            category=CATEGORY,
            description=(
                "Read an EXR frame as linear float32, keeping values above 1.0. The "
                "ordinary image loader would flatten them to 1.0 without saying so."
            ),
            search_aliases=["exr", "load exr", "hdr image", "openexr"],
            inputs=[
                IO.Combo.Input(
                    "exr_file",
                    options=files,
                    tooltip="An .exr from ComfyUI's input folder, subfolders included.",
                ),
                IO.String.Input(
                    "path_override",
                    default="",
                    tooltip=(
                        "Full path to an .exr anywhere on disk. Wins over the list "
                        "above — renders usually do not live in ComfyUI's input folder."
                    ),
                ),
                IO.Combo.Input(
                    "backend",
                    options=list(BACKENDS),
                    default="auto",
                    advanced=True,
                    tooltip=(
                        "auto prefers OpenImageIO (what the official LTX pipeline "
                        "uses), then PyAV, which ships with ComfyUI and needs no "
                        "setup. OpenCV only reads EXR when OPENCV_IO_ENABLE_OPENEXR=1 "
                        "was set before ComfyUI started."
                    ),
                ),
                HdrConfigIO.Input(
                    "config",
                    optional=True,
                    tooltip=(
                        "Settings node. Its input_color_space becomes the label this "
                        "frame carries downstream."
                    ),
                ),
            ],
            outputs=[
                HdrImageIO.Output(
                    display_name="hdr_linear",
                    tooltip="The frame, tagged with the colour space it is in.",
                ),
                IO.String.Output(
                    display_name="info",
                    tooltip="Size, backend, range and how much of the frame is above 1.0.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, exr_file="", path_override="", backend="auto", config=None):
        """Пересчитать, если файл подменили на месте.

        Один путь и одно имя ничего не говорят о содержимом: художник
        перерендерил кадр под тем же именем, и без времени изменения нода
        отдала бы старую картинку.
        """
        try:
            path = _resolve(str(exr_file), str(path_override))
            stat = path.stat()
            stamp = f"{stat.st_mtime_ns}:{stat.st_size}"
        except Exception:                                # noqa: BLE001 - пусть решает execute
            stamp = "unresolved"
        return f"{exr_file}|{path_override}|{backend}|{stamp}"

    @classmethod
    def execute(cls, exr_file="", path_override="", backend="auto", config=None) -> IO.NodeOutput:
        settings = as_config(config)
        path = _resolve(str(exr_file), str(path_override))
        tensor, meta = load_exr(path, backend=str(backend), max_pixels=DEFAULT_MAX_PIXELS)

        if meta["non_finite"] and settings.strict_validation:
            raise ExrError(
                f"{LOG_PREFIX} {meta['name']} contains {meta['non_finite']} NaN/Inf "
                "samples. Fix the render, or turn strict_validation off to continue."
            )

        space = settings.input_color_space
        image = HdrImage(tensor=tensor, color_space=space, meta=meta)
        share = 100.0 * meta["above_one"] / max(1, tensor.numel())
        info = (
            f"{meta['name']}  {meta['width']}x{meta['height']}  via {meta['backend']}\n"
            f"declared as {space}\n"
            f"range: min {meta['min']:.4g}  max {meta['max']:.4g}\n"
            f"above 1.0: {share:.2f}% of samples"
        )
        if meta["dropped_channels"]:
            info += f"\ndropped channels: {', '.join(meta['dropped_channels'])}"
        if meta["max"] <= 1.0:
            info += ("\n! Nothing above 1.0 — this file carries no highlights, so the "
                     "HDR path has nothing extra to preserve.")

        logger.info("%s loaded %s (%dx%d, max %.4g) via %s", LOG_PREFIX, meta["name"],
                    meta["width"], meta["height"], meta["max"], meta["backend"])
        return IO.NodeOutput(image, info)


NODE_CLASS_MAPPINGS = {"TS_LTXHDRLoadEXR": TS_LTXHDRLoadEXR}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LTXHDRLoadEXR": "TS LTX Load HDR EXR"}
