"""TS LTX HDR Guide — развилка SDR/HDR и подготовка опорных кадров разом.

В плане это были две ноды: «пирамида» и «роутер». Здесь они слиты в одну, и это
осознанное отступление — роутер требовал восьми проводов и отдельного типа
ноды, притом что развилка и подготовка кадра происходят в одной точке графа.
Одна нода на опорный кадр: одна для первого, одна для последнего.

⚠️ **Половинный кадр готовится из оригинала, а не уменьшением полного.**
Официальный двухстадийный pipeline собирает image conditioning заново для
каждого разрешения. Уменьшить готовый рабочий сигнал — не то же самое:
усреднять надо линейный свет, а не логарифмические коды.

⚠️ **В HDR-режиме ``LTXVPreprocess`` обходится полностью.** Он гоняет кадр
через H.264 и восемь бит (``(image * 255.0).byte()`` в исходнике ядра) — для
имитации сжатия это нормально, для HDR это уничтожение диапазона. Ленивые входы
устроены так, что в HDR-режиме ветка с ``LTXVPreprocess`` **не вычисляется**, а
в SDR-режиме не читается ни один EXR.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.v0_0_2 import IO

from ._acescct import linear_ap1_to_acescct, srgb_to_linear
from ._hdr_types import LOG_PREFIX, HdrImage, as_config, to_working_image
from ._resize import FIT_MODES, fit_linear
from ._schema import CATEGORY, MISSING, HdrConfigIO, HdrImageIO

logger = logging.getLogger("comfyui_timesaver.ts_ltx_hdr.guide")

# LTX 2.5 сжимает кадр в 32 раза по каждой стороне — размер обязан делиться.
_SIZE_STEP = 32


def _check_sizes(width1: int, height1: int, width2: int, height2: int, *, strict: bool) -> None:
    for name, value in (("stage1_width", width1), ("stage1_height", height1),
                        ("stage2_width", width2), ("stage2_height", height2)):
        if value <= 0:
            raise RuntimeError(f"{LOG_PREFIX} {name} must be positive, got {value}.")
        if value % _SIZE_STEP:
            message = (f"{LOG_PREFIX} {name}={value} is not a multiple of {_SIZE_STEP}; "
                       "LTX compresses by 32 in each direction.")
            if strict:
                raise RuntimeError(message + " Turn strict_validation off to continue anyway.")
            logger.warning("%s", message)

    # Две законные схемы: со spatial-апскейлером (полный вдвое больше) и без
    # него (размеры совпадают). Всё прочее — почти наверняка ошибка разводки.
    doubled = (width2 == width1 * 2 and height2 == height1 * 2)
    same = (width2 == width1 and height2 == height1)
    if not (doubled or same):
        message = (
            f"{LOG_PREFIX} Stage sizes look mis-wired: stage 1 is {width1}x{height1} and "
            f"stage 2 is {width2}x{height2}. Expected either the same size (no latent "
            "upscaler) or exactly double (with the x2 upscaler)."
        )
        if strict:
            raise RuntimeError(message + " Turn strict_validation off to continue anyway.")
        logger.warning("%s", message)


class TS_LTXHDRGuide(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LTXHDRGuide",
            display_name="TS LTX HDR Guide",
            category=CATEGORY,
            description=(
                "Prepare one guide frame for both LTX stages, from an EXR when HDR is "
                "on and from the ordinary SDR chain when it is off. The unused branch "
                "is never computed."
            ),
            search_aliases=["hdr guide", "exr guide", "first frame", "last frame", "ltx guide"],
            inputs=[
                HdrConfigIO.Input(
                    "config",
                    tooltip="Settings node. Its enabled switch picks the branch.",
                ),
                IO.Int.Input(
                    "stage1_width", default=960, min=32, max=16384, step=32,
                    tooltip="Width of the half-resolution first stage.",
                ),
                IO.Int.Input(
                    "stage1_height", default=544, min=32, max=16384, step=32,
                    tooltip="Height of the half-resolution first stage.",
                ),
                IO.Int.Input(
                    "stage2_width", default=1920, min=32, max=16384, step=32,
                    tooltip="Width of the full-resolution second stage.",
                ),
                IO.Int.Input(
                    "stage2_height", default=1088, min=32, max=16384, step=32,
                    tooltip="Height of the full-resolution second stage.",
                ),
                IO.Image.Input(
                    "sdr_image",
                    optional=True,
                    lazy=True,
                    tooltip=(
                        "The existing SDR chain, LTXVPreprocess included. Used only "
                        "when HDR is off, and not computed at all when it is on."
                    ),
                ),
                HdrImageIO.Input(
                    "hdr_linear",
                    optional=True,
                    lazy=True,
                    tooltip=(
                        "The EXR loader. Used only when HDR is on, and not read at all "
                        "when it is off — a broken EXR path cannot break an SDR run."
                    ),
                ),
                IO.Combo.Input(
                    "fit_mode",
                    options=list(FIT_MODES),
                    default="center_crop",
                    advanced=True,
                    tooltip=(
                        "center_crop fills the frame and trims the edges; reflect_pad "
                        "fits the whole picture and invents the margins."
                    ),
                ),
                IO.Combo.Input(
                    "interpolation",
                    options=["area", "bicubic", "bilinear", "nearest-exact"],
                    default="area",
                    advanced=True,
                    tooltip="area averages by coverage when shrinking — the safe default.",
                ),
                # ⚠️ ПОСЛЕДНИМ В СХЕМЕ: порядок входов — часть контракта
                # сохранённого workflow (§4 CLAUDE.md).
                IO.Image.Input(
                    "image_guide",
                    optional=True,
                    lazy=True,
                    tooltip=(
                        "An ordinary picture (JPG/PNG) to guide an HDR run when you have "
                        "no EXR. Used only when HDR is on and hdr_linear is not "
                        "connected. The gamma is removed properly — treating the codes "
                        "as linear light is off by up to 2.3 stops in the mid-tones. It "
                        "adds no headroom: white stays at 1.0, and whether the model "
                        "generates anything brighter is for TS LTX HDR Stats to say."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="stage1",
                    tooltip="Guide for the half-resolution stage.",
                ),
                IO.Image.Output(
                    display_name="stage2",
                    tooltip="Guide for the full-resolution stage, prepared independently.",
                ),
                IO.String.Output(display_name="info", tooltip="Which branch ran, and what it produced."),
            ],
        )

    @classmethod
    def check_lazy_status(
        cls,
        config=None,
        stage1_width=0, stage1_height=0, stage2_width=0, stage2_height=0,
        sdr_image=MISSING, hdr_linear=MISSING,
        fit_mode="center_crop", interpolation="area",
        image_guide=MISSING,
    ):
        """Попросить ровно ту ветку, которая понадобится.

        ⚠️ ``None`` означает «подключено, но ещё не вычислено», а :data:`MISSING`
        — «не подключено вовсе». Просить невычислимое нельзя: получится
        бесконечный круг вместо внятной ошибки из ``execute``.

        При включённом HDR источник выбирается по тому, что вообще подключено:
        EXR важнее обычной картинки. Отдельного переключателя нет намеренно —
        достаточно обвести загрузчик EXR в обход, и ветка сама переключится.
        """
        settings = as_config(config)
        # ⚠️ В режиме IC-LoRA опорный кадр — ОБЫЧНЫЙ SDR: диапазон выращивает
        # модель, а не мы. EXR здесь не нужен и не читается.
        if not settings.enabled or settings.expands_sdr:
            return ["sdr_image"] if sdr_image is None else []
        if hdr_linear is not MISSING:
            return ["hdr_linear"] if hdr_linear is None else []
        if image_guide is not MISSING:
            return ["image_guide"] if image_guide is None else []
        return []

    @classmethod
    def execute(
        cls,
        config=None,
        stage1_width: int = 960, stage1_height: int = 544,
        stage2_width: int = 1920, stage2_height: int = 1088,
        sdr_image=None, hdr_linear=None,
        fit_mode: str = "center_crop", interpolation: str = "area",
        image_guide=None,
    ) -> IO.NodeOutput:
        settings = as_config(config)

        if not settings.enabled or settings.expands_sdr:
            if sdr_image is None:
                raise RuntimeError(
                    f"{LOG_PREFIX} This node needs its sdr_image input connected — "
                    "that is the ordinary LoadImage → resize → LTXVPreprocess chain. "
                    + ("The IC-LoRA mode guides from ordinary SDR: the range is grown "
                       "by the model, not prepared by us."
                       if settings.expands_sdr else "")
                )
            info = ("HDR on, IC-LoRA mode — the SDR guide passed through untouched; "
                    "the model grows the range, and the decode undoes LogC3."
                    if settings.expands_sdr
                    else "HDR off — the SDR guide passed through untouched.")
            return IO.NodeOutput(sdr_image, sdr_image, info)

        headroom = ""
        if hdr_linear is None and image_guide is not None:
            # Обычная картинка вместо EXR. Гамму снимаем ЗДЕСЬ и до ресайза:
            # усреднять надо линейный свет, а коды с гаммой — нельзя.
            linear = srgb_to_linear(image_guide)
            hdr_linear = HdrImage(linear, "REC709_LINEAR", {"source": "ordinary image"})
            white = float(linear_ap1_to_acescct(torch.tensor([1.0])))
            headroom = (
                f"\n! Guided by an ordinary picture, not an EXR. Its white sits at "
                f"ACEScct {white:.3f}, so {100.0 * (1.0 - white):.0f}% of the working "
                "range above it is empty: the file simply has no highlights. The model "
                "may still generate into that headroom — check with TS LTX HDR Stats."
            )

        if hdr_linear is None:
            raise RuntimeError(
                f"{LOG_PREFIX} HDR is on, so this node needs a guide: either hdr_linear "
                "from a TS LTX Load HDR EXR node, or image_guide with an ordinary "
                "picture."
            )
        if not isinstance(hdr_linear, HdrImage):
            raise RuntimeError(
                f"{LOG_PREFIX} hdr_linear carries {type(hdr_linear).__name__}, not an "
                "HDR frame. Connect the EXR loader, not an ordinary image — an ordinary "
                "picture goes into image_guide."
            )

        _check_sizes(int(stage1_width), int(stage1_height),
                     int(stage2_width), int(stage2_height),
                     strict=settings.strict_validation)

        # Каждая стадия готовится ОТДЕЛЬНО из одного и того же оригинала.
        stages = []
        for width, height in ((int(stage1_width), int(stage1_height)),
                              (int(stage2_width), int(stage2_height))):
            if hdr_linear.color_space == "ACESCCT":
                # Уже закодированный вход ресайзим как есть: раскодировать и
                # закодировать обратно значило бы дважды пройти кривую.
                fitted = fit_linear(hdr_linear.tensor, width, height,
                                    fit_mode=fit_mode, interpolation=interpolation)
                working = HdrImage(fitted, "ACESCCT", hdr_linear.meta)
            else:
                linear = fit_linear(hdr_linear.to_linear_ap1(), width, height,
                                    fit_mode=fit_mode, interpolation=interpolation)
                working = HdrImage(linear, "ACESCG", hdr_linear.meta)
            stages.append(to_working_image(working))

        source_w, source_h = hdr_linear.size
        info = (
            f"HDR on — guides built from {source_w}x{source_h} {hdr_linear.color_space}\n"
            f"stage 1: {stage1_width}x{stage1_height}   "
            f"stage 2: {stage2_width}x{stage2_height}\n"
            f"working signal ACEScct, {fit_mode} / {interpolation}" + headroom
        )
        logger.info("%s guides %dx%d and %dx%d from %s", LOG_PREFIX,
                    stage1_width, stage1_height, stage2_width, stage2_height,
                    hdr_linear.color_space)
        return IO.NodeOutput(stages[0], stages[1], info)


NODE_CLASS_MAPPINGS = {"TS_LTXHDRGuide": TS_LTXHDRGuide}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LTXHDRGuide": "TS LTX HDR Guide"}
