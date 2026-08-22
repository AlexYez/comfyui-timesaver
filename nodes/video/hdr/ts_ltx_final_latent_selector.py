"""TS LTX Final Latent Selector — выбрать результат ДО декодирования.

Обычная разводка двухстадийного графа выбирает картинку **после** двух
отдельных VAE-decode: декодируется и первая стадия, и вторая, а потом одна из
них выбрасывается. Это лишний проход тяжёлого декодера и лишние гигабайты.

Здесь выбор перенесён на уровень латента, а входы сделаны ленивыми: невыбранная
ветка не декодируется — и, что важнее, **не считается вовсе**. Выключенный
второй проход перестаёт стоить времени сэмплера, а не только декодера.

Дальше по графу остаётся ровно один финальный decode — и, значит, ровно одно
место, где применяется HDR-преобразование.
"""

from __future__ import annotations

import logging

from comfy_api.v0_0_2 import IO

from ._hdr_types import LOG_PREFIX
from ._schema import CATEGORY, MISSING

logger = logging.getLogger("comfyui_timesaver.ts_ltx_hdr.selector")


class TS_LTXFinalLatentSelector(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LTXFinalLatentSelector",
            display_name="TS LTX Final Latent Selector",
            category=CATEGORY,
            description=(
                "Pick the first- or second-stage latent before decoding, so only one "
                "VAE decode happens — and the branch you did not pick is never computed."
            ),
            search_aliases=["latent switch", "stage select", "two stage", "ltx upscale switch"],
            inputs=[
                IO.Boolean.Input(
                    "use_stage2",
                    default=True,
                    tooltip=(
                        "On: the full-resolution second stage. Off: stop after the "
                        "first stage — the upscaler and second sampler do not run."
                    ),
                ),
                IO.Latent.Input(
                    "stage1_latent",
                    optional=True,
                    lazy=True,
                    tooltip="Video latent after the first stage, guide frames already cropped off.",
                ),
                IO.Latent.Input(
                    "stage2_latent",
                    optional=True,
                    lazy=True,
                    tooltip="Video latent after the second stage.",
                ),
            ],
            outputs=[
                IO.Latent.Output(
                    display_name="latent",
                    tooltip="The one latent that goes on to the single final decode.",
                ),
            ],
        )

    @classmethod
    def check_lazy_status(cls, use_stage2=True, stage1_latent=MISSING, stage2_latent=MISSING):
        wanted = stage2_latent if use_stage2 else stage1_latent
        if wanted is None:
            return ["stage2_latent" if use_stage2 else "stage1_latent"]
        return []

    @classmethod
    def execute(cls, use_stage2: bool = True, stage1_latent=None, stage2_latent=None) -> IO.NodeOutput:
        chosen = stage2_latent if use_stage2 else stage1_latent
        if chosen is None:
            which = "stage2_latent" if use_stage2 else "stage1_latent"
            raise RuntimeError(
                f"{LOG_PREFIX} {which} is not connected, but use_stage2="
                f"{bool(use_stage2)} asks for it.")
        logger.debug("%s selected %s", LOG_PREFIX, "stage 2" if use_stage2 else "stage 1")
        return IO.NodeOutput(chosen)


NODE_CLASS_MAPPINGS = {"TS_LTXFinalLatentSelector": TS_LTXFinalLatentSelector}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LTXFinalLatentSelector": "TS LTX Final Latent Selector"}
