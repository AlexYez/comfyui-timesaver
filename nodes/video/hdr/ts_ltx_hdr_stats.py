"""TS LTX HDR Stats — проверить, что HDR действительно доехал.

Потерянный HDR выглядит нормально. Картинка на экране та же, файл записался,
ошибок нет — просто в нём ничего нет ярче единицы, и узнают об этом в
монтажной, когда попробуют вытянуть небо. Эта нода отвечает на вопрос
«диапазон на месте?» числами, а не на глаз.

Считает кусками по кадрам: полный ролик 129×1920×1088 в float32 весит 3 ГиБ, и
копия ради сортировки перцентилей удвоила бы цифру. Перцентили берутся по
разрежённой выборке — до третьего знака здесь никому не нужно.
"""

from __future__ import annotations

import logging

from comfy_api.v0_0_2 import IO, UI

from ._hdr_types import LOG_PREFIX, HdrImage, as_config, describe, format_report, warnings_for
from ._schema import CATEGORY, HdrConfigIO, HdrImageIO

logger = logging.getLogger("comfyui_timesaver.ts_ltx_hdr.stats")


class TS_LTXHDRStats(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LTXHDRStats",
            display_name="TS LTX HDR Stats",
            category=CATEGORY,
            description=(
                "Report the actual range of an HDR master: how much is above 1.0, how "
                "many stops it spans, whether anything was clipped or turned into NaN."
            ),
            search_aliases=["hdr stats", "hdr check", "range check", "validate hdr"],
            inputs=[
                HdrImageIO.Input(
                    "hdr_linear",
                    tooltip="The master from TS LTX HDR Decode, or a frame from the EXR loader.",
                ),
                HdrConfigIO.Input(
                    "config",
                    optional=True,
                    tooltip="Settings node. strict_validation turns the complaints into errors.",
                ),
                IO.Boolean.Input(
                    "raise_on_problem",
                    default=False,
                    tooltip=(
                        "Stop the run when something looks wrong, instead of only "
                        "printing it. Useful when a batch runs unattended."
                    ),
                ),
            ],
            outputs=[
                IO.String.Output(display_name="report", tooltip="The whole thing, as text."),
                IO.Boolean.Output(
                    display_name="looks_ok",
                    tooltip="False when anything in the report deserved an exclamation mark.",
                ),
                IO.Float.Output(display_name="max", tooltip="Brightest sample in the master."),
            ],
        )

    @classmethod
    def execute(cls, hdr_linear=None, config=None, raise_on_problem: bool = False) -> IO.NodeOutput:
        if hdr_linear is None:
            raise RuntimeError(f"{LOG_PREFIX} Nothing connected to hdr_linear.")
        if not isinstance(hdr_linear, HdrImage):
            raise RuntimeError(
                f"{LOG_PREFIX} hdr_linear carries {type(hdr_linear).__name__}; connect "
                "the HDR decode or the EXR loader."
            )

        settings = as_config(config)
        stats = describe(hdr_linear.tensor, color_space=hdr_linear.color_space,
                         ceiling=settings.working_ceiling)
        notes = warnings_for(stats, strict=settings.strict_validation)
        report = format_report(stats, notes)

        for line in report.splitlines():
            logger.info("%s %s", LOG_PREFIX, line)
        if notes and bool(raise_on_problem):
            raise RuntimeError(f"{LOG_PREFIX} HDR check failed:\n{report}")

        return IO.NodeOutput(report, not notes, float(stats["max"]),
                             ui=UI.PreviewText(report))


NODE_CLASS_MAPPINGS = {"TS_LTXHDRStats": TS_LTXHDRStats}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LTXHDRStats": "TS LTX HDR Stats"}
