"""TS LTX HDR Settings — единственная галочка, которая включает весь HDR-путь.

Одна нода на весь граф. Её выход расходится по остальным HDR-нодам, и они сами
решают, что им делать: загружать EXR или не загружать, брать FP32-VAE или
обычный, отдавать мастер или заблокировать сохранятель. Человеку остаётся один
переключатель вместо пяти согласованных.

Настройки превью сюда попали намеренно: они видны рядом с главной галочкой, и
сразу понятно, что экспозиция и тон-маппинг относятся к просмотру, а не к тому,
что запишется в файл.
"""

from __future__ import annotations

import logging

from comfy_api.v0_0_2 import IO

from ._hdr_types import (HDR_MODES, INPUT_SPACES, LOG_PREFIX, MODE_PRESERVE,
                         OUTPUT_SPACES, HdrConfig)
from ._schema import CATEGORY, HdrConfigIO
from ._tonemap import OPERATORS

logger = logging.getLogger("comfyui_timesaver.ts_ltx_hdr.settings")


class TS_LTXHDRSettings(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LTXHDRSettings",
            display_name="TS LTX HDR Settings",
            category=CATEGORY,
            description=(
                "One switch for the whole native HDR path of LTX 2.5. Off, the graph "
                "behaves exactly as before and nothing HDR is loaded or computed."
            ),
            search_aliases=["hdr", "exr", "ltx hdr", "acescct", "native hdr"],
            inputs=[
                IO.Boolean.Input(
                    "enabled",
                    default=False,
                    tooltip=(
                        "Turn the native HDR path on. Off by default so that an old "
                        "workflow opens behaving exactly as it did."
                    ),
                ),
                IO.Combo.Input(
                    "input_color_space",
                    options=list(INPUT_SPACES),
                    default="ACESCG",
                    tooltip=(
                        "What the EXR files already are. ACESCG and SRGB_LINEAR are "
                        "scene-linear and get encoded to the working curve for you; "
                        "ACESCCT means the file is already encoded and is passed "
                        "through. Same three choices as the official --hdr flag."
                    ),
                ),
                IO.Combo.Input(
                    "output_color_space",
                    options=list(OUTPUT_SPACES),
                    default="REC709_LINEAR",
                    tooltip=(
                        "What the master comes out as. Scene-linear Rec.709 is what "
                        "EXR savers assume when they say 'linear'."
                    ),
                ),
                IO.Boolean.Input(
                    "strict_validation",
                    default=True,
                    tooltip=(
                        "Refuse to continue on mismatched sizes, NaN/Inf or an SDR "
                        "image fed into the HDR branch, instead of quietly making a "
                        "master nobody can use."
                    ),
                ),
                IO.Float.Input(
                    "preview_exposure_ev",
                    default=0.0,
                    min=-16.0,
                    max=16.0,
                    step=0.1,
                    tooltip=(
                        "Exposure of the SDR preview, in stops. Affects only what you "
                        "look at — never the EXR master."
                    ),
                ),
                IO.Combo.Input(
                    "preview_tonemap",
                    options=list(OPERATORS),
                    default="reinhard_luma",
                    tooltip=(
                        "How the preview squeezes HDR onto an ordinary screen. "
                        "reinhard_luma is the neutral one, aces_filmic is punchier, "
                        "clip just cuts everything above 1.0."
                    ),
                ),
                IO.Combo.Input(
                    "preview_dtype",
                    options=["FP16", "FP32"],
                    default="FP16",
                    advanced=True,
                    tooltip=(
                        "FP16 halves the memory the preview takes. For 129 frames at "
                        "1920x1088 that is 1.5 GiB instead of 3.0 GiB."
                    ),
                ),
                # ⚠️ ПОСЛЕДНИМ В СХЕМЕ: порядок виджетов позиционен в
                # сохранённом workflow (§4 CLAUDE.md). Умолчание — нативный
                # путь, поэтому старый граф ведёт себя как прежде.
                IO.Combo.Input(
                    "hdr_mode",
                    options=list(HDR_MODES),
                    default=MODE_PRESERVE,
                    tooltip=(
                        "Which HDR technology this graph uses. 'preserve' is the native "
                        "LTX 2.5 path: the range came in from an EXR and must survive "
                        "(ACEScct). 'expand' is the HDR IC-LoRA: there was no range on "
                        "the way in and the model grows it out of ordinary SDR (LogC3). "
                        "They are different technologies with different curves — mixing "
                        "them shifts the colour. The LoRA is validated on LTX 2.3; "
                        "support for 2.5 is, officially, in development."
                    ),
                ),
            ],
            outputs=[
                HdrConfigIO.Output(
                    display_name="config",
                    tooltip="Feed this into every other TS LTX HDR node.",
                ),
                IO.Boolean.Output(
                    display_name="enabled",
                    tooltip="The same switch as a plain boolean, for your own gates.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        enabled: bool = False,
        input_color_space: str = "ACESCG",
        output_color_space: str = "REC709_LINEAR",
        strict_validation: bool = True,
        preview_exposure_ev: float = 0.0,
        preview_tonemap: str = "reinhard_luma",
        preview_dtype: str = "FP16",
        hdr_mode: str = MODE_PRESERVE,
    ) -> IO.NodeOutput:
        config = HdrConfig(
            enabled=bool(enabled),
            input_color_space=str(input_color_space),
            output_color_space=str(output_color_space),
            strict_validation=bool(strict_validation),
            preview_exposure_ev=float(preview_exposure_ev),
            preview_tonemap=str(preview_tonemap),
            preview_dtype=str(preview_dtype),
            hdr_mode=str(hdr_mode),
        )
        logger.debug("%s %s", LOG_PREFIX, config)
        return IO.NodeOutput(config, config.enabled)


NODE_CLASS_MAPPINGS = {"TS_LTXHDRSettings": TS_LTXHDRSettings}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LTXHDRSettings": "TS LTX HDR Settings"}
