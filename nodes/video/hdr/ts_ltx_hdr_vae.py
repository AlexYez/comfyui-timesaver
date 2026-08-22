"""TS LTX HDR VAE — тот же VAE, но в float32, только для финального decode.

Штатный ``Load VAE`` точность не спрашивает: её выбирает model management, и для
LTX 2.5 это bf16. Для картинки этого хватает с запасом, для HDR-мастера — нет:
в тенях и в верхних стопах bf16 даёт заметный шаг квантования, а именно там и
живёт весь смысл HDR.

Что остаётся на обычном VAE и **не должно** переезжать сюда:

- кодирование опорных кадров обеих стадий;
- латентный апскейлер;
- обычный SDR-decode.

⚠️ Две копии одного видео-VAE в памяти стоят дорого, поэтому нода задумана
ленивой: провод идёт в ленивый вход ``hdr_vae`` декодера, и при выключенном HDR
ComfyUI сюда просто не заходит. Не подключайте её выход никуда ещё — этим
ленивость и потеряется.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.v0_0_2 import IO

from ._hdr_types import LOG_PREFIX, as_config
from ._schema import CATEGORY, HdrConfigIO
from ._vae import load_vae, vae_names

logger = logging.getLogger("comfyui_timesaver.ts_ltx_hdr.vae_node")

NO_VAE = "(no VAE files found)"

_PRECISIONS = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


class TS_LTXHDRVAE(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        names = vae_names() or [NO_VAE]
        return IO.Schema(
            node_id="TS_LTXHDRVAE",
            display_name="TS LTX HDR VAE",
            category=CATEGORY,
            description=(
                "Load a VAE at an explicit precision. Used for the final HDR decode "
                "only — the rest of the graph keeps the VAE it already had."
            ),
            search_aliases=["fp32 vae", "float32 vae", "hdr vae", "vae precision"],
            inputs=[
                IO.Combo.Input(
                    "vae_name",
                    options=names,
                    tooltip="Normally the same file the rest of the graph uses.",
                ),
                IO.Combo.Input(
                    "precision",
                    options=list(_PRECISIONS),
                    default="float32",
                    tooltip=(
                        "float32 is the point of this node. The others are here for "
                        "measuring what the precision actually buys you."
                    ),
                ),
                HdrConfigIO.Input(
                    "config",
                    optional=True,
                    tooltip="Settings node. Only used to say so in the log.",
                ),
            ],
            outputs=[
                IO.Vae.Output(
                    display_name="vae",
                    tooltip="Connect to the hdr_vae input of TS LTX HDR Decode, and nowhere else.",
                ),
            ],
        )

    @classmethod
    def execute(cls, vae_name: str = "", precision: str = "float32", config=None) -> IO.NodeOutput:
        name = str(vae_name or "")
        if not name or name == NO_VAE:
            raise RuntimeError(
                f"{LOG_PREFIX} No VAE chosen. Put the LTX 2.5 video VAE into "
                "models/vae and pick it here."
            )
        dtype = _PRECISIONS.get(str(precision), torch.float32)
        settings = as_config(config)
        logger.info("%s loading %s at %s (%s)", LOG_PREFIX, name,
                    str(dtype).replace("torch.", ""), settings)
        return IO.NodeOutput(load_vae(name, dtype))


NODE_CLASS_MAPPINGS = {"TS_LTXHDRVAE": TS_LTXHDRVAE}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LTXHDRVAE": "TS LTX HDR VAE"}
