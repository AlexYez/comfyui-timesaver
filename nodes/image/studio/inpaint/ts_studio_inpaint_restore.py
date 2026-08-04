"""Возврат перерисованного выреза на место: вторая половина пары.

Патч масштабируется обратно в родной размер рамки, у него снимается цветовой
сдвиг кругового рейса VAE (оценка берётся по сохранённому кольцу вокруг маски,
где содержимое не менялось), и он вкладывается в кадр по растушёванной альфе —
в линейном свете, иначе по краю остаётся тёмная линия.

Первая половина — `TS_StudioInpaintCrop`. Технология общая со `TS Smart
Inpaint`: `nodes/image/_inpaint_crop.py`.

node_id: TS_StudioInpaintRestore
"""
from __future__ import annotations

import logging

import torch
from comfy_api.v0_0_2 import IO

from ..markers._marker_shared import CATEGORY
from ._plan import PLAN_TYPE, CropPlanHolder

logger = logging.getLogger("comfyui_timesaver.ts_studio_inpaint_restore")
LOG_PREFIX = "[TS Studio Inpaint Restore]"

from ..._inpaint_crop import paste_back  # noqa: E402  (после логгера — читается вместе с ним)


class TS_StudioInpaintRestore(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_StudioInpaintRestore",
            display_name="TS Studio Inpaint Restore",
            category=CATEGORY,
            description=(
                "Pastes a repainted crop back into the frame it came from: "
                "scales it to the native size of the crop, neutralises the "
                "colour shift of the VAE round-trip using the preserved ring "
                "around the mask, and composites it under the soft edge in "
                "linear light. Pair it with TS Studio Inpaint Crop."
            ),
            inputs=[
                IO.Image.Input("image", tooltip="The original full frame."),
                IO.Image.Input("patch", tooltip="The repainted crop."),
                PLAN_TYPE.Input("plan", tooltip="Plan from TS Studio Inpaint Crop."),
            ],
            outputs=[IO.Image.Output(display_name="image")],
            search_aliases=["studio inpaint restore", "paste crop back"],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, patch: torch.Tensor, plan) -> IO.NodeOutput:
        holder = plan.plan if isinstance(plan, CropPlanHolder) else plan
        if not hasattr(holder, "y0"):
            raise RuntimeError(
                f"{LOG_PREFIX} plan input is not a crop plan — connect TS Studio Inpaint Crop"
            )
        if image.ndim != 4 or patch.ndim != 4:
            raise RuntimeError(
                f"{LOG_PREFIX} image and patch must be [B,H,W,C], got "
                f"{tuple(image.shape)} and {tuple(patch.shape)}"
            )
        out = paste_back(image, patch, holder)
        logger.info(
            "%s pasted %dx%d back into %dx%d at (%d,%d)",
            LOG_PREFIX, int(patch.shape[2]), int(patch.shape[1]),
            int(image.shape[2]), int(image.shape[1]), holder.x0, holder.y0,
        )
        return IO.NodeOutput(out)


NODE_CLASS_MAPPINGS = {"TS_StudioInpaintRestore": TS_StudioInpaintRestore}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_StudioInpaintRestore": "TS Studio Inpaint Restore"}
