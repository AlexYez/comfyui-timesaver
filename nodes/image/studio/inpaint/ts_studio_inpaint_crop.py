"""Вырез по маске для перерисовки: первая половина пары.

Перерисовывать весь кадр — дорого и глупо: модель тратит всю свою способность
на пиксели, которые никто не просил менять, а маленькая правка получает столько
разрешения, сколько ей досталось от кадра. Эта нода вырезает область маски с
запасом контекста и приводит вырез к выбранному бюджету мегапикселей — тому
разрешению, на котором модель действительно рисует детали.

Вторая половина — `TS_StudioInpaintRestore`: она возвращает перерисованный
вырез на место. Между ними стоит любой сэмплер; в студии это LanPaint.

Технология общая со `TS Smart Inpaint` — `nodes/image/_inpaint_crop.py`.

node_id: TS_StudioInpaintCrop
"""
from __future__ import annotations

import logging

import torch
from comfy_api.v0_0_2 import IO

from ..markers._marker_shared import CATEGORY
from ._plan import PLAN_TYPE, CropPlanHolder

logger = logging.getLogger("comfyui_timesaver.ts_studio_inpaint_crop")
LOG_PREFIX = "[TS Studio Inpaint Crop]"

from ..._inpaint_crop import REPLACE_DENOISE, plan_and_crop  # noqa: E402  (после логгера — читается вместе с ним)


class TS_StudioInpaintCrop(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_StudioInpaintCrop",
            display_name="TS Studio Inpaint Crop",
            category=CATEGORY,
            description=(
                "Crops the masked area with a context band and scales it to the "
                "processing budget, so a repaint spends the model's resolution "
                "where the mask is instead of on the whole frame. Feed the crop "
                "and its mask to any inpainting sampler, then hand the result "
                "plus this node's plan to TS Studio Inpaint Restore."
            ),
            inputs=[
                IO.Image.Input("image", tooltip="Full source frame."),
                IO.Mask.Input("mask", tooltip="Painted mask: white = repaint."),
                IO.Float.Input(
                    "megapixels", default=1.0, min=0.1, max=8.0, step=0.1,
                    tooltip=(
                        "Resolution the crop is actually processed at. Small "
                        "selections are enlarged toward it (up to 3x) for detail; "
                        "oversized ones are reduced to it so a big mask in an 8K "
                        "frame cannot hang the machine."
                    ),
                ),
                IO.Float.Input(
                    "context_pct", default=8.0, min=0.0, max=50.0, step=0.5,
                    tooltip=(
                        "How much context to take around the mask, as a percent "
                        "of the mask's own size. More context helps the model "
                        "match the surroundings; it also enlarges the crop, so "
                        "less of the budget lands on the mask itself. Values "
                        "below the 32px colour-analysis ring change nothing."
                    ),
                ),
                IO.Float.Input(
                    "denoise", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip=(
                        "How much of the masked area the sampler will replace. "
                        "Full replacement leaves the model no pixels to anchor on, "
                        "so the crop then takes more context and is enlarged less "
                        "— otherwise it paints a scene of its own inside the mask."
                    ),
                ),
                IO.Float.Input(
                    "feather_pct", default=3.0, min=0.0, max=25.0, step=0.5,
                    tooltip=(
                        "Width of the soft edge used when the repaint is pasted "
                        "back, as a percent of the mask's own size."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="crop"),
                IO.Mask.Output(display_name="crop_mask"),
                IO.Mask.Output(
                    display_name="crop_mask_soft",
                    tooltip=(
                        "The same mask with its soft edge kept. Feed this one when "
                        "Differential Diffusion patches the model: it turns the ramp "
                        "into a per-pixel denoise schedule instead of throwing it away."
                    ),
                ),
                PLAN_TYPE.Output(
                    display_name="plan",
                    tooltip="Crop geometry and its soft edge — feed to TS Studio Inpaint Restore.",
                ),
                IO.Image.Output(display_name="source", tooltip="The frame, passed through."),
            ],
            search_aliases=["studio inpaint crop", "crop to mask for inpaint"],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        mask: torch.Tensor,
        megapixels: float,
        context_pct: float,
        denoise: float,
        feather_pct: float,
    ) -> IO.NodeOutput:
        if image.ndim != 4:
            raise RuntimeError(
                f"{LOG_PREFIX} image must be [B,H,W,C], got {tuple(image.shape)}"
            )
        # Доработка НЕ увеличивает вырез.
        #
        # Замерено 2026-08-04 на одном сиде: тот же вырез, та же маска, сила
        # 0.45. При бюджете 0.8 МП (увеличение в 1.28 раза) кожа выходила
        # потрескавшейся — модель «дорисовывала детали» по интерполированным
        # пикселям и запекала их в сетку трещин. При родном масштабе того же
        # выреза — чистое лицо. Разница только в увеличении.
        #
        # Смысл прямой: доработка улучшает то, что есть, а не выдумывает то,
        # чего нет. Бюджет мегапикселей при этом продолжает работать вниз —
        # большая маска по-прежнему ужимается до него, чтобы не разорить
        # память. Полная замена увеличивать по-прежнему вправе: там пикселей
        # под маской всё равно не остаётся.
        refining = float(denoise) < REPLACE_DENOISE
        result = plan_and_crop(
            image, mask,
            megapixels=float(megapixels),
            context_pct=float(context_pct),
            feather_pct=float(feather_pct),
            denoise=float(denoise),
            max_linear=1.0 if refining else None,
        )
        if result is None:
            # Пустая маска — не ошибка: перерисовывать нечего. Отдаём кадр целиком
            # и план во весь кадр, чтобы возврат положил ответ ровно назад и
            # граф не падал на полпути.
            logger.info("%s empty mask — the frame passes through unchanged", LOG_PREFIX)
            plan = CropPlanHolder.whole_frame(image)
            empty = torch.zeros_like(image[:1, ..., 0])
            return IO.NodeOutput(image, empty, empty, plan, image)

        crop, crop_mask, crop_soft, plan = result
        logger.info(
            "%s crop=%dx%d -> %dx%d (%.2f MP) context=%.1f%%->%dpx feather=%.1f%%->%dpx",
            LOG_PREFIX, plan.crop_w, plan.crop_h, plan.out_w, plan.out_h,
            plan.out_w * plan.out_h / 1_000_000.0,
            float(context_pct), int(round(plan.context_px)),
            float(feather_pct), int(round(plan.feather_px)),
        )
        soft = crop_soft if crop_soft is not None else crop_mask
        return IO.NodeOutput(crop, crop_mask, soft, CropPlanHolder(plan), image)


NODE_CLASS_MAPPINGS = {"TS_StudioInpaintCrop": TS_StudioInpaintCrop}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_StudioInpaintCrop": "TS Studio Inpaint Crop"}
