"""План выреза, который ездит по графу между двумя половинами перерисовки.

Держать геометрию в отдельном типе, а не пересчитывать её при возврате, —
осознанное решение: маска, растушёвка и рамка обязаны быть теми же самыми, что
видела модель. Пересчёт по тем же настройкам дал бы почти те же числа, и
«почти» вылезло бы швом по краю.
"""
from __future__ import annotations

import torch
from comfy_api.v0_0_2 import IO

from ..._inpaint_crop import CropPlan

PLAN_TYPE = IO.Custom("TS_INPAINT_PLAN")


class CropPlanHolder:
    """Обёртка над `CropPlan` — то, что течёт по проводу графа.

    Отдельный класс нужен, чтобы у значения был понятный repr в интерфейсе и
    чтобы позже можно было добавить поля, не трогая тип провода.
    """

    def __init__(self, plan: CropPlan) -> None:
        self.plan = plan

    @classmethod
    def whole_frame(cls, image: torch.Tensor) -> "CropPlanHolder":
        """План «весь кадр целиком» — для случая пустой маски.

        Альфа нулевая: возврат тогда оставит кадр как есть, что и требуется,
        когда перерисовывать нечего.
        """
        h = int(image.shape[1])
        w = int(image.shape[2])
        alpha = torch.zeros((1, h, w, 1), dtype=torch.float32)
        return cls(CropPlan(
            y0=0, y1=h, x0=0, x1=w, out_w=w, out_h=h,
            alpha=alpha, feather_px=0.0, context_px=0.0, color_correct=False,
        ))

    def __repr__(self) -> str:  # pragma: no cover — только для интерфейса
        p = self.plan
        return (f"CropPlan({p.x0},{p.y0})-({p.x1},{p.y1}) "
                f"-> {p.out_w}x{p.out_h}")
