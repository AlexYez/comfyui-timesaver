"""Depth Anything V2 — модель для ОДНОГО кадра.

⚠️ Отдельная модель, а не режим видео-модели. Video-Depth-Anything обучена на
32-кадровом окне, и одиночный снимок ей приходится подавать дублированным — от
этого динамический диапазон глубины сплющивается (замерено: лицо и волосы
выбиваются в белое, структура теряется). Depth Anything V2 обучена ровно на
стоп-кадр и на нём же и лучше, и в разы быстрее.

⚠️ Своего кода архитектуры здесь почти нет, и это НЕ лень. Video-Depth-Anything
построена поверх Depth Anything V2: её ``DPTHeadTemporal`` наследует ``DPTHead``,
а костяк у обеих — один и тот же DINOv2. Проверено загрузкой опубликованного
чекпойнта DA-V2 Large: ``strict=True`` проходит без единого лишнего или
недостающего ключа. Вендорить вторую копию тех же слоёв означало бы держать в
паке два экземпляра одного кода, которые разойдутся при первой же правке.

Модуль знает только torch и соседние вендоренные модули.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dinov2 import DINOv2
from .dpt import DPTHead

# Из каких блоков DINOv2 берутся признаки для головы. Числа — контракт
# опубликованных весов, а не настройка: сдвинешь их, и модель соберётся, но
# будет читать не те слои и вернёт правдоподобную чушь.
INTERMEDIATE_LAYERS = {
    "vits": [2, 5, 8, 11],
    "vitb": [2, 5, 8, 11],
    "vitl": [4, 11, 17, 23],
    "vitg": [9, 19, 29, 39],
}

# Конфигурации из репозитория авторов.
MODEL_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


class DepthAnythingV2(nn.Module):
    """DINOv2 + DPT-голова. Вход ``(B, 3, H, W)``, выход ``(B, H, W)``."""

    def __init__(self, encoder: str = "vitl", features: int | None = None,
                 out_channels: list[int] | None = None):
        super().__init__()
        if encoder not in INTERMEDIATE_LAYERS:
            raise ValueError(f"Unknown encoder {encoder!r}; expected one of {list(INTERMEDIATE_LAYERS)}.")
        config = MODEL_CONFIGS[encoder]
        self.encoder = encoder
        self.intermediate_layer_idx = INTERMEDIATE_LAYERS[encoder]
        self.pretrained = DINOv2(model_name=encoder)
        self.depth_head = DPTHead(
            self.pretrained.embed_dim,
            features if features is not None else config["features"],
            out_channels=out_channels if out_channels is not None else config["out_channels"],
            use_clstoken=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Оценить глубину.

        Args:
            x: ``(B, 3, H, W)``, нормированный по ImageNet. H и W обязаны быть
                кратны 14 — этого требует сетка патчей DINOv2.

        Returns:
            ``(B, H, W)`` — относительная обратная глубина, неотрицательная.
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"DepthAnythingV2 expects (B, 3, H, W), got {tuple(x.shape)}.")
        if x.shape[-2] % 14 or x.shape[-1] % 14:
            raise ValueError(
                f"Both sides must be multiples of 14 (DINOv2 patch grid); got {tuple(x.shape[-2:])}."
            )
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx, return_class_token=True,
        )
        depth = self.depth_head(features, patch_h, patch_w)
        return F.relu(depth).squeeze(1)
