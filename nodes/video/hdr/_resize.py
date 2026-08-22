"""Изменение размера в линейном свете.

Обычные ноды масштабирования пака сюда не годятся: они рассчитаны на диапазон
``[0, 1]`` и местами уходят в восемь бит. HDR-кадр надо ресайзить **до** любой
кривой, прямо в линейных значениях — так делает официальный путь LTX, и так
сохраняется энергия: среднее двух пикселей в линейном свете это и есть их
физическая сумма, а среднее их же логарифмов — нет.

⚠️ Половинный кадр для первой стадии готовится **из оригинала**, а не
уменьшением уже готового полного. Официальный двухстадийный pipeline собирает
image conditioning заново для каждого разрешения; уменьшение готового
рабочего сигнала дало бы другие числа.

Модуль знает только torch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Как вписывать кадр в целевой размер.
FIT_MODES = ("center_crop", "reflect_pad")


def _to_nchw(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 3, 1, 2).contiguous()


def _to_nhwc(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 2, 3, 1).contiguous()


def _scale(image: torch.Tensor, width: int, height: int, interpolation: str) -> torch.Tensor:
    """Масштабировать ``[B, C, H, W]`` до точного размера.

    ``area`` для уменьшения — усреднение по площади, без муара; для увеличения
    оно бессмысленно, поэтому там всегда bicubic/bilinear.
    """
    source_h, source_w = int(image.shape[-2]), int(image.shape[-1])
    if source_h == height and source_w == width:
        return image

    shrinking = width <= source_w and height <= source_h
    mode = "area" if (interpolation == "area" and shrinking) else interpolation
    if mode == "area":
        return F.interpolate(image, size=(height, width), mode="area")
    if mode not in ("bicubic", "bilinear", "nearest-exact"):
        mode = "bicubic"
    kwargs = {} if mode == "nearest-exact" else {"align_corners": False, "antialias": shrinking}
    return F.interpolate(image, size=(height, width), mode=mode, **kwargs)


def _pad_reflect(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Дополнить до целевого размера отражением краёв.

    ⚠️ ``reflect`` в torch требует, чтобы поле было меньше самой стороны. При
    сильно непохожих пропорциях это не так, и тогда берётся ``replicate``: он
    выглядит хуже, но не роняет прогон.
    """
    source_h, source_w = int(image.shape[-2]), int(image.shape[-1])
    pad_x, pad_y = max(0, width - source_w), max(0, height - source_h)
    if pad_x == 0 and pad_y == 0:
        return image
    left, top = pad_x // 2, pad_y // 2
    padding = (left, pad_x - left, top, pad_y - top)
    mode = "reflect" if (max(padding[:2]) < source_w and max(padding[2:]) < source_h) else "replicate"
    return F.pad(image, padding, mode=mode)


def _crop_center(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    source_h, source_w = int(image.shape[-2]), int(image.shape[-1])
    left = max(0, (source_w - width) // 2)
    top = max(0, (source_h - height) // 2)
    return image[..., top:top + height, left:left + width]


def fit_linear(
    image: torch.Tensor,
    width: int,
    height: int,
    *,
    fit_mode: str = "center_crop",
    interpolation: str = "area",
) -> torch.Tensor:
    """Вписать линейный кадр ``[B, H, W, 3]`` в размер ``width × height``.

    ``center_crop`` масштабирует так, чтобы кадр накрыл цель, и срезает лишнее
    по центру — композиция сохраняется, края теряются. ``reflect_pad``
    масштабирует так, чтобы кадр целиком поместился, и достраивает поля
    отражением — ничего не теряется, но по краям появляется выдумка.

    Args:
        image: линейный HDR-кадр, значения могут быть больше 1.
        width: целевая ширина, пикселей.
        height: целевая высота, пикселей.
        fit_mode: один из :data:`FIT_MODES`.
        interpolation: ``area``, ``bicubic``, ``bilinear`` или ``nearest-exact``.

    Returns:
        Тензор ``[B, height, width, 3]``, float32.
    """
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(
            f"Linear resize expects [B, H, W, 3], got {tuple(image.shape)}.")
    if width <= 0 or height <= 0:
        raise ValueError(f"Target size must be positive, got {width}x{height}.")
    if fit_mode not in FIT_MODES:
        raise ValueError(f"Unknown fit mode '{fit_mode}'. Available: {', '.join(FIT_MODES)}.")

    source = _to_nchw(image.to(torch.float32))
    source_h, source_w = int(source.shape[-2]), int(source.shape[-1])

    # cover — накрыть цель и срезать; contain — поместиться целиком и дополнить.
    ratio_x, ratio_y = width / source_w, height / source_h
    ratio = max(ratio_x, ratio_y) if fit_mode == "center_crop" else min(ratio_x, ratio_y)
    scaled_w = max(1, int(round(source_w * ratio)))
    scaled_h = max(1, int(round(source_h * ratio)))

    scaled = _scale(source, scaled_w, scaled_h, interpolation)
    if fit_mode == "center_crop":
        fitted = _crop_center(scaled, width, height)
    else:
        fitted = _pad_reflect(scaled, width, height)
        fitted = _crop_center(fitted, width, height)

    # Округление могло дать пиксель недобора — дотягиваем точно, без сюрпризов
    # для нод, которые ждут ровно запрошенный размер.
    if int(fitted.shape[-2]) != height or int(fitted.shape[-1]) != width:
        fitted = _scale(fitted, width, height, interpolation)
    return _to_nhwc(fitted)
