"""Тон-маппинг — **только для просмотра**.

Мастер и превью здесь разведены намеренно и жёстко. Всё, что делает этот
модуль, — способ посмотреть HDR на обычном мониторе; ни экспозиция, ни оператор
не имеют права протечь в EXR. Стоит один раз смешать эти два пути, и человек
сохранит красивую картинку вместо сцены, а поймёт это уже в монтажной.

Считаем кусками по кадрам: 129 кадров 1920×1088 в float32 — это 3 ГиБ на один
тензор, и промежуточные копии тон-маппинга удвоили бы цифру.

Модуль знает только torch.
"""

from __future__ import annotations

import torch

OPERATORS = ("reinhard_luma", "aces_filmic", "clip")

# Веса яркости Rec.709 (BT.709-6).
_LUMA = (0.2126, 0.7152, 0.0722)

# Сколько кадров обрабатывать за раз. 8 кадров 1920×1088 float32 — около 190 МиБ
# со всеми промежуточными копиями: заметно дешевле целого ролика и не настолько
# мелко, чтобы накладные расходы стали видны.
_CHUNK = 8


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb[..., 0] * _LUMA[0] + rgb[..., 1] * _LUMA[1] + rgb[..., 2] * _LUMA[2])


def _reinhard_luma(rgb: torch.Tensor) -> torch.Tensor:
    """Райнхард по яркости: сжимается светимость, а не каналы по отдельности.

    Поканальный вариант тянет насыщенные цвета к белому; здесь оттенок
    сохраняется, потому что все три канала делятся на одно и то же число.
    """
    luma = _luminance(rgb)
    scale = torch.where(luma > 1e-8, (luma / (1.0 + luma)) / luma.clamp_min(1e-8),
                        torch.ones_like(luma))
    return rgb * scale.unsqueeze(-1)


def _aces_filmic(rgb: torch.Tensor) -> torch.Tensor:
    """Известное рациональное приближение ACES RRT+ODT (Narkowicz, 2015).

    Контрастнее Райнхарда и ближе к тому, как HDR выглядит после нормальной
    цветокоррекции. По-прежнему только превью.
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)


def _srgb_oetf(x: torch.Tensor) -> torch.Tensor:
    """Прямая кривая sRGB (IEC 61966-2-1): линейный свет → код монитора."""
    low = x * 12.92
    high = 1.055 * torch.pow(x.clamp_min(1e-8), 1.0 / 2.4) - 0.055
    return torch.where(x <= 0.0031308, low, high)


def make_sdr_preview(
    hdr_linear_rec709: torch.Tensor,
    *,
    exposure_ev: float = 0.0,
    operator: str = "reinhard_luma",
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Сделать обычную SDR-картинку из сцены в линейном Rec.709.

    Args:
        hdr_linear_rec709: ``[B, H, W, 3]``, линейный свет, значения могут быть
            больше 1. **Не изменяется.**
        exposure_ev: сдвиг экспозиции в стопах, только для просмотра.
        operator: один из :data:`OPERATORS`.
        output_dtype: тип превью; float16 вдвое дешевле по памяти.

    Returns:
        ``[B, H, W, 3]`` в ``[0, 1]``, закодированный кривой sRGB.
    """
    if hdr_linear_rec709.ndim != 4 or hdr_linear_rec709.shape[-1] != 3:
        raise ValueError(
            f"Preview expects [B, H, W, 3], got {tuple(hdr_linear_rec709.shape)}.")
    if operator not in OPERATORS:
        raise ValueError(f"Unknown tonemap '{operator}'. Available: {', '.join(OPERATORS)}.")

    gain = float(2.0 ** float(exposure_ev))
    frames = int(hdr_linear_rec709.shape[0])
    out = torch.empty(hdr_linear_rec709.shape, dtype=output_dtype,
                      device=hdr_linear_rec709.device)

    for start in range(0, frames, _CHUNK):
        stop = min(frames, start + _CHUNK)
        # Отрицательные компоненты вне гамута монитор всё равно не покажет;
        # для превью — и только для него — их можно просто отбросить.
        chunk = hdr_linear_rec709[start:stop].to(torch.float32).clamp_min(0.0) * gain
        if operator == "reinhard_luma":
            chunk = _reinhard_luma(chunk)
        elif operator == "aces_filmic":
            chunk = _aces_filmic(chunk)
        chunk = _srgb_oetf(chunk.clamp(0.0, 1.0))
        out[start:stop] = chunk.clamp(0.0, 1.0).to(output_dtype)

    return out
