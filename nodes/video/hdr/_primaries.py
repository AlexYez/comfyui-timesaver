"""Смена примариев: Rec.709 ↔ ACEScg (AP1).

Матрицы не переписаны откуда-то цифрами, а **выводятся из опубликованных
хроматичностей** BT.709 и ACES прямо здесь, при импорте. Так видно, откуда
взялось каждое число, и пакет не наследует чужую лицензию (§19 плана).

Сверено: результат совпадает с общепринятой матрицей Rec.709 → ACEScg с
точностью 4.8e-7, а строки дают ровно 1.0 — то есть белый остаётся белым.
Оба факта стережёт ``tests/test_hdr_color_engine.py``.

Адаптация белой точки D65 → ACES выполняется преобразованием Брэдфорда: у ACES
своя белая точка (0.32168, 0.33767), и без адаптации нейтральный серый уехал бы
в зелень.

⚠️ Гамут не подрезаем. После ACEScg → Rec.709 у цветов вне гамута Rec.709
появляются отрицательные компоненты — это не ошибка, а честное «такой цвет
здесь непредставим». Зажимаем снизу только на самой границе мастера, там же,
где это делает официальный путь.

Модуль знает только torch.
"""

from __future__ import annotations

import torch

Matrix3 = list[list[float]]

# Хроматичности из стандартов. Первый элемент — R/G/B, второй — белая точка.
REC709_CHROMA = (((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), (0.3127, 0.3290))
AP1_CHROMA = (((0.713, 0.293), (0.165, 0.830), (0.128, 0.044)), (0.32168, 0.33767))

# Матрица конусного отклика Брэдфорда (CIE 159:2004).
_BRADFORD: Matrix3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
]


def _inverse(m: Matrix3) -> Matrix3:
    """Обратная матрица 3×3 через алгебраические дополнения."""
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ValueError("Singular 3x3 matrix: check the chromaticities.")
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]


def _matmul(x: Matrix3, y: Matrix3) -> Matrix3:
    return [[sum(x[r][k] * y[k][c] for k in range(3)) for c in range(3)] for r in range(3)]


def _apply(m: Matrix3, v: list[float]) -> list[float]:
    return [sum(m[r][k] * v[k] for k in range(3)) for r in range(3)]


def _rgb_to_xyz(chroma) -> tuple[Matrix3, list[float]]:
    """Матрица RGB → XYZ и XYZ белой точки для заданных хроматичностей."""
    ((xr, yr), (xg, yg), (xb, yb)), (xw, yw) = chroma
    basis: Matrix3 = [
        [xr, xg, xb],
        [yr, yg, yb],
        [1.0 - xr - yr, 1.0 - xg - yg, 1.0 - xb - yb],
    ]
    white = [xw / yw, 1.0, (1.0 - xw - yw) / yw]
    scale = _apply(_inverse(basis), white)
    return [[basis[r][c] * scale[c] for c in range(3)] for r in range(3)], white


def _adaptation(source_white: list[float], target_white: list[float]) -> Matrix3:
    """Преобразование Брэдфорда между двумя белыми точками."""
    src = _apply(_BRADFORD, source_white)
    dst = _apply(_BRADFORD, target_white)
    ratio: Matrix3 = [
        [dst[0] / src[0], 0.0, 0.0],
        [0.0, dst[1] / src[1], 0.0],
        [0.0, 0.0, dst[2] / src[2]],
    ]
    return _matmul(_inverse(_BRADFORD), _matmul(ratio, _BRADFORD))


def _derive() -> tuple[Matrix3, Matrix3]:
    rec709, white_709 = _rgb_to_xyz(REC709_CHROMA)
    ap1, white_ap1 = _rgb_to_xyz(AP1_CHROMA)
    forward = _matmul(_inverse(ap1), _matmul(_adaptation(white_709, white_ap1), rec709))
    return forward, _inverse(forward)


REC709_TO_AP1, AP1_TO_REC709 = _derive()

# Тензоры матриц дороже собирать, чем хранить: один и тот же кадр за прогон
# проходит через них десятки раз. Модульный словарь — санкционированный кэш
# (§5 CLAUDE.md); ключ включает устройство, иначе GPU-кадр потянул бы матрицу
# с CPU и уронил бы прогон.
_tensor_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _as_tensor(matrix: Matrix3, like: torch.Tensor) -> torch.Tensor:
    key = (id(matrix), str(like.device), torch.float32)
    cached = _tensor_cache.get(key)
    if cached is None:
        cached = torch.tensor(matrix, dtype=torch.float32, device=like.device)
        _tensor_cache[key] = cached
    return cached


def _convert(x: torch.Tensor, matrix: Matrix3) -> torch.Tensor:
    """Применить матрицу к последней оси тензора ``[..., 3]``."""
    if x.shape[-1] != 3:
        raise ValueError(
            f"Primaries conversion needs 3 channels in the last axis, got {tuple(x.shape)}.")
    value = x.to(torch.float32)
    return value @ _as_tensor(matrix, value).transpose(0, 1)


def rec709_to_acescg(x: torch.Tensor) -> torch.Tensor:
    """Линейный Rec.709 → линейный ACEScg (AP1). Форма ``[..., 3]``."""
    return _convert(x, REC709_TO_AP1)


def acescg_to_rec709(x: torch.Tensor) -> torch.Tensor:
    """Линейный ACEScg (AP1) → линейный Rec.709. Форма ``[..., 3]``.

    Отрицательные компоненты на выходе — норма для цветов вне гамута Rec.709;
    здесь их не трогаем, зажим снизу делается один раз на границе мастера.
    """
    return _convert(x, AP1_TO_REC709)
