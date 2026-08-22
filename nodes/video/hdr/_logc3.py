"""ARRI LogC3 — кривая второго HDR-пути LTX, того что через IC-LoRA.

Два HDR-пути LTX — это две РАЗНЫЕ технологии, и путать их кривые нельзя:

- **нативный HDR 2.5** сохраняет диапазон, который пришёл из EXR, и работает в
  ACEScct (см. :mod:`._acescct`);
- **HDR IC-LoRA** (проверена на 2.3, для 2.5 «в разработке») наоборот
  ВЫРАЩИВАЕТ диапазон из обычного SDR-материала и работает в LogC3.

Отдать вывод одного обратной кривой другого — гарантированный сдвиг цвета,
поэтому режим выбирается один раз в ноде настроек и дальше едет с конфигом.

⚠️ **В пути IC-LoRA примарии НЕ меняются.** Официальная нода
``LTXVHDRDecodePostprocess`` раскодирует LogC3 и отдаёт линейный HDR как есть,
без перехода AP1 → Rec.709: модель уже выдаёт нужные примарии. Применить здесь
матрицу ACES-пути значило бы увести цвет.

Константы — опубликованные коэффициенты ARRI LogC3 при EI 800. Реализация
независимая, но поведение сверено с официальной нодой пак-ом LTX: тест
``test_logc3_matches_the_official_implementation``.

Модуль знает только torch.
"""

from __future__ import annotations

import torch

# Опубликованные коэффициенты ARRI LogC3, EI 800.
_A = 5.555556
_B = 0.052272
_C = 0.247190
_D = 0.385537
_E = 5.367655
_F = 0.092809
_CUT = 0.010591

# Стык участков со стороны кода: тот же перелом, посчитанный из линейного.
_CUT_CODE = _E * _CUT + _F

# Линейная яркость, которой соответствует код 1.0. У LogC3 потолок заметно ниже,
# чем у ACEScct (222.86), и об этом стоит знать, глядя на статистику.
WORKING_CEILING = float((10.0 ** ((1.0 - _D) / _C) - _B) / _A)

# Официальная нода после раскодирования зажимает результат сверху этим числом.
# Повторяем, чтобы мастер совпадал с тем, что человек увидит в их же ветке.
CEILING_CLAMP = 1e4


def linear_to_logc3(x: torch.Tensor) -> torch.Tensor:
    """Линейный свет → код LogC3 в ``[0, 1]``."""
    value = x.to(torch.float32).clamp_min(0.0)
    log_part = _C * torch.log10(_A * value + _B) + _D
    lin_part = _E * value + _F
    return torch.where(value >= _CUT, log_part, lin_part).clamp(0.0, 1.0)


def logc3_to_linear(code: torch.Tensor) -> torch.Tensor:
    """Код LogC3 → линейный свет.

    Зажимает снизу нулём и сверху :data:`CEILING_CLAMP` — ровно так же, как
    официальная нода: без верхнего предела единичный шум в тенях кривой
    превращался бы в астрономические числа.
    """
    logc = code.to(torch.float32).clamp(0.0, 1.0)
    from_log = (torch.pow(torch.tensor(10.0, dtype=torch.float32, device=logc.device),
                          (logc - _D) / _C) - _B) / _A
    from_lin = (logc - _F) / _E
    linear = torch.where(logc >= _CUT_CODE, from_log, from_lin)
    return linear.clamp(0.0, CEILING_CLAMP)
