"""Общие типы HDR-ветки: настройки, помеченная картинка, статистика.

Здесь же — единственное место, где живут имена пользовательских типов ComfyUI.
Их нельзя менять: они уезжают в сохранённый workflow как тип сокета, и
переименование разорвёт связи в чужих графах (§4 CLAUDE.md).

⚠️ **Картинка ходит с ярлыком, а не голым тензором.** Линейный Rec.709,
линейный ACEScg и уже закодированный ACEScct — это три разных числовых мира,
внешне неотличимых: во всех трёх тензор float32 формы ``[B, H, W, 3]``.
Перепутать их означает получить сдвиг цвета, который заметят только в
монтажной. Поэтому :class:`HdrImage` несёт пространство с собой, а ноды
сверяют ярлык вместо того, чтобы верить человеку на слово.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch

from ._acescct import WORKING_CEILING, acescct_to_linear_ap1, linear_ap1_to_acescct
from ._logc3 import WORKING_CEILING as LOGC3_CEILING
from ._logc3 import logc3_to_linear
from ._primaries import acescg_to_rec709, rec709_to_acescg

LOG_PREFIX = "[TS LTX HDR]"

# ⚠️ Публичный контракт: эти строки становятся типами сокетов в workflow.
HDR_CONFIG_TYPE = "TS_LTX_HDR_CONFIG"
HDR_IMAGE_TYPE = "TS_LTX_HDR_IMAGE"

# Чем объявлен вход. Совпадает с ключами официального CLI (--hdr ...), чтобы
# человек, читавший документацию LTX, не гадал.
INPUT_SPACES = ("ACESCG", "SRGB_LINEAR", "ACESCCT")

# Что отдаёт финальный decode. В v1 один вариант: у встроенного сохранятеля
# EXR примарии Rec.709 зашиты, и отдать ему ACEScg значило бы записать неверные
# хроматичности (§10.2 плана).
OUTPUT_SPACES = ("REC709_LINEAR",)

PREVIEW_DTYPES = MappingProxyType({"FP16": torch.float16, "FP32": torch.float32})

# ⚠️ ДВЕ РАЗНЫЕ ТЕХНОЛОГИИ, а не два оттенка одной.
#
# «preserve» — нативный HDR LTX 2.5: диапазон ПРИШЁЛ из EXR, наша задача его не
# потерять. Рабочее пространство ACEScct, на выходе меняются примарии AP1 →
# Rec.709.
#
# «expand» — HDR IC-LoRA: диапазона на входе НЕ БЫЛО, модель его выращивает из
# обычного SDR. Рабочее пространство LogC3, примарии НЕ меняются — модель уже
# отдаёт нужные. Официально LoRA проверена на LTX 2.3; для 2.5 поддержка
# заявлена как «в разработке».
#
# Строки уезжают в сохранённый workflow — менять нельзя.
MODE_PRESERVE = "preserve HDR from EXR (ACEScct)"
MODE_EXPAND = "expand HDR from SDR (LogC3 IC-LoRA)"
HDR_MODES = (MODE_PRESERVE, MODE_EXPAND)


@dataclass(frozen=True)
class HdrConfig:
    """Один объект настроек на всю HDR-ветку.

    Заморожен намеренно: ComfyUI кэширует результаты нод по входам, и
    изменяемый конфиг дал бы «поменял галочку — ничего не пересчиталось».
    """

    enabled: bool = False
    input_color_space: str = "ACESCG"
    output_color_space: str = "REC709_LINEAR"
    strict_validation: bool = True
    preview_exposure_ev: float = 0.0
    preview_tonemap: str = "reinhard_luma"
    preview_dtype: str = "FP16"
    # Умолчание — нативный путь: так старый workflow ведёт себя как прежде.
    hdr_mode: str = MODE_PRESERVE

    @property
    def preview_torch_dtype(self) -> torch.dtype:
        return PREVIEW_DTYPES.get(self.preview_dtype, torch.float16)

    @property
    def expands_sdr(self) -> bool:
        """Режим IC-LoRA: диапазон выращивается из SDR, а не сохраняется из EXR."""
        return self.hdr_mode == MODE_EXPAND

    @property
    def working_ceiling(self) -> float:
        """Линейная яркость, соответствующая коду 1.0 в рабочей кривой режима."""
        return LOGC3_CEILING if self.expands_sdr else WORKING_CEILING

    def __str__(self) -> str:
        if not self.enabled:
            return "Native HDR: off"
        return (f"HDR: on, mode {self.hdr_mode!r}, input {self.input_color_space}, "
                f"output {self.output_color_space}")


DEFAULT_CONFIG = HdrConfig()


def as_config(value) -> HdrConfig:
    """Привести вход к :class:`HdrConfig`.

    Ноды HDR-ветки работают и без подключённой ноды настроек — тогда действует
    умолчание «HDR выключен». Так граф не падает, если человек ещё не дотянул
    провод.
    """
    if isinstance(value, HdrConfig):
        return value
    return DEFAULT_CONFIG


@dataclass(frozen=True)
class HdrImage:
    """Кадр с ярлыком цветового пространства.

    Attributes:
        tensor: ``[B, H, W, 3]`` float32. Для линейных пространств значения
            могут быть больше 1; для ``ACESCCT`` лежат в ``[0, 1]``.
        color_space: одно из :data:`INPUT_SPACES` или ``REC709_LINEAR``.
        meta: откуда пришло и что при этом было замечено.
    """

    tensor: torch.Tensor
    color_space: str
    meta: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError(f"{LOG_PREFIX} HdrImage needs a tensor, got {type(self.tensor)!r}.")
        if self.tensor.ndim != 4 or self.tensor.shape[-1] != 3:
            raise ValueError(
                f"{LOG_PREFIX} HdrImage expects [B, H, W, 3], got {tuple(self.tensor.shape)}.")

    @property
    def size(self) -> tuple[int, int]:
        """Ширина и высота, в том порядке, в каком их спрашивают люди."""
        return int(self.tensor.shape[2]), int(self.tensor.shape[1])

    def to_linear_ap1(self) -> torch.Tensor:
        """Привести к линейному ACEScg (AP1) — общему знаменателю всей ветки."""
        value = self.tensor.to(torch.float32)
        if self.color_space == "ACESCG":
            return value
        if self.color_space in ("SRGB_LINEAR", "REC709_LINEAR"):
            return rec709_to_acescg(value)
        if self.color_space == "ACESCCT":
            return acescct_to_linear_ap1(value)
        raise ValueError(f"{LOG_PREFIX} Unknown colour space '{self.color_space}'.")


def to_working_image(hdr: HdrImage) -> torch.Tensor:
    """Кадр → рабочий сигнал ACEScct в ``[0, 1]``, готовый для VAE.

    ⚠️ Если вход **уже** объявлен как ``ACESCCT``, повторное лог-кодирование
    было бы грубой ошибкой: сигнал сжался бы дважды. Такой вход только
    зажимается в рабочий диапазон и идёт дальше как есть.
    """
    if hdr.color_space == "ACESCCT":
        return hdr.tensor.to(torch.float32).clamp(0.0, 1.0)
    return linear_ap1_to_acescct(hdr.to_linear_ap1())


def working_to_scene_linear(
    working: torch.Tensor,
    output_space: str = "REC709_LINEAR",
    *,
    expand: bool = False,
) -> torch.Tensor:
    """Рабочий сигнал → сцена в линейном свете.

    Зажимает только снизу: сверху там и находится HDR.

    Args:
        working: выход VAE в ``[0, 1]``.
        output_space: целевые примарии — только для пути ACEScct.
        expand: путь IC-LoRA. Тогда кривая LogC3, и ⚠️ **примарии не меняются**:
            официальная нода отдаёт линейный HDR как есть, матрица ACES-пути
            увела бы цвет.
    """
    if expand:
        return logc3_to_linear(working)
    linear_ap1 = acescct_to_linear_ap1(working)
    if output_space == "ACESCG":
        return linear_ap1
    if output_space != "REC709_LINEAR":
        raise ValueError(f"{LOG_PREFIX} Unknown output colour space '{output_space}'.")
    return acescg_to_rec709(linear_ap1).clamp_min(0.0)


# Кусок для поканальной обработки видео. Полный ролик 129×1920×1088 в float32 —
# 3 ГиБ; каждое промежуточное преобразование целиком удвоило бы это число.
_CONVERT_CHUNK = 4


def convert_working_to_master(
    working: torch.Tensor,
    output_space: str = "REC709_LINEAR",
    *,
    expand: bool = False,
    chunk: int = _CONVERT_CHUNK,
) -> torch.Tensor:
    """То же, что :func:`working_to_scene_linear`, но кусками по кадрам.

    Результат пишется в заранее выделенный тензор, поэтому сверх самого мастера
    в памяти живёт лишь один кусок, а не вторая копия всего ролика.
    """
    master = torch.empty(working.shape, dtype=torch.float32, device=working.device)
    for start in range(0, int(working.shape[0]), max(1, chunk)):
        stop = min(int(working.shape[0]), start + max(1, chunk))
        master[start:stop] = working_to_scene_linear(
            working[start:stop], output_space, expand=expand)
    return master


# ─────────────────────────────── статистика ────────────────────────────

# Считаем по кадрам: у 129 кадров 1920×1088 один тензор float32 весит 3 ГиБ, и
# копия ради сортировки перцентилей удвоила бы цифру.
_STATS_CHUNK = 4
_PERCENTILES = (0.5, 0.9, 0.99, 0.999)


def describe(tensor: torch.Tensor, *, color_space: str = "REC709_LINEAR",
             ceiling: float = WORKING_CEILING) -> dict:
    """Собрать статистику HDR-тензора, не копируя его целиком."""
    total = int(tensor.numel())
    frames = int(tensor.shape[0]) if tensor.ndim == 4 else 1
    low, high, running_sum = math.inf, -math.inf, 0.0
    non_finite = negatives = above_one = 0
    samples = []

    for start in range(0, frames, _STATS_CHUNK):
        chunk = tensor[start:start + _STATS_CHUNK].to(torch.float32)
        finite_mask = torch.isfinite(chunk)
        non_finite += int((~finite_mask).sum().item())
        finite = chunk[finite_mask]
        if not finite.numel():
            continue
        low = min(low, float(finite.min().item()))
        high = max(high, float(finite.max().item()))
        running_sum += float(finite.sum().item())
        negatives += int((finite < 0.0).sum().item())
        above_one += int((finite > 1.0).sum().item())
        # Для перцентилей хватает разрежённой выборки: точность до третьего
        # знака здесь не нужна, а полная сортировка трёх гигабайт — нужна ещё
        # меньше.
        flat = finite.flatten()
        step = max(1, flat.numel() // 20000)
        samples.append(flat[::step].cpu())

    if low is math.inf:
        low = high = 0.0
    pool = torch.cat(samples) if samples else torch.zeros(1)
    quantiles = torch.quantile(pool, torch.tensor(_PERCENTILES)) if pool.numel() else None

    stops = 0.0
    if high > 0.0:
        floor = max(low, 1e-4) if low > 0.0 else 1e-4
        stops = math.log2(high / floor)

    return {
        "color_space": color_space,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": tuple(int(s) for s in tensor.shape),
        "min": low,
        "max": high,
        "mean": running_sum / max(1, total - non_finite),
        "percentiles": {
            f"p{int(p * 1000) / 10:g}": float(quantiles[i])
            for i, p in enumerate(_PERCENTILES)
        } if quantiles is not None else {},
        "above_one_share": 100.0 * above_one / max(1, total),
        "negatives": negatives,
        "non_finite": non_finite,
        "dynamic_range_stops": stops,
        "working_ceiling": float(ceiling),
    }


def warnings_for(stats: dict, *, strict: bool = True) -> list[str]:
    """Что в этой статистике выглядит как потерянный HDR."""
    notes: list[str] = []
    if stats["non_finite"]:
        notes.append(f"{stats['non_finite']} NaN/Inf samples — the decode went wrong.")
    if stats["max"] <= 1.0:
        notes.append(
            "Nothing above 1.0: either the scene really is that dim, or the highlights "
            "were clipped somewhere upstream.")
    if stats["negatives"] and stats["color_space"] == "REC709_LINEAR":
        notes.append(
            f"{stats['negatives']} negative samples after the final conversion — "
            "they should have been clamped.")
    if stats["dtype"] != "float32":
        notes.append(f"Master is {stats['dtype']}, not float32.")
    if stats["max"] >= stats["working_ceiling"] * 0.98:
        notes.append(
            f"Highlights reach the ACEScct working ceiling ({stats['working_ceiling']:.0f}); "
            "anything brighter was flattened on the way into the model.")
    if strict and stats["non_finite"]:
        notes.append("Strict validation is on: fix the NaN/Inf before saving a master.")
    return notes


def format_report(stats: dict, notes: list[str]) -> str:
    """Человекочитаемый отчёт — то, что нода показывает и отдаёт строкой."""
    percentiles = "  ".join(f"{k} {v:.4g}" for k, v in stats["percentiles"].items())
    lines = [
        f"colour space : {stats['color_space']}",
        f"tensor       : {stats['dtype']} {stats['shape']}",
        f"range        : min {stats['min']:.4g}   max {stats['max']:.4g}   "
        f"mean {stats['mean']:.4g}",
        f"percentiles  : {percentiles}" if percentiles else "percentiles  : n/a",
        f"above 1.0    : {stats['above_one_share']:.2f}% of samples",
        f"dynamic range: {stats['dynamic_range_stops']:.1f} stops",
        f"negatives    : {stats['negatives']}    NaN/Inf: {stats['non_finite']}",
    ]
    if notes:
        lines.append("")
        lines.extend(f"! {note}" for note in notes)
    return "\n".join(lines)
