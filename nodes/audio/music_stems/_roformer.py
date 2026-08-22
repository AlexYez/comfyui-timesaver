"""Инференс роформеров: планирование чанков, склейка, точный остаток.

Три решения здесь стоят того, чтобы их не «упростить» обратно.

**Последний чанк сдвигается назад, а не добивается тишиной.** У роформера
внимание идёт по всей длине сегмента, поэтому синтетическая тишина в хвосте —
это не нейтральный ноль, а вход, которого в обучении не было. Вместо padding
последний чанк начинается с ``frames - chunk``: он перекрывается с предыдущим
сильнее обычного, но состоит только из настоящего звука.

**Окно — приподнятый косинус на всю длину чанка**, а не линейный фейд
фиксированной длины. Края чанка — ровно те кадры, у которых меньше всего
контекста; косинус давит их плавно. У краёв самого трека окно выравнивается в
единицу: там соседа нет, и подавлять нечего.

**Один стем не берётся у модели, а считается как ``микс − остальные``.** Маски
не суммируются обратно в микс точно: расхождение слышно любым null-тестом.
Остаточный стем делает сумму точной по построению. Считать его можно почанково:
overlap-add — это взвешенное среднее, а вход у всех чанков в данном кадре один
и тот же, поэтому ``OLA(x − Σy) = x − Σ OLA(y)``.

Модуль знает только стандартную библиотеку и torch.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

logger = logging.getLogger("comfyui_timesaver.ts_music_stems.roformer")
LOG_PREFIX = "[TS Music Stems]"


# ── Точность ──────────────────────────────────────────────────────────────
#
# ⚠️ **bfloat16 для этих моделей НЕВОЗМОЖЕН**, и это не наша недоработка:
# маска собирается через ``torch.view_as_complex``, который принимает только
# half, float и double. bf16 падает на первом же чанке. Не «чинить» попыткой
# добавить его в список.
#
# ⚠️ float16 безопасен, и это ЗАМЕРЕНО, а не предположено. На реальном миксе
# ошибка fp16 против fp32 лежит на −61 дБFS в самом громком стеме и ниже во
# всех остальных — под шумовой полкой записи. Относительный SNR по стему для
# этого решения не годится: на дорожке без баса он показывает пугающие 2.2 дБ
# просто потому, что и стем, и ошибка там одинаковая тишина (−89 против
# −91 дБFS). Считать надо абсолютный уровень ошибки.
PRECISIONS = ("fp16", "fp32")

_DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def resolve_dtype(precision: str, device: torch.device) -> tuple[torch.dtype, bool]:
    """Выбрать тип вычислений. Возвращает ``(dtype, нужен ли autocast)``.

    Половинная точность включается только на CUDA: на CPU половина не быстрее,
    а часть операций там попросту не реализована.
    """
    if precision == "fp16" and device.type == "cuda":
        return torch.float16, True
    return torch.float32, False


@dataclass(frozen=True)
class Chunk:
    """Один отрезок исходника, который уедет в модель."""

    start: int
    valid: int
    zero_padded: bool


def plan_chunks(frames: int, chunk: int, overlap: float) -> list[Chunk]:
    """Разложить дорожку на чанки так, чтобы в них был только настоящий звук.

    Args:
        frames: длина дорожки в сэмплах.
        chunk: родная длина чанка модели.
        overlap: доля перекрытия, 0…0.5.

    Returns:
        Чанки по возрастанию начала. Единственный случай, когда в чанке
        появляется тишина, — дорожка короче одного чанка: добить там нечем.
    """
    if frames <= 0:
        raise ValueError(f"{LOG_PREFIX} В дорожке нет ни одного сэмпла.")
    if chunk <= 0:
        raise ValueError(f"{LOG_PREFIX} Длина чанка должна быть больше нуля.")
    if not 0.0 <= overlap <= 0.5:
        raise ValueError(f"{LOG_PREFIX} Перекрытие должно лежать в [0, 0.5], получено {overlap}.")

    if frames <= chunk:
        return [Chunk(0, frames, frames < chunk)]

    stride = max(1, chunk - int(round(chunk * overlap)))
    last_start = frames - chunk

    chunks: list[Chunk] = []
    start = 0
    while True:
        if start >= last_start:
            chunks.append(Chunk(last_start, chunk, False))
            return chunks
        chunks.append(Chunk(start, chunk, False))
        start += stride


def overlap_window(chunk: int, is_first: bool, is_last: bool,
                   *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Веса склейки для одного чанка.

    ``is_first`` / ``is_last`` — чанк упирается в начало или конец ТРЕКА, а не
    просто идёт первым в списке. Там таперить нечем: соседа, который подхватит
    затухание, не существует, и нормировка делила бы почти-нули на почти-ноль.
    """
    position = (torch.arange(chunk, device=device, dtype=dtype) + 0.5) / chunk
    window = 0.5 - 0.5 * torch.cos(2.0 * math.pi * position)
    half = chunk // 2
    if is_first:
        window[:half] = 1.0
    if is_last:
        window[half:] = 1.0
    # Строго положительный вес — иначе деление на накопленный вес взорвётся.
    return window.clamp_min(torch.finfo(dtype).eps)


def build_model(model, weights_path):
    """Собрать архитектуру по каталогу и залить в неё веса.

    ``strict=True`` намеренно: несовпадение ключей значит, что конфиг разошёлся
    с чекпойнтом, и такая модель не падает, а возвращает правдоподобный мусор.
    Лучше отказаться на загрузке.
    """
    from pathlib import Path

    if model.architecture == "bs_roformer":
        from .roformer_models.bs_roformer import BSRoformer as Architecture
    elif model.architecture == "mel_band_roformer":
        from .roformer_models.mel_band_roformer import MelBandRoformer as Architecture
    else:
        raise RuntimeError(f"{LOG_PREFIX} Неизвестная архитектура '{model.architecture}'.")

    net = Architecture(**dict(model.config)).eval()

    path = Path(weights_path)
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            state = state.get("state_dict", state)

    net.load_state_dict(state, strict=True)
    return net


def _as_stem_batch(output: torch.Tensor, num_stems: int) -> torch.Tensor:
    """Привести выход модели к ``(stems, channels, samples)``.

    Односте́мные модели возвращают ``(batch, ch, samples)``, многостемные —
    ``(batch, stems, ch, samples)``. Разбирать это в вызывающем коде значит
    повторять одну и ту же ошибку в двух местах.
    """
    if output.ndim == 4:
        return output[0]
    if output.ndim == 3:
        if num_stems != 1:
            raise RuntimeError(
                f"{LOG_PREFIX} Модель обещала {num_stems} стемов, а вернула один "
                f"тензор формы {tuple(output.shape)}."
            )
        return output[0].unsqueeze(0)
    raise RuntimeError(f"{LOG_PREFIX} Непонятная форма выхода модели: {tuple(output.shape)}.")


def separate(
    model,
    mix: torch.Tensor,
    *,
    num_stems: int,
    chunk: int,
    overlap: float = 0.25,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    autocast: bool = False,
    residual_index: int | None = None,
    on_chunk: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Разделить дорожку на стемы.

    Args:
        model: уже загруженная модель в ``eval()``.
        mix: ``(channels, samples)`` на CPU, float32.
        num_stems: сколько голов у модели.
        chunk: родная длина чанка.
        overlap: доля перекрытия соседних чанков.
        device: где считать.
        dtype: тип вычислений (``float32`` / ``float16`` / ``bfloat16``).
        autocast: считать под ``torch.autocast``. ⚠️ Для половинной точности это
            не «ускорение по желанию», а единственный способ вообще посчитать:
            модель делает STFT внутри себя, тот всегда отдаёт float32, и
            половинные веса встречают его несовпадением типов. autocast
            приводит типы пооперационно и решает это.
        residual_index: какой стем заменить на ``микс − остальные``. ``None`` —
            отдать всё как есть.
        on_chunk: колбэк ``(сделано, всего)`` — прогресс и отмена.

    Returns:
        ``(stems, channels, samples)`` на CPU, float32.
    """
    if mix.ndim != 2:
        raise ValueError(f"{LOG_PREFIX} Ожидался тензор (каналы, сэмплы), получен {tuple(mix.shape)}.")
    channels, frames = mix.shape
    chunks = plan_chunks(frames, chunk, overlap)

    # Аккумуляторы держим на устройстве расчёта в float32: складывать сотни
    # чанков в половинной точности — верный способ накопить шум там, где его
    # быть не должно.
    stems_sum = torch.zeros((num_stems, channels, frames), device=device, dtype=torch.float32)
    weight_sum = torch.zeros(frames, device=device, dtype=torch.float32)

    mix_device = mix.to(device=device, dtype=torch.float32)

    for index, piece in enumerate(chunks):
        part = mix_device[:, piece.start:piece.start + piece.valid]
        if piece.valid < chunk:
            part = torch.nn.functional.pad(part, (0, chunk - piece.valid))

        # Вход всегда float32: внутренний STFT в половинной точности не считает.
        with torch.no_grad():
            if autocast:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    raw = model(part.unsqueeze(0))
            else:
                raw = model(part.unsqueeze(0))
        stems = _as_stem_batch(raw, num_stems).to(torch.float32)

        if residual_index is not None:
            others = torch.arange(num_stems, device=stems.device) != residual_index
            stems[residual_index] = part - stems[others].sum(dim=0)

        window = overlap_window(
            chunk,
            is_first=piece.start == 0,
            is_last=piece.start + piece.valid >= frames,
            device=device,
        )[: piece.valid]

        end = piece.start + piece.valid
        stems_sum[..., piece.start:end] += stems[..., : piece.valid] * window
        weight_sum[piece.start:end] += window

        if on_chunk is not None:
            on_chunk(index + 1, len(chunks))

    result = stems_sum / weight_sum.clamp_min(torch.finfo(torch.float32).eps)
    return result.cpu()
