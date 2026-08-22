"""Старый путь ноды — Demucs.

⚠️ Этот модуль намеренно НЕ улучшается. Значения ``htdemucs``, ``htdemucs_ft`` и
``hdemucs_mmi`` лежат в сохранённых графах, и человек, открывший свой прошлогодний
workflow, обязан получить ровно тот же звук, что и раньше. Любая «заодно почищу»
правка здесь — это тихое изменение чужого результата.

Код перенесён из ``nodes/audio/ts_music_stems.py`` как есть: нормировка по
mean/std, порядок стемов Demucs (drums, bass, other, vocals), инструментал
суммой. Новая математика (остаточный стем, приподнятый косинус) живёт в
``_roformer.py`` и сюда не распространяется.
"""

from __future__ import annotations

import logging
import os
import threading

import torch

logger = logging.getLogger("comfyui_timesaver.ts_music_stems.demucs")
LOG_PREFIX = "[TS Music Stems]"

MODEL_NAMES = ("htdemucs", "htdemucs_ft", "hdemucs_mmi")

_model_cache: dict = {}


def prepare_waveform(waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Свести вход к стерео, как того требует Demucs."""
    original_channels = int(waveform.shape[1])
    if original_channels == 1:
        logger.info(
            "%s Mono input detected, duplicating channel to stereo for Demucs compatibility.",
            LOG_PREFIX,
        )
        return waveform.repeat(1, 2, 1), original_channels
    if original_channels == 2:
        return waveform, original_channels

    logger.info(
        "%s Input has %d channels, using the first two channels for Demucs.",
        LOG_PREFIX, original_channels,
    )
    return waveform[:, :2, :], original_channels


def start_ui_progress(pbar, total_steps, processing_start_step, processing_cap_step):
    """Пульсирующий индикатор.

    ⚠️ Честного прогресса тут нет и не будет: ``apply_model`` из Demucs не
    отдаёт наружу свой ход. У роформеров прогресс настоящий, по чанкам.
    """
    stop_event = threading.Event()

    def pulse():
        current_value = processing_start_step
        while not stop_event.wait(0.5):
            if current_value >= processing_cap_step:
                continue
            current_value += 1
            pbar.update_absolute(current_value, total=total_steps)

    worker = threading.Thread(target=pulse, name="ts-music-stems-progress", daemon=True)
    worker.start()
    return stop_event, worker


def load(model_name: str, models_dir: str):
    """Загрузить модель Demucs, кэшируя её между прогонами.

    ⚠️ Кэш — мутация словаря, объявленного в модуле, а не присваивание атрибута
    классу: V3 запрещает второе на залоченном классе (CLAUDE.md §5).
    """
    from ..._deps import TSDependencyManager

    pretrained = TSDependencyManager.import_optional("demucs.pretrained")
    get_model = getattr(pretrained, "get_model", None) if pretrained is not None else None
    if get_model is None:
        raise RuntimeError(
            "[TS Music Stems] Missing dependency 'demucs'. Install it to enable stem separation."
        )

    demucs_model_path = os.path.join(models_dir, "demucs")
    os.makedirs(demucs_model_path, exist_ok=True)

    original_hub_dir = torch.hub.get_dir()
    torch.hub.set_dir(demucs_model_path)
    try:
        if model_name not in _model_cache:
            model = get_model(model_name)
            _model_cache[model_name] = model
            submodels = getattr(model, "models", None)
            if submodels is not None and len(submodels) > 1:
                logger.info(
                    "%s Model '%s' is a Demucs bag model; the first run downloads %d checkpoints once.",
                    LOG_PREFIX, model_name, len(submodels),
                )
        else:
            model = _model_cache[model_name]
    except Exception as exc:
        raise RuntimeError(f"[TS Music Stems] Model load failed: {exc}") from exc
    finally:
        torch.hub.set_dir(original_hub_dir)
    return model


def apply(model, waveform: torch.Tensor, *, shifts: int, overlap: float,
          jobs: int, device: torch.device) -> torch.Tensor:
    """Прогнать Demucs. Возвращает ``(batch, stems, channels, samples)``."""
    from ..._deps import TSDependencyManager

    apply_module = TSDependencyManager.import_optional("demucs.apply")
    apply_model = getattr(apply_module, "apply_model", None) if apply_module is not None else None
    if apply_model is None:
        raise RuntimeError(
            "[TS Music Stems] Missing dependency 'demucs'. Install it to enable stem separation."
        )

    with torch.no_grad():
        return apply_model(
            model, waveform, shifts=shifts, split=True, overlap=overlap,
            progress=True, num_workers=jobs, device=device,
        )
