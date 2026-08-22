"""Что за модели умеет нода и где лежат их веса.

Каталог — данные, а не код: контракт модели (сколько стемов, в каком порядке,
какой чанк, какой STFT) описан один раз здесь, и цикл инференса читает его,
вместо того чтобы знать про конкретные модели.

⚠️ **Порядок стемов — не украшение.** У BS-RoFormer SW головы идут
``bass, drums, other, vocals, guitar, piano`` — не в том порядке, в котором
модель описана в интернете. Перепутанный порядок не падает, а возвращает
уверенную чушь: барабаны в выходе баса. Порядок взят из ``training.instruments``
самого конфига модели и проверен тестом против опубликованного чекпойнта.

Модуль знает только стандартную библиотеку, torch и folder_paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger("comfyui_timesaver.ts_music_stems.catalog")
LOG_PREFIX = "[TS Music Stems]"

# Куда пак кладёт свои роформеры. Регистрируется в folder_paths, поэтому
# extra_model_paths.yaml тоже работает.
MODEL_FOLDER_NAME = "roformer"

SAMPLE_RATE = 44100


@dataclass(frozen=True)
class RoformerModel:
    """Контракт одной модели: чем её строить, чем кормить, что она вернёт."""

    key: str
    """Значение виджета ``model_name``. ⚠️ Часть контракта ноды — не менять."""

    display: str
    architecture: str
    """``bs_roformer`` или ``mel_band_roformer``."""

    stems: tuple[str, ...]
    """Имена голов В ТОМ ПОРЯДКЕ, В КОТОРОМ ИХ ВЫДАЁТ МОДЕЛЬ."""

    chunk_samples: int
    """Родная длина чанка. Из ``audio.chunk_size`` конфига модели."""

    config: Mapping[str, Any]
    """Аргументы конструктора архитектуры."""

    repo_id: str
    weight_files: tuple[str, ...]
    """Имена файлов у автора. Первый — веса, остальные попутные."""

    licence: str
    notes: str = ""
    extra_dirs: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """Где ещё поискать веса: (папка folder_paths, имя файла)."""

    @property
    def num_stems(self) -> int:
        return len(self.stems)


# ── BS-RoFormer SW ────────────────────────────────────────────────────────
#
# Конфиг перенесён из BS-Rofo-SW-Fixed.yaml дословно. Полосы заданы явным
# списком, потому что так их задаёт автор: сумма обязана дать 1025 бинов, и
# это проверяется тестом — разойдись она на единицу, модель построится и будет
# читать спектр не там.
_SW_BANDS = (
    (2,) * 24
    + (4,) * 12
    + (12,) * 8
    + (24,) * 8
    + (48,) * 8
    + (128, 129)
)

BS_ROFORMER_SW = RoformerModel(
    key="bs_roformer_sw",
    display="BS-RoFormer SW (6 stems)",
    architecture="bs_roformer",
    # ⚠️ Из training.instruments конфига. Не «vocals, drums, bass…».
    stems=("bass", "drums", "other", "vocals", "guitar", "piano"),
    chunk_samples=588800,
    config={
        "dim": 256,
        "depth": 12,
        "stereo": True,
        "num_stems": 6,
        "time_transformer_depth": 1,
        "freq_transformer_depth": 1,
        "linear_transformer_depth": 0,
        "freqs_per_bands": _SW_BANDS,
        "dim_head": 64,
        "heads": 8,
        "attn_dropout": 0.0,
        "ff_dropout": 0.0,
        "flash_attn": True,
        "dim_freqs_in": 1025,
        "stft_n_fft": 2048,
        "stft_hop_length": 512,
        "stft_win_length": 2048,
        "stft_normalized": False,
        "mask_estimator_depth": 2,
        "multi_stft_resolution_loss_weight": 1.0,
        "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
        "multi_stft_hop_size": 147,
        "multi_stft_normalized": False,
        "mlp_expansion_factor": 4,
        "use_torch_checkpoint": False,
        "skip_connection": False,
    },
    repo_id="enerjazzer/BS-ROFO-SW-Fixed",
    weight_files=("BS-Rofo-SW-Fixed.ckpt",),
    # ⚠️ Автор лицензию не объявил. Поэтому пак ВЕСА НЕ ВОЗИТ и не зеркалит —
    # только скачивает у первоисточника по требованию, оставляя решение
    # человеку. Карточка репозитория лицензией не является.
    licence="none declared by the author — fetched from the original source",
    notes="Guitar and piano are the weakest of the six heads.",
)


# ── Mel-Band RoFormer, вокал ──────────────────────────────────────────────
#
# Модель выдаёт ОДИН стем. Инструментал получается остатком, и именно поэтому
# он вычитается точно, а не собирается суммой (см. _roformer.py).
MELBAND_VOCALS = RoformerModel(
    key="melband_roformer",
    display="Mel-Band RoFormer (vocals + instrumental)",
    architecture="mel_band_roformer",
    stems=("vocals",),
    chunk_samples=352800,
    config={
        "dim": 384,
        "depth": 6,
        "stereo": True,
        "num_stems": 1,
        "time_transformer_depth": 1,
        "freq_transformer_depth": 1,
        "num_bands": 60,
        "dim_head": 64,
        "heads": 8,
        "attn_dropout": 0.0,
        "ff_dropout": 0.0,
        "flash_attn": True,
        "dim_freqs_in": 1025,
        "sample_rate": SAMPLE_RATE,
        "stft_n_fft": 2048,
        "stft_hop_length": 441,
        "stft_win_length": 2048,
        "stft_normalized": False,
        "mask_estimator_depth": 2,
    },
    repo_id="Kijai/MelBandRoFormer_comfy",
    weight_files=("MelBandRoformer_fp32.safetensors",),
    licence="MIT",
    notes="A specialist: all its capacity on one boundary, which is why its vocals are better.",
    # У многих эта модель уже лежит от ComfyUI-MelBandRoFormer — незачем
    # качать 912 МБ второй раз.
    extra_dirs=(("diffusion_models", "melbandroformer/MelBandRoformer_fp32.safetensors"),),
)


MODELS: tuple[RoformerModel, ...] = (BS_ROFORMER_SW, MELBAND_VOCALS)
BY_KEY: Mapping[str, RoformerModel] = {model.key: model for model in MODELS}


def is_roformer(model_name: str) -> bool:
    return model_name in BY_KEY


def register_model_folder() -> Path:
    """Объявить ``models/roformer`` в folder_paths и вернуть путь."""
    import folder_paths

    base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    if hasattr(folder_paths, "add_model_folder_path"):
        folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(base))
    return base


def _existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def locate_weights(model: RoformerModel, *, download: bool = True) -> Path:
    """Найти веса, а если их нет — скачать у первоисточника.

    Порядок поиска: своя папка ``models/roformer`` → места, где файл мог
    остаться от других паков → загрузка. Смысл среднего шага прозаичный: у
    MelBand это 912 МБ, которые у многих уже на диске.
    """
    import folder_paths

    base = register_model_folder()
    file_name = model.weight_files[0]

    # ⚠️ Половинных копий на диске НЕТ и не заводим: fp16 получается приведением
    # при загрузке. Один файл на диске — один источник истины, совпадающий с
    # тем, что опубликовал автор.
    candidates = [base / file_name]

    for folder, relative in model.extra_dirs:
        try:
            for root in folder_paths.get_folder_paths(folder):
                candidates.append(Path(root) / relative)
        except (KeyError, AttributeError):
            continue

    found = _existing(candidates)
    if found is not None:
        logger.info("%s Weights found: %s", LOG_PREFIX, found.name)
        return found

    if not download:
        raise RuntimeError(
            f"{LOG_PREFIX} Веса '{file_name}' не найдены. Положите их в "
            f"models/{MODEL_FOLDER_NAME}/ или разрешите загрузку."
        )

    from ..._hf_download import snapshot_download_resilient

    logger.info(
        "%s First run of '%s': downloading the weights from %s. Licence: %s",
        LOG_PREFIX, model.display, model.repo_id, model.licence,
    )
    snapshot_download_resilient(
        repo_id=model.repo_id,
        local_dir=str(base),
        allow_patterns=list(model.weight_files),
        log=logger,
        log_prefix=LOG_PREFIX,
    )
    downloaded = _existing([base / file_name])
    if downloaded is None:
        raise RuntimeError(
            f"{LOG_PREFIX} Загрузка прошла, но '{file_name}' на месте не оказалось."
        )
    return downloaded
