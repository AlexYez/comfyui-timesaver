"""Загрузка VAE с явной точностью и один общий путь декодирования.

Два факта из исходников ComfyUI, на которых всё держится (проверено в
``comfy/sd.py`` установленной сборки):

1. ``comfy.sd.VAE`` принимает ``dtype``; штатный ``VAELoader`` его не передаёт и
   позволяет model management выбрать bf16. Для финального HDR-decode нужен
   float32 — и у LTX 2.5 VAE ``working_dtypes`` его содержит, так что это
   законный режим, а не насилие над моделью.

2. У ветки LTX 2.4/2.5 (``decoder.conv_in_x_t.weight`` в state dict)
   ``process_output`` не переопределён, то есть работает штатное
   ``(x + 1) / 2`` с зажимом в ``[0, 1]``. Для нас это **не потеря**: рабочий
   сигнал ACEScct по определению живёт в ``[0, 1]``, и официальный путь LTX
   зажимает его там же. HDR появляется на обратном преобразовании, уже после
   декодера.

⚠️ Точность существующего загруженного VAE в runtime не подменяем: у объекта
есть ModelPatcher и политика выгрузки, и подмена ``vae_dtype`` на живом объекте
конфликтует с ними. Второй экземпляр честнее.
"""

from __future__ import annotations

import logging

import torch

from ._hdr_types import LOG_PREFIX

logger = logging.getLogger("comfyui_timesaver.ts_ltx_hdr.vae")


def vae_names() -> list[str]:
    """Список файлов VAE — тот же, что видит штатный загрузчик."""
    try:
        import folder_paths

        return list(folder_paths.get_filename_list("vae"))
    except Exception:                                    # noqa: BLE001 - вне ComfyUI
        return []


def load_vae(name: str, dtype: torch.dtype = torch.float32):
    """Загрузить VAE из ``models/vae`` с явно заданной точностью."""
    try:
        import comfy.sd
        import comfy.utils
        import folder_paths
    except Exception as error:                           # noqa: BLE001
        raise RuntimeError(f"{LOG_PREFIX} ComfyUI is required to load a VAE: {error}") from error

    path = folder_paths.get_full_path_or_raise("vae", str(name))
    state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=state_dict, dtype=dtype, metadata=metadata)
    vae.throw_exception_if_invalid()
    logger.info("%s loaded %s at %s", LOG_PREFIX, name, str(dtype).replace("torch.", ""))
    return vae


def decode_latent(
    vae,
    samples,
    *,
    tiled: bool = True,
    tile_size: int = 512,
    overlap: int = 64,
    temporal_size: int = 128,
    temporal_overlap: int = 32,
):
    """Декодировать латент в кадры ``[F, H, W, C]``.

    Повторяет арифметику ядровой ``VAEDecodeTiled`` — включая деление тайлов на
    коэффициент сжатия и пересчёт временного окна, — чтобы плиточный режим вёл
    себя ровно так же, как привычная нода.
    """
    latent = samples["samples"]
    if getattr(latent, "is_nested", False):
        latent = latent.unbind()[0]

    if not tiled:
        images = vae.decode(latent)
    else:
        if tile_size < overlap * 4:
            overlap = tile_size // 4
        if temporal_size < temporal_overlap * 2:
            temporal_overlap = temporal_overlap // 2
        compression_t = vae.temporal_compression_decode()
        if compression_t is not None:
            temporal_size = max(2, temporal_size // compression_t)
            temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // compression_t))
        else:
            temporal_size = None
            temporal_overlap = None
        compression = vae.spacial_compression_decode()
        images = vae.decode_tiled(
            latent,
            tile_x=tile_size // compression,
            tile_y=tile_size // compression,
            overlap=overlap // compression,
            tile_t=temporal_size,
            overlap_t=temporal_overlap,
        )

    if len(images.shape) == 5:                           # склеиваем батчи, как ядро
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images
