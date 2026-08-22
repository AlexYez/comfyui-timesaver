"""TS LTX HDR Decode — единственный финальный декодер графа.

Отсюда выходят две разные вещи, и путать их нельзя:

- ``preview_sdr`` — то, на что смотрят. Тон-маппинг, экспозиция, кривая sRGB.
- ``hdr_linear`` — то, что сохраняют. Сцена в линейном свете, без тон-маппинга,
  без гаммы, без зажима сверху. Ни одна настройка превью на него не влияет.

При выключенном HDR нода работает как обычный VAE-decode, а слот ``hdr_linear``
отдаёт ``ExecutionBlocker``: подключённый EXR-сохранятель не запустится и не
создаст файл-пустышку. Это лучше, чем отдать чёрный кадр — файла просто не будет.

⚠️ FP32-VAE берётся ленивым входом. При выключенном HDR он **не загружается**:
две копии одного видео-VAE в памяти стоят дорого, и платить за них в SDR-режиме
не за что.
"""

from __future__ import annotations

import logging

import torch
from comfy_api.v0_0_2 import IO

from ._hdr_types import (
    LOG_PREFIX,
    HdrImage,
    as_config,
    convert_working_to_master,
    describe,
    format_report,
    warnings_for,
)
from ._schema import CATEGORY, MISSING, HdrConfigIO, HdrImageIO
from ._tonemap import make_sdr_preview
from ._vae import decode_latent

logger = logging.getLogger("comfyui_timesaver.ts_ltx_hdr.decode")


def _blocker():
    """``ExecutionBlocker`` там, где он есть; иначе ошибка вместо пустышки."""
    try:
        from comfy_execution.graph_utils import ExecutionBlocker

        # Сообщение None — молчаливая блокировка: это не сбой, это выключенный
        # режим, и красная рамка на сохранятеле только сбивала бы с толку.
        return ExecutionBlocker(None)
    except Exception as error:                           # noqa: BLE001 - очень старое ядро
        raise RuntimeError(
            f"{LOG_PREFIX} This ComfyUI has no ExecutionBlocker, so the EXR saver "
            f"cannot be held back while HDR is off: {error}"
        ) from error


class TS_LTXHDRDecode(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LTXHDRDecode",
            display_name="TS LTX HDR Decode",
            category=CATEGORY,
            description=(
                "The single final decode. With HDR on it runs in float32 and turns the "
                "ACEScct working signal back into scene-linear light; with HDR off it "
                "is an ordinary decode and the HDR output is blocked."
            ),
            search_aliases=["hdr decode", "vae decode hdr", "exr decode", "final decode"],
            inputs=[
                IO.Latent.Input("samples", tooltip="The one latent chosen before decoding."),
                HdrConfigIO.Input("config", tooltip="Settings node."),
                IO.Vae.Input(
                    "sdr_vae",
                    optional=True,
                    lazy=True,
                    tooltip=(
                        "The VAE the rest of the graph already uses. Needed only when "
                        "HDR is off."
                    ),
                ),
                IO.Vae.Input(
                    "hdr_vae",
                    optional=True,
                    lazy=True,
                    tooltip=(
                        "The float32 VAE from TS LTX HDR VAE. Needed only when HDR is "
                        "on, and not loaded at all when it is off."
                    ),
                ),
                IO.Boolean.Input(
                    "tiled",
                    default=True,
                    tooltip="Decode in tiles. Keeps long clips inside VRAM.",
                ),
                IO.Int.Input("tile_size", default=512, min=64, max=4096, step=32, advanced=True,
                             tooltip="Tile side in pixels."),
                IO.Int.Input("overlap", default=64, min=0, max=4096, step=32, advanced=True,
                             tooltip="Overlap between tiles, to hide the seams."),
                IO.Int.Input("temporal_size", default=128, min=8, max=4096, step=4, advanced=True,
                             tooltip="How many frames are decoded at a time."),
                IO.Int.Input("temporal_overlap", default=32, min=4, max=4096, step=4, advanced=True,
                             tooltip="How many frames overlap between those groups."),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="preview_sdr",
                    tooltip="Tone-mapped for looking at. Send this to the video saver.",
                ),
                HdrImageIO.Output(
                    display_name="hdr_linear",
                    tooltip=(
                        "Scene-linear Rec.709 float32 master, highlights above 1.0 "
                        "intact. Blocked entirely while HDR is off."
                    ),
                ),
                IO.String.Output(display_name="info", tooltip="What came out, and what looks wrong."),
            ],
        )

    @classmethod
    def check_lazy_status(cls, samples=None, config=None, sdr_vae=MISSING, hdr_vae=MISSING,
                          tiled=True, tile_size=512, overlap=64,
                          temporal_size=128, temporal_overlap=32):
        if as_config(config).enabled:
            return ["hdr_vae"] if hdr_vae is None else []
        return ["sdr_vae"] if sdr_vae is None else []

    @classmethod
    def execute(cls, samples=None, config=None, sdr_vae=None, hdr_vae=None,
                tiled: bool = True, tile_size: int = 512, overlap: int = 64,
                temporal_size: int = 128, temporal_overlap: int = 32) -> IO.NodeOutput:
        settings = as_config(config)
        if samples is None:
            raise RuntimeError(f"{LOG_PREFIX} No latent connected to samples.")

        vae = hdr_vae if settings.enabled else sdr_vae
        if vae is None:
            which = "hdr_vae (from TS LTX HDR VAE)" if settings.enabled else "sdr_vae"
            raise RuntimeError(f"{LOG_PREFIX} HDR is {'on' if settings.enabled else 'off'}, "
                               f"so {which} must be connected.")

        with torch.no_grad():
            decoded = decode_latent(
                vae, samples,
                tiled=bool(tiled), tile_size=int(tile_size), overlap=int(overlap),
                temporal_size=int(temporal_size), temporal_overlap=int(temporal_overlap),
            )

        if not settings.enabled:
            logger.debug("%s SDR decode: %s", LOG_PREFIX, tuple(decoded.shape))
            return IO.NodeOutput(decoded, _blocker(), "HDR off — ordinary decode, no master written.")

        # Декодер отдал рабочий сигнал в [0, 1]. HDR появляется здесь — и кривая
        # зависит от того, какая это технология: ACEScct у нативного пути,
        # LogC3 у IC-LoRA. Перепутать их значит увести цвет.
        master = convert_working_to_master(decoded.to(torch.float32),
                                           settings.output_color_space,
                                           expand=settings.expands_sdr)
        preview = make_sdr_preview(
            master,
            exposure_ev=settings.preview_exposure_ev,
            operator=settings.preview_tonemap,
            output_dtype=settings.preview_torch_dtype,
        )

        stats = describe(master, color_space=settings.output_color_space,
                         ceiling=settings.working_ceiling)
        notes = warnings_for(stats, strict=settings.strict_validation)
        for note in notes:
            logger.warning("%s %s", LOG_PREFIX, note)
        if settings.strict_validation and stats["non_finite"]:
            raise RuntimeError(
                f"{LOG_PREFIX} The decode produced {stats['non_finite']} NaN/Inf samples; "
                "refusing to hand that on as a master. Turn strict_validation off to "
                "continue anyway."
            )

        image = HdrImage(master, settings.output_color_space,
                         {"stats": stats, "mode": settings.hdr_mode})
        logger.info("%s HDR decode (%s): %s, max %.4g, %.2f%% above 1.0", LOG_PREFIX,
                    "LogC3 IC-LoRA" if settings.expands_sdr else "ACEScct native",
                    tuple(master.shape), stats["max"], stats["above_one_share"])
        report = f"mode         : {settings.hdr_mode}\n" + format_report(stats, notes)
        return IO.NodeOutput(preview, image, report)


NODE_CLASS_MAPPINGS = {"TS_LTXHDRDecode": TS_LTXHDRDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LTXHDRDecode": "TS LTX HDR Decode"}
