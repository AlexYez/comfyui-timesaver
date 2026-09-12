"""TS Latent Upscale — re-sample a MiniMax H3 AV latent at a larger size, in one node.

Three nodes from Comfyui-MMH3-UltimateUpscale (MIT, bbaudio-2025) folded into
one: the pipeline itself, the model-based latent upscale settings, and the
temporal split settings. In the original those settings travel as wires between
`MMH3 Latent Upscale with Model Params`, `MMH3 Temporal Split Params` and
`MMH3 Ultimate Upscale`; here they are simply inputs, because nobody ever wanted
one without the others.

**Spatial tiling is deliberately absent.** The original also splits each chunk
into tiles and stitches them back. It was dropped on purpose along with its
input: the seams need their own fade and blend settings, and a clip that needs
tiling is better served by shorter chunks.

What the node does, per chunk of the clip:

1. cut the latent into overlapping chunks along time;
2. upscale that chunk's video latent with the H3 3D upscaler (audio untouched);
3. re-anchor the conditioning in time and pin frame 0 to the previous chunk's
   re-sampled frame, so the seam does not drift;
4. sample the chunk;
5. stitch it back with a cross-fade over the overlap.

Peak VRAM is therefore one chunk, not one clip — and the diffusion model is
offloaded while the upscaler works, since the two are never needed at once.

node_id: TS_LatentUpscale
"""

from __future__ import annotations

import logging

from comfy_api.v0_0_2 import IO

from . import _h3_core as core

logger = logging.getLogger("comfyui_timesaver.ts_latent_upscale")
LOG_PREFIX = "[TS Latent Upscale]"

#: Keyframe grid of the H3 model: chunk length and overlap have to land on it.
_FRAME_GRID = 17

#: Второй путь апскейла — без модели вовсе, обычной интерполяцией. Он есть у
#: автора отдельной нодой (`MMH3 Latent Upscale Params`); здесь это пункт того
#: же списка, потому что выбор «чем увеличить» один, а не два.
_INTERPOLATION_CHOICES = {
    "Interpolation: bilinear (no model)": "bilinear",
    "Interpolation: bicubic (no model)": "bicubic",
    "Interpolation: area (no model)": "area",
    "Interpolation: nearest (no model)": "nearest-exact",
}


def _upscale_options():
    """Модели из папки плюс безмодельные способы — одним списком."""
    return list(_INTERPOLATION_CHOICES) + core._scan_models()


def _upscale_param(choice, width, height, device, precision):
    """Настройки апскейла для ядра: по модели или по интерполяции."""
    width = int(round(int(width) / 32.0)) * 32
    height = int(round(int(height) / 32.0)) * 32
    if choice in _INTERPOLATION_CHOICES:
        return {"method": _INTERPOLATION_CHOICES[choice], "width": width, "height": height}
    return {
        "model_name": choice, "width": width, "height": height,
        "device": device, "precision": precision,
    }


class TS_LatentUpscale(IO.ComfyNode):
    """Chunked latent upscale + re-sample for MiniMax H3 AV latents."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_LatentUpscale",
            display_name="TS Latent Upscale",
            category="TS/Video",
            description=(
                "Re-sample an already-denoised MiniMax H3 audio+video latent at a larger "
                "size: the clip is cut into overlapping chunks along time, each chunk is "
                "upscaled by the H3 3D latent upscaler and re-sampled, then stitched back "
                "with the seam pinned to the previous chunk. Peak VRAM is one chunk rather "
                "than the whole clip. Upscale models are read from models/latent_upscale_models, "
                "including subfolders."
            ),
            search_aliases=[
                "latent upscale", "h3 upscale", "ultimate upscale", "minimax upscale",
                "video latent upscale",
            ],
            inputs=[
                IO.Model.Input(
                    "model",
                    tooltip="Diffusion model used to re-sample every chunk (the guider is built internally).",
                ),
                IO.Conditioning.Input(
                    "conditioning",
                    tooltip=(
                        "Conditioning this latent was generated with. It is re-anchored in time "
                        "per chunk, and its frame-0 keyframe is pinned to the previous chunk's "
                        "re-sampled frame."
                    ),
                ),
                IO.Latent.Input("latent", tooltip="Denoised MiniMax H3 audio+video latent to enhance."),
                IO.Noise.Input("noise", tooltip="Noise source; one noise tensor per chunk."),
                IO.Sampler.Input("sampler", tooltip="Sampler used for every chunk."),
                IO.Sigmas.Input("sigmas", tooltip="Sigma schedule used for every chunk."),
                IO.Combo.Input(
                    "upscale_model",
                    options=_upscale_options(),
                    tooltip=(
                        "How to enlarge each chunk. The H3 latent upscale checkpoints come from "
                        "models/latent_upscale_models, subfolders included and listed as "
                        "'subfolder/file.safetensors' — these are the minimax_h3_latent_upscaler_3d "
                        "weights, and an upscaler for another model family will not load here. "
                        "The 'Interpolation' entries need no model at all: quicker and lighter, "
                        "but they invent no detail."
                    ),
                ),
                IO.Int.Input(
                    "width", default=1280, min=64, max=4096, step=32,
                    tooltip=(
                        "Target frame width in pixels, snapped to a multiple of 32 (the "
                        "upscaler's grid). Must match the size the conditioning was made for."
                    ),
                ),
                IO.Int.Input(
                    "height", default=704, min=64, max=4096, step=32,
                    tooltip=(
                        "Target frame height in pixels, snapped to a multiple of 32. Must match "
                        "the size the conditioning was made for."
                    ),
                ),
                IO.Int.Input(
                    "chunk_length", default=136, min=17, max=100000, step=17,
                    tooltip=(
                        "Pixel frames per chunk at 24 fps, and a multiple of 17 — one keyframe "
                        "grid step. 136 ≈ 5.7 s, 153 ≈ 6.4 s. Aim for the longest chunk that "
                        "keeps peak VRAM just under capacity: the original author's starting "
                        "points are 34–68 frames on 8 GB, 51–102 on 12 GB, 102–153 on 16 GB and "
                        "136–170 on 24 GB. Shorter chunks are safer but pay the overlap tax on "
                        "every seam."
                    ),
                ),
                IO.Int.Input(
                    "temporal_overlap", default=17, min=0, max=100000, step=17,
                    tooltip=(
                        "Frames shared between consecutive chunks, a multiple of 17 and smaller "
                        "than chunk_length. This is the material the seam cross-fades over; 17 "
                        "is the usual choice."
                    ),
                ),
                IO.Float.Input(
                    "anchor_strength", default=0.999, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How firmly each chunk's first frame is held to the previous chunk's "
                        "result. 1.0 = exactly that frame, 0.999 = the model's own default, "
                        "0.0 = no anchoring at all (expect the seam to drift)."
                    ),
                ),
                IO.Combo.Input(
                    "precision", options=["fp16", "fp32", "bf16"], default="fp16",
                    tooltip=(
                        "Precision the upscaler runs at. fp16 is the usual choice — measured "
                        "against fp32 on the H3 checkpoint it is also the more accurate of the "
                        "two half-precision paths (0.38% vs 2.67% deviation), because bf16 trades "
                        "mantissa bits for a range these weights never use. On a card without "
                        "native bfloat16 (Turing and older) bf16 falls back to fp16 automatically."
                    ),
                ),
                IO.Combo.Input(
                    "device", options=["cuda", "cpu"], default="cuda",
                    tooltip="Where the upscaler runs. On cuda the diffusion model is offloaded first, so the two never sit on the card together.",
                ),
                IO.Conditioning.Input(
                    "negative", optional=True,
                    tooltip="Negative conditioning. Connected, sampling uses a CFG guider with the value below; otherwise the positive prompt alone.",
                ),
                IO.Float.Input(
                    "cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01, optional=True,
                    tooltip="CFG scale, used only when 'negative' is connected.",
                ),
            ],
            outputs=[
                IO.Latent.Output("latent", tooltip="Upscaled, re-sampled and stitched H3 audio+video latent."),
                IO.Custom("DICT").Output(
                    "segments_info",
                    tooltip="Debug: per-chunk frame ranges, token ranges and the size each chunk was upscaled to.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, chunk_length=136, temporal_overlap=17, upscale_model="",
                        **_kwargs) -> bool | str:
        # ⚠️ Проверяем ЗДЕСЬ, а не в `execute`: сетка в 17 кадров — требование
        # модели, и узнать о ней стоит до того, как посчитается половина клипа.
        if int(chunk_length) % _FRAME_GRID != 0:
            return (f"chunk_length must be a multiple of {_FRAME_GRID} "
                    f"(the model's keyframe grid step); got {chunk_length}.")
        if int(temporal_overlap) % _FRAME_GRID != 0:
            return (f"temporal_overlap must be a multiple of {_FRAME_GRID}; "
                    f"got {temporal_overlap}.")
        if int(temporal_overlap) >= int(chunk_length):
            return "temporal_overlap must be smaller than chunk_length."
        if str(upscale_model).startswith("(no upscale models found"):
            return ("No latent upscale model found. Put the H3 upscaler into "
                    "models/latent_upscale_models (subfolders are fine).")
        return True

    @classmethod
    def execute(cls, model, conditioning, latent, noise, sampler, sigmas,
                upscale_model="", width=1280, height=704,
                chunk_length=136, temporal_overlap=17, anchor_strength=0.999,
                precision="fp16", device="cuda",
                negative=None, cfg=1.0) -> IO.NodeOutput:
        import comfy.model_management
        import comfy.nested_tensor
        import torch

        samples = latent["samples"]
        if not core.is_h3_av_latent(samples):
            raise ValueError(
                f"{LOG_PREFIX} expects a MiniMax H3 audio+video latent "
                "(nested video [B,24,T,H,W] + audio [B,32,2,T])."
            )
        video, audio = samples.tensors[0], samples.tensors[1]
        if video.shape[0] != 1:
            raise ValueError(f"{LOG_PREFIX} expects a single-video latent (batch 1).")

        # Размеры кладутся на сетку 32 — так же, как это делала снятая нода
        # параметров, чтобы поведение не изменилось от переезда.
        upscale_param = _upscale_param(upscale_model, width, height, device, precision)

        conditioning = core.normalize_minimax_refs(conditioning)

        # ⚠️ Требование «размер обязан совпадать с размером кондиционирования»
        # в оригинале только написано в описании, но нигде не проверяется, и
        # расхождение всплывает глубоко внутри семплирования. Ключевые кадры
        # кондиционирования знают свой размер — сверяем с целевым тут же.
        cls._warn_on_conditioning_size_mismatch(
            conditioning, upscale_param["width"], upscale_param["height"],
        )

        total_tokens = video.shape[2]
        audio_tokens = audio.shape[-1]
        bounds, _frame_count = core.compute_segments(
            total_tokens, int(chunk_length), int(temporal_overlap),
        )

        # Цена нарезки, посчитанная заранее: за перекрытие платят лишним счётом,
        # и человеку полезно видеть, сколько именно он платит. У автора это
        # (chunk + overlap) / chunk — при 136/17 выходит ×1.13.
        redundancy = (int(chunk_length) + int(temporal_overlap)) / max(1, int(chunk_length))
        logger.info(
            "%s %d chunk(s) of %d frames, overlap %d — redundancy x%.2f",
            LOG_PREFIX, len(bounds), int(chunk_length), int(temporal_overlap), redundancy,
        )

        acc_v = acc_a = None
        segments = []

        # ⚠️ Апскейл идёт ГРУППАМИ, а не по одному куску.
        #
        # Апскейл и семплирование требуют разных моделей, и держать обе на карте
        # нельзя, поэтому между ними идёт выгрузка. В исходной схеме «апскейл →
        # семпл» на каждом куске это означало полную перезагрузку многогигабайтной
        # диффузионной модели СТОЛЬКО РАЗ, сколько кусков в клипе.
        #
        # Апскейл от порядка не зависит (в отличие от семплирования, где каждый
        # кусок якорится за результат предыдущего), поэтому куски апскейлятся
        # пачками: одна выгрузка на пачку вместо одной на кусок. Размер пачки —
        # по свободной оперативной памяти, потому что готовые куски ждут там же.
        group = cls._upscale_group_size(video, bounds, upscale_param)
        if group > 1:
            logger.info("%s Upscaling in groups of %d chunk(s).", LOG_PREFIX, group)

        upscaled_cache: dict[int, "object"] = {}

        for index, (k0, f0, k1, f1) in enumerate(bounds):
            a0, a1 = core.audio_range(f0, f1)
            chunk_a = audio[:, :, :, a0:min(a1, audio_tokens)].contiguous()

            if index not in upscaled_cache:
                # Пока считает апскейлер, диффузионная модель не нужна — убираем
                # её с карты, чтобы они не лежали там вдвоём. Следующий семпл
                # вернёт её сам.
                if device == "cuda" and hasattr(model, "clone_base_uuid"):
                    comfy.model_management.unload_model_and_clones(
                        model, unload_additional_models=False,
                    )
                    comfy.model_management.soft_empty_cache()
                for offset in range(index, min(index + group, len(bounds))):
                    b0, _bf0, b1, _bf1 = bounds[offset]
                    part = video[:, :, b0:b1].contiguous()
                    upscaled_cache[offset], _, _ = core.upscale_latent(part, upscale_param)

            chunk_v = upscaled_cache.pop(index)

            cond_i = core.reanchor_conditioning(
                conditioning, f0, f1, (chunk_v.shape[3], chunk_v.shape[4]),
            )
            if index > 0 and acc_v is not None:
                cond_i = core.anchor_conditioning(cond_i, acc_v, f0, float(anchor_strength))

            # ⚠️ Аудио на пересэмплинге ЗАКРЕПЛЯЕТСЯ (маска: видео 1, аудио 0).
            #
            # Иначе модель денойзит и звук тоже, то есть на каждом шаге видит его
            # зашумлённым — и видео нечему следовать. Для обычного ролика это
            # просто лишняя работа: результат по звуку всё равно выбрасывается
            # (ниже берётся только `out.tensors[0]`, а в `acc_a` копится исходный
            # `chunk_a`). А для липсинка это прямая потеря синхронизации: губы
            # уезжают от дорожки ровно на апскейле.
            #
            # Тот же приём, что в ноде TS H3 Audio Inject; удержание делает
            # штатный масочный путь ядра (`sample_piece` отдаёт маску сэмплеру).
            piece = {
                "samples": comfy.nested_tensor.NestedTensor((chunk_v, chunk_a)),
                "noise_mask": comfy.nested_tensor.NestedTensor((
                    torch.ones_like(chunk_v),
                    torch.zeros_like(chunk_a),
                )),
            }
            out = core.sample_piece(piece, cond_i, model, noise, sampler, sigmas, negative, cfg)
            chunk_out_v = out.tensors[0]

            acc_v, acc_a = core.temporal_append(
                acc_v, acc_a, chunk_out_v, chunk_a, index, k0, f0,
            )

            segments.append({
                "chunk": index,
                "frame_start": f0,
                "frame_count": f1 - f0,
                "video_tokens": [k0, k1],
                "audio_tokens": list(core.audio_range(f0, f1)),
                "spatial_h": chunk_v.shape[3],
                "spatial_w": chunk_v.shape[4],
            })
            logger.info(
                "%s chunk %d/%d: frames %d..%d at %dx%d",
                LOG_PREFIX, index + 1, len(bounds), f0, f1,
                chunk_v.shape[4], chunk_v.shape[3],
            )

        # Клип собран — модель больше не нужна, и тому, кто пойдёт декодировать
        # получившийся латент, эта память пригодится куда больше.
        if hasattr(model, "clone_base_uuid"):
            comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
            comfy.model_management.soft_empty_cache()

        result = {"samples": comfy.nested_tensor.NestedTensor((acc_v, acc_a))}
        return IO.NodeOutput(result, segments)


    @staticmethod
    def _upscale_group_size(video, bounds, upscale_param):
        """Сколько кусков апскейлить за один заход.

        Готовые куски ждут семплирования в оперативной памяти, поэтому группа
        считается от неё, а не от числа кусков: на 4K кусок весит около
        полугигабайта, и «все сразу» на длинном клипе означало бы десятки
        гигабайт. Берём четверть свободной памяти — остальное нужно самому
        ComfyUI.
        """
        if len(bounds) <= 1:
            return 1
        scale_w = int(upscale_param["width"]) / max(1, video.shape[4] * core.VAE_DOWNSAMPLE)
        scale_h = int(upscale_param["height"]) / max(1, video.shape[3] * core.VAE_DOWNSAMPLE)
        longest = max((b1 - b0) for b0, _f0, b1, _f1 in bounds)
        element = video.element_size() or 2
        bytes_per_chunk = (
            video.shape[1] * longest
            * video.shape[3] * scale_h * video.shape[4] * scale_w * element
        )
        try:
            import psutil

            budget = psutil.virtual_memory().available * 0.25
        except Exception:  # noqa: BLE001 - без psutil идём по одному куску
            return 1
        return max(1, min(len(bounds), int(budget // max(1.0, bytes_per_chunk))))

    @staticmethod
    def _warn_on_conditioning_size_mismatch(conditioning, width, height):
        """Сказать вслух, если кондиционирование делалось под другой размер.

        Предупреждение, а не отказ: у ключевого кадра может не быть латента, и
        тогда сверять попросту нечего — а мешать работе из-за неполной проверки
        неправильно.
        """
        expected = (int(height) // core.VAE_DOWNSAMPLE, int(width) // core.VAE_DOWNSAMPLE)
        for _cond, extra in conditioning or []:
            for keyframe in (extra or {}).get("keyframes", []) or []:
                latent = keyframe.get("latent") if isinstance(keyframe, dict) else None
                if latent is None or getattr(latent, "ndim", 0) != 5:
                    continue
                got = (int(latent.shape[3]), int(latent.shape[4]))
                if got != expected:
                    logger.warning(
                        "%s Conditioning keyframes are %dx%d latent (%dx%d pixels) but the "
                        "target is %dx%d pixels. They will be resized, and if the conditioning "
                        "was written for a different size the result will drift from it.",
                        LOG_PREFIX, got[1], got[0],
                        got[1] * core.VAE_DOWNSAMPLE, got[0] * core.VAE_DOWNSAMPLE,
                        int(width), int(height),
                    )
                    return


NODE_CLASS_MAPPINGS = {"TS_LatentUpscale": TS_LatentUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LatentUpscale": "TS Latent Upscale"}
