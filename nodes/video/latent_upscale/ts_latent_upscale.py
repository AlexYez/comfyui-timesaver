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
                    options=core._scan_models(),
                    tooltip=(
                        "H3 latent upscale checkpoint from models/latent_upscale_models. "
                        "Subfolders are listed too, as 'subfolder/file.safetensors'. These are "
                        "the minimax_h3_latent_upscaler_3d weights — an ordinary latent upscale "
                        "model will not load here."
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
                        "grid step. 136 ≈ 5.7 s, 153 ≈ 6.4 s. Shorter chunks cost less VRAM."
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
                    tooltip="Precision the upscaler runs at. fp16 is the usual choice; fp32 costs about twice the memory.",
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
        upscale_param = {
            "model_name": upscale_model,
            "width": int(round(int(width) / 32.0)) * 32,
            "height": int(round(int(height) / 32.0)) * 32,
            "device": device,
            "precision": precision,
        }

        conditioning = core.normalize_minimax_refs(conditioning)

        total_tokens = video.shape[2]
        audio_tokens = audio.shape[-1]
        bounds, _frame_count = core.compute_segments(
            total_tokens, int(chunk_length), int(temporal_overlap),
        )

        acc_v = acc_a = None
        segments = []

        for index, (k0, f0, k1, f1) in enumerate(bounds):
            chunk_v = video[:, :, k0:k1].contiguous()
            a0, a1 = core.audio_range(f0, f1)
            chunk_a = audio[:, :, :, a0:min(a1, audio_tokens)].contiguous()

            # Пока считает апскейлер, диффузионная модель не нужна — убираем её
            # с карты, чтобы они не лежали там вдвоём. Следующий семпл вернёт её
            # сам.
            if device == "cuda" and hasattr(model, "clone_base_uuid"):
                comfy.model_management.unload_model_and_clones(
                    model, unload_additional_models=False,
                )
                comfy.model_management.soft_empty_cache()
            chunk_v, _, _ = core.upscale_latent(chunk_v, upscale_param)

            cond_i = core.reanchor_conditioning(
                conditioning, f0, f1, (chunk_v.shape[3], chunk_v.shape[4]),
            )
            if index > 0 and acc_v is not None:
                cond_i = core.anchor_conditioning(cond_i, acc_v, f0, float(anchor_strength))

            piece = {"samples": comfy.nested_tensor.NestedTensor((chunk_v, chunk_a))}
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


NODE_CLASS_MAPPINGS = {"TS_LatentUpscale": TS_LatentUpscale}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_LatentUpscale": "TS Latent Upscale"}
