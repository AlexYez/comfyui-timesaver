"""StreamRender-H3 noncausal 24-layer decoder, with native VAE fallback."""
from __future__ import annotations

import logging

import torch
import folder_paths
from comfy_api.v0_0_2 import IO

from . import _regenerate as regen

LOG = logging.getLogger(__name__)


def _native(vae, video):
    if vae is None:
        raise RuntimeError("Fast H3 VAE ran out of memory. Connect the standard H3 video VAE to fallback_vae, or use a smaller clip.")
    images = vae.decode(video)
    return images.reshape(-1, *images.shape[-3:]) if images.ndim == 5 else images


def _decoder(path):
    from safetensors.torch import load_file
    import comfy.ops
    import comfy.utils
    from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD
    from .fast_vae.fast_vae_decoder import KEEP_24
    from .fast_vae.fast_vae_lean import LeanNonCausalViTDecoder

    state = load_file(path, device="cpu")
    quant = comfy.utils.detect_layer_quantization(state, "")
    operations = comfy.ops.mixed_precision_ops(quant, torch.bfloat16) if quant is not None else torch.nn
    # Avoid allocating and randomly initializing another copy in RAM.
    with torch.device("meta"):
        model = LeanNonCausalViTDecoder(LATENTS_MEAN, LATENTS_STD, KEEP_24, time_norm=5,
                                        heads=32, dim_head=64, rope_theta=100, rope_dim_ratio=0.75,
                                        eps=1e-5, latent_channels=24, operations=operations)
    model.load_state_dict(state, strict=True, assign=True)
    # Nonpersistent buffers are not part of state_dict and are still meta.
    model.latents_mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32)
    model.latents_std = torch.tensor(LATENTS_STD, dtype=torch.float32)
    model.pos_embed.inv_freq = 1 / 100.0 ** torch.arange(0, 1, 6 / 48, dtype=torch.float32)
    return model.eval().requires_grad_(False)


def _weight_bytes(path):
    """Tensor bytes from the safetensors header, including INT8 scales/metadata."""
    import json
    import struct

    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    return max(value["data_offsets"][1] for key, value in header.items() if key != "__metadata__")


class TS_H3FastVAEDecode(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        names = [p for p in folder_paths.get_filename_list("vae") if "fast_h3_vae_noncausal_24l" in p.lower()]
        return IO.Schema(
            node_id="TS_H3FastVAEDecode", display_name="TS H3 Fast VAE Decode", category="TS/Video",
            description="NVIDIA StreamRender-H3 noncausal 24-layer decoder, FP32 or native INT8 convrot weights, PyTorch SDPA path. Connect a standard H3 video VAE for automatic memory fallback.",
            inputs=[
                IO.Latent.Input("latent"),
                IO.Combo.Input("vae_name", options=names or ["(missing fast_h3_vae_noncausal_24l)"],
                               default=next((name for name in names if "int8_convrot" in name), names[0] if names else None)),
                IO.Vae.Input("fallback_vae", optional=True,
                             tooltip="Standard H3 video VAE; used if fast decoding needs more VRAM than is available. Audio is ignored."),
            ],
            outputs=[IO.Image.Output("images"), IO.String.Output("report")],
        )

    @classmethod
    def execute(cls, latent, vae_name, fallback_vae=None):
        import comfy.model_management as mm

        video, _ = regen.video_audio(latent["samples"])
        device = mm.get_torch_device()
        _, _, t, h, w = video.shape
        # Conservative upstream estimate; decide before reading the 6.5 GB file.
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        need = _weight_bytes(path) + t * h * w * 26 * 1024 + 3 * 2**30
        mm.free_memory(need, device)
        if device.type == "cuda" and need > mm.get_free_memory(device):
            report = f"Standard H3 VAE fallback: fast decode estimated {need / 2**30:.1f} GiB."
            if fallback_vae is None:
                raise RuntimeError(report + " Connect fallback_vae or reduce clip size.")
            LOG.info(report)
            return IO.NodeOutput(_native(fallback_vae, video), report)
        frames = 4 * t - 3 * ((t + 4) // 5)
        out = torch.empty(frames, h * 16, w * 16, 3, device=mm.intermediate_device(), dtype=torch.float32)
        model = _decoder(path)
        try:
            model.to(device)
            mean = torch.tensor((.485, .456, .406), device=device).view(1, 3, 1, 1, 1)
            std = torch.tensor((.229, .224, .225), device=device).view(1, 3, 1, 1, 1)

            def sink(start, pixels):
                part = (pixels.float() * std + mean).clamp_(0, 1)[0].permute(1, 2, 3, 0)
                out[start:start + part.shape[0]].copy_(part)

            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                model(video.to(device=device, dtype=torch.float32).transpose(1, 2), sink=sink)
        except torch.OutOfMemoryError:
            out = None
        finally:
            del model
            mm.soft_empty_cache()
        if out is None:
            return IO.NodeOutput(_native(fallback_vae, video), "Standard H3 VAE fallback after fast decoder OOM.")
        return IO.NodeOutput(out, f"Fast H3 VAE: {frames} frames, {w * 16}×{h * 16}, PyTorch SDPA.")


NODE_CLASS_MAPPINGS = {"TS_H3FastVAEDecode": TS_H3FastVAEDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_H3FastVAEDecode": "TS H3 Fast VAE Decode"}
