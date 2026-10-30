"""Native ComfyUI support for the Open-H3-Regenerate-2K release.

Recipe: NVlabs/Sana 781e8341, models/Open-H3-Regenerate-2K.
Diffusers/Comfy mapping and curve inversion also checked against
rdh9616/ComfyUI-Open-H3-Regenerate 9b686c5 (Apache-2.0).
No persistent changes to the base model's modules or parameters.
"""
from __future__ import annotations

import hashlib
import math
import re

import torch
from safetensors import safe_open

REFERENCE_TABLE_SHA256 = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"
SCHEDULES = {3: (0.909375, 6 / 7, 2 / 3, 0.0), 2: (0.7, 0.49411764705882355, 0.0)}


def sigmas(steps, denoise):
    steps, denoise = int(steps), float(denoise)
    if steps not in SCHEDULES or not math.isfinite(denoise) or not 0 <= denoise <= 1:
        raise ValueError("Regenerate requires 2 or 3 steps and denoise between 0 and 1.")
    if denoise == 1:
        return torch.tensor(SCHEDULES[steps], dtype=torch.float32)
    start = SCHEDULES[steps][0] * denoise
    raw = start / (6 - 5 * start)
    grid = [raw * (1 - i / steps) for i in range(steps + 1)]
    return torch.tensor([6 * s / (1 + 5 * s) for s in grid], dtype=torch.float32)


_TARGET = {
    "attn.to_q": ("attn.qkv_proj", 0), "attn.to_k": ("attn.qkv_proj", 1),
    "attn.to_v": ("attn.qkv_proj", 2), "attn.to_out.0": ("attn.out_proj", None),
    "ff.net.0.proj": ("mlp.fc1", "swiglu"), "ff.net.2": ("mlp.fc2", None),
}


def lora_target(name):
    match = re.fullmatch(r"transformer\.(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)", name)
    if match is None or match[3] not in _TARGET:
        raise ValueError(f"Unsupported Regenerate LoRA target: {name}")
    root = "blocks" if match[1] == "transformer_blocks" else "token_refiner.blocks"
    leaf, slot = _TARGET[match[3]]
    return f"diffusion_model.{root}.{match[2]}.{leaf}.weight", slot


def read_patches(path, state_dict, dtype=torch.bfloat16):
    from comfy.weight_adapter.lora import LoRAAdapter

    patches = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        names = sorted(k.removesuffix(".lora_A.weight") for k in keys if k.endswith(".lora_A.weight"))
        expected = {n + suffix for n in names for suffix in (".lora_A.weight", ".lora_B.weight")}
        if len(names) != 312 or keys != expected:
            raise ValueError("Expected the official merged Regenerate LoRA: 312 A/B pairs.")
        for name in names:
            key, slot = lora_target(name)
            a = f.get_tensor(name + ".lora_A.weight").to(dtype)
            b = f.get_tensor(name + ".lora_B.weight").to(dtype)
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError(f"Invalid Regenerate LoRA factor shapes: {name}")
            shape = state_dict[key].shape if key in state_dict else ()
            rows = b.shape[0] * (3 if isinstance(slot, int) else 1)
            if tuple(shape) != (rows, a.shape[1]):
                raise ValueError(f"Regenerate LoRA does not fit {key}: model {tuple(shape)}, adapter {(rows, a.shape[1])}")
            if slot == "swiglu":
                if b.shape[0] % 2:
                    raise ValueError(f"Odd SwiGLU output dimension: {name}")
                up, gate = b.chunk(2, dim=0)
                b = torch.cat((gate, up), dim=0)
            target = (key, (0, slot * b.shape[0], b.shape[0])) if isinstance(slot, int) else key
            # All merge strengths are already folded into B. No alpha/rank factor.
            patches[target] = LoRAAdapter(set(), (b, a, None, None, None, None))
    return patches


def read_reference_table(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        names = [k for k in f.keys() if k.endswith("adaln_t_table")]
        if len(names) != 1:
            raise ValueError("Choose the pruned Ref2VA checkpoint containing the reference AdaLN table.")
        table = f.get_tensor(names[0]).float().contiguous()
    digest = hashlib.sha256(table.numpy().tobytes()).hexdigest()
    if tuple(table.shape) != (1025, 8) or digest != REFERENCE_TABLE_SHA256:
        raise ValueError("Sharp requires the official reference AdaLN table from Ref2VA pruned (c02a6c11…).")
    return table


def table_coords(table, t):
    pos = t.float().clamp(0, 1) * (table.shape[0] - 1)
    lo = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[lo], table[lo + 1], (pos - lo).unsqueeze(-1))


def invert_curve(table, coords):
    table = table.to(device=coords.device, dtype=torch.float32)
    p, d = table[:-1], table[1:] - table[:-1]
    rel = coords.float()[:, None] - p[None]
    s = ((rel * d).sum(-1) / d.square().sum(-1).clamp_min(1e-20)).clamp(0, 1)
    dist = (rel - s[..., None] * d).square().sum(-1)
    best = dist.argmin(dim=1)
    row = torch.arange(coords.shape[0], device=coords.device)
    if dist[row, best].max().item() > 1e-6:
        raise ValueError("Sharp received coordinates outside the loaded model's AdaLN curve.")
    return (best.float() + s[row, best]) / (table.shape[0] - 1)


class SharpBlock:
    """Block replacement using native operations; no forward hooks or module copies.

    Only the small B@A matrix and two curve tables travel to the sampling device.
    The native model patcher continues to own, load and offload the block itself.
    """
    def __init__(self, dm, index, weight, own_table, reference):
        self.dm, self.index = dm, index
        self.weight, self.own_table, self.reference = weight, own_table, reference
        self.same_curve = torch.equal(own_table, reference)

    def to(self, device, _dtype_type=torch.dtype, _float32=torch.float32):
        # ComfyUI also calls .to(dtype) on patches; keep curve arithmetic FP32.
        if not isinstance(device, _dtype_type):
            self.weight = self.weight.to(device=device, dtype=_float32)
            self.own_table = self.own_table.to(device=device, dtype=_float32)
            self.reference = self.reference.to(device=device, dtype=_float32)
        return self

    def __call__(self, args, extra):
        from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift

        block = self.dm.blocks[self.index]
        x, t = args["img"], args["t_emb"]
        self.to(t.device)
        coords = t.float() if self.same_curve else table_coords(self.reference, invert_curve(self.own_table, t))
        delta = (coords @ self.weight.t()).reshape(t.shape[0] * 3, -1).chunk(6, dim=-1)
        shift, scale, gate, shift_ff, scale_ff, gate_ff = [
            base + change.to(base.dtype) for base, change in zip(block.adaln_proj(t), delta)
        ]
        segments = args["mod_segments"]
        h = _mod_scale_shift(block.norm1(x), shift, scale, segments)
        attention = args.get("attention", block.attn)
        x = _mod_gate(x, gate, attention(h, rope_freqs=args["rope_freqs"],
                                        transformer_options=args["transformer_options"]), segments)
        h = _mod_scale_shift(block.norm2(x), shift_ff, scale_ff, segments)
        return {"img": _mod_gate(x, gate_ff, block.mlp(h), segments)}


def install_sharp(model, path, reference):
    dm = model.get_model_object("diffusion_model")
    if not getattr(dm, "use_adaln_curves", False) or tuple(dm.adaln_t_table.shape) != (1025, 8):
        raise ValueError("TS Regenerate currently requires a pruned H3 model with 1025×8 AdaLN curves.")
    if len(dm.blocks) != 50:
        raise ValueError("Expected 50 native H3 transformer blocks.")
    own = dm.adaln_t_table.detach().to(device="cpu", dtype=torch.float32).clone()
    with safe_open(path, framework="pt", device="cpu") as f:
        expected = {f"transformer.transformer_blocks.{i}.adaln_proj.linear.curve_{ab}.weight"
                    for i in range(50) for ab in ("A", "B")}
        if set(f.keys()) != expected:
            raise ValueError("Expected the official Sharp file with 50 curve A/B pairs.")
        for i, block in enumerate(dm.blocks):
            root = f"transformer.transformer_blocks.{i}.adaln_proj.linear"
            a, b = (f.get_tensor(root + f".curve_{ab}.weight").float() for ab in ("A", "B"))
            if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 8 or a.shape[0] != b.shape[1] or b.shape[0] != block.adaln_proj.linear.weight.shape[0]:
                raise ValueError(f"Sharp branch {i} does not fit this model.")
            model.set_model_patch_replace(SharpBlock(dm, i, b @ a, own, reference), "dit", "double_block", i)


def text_conditioning(conditioning):
    # Refine uses the T2VA layout. Image/keyframe/audio references from draft
    # generation must not create extra frozen rows in the enlarged sequence.
    remove = {"minimax_keyframes", "minimax_refs", "minimax_visual_cond_noise_aug", "minimax_audio_cond_noise_aug"}
    return [[text, {k: v for k, v in metadata.items() if k not in remove}]
            for text, metadata in conditioning]


def encode_first_frame(image, video_vae, width, height):
    """Encode one RGB frame on the final canvas, never upscale its draft latent."""
    import comfy.utils

    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1 or image.shape[-1] < 3:
        raise ValueError("First frame must be an IMAGE batch [B,H,W,C] with at least three color channels.")
    pixels = image[:1, ..., :3]
    if tuple(pixels.shape[1:3]) != (height, width):
        pixels = comfy.utils.common_upscale(pixels.movedim(-1, 1), width, height,
                                           "lanczos", "disabled").movedim(1, -1)
    pixels = pixels.contiguous().clone()
    # This matches native H3 I2V canvas preparation. A prepared workflow image
    # already has this size; resizing is only a safeguard for other workflows.
    with torch.inference_mode():
        encoded = video_vae.encode(pixels)
    expected = (1, 24, 1, height // 16, width // 16)
    if not isinstance(encoded, torch.Tensor) or tuple(encoded.shape) != expected:
        raise ValueError(f"First-frame encoding requires the normal H3 video VAE, producing {expected}. "
                         "The fast Regenerate decoder and audio VAE cannot encode this reference.")
    return encoded.detach().to(device="cpu", dtype=torch.float32)


def first_frame_conditioning(conditioning, reference, frame_start):
    """Keep the original frame at global time zero in each chunk's local timeline."""
    # Later chunks must not relabel the source pose as their new first frame.
    # Native H3 RoPE accepts signed temporal positions; this chunked extension
    # remains experimental, as does chunking itself.
    return [[text, {**metadata, "minimax_keyframes": [
        {"resolved_frame_index": -frame_start, "latent": reference}]}]
            for text, metadata in conditioning]


def balance_short_tail(bounds, chunk_length, temporal_overlap):
    """Redistribute a sub-17-frame continuation tail without exceeding the cap.

    Native boundaries are on the 17-frame / 5-token grid. A 226-frame clip
    split at 85/17 otherwise ends with 17 locked frames and only 5 new ones.
    Balancing the existing four windows yields 68,68,68,73 instead of
    85,85,85,22, preserving the overlap, total work and exact final duration.
    """
    count = len(bounds)
    if count < 2 or bounds[-1][3] - bounds[-2][3] >= 17:
        return bounds
    frame_count = bounds[-1][3]
    groups = frame_count // 17 - temporal_overlap // 17
    hop, extra = divmod(groups, count)
    # Larger hops go first: the last window also contains the off-grid tail.
    hops = [hop + (index < extra) for index in range(count)]
    if any(value <= 0 for value in hops[:-1]):
        return bounds  # No better split exists on the native phase grid.
    starts = [0]
    for value in hops[:-1]:
        starts.append(starts[-1] + value * 17)
    result = []
    for index, start in enumerate(starts):
        end = starts[index + 1] + temporal_overlap if index + 1 < count else frame_count
        k1 = end // 17 * 5 if index + 1 < count else bounds[-1][2]
        result.append((start // 17 * 5, start, k1, end))
    if any(end - start > chunk_length for _, start, _, end in result):
        return bounds
    return result


def video_audio(samples):
    if getattr(samples, "is_nested", False):
        if len(samples.tensors) != 2:
            raise ValueError("Expected H3 video and audio streams.")
        video, audio = samples.tensors
    else:
        video, audio = samples, None
    if video.ndim != 5 or video.shape[1] != 24 or video.shape[0] != 1:
        raise ValueError("Expected one normalized H3 video latent [1,24,T,H,W].")
    if audio is not None and (audio.ndim != 4 or audio.shape[:3] != (1, 32, 2) or audio.shape[-1] == 0):
        raise ValueError("Expected an H3 audio latent [1,32,2,T].")
    return video, audio
