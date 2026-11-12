"""Open-H3-Regenerate-2K-IC on native, non-pruned H3.

The IC recipe needs the original time embedder, HyperFlow's endpoint embedder,
Sharp, and a clean low-resolution video in both Qwen and DiT context. Pruned
checkpoints cannot implement the two-time path. No global core patches are used.
Source: NVlabs/Sana, models/Open-H3-Regenerate-2K/IC (2026-11-11).
"""
from __future__ import annotations

import json
import logging
import math
import threading

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ...utils.ts_lora_unmerged import LoraBranch, LoraTarget
from . import _regenerate as regen

STANDARD = "Standard"
IC_UP3 = "IC up3 (HyperFlow)"
IC_RANK = 384
LOG = logging.getLogger(__name__)
TASK = ("Enhance <Video 1> into a high-quality version of the same video. "
        "Preserve the scene, objects, camera motion, timing, and composition of every frame. "
        "Restore fine details and remove degradation. Do not continue or change the scene.")
TIME_NAMES = {f"transformer.{root}.linear_{i}" for root in ("time_embedder", "endpoint_time_embedder") for i in (1, 2)}


def sigmas(metadata, steps, denoise):
    if int(steps) != 3:
        raise ValueError("IC up3 requires steps=3; the Standard 2-step schedule is not an IC schedule.")
    if not math.isfinite(float(denoise)) or not 0 <= float(denoise) <= 1:
        raise ValueError("IC denoise must be between 0 and 1.")
    raw = torch.tensor(json.loads(metadata["hyperflow_sigmas"]), dtype=torch.float32)
    if raw.shape != (9,) or raw[0] != 1 or raw[-1] != 0 or not bool((raw[:-1] > raw[1:]).all()):
        raise ValueError("Expected the released HyperFlow 8-step sigma grid.")
    shifted = 12 * raw / (1 + 11 * raw)
    # Scaling the trained tail is an explicit lower-denoise experiment; 1 is
    # the published up3 schedule. Zero is handled as upscale-only by the node.
    return shifted[-4:] * float(denoise)


def _factors(f, name, dtype):
    a, b = (f.get_tensor(name + f".lora_{ab}.weight").to(dtype) for ab in ("A", "B"))
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError(f"Invalid IC adapter factors: {name}")
    return a, b


def _names(f):
    keys = set(f.keys())
    names = {k.removesuffix(".lora_A.weight") for k in keys if k.endswith(".lora_A.weight")}
    if keys != {n + s for n in names for s in (".lora_A.weight", ".lora_B.weight")}:
        raise ValueError("The adapter must contain only complete A/B pairs.")
    return names


def _native_factors(name, a, b, state):
    key, slot = regen.lora_target(name)
    rows = b.shape[0] * (3 if isinstance(slot, int) else 1)
    if key not in state or tuple(state[key].shape) != (rows, a.shape[1]):
        raise ValueError(f"IC adapter does not fit {key}.")
    if slot == "swiglu":
        value, gate = b.chunk(2, dim=0)
        b = torch.cat((gate, value), dim=0)
    return key, slot, a, b


def backbone(path, state, dtype, *, hyperflow=False):
    """Map diffusers Q/K/V slices and SwiGLU, without padded fused factors."""
    from comfy.weight_adapter.lora import LoRAAdapter

    patches, targets, times = {}, [], {}
    with safe_open(path, framework="pt", device="cpu") as f:
        names, metadata = _names(f), f.metadata() or {}
        normal = names - TIME_NAMES
        if len(normal) != 312 or (names & TIME_NAMES) != (TIME_NAMES if hyperflow else set()):
            raise ValueError("Expected 312 IC pairs or 312 backbone + 4 HyperFlow time pairs.")
        if hyperflow and (metadata.get("hyperflow") != "true" or float(metadata.get("hyperflow_video_shift", 0)) != 12
                          or float(metadata.get("hyperflow_gate", -1)) != .25
                          or int(metadata.get("lora_alpha", 0)) != int(metadata.get("lora_rank", -1))):
            raise ValueError("Choose the original HyperFlow 8-step v1.0 adapter (shift 12, gate .25, alpha=rank).")
        for name in sorted(names):
            a, b = _factors(f, name, torch.float32 if name in TIME_NAMES else dtype)
            expected_rank = int(metadata["lora_rank"]) if hyperflow else IC_RANK
            if a.shape[0] != expected_rank:
                raise ValueError(f"Expected {'HyperFlow' if hyperflow else 'original IC'} rank {expected_rank}; got {a.shape[0]} at {name}. Do not use a Standard or IC+Sharp merged adapter here.")
            if name in TIME_NAMES:
                times[name] = (a, b)
                if ".endpoint_time_embedder." in name:
                    continue
                leaf = name.rsplit(".", 1)[-1]
                key = "diffusion_model.time_embedder." + {"linear_1": "proj_in", "linear_2": "proj_out"}[leaf] + ".weight"
                if tuple(state[key].shape) != (b.shape[0], a.shape[1]):
                    raise ValueError(f"HyperFlow time factors do not fit {key}.")
                patches[key] = LoRAAdapter(set(), (b, a, None, None, None, None))
                continue
            key, slot, a, b = _native_factors(name, a, b, state)
            if hyperflow:
                target = (key, (0, slot * b.shape[0], b.shape[0])) if isinstance(slot, int) else key
                patches[target] = LoRAAdapter(set(), (b, a, None, None, None, None))
            else:
                targets.append(LoraTarget(key.removeprefix("diffusion_model.").removesuffix(".weight"), a, b,
                                          offset=slot * b.shape[0] if isinstance(slot, int) else None,
                                          size=b.shape[0] if isinstance(slot, int) else None))
    return patches, targets, times, metadata


def sharp(path, state, dtype):
    from comfy.weight_adapter.lora import LoRAAdapter

    patches, curves = {}, {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in sorted(_names(f)):
            a, b = _factors(f, name, torch.float32 if ".adaln_proj.linear" in name else dtype)
            key = name + ".weight"
            if ".adaln_proj.linear" in name:
                index = int(name.split(".")[2])
                expected = state.get(f"diffusion_model.blocks.{index}.adaln_proj.linear.weight")
                if a.shape[1] != 8 or expected is None or b.shape[0] != expected.shape[0]:
                    raise ValueError(f"Sharp curve does not fit block {index}.")
                curves[index] = b.float() @ a.float()
            else:
                if key not in state or tuple(state[key].shape) != (b.shape[0], a.shape[1]):
                    raise ValueError(f"Sharp adapter does not fit {key}.")
                patches[key] = LoRAAdapter(set(), (b, a, None, None, None, None))
    if len(patches) != 208 or set(curves) != set(range(50)):
        raise ValueError("Choose the original minimax_h3_lms_v1.0_r64 adapter: 208 linears and 50 curves.")
    return patches, curves


class TimeContext(threading.local):
    def __init__(self):
        self.coords = None


class TwoTime:
    def __init__(self, embedder, times, reference):
        # Keep the endpoint copy BEFORE the base branch receives HyperFlow.
        self.weights = {}
        for i, native in ((1, "proj_in"), (2, "proj_out")):
            layer = getattr(embedder, native)
            a, b = times[f"transformer.endpoint_time_embedder.linear_{i}"]
            weight = layer.weight.detach().to(device="cpu", dtype=torch.float32).clone()
            if tuple(weight.shape) != (b.shape[0], a.shape[1]):
                raise ValueError("HyperFlow endpoint does not fit the original time embedder.")
            self.weights[native] = (weight + b @ a, layer.bias.detach().to(device="cpu", dtype=torch.float32).clone())
        self.freq_dim = embedder.freq_dim
        self.reference = reference
        self.context = TimeContext()

    def endpoint(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        phase = t.float()[:, None] * freqs[None]
        out = torch.cat((phase.cos(), phase.sin()), dim=-1)
        for name in ("proj_in", "proj_out"):
            w, b = self.weights[name]
            out = F.linear(out, w.to(t.device), b.to(t.device))
            if name == "proj_in":
                out = F.silu(out)
        return out

    def embedding(self, dm, times, device, dtype, sigma, next_sigma):
        t = torch.tensor(times, device=device, dtype=torch.float32)
        endpoint = t.clone()
        generated = torch.isclose(t, t.new_tensor(1. - sigma), atol=1e-6, rtol=0.)
        if int(generated.sum()) != 1:
            raise ValueError("IC could not identify the generated video timestep.")
        endpoint[generated] = 1. - next_sigma
        self.context.coords = regen.table_coords(self.reference.to(device), t)
        base = dm.time_embedder(t)
        return torch.lerp(base.float(), self.endpoint(endpoint), .25).to(dtype)

    def run(self, executor, *args, **kwargs):
        dm = executor.class_obj
        options = args[3] if len(args) > 3 else kwargs["transformer_options"]
        timestep = args[1] if len(args) > 1 else kwargs["timestep"]
        grid = options.get("sample_sigmas")
        if grid is None:
            raise ValueError("IC requires the sampler's sigma schedule for HyperFlow endpoints.")
        sigma = float((timestep.flatten()[0] / 1000.).float())
        distance = (grid.float() - sigma).abs()
        index = int(distance.argmin())
        if index >= len(grid) - 1 or float(distance[index]) > 1e-5:
            raise ValueError("IC requires Euler evaluations exactly on the supplied sigma grid.")
        next_sigma = float(grid[index + 1])
        # Scope all method changes to one forward of this model clone. Restore
        # even when a block raises or sampling is interrupted.
        saved = []
        def replace(obj, name, value):
            saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
            setattr(obj, name, value)
        try:
            replace(dm, "_time_embedding", lambda times, device, dtype: self.embedding(dm, times, device, dtype, sigma, next_sigma))
            return executor(*args, **kwargs)
        finally:
            self.context.coords = None
            for obj, name, existed, value in reversed(saved):
                if existed:
                    setattr(obj, name, value)
                else:
                    delattr(obj, name)


class ICSharpBlock:
    def __init__(self, dm, index, weight, context):
        self.dm, self.index, self.weight, self.context = dm, index, weight, context

    def to(self, device):
        if not isinstance(device, torch.dtype):
            self.weight = self.weight.to(device=device, dtype=torch.float32)
        return self

    def __call__(self, args, extra):
        from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift

        block = self.dm.blocks[self.index]
        x, t = args["img"], args["t_emb"]
        coords = self.context.coords
        if coords is None or coords.shape[0] != t.shape[0]:
            raise ValueError("IC Sharp requires the scoped HyperFlow time context.")
        self.to(t.device)
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


class ICBranch(LoraBranch):
    def prefetch(self, guider):
        import comfy.model_management as mm

        device = guider.model_patcher.load_device
        self.streamed = torch.device(device).type == "cuda" and mm.get_total_memory(device) <= 24 * 1024**3
        LOG.info("[TS H3 IC] LoRA storage: %s", "streamed per block (GPU <=24 GiB)" if self.streamed else "resident")
        if not self.streamed:
            super().prefetch(guider)

    def attach(self, diffusion_model, device, dtype, handles, fired, readers):
        grouped = {}
        streamed = getattr(self, "streamed", False)
        weights = None if streamed else self.weights_on(device, dtype)
        for index, target in enumerate(self.targets):
            a, b = (target.down, target.up) if streamed else weights[index]
            grouped.setdefault(target.name, []).append((target.offset, target.size, a, b, target.scale if streamed else 1.))
        # One block's factors survive its alternating fc1/fc2 row tiles. Do
        # not retransmit each pair for every tile or keep all 52 blocks in VRAM.
        block_cache = {"block": None, "weights": {}}
        def on_device(name, factors):
            if not streamed:
                return factors
            block = name.rsplit(".", 2)[0]
            if block != block_cache["block"]:
                block_cache["weights"].clear()
                block_cache["block"] = block
            cached = block_cache["weights"].get(name)
            if cached is None:
                cached = []
                for offset, size, a, b, scale in factors:
                    if scale != 1.:
                        b = b.float() * scale
                    cached.append((offset, size, a.to(device=device, dtype=dtype, non_blocking=True),
                                   b.to(device=device, dtype=dtype, non_blocking=True), 1.))
                block_cache["weights"][name] = cached
            return cached
        for name, factors in grouped.items():
            def hook(module, inputs, output, name=name, factors=factors):
                fired.add(name)
                x = inputs[0]
                if x.ndim != 2 or output.ndim != 2:
                    raise ValueError("IC expects native H3 packed 2D linear activations.")
                # A full 2K MLP delta can be several GiB. The base output is
                # fresh; update it in row tiles, preserving the release's
                # activation-dtype branch rounding without a second full copy.
                for offset, size, a, b, _scale in on_device(name, factors):
                    for start in range(0, x.shape[0], 4096):
                        delta = F.linear(F.linear(x[start:start + 4096], a.to(x.dtype)), b.to(x.dtype))
                        if offset is None:
                            output[start:start + 4096].add_(delta)
                        else:
                            output[start:start + 4096, offset:offset + size].add_(delta)
                return output
            handles.append(diffusion_model.get_submodule(name).register_forward_hook(hook))
        return set(grouped)


def install_ic_branch(patcher, targets):
    """Include H3's token refiner, which executes BEFORE the DiT wrapper."""
    import comfy.patcher_extension as extension

    branch = ICBranch(targets, "H3 Regenerate IC (312 linears)")

    def outer(executor, *args, **kwargs):
        guider = executor.class_obj
        model = guider.model_patcher
        dm = model.get_model_object("diffusion_model")
        handles, readers, saved, fired = [], [], [], set()
        try:
            branch.prefetch(guider)
            expected = branch.attach(dm, model.load_device, model.model.get_dtype_inference(), handles, fired, readers)
            for block in (*dm.token_refiner.blocks, *dm.blocks):
                mlp = block.mlp
                saved.append((mlp, "forward" in mlp.__dict__, mlp.__dict__.get("forward")))
                # The fused INT8 SwiGLU bypasses fc2.__call__. Explicit
                # activation lets both linear modules apply their IC branch.
                def forward(x, mlp=mlp):
                    def rows(part):
                        hidden = mlp.fc1(part)
                        gate, value = hidden.chunk(2, dim=-1)
                        activation = F.silu(gate)
                        activation.mul_(value)
                        del gate, value, hidden
                        return mlp.fc2(activation)
                    if x.shape[0] <= 4096:
                        return rows(x)
                    # MLP has no cross-token interaction. Avoid keeping the
                    # full [tokens, 2*ffn] and [tokens, ffn] activations alive.
                    output = torch.empty_like(x)
                    for start in range(0, x.shape[0], 4096):
                        output[start:start + 4096] = rows(x[start:start + 4096])
                    return output
                mlp.forward = forward
            result = executor(*args, **kwargs)
            if expected - fired:
                raise RuntimeError(f"IC adapter targets were skipped: {sorted(expected - fired)[:3]}")
            return result
        finally:
            for handle in handles:
                handle.remove()
            for mlp, existed, original in reversed(saved):
                if existed:
                    mlp.forward = original
                else:
                    delattr(mlp, "forward")
            branch.release()

    patcher.add_wrapper_with_key(extension.WrappersMP.OUTER_SAMPLE, "ts_h3_ic_lora", outer)


def install(model, ic_path, hyperflow_path, sharp_path, reference, dtype):
    import comfy.patcher_extension as extension

    dm = model.get_model_object("diffusion_model")
    if getattr(dm, "use_adaln_curves", True) or not hasattr(dm, "time_embedder") or len(dm.blocks) != 50:
        raise ValueError("IC needs the original non-pruned Ref2VA checkpoint, e.g. minimax_h3_ref2va_int8_convrot.safetensors. Pruned H3 has no HyperFlow time embedder.")
    state = model.model.state_dict()
    hf, _, times, meta = backbone(hyperflow_path, state, dtype, hyperflow=True)
    _, targets, _, _ = backbone(ic_path, state, dtype)
    lms, curves = sharp(sharp_path, state, dtype)
    # Validate schedule/metadata before constructing a modified model clone.
    sigmas(meta, 3, 1.)
    clock = TwoTime(dm.time_embedder, times, reference)
    result = model.clone()
    for patches in (hf, lms):
        if len(result.add_patches(patches, strength_patch=1.)) != len(patches):
            raise ValueError("IC failed to apply all HyperFlow/Sharp patches.")
    for index, weight in curves.items():
        result.set_model_patch_replace(ICSharpBlock(dm, index, weight, clock.context), "dit", "double_block", index)
    install_ic_branch(result, targets)
    result.add_wrapper_with_key(extension.WrappersMP.DIFFUSION_MODEL, "ts_h3_ic_time", clock.run)
    result.model_options["ts_h3_regenerate_ic"] = meta
    result.model_options["ts_h3_regenerate"] = True
    result.model_options["ts_h3_regenerate_ref2va"] = True
    return result


def encode_context(clip, images, caption, frame_count):
    if clip is None or images is None:
        raise ValueError("IC requires ic_clip (H3 Qwen encoder) and draft_images (decoded draft at 24 fps).")
    if images.ndim != 4 or images.shape[0] != frame_count or images.shape[-1] < 3:
        raise ValueError(f"IC draft_images must contain exactly {frame_count} RGB frames at 24 fps.")
    frames = images[::12, :, :, :3]
    tokens = clip.tokenize(TASK + "\n\nScene description: " + (caption or ""), minimax_ref_items=[
        {"type": "video", "data": frames, "timestamps": [i / 2. for i in range(len(frames))]}
    ])
    cond = clip.encode_from_tokens_scheduled(tokens)
    if not cond or not any((meta.get("minimax_token_tags", torch.ones(1)) == 0).any() for _, meta in cond):
        raise ValueError("IC Qwen conditioning is missing the draft's vision token tags; use the H3 Qwen encoder.")
    return regen.text_conditioning(cond)


def conditioning(cond, video):
    reference = {"kind": "video", "latent": video, "latent_t": video.shape[2],
                 "latent_h": video.shape[-2], "latent_w": video.shape[-1], "ref_audio_t": 0, "audio_latent": None}
    return [[text, dict(meta, minimax_refs=[reference], minimax_visual_cond_noise_aug=.999)] for text, meta in cond]


def upscale_clip(video, model_name, width, height, precision):
    """Core H3 upscaler, with whole-clip statistics before temporal sampling."""
    import comfy.model_management as mm
    from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3LatentUpscaleWithModel

    model, = LatentUpscaleModelLoader.execute(model_name).args
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    # The fresh loader result belongs to this invocation, with no shared clones.
    model.model.to(dtype=dtype)
    model.model.dtype = dtype
    mm.archive_model_dtypes(model.model)
    model.size = 0
    targets = []
    if width * height > 2560 * 1440 and video.shape[-1] * video.shape[-2] * 256 < 2560 * 1440:
        scale = math.sqrt(2560 * 1440 / (width * height))
        mid = (max(video.shape[-1] * 16, int(width * scale / 32) * 32),
               max(video.shape[-2] * 16, int(height * scale / 32) * 32))
        if (width, height) == (3840, 2176):
            mid = (2560, 1440)
        elif (width, height) == (2176, 3840):
            mid = (1440, 2560)
        targets.append(mid)
    targets.append((width, height))
    try:
        for w, h in targets:
            result, = MiniMaxH3LatentUpscaleWithModel.execute(model, {"samples": video}, w, h).args
            video = result["samples"].cpu()
        return video
    finally:
        mm.unload_model_and_clones(model, unload_additional_models=False)
        mm.soft_empty_cache()
