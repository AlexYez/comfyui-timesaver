# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

from .utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
INTERP_LEN = 8

# ImageNet stats — same constants used by the original `NormalizeImage` transform.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _compute_resize_target(h: int, w: int, input_size: int, multiple: int = 14) -> tuple[int, int]:
    """Lower-bound resize keeping aspect ratio with multiple-of-14 constraint.

    Mirrors the math of `util/transform.Resize(resize_method='lower_bound',
    keep_aspect_ratio=True)`, but as plain integer arithmetic so we can drive
    `F.interpolate` directly on the GPU.
    """
    scale = max(input_size / h, input_size / w)
    new_h = _round_to_multiple(h * scale, multiple)
    new_w = _round_to_multiple(w * scale, multiple)
    # `lower_bound`: snap up if rounding produced a sub-target dimension.
    if new_h < input_size:
        new_h = ((input_size + multiple - 1) // multiple) * multiple
    if new_w < input_size:
        new_w = ((input_size + multiple - 1) // multiple) * multiple
    return new_h, new_w


def _adapt_input_size_for_aspect(input_size: int, frame_h: int, frame_w: int) -> int:
    """Apply the same 16:9 aspect-ratio safeguard as the original model."""
    long_side = max(frame_h, frame_w)
    short_side = min(frame_h, frame_w)
    ratio = long_side / max(short_side, 1)
    if ratio > 1.78:
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14
    return max(14, int(input_size))


def _pick_backbone_sub_chunk(h: int, w: int, total: int) -> int:
    """Heuristic: choose how many frames to send through the DINOv2 backbone
    per sub-batch. Smaller chunks bound peak activation memory on 4K inputs
    where the legacy "all 32 frames at once" path hits OOM on a 16 GB card.

    Anchored on pixel count rather than aspect ratio so 21:9 and 16:9 inputs
    converge to the same chunk. Thresholds set deliberately low: a 448×798
    retry still produces ~1.2 GB refinenet1 spikes, so we sub-chunk there
    too. Erring on the side of more chunks costs only a small empty_cache
    between sub-batches.
    """
    pixels = h * w
    if pixels >= 500 * 800:      # ≥ ~400 k (518×924, 644×1148, …)
        return min(4, total)
    if pixels >= 350 * 600:      # ≥ ~210 k (448×798, 392×700, …)
        return min(8, total)
    if pixels >= 250 * 400:      # ≥ ~100 k (350×616, 308×548, …)
        return min(16, total)
    return total


def _preprocess_frames_gpu(
    frames: torch.Tensor,
    target_h: int,
    target_w: int,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int = 8,
    on_chunk_done=None,
) -> torch.Tensor:
    """Resize + ImageNet-normalize a stack of frames entirely on the GPU.

    Args:
        frames: tensor shaped (N, H, W, 3), float32 in [0, 1], any device.
        target_h, target_w: spatial size expected by the model.
        device: target device for the model input.
        dtype: dtype to cast to (fp16 / fp32) to match autocast.
        chunk_size: how many frames to push through F.interpolate at once.
            Keeps activation memory bounded for 4K inputs.
        on_chunk_done: optional callable receiving the number of frames just
            processed; used by the node to drive the preprocess slice of the
            UI progress bar.

    Returns:
        Tensor (N, 3, target_h, target_w) on ``device`` and ``dtype``.
    """
    n = frames.shape[0]
    mean = torch.tensor(_IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    # We keep an output buffer on GPU. For 4K @ 30s @ ~518 input_size the
    # buffer is ~1280×720 per frame × N × 3 × 4 bytes ≈ 3.3 GB worst case, which
    # is acceptable; the source frames live separately.
    out = torch.empty((n, 3, target_h, target_w), device=device, dtype=dtype)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = frames[start:end]
        if chunk.device != device:
            chunk = chunk.to(device, non_blocking=True)
        # NHWC float[0,1] → NCHW
        chunk = chunk.permute(0, 3, 1, 2).contiguous().float()
        chunk = F.interpolate(chunk, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
        chunk = (chunk - mean) / std
        out[start:end].copy_(chunk.to(dtype), non_blocking=True)
        del chunk
        if on_chunk_done is not None:
            on_chunk_done(end - start)

    return out


class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe='ape'
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)

    def forward(self, x, device):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        x_flat = x.flatten(0, 1)
        layer_idx = self.intermediate_layer_idx[self.encoder]

        # Sub-batch the DINOv2 backbone over T to bound activation memory on
        # 4K inputs. The backbone has no cross-frame state — we concatenate
        # the per-chunk features post-hoc, which is bit-identical to a single
        # large batch in eval mode.
        sub_chunk = _pick_backbone_sub_chunk(H, W, x_flat.shape[0])
        if sub_chunk >= x_flat.shape[0]:
            features = self.pretrained.get_intermediate_layers(
                x_flat, layer_idx, return_class_token=True
            )
        else:
            parts = []
            for i in range(0, x_flat.shape[0], sub_chunk):
                f = self.pretrained.get_intermediate_layers(
                    x_flat[i:i + sub_chunk], layer_idx, return_class_token=True
                )
                # Keep each tuple on-device; we accumulate then concat once.
                parts.append(f)
                if x.is_cuda:
                    torch.cuda.empty_cache()
            n_layers = len(parts[0])
            features = tuple(
                (
                    torch.cat([p[i][0] for p in parts], dim=0),
                    torch.cat([p[i][1] for p in parts], dim=0),
                )
                for i in range(n_layers)
            )
            del parts

        depth = self.head(features, patch_h, patch_w, T, device)
        del features
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T))  # [B, T, H, W]

    def infer_video_depth_torch(
        self,
        frames: torch.Tensor,
        input_size: int = 518,
        device=None,
        fp32: bool = False,
        pbar=None,
        interrupt_cb=None,
    ) -> torch.Tensor:
        """Run inference on a pre-resized, pre-normalised torch tensor.

        Args:
            frames: (N, 3, H_in, W_in) float, already resized to model input and
                ImageNet-normalised. Lives on ``device``. dtype must match
                ``fp32`` flag (model autocast picks the rest).
            input_size: original requested model input size (used only for
                logging / debug — actual size comes from `frames` shape).
            device: torch.device; defaults to ``frames.device``.
            fp32: disable autocast if True.
            pbar: optional ComfyUI ProgressBar.
            interrupt_cb: optional callable; when present it is invoked once per
                chunk and may raise to abort the run (typically wired to
                ``comfy.model_management.throw_exception_if_processing_interrupted``).

        Returns:
            Tensor (N, H_in, W_in) of relative inverse-depth, on CPU as float32.
            The original RGB resolution is restored later by the node-level
            postprocess; doing it here would double the upsample cost and break
            the chunked-memory budget on 4K inputs.
        """
        if device is None:
            device = frames.device
        device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]

        assert frames.ndim == 4 and frames.shape[1] == 3, \
            f"infer_video_depth_torch expects (N, 3, H, W), got {tuple(frames.shape)}"

        org_video_len = int(frames.shape[0])
        H_in, W_in = int(frames.shape[2]), int(frames.shape[3])

        frame_step = INFER_LEN - OVERLAP
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        # Pad by repeating the last frame so every sliding window has INFER_LEN
        # frames. The pad block uses `expand` (a view, no copy) before `cat`.
        last_frame = frames[-1:].clone()
        pad = last_frame.expand(append_frame_len, -1, -1, -1)
        padded = torch.cat([frames, pad], dim=0)
        del last_frame, pad

        # Collect per-frame depths chunk-by-chunk. Match the legacy semantics
        # exactly: ALL INFER_LEN frames of each sliding window land in
        # `depth_list`, so the final length is `num_chunks * INFER_LEN`
        # (overlapping frames appear twice, by design). `_align_chunks` relies
        # on that layout — collapsing the overlap here breaks alignment with
        # an IndexError downstream.
        depth_list: list[np.ndarray] = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            if interrupt_cb is not None:
                interrupt_cb()

            cur_input = padded[frame_id:frame_id + INFER_LEN].unsqueeze(0).to(device, non_blocking=True)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=device_type, enabled=(not fp32)):
                    depth = self.forward(cur_input, device_type)  # [1, T, H, W]

            # Cast to float32 (legacy expects fp32 numpy frames in alignment),
            # ship to CPU, append individual frames. This is the same shape
            # contract as the legacy `infer_video_depth` numpy path.
            depth_squeezed = depth.squeeze(0).float().cpu().numpy()
            depth_list.extend(depth_squeezed[i] for i in range(depth_squeezed.shape[0]))

            pre_input = cur_input
            del cur_input, depth, depth_squeezed
            # Release intermediate activations between sliding windows so the
            # next chunk starts from a clean budget. Cheap on CUDA, no-op on CPU.
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if pbar is not None:
                pbar.update(frame_step)

        del padded
        if device.type == "cuda":
            torch.cuda.empty_cache()

        aligned = _align_chunks(depth_list)
        out = np.stack(aligned[:org_video_len], axis=0)
        return torch.from_numpy(out)

    def infer_video_depth(self, frames, input_size=518, device='cuda', fp32=False, pbar=None):
        """Legacy numpy-in numpy-out API kept for backwards compatibility.

        The TS_VideoDepth node uses ``infer_video_depth_torch`` directly; this
        method exists so external callers and stand-alone scripts that still
        pass a ``(N, H, W, 3) uint8`` ndarray keep working.
        """
        frame_height, frame_width = frames[0].shape[:2]
        input_size = _adapt_input_size_for_aspect(input_size, frame_height, frame_width)

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)),
            PrepareForNet(),
        ])

        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            cur_list = []
            for i in range(INFER_LEN):
                cur_list.append(torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0))
            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=(not fp32)):
                    depth = self.forward(cur_input, device.type)

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

            pre_input = cur_input
            if pbar is not None:
                pbar.update(frame_step)

        del frame_list
        aligned = _align_chunks(depth_list)
        return np.stack(aligned[:org_video_len], axis=0)


def _align_chunks(depth_list):
    """Scale-and-shift align overlapping INFER_LEN windows.

    Extracted from `infer_video_depth` so both the legacy numpy path and the
    new torch path share one alignment routine.
    """
    depth_list_aligned = []
    ref_align = []
    align_len = OVERLAP - INTERP_LEN
    kf_align_list = KEYFRAMES[:align_len]

    for frame_id in range(0, len(depth_list), INFER_LEN):
        if len(depth_list_aligned) == 0:
            depth_list_aligned += depth_list[:INFER_LEN]
            for kf_id in kf_align_list:
                ref_align.append(depth_list[frame_id + kf_id])
        else:
            curr_align = []
            for i in range(len(kf_align_list)):
                curr_align.append(depth_list[frame_id + i])
            scale, shift = compute_scale_and_shift(
                np.concatenate(curr_align),
                np.concatenate(ref_align),
                np.concatenate(np.ones_like(ref_align) == 1),
            )

            pre_depth_list = depth_list_aligned[-INTERP_LEN:]
            post_depth_list = depth_list[frame_id + align_len:frame_id + OVERLAP]
            for i in range(len(post_depth_list)):
                post_depth_list[i] = post_depth_list[i] * scale + shift
                post_depth_list[i][post_depth_list[i] < 0] = 0
            depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

            for i in range(OVERLAP, INFER_LEN):
                new_depth = depth_list[frame_id + i] * scale + shift
                new_depth[new_depth < 0] = 0
                depth_list_aligned.append(new_depth)

            ref_align = ref_align[:1]
            for kf_id in kf_align_list[1:]:
                new_depth = depth_list[frame_id + kf_id] * scale + shift
                new_depth[new_depth < 0] = 0
                ref_align.append(new_depth)

    return depth_list_aligned
