"""Reduce exposure flicker in ordered frames without temporal pixel blending."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.v0_0_2 import IO


logger = logging.getLogger(__name__)
_LUMA = (0.2126, 0.7152, 0.0722)
_MIN_CORRECTION = 0.0015
_MAX_CORRECTION = 0.06
_MIN_SUPPORT = 0.60


def _thumbnails(images, grid):
    """Measure small regions with bounded working memory, excluding hidden RGB."""
    height, width, channels = images.shape[1:]
    rows, columns = grid
    features, visible = [], []
    for frames in images.split(32):
        pixels = frames[:, ::max(1, height // 144), ::max(1, width // 96)].float().movedim(-1, 1)
        rgb = pixels[:, :3].expand(-1, 3, -1, -1) if channels == 1 else pixels[:, :3]
        if channels == 4:
            alpha = pixels[:, 3:4].clamp(0, 1)
            weight = F.adaptive_avg_pool2d(alpha, (rows, columns))
            tiles = F.adaptive_avg_pool2d(rgb * alpha, (rows, columns)) / weight.clamp_min(1e-6)
            visible.append((weight.flatten(1) >= 0.5).cpu().numpy())
        else:
            tiles = F.adaptive_avg_pool2d(rgb, (rows, columns))
            visible.append(np.ones((len(frames), rows * columns), dtype=bool))
        features.append(tiles.flatten(2).transpose(1, 2).cpu().numpy())
    tiles = np.concatenate(features)
    return tiles, np.concatenate(visible) & np.isfinite(tiles).all(axis=2)


def _shot_ranges(tiles, visible):
    """Split edits by changes of spatial structure or strong unreferenced flashes."""
    start = 0
    for frame in range(1, len(tiles)):
        valid = visible[frame - 1] & visible[frame]
        if np.any(valid):
            difference = tiles[frame, valid] - tiles[frame - 1, valid]
            neutral = float(np.median(difference @ _LUMA))
            structural = np.max(np.abs(difference - neutral), axis=1)
            if abs(neutral) >= 0.14 or np.quantile(structural, 0.75) >= 0.16:
                yield start, frame
                start = frame
    yield start, len(tiles)


def _seam_ranges(segments_info, frames):
    """Convert validated diffusion windows to disjoint [start, end) ranges."""
    if not isinstance(segments_info, (list, tuple)):
        raise ValueError("segments_info must be the list of windows from TS Latent Upscale.")
    ranges = []
    previous_start = -1
    previous_end = 0
    for segment in segments_info:
        if not isinstance(segment, Mapping):
            raise ValueError("Each segments_info entry must describe a window.")
        start, length = segment.get("frame_start"), segment.get("frame_count")
        if type(start) is not int or type(length) is not int or length <= 0:
            raise ValueError("segments_info requires integer frame_start and positive frame_count.")
        end = start + length
        if start <= previous_start or start > previous_end or end <= previous_end or end > frames:
            raise ValueError("segments_info does not match the timeline of the input images.")
        if previous_end:
            left, right = max(0, start - 2), min(frames, previous_end + 4)
            if ranges and ranges[-1][1] >= left:
                left, _ = ranges.pop()
            ranges.append((left, right))
        previous_start, previous_end = start, end
    if segments_info and previous_end != frames:
        raise ValueError("segments_info must describe the entire input timeline.")
    return ranges


def _common_offset(candidate, valid):
    values = candidate[valid & np.isfinite(candidate)]
    if values.size < min(4, candidate.size):
        return 0.0
    offset = np.median(values)
    if abs(offset) < _MIN_CORRECTION:
        return 0.0
    support = np.mean(np.sign(offset) * values >= max(_MIN_CORRECTION / 2, abs(offset) / 4))
    if support < _MIN_SUPPORT:
        return 0.0
    return float(np.clip(offset, -_MAX_CORRECTION, _MAX_CORRECTION))


def _offsets(values, valid, shots, seams, radius, guided):
    changes = np.zeros(len(values), dtype=np.float32)
    for shot_start, shot_end in shots:
        areas = [(shot_start, shot_end)] if seams is None else seams
        for left, right in areas:
            start, end = max(left, shot_start), min(right, shot_end)
            if end - start < 3:
                continue
            if seams is not None and guided:
                # Keep the endpoints. Only smooth the residual after subtracting
                # the reference, so its intended exposure changes pass through.
                t = np.linspace(0, 1, end - start, dtype=np.float32)[:, None]
                weight = t * t * (3 - 2 * t)
                trend = values[start] + weight * (values[end - 1] - values[start])
                trusted = valid[start] & valid[end - 1]
                for frame in range(start + 1, end - 1):
                    changes[frame] = _common_offset(trend[frame - start] - values[frame], trusted & valid[frame])
            else:
                for frame in range(max(start, shot_start + 1), min(end, shot_end - 1)):
                    span = min(radius, frame - shot_start, shot_end - frame - 1)
                    neighborhood = slice(frame - span, frame + span + 1)
                    trusted = valid[neighborhood].all(axis=0)
                    target = np.median(values[neighborhood], axis=0)
                    changes[frame] = _common_offset(target - values[frame], trusted & valid[frame])
    return changes


def _validate_images(images, name):
    if (not isinstance(images, torch.Tensor) or images.ndim != 4 or not images.is_floating_point()
            or images.shape[-1] not in (1, 3, 4) or not min(images.shape[1:3])):
        raise ValueError(f"{name} must be an IMAGE batch [frames, height, width, 1/3/4].")


class TS_Deflicker(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Deflicker",
            display_name="TS Deflicker",
            category="TS/Video",
            description="Reduce brief exposure flicker in ordered frames. Optional window metadata and "
                        "aligned reference frames keep correction local and preserve intended lighting.",
            search_aliases=["deflicker", "exposure stabilization", "flicker removal", "chunk seams"],
            inputs=[
                IO.Image.Input("images", tooltip="Ordered frames from a video or image sequence."),
                IO.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Amount of exposure correction. Zero passes frames through."),
                IO.Int.Input("radius", default=5, min=1, max=30,
                             tooltip="Neighboring frames on either side for short flicker detection. "
                                     "Guided seam correction uses the window ranges instead."),
                IO.Custom("DICT").Input("segments_info", optional=True,
                                        tooltip="Frame windows from TS Latent Upscale. Limits correction to seams."),
                IO.Image.Input("reference", optional=True,
                               tooltip="Aligned draft with the same frames and aspect ratio. Its lighting changes "
                                       "are preserved. Draft resolution may be lower."),
            ],
            outputs=[IO.Image.Output("images", tooltip="Corrected frames in their original order.")],
        )

    @classmethod
    @torch.inference_mode()
    def execute(cls, images, strength=1.0, radius=5, segments_info=None, reference=None) -> IO.NodeOutput:
        _validate_images(images, "images")
        if not math.isfinite(strength) or not 0 <= strength <= 1 or not 1 <= radius <= 30:
            raise ValueError("strength must be between 0 and 1 and radius between 1 and 30.")
        if len(images) < 3 or strength == 0:
            return IO.NodeOutput(images)
        seams = None if segments_info is None else _seam_ranges(segments_info, len(images))
        if seams == []:
            return IO.NodeOutput(images)

        if reference is not None:
            _validate_images(reference, "reference")
            if len(reference) != len(images):
                raise ValueError("reference must have the same frame count as images.")
            cross_aspect = reference.shape[2] * images.shape[1] / (images.shape[2] * reference.shape[1])
            if not 0.95 <= cross_aspect <= 1.05:
                raise ValueError("reference must have the same aspect ratio as images.")
        height, width = images.shape[1:3]
        if reference is not None:
            height = min(height, reference.shape[1])
            width = min(width, reference.shape[2])
        grid = (min(12, height), min(8, width))
        tiles, visible = _thumbnails(images, grid)
        values = tiles @ _LUMA
        if reference is None:
            guide, guide_visible = tiles, visible
        else:
            guide, guide_visible = _thumbnails(reference, grid)
            values -= guide @ _LUMA
        changes = _offsets(values, visible & guide_visible, _shot_ranges(guide, guide_visible),
                           seams, int(radius), reference is not None) * strength
        active = np.flatnonzero(changes)
        if not active.size:
            return IO.NodeOutput(images)

        output = images.clone()
        channels = min(images.shape[-1], 3)
        for frame in active:
            pixels = output[frame, :, :, :channels]
            # Taper at black and white and keep RGB inside gamut with the same
            # additive shift on all channels. Copying alpha leaves it exact.
            low, high = pixels.amin(dim=-1), pixels.amax(dim=-1)
            weight = torch.minimum(low, 1 - high).mul_(8).clamp_(0, 1)
            if images.shape[-1] == 4:
                weight.mul_(output[frame, :, :, 3] > 0)
            pixels.add_(weight.unsqueeze(-1) * float(changes[frame]))
        logger.info("[TS Deflicker] Corrected %d of %d frames.", len(active), len(images))
        return IO.NodeOutput(output)


NODE_CLASS_MAPPINGS = {"TS_Deflicker": TS_Deflicker}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Deflicker": "TS Deflicker"}
