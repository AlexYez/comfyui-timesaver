"""Fixed-size, keyframed video crops and their reversible frame geometry."""

from __future__ import annotations

import bisect
import json
import math


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Video crop: {name} must be a finite number.")
    return float(value)


def parse_crop(raw):
    if not raw:
        return {"enabled": False}
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError) as exc:
        raise ValueError("Video crop: invalid saved crop data. Reset the crop in the fullscreen editor.") from exc
    if not isinstance(data, dict):
        raise ValueError("Video crop: expected an object.")
    if not data.get("enabled", False):
        return {"enabled": False}
    if data.get("version") != 1:
        raise ValueError("Video crop: unsupported crop version.")
    w = _finite(data.get("width"), "width")
    h = _finite(data.get("height"), "height")
    if not 0 < w <= 1 or not 0 < h <= 1:
        raise ValueError("Video crop: width and height must be between 0 and 1.")
    multiple = data.get("multiple", 32)
    if type(multiple) is not int or not 1 <= multiple <= 256:
        raise ValueError("Video crop: multiple must be an integer from 1 to 256.")
    interpolation = data.get("interpolation", "smooth")
    if interpolation not in ("linear", "smooth"):
        raise ValueError("Video crop: unknown interpolation.")

    def position(point):
        return (max(0., min(1. - w, _finite(point.get("x"), "x"))),
                max(0., min(1. - h, _finite(point.get("y"), "y"))))

    x, y = position(data)
    keys = data.get("keyframes", [])
    if not isinstance(keys, list) or len(keys) > 10000:
        raise ValueError("Video crop: invalid keyframe list (maximum 10000).")
    by_time = {}
    for key in keys:
        if not isinstance(key, dict):
            raise ValueError("Video crop: invalid keyframe.")
        t = _finite(key.get("time"), "keyframe time")
        if t < 0:
            raise ValueError("Video crop: keyframe time cannot be negative.")
        kx, ky = position(key)
        by_time[t] = {"time": t, "x": kx, "y": ky}
    return {"version": 1, "enabled": True, "width": w, "height": h,
            "x": x, "y": y, "multiple": multiple, "interpolation": interpolation,
            "keyframes": [by_time[t] for t in sorted(by_time)]}


def crop_position(config, seconds):
    keys = config["keyframes"]
    if not keys:
        return config["x"], config["y"]
    times = [key["time"] for key in keys]
    right = bisect.bisect_right(times, seconds)
    if right == 0:
        return keys[0]["x"], keys[0]["y"]
    if right >= len(keys):
        return keys[-1]["x"], keys[-1]["y"]
    a, b = keys[right - 1], keys[right]
    u = (seconds - a["time"]) / (b["time"] - a["time"])
    if config["interpolation"] == "smooth":
        u = u * u * (3. - 2. * u)
    return a["x"] + (b["x"] - a["x"]) * u, a["y"] + (b["y"] - a["y"]) * u


def crop_geometry(config, width, height, seconds):
    step = config["multiple"]
    pixels_w = max(1, math.floor(config["width"] * width + .5))
    pixels_h = max(1, math.floor(config["height"] * height + .5))
    cw = ((pixels_w + step - 1) // step) * step
    ch = ((pixels_h + step - 1) // step) * step
    nx, ny = crop_position(config, seconds)
    x = min(max(0, width - cw), max(0, math.floor((nx + config["width"] / 2) * width - cw / 2 + .5)))
    y = min(max(0, height - ch), max(0, math.floor((ny + config["height"] / 2) * height - ch / 2 + .5)))
    return {"x": x, "y": y, "width": min(cw, width), "height": min(ch, height),
            "output_width": cw, "output_height": ch, "time": float(seconds)}


def _allocate(shape, allow_disk, resident_bytes, tag):
    from ._decode import _allocate_frames, _max_bytes

    amount = math.prod(shape) * 4
    on_disk = resident_bytes + amount > _max_bytes()
    if on_disk and not allow_disk:
        raise RuntimeError(
            "Video crop: full frames, crop and restored output exceed the video memory "
            "limit. Trim the clip, reduce the full-frame resolution, or choose 'use disk' "
            "in TS Video Loader."
        )
    return _allocate_frames(shape, on_disk, tag)


def crop_frames(images, raw, frame_times, *, start_time=0., fps=0., allow_disk=False):
    import torch.nn.functional as F

    config = parse_crop(raw)
    n, h, w, channels = images.shape
    if len(frame_times) != n:
        if config.get("keyframes"):
            raise ValueError("Video crop: decoded frame timestamps are missing; animation cannot be aligned safely.")
        frame_times = [start_time + (i / fps if fps > 0 else 0.) for i in range(n)]
    if config["enabled"]:
        frames = [crop_geometry(config, w, h, when) for when in frame_times]
    else:
        frames = [{"x": 0, "y": 0, "width": w, "height": h,
                   "output_width": w, "output_height": h, "time": float(t)} for t in frame_times]
    data = {"kind": "ts_video_crop", "version": 1, "enabled": config["enabled"],
            "full_width": w, "full_height": h, "frame_count": n,
            "allow_disk": bool(allow_disk), "frames": frames}
    if not config["enabled"]:
        return images, data
    if not frames:
        raise ValueError("Video crop: no decoded frames.")
    cw, ch = frames[0]["output_width"], frames[0]["output_height"]
    output = _allocate((n, ch, cw, channels), allow_disk,
                       images.numel() * images.element_size(), "crop")
    from comfy.model_management import throw_exception_if_processing_interrupted

    for i, region in enumerate(frames):
        throw_exception_if_processing_interrupted()
        x, y, rw, rh = (region[key] for key in ("x", "y", "width", "height"))
        piece = images[i, y:y + rh, x:x + rw]
        if rh < ch or rw < cw:
            # Padding, never a stretch: remove these pixels again on restore.
            piece = F.pad(piece.permute(2, 0, 1).unsqueeze(0),
                          (0, cw - rw, 0, ch - rh), mode="replicate")[0].permute(1, 2, 0)
        output[i].copy_(piece)
    return output, data


def restore_video_crop(original, processed, data, blur, device, blur_mask):
    """Restore one frame at a time; never move the full original clip to VRAM."""
    import torch
    import torch.nn.functional as F
    from comfy.model_management import throw_exception_if_processing_interrupted

    from ...image._alpha import resize_with_alpha, to_channels
    from ...image._inpaint_crop import blend_pixels

    n, h, w, channels = original.shape
    if data.get("version") != 1 or data.get("frame_count") != n or len(data.get("frames", [])) != n:
        raise ValueError("Video crop restore: crop geometry does not match the original frame count.")
    if data.get("full_width") != w or data.get("full_height") != h:
        raise ValueError("Video crop restore: use full_images from the same TS Video Loader without resizing.")
    if len(processed) != n:
        raise ValueError(
            f"Video crop restore: expected {n} processed frames, got {len(processed)}. "
            "The model must preserve frame count and order."
        )
    if processed.shape[-1] == 4 and channels in (1, 3):
        channels = 4
    output = _allocate((n, h, w, channels), bool(data.get("allow_disk")),
                       original.numel() * original.element_size() + processed.numel() * processed.element_size(),
                       "restore")
    masks = {}
    for i, region in enumerate(data["frames"]):
        throw_exception_if_processing_interrupted()
        x, y, rw, rh, cw, ch = (region[key] for key in
                               ("x", "y", "width", "height", "output_width", "output_height"))
        if (min(x, y) < 0 or min(rw, rh, cw, ch) < 1 or x + rw > w or y + rh > h
                or rw > cw or rh > ch):
            raise ValueError("Video crop restore: invalid crop bounds.")
        source = to_channels(original[i], channels)
        base = source[y:y + rh, x:x + rw].to(device)
        piece = to_channels(processed[i].to(device), 3 if channels == 4 else channels)
        if piece.shape[-1] > channels:
            piece = piece[..., :channels]
        elif piece.shape[-1] < channels and channels != 4:
            piece = torch.cat((piece, piece.new_ones((*piece.shape[:2], channels - piece.shape[-1]))), -1)
        if tuple(piece.shape[:2]) != (ch, cw):
            piece = resize_with_alpha(piece[None], cw, ch, "bilinear")[0]
        piece = piece[:rh, :rw]
        if channels == 4 and piece.shape[-1] == 3:
            piece = torch.cat((piece, base[..., 3:4]), dim=-1)
        # Edge blur stays inside the patch, and does not fade at frame borders.
        feather = min(max(0, int(blur)), max(0, (min(rw, rh) - 1) // 4))
        edges = (x > 0, y > 0, x + rw < w, y + rh < h)
        key = (rw, rh, feather, edges)
        if key not in masks:
            mask = torch.ones((rh, rw), device=device)
            if feather and any(edges):
                left, top, right, bottom = [feather if edge else 0 for edge in edges]
                mask[:top] = 0
                if bottom:
                    mask[-bottom:] = 0
                mask[:, :left] = 0
                if right:
                    mask[:, -right:] = 0
                # Replicate padding preserves alpha=1 at touching frame edges.
                padded = F.pad(mask[None, None], (feather,) * 4, mode="replicate")[0]
                mask = blur_mask(padded, feather, device)[0, feather:-feather, feather:-feather]
            masks[key] = mask[..., None]
        alpha = masks[key]
        mixed = blend_pixels(base, piece, alpha)
        blended = torch.where(alpha >= 1, piece, torch.where(alpha <= 0, base, mixed))
        blended = torch.where(piece == base, base, blended)
        output[i].copy_(source)
        output[i, y:y + rh, x:x + rw].copy_(blended.to(device="cpu", dtype=output.dtype))
    return output
