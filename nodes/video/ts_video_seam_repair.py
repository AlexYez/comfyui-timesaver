"""Repair persistent H3 decoder seams using the native grid and a synchronised draft."""

import logging

import numpy as np
import torch
from comfy_api.v0_0_2 import IO

from .ts_deflicker import _validate_images

_SECTION_SPAN = 32


def _luma(frame):
    if frame.shape[-1] == 1:
        return frame[..., 0]
    return frame[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _row_profile(frame):
    # Keep vertical resolution and measure independent areas across the image.
    samples = frame[:, :: max(1, frame.shape[1] // 384)].detach().float().cpu().numpy()
    edges = np.abs(np.diff(_luma(samples), axis=0))
    if samples.shape[-1] == 4:
        edges *= np.minimum(samples[:-1, :, 3], samples[1:, :, 3]).clip(0, 1)
    areas = np.stack([part.mean(axis=1) for part in np.array_split(edges, 8, axis=1)], axis=1)
    neighbours = np.mean([np.roll(areas, n, axis=0) for n in (-5, -4, -3, -2, 2, 3, 4, 5)], axis=0)
    # A near-flat background must not amplify numerical noise into a candidate.
    return (areas + 0.0005) / (neighbours + 0.0005)


def _axis_boundaries(size, vae=None):
    """Internal window boundaries in decoded pixels, clipped exactly as in H3."""
    model = getattr(vae, "first_stage_model", None)
    if vae is not None:
        if not hasattr(model, "noncausal") or not callable(getattr(model, "split_tiles", None)):
            raise ValueError("Connect the MiniMax H3 video VAE used for decoding.")
        if not model.noncausal:
            if not model.tiling:
                return []
            starts, lengths, _ = model.split_tiles(size)
            return sorted(
                {edge for start, length in zip(starts, lengths) for edge in (start, start + length) if 0 < edge < size}
            )
    # Sol/Fast V2 defaults. Read the actual geometry when the VAE is connected.
    tokens = model.decoder.window if model is not None else 16
    patch = model.decoder.patch_size if model is not None else 16
    period = tokens * patch
    shifted_start = (tokens - tokens // 2) * patch
    return sorted(set(range(period, size, period)) | set(range(shifted_start, size, period)))


def _find_rows(images, reference, vae=None):
    count, height, width = images.shape[:3]
    ref_height = reference.shape[1]
    if count < 4 or min(height, ref_height, width, reference.shape[2]) < 32:
        return []
    rows = np.array([r for r in _axis_boundaries(height, vae) if 8 <= r < height - 8], dtype=int)
    if not rows.size:
        return []
    ref_index = np.rint(rows * ref_height / height).astype(int) - 1
    scores = []
    broad_support = []
    for frame in np.linspace(0, count - 1, min(32, count), dtype=int):
        target = _row_profile(images[frame])[rows - 1]
        draft = _row_profile(reference[frame])
        # Allow for rounding and a small displacement of a real scene edge.
        natural = np.maximum.reduce([draft[(ref_index + d).clip(0, len(draft) - 1)] for d in (-2, -1, 0, 1, 2)])
        scores.append(np.median(target / np.maximum(1, natural), axis=1))
        broad_support.append(np.median(target, axis=1))
    # At least three quarters of the sampled frames and half the width must
    # support a line that is considerably stronger than the draft.
    confidence = np.quantile(scores, 0.25, axis=0)
    broad = np.quantile(broad_support, 0.25, axis=0)
    selected = []
    for index in np.argsort(confidence)[::-1]:
        # Positions already come from the decoder, requiring separate evidence
        # from the frames avoids treating every window edge as a visible seam.
        if confidence[index] < 1.15 or broad[index] < 1.35:
            continue
        row = int(rows[index])
        if all(abs(row - previous) >= 64 for previous in selected):
            selected.append(row)
    return sorted(selected)


def _section_profiles(frame, height, width, starts, span):
    import cv2

    pixels = frame.detach().float().cpu().numpy()
    luma = _luma(pixels)
    if luma.shape != (height, width):
        luma = cv2.resize(luma, (width, height), interpolation=cv2.INTER_LINEAR)
    edges = np.abs(np.diff(luma, axis=1))
    if pixels.shape[-1] == 4:
        alpha = cv2.resize(pixels[..., 3], (width, height), interpolation=cv2.INTER_LINEAR)
        edges *= np.minimum(alpha[:, :-1], alpha[:, 1:]).clip(0, 1)
    # Limit the influence of a single bright pixel crossing a section.
    np.minimum(edges, 0.12, out=edges)
    cumulative = np.cumsum(edges, axis=0, dtype=np.float32)
    sums = cumulative[starts + span - 1].copy()
    sums[1:] -= cumulative[starts[1:] - 1]
    return sums / span


def _section_scores(frame, reference, starts, span, columns):
    height, width = frame.shape[:2]
    source = _section_profiles(frame, height, width, starts, span)
    draft = _section_profiles(reference, height, width, starts, span)
    i = columns - 1
    background = np.mean([source[:, i + d] for d in (-5, -4, -3, -2, 2, 3, 4, 5)], axis=0)
    natural = np.maximum.reduce([draft[:, i + d] for d in (-2, -1, 0, 1, 2)])
    # Require both a sharp single-column step and more contrast than in the draft.
    return source[:, i] / np.maximum(2.0 * background + 0.004, 1.35 * natural + 0.004)


def _vertical_sections(images, reference, vae, whole_columns):
    import comfy.model_management as mm

    count, height, width = images.shape[:3]
    if count < 4 or min(height, width, *reference.shape[1:3]) < 32:
        return {}
    grid = set(_axis_boundaries(width, vae))
    model = getattr(vae, "first_stage_model", None)
    if model is not None and not model.noncausal and not model.tiling:
        return {}
    # Half-overlapping windows, including the remainder at the bottom.
    span = _SECTION_SPAN
    starts = np.unique(np.r_[np.arange(0, height - span + 1, span // 2), height - span])
    columns = np.arange(8, width - 8)
    sampled = np.linspace(0, count - 1, min(24, count), dtype=int)
    votes = np.zeros((len(starts), len(columns)), dtype=np.uint8)
    for index in sampled:
        mm.throw_exception_if_processing_interrupted()
        scores = _section_scores(images[index], reference[index], starts, span, columns)
        votes += scores >= 1.8
    # Outside the decoder grid require strong evidence at exactly the same
    # position in three quarters of the sample. Moving silhouettes fail this test.
    persistent = votes >= np.ceil(0.75 * len(sampled))
    on_grid = np.isin(columns, list(grid))
    keep = (on_grid | persistent.any(axis=0)) & ~np.isin(columns, whole_columns)
    if not keep.any():
        return {}
    columns, on_grid, persistent = columns[keep], on_grid[keep], persistent[:, keep]
    weights = np.empty((count, len(starts), len(columns)), dtype=np.float32)
    for index, (frame, draft) in enumerate(zip(images, reference)):
        mm.throw_exception_if_processing_interrupted()
        scores = _section_scores(frame, draft, starts, span, columns)
        weights[index] = ((scores - 1.0) / 0.7).clip(0, 1)
    # An on-grid seam can become obvious only as an object crosses the
    # boundary. Require recurrence, then correct according to each frame.
    supported = (on_grid & ((weights >= 0.5).sum(axis=0) >= 2)) | persistent
    weights *= supported
    centers = starts + (span - 1) / 2
    return {int(col): (centers, weights[:, :, j]) for j, col in enumerate(columns) if weights[:, :, j].any()}


def _section_blend(pixels, guide, boundary, centers, weights):
    import cv2

    # A section score describes 32 pixels, including the ends of the window.
    # Interpolating only its centers undercorrects T-junctions. Extend the
    # weight there only if a finer measurement confirms the step.
    along = np.interp(np.arange(pixels.shape[1]), centers, weights).astype(np.float32)
    envelope = cv2.dilate(along[:, None], np.ones((_SECTION_SPAN + 1, 1), dtype=np.uint8))[:, 0]

    def gradients(strip):
        sample = _luma(strip[boundary - 5 : boundary + 5])
        edges = np.abs(np.diff(sample, axis=0))
        return cv2.blur(edges, (5, 1), borderType=cv2.BORDER_REPLICATE)

    edges = gradients(pixels)
    nearby = np.median(edges[[0, 1, 2, 6, 7, 8]], axis=0)
    natural = np.max(gradients(guide)[3:6], axis=0)
    floor = np.maximum(1.7 * nearby, 1.3 * natural) + 0.002
    confidence = ((edges[4] / floor - 1) / 0.65).clip(0, 1)
    return np.maximum(along, envelope * confidence)


def _repair_strip(target, reference, row, flow_solver, section=None):
    import cv2

    height, width = target.shape[:2]
    top, end = max(0, row - 80), min(height, row + 80)
    strip = target[top:end].detach().float().cpu().numpy()
    ref_h, ref_w = reference.shape[:2]
    rt, rb = top * ref_h // height, (end * ref_h + height - 1) // height
    ref = reference[rt:rb].detach().float().cpu().numpy()
    # Sample the reference in target coordinates, including fractional scale.
    y, x = np.mgrid[top:end, :width].astype(np.float32)
    rx = (x + 0.5) * (ref_w / width) - 0.5
    ry = (y + 0.5) * (ref_h / height) - 0.5 - rt
    guide = cv2.remap(ref, rx, ry, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if guide.ndim == 2:
        guide = guide[..., None]

    def flow_input(pixels):
        luma = _luma(pixels)
        if pixels.shape[-1] == 4:
            luma = luma * pixels[..., 3]
        return cv2.resize(
            (luma.clip(0, 1) * 255).astype(np.uint8), (width // 2, len(strip) // 2), interpolation=cv2.INTER_AREA
        )

    flow = flow_solver.calc(flow_input(strip), flow_input(guide), None)
    flow = cv2.resize(cv2.GaussianBlur(flow, (0, 0), 3), (width, len(strip)))
    guide = cv2.remap(
        ref,
        rx + flow[..., 0] * (ref_w / (width // 2)),
        ry + flow[..., 1] * (ref_h / height * len(strip) / (len(strip) // 2)),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if guide.ndim == 2:
        guide = guide[..., None]
    guide_rgb = guide[..., :3] if strip.shape[-1] != 1 else _luma(guide)[..., None]
    pixels = strip[..., :3]
    error = pixels - guide_rgb
    distance = np.abs(y[:, :1] + 0.5 - row)

    def feather(flat, outer):
        return (0.5 + 0.5 * np.cos(np.pi * ((distance - flat) / (outer - flat)).clip(0, 1)))[..., None]

    def correction(sigma):
        band = cv2.GaussianBlur(error, (0, 1), sigma)
        delta = cv2.GaussianBlur(band, (1, 0), 0, sigmaY=12) - band
        return delta.reshape(error.shape)

    narrow = feather(1, 7)
    weight = feather(5, 32)
    if strip.shape[-1] == 4:
        weight = weight * strip[..., 3:4].clip(0, 1)
    if guide.shape[-1] == 4:
        weight = weight * guide[..., 3:4].clip(0, 1)
    delta = (correction(2.4) * (1 - narrow) + correction(0.7) * narrow) * weight
    if section is not None:
        along = _section_blend(pixels, guide_rgb, row - top, *section)
        delta *= along[None, :, None]
    start, stop = max(0, row - 32), min(height, row + 32)
    region = slice(start - top, stop - top)
    corrected = pixels[region] + delta[region]
    return start, stop, torch.from_numpy(corrected)


def _finish_horizontal(frame, original, reference, row):
    """Bridge a confirmed residual after both axes have been composed."""
    import cv2

    height, width = frame.shape[:2]
    radius = 5
    band = frame[row - radius : row + radius]
    pixels = band.detach().float().cpu().numpy()
    source = original[row - radius : row + radius].detach().float().cpu().numpy()
    ref_h, ref_w = reference.shape[:2]
    y, x = np.mgrid[row - radius : row + radius, :width].astype(np.float32)
    ry = (y + 0.5) * (ref_h / height) - 0.5
    top = max(0, int(np.floor(ry.min())))
    end = min(ref_h, int(np.floor(ry.max())) + 2)
    guide = cv2.remap(
        reference[top:end].detach().float().cpu().numpy(),
        (x + 0.5) * (ref_w / width) - 0.5, ry - top, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if guide.ndim == 2:
        guide = guide[..., None]
    center = radius - 1
    # Compare the initial step with local texture and the strongest nearby
    # draft edge before deciding whether the final strip needs adjustment.
    def edge_profile(strip):
        edges = np.abs(np.diff(_luma(strip), axis=0))
        if strip.shape[-1] == 4:
            edges *= np.minimum(strip[:-1, :, 3], strip[1:, :, 3]).clip(0, 1)
        return edges

    edges = edge_profile(source)
    texture = np.median(edges[[0, 1, 2, 6, 7, 8]], axis=0)
    natural = cv2.dilate(edge_profile(guide), np.ones((3, 3), np.uint8))[center]
    step, texture, natural = cv2.blur(np.stack((edges[center], texture, natural)), (9, 1))
    strength = ((step - 0.0015 - np.maximum(texture * 1.25, natural * 1.1)) / 0.014).clip(0, 1)
    if not strength.any():
        return
    rgb = pixels[..., :3]
    t = np.linspace(0, 1, len(band), dtype=np.float32)[:, None, None]
    bridge = rgb[:1] * (1 - t) + rgb[-1:] * t
    # Keep fine texture across the width and fade into both unmodified ends.
    delta = bridge - rgb
    weight = np.sqrt(np.maximum(0, np.sin(np.pi * t))) * strength[None, :, None]
    for visible in (pixels, guide):
        if visible.shape[-1] == 4:
            # Hidden RGB at either anchor must not bleed into a visible row.
            anchors = np.minimum(visible[:1, :, 3:4], visible[-1:, :, 3:4])
            opacity = np.minimum(visible[..., 3:4], anchors).clip(0, 1)
            delta *= opacity
            weight *= opacity
    delta = cv2.GaussianBlur(delta, (5, 1), 0.9).reshape(rgb.shape)
    band[..., :3].copy_(torch.from_numpy(rgb + delta * weight))


class TS_VideoSeamRepair(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TS_VideoSeamRepair",
            display_name="TS Video Seam Repair",
            category="TS/Video",
            description="Automatically repair visible MiniMax H3 decoder grid lines using a clean draft.",
            inputs=[
                IO.Image.Input("images", tooltip="Decoded upscale frames, in order."),
                IO.Image.Input("reference", tooltip="Same frames before upscale. Match the framing and frame count."),
                IO.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="Connect the decoding H3 VAE for its exact grid. Without it, use the Fast V2 window geometry.",
                ),
            ],
            outputs=[IO.Image.Output("images")],
        )

    @classmethod
    def execute(cls, images, reference, vae=None):
        for batch, name in ((images, "images"), (reference, "reference")):
            _validate_images(batch, name)
        if len(images) != len(reference):
            raise ValueError("images and reference must contain the same number of frames.")
        aspect = images.shape[2] * reference.shape[1] / (images.shape[1] * reference.shape[2])
        if not 0.98 <= aspect <= 1.02:
            raise ValueError("images and reference must have matching aspect ratios.")
        rows = _find_rows(images, reference, vae)
        columns = _find_rows(images.transpose(1, 2), reference.transpose(1, 2), vae)
        sections = _vertical_sections(images, reference, vae, columns)
        if not rows and not columns and not sections:
            return IO.NodeOutput(images)
        import comfy.model_management as mm
        import cv2

        logging.getLogger(__name__).info(
            "[TS Video Seam Repair] Correcting rows %s, columns %s, partial columns %s", rows, columns, list(sections)
        )
        solver = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        output = images.clone()
        for index, frame in enumerate(images):
            mm.throw_exception_if_processing_interrupted()
            for row in rows:
                start, stop, corrected = _repair_strip(frame, reference[index], row, solver)
                output[index, start:stop, :, :3].copy_(corrected)
            for col in columns:
                # Preserve the horizontal correction at band intersections.
                vertical = output[index].transpose(0, 1)
                start, stop, corrected = _repair_strip(vertical, reference[index].transpose(0, 1), col, solver)
                vertical[start:stop, :, :3].copy_(corrected)
            for col, (centers, weights) in sections.items():
                if not weights[index].any():
                    continue
                vertical = output[index].transpose(0, 1)
                start, stop, corrected = _repair_strip(
                    vertical, reference[index].transpose(0, 1), col, solver, (centers, weights[index])
                )
                vertical[start:stop, :, :3].copy_(corrected)
            for row in rows:
                _finish_horizontal(output[index], frame, reference[index], row)
        return IO.NodeOutput(output)


NODE_CLASS_MAPPINGS = {"TS_VideoSeamRepair": TS_VideoSeamRepair}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_VideoSeamRepair": "TS Video Seam Repair"}
