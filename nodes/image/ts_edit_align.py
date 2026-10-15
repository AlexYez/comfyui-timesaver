"""TS Edit Align — put an edited image back exactly over its source.

node_id: TS_EditAlign

Edit models (Qwen Image Edit, Qwen Image 2.1, Flux Kontext...) redraw the whole
frame, and the result drifts: measured on Qwen Image 2.1 in the owner's
"Image Gen Edit Qwen 2" graph (15.10.2026), a 960x720 edit came back shifted by
2.8 px with a 0.5% stretch — 5 px at the frame edge; a tall 1100x1980 one by
3.4 px and 0.65% — 7 px. Even with the latent sized exactly like the reference
(the core node's own advice) the model kept a 0.3% zoom and a 2.7 px offset.
Wiring cannot remove that: it is the model's own error. So it is measured and
undone after the fact.

1. **Same frame.** The edit is resized to the source's exact size — the model
   works on a reference rounded to its grid, and only this undoes that
   rounding exactly (an independent stretch per axis is an affine map, and its
   inverse is exact). With ``output_size = edit`` it goes the other way: the
   source is stretched to the size the model rendered at, so a small photo is
   not squeezed back and the generated detail survives for an upscaler.
2. **Measure.** Luma of both, a grid of tiles, phase correlation per tile with
   a sub-pixel peak: a field of local offsets. An edit is local, so its tiles
   disagree with the rest; a robust fit (least squares, outliers dropped by
   their residual) gives scale and offset per axis from the tiles that did not
   change.
3. **Undo.** One bicubic warp of the full-resolution edit with the inverse
   map. Measured and applied twice — the second pass mops up what the first
   estimate left.

Nothing is moved when there is not enough agreeing texture to trust the fit
(a flat image, an edit that changed everything): the node says so instead of
guessing. Without a source it passes the image through, so the same graph
still does plain generation.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from comfy_api.v0_0_2 import IO, UI

logger = logging.getLogger("comfyui_timesaver.ts_edit_align")
LOG_PREFIX = "[TS Edit Align]"

#: Estimation runs on a copy whose long side is at most this; the warp is full size.
_WORK_SIDE = 1024
_TILE = 96
#: A tile counts only with a clear correlation peak (texture to lock onto).
_MIN_PEAK = 0.06
_MIN_TILES = 8
_PASSES = 2


def _luma(image: torch.Tensor) -> torch.Tensor:
    """[1, H, W, C] -> [1, 1, H, W] luma (Rec.709 on the first three channels)."""
    rgb = image[..., :3]
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device)
    if rgb.shape[-1] < 3:
        return rgb.mean(-1)[:, None]
    return (rgb * weights).sum(-1)[:, None]


def _resize(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """[N, C, H, W] bicubic with antialias."""
    if x.shape[-2:] == (h, w):
        return x
    return F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)


def _tile_offsets(ref: torch.Tensor, img: torch.Tensor, tile: int):
    """Phase correlation on a tile grid. ``ref``/``img``: [H, W].

    Returns rows (cx, cy, dx, dy, peak) where (dx, dy) is how far ``img``'s
    content sits from ``ref``'s at that tile: img(p + d) ~ ref(p).
    """
    h, w = ref.shape
    step = tile // 2
    ys = list(range(0, h - tile + 1, step))
    xs = list(range(0, w - tile + 1, step))
    if not ys or not xs:
        return torch.empty(0, 5)
    a = torch.stack([ref[y:y + tile, x:x + tile] for y in ys for x in xs])
    b = torch.stack([img[y:y + tile, x:x + tile] for y in ys for x in xs])
    win = torch.hann_window(tile, periodic=False, dtype=a.dtype, device=a.device)
    win = win[:, None] * win[None, :]
    a = (a - a.mean((1, 2), keepdim=True)) * win
    b = (b - b.mean((1, 2), keepdim=True)) * win
    cross = torch.fft.fft2(b) * torch.conj(torch.fft.fft2(a))
    cross = cross / (cross.abs() + 1e-9)
    corr = torch.fft.fftshift(torch.fft.ifft2(cross).real, dim=(1, 2))
    n = corr.shape[0]
    flat = corr.reshape(n, -1)
    peak, idx = flat.max(1)
    py, px = idx // tile, idx % tile

    def sub(c0, cm, cp):
        d = cm - 2 * c0 + cp
        return torch.where(d.abs() > 1e-12, 0.5 * (cm - cp) / d, torch.zeros_like(d))

    ar = torch.arange(n, device=corr.device)
    ym, yp = (py - 1).clamp(0, tile - 1), (py + 1).clamp(0, tile - 1)
    xm, xp = (px - 1).clamp(0, tile - 1), (px + 1).clamp(0, tile - 1)
    c0 = corr[ar, py, px]
    dy = (py - tile // 2).to(a.dtype) + sub(c0, corr[ar, ym, px], corr[ar, yp, px]).clamp(-0.5, 0.5)
    dx = (px - tile // 2).to(a.dtype) + sub(c0, corr[ar, py, xm], corr[ar, py, xp]).clamp(-0.5, 0.5)
    centres = torch.tensor([(x + tile / 2, y + tile / 2) for y in ys for x in xs], dtype=a.dtype, device=a.device)
    return torch.stack([centres[:, 0], centres[:, 1], dx, dy, peak], dim=1)


def _robust_fit(rows: torch.Tensor, w: int, h: int, limit: float):
    """Fit d = s * (p - centre) + t per axis on the tiles that agree.

    Returns (sx, tx, sy, ty, used) or None when too few tiles agree.
    """
    keep = (rows[:, 4] >= _MIN_PEAK) & (rows[:, 2].abs() <= limit) & (rows[:, 3].abs() <= limit)
    rows = rows[keep]
    if rows.shape[0] < _MIN_TILES:
        return None
    px = rows[:, 0] - w / 2
    py = rows[:, 1] - h / 2
    mask = torch.ones(rows.shape[0], dtype=torch.bool)
    fit = None
    for _ in range(4):
        if int(mask.sum()) < _MIN_TILES:
            return None
        ax = torch.stack([px[mask], torch.ones_like(px[mask])], 1)
        ay = torch.stack([py[mask], torch.ones_like(py[mask])], 1)
        sx, tx = torch.linalg.lstsq(ax, rows[mask, 2:3]).solution.flatten().tolist()
        sy, ty = torch.linalg.lstsq(ay, rows[mask, 3:4]).solution.flatten().tolist()
        res = torch.hypot(rows[:, 2] - (sx * px + tx), rows[:, 3] - (sy * py + ty))
        # Outliers: the edited tiles. Robust spread, floor at a quarter pixel.
        spread = max(0.25, 1.4826 * float(res[mask].median()))
        new_mask = res <= 3.0 * spread
        fit = (sx, tx, sy, ty, int(new_mask.sum()))
        if torch.equal(new_mask, mask):
            break
        mask = new_mask
    return fit


def _warp(image: torch.Tensor, sx: float, tx: float, sy: float, ty: float) -> torch.Tensor:
    """Pull the edit back: out(p) = edit(p + d(p)), d = s*(p - centre) + t. image [1, C, H, W]."""
    _, _, h, w = image.shape
    ys = torch.arange(h, dtype=torch.float32, device=image.device) + 0.5
    xs = torch.arange(w, dtype=torch.float32, device=image.device) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    src_x = gx + sx * (gx - w / 2) + tx
    src_y = gy + sy * (gy - h / 2) + ty
    grid = torch.stack([src_x / w * 2 - 1, src_y / h * 2 - 1], dim=-1)[None]
    out = F.grid_sample(image.float(), grid, mode="bicubic", padding_mode="border", align_corners=False)
    return out.clamp(0.0, 1.0).to(image.dtype)


def align_frame(edit: torch.Tensor, source: torch.Tensor, max_shift: float, output_size: str = "source"):
    """One frame. ``edit``/``source``: [1, H, W, C] (any sizes). Returns (aligned, report).

    ``output_size="source"`` brings the edit to the source's size; ``"edit"``
    keeps the size the model rendered at and stretches the SOURCE to it for
    the measurement instead, so no generated detail is thrown away.
    ``max_shift`` and the reported offsets are in pixels of the output frame.
    """
    if output_size == "edit":
        h, w = int(edit.shape[1]), int(edit.shape[2])
    else:
        h, w = int(source.shape[1]), int(source.shape[2])
    pic = _resize(edit.permute(0, 3, 1, 2).float(), h, w).clamp(0.0, 1.0)

    scale = min(1.0, _WORK_SIDE / max(h, w))
    wh, ww = max(1, round(h * scale)), max(1, round(w * scale))
    ref_small = _resize(_luma(source.float()), wh, ww)[0, 0]
    tile = min(_TILE, max(16, min(wh, ww) // 4 // 2 * 2))

    total = [0.0, 0.0, 0.0, 0.0]  # sx, tx, sy, ty in FULL-resolution pixels
    used = 0
    for _ in range(_PASSES):
        current = _warp(pic, *total) if any(total) else pic
        cur_small = _resize(_luma(current.permute(0, 2, 3, 1)), wh, ww)[0, 0]
        rows = _tile_offsets(ref_small, cur_small, tile)
        fit = _robust_fit(rows, ww, wh, max_shift * scale) if rows.numel() else None
        if fit is None:
            break
        sx, tx, sy, ty, used = fit
        # Compose: offsets measured at work scale; scale terms are unitless.
        total = [total[0] + sx, total[1] + tx / scale, total[2] + sy, total[3] + ty / scale]
    if used == 0:
        return pic.permute(0, 2, 3, 1).to(edit.dtype), None
    aligned = _warp(pic, *total)
    report = {"dx": total[1], "dy": total[3], "sx": total[0], "sy": total[2], "tiles": used}
    return aligned.permute(0, 2, 3, 1).to(edit.dtype), report


class TS_EditAlign(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_EditAlign",
            display_name="TS Edit Align",
            category="TS/Image/Retouch",
            description=(
                "Put an edited image back exactly over its source. Edit models (Qwen Image Edit, "
                "Qwen Image 2.1, Kontext) return the frame a few pixels off and slightly zoomed. "
                "This node brings the edit to the source's size, measures the drift on the parts "
                "that were not edited and undoes it with sub-pixel accuracy. Without a source the "
                "image passes through, so the same graph still generates from scratch."
            ),
            search_aliases=["edit align", "qwen edit shift", "align to source", "registration", "offset fix", "kontext shift"],
            inputs=[
                IO.Image.Input("image", tooltip="The edited image, straight from VAE Decode (a batch is aligned frame by frame)."),
                IO.Image.Input("source", optional=True, tooltip="The original the edit was made from. Not connected = pass through."),
                IO.Float.Input(
                    "max_shift", default=24.0, min=1.0, max=128.0, step=1.0,
                    tooltip="Largest drift (pixels of the output) the node will correct. Tiles moved further count as edited.",
                ),
                IO.Combo.Input(
                    "output_size", options=["source", "edit"], default="source",
                    tooltip=(
                        "source: the edit is brought to the original's size and lies on it pixel for pixel. "
                        "edit: keep the size the model rendered at (no generated detail lost) and align it to "
                        "the original stretched to that size — better before an upscaler, e.g. for small photos."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image", tooltip="The edit, in the original's position, at the chosen size."),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, source: Optional[torch.Tensor] = None, max_shift: float = 24.0,
                output_size: str = "source") -> IO.NodeOutput:
        output_size = output_size if output_size in ("source", "edit") else "source"
        if source is None:
            return IO.NodeOutput(image, ui=UI.PreviewText("no source connected - passed through"))
        if image.ndim != 4 or source.ndim != 4:
            raise ValueError(f"{LOG_PREFIX} Expected IMAGE [B, H, W, C] for both inputs.")
        frames, lines = [], []
        with torch.no_grad():
            for i in range(image.shape[0]):
                src = source[min(i, source.shape[0] - 1):min(i, source.shape[0] - 1) + 1]
                out, rep = align_frame(image[i:i + 1], src, float(max_shift), output_size)
                if out.shape[-1] != image.shape[-1]:
                    out = out[..., :image.shape[-1]]
                frames.append(out)
                if rep is None:
                    lines.append("not enough unchanged texture to measure - resized only")
                else:
                    edge = max(abs(rep["sx"]) * out.shape[2] / 2 + abs(rep["dx"]),
                               abs(rep["sy"]) * out.shape[1] / 2 + abs(rep["dy"]))
                    lines.append(
                        f"corrected shift {rep['dx']:+.2f}, {rep['dy']:+.2f} px, scale "
                        f"{rep['sx'] * 100:+.2f}% / {rep['sy'] * 100:+.2f}% (up to {edge:.1f} px at the edge), "
                        f"{rep['tiles']} tiles"
                    )
        result = torch.cat(frames, dim=0)
        text = lines[0] if len(set(lines)) == 1 else " | ".join(lines[:4])
        text += f" | {result.shape[2]}x{result.shape[1]} ({output_size} size)"
        logger.info("%s %s", LOG_PREFIX, text)
        return IO.NodeOutput(result, ui=UI.PreviewText(text))


NODE_CLASS_MAPPINGS = {"TS_EditAlign": TS_EditAlign}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_EditAlign": "TS Edit Align"}
