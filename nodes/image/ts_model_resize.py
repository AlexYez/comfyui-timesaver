"""TS Model Resize — fit an image to the resolution a model was trained on.

node_id: TS_ModelResize

Pick the model, feed any picture: the node reads its orientation and aspect,
chooses the model's own resolution that matches it best and brings the picture
there — cropped, padded or stretched. Two kinds of presets:

* **Bucket** — the model ships a fixed list of sizes (Wan, Qwen Image, SDXL,
  Z-Image). The bucket with the nearest aspect ratio wins; a vertical frame
  lands on a vertical bucket by construction.
* **Area** — the model takes any aspect at a pixel budget, on a grid (Flux,
  LTX, HunyuanVideo). The picture keeps its own aspect at that area, rounded
  to the grid, so only a few pixels are ever cropped.

Sources of the tables (checked 15.10.2026):

* Wan 2.1 / 2.2 — ``SUPPORTED_SIZES`` in Wan-Video/Wan2.2 ``wan/configs``:
  A14B 1280×720 / 832×480 (both orientations), TI2V-5B 1280×704.
* Qwen Image — the ``aspect_ratios`` dict in QwenLM/Qwen-Image README. ⚠️ 4:3
  is 1472×1104; many community nodes say 1472×1140, which is not the table.
* SDXL — the bucket table of the SDXL report (arXiv 2307.01952), 1:1 to ~2.4:1.
* Z-Image Turbo — the 1024 tier of its official Hugging Face Space.
* SD 1.5 — trained at 512; 512×512 and 768×512 are the sizes it holds together at.
* FLUX.1 / FLUX.2 — multiples of 16 around 1 MP (BFL: 16 px step, ≤2 MP).
* LTX-2 / 2.5 — multiples of 32. HunyuanVideo 1.5 — 720p, multiples of 16.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from comfy_api.v0_0_2 import IO, UI

logger = logging.getLogger("comfyui_timesaver.ts_model_resize")
LOG_PREFIX = "[TS Model Resize]"


@dataclass(frozen=True)
class ModelPreset:
    name: str
    #: Landscape-or-square sizes (w >= h); portrait twins are added automatically.
    buckets: tuple[tuple[int, int], ...] = ()
    #: Area mode: pixel budget and grid. Used when ``buckets`` is empty.
    area: int = 0
    multiple: int = 16


def _sdxl() -> tuple[tuple[int, int], ...]:
    return (
        (1024, 1024), (1088, 960), (1088, 896), (1152, 896), (1152, 832), (1216, 832),
        (1280, 768), (1344, 768), (1408, 704), (1472, 704), (1536, 640),
    )


# ⚠️ The NAMES are workflow data (a saved graph stores the chosen preset by
# name): add new ones freely, never rename or reorder the existing ones.
PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset("Wan 2.1 / 2.2 14B — 720p", buckets=((1280, 720),)),
    ModelPreset("Wan 2.1 / 2.2 14B — 480p", buckets=((832, 480),)),
    ModelPreset("Wan 2.2 TI2V 5B — 720p", buckets=((1280, 704),)),
    ModelPreset("Qwen Image", buckets=((1328, 1328), (1664, 928), (1472, 1104), (1584, 1056))),
    ModelPreset("SDXL / Illustrious / Pony", buckets=_sdxl()),
    ModelPreset("SD 1.5", buckets=((512, 512), (768, 512))),
    ModelPreset("Z-Image Turbo", buckets=(
        (1024, 1024), (1152, 896), (1152, 864), (1248, 832), (1280, 720), (1344, 576),
    )),
    ModelPreset("FLUX.1 / FLUX.2 — 1 MP", area=1024 * 1024, multiple=16),
    ModelPreset("FLUX.2 — 2 MP", area=1920 * 1088, multiple=16),
    ModelPreset("LTX-2 / 2.5 — 720p", area=1280 * 704, multiple=32),
    ModelPreset("HunyuanVideo 1.5 — 720p", area=1280 * 720, multiple=16),
)
PRESET_BY_NAME = {p.name: p for p in PRESETS}

FITS = ("crop", "pad", "stretch")
ANCHORS = ("center", "top", "bottom", "left", "right")
PAD_COLORS = {"black": 0.0, "white": 1.0, "gray": 0.5}
PAD_OPTIONS = ("black", "white", "gray", "edge")


def all_buckets(preset: ModelPreset) -> list[tuple[int, int]]:
    """Every bucket of a preset in both orientations, without duplicates."""
    out: list[tuple[int, int]] = []
    for w, h in preset.buckets:
        for size in ((w, h), (h, w)):
            if size not in out:
                out.append(size)
    return out


def target_size(preset: ModelPreset, width: int, height: int) -> tuple[int, int]:
    """The model resolution for a picture of this size.

    Aspect ratios are compared in log space: 2:1 and 1:2 are equally far from
    1:1, as the eye judges them.
    """
    aspect = math.log(max(width, 1) / max(height, 1))
    if preset.buckets:
        return min(all_buckets(preset), key=lambda s: (abs(math.log(s[0] / s[1]) - aspect), -s[0] * s[1]))
    m = preset.multiple
    ratio = width / max(height, 1)
    w = math.sqrt(preset.area * ratio)
    h = w / ratio
    return max(m, int(round(w / m)) * m), max(m, int(round(h / m)) * m)


def _resize(frame: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """[1, C, H, W] -> [1, C, h, w]; antialiased bicubic, clamped to 0..1."""
    w, h = size
    if frame.shape[-2:] == (h, w):
        return frame
    out = F.interpolate(frame, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    return out.clamp(0.0, 1.0)


def _offset(free: int, anchor: str, start: str, end: str) -> int:
    if anchor == start:
        return 0
    if anchor == end:
        return free
    return free // 2


def fit_frame(frame: torch.Tensor, size: tuple[int, int], fit: str, anchor: str, pad: str):
    """One frame [1, C, H, W] into ``size``; returns (frame, content_mask [1, h, w]).

    The mask is 1 where the picture is and 0 on padding — wire it to an
    outpaint to fill the bars instead of leaving them flat.
    """
    w, h = size
    src_h, src_w = frame.shape[-2:]
    full = torch.ones((1, h, w), dtype=frame.dtype, device=frame.device)
    if fit == "stretch":
        return _resize(frame, size), full
    if fit == "crop":
        scale = max(w / src_w, h / src_h)
        mid = (max(w, round(src_w * scale)), max(h, round(src_h * scale)))
        big = _resize(frame, mid)
        x = _offset(mid[0] - w, anchor, "left", "right")
        y = _offset(mid[1] - h, anchor, "top", "bottom")
        return big[..., y:y + h, x:x + w], full
    # pad
    scale = min(w / src_w, h / src_h)
    inner = (min(w, max(1, round(src_w * scale))), min(h, max(1, round(src_h * scale))))
    small = _resize(frame, inner)
    x = _offset(w - inner[0], anchor, "left", "right")
    y = _offset(h - inner[1], anchor, "top", "bottom")
    pads = (x, w - inner[0] - x, y, h - inner[1] - y)
    if pad == "edge":
        out = F.pad(small, pads, mode="replicate")
    else:
        out = F.pad(small, pads, mode="constant", value=PAD_COLORS.get(pad, 0.0))
    if frame.shape[1] > 3:
        # Padding is not picture: transparent where an alpha channel exists.
        out[:, 3:] = F.pad(small[:, 3:], pads, mode="constant", value=0.0)
    mask = torch.zeros((1, h, w), dtype=frame.dtype, device=frame.device)
    mask[:, y:y + inner[1], x:x + inner[0]] = 1.0
    return out, mask


class TS_ModelResize(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_ModelResize",
            display_name="TS Model Resize",
            category="TS/Image/Size",
            description=(
                "Bring any image to the resolution a model was trained on. Pick the model; the "
                "node reads whether the picture is horizontal, vertical or square, takes the "
                "model's matching size (Wan 1280x720 or 720x1280, Qwen Image's seven official "
                "sizes, SDXL buckets...) and crops, pads or stretches the picture into it."
            ),
            search_aliases=[
                "model resize", "model resolution", "wan resize", "qwen resolution", "sdxl buckets",
                "fit to model", "aspect ratio", "bucket", "flux resolution", "ltx resolution",
            ],
            inputs=[
                IO.Image.Input("image", tooltip="Any size, any orientation; a batch is resized frame by frame."),
                IO.Combo.Input(
                    "model",
                    options=[p.name for p in PRESETS],
                    default=PRESETS[0].name,
                    tooltip="The model the picture is meant for. Its own resolutions are used, in the orientation of the picture.",
                ),
                IO.Combo.Input(
                    "fit",
                    options=list(FITS),
                    default="crop",
                    tooltip=(
                        "crop: fill the frame and trim what sticks out (nothing is distorted). "
                        "pad: show the whole picture and add bars. stretch: distort to the exact size."
                    ),
                ),
                IO.Combo.Input(
                    "anchor",
                    options=list(ANCHORS),
                    default="center",
                    tooltip="Which part is kept when cropping (or where the picture sits when padding). top keeps faces in tall shots.",
                ),
                IO.Combo.Input(
                    "pad_color",
                    options=list(PAD_OPTIONS),
                    default="black",
                    tooltip="Bars for 'pad'. edge repeats the border pixels.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image", tooltip="The picture at the model's resolution."),
                IO.Int.Output(display_name="width"),
                IO.Int.Output(display_name="height"),
                IO.Mask.Output(display_name="mask", tooltip="1 on the picture, 0 on the bars added by 'pad' (all 1 otherwise)."),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, model: str = PRESETS[0].name, fit: str = "crop",
                anchor: str = "center", pad_color: str = "black") -> IO.NodeOutput:
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError(f"{LOG_PREFIX} Expected IMAGE [B, H, W, C].")
        preset = PRESET_BY_NAME.get(model)
        if preset is None:
            raise ValueError(f"{LOG_PREFIX} Unknown model preset {model!r}.")
        fit = fit if fit in FITS else "crop"
        anchor = anchor if anchor in ANCHORS else "center"
        batch, src_h, src_w, _ = image.shape
        size = target_size(preset, src_w, src_h)

        frames, masks = [], []
        with torch.no_grad():
            for i in range(batch):
                frame = image[i:i + 1].permute(0, 3, 1, 2).float()
                out, mask = fit_frame(frame, size, fit, anchor, pad_color)
                frames.append(out.permute(0, 2, 3, 1).to(image.dtype))
                masks.append(mask)
        result = torch.cat(frames, dim=0)
        mask = torch.cat(masks, dim=0)

        line = f"{src_w}x{src_h} -> {size[0]}x{size[1]} ({fit}) for {model}"
        logger.info("%s %s", LOG_PREFIX, line)
        return IO.NodeOutput(result, size[0], size[1], mask, ui=UI.PreviewText(line))


NODE_CLASS_MAPPINGS = {"TS_ModelResize": TS_ModelResize}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_ModelResize": "TS Model Resize"}
