"""TS Multi Reference — attach up to three IMAGE references to a CONDITIONING.

For each connected reference image the node:
  1. Crops to the bounding box of the mask (with 16 px padding) when a
     MASK input is connected, OR to the bounding box of the embedded
     alpha for RGBA images, so the VAE does not waste resolution on
     transparent margins. The MASK is binarised at 0.5 and its
     orientation is auto-detected from the four corners — both
     ComfyUI Load Image alpha (1.0=transparent) and segmentation
     outputs (1.0=subject, e.g. SAM, BiRefNet, RemBG, Mask Editor) are
     handled correctly without any toggles.
  2. Pixels INSIDE the bbox are preserved as-is — the mask shape is
     NOT cut out. Only embedded RGBA alpha (rare) is composited onto
     white to avoid premultiplied-black halos.
  3. Resizes to fit max_megapixels on a configurable pixel grid
     (divide_by widget, default 32).
  4. Encodes through the supplied VAE and appends the result as a
     reference_latent on the conditioning.

The result is equivalent to chaining three native ReferenceLatent
nodes after three VAE Encode nodes that share a VAE, with the bonus
of mask-aware bbox cropping.

The node accepts up to three IMAGE inputs (image_1/image_2/image_3),
all optional. Use any standard ComfyUI image loader to feed them.
Each image_N has a matching optional mask_N MASK input — connect any
mask source (Load Image alpha, SAM3 Detect, BiRefNet, etc.) to tell
the node where the subject lives so it can bbox-crop around it.

Each image_N input has a matching image_N output that returns the
resized version (or ExecutionBlocker if the slot is empty so any
downstream consumer is silently skipped).

Behaviour:
- All inputs unconnected: conditioning passes through unchanged, all
  three image outputs return ExecutionBlocker.
- Some images connected, no conditioning: only resize for the matching
  image outputs, no encoding (no sink for reference_latents).
- Some images connected, no VAE: clear RuntimeError.
- Standard case (conditioning + vae + N images): N reference_latents
  attached, the corresponding image_N outputs return the resized image,
  the rest return ExecutionBlocker.

node_id: TS_MultiReference
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch

import comfy.utils
import node_helpers
from comfy_api.v0_0_2 import IO
from comfy_execution.graph_utils import ExecutionBlocker


logger = logging.getLogger(__name__)

_IMAGE_SLOT_COUNT = 3
_DEFAULT_SIZE_MULTIPLE = 32
_MIN_SIZE_MULTIPLE = 1
_MAX_SIZE_MULTIPLE = 128
_DEFAULT_MEGAPIXELS = 1.0
# Padding (in source-image pixels) added around the opaque bounding box
# when a MASK or RGBA alpha lets us tell which region of the reference
# is the actual subject. Matters most for product photos with a lot of
# transparent margin around the object.
_BBOX_PADDING = 16
# Pixel intensity considered opaque-enough to count toward the bbox.
# Treated as "alpha > 0.01"; in MASK convention that's "mask < 0.99".
_OPACITY_BBOX_THRESHOLD = 0.01
# Hardcoded resize method used when scaling reference images before VAE
# encoding. Kept as a module-level constant (not a node input) so the UI
# stays compact. Edit here if you need a different default.
_UPSCALE_METHOD = "area"
# Threshold for binarizing the input mask before bbox + composite. Anything
# >= this is "subject side", below is "background side". Hard binarisation
# eliminates fuzz from soft-edge / low-confidence segmentation masks.
_BINARIZE_THRESHOLD = 0.5
# When auto-detecting mask orientation, sample a small square in each of
# the four image corners. If most corners are bright the mask is treated
# as ComfyUI-native (1.0 = transparent / background). If most are dark
# we assume segmentation convention (1.0 = subject) and invert before any
# downstream operation.
_CORNER_SAMPLE_FRACTION = 16  # corner box side = min(H, W) // this


def _target_dimensions(
    width: int,
    height: int,
    max_megapixels: float,
    multiple: int = _DEFAULT_SIZE_MULTIPLE,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if max_megapixels <= 0:
        raise ValueError("max_megapixels must be greater than zero.")

    if multiple < 1:
        raise ValueError("multiple must be >= 1.")

    source_pixels = width * height
    target_pixels = int(max_megapixels * 1_000_000)
    scale = min(1.0, math.sqrt(target_pixels / source_pixels))

    target_width = max(multiple, int(math.floor((width * scale) / multiple)) * multiple)
    target_height = max(multiple, int(math.floor((height * scale) / multiple)) * multiple)
    return target_width, target_height


def _coerce_mask(mask: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
    """Bring a MASK tensor to [B, H, W, 1] matching the image grid.

    ComfyUI MASK convention is 1.0 = transparent, 0.0 = opaque (an
    inverted alpha). The standard Load Image node already uses that.
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # [H, W] → [1, H, W]
    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)  # [B, H, W] → [B, H, W, 1]
    if mask.ndim != 4 or mask.shape[-1] != 1:
        raise ValueError(f"Expected MASK tensor [B,H,W] or [B,H,W,1], got shape {tuple(mask.shape)}.")

    mask = mask.clamp(0.0, 1.0).to(dtype=torch.float32)

    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1, -1)
    elif mask.shape[0] != batch:
        # Best-effort — fall back to the first frame if batch sizes diverge.
        mask = mask[:1].expand(batch, -1, -1, -1)

    if mask.shape[1] != height or mask.shape[2] != width:
        # Bilinear resize to match the image grid; mask is BHWC so move to BCHW.
        bchw = mask.movedim(-1, 1)
        bchw = comfy.utils.common_upscale(bchw, width, height, "bilinear", "disabled")
        mask = bchw.movedim(1, -1)

    return mask.clamp(0.0, 1.0)


def _binarize_and_orient_mask(
    mask: torch.Tensor,
    image_height: int,
    image_width: int,
    batch: int,
) -> torch.Tensor:
    """Coerce, binarize and auto-orient a MASK to native ComfyUI convention.

    Two things happen here, both invisible to the user:

    1. **Binarisation** at ``_BINARIZE_THRESHOLD``: removes fuzz from
       low-confidence segmentation outputs (BiRefNet, SAM, RemBG) so the
       bbox is sharp and the white-background composite has hard edges
       instead of a halo. Soft-edge feathering is intentionally lost —
       the original feathered behaviour ran into the "translucent circle
       on top" symptom when paired with the wrong convention.

    2. **Orientation** by 4-corner majority voting. Almost every real
       reference image has its background in the corners. Sample a small
       square in each of the four corners after binarisation:

       - Most corners bright (≈1) → mask already follows native ComfyUI
         convention (1.0 = transparent / background). Pass through.
       - Most corners dark (≈0) → mask follows segmentation convention
         (1.0 = subject). Invert, so downstream code keeps using the
         "alpha = 1 - mask" formula.

       Ties (2/2) are treated as the native case to preserve the
       historic behaviour for ambiguous masks.

    Returns ``[B, H, W, 1]`` tensor in {0.0, 1.0}, native convention.
    """

    aligned = _coerce_mask(mask, batch=batch, height=image_height, width=image_width)
    binary = (aligned > _BINARIZE_THRESHOLD).to(dtype=torch.float32)

    h, w = binary.shape[1], binary.shape[2]
    cs = max(4, min(h, w) // _CORNER_SAMPLE_FRACTION)
    cs = min(cs, h, w)
    corners = (
        binary[:, :cs, :cs, :].mean().item(),
        binary[:, :cs, -cs:, :].mean().item(),
        binary[:, -cs:, :cs, :].mean().item(),
        binary[:, -cs:, -cs:, :].mean().item(),
    )
    bright_corners = sum(1 for value in corners if value > 0.5)
    if bright_corners < 2:
        # Background lives in the corners and is "0" → segmentation
        # convention. Invert so downstream code sees the native form.
        binary = 1.0 - binary
    return binary


def _crop_to_mask_bbox(
    image: torch.Tensor,
    mask: torch.Tensor | None,
    padding: int = _BBOX_PADDING,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Crop image (and mask if present) to the opaque bounding box + padding.

    Source of opacity, in priority order:
      1. ``mask`` argument (ComfyUI MASK convention: 1.0 = transparent).
      2. Embedded alpha channel of an RGBA image.

    If neither is available the inputs are returned unchanged. If the
    chosen alpha is fully transparent everywhere (degenerate input) we
    also leave the image alone — cropping to nothing is worse than
    cropping to "everything".
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        return image, mask

    batch, height, width, _ = image.shape

    aligned_mask: torch.Tensor | None = None
    if isinstance(mask, torch.Tensor):
        aligned_mask = _coerce_mask(mask, batch, height, width)
        # MASK: 1 = transparent, 0 = opaque → alpha = 1 - mask.
        opaque = (1.0 - aligned_mask.squeeze(-1)) > _OPACITY_BBOX_THRESHOLD
    elif image.shape[-1] >= 4:
        opaque = image[..., 3] > _OPACITY_BBOX_THRESHOLD
    else:
        # No alpha source — nothing to bbox against.
        return image, mask

    if not torch.any(opaque):
        # Fully transparent input: don't shrink to zero.
        return image, aligned_mask if aligned_mask is not None else mask

    # Union over batch frames so a multi-frame reference shares one bbox.
    union = opaque.any(dim=0)  # [H, W]

    rows = union.any(dim=1)
    cols = union.any(dim=0)
    ys = torch.nonzero(rows, as_tuple=False).flatten()
    xs = torch.nonzero(cols, as_tuple=False).flatten()
    y_min = int(ys[0].item())
    y_max = int(ys[-1].item()) + 1
    x_min = int(xs[0].item())
    x_max = int(xs[-1].item()) + 1

    pad = max(0, int(padding))
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(height, y_max + pad)
    x_max = min(width, x_max + pad)

    cropped_image = image[:, y_min:y_max, x_min:x_max, :]
    cropped_mask: torch.Tensor | None = None
    if aligned_mask is not None:
        cropped_mask = aligned_mask[:, y_min:y_max, x_min:x_max, :]

    return cropped_image, cropped_mask


def _normalize_image_tensor(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected IMAGE tensor, got {type(image)}.")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {image.ndim} dimensions.")
    if image.shape[-1] < 3:
        raise ValueError(f"Expected IMAGE tensor with at least 3 channels, got {image.shape[-1]}.")

    if image.shape[-1] >= 4:
        # Composite RGBA onto a white background using the embedded alpha.
        # Straight (non-premultiplied) convention — this matches PIL and
        # ComfyUI VideoFromFile (which keeps RGB and alpha separated).
        rgb = image[:, :, :, :3].clamp(0.0, 1.0)
        alpha = image[:, :, :, 3:4].clamp(0.0, 1.0)
        return (rgb * alpha + (1.0 - alpha)).clamp(0.0, 1.0)

    rgb = image[:, :, :, :3].clamp(0.0, 1.0)

    if isinstance(mask, torch.Tensor):
        # Standard ComfyUI Load Image returns MASK separately from IMAGE,
        # so the IMAGE we receive here is plain RGB with the transparent
        # pixels already flattened to (typically) black. Use the supplied
        # MASK to recover the alpha and composite on white.
        # ComfyUI MASK convention: 1.0 = transparent, 0.0 = opaque.
        batch, height, width, _ = rgb.shape
        mask_4d = _coerce_mask(mask, batch, height, width)
        # alpha = 1 - mask, composite on white:
        # out = rgb*alpha + (1-alpha)*white = rgb*(1-mask) + mask
        return (rgb * (1.0 - mask_4d) + mask_4d).clamp(0.0, 1.0)

    return rgb.clone()


def _resize_reference_image(
    image: torch.Tensor,
    max_megapixels: float,
    upscale_method: str,
    size_multiple: int = _DEFAULT_SIZE_MULTIPLE,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Step 0: when a MASK is supplied, binarize it AND auto-detect its
    # convention (Load Image alpha vs. segmentation models). Producing a
    # canonical native-convention binary mask once here means downstream
    # bbox code stays simple — no awareness of inverted masks needed.
    if isinstance(mask, torch.Tensor):
        normalized = image if image.ndim == 4 else image.unsqueeze(0)
        mask = _binarize_and_orient_mask(
            mask,
            image_height=int(normalized.shape[1]),
            image_width=int(normalized.shape[2]),
            batch=int(normalized.shape[0]),
        )

    # Step 1: tighten the reference around the actual subject using the
    # mask only as a bounding-box hint (with 16 px padding). The pixels
    # INSIDE the bbox are preserved as-is — we deliberately do NOT
    # composite by mask shape, because the user wants the original RGB
    # surrounding to stay intact (silhouette cut-outs were the wrong
    # default; user explicitly asked for "bbox only").
    image, _ = _crop_to_mask_bbox(image, mask, padding=_BBOX_PADDING)

    # Step 2: only composite when the IMAGE itself carries embedded
    # RGBA alpha (rare — most ComfyUI loaders return RGB + MASK
    # separately). Without this, premultiplied-alpha PNGs would feed
    # black halos to the VAE. The MASK input is intentionally not
    # passed here.
    image = _normalize_image_tensor(image)

    height = int(image.shape[1])
    width = int(image.shape[2])
    target_width, target_height = _target_dimensions(
        width, height, max_megapixels, multiple=size_multiple,
    )

    if target_width == width and target_height == height:
        return image.clamp(0.0, 1.0)

    samples = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(
        samples,
        target_width,
        target_height,
        upscale_method,
        "disabled",
    ).movedim(1, -1)
    return resized.clamp(0.0, 1.0)


def _encode_reference_latent(vae, image: torch.Tensor) -> dict:
    with torch.no_grad():
        samples = vae.encode(image)
    return {"samples": samples}


def _append_reference_latent(conditioning, latent: dict):
    # Equivalent to the native ReferenceLatent node.
    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [latent["samples"]]},
        append=True,
    )


class TS_MultiReference(IO.ComfyNode):
    """Attach up to three IMAGE references to a CONDITIONING via VAE + reference_latents."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_MultiReference",
            display_name="TS Multi Reference",
            category="TS/Conditioning",
            description=(
                "Resizes up to three reference images down to a "
                "divide_by-pixel grid (area downscale), encodes each "
                "through the VAE, and appends them as reference_latents "
                "on the conditioning. Equivalent to chaining three "
                "ReferenceLatent nodes after three VAE Encode nodes. "
                "Feed images from any standard Load Image node. Empty "
                "inputs are skipped silently."
            ),
            inputs=[
                IO.Conditioning.Input(
                    "conditioning",
                    optional=True,
                    tooltip=(
                        "Conditioning that will receive reference_latents. "
                        "Optional: when not connected, the node skips VAE "
                        "encoding and only resizes images for multi_images."
                    ),
                ),
                IO.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip=(
                        "VAE used to encode reference images into latents. "
                        "Required only when both images and conditioning are connected."
                    ),
                ),
                IO.Float.Input(
                    "max_megapixels",
                    default=_DEFAULT_MEGAPIXELS,
                    min=0.01,
                    max=16.0,
                    step=0.01,
                    tooltip="Maximum size for each reference image before VAE encoding.",
                ),
                IO.Int.Input(
                    "divide_by",
                    default=_DEFAULT_SIZE_MULTIPLE,
                    min=_MIN_SIZE_MULTIPLE,
                    max=_MAX_SIZE_MULTIPLE,
                    step=1,
                    tooltip=(
                        "Resized dimensions are rounded down to a multiple of this "
                        "value before VAE encoding. Most VAEs need 8 or 16; the "
                        "default 32 is a safe choice for Flux 2, Qwen image edit, etc."
                    ),
                ),
                IO.Boolean.Input(
                    "block_empty_slots",
                    default=True,
                    tooltip=(
                        "When enabled (default), empty image_N slots return "
                        "ExecutionBlocker so downstream nodes are silently "
                        "skipped (e.g. Save Image / PreviewImage). Disable "
                        "to pass None on empty slots instead, so downstream "
                        "nodes with optional IMAGE inputs (like TS Resolution "
                        "Selector) keep running and apply their own fallback."
                    ),
                ),
                IO.Image.Input(
                    "image_1",
                    optional=True,
                    tooltip="Reference image 1. Connect any IMAGE source.",
                ),
                IO.Mask.Input(
                    "mask_1",
                    optional=True,
                    tooltip=(
                        "Optional MASK for image_1. Used ONLY as a bounding-"
                        "box hint with 16 px padding — pixels inside the "
                        "bbox are preserved as-is (the mask shape is NOT "
                        "cut out). The mask is binarised at 0.5 and its "
                        "orientation is auto-detected from the four corners, "
                        "so both ComfyUI Load Image alpha (1.0=transparent) "
                        "and segmentation outputs (1.0=subject, e.g. SAM, "
                        "BiRefNet, RemBG, Mask Editor) are handled correctly "
                        "without any toggles."
                    ),
                ),
                IO.Image.Input(
                    "image_2",
                    optional=True,
                    tooltip="Reference image 2. Connect any IMAGE source.",
                ),
                IO.Mask.Input(
                    "mask_2",
                    optional=True,
                    tooltip="Optional MASK for image_2. See mask_1 tooltip.",
                ),
                IO.Image.Input(
                    "image_3",
                    optional=True,
                    tooltip="Reference image 3. Connect any IMAGE source.",
                ),
                IO.Mask.Input(
                    "mask_3",
                    optional=True,
                    tooltip="Optional MASK for image_3. See mask_1 tooltip.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image_1"),
                IO.Image.Output(display_name="image_2"),
                IO.Image.Output(display_name="image_3"),
                IO.Conditioning.Output(display_name="conditioning"),
            ],
            search_aliases=[
                "TS Multi Reference",
                "ReferenceLatent",
                "Flux 2 reference",
                "Qwen image edit reference",
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        max_megapixels: float,
        divide_by: int = _DEFAULT_SIZE_MULTIPLE,
        **_,
    ):
        if max_megapixels <= 0:
            return "max_megapixels must be greater than zero."
        if divide_by < _MIN_SIZE_MULTIPLE:
            return f"divide_by must be at least {_MIN_SIZE_MULTIPLE}."
        return True

    @classmethod
    def execute(
        cls,
        max_megapixels: float,
        divide_by: int = _DEFAULT_SIZE_MULTIPLE,
        block_empty_slots: bool = True,
        conditioning=None,
        vae=None,
        image_1: Optional[torch.Tensor] = None,
        image_2: Optional[torch.Tensor] = None,
        image_3: Optional[torch.Tensor] = None,
        mask_1: Optional[torch.Tensor] = None,
        mask_2: Optional[torch.Tensor] = None,
        mask_3: Optional[torch.Tensor] = None,
    ):
        # Per-slot input → per-slot output (resized image, ExecutionBlocker, or None).
        input_slots = (image_1, image_2, image_3)
        mask_slots = (mask_1, mask_2, mask_3)

        # Determine which slots got real IMAGE tensors.
        has_image = [isinstance(img, torch.Tensor) for img in input_slots]

        # ReferenceLatent is only attached when we have BOTH conditioning
        # and VAE. With only images + VAE, we still resize for the matching
        # image outputs but skip encoding (no sink to attach to).
        attach_reference = any(has_image) and conditioning is not None
        if attach_reference and vae is None:
            raise RuntimeError(
                "[TS Multi Reference] VAE input is required when reference "
                "images are provided together with a conditioning."
            )

        empty_slot_value = ExecutionBlocker(None) if block_empty_slots else None
        current_conditioning = conditioning
        output_images: list = []
        for slot_image, slot_mask, slot_has in zip(input_slots, mask_slots, has_image):
            if not slot_has:
                # Empty slot → block downstream silently or pass None through.
                output_images.append(empty_slot_value)
                continue

            processed_image = _resize_reference_image(
                slot_image,
                max_megapixels=max_megapixels,
                upscale_method=_UPSCALE_METHOD,
                size_multiple=divide_by,
                mask=slot_mask if isinstance(slot_mask, torch.Tensor) else None,
            )
            if attach_reference:
                latent = _encode_reference_latent(vae, processed_image)
                current_conditioning = _append_reference_latent(current_conditioning, latent)
            output_images.append(processed_image)

        # Output a valid CONDITIONING value even if nothing was supplied:
        # an empty list is a valid ComfyUI conditioning structure.
        output_conditioning = current_conditioning if current_conditioning is not None else []

        if not any(has_image):
            logger.debug("[TS Multi Reference] No reference images provided.")

        return IO.NodeOutput(*output_images, output_conditioning)


NODE_CLASS_MAPPINGS = {
    "TS_MultiReference": TS_MultiReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_MultiReference": "TS Multi Reference",
}
