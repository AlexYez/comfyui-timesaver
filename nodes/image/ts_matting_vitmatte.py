"""TS Matting (ViTMatte) - guided matting via Hugging Face ViTMatte models.

node_id: TS_Matting_ViTMatte

Refines a coarse SAM-style MASK into a photo-realistic alpha matte. Builds
an auto-trimap from the mask (erode -> confident foreground, dilate -> band,
outside -> background), then feeds image + trimap into ViTMatte. Designed
to consume the output of SAM3 Detect / SAM3 Video Track so users get crisp
edges, hair and semi-transparency without dropping into Photoshop.

Models cached under ``models/vitmatte/<variant>/``. First call downloads
weights from Hugging Face via ``snapshot_download``.

Reuses the post-processing pipeline of TS_BGRM_BiRefNet via direct helper
imports — exactly the same blur/offset/invert/background contract so the two
nodes are drop-in replacements for each other.
"""

from __future__ import annotations

import logging
from pathlib import Path

import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar
from comfy_api.v0_0_2 import IO
from PIL import Image, ImageFilter

from .ts_bgrm_birefnet import (
    _format_device_label,
    _get_target_device,
    _safe_empty_cache,
    _update_progress,
    hex_to_rgba,
    pil2tensor,
    tensor2pil,
)


logger = logging.getLogger(__name__)
_LOG_PREFIX = "[TS Matting ViTMatte]"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


MODEL_FOLDER_NAME = "vitmatte"

MODEL_VARIANTS: dict[str, dict[str, str]] = {
    "vitmatte-small-composition-1k": {
        "repo_id": "hustvl/vitmatte-small-composition-1k",
        "description": "Small ViT-S backbone, Composition-1K. ~96 MB.",
    },
    "vitmatte-base-composition-1k": {
        "repo_id": "hustvl/vitmatte-base-composition-1k",
        "description": "Base ViT-B backbone, Composition-1K. ~370 MB.",
    },
    "vitmatte-small-distinctions-646": {
        "repo_id": "hustvl/vitmatte-small-distinctions-646",
        "description": "Small ViT-S, Distinctions-646. ~96 MB.",
    },
    "vitmatte-base-distinctions-646": {
        "repo_id": "hustvl/vitmatte-base-distinctions-646",
        "description": "Base ViT-B, Distinctions-646. ~370 MB.",
    },
}


_PRECISION_OPTIONS = ("auto", "fp16", "bf16", "fp32")
_TEMPORAL_SMOOTH_OPTIONS = ("off", "median3", "median5", "ema_causal")


def _temporal_smooth_alphas(
    alphas: list,
    mode: str,
    ema_alpha: float,
) -> list:
    """Apply temporal smoothing across the alpha sequence.

    Per-frame matting has no concept of time, so identical objects in
    adjacent frames produce slightly different alphas — visually the edge
    "boils". This pass operates on the already-computed alphas:

    - ``median3`` / ``median5`` — N-frame temporal median. Best for random
      flicker; introduces ``N // 2`` frames of lag at the very edge of the
      batch (handled with ``mode="nearest"`` reflection in scipy).
    - ``ema_causal`` — exponential moving average. Causal, no lag, but a
      sudden alpha change in the middle of the clip is blended with the
      past, which can read as motion blur for fast objects.
    """
    n = len(alphas)
    if mode == "off" or n <= 1:
        return alphas
    stack = np.stack(alphas, axis=0)  # [N, H, W]
    if mode in ("median3", "median5"):
        size = 3 if mode == "median3" else 5
        try:
            import scipy.ndimage as _ndi

            stack = _ndi.median_filter(stack, size=(size, 1, 1), mode="nearest")
        except ImportError:
            # Pure-numpy fallback (slow for huge batches but correctness-safe).
            radius = size // 2
            padded = np.empty((n + 2 * radius,) + stack.shape[1:], dtype=stack.dtype)
            padded[:radius] = stack[0]
            padded[radius : radius + n] = stack
            padded[radius + n :] = stack[-1]
            out = np.empty_like(stack)
            for i in range(n):
                out[i] = np.median(padded[i : i + size], axis=0)
            stack = out
    elif mode == "ema_causal":
        a = float(np.clip(ema_alpha, 0.0, 0.99))
        if a > 0.0:
            for i in range(1, n):
                stack[i] = a * stack[i - 1] + (1.0 - a) * stack[i]
    return [stack[i] for i in range(n)]


def _resolve_dtype(target_device: torch.device, precision: str) -> torch.dtype:
    """Map the user-facing precision combo to a torch dtype.

    'auto' picks bf16 when the GPU supports it (Ampere+ with Tensor Core
    bf16; H100/L40/RTX 30+ etc.), falling back to fp16 elsewhere. CPU stays
    fp32 — running ViTMatte in half precision on CPU is dramatically slower
    than fp32 because Intel/AMD don't have hardware fp16 matmul.
    """
    device_type = getattr(target_device, "type", str(target_device))
    if device_type != "cuda":
        return torch.float32
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    # auto
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


# ---------------------------------------------------------------------------
# Idempotent SDPA monkey-patch for transformers' VitDetAttention.
#
# The stock implementation materialises a `(N, N)` attention matrix and runs
# its own matmul + softmax. On 4K input that single matrix is ~25 GB.
# `torch.nn.functional.scaled_dot_product_attention` picks the best backend
# available (memory-efficient / Flash / math) and never stores the full
# scores tensor, which both speeds up attention 1.5-3x and removes the OOM
# bottleneck for global-attention blocks.
#
# Relative-position embeddings are handled via the additive `attn_mask`
# argument of SDPA, exactly equivalent to the original "scores + rel_bias"
# step.
# ---------------------------------------------------------------------------


_VITDET_SDPA_PATCHED = False


def _ensure_vitdet_sdpa_patch() -> None:
    global _VITDET_SDPA_PATCHED
    if _VITDET_SDPA_PATCHED:
        return
    try:
        from transformers.models.vitdet import modeling_vitdet as _vitdet_mod
    except Exception as exc:
        logger.warning(
            "%s Could not import transformers VitDet module for SDPA patch: %s",
            _LOG_PREFIX, exc,
        )
        _VITDET_SDPA_PATCHED = True  # don't keep retrying
        return
    if getattr(_vitdet_mod.VitDetAttention.forward, "_ts_sdpa_patched", False):
        _VITDET_SDPA_PATCHED = True
        return

    get_rel_pos = _vitdet_mod.get_rel_pos

    def _decomposed_rel_pos_bias(queries, rel_pos_h, rel_pos_w, q_size, k_size):
        """Compute the additive position bias that the stock code adds to
        ``attention_scores`` after the q @ k.T step. Returned shape
        ``[batch_size * num_heads, HW, HW]`` matches the SDPA ``attn_mask``
        contract.
        """
        queries_height, queries_width = q_size
        keys_height, keys_width = k_size
        relative_height = get_rel_pos(queries_height, keys_height, rel_pos_h)
        relative_width = get_rel_pos(queries_width, keys_width, rel_pos_w)
        batch_size, _, dim = queries.shape
        r_q = queries.reshape(batch_size, queries_height, queries_width, dim)
        rel_h_emb = torch.einsum("bhwc,hkc->bhwk", r_q, relative_height)
        rel_w_emb = torch.einsum("bhwc,wkc->bhwk", r_q, relative_width)
        bias = (
            rel_h_emb[:, :, :, :, None] + rel_w_emb[:, :, :, None, :]
        ).reshape(
            batch_size,
            queries_height * queries_width,
            keys_height * keys_width,
        )
        return bias

    def _patched_forward(self, hidden_state, output_attentions=False):
        # The original path supports ``output_attentions=True`` to expose
        # ``attention_probs``. SDPA does not expose that intermediate, so we
        # fall back to the unpatched implementation in that (rare) case.
        if output_attentions:
            return _original_forward(self, hidden_state, output_attentions=True)

        batch_size, height, width, _ = hidden_state.shape
        # Same qkv reshape as the original code; ends with
        # queries/keys/values of shape [B*num_heads, HW, head_dim].
        qkv = (
            self.qkv(hidden_state)
            .reshape(batch_size, height * width, 3, self.num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        queries, keys, values = qkv.reshape(
            3, batch_size * self.num_heads, height * width, -1
        ).unbind(0)

        attn_bias = None
        if self.use_relative_position_embeddings:
            attn_bias = _decomposed_rel_pos_bias(
                queries,
                self.rel_pos_h,
                self.rel_pos_w,
                (height, width),
                (height, width),
            )

        # SDPA applies the scale internally; pass it explicitly to match the
        # `(q * self.scale) @ k.T` semantics of the original.
        attn_out = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attn_bias,
            dropout_p=0.0,
            scale=self.scale,
        )

        attn_out = attn_out.view(batch_size, self.num_heads, height, width, -1)
        attn_out = attn_out.permute(0, 2, 3, 1, 4)
        attn_out = attn_out.reshape(batch_size, height, width, -1)
        attn_out = self.proj(attn_out)
        return (attn_out,)

    _original_forward = _vitdet_mod.VitDetAttention.forward
    _patched_forward._ts_sdpa_patched = True
    _vitdet_mod.VitDetAttention.forward = _patched_forward
    _VITDET_SDPA_PATCHED = True
    logger.info("%s Patched VitDetAttention.forward to use SDPA.", _LOG_PREFIX)


def _register_model_folder() -> None:
    """Register ``models/vitmatte/`` with ComfyUI's folder_paths."""
    try:
        base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
        base.mkdir(parents=True, exist_ok=True)
        if hasattr(folder_paths, "add_model_folder_path"):
            folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(base))
    except Exception as exc:
        logger.warning(
            "%s Failed to register '%s' model folder: %s",
            _LOG_PREFIX, MODEL_FOLDER_NAME, exc,
        )


_register_model_folder()


def _resolve_model_dir(variant: str) -> Path:
    base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME / variant
    base.mkdir(parents=True, exist_ok=True)
    return base


def _model_files_present(variant: str) -> bool:
    """Return True iff the ViTMatte variant is already on disk."""
    local_dir = _resolve_model_dir(variant)
    if not (local_dir / "config.json").is_file():
        return False
    return (
        (local_dir / "model.safetensors").is_file()
        or (local_dir / "pytorch_model.bin").is_file()
    )


def _download_model_files(variant: str) -> Path:
    """Download the ViTMatte checkpoint into ``models/vitmatte/<variant>/``.

    Caller decides whether to display a progress bar — this function just
    fetches the files. ``huggingface_hub`` prints its own per-file tqdm to
    the console, which provides the byte-level progress feedback ComfyUI's
    progress bar cannot.
    """
    local_dir = _resolve_model_dir(variant)
    repo_id = MODEL_VARIANTS[variant]["repo_id"]
    logger.info("%s Downloading %s from %s", _LOG_PREFIX, variant, repo_id)

    # huggingface_hub.snapshot_download pulls in HTTP/auth/cache infra and is
    # only needed on the very first run; keep it lazy so cold startup is fast.
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision="main",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt"],
    )
    return local_dir


# ---------------------------------------------------------------------------
# Singleton state (CLAUDE.md §5 - V3 locked classes forbid cls._x = ...)
# ---------------------------------------------------------------------------


class _ViTMatteState:
    """Module-level cache for the loaded ViTMatte model + processor."""

    model: object | None = None
    processor: object | None = None
    variant: str = ""
    device: str = ""
    dtype: object | None = None
    precision: str = ""


_state = _ViTMatteState()


def _state_matches(
    variant: str,
    target_device: torch.device,
    target_dtype: torch.dtype,
    precision: str,
) -> bool:
    return (
        _state.model is not None
        and _state.variant == variant
        and _state.device == str(target_device)
        and _state.dtype == target_dtype
        and _state.precision == precision
    )


def _load_model(
    variant: str,
    target_device: torch.device,
    target_dtype: torch.dtype,
    precision: str,
    progress: ProgressBar | None = None,
):
    """Load ViTMatte weights from disk and move them onto the target device.

    Assumes the files are already present (``_download_model_files`` was
    called first if needed). The progress bar (if provided) ranges 0..100
    across this single phase:
        - 0..30:  ``from_pretrained`` for processor + model (CPU load).
        - 30..80: ``model.to(target_device)`` (VRAM transfer / dtype cast).
        - 80..100: bookkeeping.
    """
    target_device_key = str(target_device)

    # Free a stale model off-GPU first to free VRAM for the new one.
    if _state.model is not None:
        try:
            _state.model.cpu()
        except Exception:
            pass
        _state.model = None
        _state.processor = None
        _state.variant = ""
        _state.device = ""
        _state.dtype = None
        _state.precision = ""
        _safe_empty_cache()

    # Apply the SDPA patch once before constructing the model — patch is
    # idempotent so calling it again on cache miss is free.
    _ensure_vitdet_sdpa_patch()

    local_dir = _resolve_model_dir(variant)

    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    if progress is not None:
        _update_progress(progress, 5)
    # local_dir is a fully resolved on-disk path populated by
    # _download_model_files(); from_pretrained will not hit the network here,
    # but bandit B615 still requires an explicit revision argument on every
    # from_pretrained call regardless of source.
    processor = VitMatteImageProcessor.from_pretrained(str(local_dir), revision="main")
    if progress is not None:
        _update_progress(progress, 15)
    model = VitMatteForImageMatting.from_pretrained(str(local_dir), revision="main")
    model.eval()
    if target_dtype == torch.float16:
        model.half()
    elif target_dtype == torch.bfloat16:
        model.to(dtype=torch.bfloat16)
    else:
        model.float()
    if progress is not None:
        _update_progress(progress, 30)
    model.to(target_device)
    # Channels-last layout helps the matting head's conv decoder on Tensor
    # Cores; harmless for the ViT backbone which already operates in
    # (B, H, W, C). One-time cost at load time.
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception as exc:
        logger.warning(
            "%s channels_last conversion failed (continuing): %s",
            _LOG_PREFIX, exc,
        )
    if progress is not None:
        _update_progress(progress, 80)

    _state.model = model
    _state.processor = processor
    _state.variant = variant
    _state.device = target_device_key
    _state.dtype = target_dtype
    _state.precision = precision

    logger.info(
        "%s Loaded %s on %s (precision=%s, dtype=%s, SDPA=on, channels_last=on)",
        _LOG_PREFIX, variant,
        _format_device_label(target_device),
        precision, str(target_dtype),
    )
    if progress is not None:
        _update_progress(progress, 100)
    return model, processor


# ---------------------------------------------------------------------------
# Trimap construction
# ---------------------------------------------------------------------------


def _make_trimap(
    mask_float: np.ndarray,
    trimap_erode_px: int,
    trimap_dilate_px: int,
) -> np.ndarray:
    """Build an uint8 trimap: 0 = background, 128 = unknown, 255 = foreground.

    - Erode the binary mask to get the confident foreground core.
    - Dilate the binary mask to extend the unknown band into the surroundings.
    - Everything outside the dilation is hard background.

    Wider erode_px shrinks the certain-foreground region (lets ViTMatte choose
    softer hair-style edges around the body). Wider dilate_px extends the
    unknown band into the background (more context for fly-away hair, fur).
    """
    import cv2

    binary = (mask_float > 0.5).astype(np.uint8) * 255
    if trimap_erode_px > 0:
        ek = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(trimap_erode_px) * 2 + 1, int(trimap_erode_px) * 2 + 1),
        )
        fg = cv2.erode(binary, ek, iterations=1)
    else:
        fg = binary
    if trimap_dilate_px > 0:
        dk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(trimap_dilate_px) * 2 + 1, int(trimap_dilate_px) * 2 + 1),
        )
        bg_band = cv2.dilate(binary, dk, iterations=1)
    else:
        bg_band = binary

    trimap = np.zeros_like(binary)
    trimap[bg_band > 0] = 128  # unknown band
    trimap[fg > 0] = 255       # confident foreground overwrites unknown
    return trimap


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class TS_Matting_ViTMatte(IO.ComfyNode):
    """ViTMatte-based guided matting from a coarse SAM-style mask."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Matting_ViTMatte",
            display_name="TS Matting (ViTMatte)",
            category="TS/Image",
            description=(
                "Guided matting via Hugging Face ViTMatte. Builds an "
                "auto-trimap from a coarse binary MASK (erode for confident "
                "foreground, dilate for the unknown band), then refines the "
                "alpha with hair / fur / semi-transparency. Drop-in for "
                "TS Remove Background when you already have a SAM3 mask."
            ),
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="Source image batch [B, H, W, 3].",
                ),
                IO.Mask.Input(
                    "mask",
                    tooltip=(
                        "Coarse binary or soft MASK [B, H, W] (e.g. from "
                        "SAM3 Detect or SAM3 Video Track). A single-frame "
                        "mask is broadcast across the image batch."
                    ),
                ),
                IO.Combo.Input(
                    "model",
                    options=list(MODEL_VARIANTS.keys()),
                    default="vitmatte-base-composition-1k",
                    tooltip=(
                        "ViTMatte variant. 'base' = ~370 MB (recommended for "
                        "best edge quality), 'small' = ~96 MB (faster, lower "
                        "detail). 'composition-1k' is the standard matting "
                        "benchmark; 'distinctions-646' is more diverse."
                    ),
                ),
                IO.Int.Input(
                    "trimap_erode_px",
                    default=10,
                    min=0,
                    max=128,
                    step=1,
                    tooltip=(
                        "Erosion radius for the confident foreground core. "
                        "Higher = thinner certain-foreground, more soft "
                        "edges (better for hair)."
                    ),
                ),
                IO.Int.Input(
                    "trimap_dilate_px",
                    default=20,
                    min=0,
                    max=128,
                    step=1,
                    tooltip=(
                        "Dilation radius for the unknown band beyond the "
                        "mask. Higher = ViTMatte gets more context around "
                        "the object (helps fly-away hair / fur). 20 is a "
                        "good default; bump to 32+ if the source mask is "
                        "tight around hair/fur."
                    ),
                ),
                IO.Int.Input(
                    "max_resolution",
                    default=2048,
                    min=0,
                    max=4096,
                    step=64,
                    optional=True,
                    tooltip=(
                        "Long-edge cap (px) for ViTMatte inference. ViTMatte "
                        "uses global self-attention in some ViT blocks; on "
                        "4K input that single attention matrix needs ~25 GB "
                        "of VRAM. The frame is downscaled before inference "
                        "and the resulting alpha is upscaled back to native "
                        "size. 0 = use the native resolution (only safe for "
                        "small images / lots of VRAM)."
                    ),
                ),
                IO.Boolean.Input(
                    "auto_crop_by_mask",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Crop the frame to the mask's bounding box (plus "
                        "padding) before running ViTMatte. Saves enormous "
                        "amounts of VRAM and time when the object is small "
                        "compared to the frame. Has no effect when the mask "
                        "covers the whole frame."
                    ),
                ),
                IO.Int.Input(
                    "crop_padding_pct",
                    default=15,
                    min=0,
                    max=100,
                    step=1,
                    optional=True,
                    tooltip=(
                        "Padding around the mask bbox in percent of bbox "
                        "size when ``auto_crop_by_mask`` is on. 15 = 15% "
                        "context margin."
                    ),
                ),
                IO.Combo.Input(
                    "precision",
                    options=list(_PRECISION_OPTIONS),
                    default="auto",
                    optional=True,
                    tooltip=(
                        "Inference precision on CUDA. 'auto' picks bf16 on "
                        "Ampere+ (more numerically robust, no fp16 inf/nan "
                        "on extreme alpha) and fp16 elsewhere. Force 'fp32' "
                        "for diagnostics. CPU always runs fp32."
                    ),
                ),
                IO.Combo.Input(
                    "temporal_smooth",
                    options=list(_TEMPORAL_SMOOTH_OPTIONS),
                    default="median3",
                    optional=True,
                    tooltip=(
                        "Smooth alpha across frames to reduce 'boiling' "
                        "edges in video. 'median3' (default) kills random "
                        "1-frame flicker with minimal overhead and is a "
                        "no-op for single images. 'median5' for stronger "
                        "flicker (2-frame lag at clip boundaries). "
                        "'ema_causal' is causal exponential averaging — "
                        "no lag, but can blur fast motion. 'off' disables. "
                        "Peak RAM grows by ~N*H*W*4 bytes during the pass."
                    ),
                ),
                IO.Float.Input(
                    "ema_alpha",
                    default=0.5,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Strength of the causal EMA when temporal_smooth = "
                        "'ema_causal'. Higher = more smoothing (lag for "
                        "moving objects); lower = closer to raw per-frame "
                        "alpha."
                    ),
                ),
                IO.Int.Input(
                    "mask_blur",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    optional=True,
                    tooltip="Final Gaussian blur on the alpha output (px).",
                ),
                IO.Int.Input(
                    "mask_offset",
                    default=0,
                    min=-20,
                    max=20,
                    step=1,
                    optional=True,
                    tooltip="Shrink (-) / expand (+) the final alpha mask.",
                ),
                IO.Boolean.Input(
                    "invert_output",
                    default=False,
                    optional=True,
                    tooltip="Invert the alpha output.",
                ),
                IO.Combo.Input(
                    "background",
                    options=["Alpha", "Color"],
                    default="Alpha",
                    optional=True,
                    tooltip="Output background mode.",
                ),
                IO.Color.Input(
                    "background_color",
                    default="#ffffff",
                    optional=True,
                    tooltip="Solid background colour when 'background' = 'Color'.",
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="IMAGE"),
                IO.Mask.Output(display_name="MASK"),
                IO.Image.Output(display_name="MASK_IMAGE"),
            ],
            search_aliases=[
                "vitmatte",
                "matting",
                "alpha matte",
                "guided matting",
                "hair",
                "trimap",
            ],
        )

    @classmethod
    def execute(
        cls,
        image: "torch.Tensor",
        mask: "torch.Tensor",
        model: str = "vitmatte-base-composition-1k",
        trimap_erode_px: int = 10,
        trimap_dilate_px: int = 20,
        max_resolution: int = 2048,
        auto_crop_by_mask: bool = True,
        crop_padding_pct: int = 15,
        precision: str = "auto",
        temporal_smooth: str = "median3",
        ema_alpha: float = 0.5,
        mask_blur: int = 0,
        mask_offset: int = 0,
        invert_output: bool = False,
        background: str = "Alpha",
        background_color: str = "#ffffff",
    ) -> IO.NodeOutput:
        if model not in MODEL_VARIANTS:
            raise RuntimeError(f"{_LOG_PREFIX} Unknown model variant '{model}'.")
        if precision not in _PRECISION_OPTIONS:
            raise RuntimeError(
                f"{_LOG_PREFIX} Unknown precision '{precision}'. "
                f"Expected one of {_PRECISION_OPTIONS}."
            )
        if temporal_smooth not in _TEMPORAL_SMOOTH_OPTIONS:
            raise RuntimeError(
                f"{_LOG_PREFIX} Unknown temporal_smooth '{temporal_smooth}'. "
                f"Expected one of {_TEMPORAL_SMOOTH_OPTIONS}."
            )
        if image.ndim != 4:
            raise RuntimeError(
                f"{_LOG_PREFIX} image must be [B, H, W, 3], got shape {tuple(image.shape)}."
            )

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3:
            raise RuntimeError(
                f"{_LOG_PREFIX} mask must be [B, H, W] or [H, W], got shape {tuple(mask.shape)}."
            )

        batch_size = image.shape[0]
        if mask.shape[0] != batch_size:
            # Single-frame mask -> broadcast to the whole image batch. Useful
            # when the user only painted the first frame for a video — same
            # static cut-out applied frame by frame.
            if mask.shape[0] == 1:
                mask = mask.repeat(batch_size, 1, 1)
            else:
                raise RuntimeError(
                    f"{_LOG_PREFIX} mask batch ({mask.shape[0]}) does not "
                    f"match image batch ({batch_size})."
                )
        h, w = int(image.shape[1]), int(image.shape[2])
        if int(mask.shape[-2]) != h or int(mask.shape[-1]) != w:
            raise RuntimeError(
                f"{_LOG_PREFIX} mask spatial size {tuple(mask.shape[-2:])} "
                f"does not match image spatial size ({h}, {w})."
            )

        target_device = _get_target_device()
        target_dtype = _resolve_dtype(target_device, precision)
        logger.info(
            "%s model=%s batch=%d size=%dx%d device=%s precision=%s dtype=%s",
            _LOG_PREFIX, model, batch_size, w, h,
            _format_device_label(target_device),
            precision, str(target_dtype),
        )

        # ---- Phase 1: download model files (skipped if already on disk) ----
        if not _model_files_present(model):
            dl_pbar = ProgressBar(100)
            _update_progress(dl_pbar, 0)
            try:
                _download_model_files(model)
            except Exception as exc:
                raise RuntimeError(
                    f"{_LOG_PREFIX} Failed to download {model}: {exc}"
                ) from exc
            _update_progress(dl_pbar, 100)

        # ---- Phase 2: load model into VRAM (skipped if already cached) ----
        if _state_matches(model, target_device, target_dtype, precision):
            net, processor = _state.model, _state.processor
        else:
            load_pbar = ProgressBar(100)
            _update_progress(load_pbar, 0)
            try:
                net, processor = _load_model(
                    model,
                    target_device,
                    target_dtype,
                    precision,
                    progress=load_pbar,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"{_LOG_PREFIX} Failed to load ViTMatte: {exc}"
                ) from exc

        # ---- Phase 3: per-frame ViTMatte inference (collect raw alphas) ----
        # Two-pass: we need all alphas in memory before temporal smoothing
        # so the median/EMA pass can look at neighbouring frames. Memory
        # cost: N * H * W * 4 bytes (float32). 4K * 96 frames ≈ 3 GB —
        # large but typical for video matting pipelines.
        inf_pbar = ProgressBar(batch_size)
        _update_progress(inf_pbar, 0, total=batch_size)

        effective_max_res = int(max_resolution or 0)
        raw_alphas: list[np.ndarray] = []

        for frame_index in range(batch_size):
            # OOM-retry: ViTMatte global attention is quadratic in token
            # count; on OOM we halve max_resolution until the frame fits or
            # we hit 256 px (at which point GPU is truly out of room).
            current_max_res = effective_max_res
            alpha = None
            while alpha is None:
                try:
                    alpha = cls._run_frame(
                        net,
                        processor,
                        image[frame_index],
                        mask[frame_index],
                        trimap_erode_px=int(trimap_erode_px),
                        trimap_dilate_px=int(trimap_dilate_px),
                        max_resolution=int(current_max_res),
                        auto_crop=bool(auto_crop_by_mask),
                        crop_padding_pct=int(crop_padding_pct),
                        target_device=target_device,
                        target_dtype=target_dtype,
                    )
                except torch.cuda.OutOfMemoryError:
                    _safe_empty_cache()
                    new_max_res = (
                        max(256, current_max_res // 2)
                        if current_max_res > 0
                        else 1024
                    )
                    if current_max_res > 0 and new_max_res >= current_max_res:
                        raise
                    logger.warning(
                        "%s CUDA OOM on frame %d; retrying with max_resolution=%d",
                        _LOG_PREFIX, frame_index, new_max_res,
                    )
                    current_max_res = new_max_res
            raw_alphas.append(alpha)
            _update_progress(inf_pbar, frame_index + 1, total=batch_size)

        # ---- Phase 4: temporal smoothing (skipped for single-frame inputs) ----
        if temporal_smooth != "off" and batch_size > 1:
            logger.info(
                "%s Temporal smoothing: mode=%s ema_alpha=%.2f frames=%d",
                _LOG_PREFIX, temporal_smooth, float(ema_alpha), batch_size,
            )
            raw_alphas = _temporal_smooth_alphas(
                raw_alphas, temporal_smooth, float(ema_alpha)
            )

        # ---- Phase 5: per-frame post-process + composite ----
        post_pbar = ProgressBar(batch_size)
        _update_progress(post_pbar, 0, total=batch_size)
        processed_images: list[torch.Tensor] = []
        processed_masks: list[torch.Tensor] = []

        for frame_index, alpha in enumerate(raw_alphas):
            # Per-frame post-processing matches TS_BGRM_BiRefNet contract.
            pil_alpha = Image.fromarray(
                np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L"
            )
            if mask_blur > 0:
                pil_alpha = pil_alpha.filter(
                    ImageFilter.GaussianBlur(radius=int(mask_blur))
                )
            if mask_offset != 0:
                if mask_offset > 0:
                    for _ in range(int(mask_offset)):
                        pil_alpha = pil_alpha.filter(ImageFilter.MaxFilter(3))
                else:
                    for _ in range(-int(mask_offset)):
                        pil_alpha = pil_alpha.filter(ImageFilter.MinFilter(3))
            if invert_output:
                pil_alpha = Image.fromarray(255 - np.array(pil_alpha), mode="L")

            orig_image = tensor2pil(image[frame_index])
            orig_rgba = orig_image.convert("RGBA")
            r, g, b, _ = orig_rgba.split()
            foreground = Image.merge("RGBA", (r, g, b, pil_alpha))
            if background == "Alpha":
                processed_images.append(pil2tensor(foreground))
            else:
                rgba = hex_to_rgba(background_color)
                bg_image = Image.new("RGBA", orig_image.size, rgba)
                composite = Image.alpha_composite(bg_image, foreground)
                processed_images.append(pil2tensor(composite.convert("RGB")))
            processed_masks.append(pil2tensor(pil_alpha))

            _update_progress(post_pbar, frame_index + 1, total=batch_size)

        image_output = torch.cat(processed_images, dim=0)
        mask_output = torch.cat(processed_masks, dim=0)
        mask_image_output = mask_output.unsqueeze(-1).expand(-1, -1, -1, 3)

        return IO.NodeOutput(image_output, mask_output, mask_image_output)

    @classmethod
    def _run_frame(
        cls,
        net,
        processor,
        image_frame: "torch.Tensor",
        mask_frame: "torch.Tensor",
        *,
        trimap_erode_px: int,
        trimap_dilate_px: int,
        max_resolution: int,
        auto_crop: bool,
        crop_padding_pct: int,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> np.ndarray:
        """Run ViTMatte on a single frame. Returns ``float32 [H, W]`` alpha.

        Per-frame because each frame may end up with a different processed
        size after auto-crop and downscale — batching makes no sense once
        sizes diverge. Used by ``execute`` with an OOM retry loop that
        halves ``max_resolution``.
        """
        full_h = int(image_frame.shape[0])
        full_w = int(image_frame.shape[1])

        img_np = (
            image_frame.detach().cpu().numpy() * 255.0
        ).clip(0, 255).astype(np.uint8)
        mask_np = mask_frame.detach().cpu().numpy().astype(np.float32)

        binary = (mask_np > 0.5).astype(np.uint8) * 255
        if not binary.any():
            # Empty mask -> empty alpha; saves a ViTMatte forward pass.
            return np.zeros((full_h, full_w), dtype=np.float32)

        trimap_full = _make_trimap(mask_np, trimap_erode_px, trimap_dilate_px)

        if auto_crop:
            bbox = _bbox_with_padding(binary, full_w, full_h, crop_padding_pct)
        else:
            bbox = (0, 0, full_w, full_h)
        x1, y1, x2, y2 = bbox
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w <= 0 or crop_h <= 0:
            return np.zeros((full_h, full_w), dtype=np.float32)

        crop_img = img_np[y1:y2, x1:x2]
        crop_trimap = trimap_full[y1:y2, x1:x2]

        # Decide inference resolution: cap the long edge at max_resolution.
        proc_w, proc_h = crop_w, crop_h
        long_edge = max(crop_h, crop_w)
        if max_resolution > 0 and long_edge > max_resolution:
            scale = max_resolution / long_edge
            proc_h = max(1, int(round(crop_h * scale)))
            proc_w = max(1, int(round(crop_w * scale)))

        pil_img = Image.fromarray(crop_img, mode="RGB")
        pil_trimap = Image.fromarray(crop_trimap, mode="L")
        if proc_h != crop_h or proc_w != crop_w:
            pil_img = pil_img.resize((proc_w, proc_h), Image.BICUBIC)
            # Trimap uses NEAREST so the three discrete values (0/128/255)
            # survive the resize — bicubic would smear them into intermediate
            # values that ViTMatte interprets as "uncertain everywhere".
            pil_trimap = pil_trimap.resize((proc_w, proc_h), Image.NEAREST)

        inputs = processor(images=[pil_img], trimaps=[pil_trimap], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(
            device=target_device, dtype=target_dtype, non_blocking=True
        )
        # channels_last layout matches the model's memory format so the
        # matting head's conv decoder hits Tensor Cores cleanly. NCHW->NHWC
        # layout change is free on first call (no data copy by PyTorch).
        try:
            pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass

        with torch.inference_mode():
            outputs = net(pixel_values=pixel_values)
        # VitMatteImageProcessor pads inputs to a multiple of patch_size;
        # the alpha tensor shares that padded canvas. Crop it back to the
        # processed size before resizing to the (un-padded) crop size.
        alpha = outputs.alphas[0, 0].float().cpu().numpy()
        alpha = alpha[:proc_h, :proc_w]

        if proc_h != crop_h or proc_w != crop_w:
            alpha_pil = Image.fromarray(
                np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L"
            )
            alpha_pil = alpha_pil.resize((crop_w, crop_h), Image.BICUBIC)
            alpha = np.asarray(alpha_pil, dtype=np.float32) / 255.0

        # Paste the alpha back into a full-size canvas of zeros.
        full_alpha = np.zeros((full_h, full_w), dtype=np.float32)
        full_alpha[y1:y2, x1:x2] = np.clip(alpha, 0.0, 1.0).astype(np.float32)
        return full_alpha


def _bbox_with_padding(
    binary_mask: np.ndarray,
    image_w: int,
    image_h: int,
    padding_pct: int,
) -> tuple[int, int, int, int]:
    """Return ``(x1, y1, x2, y2)`` enclosing the mask, padded by ``padding_pct``
    percent of the bbox side. Clipped to the image bounds. Falls back to the
    full frame if the mask is empty."""
    ys, xs = np.where(binary_mask > 0)
    if ys.size == 0 or xs.size == 0:
        return (0, 0, int(image_w), int(image_h))
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding_pct / 100)
    pad_y = int(bh * padding_pct / 100)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(int(image_w), x2 + pad_x)
    y2 = min(int(image_h), y2 + pad_y)
    return (x1, y1, x2, y2)


NODE_CLASS_MAPPINGS = {"TS_Matting_ViTMatte": TS_Matting_ViTMatte}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Matting_ViTMatte": "TS Matting (ViTMatte)"}
