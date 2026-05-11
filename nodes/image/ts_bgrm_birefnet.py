# Model License Notice:
# - BiRefNet Models: Apache-2.0 License (https://huggingface.co/ZhengPeng7)
# Code based on : https://github.com/AILab-AI/ComfyUI-RMBG

import importlib.util
import logging
import os
import sys
import types
from contextlib import contextmanager

import comfy.model_management as model_management
import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar
from comfy_api.v0_0_2 import IO
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)
_LOG_PREFIX = "[TS Remove Background]"


def _register_birefnet_folder():
    """Register `models/BiRefNet/` so ComfyUI's folder_paths is aware of it.

    Wrapped in try/except: any monkey-patch from another custom node that
    breaks `folder_paths` should not abort our import.
    """
    try:
        folder_paths.add_model_folder_path(
            "birefnet", os.path.join(folder_paths.models_dir, "BiRefNet")
        )
    except Exception as exc:
        logger.warning("%s Failed to register 'birefnet' model folder: %s", _LOG_PREFIX, exc)


_register_birefnet_folder()

# Model configuration.
#
# Each variant has a *primary* source (``ZhengPeng7/<variant>``, the
# official author's HF account) plus a *fallback* aggregated mirror
# (``1038lab/BiRefNet``, where all weights live in one repo named
# ``<variant>.safetensors`` alongside the shared ``birefnet.py``).
#
# Why both:
#   * ZhengPeng7 ships the up-to-date ``birefnet.py`` with the September
#     2025 Swin attention rewrite, which lets us safely turn on SDPA via
#     the runtime check below. That's where the speed comes from.
#   * Some of the author's repos are gated / archived (e.g.
#     ``ZhengPeng7/BiRefNet-HR-matting`` started returning 401 in late
#     2025). The 1038lab mirror still serves the same weights publicly,
#     just bundled with an older ``birefnet.py``. SDPA stays disabled in
#     that case — quality is preserved, speed reverts to the legacy path.
#
# Each variant lives in its OWN sub-directory ``models/BiRefNet/<variant>/``
# so per-model bundled ``birefnet.py`` files never clobber each other.
MODEL_CONFIG = {
    "BiRefNet-general": {
        "repo_id": "ZhengPeng7/BiRefNet",
        "fallback_repo_id": "1038lab/BiRefNet",
        "fallback_filename": "BiRefNet-general.safetensors",
        "description": "General purpose model with balanced performance",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512,
    },
    "BiRefNet_512x512": {
        # No ZhengPeng7/BiRefNet_512x512 repo exists on HF — the 512×512
        # variant only ships through the 1038lab mirror.
        "repo_id": "1038lab/BiRefNet",
        "fallback_repo_id": None,
        "fallback_filename": "BiRefNet_512x512.safetensors",
        # Primary already targets 1038lab; reuse the same allow-pattern
        # logic by leaving fallback_filename so download_model picks the
        # right .safetensors file.
        "primary_mode": "mirror",
        "description": "Optimized for 512x512 resolution, faster processing",
        "default_res": 512,
        "max_res": 1024,
        "min_res": 256,
        "force_res": True,
    },
    "BiRefNet-HR": {
        # HF spelling uses underscore: BiRefNet_HR (not BiRefNet-HR).
        "repo_id": "ZhengPeng7/BiRefNet_HR",
        "fallback_repo_id": "1038lab/BiRefNet",
        "fallback_filename": "BiRefNet-HR.safetensors",
        "description": "High resolution general purpose model (2048x2048)",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024,
    },
    "BiRefNet-portrait": {
        "repo_id": "ZhengPeng7/BiRefNet-portrait",
        "fallback_repo_id": "1038lab/BiRefNet",
        "fallback_filename": "BiRefNet-portrait.safetensors",
        "description": "Optimized for portrait / human matting",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512,
    },
    "BiRefNet-matting": {
        "repo_id": "ZhengPeng7/BiRefNet-matting",
        "fallback_repo_id": "1038lab/BiRefNet",
        "fallback_filename": "BiRefNet-matting.safetensors",
        "description": "General purpose matting model",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512,
    },
    "BiRefNet-HR-matting": {
        # HF spelling: BiRefNet_HR-matting (underscore before HR).
        "repo_id": "ZhengPeng7/BiRefNet_HR-matting",
        "fallback_repo_id": "1038lab/BiRefNet",
        "fallback_filename": "BiRefNet-HR-matting.safetensors",
        "description": "High resolution matting model (2048x2048)",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024,
    },
    "BiRefNet_lite": {
        "repo_id": "ZhengPeng7/BiRefNet_lite",
        "fallback_repo_id": "1038lab/BiRefNet",
        "fallback_filename": "BiRefNet_lite.safetensors",
        "description": "Lightweight version for faster processing",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512,
    },
    "BiRefNet_lite-2K": {
        "repo_id": "ZhengPeng7/BiRefNet_lite-2K",
        "fallback_repo_id": "1038lab/BiRefNet",
        "fallback_filename": "BiRefNet_lite-2K.safetensors",
        "description": "Lightweight version optimized for 2K resolution",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024,
    },
    "BiRefNet_dynamic": {
        "repo_id": "ZhengPeng7/BiRefNet_dynamic",
        "fallback_repo_id": "1038lab/BiRefNet",
        "fallback_filename": "BiRefNet_dynamic.safetensors",
        "description": "Any-resolution model trained on 256..2304 (Mar 2025)",
        "default_res": 1024,
        "max_res": 2304,
        "min_res": 256,
    },
    "BiRefNet_dynamic-matting": {
        "repo_id": "ZhengPeng7/BiRefNet_dynamic-matting",
        # No 1038lab mirror for this Feb-2025 variant. If the primary repo
        # is gated for the user, they will get a clear error — there is no
        # alternative source we know of.
        "description": "Any-resolution matting (256..2304)",
        "default_res": 1024,
        "max_res": 2304,
        "min_res": 256,
    },
    "BiRefNet_lite-matting": {
        "repo_id": "ZhengPeng7/BiRefNet_lite-matting",
        "fallback_repo_id": "1038lab/BiRefNet",
        "fallback_filename": "BiRefNet_lite-matting.safetensors",
        "description": "Lightweight matting model",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512,
    },
}

# Utility functions
def _get_target_device():
    """Resolve the inference device strictly via ComfyUI's model_management.

    Previously this function silently overrode CPU back to cuda whenever CUDA
    was physically present, which broke `--cpu`, lowvram fallback, and
    multi-GPU index selection. Trusting `model_management.get_torch_device()`
    matches the documented ComfyUI contract — if the user asked for CPU, they
    get CPU; if ComfyUI chose `cuda:N`, that index is preserved.
    """
    try:
        return model_management.get_torch_device()
    except Exception as exc:
        logger.warning("%s Could not resolve ComfyUI device, using CPU: %s", _LOG_PREFIX, exc)
        return torch.device("cpu")


def _target_dtype(target_device):
    device_type = getattr(target_device, "type", str(target_device))
    return torch.float16 if device_type == "cuda" else torch.float32


_PRECISION_OPTIONS = ("auto", "fp16", "bf16", "fp32")


def _resolve_dtype(target_device, precision: str) -> "torch.dtype":
    """Map the user-facing precision combo to a torch dtype.

    'auto' picks bf16 when the GPU supports it (Ampere+ with Tensor Core
    bf16 — RTX 30/40, A100, H100, etc.), otherwise fp16. CPU always runs
    fp32 — half precision on CPU has no Tensor Core path and is markedly
    slower than fp32 in practice.

    For BiRefNet specifically use ``_resolve_birefnet_dtype`` instead —
    it patches the bf16 path because torchvision's ``deform_conv2d``
    (used in the HR variants' ASPP head) has no BF16 CUDA kernel.
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


_TEMPORAL_SMOOTH_OPTIONS = ("off", "median3", "median5", "ema_causal")


def _temporal_smooth_alphas(
    alphas: list,
    mode: str,
    ema_alpha: float,
) -> list:
    """Smooth alpha across the time axis to suppress per-frame edge wobble.

    Per-frame BiRefNet inference has no temporal model; identical objects
    in adjacent frames produce slightly different alphas → the edge
    "boils". Applied after inference on the already-computed alpha
    sequence:

    - ``median3`` / ``median5`` — N-frame temporal median. Best for
      random flicker; mild lag at clip boundaries (handled by
      ``mode="nearest"`` reflection in scipy).
    - ``ema_causal`` — exponential moving average. Causal, no lag, but a
      sudden alpha change blends with the past and can read as motion
      blur on fast objects.
    - ``off`` — passthrough.
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
            # Pure-numpy fallback if scipy isn't available.
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


def _resolve_birefnet_dtype(target_device, precision: str) -> "torch.dtype":
    """Like ``_resolve_dtype`` but downgrades bf16 → fp16.

    BiRefNet's HR variants run ``torchvision.ops.deform_conv2d`` inside the
    ASPP block. As of torchvision 0.20 / PyTorch 2.5 that op has no
    ``BFloat16`` CUDA kernel — running the model in bf16 raises
    ``"deformable_im2col" not implemented for 'BFloat16'``. We log the
    fallback (info for ``auto`` which picks bf16 silently; warning when the
    user explicitly chose ``bf16``) and use fp16 instead.
    """
    dtype = _resolve_dtype(target_device, precision)
    if dtype == torch.bfloat16:
        if precision == "bf16":
            logger.warning(
                "%s BiRefNet uses torchvision.deform_conv2d which has no "
                "BF16 CUDA kernel — falling back to FP16.",
                _LOG_PREFIX,
            )
        else:
            logger.info(
                "%s BiRefNet uses torchvision.deform_conv2d which has no "
                "BF16 CUDA kernel — auto-precision picks FP16.",
                _LOG_PREFIX,
            )
        return torch.float16
    return dtype


def _format_device_label(target_device):
    device_type = getattr(target_device, "type", str(target_device))
    if device_type == "cuda":
        index = getattr(target_device, "index", None)
        if index is None:
            index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        try:
            name = torch.cuda.get_device_name(index)
        except Exception:
            name = "unknown GPU"
        return f"cuda ({name})"
    return "cpu"


def _safe_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _update_progress(progress_bar, value, total=100):
    if progress_bar is not None:
        progress_bar.update_absolute(int(value), total=total)


@contextmanager
def _progress_pulse(progress_bar, start_step, cap_step, total_steps=100, interval=0.75):
    """No-op context manager. The previous implementation spawned a
    background thread per phase to animate the progress bar; on a
    multi-phase execute (download / load module / load weights / model.to /
    inference) that meant 5-7 thread spawn+join cycles costing 100-200 ms
    of overhead with no functional benefit. Phase boundaries still emit
    discrete `_update_progress` calls."""
    if progress_bar is not None and start_step:
        _update_progress(progress_bar, start_step, total_steps)
    yield


def _estimate_inference_chunk_size(batch_size, process_res, target_device):
    device_type = getattr(target_device, "type", str(target_device))
    if device_type != "cuda":
        return 1
    if process_res <= 512:
        return min(batch_size, 8)
    if process_res <= 1024:
        return min(batch_size, 4)
    if process_res <= 1536:
        return min(batch_size, 2)
    return 1


def _safe_module_name(model_name):
    return "".join(ch if ch.isalnum() else "_" for ch in model_name)


def _mask_to_pil(mask):
    mask_np = np.clip(mask.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_np)


def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        a = 255
    elif len(hex_color) == 8:
        r, g, b, a = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), int(hex_color[6:8], 16)
    else:
        raise ValueError("Invalid color format")
    return (r, g, b, a)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def handle_model_error(message, cause: Exception | None = None):
    """Log and raise a normalized RuntimeError. When called from inside an
    `except` block, pass the caught exception via ``cause`` so the original
    traceback is chained (`raise ... from cause`) — without it, debugging a
    BiRefNet failure means staring at a generic message with no origin.
    """
    logger.error("%s %s", _LOG_PREFIX, message)
    if cause is not None:
        raise RuntimeError(message) from cause
    raise RuntimeError(message)

def _make_swin_sdpa_forward(original_forward):
    """Create a drop-in ``WindowAttention.forward`` that uses SDPA.

    The original Swin window-attention implementation manually materialises
    a ``[B*nW, num_heads, N, N]`` attention-score matrix, adds
    ``relative_position_bias`` and the optional shifted-window ``mask``,
    softmaxes and multiplies by V. PyTorch's
    ``F.scaled_dot_product_attention`` does all of that with a fused kernel
    (Flash / mem-efficient / math) — we just have to assemble the additive
    ``attn_mask`` (bias + window mask) up-front.
    """
    import torch.nn.functional as _F

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        head_dim = C // self.num_heads
        # qkv: (3, B_, num_heads, N, head_dim)
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Relative position bias: [num_heads, N, N]
        rel_pos = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()

        if mask is not None:
            nW = mask.shape[0]
            # bias: [1, num_heads, N, N]  +  mask: [nW, 1, N, N]
            # -> [nW, num_heads, N, N], then tile over B groups of nW windows
            attn_mask = rel_pos.unsqueeze(0) + mask.unsqueeze(1)
            B = B_ // nW
            attn_mask = attn_mask.repeat(B, 1, 1, 1)
        else:
            # Single broadcastable bias for the whole batch.
            attn_mask = rel_pos.unsqueeze(0)

        # SDPA applies the scale internally; pass ``self.scale`` explicitly
        # to match the original ``(q * self.scale) @ k.T`` semantics.
        out = _F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=getattr(self, "attn_drop_prob", 0.0),
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    forward._ts_sdpa_patched = True
    forward._ts_original_forward = original_forward
    return forward


def _enable_birefnet_sdpa(model, model_module) -> str:
    """Enable SDPA across both PVT-v2 and Swin attention layers in a
    loaded BiRefNet model.

    Strategy:
      - Set the module-level ``config.SDPA_enabled = True``. In PVT-v2's
        ``Attention.forward`` this routes through a vanilla SDPA call
        (no relative-position bias involved — safe even in legacy code).
      - For Swin, the legacy SDPA branch drops both
        ``relative_position_bias`` and the shifted-window ``mask`` —
        instead of relying on that branch, we runtime-patch the
        ``WindowAttention.forward`` we actually see on the loaded model
        with an SDPA implementation that threads the bias and the mask
        through ``attn_mask`` correctly.

    Returns a short status string for logging:
        - ``"enabled (PVT + Swin)"``
        - ``"enabled (PVT)"`` if no Swin layers exist (pure PVT model)
        - ``"enabled (Swin)"`` if config flag could not be flipped
        - ``"failed"`` if neither path worked
    """
    cfg_flipped = False
    cfg = getattr(model_module, "config", None)
    if cfg is not None:
        try:
            cfg.SDPA_enabled = True
            cfg_flipped = True
        except Exception:
            pass

    # Patch every Swin WindowAttention class we encounter on this model.
    # Detection: it's the only attention block that owns
    # ``relative_position_bias_table`` plus a ``qkv`` linear and a ``proj``.
    patched_classes: set = set()
    for module in model.modules():
        if (
            hasattr(module, "relative_position_bias_table")
            and hasattr(module, "relative_position_index")
            and hasattr(module, "qkv")
            and hasattr(module, "proj")
        ):
            cls = type(module)
            if cls in patched_classes:
                continue
            patched_classes.add(cls)
            current_forward = cls.forward
            if getattr(current_forward, "_ts_sdpa_patched", False):
                continue
            cls.forward = _make_swin_sdpa_forward(current_forward)

    if cfg_flipped and patched_classes:
        return f"enabled (PVT + Swin x{len(patched_classes)})"
    if cfg_flipped:
        return "enabled (PVT)"
    if patched_classes:
        return f"enabled (Swin x{len(patched_classes)})"
    return "failed"


def _resolve_birefnet_cache_dir() -> str:
    """Pick the first registered `birefnet` folder, falling back to the
    default `models/BiRefNet/` only when nothing else is registered. This
    lets `extra_model_paths.yaml` redirect BiRefNet weights to a shared
    network mount the same way it works for any other ComfyUI model type.
    """
    try:
        registered = folder_paths.get_folder_paths("birefnet")
    except Exception:
        registered = []
    if registered:
        return registered[0]
    return os.path.join(folder_paths.models_dir, "BiRefNet")


class BiRefNetModel:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.current_device = None
        self.current_dtype = None
        self.base_cache_dir = _resolve_birefnet_cache_dir()

    def get_cache_dir(self, model_name):
        """Per-variant sub-directory so different models' bundled birefnet.py
        files never overwrite each other."""
        return os.path.join(self.base_cache_dir, model_name)

    @staticmethod
    def _find_weights_file(cache_dir):
        """Return the first weights file in ``cache_dir`` (safetensors first,
        then .bin). HF repos use the canonical ``model.safetensors`` name but
        some legacy mirrors used variant-specific names — accept both."""
        if not os.path.isdir(cache_dir):
            return None
        for name in os.listdir(cache_dir):
            if name.endswith(".safetensors"):
                return os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir):
            if name.endswith(".bin"):
                return os.path.join(cache_dir, name)
        return None

    @staticmethod
    def _find_model_py(cache_dir):
        """Find the BiRefNet Python module shipped with the checkpoint:
        ``birefnet.py`` for full models, ``birefnet_lite.py`` for lite ones."""
        if not os.path.isdir(cache_dir):
            return None
        for candidate in ("birefnet.py", "birefnet_lite.py"):
            path = os.path.join(cache_dir, candidate)
            if os.path.isfile(path):
                return path
        return None

    def check_model_cache(self, model_name):
        cache_dir = self.get_cache_dir(model_name)
        if not os.path.exists(cache_dir):
            return False, "Model directory not found"
        if not os.path.isfile(os.path.join(cache_dir, "config.json")):
            return False, "Missing config.json"
        if self._find_weights_file(cache_dir) is None:
            return False, "Missing model weights (.safetensors / .bin)"
        if self._find_model_py(cache_dir) is None:
            return False, "Missing birefnet.py / birefnet_lite.py"
        return True, "Model cache verified"

    def download_model(self, model_name, progress_bar=None, start_step=5, end_step=30):
        """Download a BiRefNet variant.

        Tries the configured ``repo_id`` first (typically ``ZhengPeng7/…``,
        which ships the up-to-date ``birefnet.py`` with the post-Sep-2025
        SDPA rewrite). If that fails (most commonly because the upstream
        repo is gated or archived — late-2025 ZhengPeng7 closed several
        of them), falls back to the ``1038lab/BiRefNet`` aggregated
        mirror, which still serves the weights publicly under a single
        repo with variant-specific filenames.

        After a fallback download the runtime SDPA check sees the older
        ``birefnet.py`` and stays disabled — quality is preserved, just no
        speedup.
        """
        config = MODEL_CONFIG[model_name]
        cache_dir = self.get_cache_dir(model_name)
        primary_repo = str(config.get("repo_id") or "")
        fallback_repo = config.get("fallback_repo_id")
        fallback_filename = config.get("fallback_filename")

        os.makedirs(cache_dir, exist_ok=True)

        # Lazy: huggingface_hub pulls in HTTP/auth/cache infra. Only
        # imported on the first download attempt.
        from huggingface_hub import snapshot_download

        # When the primary repo is actually the 1038lab aggregated mirror
        # (because some variants — e.g. ``BiRefNet_512x512`` — exist only
        # there) we must restrict the snapshot to the right filename;
        # otherwise snapshot_download would pull every .safetensors in
        # that ~2 GB repo. We use the same narrow patterns as the
        # fallback path in that case.
        primary_is_mirror = (
            primary_repo == fallback_repo
            or primary_repo.split("/", 1)[0] == "1038lab"
        )
        if primary_is_mirror and fallback_filename:
            primary_patterns = [
                fallback_filename,
                "birefnet.py",
                "birefnet_lite.py",
                "BiRefNet_config.py",
                "config.json",
            ]
        else:
            primary_patterns = [
                "*.json", "*.py", "*.safetensors", "*.bin", "*.txt",
            ]

        primary_exc: Exception | None = None
        if primary_repo:
            logger.info(
                "%s Downloading %s from %s into %s",
                _LOG_PREFIX, model_name, primary_repo, cache_dir,
            )
            try:
                _update_progress(progress_bar, start_step)
                with _progress_pulse(progress_bar, start_step, max(start_step, end_step - 1)):
                    snapshot_download(
                        repo_id=primary_repo,
                        local_dir=cache_dir,
                        local_dir_use_symlinks=False,
                        allow_patterns=primary_patterns,
                    )
                _update_progress(progress_bar, end_step)
                return True, f"Downloaded from {primary_repo}"
            except Exception as exc:
                primary_exc = exc
                logger.warning(
                    "%s Primary repo %s failed: %s",
                    _LOG_PREFIX, primary_repo, str(exc).split("\n", 1)[0][:160],
                )

        if not fallback_repo or not fallback_filename:
            return False, (
                f"Error downloading {model_name} from {primary_repo}: "
                f"{primary_exc}. No fallback repo configured for this variant."
            )

        logger.info(
            "%s Falling back to mirror %s (file %s)",
            _LOG_PREFIX, fallback_repo, fallback_filename,
        )
        try:
            # The 1038lab mirror keeps every variant's weights in one repo
            # with a variant-specific filename, plus a single shared
            # ``birefnet.py``/``birefnet_lite.py``/``BiRefNet_config.py``.
            # We pull just the files we actually need.
            with _progress_pulse(progress_bar, start_step, max(start_step, end_step - 1)):
                snapshot_download(
                    repo_id=fallback_repo,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False,
                    allow_patterns=[
                        fallback_filename,
                        "birefnet.py",
                        "birefnet_lite.py",
                        "BiRefNet_config.py",
                        "config.json",
                    ],
                )
            _update_progress(progress_bar, end_step)
            return True, f"Downloaded from fallback mirror {fallback_repo}"
        except Exception as fb_exc:
            return False, (
                f"Error downloading {model_name}. Primary ({primary_repo}): "
                f"{primary_exc}. Fallback ({fallback_repo}): {fb_exc}"
            )

    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            self.current_model_version = None
            self.current_device = None
            self.current_dtype = None
            _safe_empty_cache()
            logger.info("%s Model cleared from memory", _LOG_PREFIX)

    def load_model(self, model_name, progress_bar=None, start_step=30, end_step=55, target_device=None, target_dtype=None):
        if target_device is None:
            target_device = _get_target_device()
        if target_dtype is None:
            target_dtype = _target_dtype(target_device)
        target_device_key = str(target_device)

        if (
            self.model is None
            or self.current_model_version != model_name
            or self.current_device != target_device_key
            or self.current_dtype != target_dtype
        ):
            self.clear_model()

            cache_dir = self.get_cache_dir(model_name)
            model_path = self._find_model_py(cache_dir)
            if model_path is None:
                handle_model_error(
                    f"Could not find birefnet.py / birefnet_lite.py in {cache_dir}. "
                    "Re-run the node to re-download the model."
                )
            config_path = os.path.join(cache_dir, "BiRefNet_config.py")
            if not os.path.isfile(config_path):
                handle_model_error(
                    f"Could not find BiRefNet_config.py in {cache_dir}. "
                    "Re-run the node to re-download the model."
                )
            weights_path = self._find_weights_file(cache_dir)
            if weights_path is None:
                handle_model_error(
                    f"Could not find weights (.safetensors / .bin) in {cache_dir}. "
                    "Re-run the node to re-download the model."
                )
            model_filename = os.path.basename(model_path)
            weights_filename = os.path.basename(weights_path)

            try:
                package_name = f"_ts_birefnet_{_safe_module_name(model_name)}"
                package_module = types.ModuleType(package_name)
                package_module.__path__ = [cache_dir]
                sys.modules[package_name] = package_module

                _update_progress(progress_bar, start_step)

                spec = importlib.util.spec_from_file_location(f"{package_name}.BiRefNet_config", config_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError("Could not load BiRefNet_config module")
                config_module = importlib.util.module_from_spec(spec)
                sys.modules[f"{package_name}.BiRefNet_config"] = config_module
                sys.modules["BiRefNet_config"] = config_module
                spec.loader.exec_module(config_module)

                model_module_name = os.path.splitext(model_filename)[0]
                spec = importlib.util.spec_from_file_location(f"{package_name}.{model_module_name}", model_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError("Could not load BiRefNet model module")
                model_module = importlib.util.module_from_spec(spec)
                sys.modules[f"{package_name}.{model_module_name}"] = model_module
                sys.modules[model_module_name] = model_module
                with _progress_pulse(progress_bar, start_step + 2, start_step + 8):
                    spec.loader.exec_module(model_module)

                _update_progress(progress_bar, start_step + 10)
                self.model = model_module.BiRefNet(config_module.BiRefNetConfig())

                with _progress_pulse(progress_bar, start_step + 12, end_step - 8):
                    # Accept both safetensors (preferred) and legacy
                    # .bin pickles. Lazy import keeps cold startup fast.
                    if weights_path.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        state_dict = load_file(weights_path)
                    else:
                        state_dict = torch.load(
                            weights_path, map_location="cpu", weights_only=True
                        )
                self.model.load_state_dict(state_dict)

                self.model.eval()
                if target_dtype == torch.float16:
                    self.model.half()
                    torch.set_float32_matmul_precision('high')
                elif target_dtype == torch.bfloat16:
                    self.model.to(dtype=torch.bfloat16)
                    torch.set_float32_matmul_precision('high')
                else:
                    self.model.float()

                with _progress_pulse(progress_bar, end_step - 8, end_step - 1):
                    self.model.to(target_device)

                # Channels-last layout helps the BiRefNet decoder's conv
                # stack hit Tensor Cores cleanly. One-time cost at load.
                try:
                    self.model = self.model.to(memory_format=torch.channels_last)
                except Exception as exc:
                    logger.warning(
                        "%s channels_last conversion failed (continuing): %s",
                        _LOG_PREFIX, exc,
                    )

                # SDPA: PVT-v2 attention is wired through the module-level
                # ``config.SDPA_enabled`` flag and the bundled SDPA branch
                # there has always been correct. Swin's legacy SDPA branch
                # would silently drop ``relative_position_bias`` and the
                # shifted-window mask, so instead of trusting it we
                # runtime-patch ``WindowAttention.forward`` on the loaded
                # model with our own SDPA implementation that threads both
                # through ``attn_mask``. Works regardless of how old the
                # downloaded ``birefnet.py`` is.
                sdpa_status = _enable_birefnet_sdpa(self.model, model_module)
                logger.info("%s SDPA: %s", _LOG_PREFIX, sdpa_status)

                self.current_model_version = model_name
                self.current_device = target_device_key
                self.current_dtype = target_dtype
                _update_progress(progress_bar, end_step)

            except Exception as e:
                handle_model_error(f"Error loading BiRefNet model: {str(e)}", cause=e)

    def _process_mask_chunk(self, image_chunk, process_res, target_device, target_dtype):
        if image_chunk.ndim != 4 or image_chunk.shape[-1] < 3:
            raise ValueError(f"Expected IMAGE tensor [B, H, W, C>=3], got {tuple(image_chunk.shape)}")

        _, height, width, _ = image_chunk.shape
        input_tensor = image_chunk[..., :3].detach().movedim(-1, 1).to(
            device=target_device,
            dtype=target_dtype,
            non_blocking=True,
        )
        input_tensor = F.interpolate(input_tensor, size=(process_res, process_res), mode="bicubic", align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=target_device, dtype=target_dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=target_device, dtype=target_dtype).view(1, 3, 1, 1)
        input_tensor = (input_tensor - mean) / std
        # Match the model's memory layout — set by load_model() above. The
        # contiguous() call is a no-op when already in channels_last.
        try:
            input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass

        with torch.inference_mode():
            preds = self.model(input_tensor)
            pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            elif pred.ndim == 4 and pred.shape[1] != 1 and pred.shape[-1] == 1:
                pred = pred.movedim(-1, 1)
            pred = pred.sigmoid()

        # CPU + PIL bicubic resize: at 4K/8K output sizes, GPU bicubic on
        # FP32 is dominated by the cast + interpolate kernel; PIL's libjpeg-
        # tuned bicubic on CPU comes out ahead, especially when the
        # postprocessing loop is CPU-bound anyway (PIL filters / merge).
        pred_cpu = pred[:, 0].float().cpu().numpy()
        out = np.empty((pred_cpu.shape[0], height, width), dtype=np.float32)
        for i in range(pred_cpu.shape[0]):
            arr = np.clip(pred_cpu[i] * 255.0, 0, 255).astype(np.uint8)
            pil_mask = Image.fromarray(arr).resize((width, height), Image.BICUBIC)
            out[i] = np.asarray(pil_mask, dtype=np.float32) / 255.0
        return torch.from_numpy(out)

    def process_masks(self, image, params, progress_bar=None, start_step=0, end_step=100, target_device=None, target_dtype=None):
        """Run BiRefNet on a batch of frames.

        ``progress_bar`` is updated with ``(processed, total=batch_size)`` so
        the ComfyUI UI shows "frame N / batch" — much more informative than
        a percentage span. ``start_step``/``end_step`` are kept for backward
        compatibility with older callers but no longer drive the bar.
        """
        try:
            if self.model is None:
                raise RuntimeError("BiRefNet model is not loaded")

            if image.ndim != 4:
                raise ValueError(f"Expected IMAGE tensor [B, H, W, C], got {tuple(image.shape)}")

            batch_size = image.shape[0]
            process_res = params["process_res"]
            if target_device is None:
                target_device = _get_target_device()
            if target_dtype is None:
                target_dtype = _target_dtype(target_device)
            chunk_size = _estimate_inference_chunk_size(batch_size, process_res, target_device)

            if progress_bar is not None:
                try:
                    progress_bar.update_absolute(0, total=batch_size)
                except Exception:
                    pass

            masks = []
            processed = 0
            while processed < batch_size:
                current_chunk_size = min(chunk_size, batch_size - processed)
                chunk = image[processed:processed + current_chunk_size]

                try:
                    masks.append(self._process_mask_chunk(chunk, process_res, target_device, target_dtype))
                    processed += current_chunk_size
                    if progress_bar is not None:
                        try:
                            progress_bar.update_absolute(processed, total=batch_size)
                        except Exception:
                            pass
                except torch.cuda.OutOfMemoryError:
                    _safe_empty_cache()
                    if current_chunk_size <= 1:
                        raise
                    chunk_size = max(1, current_chunk_size // 2)
                    logger.warning(
                        "%s CUDA OOM at batch chunk %s, retrying with chunk size %s",
                        _LOG_PREFIX,
                        current_chunk_size,
                        chunk_size,
                    )

            return torch.cat(masks, dim=0)

        except Exception as e:
            handle_model_error(f"Error in BiRefNet processing: {str(e)}", cause=e)

class _BiRefNetState:
    """Module-level mutable state. ComfyUI V3 `lock_class` blocks
    `cls._x = ...` on registered nodes, so the cached model lives here
    instead of on the node class."""
    model: "BiRefNetModel | None" = None


_state = _BiRefNetState()


class TS_BGRM_BiRefNet(IO.ComfyNode):
    @classmethod
    def _ensure_model(cls):
        if _state.model is None:
            _state.model = BiRefNetModel()
        return _state.model

    @classmethod
    def define_schema(cls) -> IO.Schema:
        tooltips = {
            "enable": "Enable or disable the background removal process. If disabled, the original image will be passed through.",
            "image": "Input image to be processed for background removal.",
            "model": "Select the BiRefNet model variant to use.",
            "use_custom_resolution": "Enable to use a custom resolution specified below. If disabled, the model's default resolution will be used.",
            "process_resolution": "The resolution for processing the image. It will be adjusted to the nearest multiple of 64.",
            "mask_blur": "Specify the amount of blur to apply to the mask edges (0 for no blur, higher values for more blur).",
            "mask_offset": "Adjust the mask boundary (positive values expand the mask, negative values shrink it).",
            "invert_output": "Enable to invert both the image and mask output (useful for certain effects).",
            "background": "Choose background type: Alpha (transparent) or Color (custom background color).",
            "background_color": "Background color when 'background' is set to 'Color'. COLOR widget supports precise eyedropper picking.",
            "precision": "Inference precision on CUDA. 'auto' picks bf16 on Ampere+ (more numerically robust) and fp16 elsewhere. Force 'fp32' for diagnostics. CPU always runs fp32.",
            "temporal_smooth": "Smooth alpha across frames to reduce 'boiling' edges in video. 'median3' (default) kills random 1-frame flicker; 'median5' is stronger (2-frame lag at clip boundaries); 'ema_causal' is causal averaging (no lag, can blur fast motion); 'off' disables. No-op for single images. Adds ~N*H*W*4 bytes RAM during the pass.",
            "ema_alpha": "Strength of the causal EMA when temporal_smooth = 'ema_causal'. Higher = more smoothing (more lag for moving objects); lower = closer to the raw per-frame alpha.",
        }
        return IO.Schema(
            node_id="TS_BGRM_BiRefNet",
            display_name="TS Remove Background",
            category="TS/Image",
            inputs=[
                IO.Image.Input("image", tooltip=tooltips["image"]),
                IO.Boolean.Input("enable", default=True, tooltip=tooltips["enable"]),
                IO.Combo.Input("model", options=list(MODEL_CONFIG.keys()), default="BiRefNet-HR-matting", tooltip=tooltips["model"]),
                IO.Boolean.Input("use_custom_resolution", default=False, optional=True, tooltip=tooltips["use_custom_resolution"]),
                IO.Int.Input("process_resolution", default=1024, min=256, max=4096, step=64, optional=True, tooltip=tooltips["process_resolution"]),
                IO.Int.Input("mask_blur", default=0, min=0, max=64, step=1, optional=True, tooltip=tooltips["mask_blur"]),
                IO.Int.Input("mask_offset", default=0, min=-20, max=20, step=1, optional=True, tooltip=tooltips["mask_offset"]),
                IO.Boolean.Input("invert_output", default=False, optional=True, tooltip=tooltips["invert_output"]),
                IO.Combo.Input("background", options=["Alpha", "Color"], default="Alpha", optional=True, tooltip=tooltips["background"]),
                IO.Color.Input("background_color", default="#ffffff", optional=True, tooltip=tooltips["background_color"]),
                IO.Combo.Input("precision", options=list(_PRECISION_OPTIONS), default="auto", optional=True, tooltip=tooltips["precision"]),
                IO.Combo.Input("temporal_smooth", options=list(_TEMPORAL_SMOOTH_OPTIONS), default="median3", optional=True, tooltip=tooltips["temporal_smooth"]),
                IO.Float.Input("ema_alpha", default=0.5, min=0.0, max=0.99, step=0.01, optional=True, tooltip=tooltips["ema_alpha"]),
            ],
            outputs=[
                IO.Image.Output(display_name="IMAGE"),
                IO.Mask.Output(display_name="MASK"),
                IO.Image.Output(display_name="MASK_IMAGE"),
            ],
        )

    @classmethod
    def execute(cls, image, enable, model, use_custom_resolution=False, process_resolution=1024, **params) -> IO.NodeOutput:
        params = {
            "mask_blur": 0,
            "mask_offset": 0,
            "invert_output": False,
            "background": "Alpha",
            "background_color": "#ffffff",
            "precision": "auto",
            "temporal_smooth": "median3",
            "ema_alpha": 0.5,
            **params,
        }
        precision = str(params.get("precision", "auto") or "auto")
        if precision not in _PRECISION_OPTIONS:
            logger.warning(
                "%s Unknown precision '%s', falling back to 'auto'.",
                _LOG_PREFIX, precision,
            )
            precision = "auto"
        temporal_smooth = str(params.get("temporal_smooth", "median3") or "median3")
        if temporal_smooth not in _TEMPORAL_SMOOTH_OPTIONS:
            logger.warning(
                "%s Unknown temporal_smooth '%s', falling back to 'median3'.",
                _LOG_PREFIX, temporal_smooth,
            )
            temporal_smooth = "median3"
        try:
            ema_alpha = float(params.get("ema_alpha", 0.5))
        except (TypeError, ValueError):
            ema_alpha = 0.5

        if not enable:
            b, h, w, c = image.shape
            mask_output = torch.ones((b, h, w), dtype=torch.float32, device=image.device)
            mask_image_output = torch.ones((b, h, w, 3), dtype=torch.float32, device=image.device)
            return IO.NodeOutput(image, mask_output, mask_image_output)

        try:
            model_config = MODEL_CONFIG[model]

            if use_custom_resolution:
                # Use the user-provided resolution, ensuring it's a multiple of 64
                process_res = max(64, (int(process_resolution) // 64) * 64)
            else:
                # Use the default resolution from the model's config
                process_res = model_config.get("default_res", 1024)
                if model_config.get("force_res", False):
                    base_res = 512
                    process_res = ((process_res + base_res - 1) // base_res) * base_res
                else:
                    process_res = process_res // 32 * 32

            logger.info("%s Using %s model with %s resolution", _LOG_PREFIX, model, process_res)
            params["process_res"] = process_res

            processed_images = []
            processed_masks = []
            bg_model = cls._ensure_model()

            # ---- Phase 1: download model files (only if missing) ----
            cache_status, message = bg_model.check_model_cache(model)
            if not cache_status:
                logger.info("%s Cache check: %s", _LOG_PREFIX, message)
                dl_pbar = ProgressBar(100)
                _update_progress(dl_pbar, 0)
                download_status, download_message = bg_model.download_model(
                    model, progress_bar=dl_pbar, start_step=0, end_step=100,
                )
                if not download_status:
                    handle_model_error(download_message)
                _update_progress(dl_pbar, 100)
                logger.info("%s Model files downloaded successfully", _LOG_PREFIX)

            target_device = _get_target_device()
            target_dtype = _resolve_birefnet_dtype(target_device, precision)
            logger.info(
                "%s Processing device: %s precision=%s dtype=%s",
                _LOG_PREFIX, _format_device_label(target_device),
                precision, str(target_dtype),
            )

            # ---- Phase 2: load model into VRAM (skipped if already cached) ----
            # ``getattr`` lets a slim test double (no internal attrs) flow
            # through this branch — it will simply re-load every call.
            already_loaded = (
                getattr(bg_model, "model", None) is not None
                and getattr(bg_model, "current_model_version", None) == model
                and getattr(bg_model, "current_device", None) == str(target_device)
                and getattr(bg_model, "current_dtype", None) == target_dtype
            )
            if not already_loaded:
                load_pbar = ProgressBar(100)
                _update_progress(load_pbar, 0)
                bg_model.load_model(
                    model,
                    progress_bar=load_pbar,
                    start_step=0,
                    end_step=100,
                    target_device=target_device,
                    target_dtype=target_dtype,
                )

            batch_size = image.shape[0]

            # ---- Phase 3: BiRefNet inference (frame N / batch_size) ----
            inf_pbar = ProgressBar(batch_size)
            _update_progress(inf_pbar, 0, total=batch_size)
            masks = bg_model.process_masks(
                image, params, progress_bar=inf_pbar,
                target_device=target_device, target_dtype=target_dtype,
            )

            # Temporal smoothing across frames — skipped for single images.
            # Operates on the raw BiRefNet output before per-frame
            # blur / offset / invert / composite so those see a stable
            # alpha sequence too. No dedicated progress bar — fast op.
            if temporal_smooth != "off" and batch_size > 1:
                logger.info(
                    "%s Temporal smoothing: mode=%s ema_alpha=%.2f frames=%d",
                    _LOG_PREFIX, temporal_smooth, float(ema_alpha), batch_size,
                )
                masks_np = masks.detach().cpu().numpy().astype(np.float32)
                smoothed = _temporal_smooth_alphas(
                    [masks_np[i] for i in range(batch_size)],
                    temporal_smooth,
                    float(ema_alpha),
                )
                masks = torch.from_numpy(np.stack(smoothed, axis=0))

            # ---- Phase 4: per-frame post-process + composite ----
            post_pbar = ProgressBar(batch_size)
            _update_progress(post_pbar, 0, total=batch_size)

            for index, img in enumerate(image):
                mask = _mask_to_pil(masks[index])
                if params["mask_blur"] > 0:
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=params["mask_blur"]))
                if params["mask_offset"] != 0:
                    if params["mask_offset"] > 0:
                        for _ in range(params["mask_offset"]):
                            mask = mask.filter(ImageFilter.MaxFilter(3))
                    else:
                        for _ in range(-params["mask_offset"]):
                            mask = mask.filter(ImageFilter.MinFilter(3))
                if params["invert_output"]:
                    mask = Image.fromarray(255 - np.array(mask))
                orig_image = tensor2pil(img)
                orig_rgba = orig_image.convert("RGBA")
                r, g, b, _ = orig_rgba.split()
                foreground = Image.merge('RGBA', (r, g, b, mask))
                if params["background"] == "Alpha":
                    processed_images.append(pil2tensor(foreground))
                else:
                    background_color = params.get("background_color", "#ffffff")
                    rgba = hex_to_rgba(background_color)
                    bg_image = Image.new('RGBA', orig_image.size, rgba)
                    composite_image = Image.alpha_composite(bg_image, foreground)
                    processed_images.append(pil2tensor(composite_image.convert("RGB")))
                processed_masks.append(pil2tensor(mask))
                _update_progress(post_pbar, index + 1, total=batch_size)

            image_output = torch.cat(processed_images, dim=0)
            mask_output = torch.cat(processed_masks, dim=0)
            mask_image_output = mask_output.unsqueeze(-1).expand(-1, -1, -1, 3)
            return IO.NodeOutput(image_output, mask_output, mask_image_output)
        except Exception as e:
            handle_model_error(f"Error in image processing: {str(e)}", cause=e)

# Node Mapping
NODE_CLASS_MAPPINGS = {
    "TS_BGRM_BiRefNet": TS_BGRM_BiRefNet
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_BGRM_BiRefNet": "TS Remove Background"
}

