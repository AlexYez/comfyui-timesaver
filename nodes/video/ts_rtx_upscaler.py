import logging
from enum import Enum

import comfy.model_management as model_management
import comfy.utils
import torch

from comfy_api.v0_0_2 import IO

try:
    import nvvfx
    TS_NVVFX_IMPORT_ERROR = None
except Exception as import_error:
    nvvfx = None
    TS_NVVFX_IMPORT_ERROR = import_error

logger = logging.getLogger("comfyui_timesaver.ts_rtx_upscaler")
LOG_PREFIX = "[TS RTX Upscaler]"


class TS_UpscaleType(str, Enum):
    SCALE_BY = "scale by multiplier"
    TARGET_DIMENSIONS = "target dimensions"


class TS_RTX_Upscaler(IO.ComfyNode):
    MAX_PIXELS_PER_BATCH = 1024 * 1024 * 16

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_RTX_Upscaler",
            display_name="TS RTX Upscaler",
            category="TS/Video",
            inputs=[
                IO.Image.Input("images"),
                IO.Combo.Input(
                    "resize_type",
                    options=[TS_UpscaleType.SCALE_BY.value, TS_UpscaleType.TARGET_DIMENSIONS.value],
                    default=TS_UpscaleType.SCALE_BY.value,
                ),
                IO.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.01),
                IO.Int.Input("width", default=1920, min=64, max=8192, step=8),
                IO.Int.Input("height", default=1080, min=64, max=8192, step=8),
                IO.Combo.Input("quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="ULTRA"),
            ],
            outputs=[IO.Image.Output(display_name="upscaled_images")],
        )

    @classmethod
    def validate_inputs(cls, **_):
        return True

    @classmethod
    def execute(cls, images, resize_type, scale, width, height, quality) -> IO.NodeOutput:
        cls._ensure_runtime_ready()
        cls._validate_images(images)

        output_width, output_height = cls._resolve_output_size(
            images=images,
            resize_type=resize_type,
            scale=scale,
            width=width,
            height=height,
        )
        quality_level = cls._resolve_quality_level(quality)
        batch_size = cls._resolve_batch_size(output_width, output_height)

        logger.info(
            "%s input=%s target=%dx%d quality=%s batch_size=%d",
            LOG_PREFIX,
            tuple(images.shape),
            output_width,
            output_height,
            quality,
            batch_size,
        )

        rgb_images, alpha_images = cls._split_alpha(images)
        upscaled_rgb = cls._run_nvvfx_upscale(
            images_rgb=rgb_images,
            output_width=output_width,
            output_height=output_height,
            quality_level=quality_level,
            batch_size=batch_size,
        )

        if alpha_images is not None:
            upscaled_alpha = cls._resize_alpha(alpha_images, output_width, output_height)
            final_images = torch.cat([upscaled_rgb, upscaled_alpha], dim=-1)
        else:
            final_images = upscaled_rgb

        final_images = final_images.clamp(0.0, 1.0).to(torch.float32).cpu()

        logger.info("%s output=%s", LOG_PREFIX, tuple(final_images.shape))
        return IO.NodeOutput(final_images)

    @staticmethod
    def _ensure_runtime_ready():
        if nvvfx is None:
            message = (
                "[TS RTX Upscaler] nvidia-vfx is not installed. Install it with "
                "`pip install nvidia-vfx` in your ComfyUI environment."
            )
            if TS_NVVFX_IMPORT_ERROR is not None:
                message = f"{message} Original import error: {TS_NVVFX_IMPORT_ERROR}"
            raise RuntimeError(message)

        if not torch.cuda.is_available():
            raise RuntimeError("[TS RTX Upscaler] CUDA is required. No CUDA device is available.")

        device = model_management.get_torch_device()
        if "cuda" not in str(device).lower():
            raise RuntimeError(f"[TS RTX Upscaler] Expected CUDA device, got: {device}")

    @staticmethod
    def _validate_images(images):
        if not isinstance(images, torch.Tensor):
            raise TypeError("[TS RTX Upscaler] `images` must be a torch.Tensor.")
        if images.ndim != 4:
            raise ValueError(
                f"[TS RTX Upscaler] Expected IMAGE tensor shape [B,H,W,C], got: {tuple(images.shape)}"
            )
        if images.shape[0] <= 0:
            raise ValueError("[TS RTX Upscaler] Batch is empty.")
        if images.shape[-1] not in (3, 4):
            raise ValueError(
                f"[TS RTX Upscaler] Supported channels are 3 or 4, got: {images.shape[-1]}"
            )

    @classmethod
    def _resolve_output_size(cls, images, resize_type, scale, width, height):
        _, input_h, input_w, _ = images.shape

        if resize_type == TS_UpscaleType.SCALE_BY.value:
            out_w = int(round(input_w * float(scale)))
            out_h = int(round(input_h * float(scale)))
        elif resize_type == TS_UpscaleType.TARGET_DIMENSIONS.value:
            out_w = int(width)
            out_h = int(height)
        else:
            raise ValueError(f"[TS RTX Upscaler] Unsupported resize_type: {resize_type}")

        out_w = cls._align_to_8(out_w)
        out_h = cls._align_to_8(out_h)
        return out_w, out_h

    @classmethod
    def _resolve_batch_size(cls, output_width, output_height):
        out_pixels = output_width * output_height
        if out_pixels <= 0:
            raise ValueError("[TS RTX Upscaler] Invalid output dimensions.")
        return max(1, cls.MAX_PIXELS_PER_BATCH // out_pixels)

    @staticmethod
    def _resolve_quality_level(quality):
        effects = getattr(nvvfx, "effects", None)
        quality_level = getattr(effects, "QualityLevel", None) if effects is not None else None
        if quality_level is None:
            raise RuntimeError("[TS RTX Upscaler] nvidia-vfx QualityLevel enum was not found.")

        mapping = {
            "LOW": getattr(quality_level, "LOW", None),
            "MEDIUM": getattr(quality_level, "MEDIUM", None),
            "HIGH": getattr(quality_level, "HIGH", None),
            "ULTRA": getattr(quality_level, "ULTRA", None),
        }

        selected = mapping.get(quality)
        if selected is not None:
            return selected

        for fallback_name in ("HIGH", "MEDIUM", "LOW", "ULTRA"):
            if mapping.get(fallback_name) is not None:
                logger.warning(
                    "%s quality=%s not available. Fallback to %s.",
                    LOG_PREFIX,
                    quality,
                    fallback_name,
                )
                return mapping[fallback_name]

        raise RuntimeError("[TS RTX Upscaler] No supported quality levels found in nvidia-vfx.")

    @classmethod
    def _run_nvvfx_upscale(cls, images_rgb, output_width, output_height, quality_level, batch_size):
        device = model_management.get_torch_device()
        upscaled_batches = []

        with nvvfx.VideoSuperRes(quality_level) as super_res:
            super_res.output_width = output_width
            super_res.output_height = output_height
            super_res.load()

            for start in range(0, images_rgb.shape[0], batch_size):
                batch = images_rgb[start:start + batch_size]
                batch_cuda = batch.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous()

                batch_outputs = []
                for frame_idx in range(batch_cuda.shape[0]):
                    frame_chw = batch_cuda[frame_idx]
                    dlpack_output = super_res.run(frame_chw).image
                    output = torch.from_dlpack(dlpack_output).clone()
                    output_hwc = cls._to_hwc(output)
                    batch_outputs.append(cls._normalize_output(output_hwc))

                upscaled_batches.append(torch.stack(batch_outputs, dim=0).cpu())

        if not upscaled_batches:
            raise RuntimeError("[TS RTX Upscaler] Upscaler produced no output frames.")

        return torch.cat(upscaled_batches, dim=0)

    @staticmethod
    def _split_alpha(images):
        if images.shape[-1] == 4:
            return images[..., :3], images[..., 3:4]
        return images, None

    @staticmethod
    def _resize_alpha(alpha_images, output_width, output_height):
        alpha_nchw = alpha_images.permute(0, 3, 1, 2).contiguous()
        resized_alpha = comfy.utils.common_upscale(
            alpha_nchw, output_width, output_height, "bilinear", crop="disabled"
        )
        return resized_alpha.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _to_hwc(frame):
        if frame.ndim != 3:
            raise ValueError(f"[TS RTX Upscaler] Unexpected frame shape from nvidia-vfx: {tuple(frame.shape)}")

        if frame.shape[0] in (1, 3, 4):
            frame = frame.permute(1, 2, 0)
        elif frame.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"[TS RTX Upscaler] Unsupported frame layout from nvidia-vfx: {tuple(frame.shape)}"
            )

        if frame.shape[-1] == 1:
            frame = frame.repeat(1, 1, 3)
        elif frame.shape[-1] == 4:
            frame = frame[..., :3]
        elif frame.shape[-1] != 3:
            raise ValueError(
                f"[TS RTX Upscaler] Unexpected channel count after conversion: {frame.shape[-1]}"
            )

        return frame.contiguous()

    @staticmethod
    def _normalize_output(frame):
        if torch.is_floating_point(frame):
            return frame.to(torch.float32)
        return frame.to(torch.float32) / 255.0

    @staticmethod
    def _align_to_8(value):
        return max(8, int(round(float(value) / 8.0) * 8))


NODE_CLASS_MAPPINGS = {
    "TS_RTX_Upscaler": TS_RTX_Upscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_RTX_Upscaler": "TS RTX Upscaler",
}
