"""TS Video Upscale With Model — frame-by-frame upscaling with Spandrel-loaded models.

node_id: TS_Video_Upscale_With_Model
"""

import gc
import logging
import os

import torch

import comfy.model_management as model_management
import comfy.utils
import folder_paths

from comfy_api.v0_0_2 import IO

logger = logging.getLogger("comfyui_timesaver.ts_video_upscale_with_model")
LOG_PREFIX = "[TS Video Upscale]"

try:
    from spandrel import ModelLoader
except ImportError:
    logger.warning("%s Spandrel library not found. Please install spandrel.", LOG_PREFIX)
    ModelLoader = None


_UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic"]


class TS_Video_Upscale_With_Model(IO.ComfyNode):
    """Memory-efficient per-frame upscaler with three device strategies."""

    upscale_methods = _UPSCALE_METHODS

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Video_Upscale_With_Model",
            display_name="TS Video Upscale With Model",
            category="TS/Video",
            inputs=[
                IO.Combo.Input("model_name", options=folder_paths.get_filename_list("upscale_models")),
                IO.Image.Input("images"),
                IO.Combo.Input("upscale_method", options=_UPSCALE_METHODS),
                IO.Float.Input("factor", default=2.0, min=0.1, max=8.0, step=0.1),
                IO.Combo.Input("device_strategy", options=["auto", "load_unload_each_frame", "keep_loaded", "cpu_only"], default="auto"),
            ],
            outputs=[IO.Image.Output(display_name="IMAGE")],
        )

    @classmethod
    def load_upscale_model_with_spandrel(cls, model_path):
        if ModelLoader is None:
            logger.error("%s Spandrel library is not available. Cannot load model.", LOG_PREFIX)
            return None

        try:
            model_descriptor = ModelLoader(device="cpu").load_from_file(model_path)
            upscale_model = model_descriptor.model
            upscale_model.eval()

            if not hasattr(upscale_model, 'scale'):
                if hasattr(model_descriptor, 'scale') and model_descriptor.scale is not None:
                    upscale_model.scale = model_descriptor.scale
                    logger.info("%s Inferred model scale %s from Spandrel descriptor for %s", LOG_PREFIX, upscale_model.scale, model_path)
                else:
                    scale_from_name = 1
                    for part in ["x2", "x3", "x4", "x8"]:
                        if part in model_path.lower():
                            scale_from_name = int(part[1:])
                            break
                    upscale_model.scale = scale_from_name
                    logger.warning(
                        "%s Model scale not directly found for %s; set to %s (inferred from name or default 1).",
                        LOG_PREFIX,
                        model_path,
                        upscale_model.scale,
                    )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return upscale_model
        except Exception as e:
            logger.error("%s Error loading model '%s' with Spandrel: %s", LOG_PREFIX, os.path.basename(model_path), e)
            return None

    @staticmethod
    def _upscale_on_cpu(upscale_model, images, upscale_method, new_width, new_height):
        result_frames = []
        model_scale_factor = getattr(upscale_model, 'scale', 1)
        pbar = comfy.utils.ProgressBar(images.shape[0])

        for i in range(images.shape[0]):
            frame = images[i:i + 1]
            in_img = frame.movedim(-1, -3)

            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=64,
                tile_y=64,
                overlap=8,
                upscale_amount=model_scale_factor,
            )

            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
            samples = upscaled.movedim(-1, 1)
            s_resized = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
            s_resized = s_resized.movedim(1, -1)

            result_frames.append(s_resized[0])
            pbar.update(1)

            del in_img, s, upscaled, samples, s_resized
            gc.collect()

        return result_frames

    @staticmethod
    def _upscale_batch_keep_loaded(upscale_model, images, device, upscale_method, new_width, new_height):
        result_frames = []
        model_scale_factor = getattr(upscale_model, 'scale', 1)
        pbar = comfy.utils.ProgressBar(images.shape[0])

        for i in range(images.shape[0]):
            frame = images[i:i + 1]
            in_img = frame.movedim(-1, -3).to(device)

            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=128,
                tile_y=128,
                overlap=8,
                upscale_amount=model_scale_factor,
            )

            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
            samples = upscaled.movedim(-1, 1)
            s_resized = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
            s_resized = s_resized.movedim(1, -1).cpu()

            result_frames.append(s_resized[0])
            pbar.update(1)

            del in_img, s, upscaled, samples, s_resized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result_frames

    @staticmethod
    def _upscale_batch_load_unload(upscale_model, images, device, upscale_method, new_width, new_height):
        result_frames = []
        model_scale_factor = getattr(upscale_model, 'scale', 1)
        pbar = comfy.utils.ProgressBar(images.shape[0])

        for i in range(images.shape[0]):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_model = upscale_model.to(device)

            frame = images[i:i + 1]
            in_img = frame.movedim(-1, -3).to(device)

            # Bind `current_model` via the default-arg trick so the closure
            # captures *this iteration's* model reference, not the loop
            # variable. tiled_scale is synchronous today, but a future async
            # variant would otherwise see a `current_model` that was already
            # moved back to CPU and del'd at the end of the iteration.
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a, m=current_model: m(a),
                tile_x=96,
                tile_y=96,
                overlap=8,
                upscale_amount=model_scale_factor,
            )

            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
            samples = upscaled.movedim(-1, 1)
            s_resized = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
            s_resized = s_resized.movedim(1, -1).cpu()

            result_frames.append(s_resized[0])
            pbar.update(1)

            del in_img, s, upscaled, samples, s_resized

            current_model = current_model.to("cpu")
            del current_model

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result_frames

    @classmethod
    def execute(cls, model_name, images, upscale_method, factor, device_strategy="auto") -> IO.NodeOutput:
        upscale_model_path = folder_paths.get_full_path("upscale_models", model_name)

        upscale_model = cls.load_upscale_model_with_spandrel(upscale_model_path)
        if upscale_model is None:
            raise RuntimeError(f"Failed to load upscale model '{model_name}' using Spandrel. Ensure Spandrel is installed and model is compatible.")

        device = model_management.get_torch_device()
        if device_strategy == "auto":
            if torch.cuda.is_available():
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    reserved_memory = torch.cuda.memory_reserved(0)

                    if (total_memory - reserved_memory) / total_memory > 0.5:
                        device_strategy = "keep_loaded"
                    else:
                        device_strategy = "load_unload_each_frame"
                except Exception as e:
                    logger.warning("%s GPU memory probe failed, defaulting to load_unload_each_frame: %s", LOG_PREFIX, e)
                    device_strategy = "load_unload_each_frame"
            else:
                device_strategy = "cpu_only"

        num_frames = images.shape[0]
        old_height = images.shape[1]
        old_width = images.shape[2]
        new_height = int(old_height * factor)
        new_width = int(old_width * factor)

        logger.info(
            "%s Processing %d frames from %dx%d to %dx%d with %s strategy using model %s",
            LOG_PREFIX,
            num_frames,
            old_width,
            old_height,
            new_width,
            new_height,
            device_strategy,
            model_name,
        )

        if device_strategy == "cpu_only":
            upscale_model = upscale_model.to("cpu")
            result_frames = cls._upscale_on_cpu(upscale_model, images, upscale_method, new_width, new_height)
        elif device_strategy == "keep_loaded":
            upscale_model = upscale_model.to(device)
            result_frames = cls._upscale_batch_keep_loaded(upscale_model, images, device, upscale_method, new_width, new_height)
        else:
            result_frames = cls._upscale_batch_load_unload(upscale_model, images, device, upscale_method, new_width, new_height)

        return IO.NodeOutput(torch.stack(result_frames))


NODE_CLASS_MAPPINGS = {"TS_Video_Upscale_With_Model": TS_Video_Upscale_With_Model}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Video_Upscale_With_Model": "TS Video Upscale With Model"}
