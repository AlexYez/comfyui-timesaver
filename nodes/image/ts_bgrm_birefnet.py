# Model License Notice:
# - BiRefNet Models: Apache-2.0 License (https://huggingface.co/ZhengPeng7)
# Code based on : https://github.com/AILab-AI/ComfyUI-RMBG

import importlib.util
import logging
import os
import sys
import threading
import types
from contextlib import contextmanager

import cv2
import comfy.model_management as model_management
import folder_paths
import numpy as np
import torch
import torch.nn.functional as F
from comfy.utils import ProgressBar
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter
from safetensors.torch import load_file

logger = logging.getLogger(__name__)
_LOG_PREFIX = "[TS Remove Background]"

# Add model path to ComfyUI/models/BiRefNet/
folder_paths.add_model_folder_path("birefnet", os.path.join(folder_paths.models_dir, "BiRefNet"))

# Model configuration
MODEL_CONFIG = {
    "BiRefNet-general": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-general.safetensors": "BiRefNet-general.safetensors",
            "config.json": "config.json"
        },
        "description": "General purpose model with balanced performance",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet_512x512": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_512x512.safetensors": "BiRefNet_512x512.safetensors",
            "config.json": "config.json"
        },
        "description": "Optimized for 512x512 resolution, faster processing",
        "default_res": 512,
        "max_res": 1024,
        "min_res": 256,
        "force_res": True
    },
    "BiRefNet-HR": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-HR.safetensors": "BiRefNet-HR.safetensors",
            "config.json": "config.json"
        },
        "description": "High resolution general purpose model",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024
    },
    "BiRefNet-portrait": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-portrait.safetensors": "BiRefNet-portrait.safetensors",
            "config.json": "config.json"
        },
        "description": "Optimized for portrait/human matting",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet-matting": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-matting.safetensors": "BiRefNet-matting.safetensors",
            "config.json": "config.json"
        },
        "description": "General purpose matting model",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet-HR-matting": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet-HR-matting.safetensors": "BiRefNet-HR-matting.safetensors",
            "config.json": "config.json"
        },
        "description": "High resolution matting model",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024
    },
    "BiRefNet_lite": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet_lite.py": "birefnet_lite.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_lite.safetensors": "BiRefNet_lite.safetensors",
            "config.json": "config.json"
        },
        "description": "Lightweight version for faster processing",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet_lite-2K": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet_lite.py": "birefnet_lite.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_lite-2K.safetensors": "BiRefNet_lite-2K.safetensors",
            "config.json": "config.json"
        },
        "description": "Lightweight version optimized for 2K resolution",
        "default_res": 2048,
        "max_res": 2560,
        "min_res": 1024
    },
    "BiRefNet_dynamic": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_dynamic.safetensors": "BiRefNet_dynamic.safetensors",
            "config.json": "config.json"
        },
        "description": "Dynamic model for high-resolution dichotomous image segmentation",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    },
    "BiRefNet_lite-matting": {
        "repo_id": "1038lab/BiRefNet",
        "files": {
            "birefnet_lite.py": "birefnet_lite.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
            "BiRefNet_lite-matting.safetensors": "BiRefNet_lite-matting.safetensors",
            "config.json": "config.json"
        },
        "description": "Lightweight matting model for general purpose",
        "default_res": 1024,
        "max_res": 2048,
        "min_res": 512
    }
}

# Utility functions
def _get_target_device():
    try:
        target_device = model_management.get_torch_device()
    except Exception as exc:
        logger.warning("%s Could not resolve ComfyUI device, using CPU: %s", _LOG_PREFIX, exc)
        target_device = torch.device("cpu")

    if torch.cuda.is_available() and getattr(target_device, "type", str(target_device)) == "cpu":
        logger.info("%s CUDA is available; using GPU for BiRefNet inference", _LOG_PREFIX)
        return torch.device("cuda")

    return target_device


def _target_dtype(target_device):
    device_type = getattr(target_device, "type", str(target_device))
    return torch.float16 if device_type == "cuda" else torch.float32


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
    if progress_bar is None or cap_step <= start_step:
        yield
        return

    stop_event = threading.Event()
    current_value = int(start_step)
    cap_value = int(cap_step)

    def pulse():
        nonlocal current_value
        while not stop_event.wait(interval):
            if current_value >= cap_value:
                continue
            current_value += 1
            _update_progress(progress_bar, current_value, total_steps)

    _update_progress(progress_bar, current_value, total_steps)
    worker = threading.Thread(target=pulse, name="ts-bgrm-progress", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop_event.set()
        worker.join(timeout=1.0)


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


def handle_model_error(message):
    logger.error("%s %s", _LOG_PREFIX, message)
    raise RuntimeError(message)

def refine_foreground(image_bchw, masks_b1hw):
    b, c, h, w = image_bchw.shape
    if b != masks_b1hw.shape[0]:
        raise ValueError("images and masks must have the same batch size")
    
    image_np = image_bchw.cpu().numpy()
    mask_np = masks_b1hw.cpu().numpy()
    
    refined_fg = []
    for i in range(b):
        mask = mask_np[i, 0]      
        thresh = 0.45
        mask_binary = (mask > thresh).astype(np.float32)
        
        edge_blur = cv2.GaussianBlur(mask_binary, (3, 3), 0)
        transition_mask = np.logical_and(mask > 0.05, mask < 0.95)
        
        alpha = 0.85
        mask_refined = np.where(transition_mask,
                              alpha * mask + (1-alpha) * edge_blur,
                              mask_binary)
        
        edge_region = np.logical_and(mask > 0.2, mask < 0.8)
        mask_refined = np.where(edge_region,
                              mask_refined * 0.98,
                              mask_refined)
        
        result = []
        for c in range(image_np.shape[1]):
            channel = image_np[i, c]
            refined = channel * mask_refined
            result.append(refined)
            
        refined_fg.append(np.stack(result))
    
    return torch.from_numpy(np.stack(refined_fg))

class BiRefNetModel:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.current_device = None
        self.current_dtype = None
        self.base_cache_dir = os.path.join(folder_paths.models_dir, "BiRefNet")
    
    def get_cache_dir(self, model_name):
        return self.base_cache_dir
    
    def check_model_cache(self, model_name):
        cache_dir = self.get_cache_dir(model_name)
        
        if not os.path.exists(cache_dir):
            return False, "Model directory not found"
        
        missing_files = []
        for filename in MODEL_CONFIG[model_name]["files"].keys():
            if not os.path.exists(os.path.join(cache_dir, filename)):
                missing_files.append(filename)
        
        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"
            
        return True, "Model cache verified"
    
    def download_model(self, model_name, progress_bar=None, start_step=5, end_step=30):
        cache_dir = self.get_cache_dir(model_name)

        try:
            os.makedirs(cache_dir, exist_ok=True)
            filenames = list(MODEL_CONFIG[model_name]["files"].keys())
            logger.info("%s Downloading %s model files", _LOG_PREFIX, model_name)

            current_step = start_step
            for index, filename in enumerate(filenames, start=1):
                next_step = start_step + int((end_step - start_step) * index / max(1, len(filenames)))
                logger.info("%s Downloading %s", _LOG_PREFIX, filename)
                with _progress_pulse(progress_bar, current_step, max(current_step, next_step - 1)):
                    hf_hub_download(
                        repo_id=MODEL_CONFIG[model_name]["repo_id"],
                        filename=filename,
                        local_dir=cache_dir,
                        local_dir_use_symlinks=False
                    )
                current_step = next_step
                _update_progress(progress_bar, current_step)

            return True, "Model files downloaded successfully"

        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"
    
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

    def load_model(self, model_name, progress_bar=None, start_step=30, end_step=55, target_device=None):
        if target_device is None:
            target_device = _get_target_device()
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
            model_filename = [k for k in MODEL_CONFIG[model_name]["files"].keys() if k.endswith('.py') and k != "BiRefNet_config.py"][0]
            model_path = os.path.join(cache_dir, model_filename)
            config_path = os.path.join(cache_dir, "BiRefNet_config.py")
            weights_filename = [k for k in MODEL_CONFIG[model_name]["files"].keys() if k.endswith('.safetensors')][0]
            weights_path = os.path.join(cache_dir, weights_filename)

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
                    state_dict = load_file(weights_path)
                self.model.load_state_dict(state_dict)

                self.model.eval()
                if target_dtype == torch.float16:
                    self.model.half()
                    torch.set_float32_matmul_precision('high')
                else:
                    self.model.float()

                with _progress_pulse(progress_bar, end_step - 8, end_step - 1):
                    self.model.to(target_device)

                self.current_model_version = model_name
                self.current_device = target_device_key
                self.current_dtype = target_dtype
                _update_progress(progress_bar, end_step)

            except Exception as e:
                handle_model_error(f"Error loading BiRefNet model: {str(e)}")

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

        with torch.inference_mode():
            preds = self.model(input_tensor)
            pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            elif pred.ndim == 4 and pred.shape[1] != 1 and pred.shape[-1] == 1:
                pred = pred.movedim(-1, 1)

            pred = pred.sigmoid().float()
            pred = F.interpolate(pred, size=(height, width), mode="bicubic", align_corners=False)

        return pred[:, 0].clamp(0.0, 1.0).detach().cpu()

    def process_masks(self, image, params, progress_bar=None, start_step=55, end_step=80, target_device=None):
        try:
            if self.model is None:
                raise RuntimeError("BiRefNet model is not loaded")

            if image.ndim != 4:
                raise ValueError(f"Expected IMAGE tensor [B, H, W, C], got {tuple(image.shape)}")

            batch_size = image.shape[0]
            process_res = params["process_res"]
            if target_device is None:
                target_device = _get_target_device()
            target_dtype = _target_dtype(target_device)
            chunk_size = _estimate_inference_chunk_size(batch_size, process_res, target_device)

            masks = []
            processed = 0
            while processed < batch_size:
                current_chunk_size = min(chunk_size, batch_size - processed)
                chunk = image[processed:processed + current_chunk_size]
                current_step = start_step + int((end_step - start_step) * processed / max(1, batch_size))
                chunk_cap = start_step + int((end_step - start_step) * (processed + current_chunk_size) / max(1, batch_size))

                try:
                    with _progress_pulse(progress_bar, current_step, max(current_step, chunk_cap - 1)):
                        masks.append(self._process_mask_chunk(chunk, process_res, target_device, target_dtype))
                    processed += current_chunk_size
                    _update_progress(progress_bar, chunk_cap)
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
            handle_model_error(f"Error in BiRefNet processing: {str(e)}")

class TS_BGRM_BiRefNet:
    def __init__(self):
        self.model = BiRefNetModel()
    
    @classmethod
    def INPUT_TYPES(s):
        tooltips = {
            "enable": "Enable or disable the background removal process. If disabled, the original image will be passed through.",
            "image": "Input image to be processed for background removal.",
            "model": "Select the BiRefNet model variant to use.",
            "use_custom_resolution": "Enable to use a custom resolution specified below. If disabled, the model's default resolution will be used.",
            "process_resolution": "The resolution for processing the image. It will be adjusted to the nearest multiple of 64.",
            "mask_blur": "Specify the amount of blur to apply to the mask edges (0 for no blur, higher values for more blur).",
            "mask_offset": "Adjust the mask boundary (positive values expand the mask, negative values shrink it).",
            "invert_output": "Enable to invert both the image and mask output (useful for certain effects).",
            "refine_foreground": "Use Fast Foreground Colour Estimation to optimize transparent background",
            "background": "Choose background type: Alpha (transparent) or Color (custom background color).",
            "background_color": "Select background color preset (black, white, green)."
        }
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "enable": ("BOOLEAN", {"default": True, "tooltip": tooltips["enable"]}),
                "model": (list(MODEL_CONFIG.keys()), {"tooltip": tooltips["model"]}),
            },
            "optional": {
                "use_custom_resolution": ("BOOLEAN", {"default": False, "tooltip": tooltips["use_custom_resolution"]}),
                "process_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64, "tooltip": tooltips["process_resolution"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
                "refine_foreground": ("BOOLEAN", {"default": False, "tooltip": tooltips["refine_foreground"]}),
                "background": (["Alpha", "Color"], {"default": "Alpha", "tooltip": tooltips["background"]}),
                "background_color": (["black", "white", "green"], {"default": "white", "tooltip": tooltips["background_color"]}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "process_image"
    CATEGORY = "TS/Image"

    def process_image(self, image, enable, model, use_custom_resolution, process_resolution, **params):
        params = {
            "mask_blur": 0,
            "mask_offset": 0,
            "invert_output": False,
            "refine_foreground": False,
            "background": "Alpha",
            "background_color": "white",
            **params,
        }

        if not enable:
            b, h, w, c = image.shape
            mask_output = torch.ones((b, h, w), dtype=torch.float32, device=image.device)
            mask_image_output = torch.ones((b, h, w, 3), dtype=torch.float32, device=image.device)
            return (image, mask_output, mask_image_output)

        try:
            pbar = ProgressBar(100)
            _update_progress(pbar, 1)
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
            _update_progress(pbar, 5)

            processed_images = []
            processed_masks = []
            cache_status, message = self.model.check_model_cache(model)
            if not cache_status:
                logger.info("%s Cache check: %s", _LOG_PREFIX, message)
                download_status, download_message = self.model.download_model(model, progress_bar=pbar, start_step=5, end_step=30)
                if not download_status:
                    handle_model_error(download_message)
                logger.info("%s Model files downloaded successfully", _LOG_PREFIX)
            else:
                _update_progress(pbar, 30)

            target_device = _get_target_device()
            logger.info("%s Processing device: %s", _LOG_PREFIX, _format_device_label(target_device))

            self.model.load_model(model, progress_bar=pbar, start_step=30, end_step=55, target_device=target_device)
            _update_progress(pbar, 55)

            masks = self.model.process_masks(image, params, progress_bar=pbar, start_step=55, end_step=80, target_device=target_device)

            batch_size = image.shape[0]
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
                img_tensor = img.detach().cpu().movedim(-1, 0).unsqueeze(0).float()
                mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).unsqueeze(0) / 255.0
                orig_image = tensor2pil(img)
                if params.get("refine_foreground", False):
                    refined_fg = refine_foreground(img_tensor, mask_tensor)
                    refined_fg = tensor2pil(refined_fg[0].permute(1, 2, 0))
                    r, g, b = refined_fg.split()
                    foreground = Image.merge('RGBA', (r, g, b, mask))
                else:
                    orig_rgba = orig_image.convert("RGBA")
                    r, g, b, _ = orig_rgba.split()
                    foreground = Image.merge('RGBA', (r, g, b, mask))
                if params["background"] == "Alpha":
                    processed_images.append(pil2tensor(foreground))
                else:
                    background_color = params.get("background_color", "white")
                    color_presets = {
                        "black": "#000000",
                        "white": "#ffffff",
                        "green": "#00ff00",
                    }
                    if isinstance(background_color, str):
                        color_key = background_color.strip().lower()
                        if color_key in color_presets:
                            background_color = color_presets[color_key]
                    rgba = hex_to_rgba(background_color)
                    bg_image = Image.new('RGBA', orig_image.size, rgba)
                    composite_image = Image.alpha_composite(bg_image, foreground)
                    processed_images.append(pil2tensor(composite_image.convert("RGB")))
                processed_masks.append(pil2tensor(mask))
                post_step = 80 + int(19 * (index + 1) / max(1, batch_size))
                _update_progress(pbar, post_step)

            image_output = torch.cat(processed_images, dim=0)
            mask_output = torch.cat(processed_masks, dim=0)
            mask_image_output = mask_output.unsqueeze(-1).expand(-1, -1, -1, 3)
            _update_progress(pbar, 100)
            return (image_output, mask_output, mask_image_output)
        except Exception as e:
            handle_model_error(f"Error in image processing: {str(e)}")

# Node Mapping
NODE_CLASS_MAPPINGS = {
    "TS_BGRM_BiRefNet": TS_BGRM_BiRefNet
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_BGRM_BiRefNet": "TS Remove Background"
}

