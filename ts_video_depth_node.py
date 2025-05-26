import os
import torch
import numpy as np
import matplotlib.cm as cm
import cv2
import gc 

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

# --- Импорт модели VideoDepthAnything ---
try:
    from .video_depth_anything.video_depth import VideoDepthAnything
except ImportError:
    try:
        from video_depth_anything.video_depth import VideoDepthAnything
    except ImportError as e:
        print(f"[TS_VideoDepth] CRITICAL IMPORT ERROR: Could not import VideoDepthAnything model.")
        print(f"[TS_VideoDepth] Error details: {e}")
        VideoDepthAnything = None
# --- Конец импорта ---

def ensure_even_vda(value):
    return int(value) if int(value) % 2 == 0 else int(value) + 1

def preprocess_vda_internal(tensor_images, max_res=-1):
    # ... (код без изменений) ...
    if not isinstance(tensor_images, torch.Tensor):
        raise TypeError("Input 'images' must be a PyTorch tensor.")
    if not (tensor_images.ndim == 4 and tensor_images.shape[0] > 0):
        raise ValueError("Input tensor 'images' must have format (N,H,W,C).")

    images_np_float = tensor_images.cpu().numpy()
    frames_uint8_list = []
    for i in range(images_np_float.shape[0]):
        frame_float = images_np_float[i]
        frame_uint8 = (np.clip(frame_float * 255.0, 0, 255)).astype(np.uint8)
        frames_uint8_list.append(frame_uint8)
    
    current_frames_np = np.array(frames_uint8_list)
    del images_np_float, frames_uint8_list
    gc.collect()

    original_height, original_width = current_frames_np.shape[1:3]
    final_frames_np = current_frames_np

    if max_res > 0 and max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        target_height = ensure_even_vda(original_height * scale)
        target_width = ensure_even_vda(original_width * scale)
        
        resized_frames_list = []
        for i in range(current_frames_np.shape[0]):
            resized_frame = cv2.resize(current_frames_np[i], (target_width, target_height), interpolation=cv2.INTER_AREA)
            resized_frames_list.append(resized_frame)
        final_frames_np = np.array(resized_frames_list)
        
        if final_frames_np is not current_frames_np:
            del current_frames_np
        del resized_frames_list
        gc.collect()
        
    return final_frames_np


def postprocess_vda_colormap_internal(depths_np_float32_input, colormap_name="gray", 
                                      dithering_strength=0.0, apply_median_blur=False, 
                                      target_h=None, target_w=None, upscale_algorithm="Lanczos4"): # Добавлен upscale_algorithm
    if depths_np_float32_input is None or depths_np_float32_input.size == 0:
        print("[TS_VideoDepth] Warning: Empty depth map received in postprocess.")
        return torch.empty((0, 0, 0, 3), dtype=torch.float32)

    num_frames, current_h, current_w = depths_np_float32_input.shape
    current_depths_processed = depths_np_float32_input.copy() 

    MEDIAN_KERNEL_SIZE = 5 
    if apply_median_blur:
        print(f"[TS_VideoDepth] Applying Median Blur with kernel size: {MEDIAN_KERNEL_SIZE} on float32 data.")
        processed_depths_list_blur = []
        for i in range(num_frames):
            slice_to_blur = np.ascontiguousarray(current_depths_processed[i], dtype=np.float32)
            try:
                blurred_slice = cv2.medianBlur(slice_to_blur, MEDIAN_KERNEL_SIZE)
                processed_depths_list_blur.append(blurred_slice)
            except cv2.error as e:
                print(f"[TS_VideoDepth] OpenCV Error during medianBlur: {e}. Skipping for this frame.")
                processed_depths_list_blur.append(slice_to_blur)
        if processed_depths_list_blur:
            current_depths_processed = np.array(processed_depths_list_blur, dtype=np.float32)
        del processed_depths_list_blur; gc.collect()
    
    # --- Апскейл до целевого разрешения ---
    interpolation_methods = {
        "Linear": cv2.INTER_LINEAR,
        "Cubic": cv2.INTER_CUBIC,
        "Lanczos4": cv2.INTER_LANCZOS4
    }
    chosen_interpolation = interpolation_methods.get(upscale_algorithm, cv2.INTER_LANCZOS4) # По умолчанию Lanczos4

    if target_h is not None and target_w is not None and (current_h != target_h or current_w != target_w):
        print(f"[TS_VideoDepth] Upscaling depth map from ({current_h},{current_w}) to ({target_h},{target_w}) using {upscale_algorithm}.")
        upscaled_depths_list = []
        for i in range(num_frames):
            upscaled_slice = cv2.resize(current_depths_processed[i], (target_w, target_h), interpolation=chosen_interpolation)
            upscaled_depths_list.append(upscaled_slice)
        current_depths_processed = np.array(upscaled_depths_list, dtype=np.float32)
        del upscaled_depths_list; gc.collect()
        h, w = target_h, target_w
    else:
        h, w = current_h, current_w

    d_min, d_max = current_depths_processed.min(), current_depths_processed.max() 
    if d_min == d_max:
        normalized_depths = np.full_like(current_depths_processed, 0.0, dtype=np.float32)
    else:
        normalized_depths = ((current_depths_processed - d_min) / (d_max - d_min)).astype(np.float32)
    del current_depths_processed; gc.collect()

    if dithering_strength > 0.0:
        noise = (np.random.rand(*normalized_depths.shape).astype(np.float32) - 0.5) * dithering_strength
        dithered_depths = normalized_depths + noise
        normalized_depths = np.clip(dithered_depths, 0.0, 1.0)
        del noise, dithered_depths; gc.collect()

    output_array_rgb = np.zeros((num_frames, h, w, 3), dtype=np.float32)
    if colormap_name == "gray":
        output_array_rgb[..., 0] = normalized_depths
        output_array_rgb[..., 1] = normalized_depths
        output_array_rgb[..., 2] = normalized_depths
    else:
        try:
            colormap_fn = cm.get_cmap(colormap_name)
        except ValueError:
            print(f"[TS_VideoDepth] Warning: Colormap '{colormap_name}' not found. Defaulting to 'gray'.")
            output_array_rgb[..., 0] = normalized_depths; output_array_rgb[..., 1] = normalized_depths; output_array_rgb[..., 2] = normalized_depths
        else:
            for i in range(num_frames):
                colored_slice_rgba = colormap_fn(normalized_depths[i])
                output_array_rgb[i] = colored_slice_rgba[..., :3].astype(np.float32)
    
    del normalized_depths; gc.collect()
    return torch.from_numpy(output_array_rgb)


class TS_VideoDepth:
    _loaded_model_instance = None
    _loaded_model_filename = None
    _model_on_device_type_str = None 
    _first_gpu_call_done = False 

    def __init__(self):
        if VideoDepthAnything is None:
            raise ImportError("[TS_VideoDepth] Node cannot be initialized: VideoDepthAnything model class failed to import.")

    @classmethod
    def INPUT_TYPES(cls):
        upscale_methods_list = ["Lanczos4", "Cubic", "Linear"] # Порядок для отображения в UI
        return {
            "required": {
                "images": ("IMAGE", ),
                "model_filename": (['video_depth_anything_vits.pth', 'video_depth_anything_vitl.pth'], 
                                   {"default": 'video_depth_anything_vitl.pth'}),
                "input_size": ("INT", {"default": 518, "min": 64, "max": 4096, "step": 2}),
                "max_res": ("INT", {"default": 1280, "min": -1, "max": 8192, "step": 1}), 
                "precision": (['fp16', 'fp32'], {"default": 'fp16'}),
                "colormap": (['gray', 'inferno', 'viridis', 'plasma', 'magma', 'cividis'], {"default": 'gray'}),
                "dithering_strength": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 0.016, "step": 0.0001, "round": 0.0001}),
                "apply_median_blur": ("BOOLEAN", {"default": True}), # ИЗМЕНЕНО: включено по умолчанию
                "upscale_algorithm": (upscale_methods_list, {"default": "Lanczos4"}), # Новый параметр
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute_process_unified" 
    CATEGORY = "Tools/Video"

    @classmethod
    def _ensure_model_loaded_on_cpu(cls, model_filename_to_load):
        # ... (код без изменений) ...
        if cls._loaded_model_instance is None or cls._loaded_model_filename != model_filename_to_load or cls._model_on_device_type_str != 'cpu':
            if cls._loaded_model_instance is not None and cls._model_on_device_type_str != 'cpu':
                print(f"[TS_VideoDepth] Offloading previous model: {cls._loaded_model_filename} from {cls._model_on_device_type_str}")
                try:
                    cls._loaded_model_instance.to('cpu')
                    if cls._model_on_device_type_str == 'cuda' and torch.cuda.is_available():
                        torch.cuda.synchronize() 
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[TS_VideoDepth] Minor error during offload of previous model: {e}")
            
            if cls._loaded_model_instance is None or cls._loaded_model_filename != model_filename_to_load:
                del cls._loaded_model_instance 
                cls._loaded_model_instance = None
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                print(f"[TS_VideoDepth] Loading model: {model_filename_to_load} to CPU")
                download_path = os.path.join(folder_paths.models_dir, "videodepthanything")
                model_path = os.path.join(download_path, model_filename_to_load)
                if not os.path.exists(model_path):
                    os.makedirs(download_path, exist_ok=True)
                    from huggingface_hub import snapshot_download
                    repo_map = { "vits": "depth-anything/Video-Depth-Anything-Small", "vitl": "depth-anything/Video-Depth-Anything-Large" }
                    model_key = next((key for key in repo_map if key in model_filename_to_load.lower()), None)
                    if not model_key: raise ValueError(f"Cannot determine repository for model: {model_filename_to_load}.")
                    snapshot_download(repo_id=repo_map[model_key], allow_patterns=[f"*{model_filename_to_load}*"], local_dir=download_path, local_dir_use_symlinks=False)
                model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                }
                encoder_key = next((key for key in model_configs if key in model_filename_to_load.lower()), None)
                if not encoder_key: raise ValueError(f"Cannot determine model config for: {model_filename_to_load}")
                vda_instance = VideoDepthAnything(**model_configs[encoder_key])
                state_dict = load_torch_file(model_path, device='cpu')
                vda_instance.load_state_dict(state_dict); del state_dict; gc.collect()
                cls._loaded_model_instance = vda_instance.eval()
                cls._loaded_model_filename = model_filename_to_load
            
            if hasattr(cls._loaded_model_instance, 'device') and cls._loaded_model_instance.device.type != 'cpu':
                 cls._loaded_model_instance.to('cpu')

            cls._model_on_device_type_str = 'cpu'
            print(f"[TS_VideoDepth] Model {model_filename_to_load} is confirmed on CPU.")
        return cls._loaded_model_instance

    # Добавлен upscale_algorithm в аргументы
    def execute_process_unified(self, images, model_filename, input_size, max_res, precision, colormap, dithering_strength, apply_median_blur, upscale_algorithm): 
        
        original_h, original_w = images.shape[1], images.shape[2]
        model_to_use = TS_VideoDepth._ensure_model_loaded_on_cpu(model_filename)
        current_processing_device = mm.get_torch_device()

        if current_processing_device.type == 'cuda':
            if not TS_VideoDepth._first_gpu_call_done:
                print("[TS_VideoDepth] First GPU call for this model in session: Performing VRAM clear.")
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize() 
            model_to_use.to(current_processing_device)
            TS_VideoDepth._model_on_device_type_str = current_processing_device.type
        else: 
            TS_VideoDepth._model_on_device_type_str = 'cpu'

        if input_size % 14 != 0:
            processed_input_size = ((input_size // 14) * 14)
            if processed_input_size == 0 and input_size > 0 : processed_input_size = 14
            if processed_input_size != input_size:
                 print(f"[TS_VideoDepth] Adjusted input_size from {input_size} to {processed_input_size} (to be multiple of 14).")
        else:
            processed_input_size = input_size

        pbar = ProgressBar(images.shape[0]) 
        images_np_uint8 = preprocess_vda_internal(images, max_res)
        
        depths_np_raw = None
        try:
            if current_processing_device.type == 'cuda': torch.cuda.synchronize()
            with torch.no_grad():
                print(f"[TS_VideoDepth] Starting inference with input_size={processed_input_size}, precision={precision} on {current_processing_device}...")
                depths_np_raw = model_to_use.infer_video_depth(
                    images_np_uint8, input_size=processed_input_size, device=current_processing_device,
                    pbar=pbar, fp32=(precision == 'fp32')
                )
            if current_processing_device.type == 'cuda':
                torch.cuda.synchronize(); TS_VideoDepth._first_gpu_call_done = True 
            print(f"[TS_VideoDepth] Inference complete.")
        except Exception as e:
            print(f"[TS_VideoDepth] EXCEPTION DURING MODEL INFERENCE: {type(e).__name__} - {e}")
            if TS_VideoDepth._model_on_device_type_str != 'cpu':
                print(f"[TS_VideoDepth] Offloading model to CPU due to exception...")
                model_to_use.to('cpu'); TS_VideoDepth._model_on_device_type_str = 'cpu'
                if current_processing_device.type == 'cuda': torch.cuda.synchronize(); torch.cuda.empty_cache()
            gc.collect(); raise e 
        finally:
            if TS_VideoDepth._model_on_device_type_str != 'cpu':
                model_to_use.to('cpu'); TS_VideoDepth._model_on_device_type_str = 'cpu'
            gc.collect()

        del images_np_uint8

        if depths_np_raw is None:
            print("[TS_VideoDepth] Error: Model inference returned None but no exception was raised.")
            return (torch.zeros((images.shape[0], original_h, original_w, 3), dtype=torch.float32, device=images.device),)

        if depths_np_raw.dtype != np.float32:
            depths_np_float32_for_postproc = depths_np_raw.astype(np.float32); del depths_np_raw 
        else:
            depths_np_float32_for_postproc = depths_np_raw
        
        output_tensor = postprocess_vda_colormap_internal(
            depths_np_float32_for_postproc, colormap, dithering_strength, 
            apply_median_blur, original_h, original_w, upscale_algorithm # Передаем новый параметр
        )
        
        del depths_np_float32_for_postproc; gc.collect()
        print(f"[TS_VideoDepth] Processing finished successfully.")
        return (output_tensor,)

# --- Регистрация ноды для ComfyUI ---
NODE_CLASS_MAPPINGS = { "TS_VideoDepthNode": TS_VideoDepth }
NODE_DISPLAY_NAME_MAPPINGS = { "TS_VideoDepthNode": "TS Video Depth" }

if VideoDepthAnything is not None:
    print("--------------------------------------------------")
    print("Custom Node: TS Video Depth - Loaded Successfully")
    print("--------------------------------------------------")
else:
    print("--------------------------------------------------")
    print("Custom Node: TS Video Depth - LOADED WITH ERRORS (VideoDepthAnything model class not found)")
    print("--------------------------------------------------")