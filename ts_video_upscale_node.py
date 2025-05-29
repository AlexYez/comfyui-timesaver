import os
import torch
import comfy.model_management as model_management
import comfy.utils
import folder_paths
import numpy as np
from PIL import Image
import gc
import time
# Попытка импортировать spandrel
try:
    from spandrel import ModelLoader
except ImportError:
    print("Spandrel library not found. Please make sure it is installed.")
    ModelLoader = None

class TS_Video_Upscale_With_Model:
    """
    A memory-efficient implementation for upscaling video frames using an upscale model
    with proper batch processing and memory management.
    """
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                    "images": ("IMAGE",),
                    "upscale_method": (s.upscale_methods,),
                    "factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                    "device_strategy": (["auto", "load_unload_each_frame", "keep_loaded", "cpu_only"], {"default": "auto"})
                }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_video"
    CATEGORY = "video"
    
    # ComfyUI progress bar integration
    def __init__(self):
        self.steps = 0
        self.step = 0
    
    # Important: ComfyUI looks for this method to track progress
    def get_progress_execution(self):
        if self.steps > 0:
            return self.step, self.steps
        return 0, 1 # Default to 0 out of 1 if steps not initialized
    
    def upscale_video(self, model_name, images, upscale_method, factor, device_strategy="auto"):
        """
        Upscale a sequence of images (video frames) efficiently using an integrated upscale model.
        """
        upscale_model_path = folder_paths.get_full_path("upscale_models", model_name)
        
        upscale_model = self.load_upscale_model_with_spandrel(upscale_model_path)
        if upscale_model is None:
            raise RuntimeError(f"Failed to load upscale model '{model_name}' using Spandrel. Ensure Spandrel is installed and model is compatible.")

        device = model_management.get_torch_device()
        if device_strategy == "auto":
            if torch.cuda.is_available():
                try:
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    reserved_memory = torch.cuda.memory_reserved(0)
                    
                    if (total_memory - reserved_memory) / total_memory > 0.5: # Heuristic: if more than 50% VRAM is free
                        device_strategy = "keep_loaded"
                    else:
                        device_strategy = "load_unload_each_frame"
                except Exception as e:
                    print(f"Could not assess GPU memory for auto strategy, defaulting to load_unload_each_frame. Error: {e}")
                    device_strategy = "load_unload_each_frame" # Fallback if props/reserved fails
            else:
                device_strategy = "cpu_only"
        
        num_frames = images.shape[0]
        old_height = images.shape[1]
        old_width = images.shape[2]
        new_height = int(old_height * factor)
        new_width = int(old_width * factor)
        
        # Initialize progress tracking for ComfyUI progress bar
        self.steps = num_frames
        self.step = 0
        
        print(f"Processing video: {num_frames} frames from {old_width}x{old_height} to {new_width}x{new_height} with {device_strategy} strategy using model {model_name}")
        
        if device_strategy == "cpu_only":
            upscale_model = upscale_model.to("cpu")
            result_frames = self._upscale_on_cpu(upscale_model, images, upscale_method, new_width, new_height)
        elif device_strategy == "keep_loaded":
            upscale_model = upscale_model.to(device)
            result_frames = self._upscale_batch_keep_loaded(upscale_model, images, device, upscale_method, new_width, new_height)
        else:  # "load_unload_each_frame"
            result_frames = self._upscale_batch_load_unload(upscale_model, images, device, upscale_method, new_width, new_height)
        
        return (torch.stack(result_frames),)
    
    def load_upscale_model_with_spandrel(self, model_path):
        """Load the upscale model from the given path using Spandrel."""
        if ModelLoader is None:
            print("Spandrel library is not available. Cannot load model.")
            return None
        
        try:
            model_descriptor = ModelLoader(device="cpu").load_from_file(model_path)
            upscale_model = model_descriptor.model
            upscale_model.eval()

            if not hasattr(upscale_model, 'scale'):
                if hasattr(model_descriptor, 'scale') and model_descriptor.scale is not None:
                     upscale_model.scale = model_descriptor.scale
                     print(f"Info: Inferred model scale {upscale_model.scale} from Spandrel descriptor for {model_path}")
                else:
                    # Attempt to infer scale from common model naming conventions if factor is e.g. x2, x4
                    scale_from_name = 1 
                    for part in ["x2", "x3", "x4", "x8"]:
                        if part in model_path.lower():
                            scale_from_name = int(part[1:])
                            break
                    upscale_model.scale = scale_from_name
                    print(f"Warning: Model scale not directly found for {model_path}. Set to {upscale_model.scale} (inferred from name or default 1). This is used by tiled_scale.")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return upscale_model
        except Exception as e:
            print(f"Error loading model '{os.path.basename(model_path)}' with Spandrel: {e}")
            return None

    def _upscale_on_cpu(self, upscale_model, images, upscale_method, new_width, new_height):
        result_frames = []
        start_time = time.time()
        model_scale_factor = getattr(upscale_model, 'scale', 1)

        for i in range(images.shape[0]):
            frame = images[i:i+1]
            in_img = frame.movedim(-1, -3)
            
            # Check if model itself handles the target factor directly
            # This is a simplified check; real models might have complex scaling.
            # The `factor` input to the node is the overall desired upscale factor.
            # `model_scale_factor` is what the model intrinsically does.
            # `comfy.utils.tiled_scale` uses `upscale_amount` for its internal logic,
            # and then `comfy.utils.common_upscale` does the final resize to new_width/new_height.
            
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=64,
                tile_y=64,
                overlap=8, 
                upscale_amount=model_scale_factor 
            )
            
            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
            samples = upscaled.movedim(-1, 1)
            s_resized = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
            s_resized = s_resized.movedim(1, -1)
            
            result_frames.append(s_resized[0])
            self.step += 1 # Update progress for ComfyUI bar
            
            elapsed = time.time() - start_time
            eta = elapsed / self.step * (self.steps - self.step) if self.step > 0 else 0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            percent = (self.step / self.steps) * 100
            # Removed ANSI color codes for standard console output
            print(f"\r|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}", end="", flush=True)
            
            del in_img, s, upscaled, samples, s_resized
            gc.collect()
        
        print() # Final newline after progress bar
        return result_frames
    
    def _upscale_batch_keep_loaded(self, upscale_model, images, device, upscale_method, new_width, new_height):
        result_frames = []
        start_time = time.time()
        model_scale_factor = getattr(upscale_model, 'scale', 1)

        for i in range(images.shape[0]):
            frame = images[i:i+1]
            in_img = frame.movedim(-1, -3).to(device)
            
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=128, 
                tile_y=128,
                overlap=8, 
                upscale_amount=model_scale_factor
            )
            
            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
            samples = upscaled.movedim(-1, 1)
            s_resized = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
            s_resized = s_resized.movedim(1, -1).cpu()
            
            result_frames.append(s_resized[0])
            self.step += 1 # Update progress for ComfyUI bar
            
            elapsed = time.time() - start_time
            eta = elapsed / self.step * (self.steps - self.step) if self.step > 0 else 0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            percent = (self.step / self.steps) * 100
            # Removed ANSI color codes
            print(f"\r|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}", end="", flush=True)
            
            del in_img, s, upscaled, samples, s_resized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print() # Final newline
        return result_frames

    def _upscale_batch_load_unload(self, upscale_model, images, device, upscale_method, new_width, new_height):
        result_frames = []
        start_time = time.time()
        model_scale_factor = getattr(upscale_model, 'scale', 1)

        for i in range(images.shape[0]):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            current_model = upscale_model.to(device) # Ensure model is on device for this frame
            
            frame = images[i:i+1]
            in_img = frame.movedim(-1, -3).to(device)
            
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: current_model(a), # Use current_model which is on device
                tile_x=96,
                tile_y=96,
                overlap=8, 
                upscale_amount=model_scale_factor
            )
            
            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
            samples = upscaled.movedim(-1, 1)
            s_resized = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
            s_resized = s_resized.movedim(1, -1).cpu()
            
            result_frames.append(s_resized[0])
            self.step += 1 # Update progress for ComfyUI bar
            
            elapsed = time.time() - start_time
            eta = elapsed / self.step * (self.steps - self.step) if self.step > 0 else 0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            percent = (self.step / self.steps) * 100
            # Removed ANSI color codes
            print(f"\r|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}", end="", flush=True)
            
            del in_img, s, upscaled, samples, s_resized
            
            current_model = current_model.to("cpu") # Move model back to CPU
            del current_model # Ensure reference is cleared before gc
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print() # Final newline
        return result_frames


# Optional companion node to manage memory explicitly during video processing
# (Код TS_Free_Video_Memory остается без изменений)
class TS_Free_Video_Memory:
    """
    A node that explicitly cleans up memory during video processing pipelines
    to avoid memory bottlenecks.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "aggressive_cleanup": (["disable", "enable"], {"default": "disable"}),
            "report_memory": (["disable", "enable"], {"default": "enable"})
        }}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "cleanup_memory"
    CATEGORY = "video"
    
    def cleanup_memory(self, images, aggressive_cleanup="disable", report_memory="enable"):
        if report_memory == "enable" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"Before cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive_cleanup == "enable":
                torch.cuda.synchronize()
                if hasattr(torch.cuda, 'caching_allocator_delete_caches'): # For newer PyTorch
                    try:
                        torch.cuda.caching_allocator_delete_caches()
                    except Exception as e:
                        print(f"Note: torch.cuda.caching_allocator_delete_caches() failed (might be older PyTorch): {e}")
        
        if report_memory == "enable" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"After cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        return (images,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "TS_Video_Upscale_With_Model": TS_Video_Upscale_With_Model,
    "TS_Free_Video_Memory": TS_Free_Video_Memory,
}

# Display name and category mappings for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Video_Upscale_With_Model": "TS Video Upscale With Model",
    "TS_Free_Video_Memory": "TS Free Video Memory",
}