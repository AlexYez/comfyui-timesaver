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
    
    def __init__(self):
        self.steps = 0
        self.step = 0
    
    def get_progress_execution(self):
        if self.steps > 0:
            return self.step, self.steps
        return 0, 1
    
    def upscale_video(self, model_name, images, upscale_method, factor, device_strategy="auto"):
        upscale_model_path = folder_paths.get_full_path("upscale_models", model_name)
        
        # Используем новую функцию загрузки
        upscale_model = self.load_upscale_model_with_spandrel(upscale_model_path)
        if upscale_model is None:
            # Обработка ошибки, если модель не загрузилась (например, spandrel не установлен или модель несовместима)
            # Можно вернуть ошибку или попытаться использовать старый метод как запасной
            print("Failed to load model with Spandrel. Falling back to old method (if available) or erroring.")
            # Для примера, вызовем исключение, если spandrel обязателен
            raise RuntimeError(f"Failed to load upscale model '{model_name}' using Spandrel.")

        device = model_management.get_torch_device()
        if device_strategy == "auto":
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = torch.cuda.memory_reserved(0)
                
                if (total_memory - reserved_memory) / total_memory > 0.5:
                    device_strategy = "keep_loaded"
                else:
                    device_strategy = "load_unload_each_frame"
            else:
                device_strategy = "cpu_only"
        
        num_frames = images.shape[0]
        old_height = images.shape[1]
        old_width = images.shape[2]
        new_height = int(old_height * factor)
        new_width = int(old_width * factor)
        
        self.steps = num_frames
        self.step = 0
        
        print(f"Processing video: {num_frames} frames from {old_width}x{old_height} to {new_width}x{new_height} with {device_strategy} strategy")
        
        if device_strategy == "cpu_only":
            upscale_model = upscale_model.to("cpu")
            result_frames = self._upscale_on_cpu(upscale_model, images, upscale_method, new_width, new_height)
        elif device_strategy == "keep_loaded":
            upscale_model = upscale_model.to(device)
            result_frames = self._upscale_batch_keep_loaded(upscale_model, images, device, upscale_method, new_width, new_height)
        else:
            result_frames = self._upscale_batch_load_unload(upscale_model, images, device, upscale_method, new_width, new_height)
        
        return (torch.stack(result_frames),)

    def load_upscale_model_with_spandrel(self, model_path):
        """Load the upscale model from the given path using Spandrel."""
        if ModelLoader is None:
            print("Spandrel library is not available. Cannot load model.")
            return None
        
        try:
            # Spandrel сам определяет архитектуру по файлу модели
            # device="cpu" гарантирует, что модель загрузится в CPU память сначала, 
            # чтобы избежать потенциальных проблем с VRAM при загрузке больших моделей,
            # перед тем как она будет перемещена на GPU в основной логике.
            model_descriptor = ModelLoader(device="cpu").load_from_file(model_path)
            upscale_model = model_descriptor.model # Получаем саму torch.nn.Module
            
            # Spandrel модели уже должны быть в режиме eval() после загрузки, но для уверенности:
            upscale_model.eval()

            # Очистка памяти (хотя Spandrel должен управлять этим лучше)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Spandrel загруженная модель также должна иметь атрибут .scale
            # Если его нет, или он не соответствует ожиданиям, это может потребовать дополнительной обработки
            # или проверки совместимости модели.
            if not hasattr(upscale_model, 'scale'):
                # Попытка определить масштаб из дескриптора, если возможно,
                # или установить значение по умолчанию/вызвать ошибку
                print(f"Warning: Loaded model {model_path} via Spandrel does not have a direct 'scale' attribute. Attempting to infer or use descriptor scale.")
                if hasattr(model_descriptor, 'scale'):
                     upscale_model.scale = model_descriptor.scale
                else:
                    # Если масштаб критичен и не может быть определен, это проблема.
                    # Для примера, здесь можно установить некий дефолтный или ожидаемый масштаб,
                    # но лучше, если модель его предоставляет.
                    # В вашем коде `upscale_model.scale` используется в `tiled_scale`,
                    # так что это важно.
                    print(f"Critical: Scale could not be determined for model {model_path}. Tiled_scale might not work as expected.")
                    # Можно присвоить дефолтное значение, если это приемлемо, например 1 или 2, или поднять ошибку.
                    # upscale_model.scale = 1 # Пример, требует аккуратности

            return upscale_model
        except Exception as e:
            print(f"Error loading model with Spandrel: {e}")
            # Тут можно добавить логику для возврата к старому методу загрузки, если он есть,
            # или просто вернуть None, чтобы обработать ошибку выше.
            # return self.load_upscale_model_old_method(model_path) # Если есть старый метод
            return None

    # Важно: Ваша функция tiled_scale ожидает, что у модели есть атрибут `scale`.
    # `upscale_amount=upscale_model.scale`
    # Spandrel обычно предоставляет это, но стоит проверить.

    # Старый метод загрузки, переименованный для возможного использования как fallback
    def load_upscale_model_old_method(self, model_path):
        """Load the upscale model from the given path (OLD METHOD)"""
        from comfy_extras.chainner_models import model_loading # Этот импорт теперь будет здесь
        
        print("WARNING: Using deprecated comfy_extras.chainner_models. Consider updating your models or node for Spandrel.")
        sd = comfy.utils.load_torch_file(model_path)
        upscale_model = model_loading.load_state_dict(sd).eval()
        
        del sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return upscale_model

    def _upscale_on_cpu(self, upscale_model, images, upscale_method, new_width, new_height):
        result_frames = []
        start_time = time.time()
        
        # Убедимся, что у модели есть атрибут scale
        model_scale_factor = getattr(upscale_model, 'scale', 1) # Если нет, ставим 1 чтобы не сломать tiled_scale

        for i in range(images.shape[0]):
            frame = images[i:i+1]
            in_img = frame.movedim(-1, -3)
            
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=64,
                tile_y=64,
                overlap=8, 
                upscale_amount=model_scale_factor # Используем полученный масштаб
            )
            
            upscaled = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
            samples = upscaled.movedim(-1, 1)
            s_resized = comfy.utils.common_upscale(samples, new_width, new_height, upscale_method, crop="disabled")
            s_resized = s_resized.movedim(1, -1)
            
            result_frames.append(s_resized[0])
            self.step += 1
            
            elapsed = time.time() - start_time
            eta = elapsed / self.step * (self.steps - self.step) if self.step > 0 else 0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            percent = (self.step / self.steps) * 100
            print(f"\r\033[32m|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}\033[0m", end="", flush=True)
            
            del in_img, s, upscaled, samples, s_resized
            gc.collect()
        
        print()
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
            self.step += 1
            
            elapsed = time.time() - start_time
            eta = elapsed / self.step * (self.steps - self.step) if self.step > 0 else 0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            percent = (self.step / self.steps) * 100
            print(f"\r\033[32m|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}\033[0m", end="", flush=True)
            
            del in_img, s, upscaled, samples, s_resized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print()
        return result_frames

    def _upscale_batch_load_unload(self, upscale_model, images, device, upscale_method, new_width, new_height):
        result_frames = []
        start_time = time.time()
        model_scale_factor = getattr(upscale_model, 'scale', 1)

        for i in range(images.shape[0]):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            upscale_model = upscale_model.to(device)
            frame = images[i:i+1]
            in_img = frame.movedim(-1, -3).to(device)
            
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
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
            self.step += 1
            
            elapsed = time.time() - start_time
            eta = elapsed / self.step * (self.steps - self.step) if self.step > 0 else 0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            percent = (self.step / self.steps) * 100
            print(f"\r\033[32m|{'█' * int(percent/5)}{' ' * (20-int(percent/5))}| {self.step}/{self.steps} [{percent:.1f}%] - {elapsed_str}<{eta_str}\033[0m", end="", flush=True)
            
            del in_img, s, upscaled, samples, s_resized
            upscale_model = upscale_model.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print()
        return result_frames

# Optional companion node to manage memory explicitly during video processing
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
                if hasattr(torch.cuda, 'caching_allocator_delete_caches'):
                    torch.cuda.caching_allocator_delete_caches()
        
        if report_memory == "enable" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"After cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        return (images,)

NODE_CLASS_MAPPINGS = {
    "TS_Video_Upscale_With_Model": TS_Video_Upscale_With_Model,
    "TS_Free_Video_Memory": TS_Free_Video_Memory,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Video_Upscale_With_Model": "TS Video Upscale With Model",
    "TS_Free_Video_Memory": "TS Free Video Memory",
}