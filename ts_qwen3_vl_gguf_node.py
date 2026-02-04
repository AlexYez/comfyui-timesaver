import os
import logging
import folder_paths
import json
import base64
import gc
import sys
import ctypes
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import comfy.model_management as mm

# ===============================================
# 1. Логирование
# ===============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TS_Qwen_3VL_GGUF")

# ===============================================
# 2. DLL Fix (Локальный фикс, не влияет на другие ноды)
# ===============================================
def fix_cuda_paths():
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            if hasattr(os, 'add_dll_directory'):
                try: os.add_dll_directory(torch_lib)
                except: pass
            # Мы меняем PATH только для текущего процесса Python, это безопасно
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception: pass

fix_cuda_paths()

# ===============================================
# 3. Импорт (Безопасный)
# ===============================================
LLAMA_CPP_AVAILABLE = False
QWEN_HANDLER = None
HANDLER_TYPE = "Unknown"

try:
    from llama_cpp import Llama
    try:
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
        QWEN_HANDLER = Qwen3VLChatHandler
        HANDLER_TYPE = "Qwen3VLChatHandler"
    except ImportError:
        try:
            from llama_cpp.llama_chat_format import Qwen2_5VLChatHandler
            QWEN_HANDLER = Qwen2_5VLChatHandler
            HANDLER_TYPE = "Qwen2_5VLChatHandler (Fallback)"
        except ImportError:
            # Для этой ноды критично иметь Qwen хендлер, иначе она не заработает корректно с VL
            pass

    if QWEN_HANDLER:
        LLAMA_CPP_AVAILABLE = True
        logger.info(f"Llama-cpp loaded locally. Using: {HANDLER_TYPE}")

except ImportError:
    pass # Тихо пропускаем, если нет либы, ошибка будет только при запуске ноды

# ===============================================
# 4. Менеджер Кэша
# ===============================================
class QwenGGUFManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QwenGGUFManager, cls).__new__(cls)
            cls._instance.cache = {}
            cls._instance.current_key = None
        return cls._instance

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, model):
        if self.current_key and self.current_key != key:
            self.unload(self.current_key)
        self.cache[key] = model
        self.current_key = key

    def unload(self, key):
        if key in self.cache:
            model = self.cache[key]
            try: model.close()
            except: pass
            del self.cache[key]
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"GGUF Model unloaded: {key}")

GGUF_MANAGER = QwenGGUFManager()

# ===============================================
# 5. Пресеты
# ===============================================
current_dir = os.path.dirname(os.path.abspath(__file__))
def load_presets(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f: 
            return json.load(f)
    except: return {}

presets_path = os.path.join(current_dir, "qwen_3_vl_presets.json")
PRESET_CONFIGS = load_presets(presets_path)
preset_keys = list(PRESET_CONFIGS.keys()) if PRESET_CONFIGS else []

# ===============================================
# 6. Нода (ИЗОЛИРОВАННАЯ)
# ===============================================
class TS_Qwen_3VL_GGUF:
    
    # Константы (для удобства правки в коде)
    N_GPU_LAYERS = -1
    N_CTX = 24576
    FLASH_ATTN = False 
    BATCH_SIZE = 512

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        preset_options = preset_keys + ["Your instruction"]
        
        # --- ЛОКАЛЬНЫЙ ПОИСК ФАЙЛОВ ---
        # Мы НЕ используем folder_paths.folder_names_and_paths, чтобы не ломать другие ноды.
        # Мы ищем строго в ComfyUI/models/LLM
        
        models_dir = folder_paths.models_dir
        llm_dir = os.path.join(models_dir, "LLM")
        
        files_found = []
        
        if os.path.exists(llm_dir):
            # Рекурсивный поиск .gguf
            for root, dirs, files in os.walk(llm_dir):
                for file in files:
                    if file.lower().endswith(".gguf"):
                        # Исключаем файлы проекторов
                        if "mmproj" not in file.lower():
                            # Формируем относительный путь для списка
                            rel_path = os.path.relpath(os.path.join(root, file), llm_dir)
                            files_found.append(rel_path)
        else:
            # Создаем папку, если её нет, чтобы пользователь знал куда класть
            try: os.makedirs(llm_dir, exist_ok=True)
            except: pass

        files_found.sort()

        return {
            "required": {
                "model_name": (files_found,),
                "system_preset": (preset_options, {"default": preset_options[0] if preset_options else "Your instruction"}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this video in detail."}),
                
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 8192}),
                
                "max_image_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "video_max_frames": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "enable": ("BOOLEAN", {"default": True}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "custom_system_prompt": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "processed_image")
    FUNCTION = "process"
    CATEGORY = "LLM/TS_Qwen"

    # --- Внутренние методы ---
    def tensor_to_pil_list(self, tensor):
        if tensor is None: return []
        images = []
        tensor = tensor.clamp(0, 1) 
        for i in range(tensor.shape[0]):
            img = Image.fromarray((tensor[i].cpu().numpy() * 255.0).astype(np.uint8))
            images.append(img)
        return images

    def resize_image(self, image, max_size):
        w, h = image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return image

    def image_to_base64_url(self, pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _find_projector_local(self, model_rel_path):
        # Ищем проектор относительно выбранного файла модели
        # model_rel_path = "QwenFolder/model.gguf"
        
        llm_dir = os.path.join(folder_paths.models_dir, "LLM")
        full_model_path = os.path.join(llm_dir, model_rel_path)
        
        model_folder = os.path.dirname(full_model_path)
        
        if not os.path.exists(model_folder):
             raise FileNotFoundError(f"Model folder not found: {model_folder}")

        files = os.listdir(model_folder)
        projectors = [f for f in files if "mmproj" in f.lower() and f.endswith(".gguf")]
        
        if not projectors:
            raise FileNotFoundError(f"Projector (mmproj) not found in {model_folder}")
            
        return os.path.join(model_folder, projectors[0]), full_model_path

    # --- Загрузка ---
    def _load_model(self, model_name):
        # Используем локальный поиск файлов, не зависящий от ComfyUI registry
        projector_path, full_model_path = self._find_projector_local(model_name)
        
        cache_key = f"{model_name}_safe_{self.N_GPU_LAYERS}_{self.N_CTX}"
        cached = GGUF_MANAGER.get(cache_key)
        if cached: return cached

        # Чистим память
        if torch.cuda.is_available(): torch.cuda.synchronize()
        logger.info("Unloading ComfyUI models...")
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        logger.info(f"Loading GGUF: {os.path.basename(full_model_path)}")
        
        if QWEN_HANDLER is None:
            raise ImportError("Qwen Chat Handler not found in llama-cpp-python")

        chat_handler = QWEN_HANDLER(
            clip_model_path=projector_path,
            verbose=False,
            force_reasoning=False
            # image_min_tokens не передаем для совместимости, либа сама выберет дефолт
        )

        llm = Llama(
            model_path=full_model_path,
            chat_handler=chat_handler,
            n_ctx=self.N_CTX,
            n_gpu_layers=self.N_GPU_LAYERS,
            flash_attn=self.FLASH_ATTN,
            n_batch=self.BATCH_SIZE,
            n_ubatch=self.BATCH_SIZE,
            logits_all=False,
            verbose=False
        )
        
        GGUF_MANAGER.set(cache_key, llm)
        return llm

    # --- Process ---
    def process(self, model_name, system_preset, prompt, seed, max_new_tokens, 
                enable, unload_after_generation, max_image_size, video_max_frames, image=None, custom_system_prompt=None):
        
        if not enable:
            return (prompt, image if image is not None else torch.zeros((1, 64, 64, 3)))

        if not LLAMA_CPP_AVAILABLE:
            return ("Error: llama-cpp-python not installed.", image)

        # 1. Загрузка
        try:
            llm = self._load_model(model_name)
        except Exception as e:
            logger.error(f"GGUF Init Crash: {e}")
            return (f"Error loading model: {e}", image)

        # 2. Параметры
        preset_data = PRESET_CONFIGS.get(system_preset)
        
        if system_preset == "Your instruction" and custom_system_prompt:
            sys_msg = custom_system_prompt
            temp_val = 0.7 
        elif preset_data:
            sys_msg = preset_data.get("system_prompt", "You are a helpful assistant.")
            temp_val = preset_data.get("gen_params", {}).get("temperature", 0.7)
        else:
            sys_msg = "You are a helpful assistant."
            temp_val = 0.7

        messages = [{"role": "system", "content": sys_msg}]
        user_content = []
        processed_tensors = []

        # 3. Обработка контента
        if image is not None:
            pil_images = self.tensor_to_pil_list(image)
            
            # Прореживание
            if len(pil_images) > video_max_frames:
                indices = np.linspace(0, len(pil_images) - 1, video_max_frames, dtype=int)
                pil_images = [pil_images[i] for i in indices]
                logger.info(f"Sampled to {len(pil_images)} frames")

            for pil_img in pil_images:
                resized = self.resize_image(pil_img, max_image_size)
                user_content.append({"type": "image_url", "image_url": {"url": self.image_to_base64_url(resized)}})
                
                img_np = np.array(resized).astype(np.float32) / 255.0
                processed_tensors.append(torch.from_numpy(img_np).unsqueeze(0))

        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        logger.info(f"Generating (Temp: {temp_val})...")
        
        if torch.cuda.is_available(): torch.cuda.synchronize()

        try:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temp_val,
                seed=seed,
                top_p=0.9,
                repeat_penalty=1.1
            )
            output_text = response["choices"][0]["message"]["content"]
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Gen Error: {e}")
            if "memory slot" in err_msg or "n_ctx" in err_msg:
                output_text = f"Error: Context Limit! Increase N_CTX (current {self.N_CTX}) or reduce frames."
            else:
                output_text = f"Generation Error: {e}"

        if torch.cuda.is_available(): torch.cuda.synchronize()

        if unload_after_generation:
            GGUF_MANAGER.unload(f"{model_name}_safe_{self.N_GPU_LAYERS}_{self.N_CTX}")

        if processed_tensors:
            ret_image = torch.cat(processed_tensors, dim=0)
        else:
            ret_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return (output_text, ret_image)

NODE_CLASS_MAPPINGS = {
    "TS_Qwen_3VL_GGUF": TS_Qwen_3VL_GGUF
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Qwen_3VL_GGUF": "TS Qwen 3 VL GGUF"
}