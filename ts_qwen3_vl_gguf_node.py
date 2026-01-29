import os
import logging
import folder_paths
import json
import base64
import gc
import inspect
import sys
import ctypes
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import comfy.model_management as mm

# ===============================================
# 1. Настройки и Логирование
# ===============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TS_Qwen_3VL_GGUF")

# Путь поиска моделей
llm_root = os.path.join(folder_paths.models_dir, "LLM")
if "llm_models" in folder_paths.folder_names_and_paths:
    base_paths, base_exts = folder_paths.folder_names_and_paths["llm_models"]
    folder_paths.folder_names_and_paths["llm_models"] = (base_paths, base_exts | {".gguf"})
else:
    folder_paths.folder_names_and_paths["llm_models"] = ([llm_root], {".gguf"})

# ===============================================
# 2. DLL Fix (Windows)
# ===============================================
def fix_cuda_paths():
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            if hasattr(os, 'add_dll_directory'):
                try: os.add_dll_directory(torch_lib)
                except: pass
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception: pass

fix_cuda_paths()

# ===============================================
# 3. Импорт Qwen Handler (Исправлено)
# ===============================================
LLAMA_CPP_AVAILABLE = False
VISION_HANDLER = None
HANDLER_NAME = "None"

try:
    from llama_cpp import Llama
    
    # Логика выбора хендлера
    try:
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
        VISION_HANDLER = Qwen3VLChatHandler
        HANDLER_NAME = "Qwen3VLChatHandler"
    except ImportError:
        try:
            from llama_cpp.llama_chat_format import Qwen2_5VLChatHandler
            VISION_HANDLER = Qwen2_5VLChatHandler
            HANDLER_NAME = "Qwen2_5VLChatHandler"
        except ImportError:
            try:
                from llama_cpp.llama_chat_format import Qwen2VLChatHandler
                VISION_HANDLER = Qwen2VLChatHandler
                HANDLER_NAME = "Qwen2VLChatHandler"
            except ImportError:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                VISION_HANDLER = Llava15ChatHandler
                HANDLER_NAME = "Llava15ChatHandler"

    LLAMA_CPP_AVAILABLE = True
    logger.info(f"Llama-cpp loaded. Handler class: {HANDLER_NAME}")

except ImportError:
    logger.error("CRITICAL: llama-cpp-python not found.")
except Exception as e:
    logger.error(f"CRITICAL INIT ERROR: {e}")

# ===============================================
# 4. Менеджер моделей
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
# 6. Нода (TS Qwen 3 VL GGUF)
# ===============================================
class TS_Qwen_3VL_GGUF:
    
    # --- ВНУТРЕННИЕ НАСТРОЙКИ (SAFETY) ---
    N_GPU_LAYERS = -1
    N_CTX = 24576
    FLASH_ATTN = False 
    BATCH_SIZE = 256 # Безопасный размер батча для предотвращения BSOD
    # -------------------------------------

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        preset_options = preset_keys + ["Your instruction"]
        
        file_list = []
        if os.path.exists(llm_root):
            for root, dirs, files in os.walk(llm_root):
                for file in files:
                    if file.lower().endswith(".gguf") and "mmproj" not in file.lower():
                        file_list.append(os.path.relpath(os.path.join(root, file), llm_root))
        file_list.sort()

        return {
            "required": {
                "model_name": (file_list,),
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

    # --- Утилиты ---
    def tensor_to_pil_list(self, tensor):
        if tensor is None: return []
        images = []
        # Clamp важен для стабильности
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
        # JPEG quality 90 для баланса скорости/качества
        pil_image.save(buffered, format="JPEG", quality=90) 
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _find_projector(self, model_filename):
        full_model_path = os.path.join(llm_root, model_filename)
        model_dir = os.path.dirname(full_model_path)
        files = os.listdir(model_dir)
        projectors = [f for f in files if "mmproj" in f.lower() and f.endswith(".gguf")]
        
        if not projectors:
            raise FileNotFoundError(f"Projector (mmproj) not found in {model_dir}")
        return os.path.join(model_dir, projectors[0]), full_model_path

    # --- Загрузка ---
    def _load_model(self, model_name):
        projector_path, full_model_path = self._find_projector(model_name)
        
        cache_key = f"{model_name}_safe_{self.N_GPU_LAYERS}_{self.N_CTX}"
        cached = GGUF_MANAGER.get(cache_key)
        if cached:
            return cached

        # 1. Жесткая синхронизация и очистка перед загрузкой
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        logger.info("Unloading ComfyUI models to free VRAM...")
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info(f"Loading GGUF (Safe Mode): {os.path.basename(full_model_path)}")
        logger.info(f"Handler Class: {HANDLER_NAME}")
        
        if VISION_HANDLER is None:
            raise ImportError(f"Vision Handler class is None. Check llama-cpp installation.")

        # Настройка Handler
        handler_kwargs = {"clip_model_path": projector_path, "verbose": False}
        if "Qwen" in HANDLER_NAME:
            handler_kwargs["force_reasoning"] = False
            handler_kwargs["image_min_tokens"] = 1024 if self.N_CTX > 8192 else 256
            
        chat_handler = VISION_HANDLER(**handler_kwargs)

        llm = Llama(
            model_path=full_model_path,
            chat_handler=chat_handler,
            n_ctx=self.N_CTX,
            n_gpu_layers=self.N_GPU_LAYERS,
            flash_attn=self.FLASH_ATTN,
            n_batch=self.BATCH_SIZE, # 256 для стабильности
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

        try:
            llm = self._load_model(model_name)
        except Exception as e:
            logger.error(f"GGUF Init Crash: {e}")
            return (f"Error loading model: {e}", image)

        # Параметры
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
        
        # Обработка картинок / Видео
        processed_tensors = []

        if image is not None:
            pil_images = self.tensor_to_pil_list(image)
            
            # Прореживание
            if len(pil_images) > video_max_frames:
                indices = np.linspace(0, len(pil_images) - 1, video_max_frames, dtype=int)
                pil_images = [pil_images[i] for i in indices]
                logger.info(f"Video: sampled {len(pil_images)} frames")

            for pil_img in pil_images:
                resized = self.resize_image(pil_img, max_image_size)
                img_url = self.image_to_base64_url(resized)
                user_content.append({"type": "image_url", "image_url": {"url": img_url}})
                
                # Возврат обработанной картинки
                img_np = np.array(resized).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                processed_tensors.append(img_tensor)

        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        logger.info(f"Generating (Temp: {temp_val})...")
        
        # 2. Синхронизация ПЕРЕД генерацией
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
            logger.error(f"Generation Error: {e}")
            if "memory slot" in err_msg or "n_ctx" in err_msg:
                output_text = f"Error: Context Limit! Increase N_CTX or reduce frames."
            else:
                output_text = f"Generation Error: {e}"

        # 3. Синхронизация ПОСЛЕ генерации
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