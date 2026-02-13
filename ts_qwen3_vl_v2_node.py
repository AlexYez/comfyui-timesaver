import torch
import os
import logging
import gc
import json
import numpy as np
from PIL import Image
from contextlib import contextmanager

# ComfyUI Imports
import comfy.model_management as mm
import folder_paths
from safetensors.torch import load_file
from safetensors import safe_open
from transformers import (
    AutoConfig, 
    AutoProcessor, 
    AutoModelForVision2Seq
)

# ===============================================
# 1. Локальные настройки и Логгер
# ===============================================
logger = logging.getLogger("TS_Qwen_3VL_V2")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)

@contextmanager
def _temporary_tf32(enabled=True):
    # Разрешаем TF32 для ускорения на Ampere+ картах
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn

# ===============================================
# 2. Обертка модели (Wrapper)
# ===============================================
class TS_QwenWrapper(torch.nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    @property
    def device(self):
        return self.model.device

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

# ===============================================
# 3. Менеджер моделей (Singleton)
# ===============================================
class TS_QwenModelManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TS_QwenModelManager, cls).__new__(cls)
            cls._instance.cache = {}
            cls._instance.current_key = None
        return cls._instance

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, wrapper):
        if self.current_key and self.current_key != key:
            self.unload(self.current_key)
        self.cache[key] = wrapper
        self.current_key = key

    def unload(self, key):
        if key in self.cache:
            del self.cache[key]
            mm.soft_empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"[TS Manager] Model unloaded: {key}")

MODEL_MANAGER = TS_QwenModelManager()

# ===============================================
# 4. Утилиты
# ===============================================
current_dir = os.path.dirname(os.path.abspath(__file__))
presets_path = os.path.join(current_dir, "qwen_3_vl_presets.json")

def load_presets(json_path):
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f: 
                return json.load(f)
        return {}
    except Exception:
        return {}

PRESET_CONFIGS = load_presets(presets_path)
preset_keys = list(PRESET_CONFIGS.keys()) if PRESET_CONFIGS else []

# ===============================================
# 5. Основной Узел (Node)
# ===============================================
class TS_Qwen_3VL_V2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        preset_options = preset_keys + ["Your instruction"]
        
        # --- Изолированный поиск Safetensors ---
        llm_root = os.path.join(folder_paths.models_dir, "LLM")
        safetensors_files = []
        
        if os.path.exists(llm_root):
            for root, dirs, files in os.walk(llm_root):
                for file in files:
                    if file.lower().endswith(".safetensors"):
                        rel_path = os.path.relpath(os.path.join(root, file), llm_root)
                        safetensors_files.append(rel_path)
        
        safetensors_files.sort()
        
        return {
            "required": {
                "model_name": (safetensors_files,),
                "system_preset": (preset_options, {"default": preset_options[0] if preset_options else "Your instruction"}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "enable": ("BOOLEAN", {"default": True}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
                "max_image_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32}),
                "video_max_frames": ("INT", {"default": 16, "min": 4, "max": 256, "step": 4}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",), # Вход для видео-тензоров (batch of frames)
                "custom_system_prompt": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "processed_image")
    FUNCTION = "process"
    CATEGORY = "TS/LLM"

    # --- Хелперы обработки изображений ---
    def tensor_to_pil_list(self, tensor):
        if tensor is None: return []
        images = []
        for i in range(tensor.shape[0]):
            # Конвертация: [H, W, C] -> PIL
            img_np = (tensor[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            images.append(img)
        return images

    def pil_to_tensor(self, pil_list):
        if not pil_list: return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        tensors = []
        for img in pil_list:
            t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
            tensors.append(t)
        return torch.cat(tensors, dim=0)

    def resize_for_vision(self, image, max_size):
        # Qwen2-VL Processor делает ресайз сам, но предварительный даунскейл 
        # экономит память CPU при подготовке inputs
        w, h = image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        return image

    # --- Детекция конфига ---
    def _detect_config(self, ckpt_path):
        configs_dir = os.path.join(current_dir, "configs")
        if not os.path.exists(configs_dir):
            raise FileNotFoundError(f"Configs folder not found at {configs_dir}")

        model_hidden_size = None
        # Читаем header safetensors для определения архитектуры
        try:
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "model.layers.0.input_layernorm.weight" in key:
                        tensor = f.get_slice(key)
                        model_hidden_size = tensor.get_shape()[0]
                        break
                    if "model.embed_tokens.weight" in key:
                        tensor = f.get_slice(key)
                        shape = tensor.get_shape()
                        if len(shape) > 1:
                            model_hidden_size = shape[1]
                        break
        except Exception as e:
            logger.error(f"Failed to read model header: {e}")
            raise RuntimeError(f"Could not read safetensors header.")

        if model_hidden_size is None:
             raise RuntimeError("Could not determine hidden_size from .safetensors file.")

        logger.info(f"[TS] Model fingerprint: hidden_size={model_hidden_size}")

        # Ищем подходящий конфиг
        subfolders = [f.path for f in os.scandir(configs_dir) if f.is_dir()]
        for folder in subfolders:
            config_file = os.path.join(folder, "config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as cf:
                        conf_data = json.load(cf)
                        # Пытаемся найти hidden_size в разных местах конфига
                        conf_hidden = conf_data.get("hidden_size")
                        if conf_hidden is None:
                            conf_hidden = conf_data.get("text_config", {}).get("hidden_size")
                        if conf_hidden is None:
                            conf_hidden = conf_data.get("llm_config", {}).get("hidden_size")

                        if conf_hidden == model_hidden_size:
                            logger.info(f"[TS] Config match found: {os.path.basename(folder)}")
                            return folder
                except Exception:
                    continue

        raise FileNotFoundError(f"No config found matching hidden_size={model_hidden_size}")

    # --- Загрузка модели ---
    def _load_model(self, model_name):
        cache_key = f"{model_name}_TS_V2"
        cached_wrapper = MODEL_MANAGER.get(cache_key)
        if cached_wrapper:
            return cached_wrapper

        # Чистка памяти
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        llm_root = os.path.join(folder_paths.models_dir, "LLM")
        ckpt_path = os.path.join(llm_root, model_name)
        
        config_dir = self._detect_config(ckpt_path)
        
        logger.info(f"[TS] Loading model from {model_name}...")
        
        # Конфигурация
        config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
        config._attn_implementation = "sdpa" # Flash Attention 2 support if available via sdpa
        
        processor = AutoProcessor.from_pretrained(config_dir, local_files_only=True)

        # Выбор типа данных
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        
        device = mm.get_torch_device()

        with torch.device(device):
            # Инициализируем скелет модели сразу на GPU и в нужном dtype
            model = AutoModelForVision2Seq.from_config(config)
            model.to(dtype)

        # Загрузка весов
        logger.info("[TS] Loading Safetensors weights...")
        state_dict = load_file(ckpt_path)
        
        # Обработка несовпадений ключей (часто бывает при конвертации)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"[TS] Missing keys: {len(missing)}")
        
        model.eval()
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        wrapper = TS_QwenWrapper(model, processor)
        MODEL_MANAGER.set(cache_key, wrapper)
        return wrapper

    # --- Основной процесс ---
    def process(self, model_name, system_preset, prompt, seed, max_new_tokens, enable, 
                unload_after_generation, max_image_size, video_max_frames, image=None, video=None, custom_system_prompt=None):
        
        if not enable:
            return (prompt, torch.zeros((1, 64, 64, 3)))

        # 1. Загрузка
        try:
            wrapper = self._load_model(model_name)
            model = wrapper.model
            processor = wrapper.processor
        except Exception as e:
            logger.error(f"Load Error: {e}")
            return (f"Error loading model: {e}", None)

        # 2. Подготовка системного промта
        preset_data = PRESET_CONFIGS.get(system_preset)
        if system_preset == "Your instruction" and custom_system_prompt:
            sys_prompt_text = custom_system_prompt
            gen_params = {"temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.05}
        elif preset_data:
            sys_prompt_text = preset_data.get("system_prompt", "You are a helpful assistant.")
            gen_params = preset_data.get("gen_params", {})
        else:
            sys_prompt_text = "You are a helpful assistant."
            gen_params = {"temperature": 0.7}
        
        # 3. Подготовка контента (Images & Video)
        content_list = []
        process_images_list = [] # Для передачи в processor(images=...)
        process_videos_list = [] # Для передачи в processor(videos=...)
        
        debug_previews = []

        # -- Обработка Изображений --
        if image is not None:
            pil_images = self.tensor_to_pil_list(image)
            for img in pil_images:
                img_resized = self.resize_for_vision(img, max_image_size)
                # Добавляем плейсхолдер в контент сообщения
                content_list.append({"type": "image", "image": img_resized})
                # Добавляем в список для процессора
                process_images_list.append(img_resized)
                debug_previews.append(img_resized)

        # -- Обработка Видео --
        if video is not None:
            video_frames_pil = self.tensor_to_pil_list(video)
            
            # Семплинг кадров
            total_frames = len(video_frames_pil)
            if total_frames > video_max_frames:
                indices = np.linspace(0, total_frames - 1, video_max_frames, dtype=int)
                video_frames_pil = [video_frames_pil[i] for i in indices]
            
            # Ресайз каждого кадра
            video_frames_resized = [self.resize_for_vision(f, max_image_size) for f in video_frames_pil]
            
            # Qwen2-VL требует структуру для видео
            content_list.append({"type": "video", "video": video_frames_resized})
            
            # В processor videos передается как список списков кадров (list of list of PIL)
            process_videos_list.append(video_frames_resized)
            
            # Для превью возьмем первый кадр видео
            if video_frames_resized:
                debug_previews.append(video_frames_resized[0])

        # -- Добавляем текст --
        content_list.append({"type": "text", "text": prompt})

        # 4. Формирование Chat Template
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt_text}]},
            {"role": "user", "content": content_list}
        ]

        # Генерируем текстовый промт с правильными токенами (<|vision_start|>, etc.)
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 5. Токенизация и подготовка Inputs
        # ВАЖНО: передаем images и videos отдельно, чтобы processor корректно сопоставил их с токенами
        processor_args = {
            "text": [text_prompt],
            "padding": True,
            "return_tensors": "pt"
        }
        
        if process_images_list:
            processor_args["images"] = process_images_list
        
        if process_videos_list:
            processor_args["videos"] = process_videos_list

        logger.info("[TS] Processing inputs (CPU)...")
        inputs = processor(**processor_args)
        
        # 6. Перенос на GPU
        device = model.device
        inputs = inputs.to(device)
        logger.info(f"[TS] Inputs moved to {device}")

        # 7. Генерация
        gen_params["max_new_tokens"] = max_new_tokens
        gen_params["do_sample"] = gen_params.get("temperature", 0.7) > 0
        
        logger.info("[TS] Generating...")
        
        try:
            with _temporary_tf32(True):
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                with torch.inference_mode():
                    generated_ids = model.generate(**inputs, **gen_params)
        except RuntimeError as e:
            # Отлов ошибок размерности
            if "match" in str(e).lower():
                logger.error("Dimension mismatch error. Check if the model config matches the safetensors file.")
            raise e

        # 8. Декодирование
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 9. Финализация
        if unload_after_generation:
            MODEL_MANAGER.unload(f"{model_name}_TS_V2")

        return (output_text, self.pil_to_tensor(debug_previews))

NODE_CLASS_MAPPINGS = {
    "TS_Qwen_3VL_V2": TS_Qwen_3VL_V2
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Qwen_3VL_V2": "TS Qwen 3 VL V2"
}
