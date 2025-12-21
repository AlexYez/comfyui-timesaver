import torch
import os
import logging
import comfy.model_management as mm
import folder_paths
import gc
import json
import numpy as np
from PIL import Image
from safetensors.torch import load_file
from safetensors import safe_open
from transformers import (
    AutoConfig, 
    AutoProcessor, 
    AutoModelForVision2Seq
)

# ===============================================
# 1. Логирование и настройки
# ===============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TS_Qwen_3VL_FP8")

if "llm_models" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["llm_models"] = ([os.path.join(folder_paths.models_dir, "LLM")], {".safetensors"})

# ===============================================
# 2. Менеджер моделей (Singleton)
# ===============================================
class QwenModelManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QwenModelManager, cls).__new__(cls)
            cls._instance.cache = {}
            cls._instance.current_key = None
        return cls._instance

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, model, processor):
        if self.current_key and self.current_key != key:
            self.unload(self.current_key)
        self.cache[key] = (model, processor)
        self.current_key = key

    def unload(self, key):
        if key in self.cache:
            del self.cache[key]
            gc.collect()
            torch.cuda.empty_cache()
            mm.soft_empty_cache()
            logger.info(f"Model unloaded: {key}")

MODEL_MANAGER = QwenModelManager()

# ===============================================
# 3. Пресеты
# ===============================================
current_dir = os.path.dirname(os.path.abspath(__file__))

def load_presets(json_path):
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f: 
                return json.load(f)
        return {}
    except Exception:
        return {}

presets_path = os.path.join(current_dir, "qwen_3_vl_presets.json")
PRESET_CONFIGS = load_presets(presets_path)
preset_keys = list(PRESET_CONFIGS.keys()) if PRESET_CONFIGS else []

# ===============================================
# 4. Основной Класс Ноды
# ===============================================
class TS_Qwen_3VL_FP8:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        preset_options = preset_keys + ["Your instruction"]
        
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("llm_models"),),
                "system_preset": (preset_options, {"default": preset_options[0] if preset_options else "Your instruction"}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                # Добавлен параметр enable
                "enable": ("BOOLEAN", {"default": True}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
                "max_image_size": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 32}),
                "video_max_frames": ("INT", {"default": 48, "min": 4, "max": 256, "step": 4}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
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
        for i in range(tensor.shape[0]):
            img = Image.fromarray((tensor[i].cpu().numpy() * 255.0).astype(np.uint8))
            images.append(img)
        return images

    def pil_to_tensor(self, pil_list):
        if not pil_list: return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        tensors = []
        for img in pil_list:
            t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
            tensors.append(t)
        return torch.cat(tensors, dim=0)

    def resize_and_crop_image(self, image, max_size, multiple_of=32):
        w, h = image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        w, h = image.size
        tw, th = w - (w % multiple_of), h - (h % multiple_of)
        if tw == 0 or th == 0: return image
        l, t = (w - tw) / 2, (h - th) / 2
        return image.crop((l, t, l + tw, t + th))

    # --- АВТО-ОПРЕДЕЛЕНИЕ КОНФИГА ---
    def _detect_config(self, ckpt_path):
        configs_dir = os.path.join(current_dir, "configs")
        if not os.path.exists(configs_dir):
            raise FileNotFoundError(f"Configs folder not found at {configs_dir}")

        model_hidden_size = None
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

        logger.info(f"Model fingerprint: hidden_size={model_hidden_size}")

        matched_config_dir = None
        subfolders = [f.path for f in os.scandir(configs_dir) if f.is_dir()]
        
        for folder in subfolders:
            config_file = os.path.join(folder, "config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as cf:
                        conf_data = json.load(cf)
                        conf_hidden = conf_data.get("hidden_size")
                        if conf_hidden is None:
                            text_config = conf_data.get("text_config")
                            if text_config and isinstance(text_config, dict):
                                conf_hidden = text_config.get("hidden_size")
                        if conf_hidden is None:
                            llm_config = conf_data.get("llm_config")
                            if llm_config and isinstance(llm_config, dict):
                                conf_hidden = llm_config.get("hidden_size")

                        if conf_hidden == model_hidden_size:
                            matched_config_dir = folder
                            logger.info(f"✔ Found matching config in: {os.path.basename(folder)}")
                            break
                except Exception as e:
                    logger.warning(f"Error parsing config in {folder}: {e}")

        if matched_config_dir:
            return matched_config_dir
        else:
            raise FileNotFoundError(f"No config found matching hidden_size={model_hidden_size} in 'configs/'. Download config from HF.")

    # --- Очистка памяти ---
    def _free_vram(self):
        logger.info("ComfyUI Memory Management: Requesting model unload...")
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    # --- Загрузка Модели ---
    def _load_model(self, model_name):
        ckpt_path = folder_paths.get_full_path("llm_models", model_name)
        
        try:
            config_dir = self._detect_config(ckpt_path)
        except Exception as e:
            raise RuntimeError(f"Config detection failed: {e}")

        cache_key = f"{model_name}_auto_final"
        cached = MODEL_MANAGER.get(cache_key)
        if cached:
            return cached

        # Освобождаем память перед загрузкой
        self._free_vram()

        device = mm.get_torch_device()
        logger.info(f"Loading Config from: {config_dir}")

        config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True, local_files_only=True)
        processor = AutoProcessor.from_pretrained(config_dir, trust_remote_code=True, local_files_only=True)

        logger.info("Initializing model on GPU (Float16)...")
        
        try:
            with torch.device(device):
                model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)
                model.to(torch.float16)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "allocation" in str(e).lower():
                logger.warning("OOM Detected! Retrying with aggressive cleanup...")
                gc.collect()
                torch.cuda.empty_cache()
                mm.soft_empty_cache()
                with torch.device(device):
                    model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)
                    model.to(torch.float16)
            else:
                raise e

        logger.info("Loading weights...")
        state_dict = load_file(ckpt_path)
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing: logger.info(f"Missing keys: {len(missing)}")
        
        model.eval()
        del state_dict
        gc.collect()
        
        MODEL_MANAGER.set(cache_key, model, processor)
        return model, processor

    # --- Process ---
    def process(self, model_name, system_preset, prompt, seed, max_new_tokens, enable, 
                unload_after_generation, max_image_size, video_max_frames, image=None, video=None, custom_system_prompt=None):
        
        logger.info(f"--- Starting Process (Enable={enable}) ---")

        # === 1. Обработка отключения (Bypass) ===
        if not enable:
            logger.info("Node disabled. Passing inputs through.")
            # Собираем входные картинки и видео в один список для возврата
            inputs_to_pass = []
            if image is not None: inputs_to_pass.append(image)
            if video is not None: inputs_to_pass.append(video)
            
            if inputs_to_pass:
                # Объединяем тензоры по батчу
                ret_image = torch.cat(inputs_to_pass, dim=0)
            else:
                # Пустой тензор (черный квадрат 64x64), если ничего нет
                ret_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            
            return (prompt, ret_image)

        # === 2. Основная работа ===
        try:
            model, processor = self._load_model(model_name)
        except Exception as e:
            logger.error(f"Error: {e}")
            self._free_vram()
            return (f"Error: {e}", None)

        # Пресеты
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

        if "temperature" not in gen_params: gen_params["temperature"] = 0.7

        # Контент
        all_processed_pil = []
        user_content = []

        if image is not None:
            pil_list = self.tensor_to_pil_list(image)
            for img in pil_list:
                proc_img = self.resize_and_crop_image(img, max_image_size)
                all_processed_pil.append(proc_img)
                user_content.append({"type": "image", "image": proc_img})

        if video is not None:
            video_frames = self.tensor_to_pil_list(video)
            total_frames = len(video_frames)
            if total_frames > video_max_frames:
                indices = np.linspace(0, total_frames - 1, video_max_frames, dtype=int)
                video_frames = [video_frames[i] for i in indices]
            
            proc_video_frames = []
            for frame in video_frames:
                proc_frame = self.resize_and_crop_image(frame, max_image_size)
                proc_video_frames.append(proc_frame)
                all_processed_pil.append(proc_frame)
            
            user_content.append({"type": "video", "video": proc_video_frames})

        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": sys_prompt_text},
            {"role": "user", "content": user_content}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_args = {}
        if all_processed_pil:
            image_args = {"images": all_processed_pil, "padding": True}

        inputs = processor(
            text=[text],
            return_tensors="pt",
            **image_args
        )
        
        inputs = inputs.to(model.device)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        gen_params["max_new_tokens"] = max_new_tokens
        temp = gen_params.get("temperature", 0.7)
        if temp > 0:
            gen_params["do_sample"] = True
        else:
            gen_params["do_sample"] = False
            
        logger.info(f"Generating with params: {gen_params}")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                **gen_params
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if unload_after_generation:
            MODEL_MANAGER.unload(f"{model_name}_auto_final")

        return (output_text, self.pil_to_tensor(all_processed_pil))

NODE_CLASS_MAPPINGS = {
    "TS_Qwen_3VL_FP8": TS_Qwen_3VL_FP8
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Qwen_3VL_FP8": "TS Qwen 3 VL FP8"
}