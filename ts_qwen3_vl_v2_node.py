import torch
import os
import logging
import comfy.model_management as mm
import folder_paths
import gc
import json
import numpy as np
import time
from PIL import Image
from safetensors.torch import load_file
from safetensors import safe_open
from transformers import (
    AutoConfig, 
    AutoProcessor, 
    AutoModelForVision2Seq
)

# ===============================================
# 1. Глобальные настройки (безопасные)
# ===============================================
def _get_logger():
    return logging.getLogger("TS_Qwen_3VL_V2")

# ===============================================
# 2. Обертка (Wrapper)
# ===============================================
class QwenComfyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def device(self):
        return self.model.device

    @device.setter
    def device(self, value):
        pass

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

# ===============================================
# 3. Менеджер моделей
# ===============================================
class QwenModelManager:
    def __init__(self):
        self.cache = {}
        self.cache_order = []

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, model, processor):
        self.cache[key] = (model, processor)
        if key not in self.cache_order:
            self.cache_order.append(key)

    def unload(self, key):
        if key in self.cache:
            del self.cache[key]
            if key in self.cache_order:
                self.cache_order.remove(key)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mm.soft_empty_cache()
            _get_logger().info(f"[TS Manager] Model unloaded: {key}")

    def unload_all(self):
        for key in list(self.cache.keys()):
            self.unload(key)

# ===============================================
# 4. Пресеты
# ===============================================

# ===============================================
# 5. Основной класс ноды
# ===============================================
class TS_Qwen_3VL_V2:
    _manager = None
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _presets_cache = None
    _presets_mtime = None
    _preset_keys_cache = None
    _models_cache = None
    _models_cache_mtime = None
    _optimal_dtype_cache = None

    def __init__(self):
        if self.__class__._manager is None:
            self.__class__._manager = QwenModelManager()
        self._logger = _get_logger()

    @classmethod
    def _get_presets(cls):
        presets_path = os.path.join(cls._current_dir, "qwen_3_vl_presets.json")
        mtime = os.path.getmtime(presets_path) if os.path.exists(presets_path) else None
        if cls._presets_cache is None or cls._presets_mtime != mtime:
            data = {}
            if mtime is not None:
                try:
                    with open(presets_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if not isinstance(data, dict):
                            data = {}
                except Exception as e:
                    _get_logger().warning(f"[TS Qwen] Failed to load presets: {e}")
                    data = {}
            cls._presets_cache = data
            cls._preset_keys_cache = list(data.keys())
            cls._presets_mtime = mtime
        return cls._presets_cache, cls._preset_keys_cache

    @classmethod
    def _get_safetensors_list(cls):
        llm_root = os.path.join(folder_paths.models_dir, "LLM")
        stamp = os.path.getmtime(llm_root) if os.path.exists(llm_root) else None
        if cls._models_cache is None or cls._models_cache_mtime != stamp:
            safetensors_files = []
            if stamp is not None:
                for root, _, files in os.walk(llm_root):
                    for file in files:
                        if file.lower().endswith(".safetensors"):
                            rel_path = os.path.relpath(os.path.join(root, file), llm_root)
                            safetensors_files.append(rel_path)
            safetensors_files.sort()
            cls._models_cache = safetensors_files
            cls._models_cache_mtime = stamp
        return cls._models_cache or []

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        cls._get_presets()
        cls._get_safetensors_list()
        return (cls._presets_mtime, cls._models_cache_mtime)

    @classmethod
    def INPUT_TYPES(cls):
        _, preset_keys = cls._get_presets()
        preset_options = preset_keys + ["Your instruction"]
        safetensors_files = cls._get_safetensors_list()
        
        # --- Локальный кэш списка моделей ---
        return {
            "required": {
                "model_name": (safetensors_files,),
                "system_preset": (preset_options, {"default": preset_options[0] if preset_options else "Your instruction"}),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "enable": ("BOOLEAN", {"default": True}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
                "max_image_size": ("INT", {"default": 1280, "min": 256, "max": 4096, "step": 32}),
                "video_max_frames": ("INT", {"default": 16, "min": 4, "max": 256, "step": 4}),
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
    CATEGORY = "TS/LLM"

    # --- Утилиты ---
    def tensor_to_pil_list(self, tensor):
        if tensor is None:
            return []
        images = []
        tensor = tensor.detach().cpu()
        for i in range(tensor.shape[0]):
            arr = np.clip(tensor[i].numpy(), 0.0, 1.0) * 255.0
            img = Image.fromarray(arr.astype(np.uint8))
            images.append(img)
        return images

    def pil_to_tensor(self, pil_list):
        if not pil_list:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        tensors = []
        for img in pil_list:
            if img.mode != "RGB":
                img = img.convert("RGB")
            t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)
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

    # --- Определение типа данных (Smart Precision) ---
    def _get_optimal_dtype(self):
        """Auto-select the best dtype for the current device."""
        if self.__class__._optimal_dtype_cache is not None:
            return self.__class__._optimal_dtype_cache

        if not torch.cuda.is_available():
            self._logger.warning("[TS Qwen] CUDA not found, falling back to float32")
            dtype = torch.float32
        elif torch.cuda.is_bf16_supported():
            self._logger.info("[TS Qwen] Detected Ampere+ GPU. Using BFloat16 mode for maximum performance.")
            dtype = torch.bfloat16
        else:
            self._logger.info("[TS Qwen] Detected older GPU (Turing/Pascal). Falling back to Float16 compatibility mode.")
            dtype = torch.float16

        self.__class__._optimal_dtype_cache = dtype
        return dtype

    def _detect_config(self, ckpt_path):
        configs_dir = os.path.join(self._current_dir, "configs")
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
            self._logger.error(f"[TS Qwen] Failed to read model header: {e}")
            raise RuntimeError(f"Could not read safetensors header.")

        if model_hidden_size is None:
             raise RuntimeError("Could not determine hidden_size from .safetensors file.")

        self._logger.info(f"[TS Qwen] Model fingerprint: hidden_size={model_hidden_size}")

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
                            self._logger.info(f"[TS Qwen] Found matching config in: {os.path.basename(folder)}")
                            break
                except Exception as e:
                    self._logger.warning(f"Error parsing config in {folder}: {e}")

        if matched_config_dir:
            return matched_config_dir
        else:
            raise FileNotFoundError(f"No config found matching hidden_size={model_hidden_size} in 'configs/'.")

    # --- Подготовка памяти ---
    def _prepare_memory(self):
        self._logger.info("[TS Qwen] Memory: Soft cleanup before model load...")
        mm.soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_model_from_config(self, config, compute_dtype):
        start = time.time()
        used_meta = False
        try:
            from accelerate import init_empty_weights
            self._logger.info("[TS Qwen] Using accelerate.init_empty_weights for fast init.")
            with init_empty_weights():
                model = AutoModelForVision2Seq.from_config(
                    config,
                    trust_remote_code=True,
                    torch_dtype=compute_dtype
                )
            used_meta = True
        except ImportError:
            model = AutoModelForVision2Seq.from_config(
                config,
                trust_remote_code=True,
                torch_dtype=compute_dtype
            )
        except Exception as e:
            self._logger.warning(f"[TS Qwen] Fast init failed, falling back. ({e})")
            model = AutoModelForVision2Seq.from_config(
                config,
                trust_remote_code=True,
                torch_dtype=compute_dtype
            )
        if any(p.is_meta for p in model.parameters()):
            used_meta = True
        self._logger.info(f"[TS Qwen] Model init done in {time.time() - start:.2f}s")
        return model, used_meta

    # --- Загрузка модели ---
    def _load_model(self, model_name):
        llm_root = os.path.join(folder_paths.models_dir, "LLM")
        ckpt_path = os.path.join(llm_root, model_name)
        
        try:
            config_dir = self._detect_config(ckpt_path)
        except Exception as e:
            raise RuntimeError(f"Config detection failed: {e}")

        # Определяем оптимальный dtype (BF16 vs FP16)
        compute_dtype = self._get_optimal_dtype()
        device = mm.get_torch_device()
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if torch.cuda.is_available() and device.type != "cuda":
            self._logger.warning(f"[TS Qwen] CUDA available but device is {device}. Forcing CUDA.")
            device = torch.device("cuda")
        cache_key = f"{model_name}|{device}|{compute_dtype}"

        cached = self.__class__._manager.get(cache_key)
        if cached:
            cached_model, cached_processor = cached
            if torch.cuda.is_available() and cached_model.device.type != "cuda":
                self._logger.warning("[TS Qwen] Cached model is on CPU. Reloading on CUDA.")
                self.__class__._manager.unload(cache_key)
            else:
                return cached

        self._prepare_memory()

        self._logger.info(f"[TS Qwen] Loading Config from: {config_dir}")

        config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True, local_files_only=True)
        if hasattr(config, "attn_implementation"):
            config.attn_implementation = "sdpa"
        
        processor = AutoProcessor.from_pretrained(config_dir, trust_remote_code=True, local_files_only=True)

        self._logger.info(f"[TS Qwen] Initializing model using {compute_dtype}...")

        try:
            model, used_meta = self._build_model_from_config(config, compute_dtype)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._logger.warning("[TS Qwen] OOM during init! Attempting soft cleanup...")
                self._prepare_memory()
                model, used_meta = self._build_model_from_config(config, compute_dtype)
            else:
                raise e
        if used_meta:
            if hasattr(model, "to_empty"):
                model = model.to_empty(device=device)
            else:
                raise RuntimeError("Meta model requires to_empty(), but it is not available.")

        self._logger.info("[TS Qwen] Loading weights and converting on-the-fly...")
        weights_start = time.time()
        state_dict = load_file(ckpt_path, device="cpu")
        self._logger.info(f"[TS Qwen] Weights loaded in {time.time() - weights_start:.2f}s")
        
        # load_state_dict автоматически кастит веса из файла в тип модели (compute_dtype)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            self._logger.info(f"[TS Qwen] Missing keys (usually fine for fine-tunes): {len(missing)}")
        if unexpected:
            self._logger.info(f"[TS Qwen] Unexpected keys: {len(unexpected)}")

        if not used_meta:
            try:
                to_start = time.time()
                model.to(device)
                self._logger.info(f"[TS Qwen] Model moved to {device} in {time.time() - to_start:.2f}s")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._logger.warning("[TS Qwen] OOM during model.to(device).")
                raise e
        if torch.cuda.is_available() and device.type == "cuda" and model.device.type != "cuda":
            raise RuntimeError("Model failed to move to CUDA device.")

        model.eval()

        wrapper = QwenComfyWrapper(model)
        
        del state_dict
        gc.collect()
        
        self.__class__._manager.set(cache_key, wrapper, processor)
        return wrapper, processor

    # --- Process ---
    def process(self, model_name, system_preset, prompt, seed, max_new_tokens, enable, 
                unload_after_generation, max_image_size, video_max_frames, image=None, video=None, custom_system_prompt=None):
        
        if not enable:
            inputs_to_pass = []
            if image is not None: inputs_to_pass.append(image)
            if video is not None: inputs_to_pass.append(video)
            ret_image = torch.cat(inputs_to_pass, dim=0) if inputs_to_pass else torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (prompt, ret_image)

        self._logger.info(f"--- Starting Generation V3.0 (Smart Precision) ---")
        if image is not None:
            self._logger.info(f"[TS Qwen] image shape: {tuple(image.shape)}")
        if video is not None:
            self._logger.info(f"[TS Qwen] video shape: {tuple(video.shape)}")
        
        try:
            model, processor = self._load_model(model_name)
        except Exception as e:
            self._logger.error(f"Load Error: {e}")
            return (f"Error: {e}", torch.zeros((1, 64, 64, 3), dtype=torch.float32))

        preset_configs, _ = self._get_presets()
        preset_data = preset_configs.get(system_preset)
        if system_preset == "Your instruction" and custom_system_prompt:
            sys_prompt_text = custom_system_prompt
            gen_params = {"temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.05}
        elif preset_data:
            sys_prompt_text = preset_data.get("system_prompt", "You are a helpful assistant.")
            gen_params = dict(preset_data.get("gen_params", {}))
        else:
            sys_prompt_text = "You are a helpful assistant."
            gen_params = {"temperature": 0.7}

        if "temperature" not in gen_params:
            gen_params["temperature"] = 0.7

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
            
            for frame in video_frames:
                proc_frame = self.resize_and_crop_image(frame, max_image_size)
                all_processed_pil.append(proc_frame)
                user_content.append({"type": "image", "image": proc_frame})

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
        gen_params["use_cache"] = True 
        
        if gen_params.get("temperature", 0) > 0:
            gen_params["do_sample"] = True
        else:
            gen_params["do_sample"] = False
            
        self._logger.info(f"Generating on {model.device} with dtype {model.model.dtype}...")

        with torch.inference_mode():
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
            # Сбрасываем кэш с учетом текущего типа данных
            current_dtype = self._get_optimal_dtype()
            device = mm.get_torch_device()
            self.__class__._manager.unload(f"{model_name}|{device}|{current_dtype}")

        return (output_text, self.pil_to_tensor(all_processed_pil))

NODE_CLASS_MAPPINGS = {
    "TS_Qwen_3VL_V2": TS_Qwen_3VL_V2
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Qwen_3VL_V2": "TS Qwen 3 VL V2"
}




