import torch
import os
import logging
import comfy.model_management
import folder_paths
from huggingface_hub import snapshot_download
from PIL import Image
import numpy as np
import gc
import json
import importlib.metadata
from contextlib import contextmanager

# ===============================================
# Логирование и проверка версий
# ===============================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TS_Qwen3_VL")

# --- Импорт класса модели с фоллбэком ---
try:
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    MODEL_CLASS = Qwen3VLForConditionalGeneration
    logger.info("Imported Qwen3VLForConditionalGeneration successfully.")
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration
        MODEL_CLASS = Qwen2VLForConditionalGeneration
        logger.warning("Qwen3VLForConditionalGeneration not found. Using Qwen2VLForConditionalGeneration instead.")
    except ImportError:
        from transformers import AutoModelForCausalLM
        MODEL_CLASS = AutoModelForCausalLM
        logger.warning("Specific QwenVL classes not found. Using AutoModelForCausalLM (might not work for VL tasks properly).")

def check_and_log_versions():
    dependencies = {
        "torch": "torch",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
        "tiktoken": "tiktoken",
        "flash_attn": "flash-attn"
    }
    logger.info("--- Checking dependency versions for TS_Qwen3_VL ---")
    for name, package in dependencies.items():
        try:
            version = importlib.metadata.version(package)
            logger.info(f"  > {name}: {version}")
        except importlib.metadata.PackageNotFoundError:
            if name == "flash_attn":
                logger.warning(f"  > {name}: Not found. Flash Attention 2 will be disabled.")
            else:
                logger.error(f"  > {name}: CRITICAL - Not found. Please install this package.")
    logger.info("----------------------------------------------------")

check_and_log_versions()

# ===============================================
# Проверка наличия библиотек
# ===============================================
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# ===============================================
# Утилиты
# ===============================================
@contextmanager
def temporary_env_vars(env_vars):
    """
    Временно устанавливает переменные окружения и возвращает их обратно после выхода.
    """
    original_vars = {key: os.environ.get(key) for key in env_vars.keys()}
    
    # Установка новых значений
    for key, value in env_vars.items():
        if value is not None and value.strip() != "":
            os.environ[key] = value

    try:
        yield
    finally:
        # Восстановление старых значений
        for key, original_value in original_vars.items():
            if original_value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = original_value

# ===============================================
# Глобальный менеджер моделей
# ===============================================
class QwenModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QwenModelManager, cls).__new__(cls)
            cls._instance.cache = {}
            logger.info("Global Model Manager initialized.")
        return cls._instance

    def get_model(self, model_id): 
        return self.cache.get(model_id)

    def set_model(self, model_id, model, processor): 
        self.cache[model_id] = (model, processor)

    def unload_model(self, model_id):
        if model_id in self.cache:
            model, processor = self.cache.pop(model_id)
            del model
            del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Model {model_id} unloaded from cache.")

MODEL_MANAGER = QwenModelManager()

# ===============================================
# Загрузка пресетов
# ===============================================
def load_presets(json_path):
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f: 
                return json.load(f)
        else:
            return {}
    except Exception as e:
        logger.error(f"Error loading presets file ({json_path}): {e}")
        return {"Error": {"system_prompt": f"Could not load {os.path.basename(json_path)}", "gen_params": {}}}

presets_path = os.path.join(os.path.dirname(__file__), "qwen_3_vl_presets.json")
PRESET_CONFIGS = load_presets(presets_path)
preset_keys = list(PRESET_CONFIGS.keys()) if PRESET_CONFIGS else []

# ===============================================
# Нода TS_Qwen3_VL
# ===============================================
class TS_Qwen3_VL_Node:
    def __init__(self):
        logger.info("TS_Qwen3_VL_Node instance created.")

    @classmethod
    def INPUT_TYPES(cls):
        preset_options = preset_keys + ["Your instruction"]
        precision_options = ["fp16", "bf16", "fp32"]
        if BITSANDBYTES_AVAILABLE:
            precision_options.extend(["int8", "int4"])
        
        return {
            "required": {
                "model_name": ("STRING", {"default": "hfmaster/Qwen3-VL-2B"}),
                "system_preset": (preset_options, {"default": preset_options[0] if preset_options else "Your instruction"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "precision": (precision_options, {"default": "fp16"}),
                "use_flash_attention_2": ("BOOLEAN", {"default": True if FLASH_ATTN_AVAILABLE else False}),
                "offline_mode": ("BOOLEAN", {"default": False}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
                "enable": ("BOOLEAN", {"default": True}),
                "hf_token": ("STRING", {"multiline": False, "default": ""}),
                "max_image_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
                "video_max_frames": ("INT", {"default": 48, "min": 4, "max": 256, "step": 4}),
            },
            "optional": { 
                "image": ("IMAGE",), 
                "video": ("IMAGE",), 
                "custom_system_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                # ИЗМЕНЕНИЕ: Подсказка о множественных зеркалах
                "hf_endpoint": ("STRING", {"default": "hf-mirror.com, huggingface.co", "multiline": False}),
                "proxy": ("STRING", {"default": "", "multiline": False, "placeholder": "http://user:pass@host:port"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("generated_text", "processed_image")
    FUNCTION = "process"
    CATEGORY = "LLM/TS_Qwen"

    def _get_torch_dtype(self, p):
        if p == "bf16": return torch.bfloat16
        if p == "fp16": return torch.float16
        return torch.float32

    def _check_model_integrity(self, path):
        config = os.path.join(path, "config.json")
        index = os.path.join(path, "model.safetensors.index.json") or os.path.join(path, "pytorch_model.bin.index.json")
        safetensors = os.path.join(path, "model.safetensors")
        bin_model = os.path.join(path, "pytorch_model.bin")
        
        has_index = os.path.exists(index)
        has_model_file = os.path.exists(safetensors) or os.path.exists(bin_model)
        
        return os.path.exists(config) and (has_index or has_model_file)

    # ===============================================
    # Загрузка модели с поддержкой нескольких зеркал
    # ===============================================
    def _load_model(self, model_id, model_name, precision, use_flash_attention, offline_mode, hf_token, hf_endpoint_str, proxy):
        cached_model = MODEL_MANAGER.get_model(model_id)
        if cached_model:
            logger.info(f"Loading model '{model_id}' from cache.")
            return cached_model

        logger.info(f"Model '{model_id}' not in cache. Starting load process...")
        models_dir = os.path.join(folder_paths.models_dir, "LLM")
        repo_name = model_name.split("/")[-1]
        local_path = os.path.join(models_dir, repo_name)

        # 1. Сначала обрабатываем Offline режим
        if offline_mode:
            if not self._check_model_integrity(local_path):
                raise FileNotFoundError(f"Offline mode: Model not found at {local_path}.")
        else:
            # 2. Если модели нет или она неполная, пытаемся скачать
            if not self._check_model_integrity(local_path):
                
                # Парсим список зеркал
                endpoints = [e.strip() for e in hf_endpoint_str.split(',') if e.strip()]
                if not endpoints:
                    # Если пользователь стер все, используем дефолт
                    endpoints = ["https://huggingface.co"]

                token_arg = hf_token.strip() if hf_token and hf_token.strip() else None
                download_success = False
                last_exception = None

                # 3. Перебираем зеркала
                for i, endpoint_raw in enumerate(endpoints):
                    # Нормализация URL
                    if not endpoint_raw.startswith("http://") and not endpoint_raw.startswith("https://"):
                        current_endpoint = "https://" + endpoint_raw
                    else:
                        current_endpoint = endpoint_raw

                    logger.info(f"Attempting download from mirror [{i+1}/{len(endpoints)}]: {current_endpoint}")
                    
                    # Подготовка переменных для конкретной попытки
                    env_vars_to_set = {"HF_ENDPOINT": current_endpoint}
                    if proxy:
                        env_vars_to_set["HTTP_PROXY"] = proxy
                        env_vars_to_set["HTTPS_PROXY"] = proxy

                    try:
                        with temporary_env_vars(env_vars_to_set):
                            snapshot_download(repo_id=model_name, local_dir=local_path, token=token_arg)
                        
                        logger.info(f"Download complete successfully from {current_endpoint}.")
                        download_success = True
                        break # Выходим из цикла, если скачали
                    except Exception as e:
                        logger.warning(f"Failed to download from {current_endpoint}. Error: {e}")
                        last_exception = e
                        # Идем к следующему зеркалу в цикле
                
                if not download_success:
                    logger.error("All mirrors failed.")
                    raise RuntimeError(f"Could not download model from any provided mirrors. Last error: {last_exception}")

        # 4. Загрузка в память (здесь код стандартный)
        device = comfy.model_management.get_torch_device()
        load_kwargs = {}

        if precision == "int4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        elif precision == "int8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )
        else:
            load_kwargs["torch_dtype"] = self._get_torch_dtype(precision)

        if use_flash_attention and FLASH_ATTN_AVAILABLE and precision in ["fp16", "bf16"]:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        logger.info(f"Loading processor from {local_path}")
        processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
        
        logger.info(f"Loading model using class: {MODEL_CLASS.__name__}")

        if precision in ["int4", "int8"]:
            load_kwargs["device_map"] = "auto"
            model = MODEL_CLASS.from_pretrained(local_path, trust_remote_code=True, **load_kwargs)
        else:
            model = MODEL_CLASS.from_pretrained(local_path, trust_remote_code=True, low_cpu_mem_usage=True, **load_kwargs)
            model.to(device)

        MODEL_MANAGER.set_model(model_id, model, processor)
        logger.info(f"Model '{model_id}' loaded and cached.")
        return model, processor

    # ===============================================
    # Вспомогательные функции
    # ===============================================
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

    # ===============================================
    # Основной метод process
    # ===============================================
    def process(self, model_name, system_preset, prompt, seed, max_new_tokens,
                precision, use_flash_attention_2, offline_mode, unload_after_generation, enable,
                hf_token, max_image_size, video_max_frames, image=None, video=None, custom_system_prompt=None, 
                hf_endpoint="hf-mirror.com, huggingface.co", proxy=None):

        logger.info("--- Starting TS_Qwen3_VL process ---")
        all_processed_images = []

        if not enable:
            if image is not None: all_processed_images.extend(self.tensor_to_pil_list(image))
            if video is not None: all_processed_images.extend(self.tensor_to_pil_list(video))
            return (prompt.strip() if prompt else "", self.pil_to_tensor(all_processed_images))

        if precision in ["int4", "int8"] and not BITSANDBYTES_AVAILABLE:
            return (f"ERROR: {precision} requires bitsandbytes.", None)

        model_id = f"{model_name}_{precision}_{use_flash_attention_2}"
        
        try:
            model, processor = self._load_model(model_id, model_name, precision, use_flash_attention_2, offline_mode, hf_token, hf_endpoint, proxy)

            preset_config = PRESET_CONFIGS.get(system_preset)
            if system_preset == "Your instruction" and custom_system_prompt:
                system_prompt, gen_params = custom_system_prompt, {"temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.0}
            elif preset_config:
                system_prompt, gen_params = preset_config.get("system_prompt", ""), preset_config.get("gen_params", {})
            else:
                system_prompt, gen_params = "", {}

            torch.manual_seed(seed)
            if torch.cuda.is_available(): 
                torch.cuda.manual_seed_all(seed)

            user_content = [{"type": "text", "text": prompt.strip() if prompt else ""}]

            if image is not None:
                pil_list = self.tensor_to_pil_list(image)
                for i, img in enumerate(pil_list):
                    processed_img = self.resize_and_crop_image(img, max_image_size)
                    all_processed_images.append(processed_img)
                    user_content.insert(i, {"type": "image", "image": processed_img})

            if video is not None:
                video_frames = self.tensor_to_pil_list(video)
                total_frames = len(video_frames)
                if total_frames > video_max_frames:
                    step = total_frames / video_max_frames
                    indices = [int(i * step) for i in range(video_max_frames)]
                    video_frames = [video_frames[i] for i in indices]
                
                processed_video_frames = []
                for frame in video_frames:
                    proc_frame = self.resize_and_crop_image(frame, max_image_size)
                    processed_video_frames.append(proc_frame)
                    all_processed_images.append(proc_frame)
                
                user_content.insert(0, {"type": "video", "video": processed_video_frames, "fps": 1.0})

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]},
                {"role": "user", "content": user_content}
            ]

            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            gen_params.update({"max_new_tokens": max_new_tokens, "pad_token_id": processor.tokenizer.eos_token_id})

            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=self._get_torch_dtype(precision)):
                generated_ids = model.generate(**inputs, **gen_params)

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
            generated_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
            final_content_str = generated_text

        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            final_content_str = f"ERROR: {e}"
            if unload_after_generation and model_id in MODEL_MANAGER.cache: 
                MODEL_MANAGER.unload_model(model_id)
            return (final_content_str, self.pil_to_tensor(all_processed_images))

        if unload_after_generation:
            MODEL_MANAGER.unload_model(model_id)

        return (final_content_str, self.pil_to_tensor(all_processed_images))

NODE_CLASS_MAPPINGS = {"TS_Qwen3_VL": TS_Qwen3_VL_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Qwen3_VL": "TS Qwen 3 VL"}