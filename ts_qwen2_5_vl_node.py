import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
import os
import logging
import comfy.model_management
import folder_paths
from huggingface_hub import snapshot_download
from PIL import Image
import numpy as np
import gc

# --- ПРОВЕРКА ЗАВИСИМОСТЕЙ ---
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False

try:
    import bitsandbytes
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

FLASH_ATTN_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- СЛОВАРЬ С ПРЕСЕТАМИ ---
PRESET_PROMPTS = {
    "Image Edit Command Translation": """You are a specialized AI that converts user requests into precise English commands for AI image editing. Your goal is to create a command that is clear, specific, and preserves the original image context.

Follow this process:

**1. Translate to English**
First, accurately translate the user's entire request into English. Keep any existing English words as they are.

**2. Rebuild as a Precise Command**
Next, restructure the translated text into a formal command using these rules:

   **A. Start with an Action Verb:** Always begin the command with a clear action like `Add`, `Replace`, `Change`, `Remove`, `Make`, or `Relight`.

   **B. Be Specific and Add Details:** If the user's request is vague, you must add logical details.
   *   **Example:** If the user says "add an animal", a better command is "Add a small, fluffy white cat sitting on the grass".

   **C. The Golden Rule: State What to Preserve.** This is the most important step. After your main command, you MUST add a clause specifying what should remain unchanged to prevent the AI from altering the entire image.
   *   **Use phrases like:** `...; keep everything else unchanged.`
   *   **Or be more specific:** `...; preserve the original font, color, and perspective.`
   *   **Or for people:** `...; keep the person's identity, expression, and clothing unchanged.`

**3. Special Rule for Text**
For any text editing, put the new text content inside English double quotes `\" \"`.
   *   **Example:** `Replace the text on the sign with \"GRAND OPENING\".`

**Final Output:**
Your final output must be ONLY the resulting single-line English editing command. Do not add any of your own comments or explanations.""",
    "Text translation to English": """You are an expert AI translation tool. Your one and only goal is to ensure the user's input text is converted into 100% fluent and accurate English.

Follow this simple process:
1. Scan the entire input text.
2. Identify all words and phrases that are **not** in English.
3. Translate these non-English parts into their most accurate English equivalent, preserving the original context.
4. Keep any words or phrases that are **already** in English exactly as they are, without changes.
5. Assemble the final text using the original English parts and your new translations to form a complete, coherent English sentence or paragraph.

Your final output must be ONLY the resulting pure English text. Do not add any of your own comments, explanations, or any text other than the translation itself.""",
    "Prompts enhance": """You are an AI Art Director creating prompts for modern AI image generators.
Your task: Take the user's idea and expand it into a single, descriptive paragraph in natural English.

1.  First, if the user's text is not in English, translate it to English.
2.  Using the English text, describe a complete scene in full sentences.
3.  Weave details about the subject, the background, the lighting, the mood, and the camera angle directly into your description.
4.  The final prompt should read like a short, vivid story or a scene from a film.
5.  IMPORTANT: Do NOT use a comma-separated list of keywords or old tags like 'masterpiece' or '8K'.

Your output must be ONLY the final descriptive paragraph.""",
    "Image generation prompt from image": """You are an AI Art Director describing a photo for a modern AI image generator.
Your task: Analyze the user's image and describe it in a single, detailed paragraph of natural English.

1.  Write in complete, descriptive sentences.
2.  Describe the main subject, their actions, the background, the lighting, and the overall mood of the scene.
3.  Include details about the camera shot, for example, "This is a close-up portrait shot..."
4.  IMPORTANT: Do NOT output a comma-separated list of keywords.

Your output must be ONLY the final descriptive paragraph.""",
    "Video generation prompt from image": """You are an AI Cinematographer creating a video prompt from a static image.
Your task: Look at the user's image and describe a short, dynamic video clip in a single paragraph.

1.  Start by describing the scene in the image.
2.  Introduce movement: describe a small action the subject is performing or how the environment is moving (wind, rain, clouds).
3.  Crucially, describe a specific camera movement (e.g., 'a slow zoom in on the subject', 'a sweeping aerial shot', 'the camera pans across the scene').
4.  Focus on creating a cinematic feeling.

Your output must be ONLY the final video prompt paragraph.""",
    "Video Prompt Enhance": """You are an AI Cinematographer creating a video prompt from a text idea.
Your task: Take the user's idea and expand it into a single, descriptive paragraph for an AI video generator.

1.  First, if the user's text is not in English, translate it to English.
2.  Describe a short scene with action and movement.
3.  Describe what the subject is doing and how the environment is moving.
4.  Crucially, add a specific camera movement (e.g., 'the camera tracks the character as they walk', 'a slow crane shot reveals the city').
5.  The final prompt should feel like a shot description from a movie script.

Your output must be ONLY the final video prompt paragraph."""
}

class TS_Qwen2_5_VL_Node:
    def __init__(self):
        self.loaded_model = None
        self.loaded_processor = None
        self.current_model_id = None
        if not QWEN_VL_UTILS_AVAILABLE: logger.error("qwen-vl-utils not installed. Please run: pip install qwen-vl-utils")
        if not BITSANDBYTES_AVAILABLE: logger.warning("bitsandbytes not installed. 4-bit and 8-bit quantization will not be available. Run: pip install bitsandbytes")
        logger.info("TS_Qwen2_5_VL_Node initialized.")

    @classmethod
    def INPUT_TYPES(cls):
        preset_options = list(PRESET_PROMPTS.keys()) + ["Your instruction"]
        precision_options = ["fp16", "bf16", "fp32"]
        if BITSANDBYTES_AVAILABLE:
            precision_options.extend(["int8", "int4"])
            
        return {
            "required": {
                "model_name": ("STRING", {"default": "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated", "multiline": False}),
                "system_preset": (preset_options, {"default": "Image Edit Command Translation"}),
                "prompt": ("STRING", {"multiline": True, "default": "сделай куртку красной кожаной"}),
                # Generation Parameters
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                # Model & Optimization Parameters
                "precision": (precision_options, {"default": "fp16"}),
                "force_full_gpu_load": ("BOOLEAN", {"default": True}),
                "use_flash_attention_2": ("BOOLEAN", {"default": True if FLASH_ATTN_AVAILABLE else False}),
                "offline_mode": ("BOOLEAN", {"default": False}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
                "enable": ("BOOLEAN", {"default": True}),
                # Image Parameters
                "max_image_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32}),
            },
            "optional": { 
                "image": ("IMAGE",),
                "custom_system_prompt": ("STRING", {"multiline": True, "forceInput": True}),
            },
            "hidden": {"hf_token": ""}
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("generated_text", "processed_image")
    FUNCTION = "process"
    CATEGORY = "LLM/TS_Qwen"

    def _get_torch_dtype(self, precision_str):
        if precision_str == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported(): return torch.bfloat16
        if precision_str == "fp16" and torch.cuda.is_available(): return torch.float16
        return torch.float32

    def _unload_model(self, reason=""):
        if self.loaded_model: del self.loaded_model
        if self.loaded_processor: del self.loaded_processor
        self.loaded_model, self.loaded_processor, self.current_model_id = None, None, None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Model and processor unloaded. Reason: {reason}")

    def _check_model_integrity(self, local_model_path):
        if not os.path.exists(local_model_path): return False
        config_path = os.path.join(local_model_path, "config.json")
        weights_index_path_safe = os.path.join(local_model_path, "model.safetensors.index.json")
        weights_index_path_bin = os.path.join(local_model_path, "pytorch_model.bin.index.json")
        return os.path.exists(config_path) and (os.path.exists(weights_index_path_safe) or os.path.exists(weights_index_path_bin))

    def _load_model(self, model_name, precision, use_flash_attention, force_full_gpu_load, offline_mode, hf_token):
        models_llm_dir = os.path.join(folder_paths.models_dir, "LLM")
        repo_name = model_name.split("/")[-1]
        local_model_path = os.path.join(models_llm_dir, repo_name)

        if offline_mode:
            logger.info(f"Offline mode. Checking for model at: {local_model_path}")
            if not self._check_model_integrity(local_model_path):
                raise FileNotFoundError(f"Offline mode: Model not found or incomplete. Run with 'offline_mode' unchecked once to download.")
        else:
            os.makedirs(models_llm_dir, exist_ok=True)
            if not self._check_model_integrity(local_model_path):
                logger.info(f"Model not found/incomplete. Starting download...")
                snapshot_download(repo_id=model_name, local_dir=local_model_path, token=hf_token if hf_token else None)

        if force_full_gpu_load and torch.cuda.is_available():
            device_map = {"": "cuda:0"}
            logger.info("Forcing full model load onto GPU.")
        else:
            device_map = "auto"
            logger.info("Using 'auto' device map for model loading.")

        load_kwargs = {"device_map": device_map}
        
        if precision == "int4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            logger.info("Loading model in 4-bit quantization.")
        elif precision == "int8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Loading model in 8-bit quantization.")
        else:
            load_kwargs["torch_dtype"] = self._get_torch_dtype(precision)
            
        if use_flash_attention and FLASH_ATTN_AVAILABLE and precision in ["fp16", "bf16"]:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2.")
        elif use_flash_attention:
            logger.warning("Flash Attention 2 is not available or not compatible with current precision. Using default attention.")

        try:
            self.loaded_processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
            self.loaded_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_model_path, trust_remote_code=True, **load_kwargs)
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise
            
        self.current_model_id = f"{model_name}_{precision}_{use_flash_attention}_{force_full_gpu_load}"
        logger.info(f"Model '{model_name}' loaded successfully from '{local_model_path}'.")

    def tensor_to_pil(self, tensor):
        if tensor is None: return None
        return Image.fromarray((tensor.cpu().numpy()[0] * 255.0).astype(np.uint8))
        
    def pil_to_tensor(self, pil_image):
        if pil_image is None: return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def resize_and_crop_image(self, image, max_size, multiple_of=32):
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width, new_height = int(width * ratio), int(height * ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        width, height = image.size
        target_width = width - (width % multiple_of)
        target_height = height - (height % multiple_of)
        
        if target_width == 0 or target_height == 0: return None
        
        left, top = (width - target_width) / 2, (height - target_height) / 2
        right, bottom = (width + target_width) / 2, (height + target_height) / 2
        
        return image.crop((left, top, right, bottom))
    
    def process(self, model_name, system_preset, prompt, seed, max_new_tokens, temperature, top_p, repetition_penalty, 
                precision, use_flash_attention_2, offline_mode, unload_after_generation, enable, 
                max_image_size, force_full_gpu_load, image=None, custom_system_prompt=None, hf_token=""):

        if not enable:
            if unload_after_generation: self._unload_model(reason="Processing disabled")
            return (prompt.strip() if prompt else "", self.pil_to_tensor(self.tensor_to_pil(image)),)
        
        if not QWEN_VL_UTILS_AVAILABLE: return ("ERROR: qwen-vl-utils not installed.", torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
        if precision in ["int4", "int8"] and not BITSANDBYTES_AVAILABLE: return (f"ERROR: {precision} requires bitsandbytes. Please install it.", torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
        
        model_id = f"{model_name}_{precision}_{use_flash_attention_2}_{force_full_gpu_load}"
        if self.loaded_model is None or self.current_model_id != model_id:
            self._unload_model(reason="Parameters changed")
            self._load_model(model_name, precision, use_flash_attention_2, force_full_gpu_load, offline_mode, hf_token)

        if system_preset == "Your instruction" and custom_system_prompt:
            system_prompt_to_use = custom_system_prompt
        else:
            system_prompt_to_use = PRESET_PROMPTS.get(system_preset, "")

        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        
        pil_image = self.tensor_to_pil(image)
        if pil_image:
            pil_image = self.resize_and_crop_image(pil_image, max_image_size)

        user_content = []
        if pil_image: user_content.append({"type": "image", "image": pil_image})
        user_content.append({"type": "text", "text": prompt.strip() if prompt else ""})
        messages = [{"role": "system", "content": system_prompt_to_use.strip()}, {"role": "user", "content": user_content}]
        
        final_content_str = ""
        try:
            text_template = self.loaded_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.loaded_processor(text=[text_template], images=image_inputs, padding=True, return_tensors="pt").to(self.loaded_model.device)

            generation_params = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": self.loaded_processor.tokenizer.eos_token_id
            }
            
            generated_ids = self.loaded_model.generate(**inputs, **generation_params)
            trimmed_ids = generated_ids[0][len(inputs.input_ids[0]):]
            final_content_str = self.loaded_processor.decode(trimmed_ids, skip_special_tokens=True).strip()

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA Out of Memory Error: {e}", exc_info=True)
            self._unload_model(reason="OOM error")
            final_content_str = "ERROR: CUDA Out of Memory. Try using a lower precision (e.g., int4), reducing max_image_size, or enabling 'unload_after_generation'."
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            if unload_after_generation: self._unload_model(reason="error during generation")
            final_content_str = f"ERROR: Generation failed - {str(e)}"
        
        if unload_after_generation:
            self._unload_model(reason="Unload after generation enabled")
            
        return (final_content_str, self.pil_to_tensor(pil_image),)

NODE_CLASS_MAPPINGS = {"TS_Qwen2.5_VL": TS_Qwen2_5_VL_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Qwen2.5_VL": "TS Qwen2.5 VL"}