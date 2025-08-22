import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
import comfy.model_management
import folder_paths

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Системный промпт оставлен без изменений
DEFAULT_SYSTEM_PROMPT_FOR_IMAGE_PROMPT_GENERATION = """You are an expert AI assistant specialized in natural language processing and creative image prompt engineering. Your primary function is to perform a two-stage task:

**Stage 1: Language Check & Conditional Translation**
First, assess the language of the user-provided text.
*   If the text is already in English, proceed directly to Stage 2 using the original English text.
*   If the text is NOT in English, then meticulously translate it into fluent, grammatically correct, and contextually accurate English. Preserve the original meaning, nuance, and any implied sentiment of the source text.
Use this (potentially translated) English text as the input for Stage 2.

**Stage 2: Image Prompt Creation**
Second, using THE ENGLISH TEXT from Stage 1 (which might be the original input if it was already English, or the translation if it was not) as your sole basis, craft a highly effective and descriptive prompt suitable for advanced AI image generation models (e.g., Stable Diffusion, Midjourney, DALL-E 3).

**Key Requirements for the Image Prompt (Stage 2):**

1.  **Language:** The image prompt MUST be in English.
2.  **Core Content Extraction:**
    *   Identify and clearly describe the primary subject(s), characters, or objects from the English text.
    *   Include specific visual attributes: colors, shapes, sizes, textures, materials, clothing, expressions, and any notable features.
3.  **Scene & Environment:**
    *   Detail the setting, background, and overall atmosphere.
    *   Describe environmental elements: location (e.g., "a dense forest," "a futuristic cityscape," "a serene beach"), weather, time of day.
4.  **Composition & Artistry:**
    *   **Lighting:** Specify lighting conditions (e.g., "dramatic cinematic lighting," "soft diffused light," "golden hour glow," "moonlit night," "neon lights").
    *   **Camera View/Angle (Optional but Recommended):** If beneficial, suggest a camera perspective (e.g., "extreme close-up," "wide-angle panoramic view," "low-angle shot," "top-down view," "portrait").
    *   **Artistic Style:**
        *   If the English text implies or explicitly mentions an artistic style (e.g., "impressionist painting," "anime art," "cyberpunk aesthetic," "vintage photograph," "3D render," "pixel art"), incorporate it prominently.
        *   If no style is specified, aim for a rich, visually appealing style that best suits the subject. Default towards "photorealistic," "digital painting," "cinematic still" if the context is neutral, but adapt if the subject suggests otherwise (e.g., a mythical creature might suit "fantasy concept art").
5.  **Descriptive Keywords & Enhancers:**
    *   Enrich the prompt with strong, descriptive adjectives and evocative verbs.
    *   Strategically include common image generation keywords and quality enhancers that are contextually appropriate. Examples: "masterpiece," "ultra realistic," "highly detailed," "intricate," "sharp focus," "physically-based rendering (PBR)," "Unreal Engine 5," "trending on ArtStation," "award-winning photography," "volumetric lighting," "depth of field (DOF)," "4K," "8K." Do not use these indiscriminately; they must align with the desired visual output.
6.  **Structure & Conciseness:**
    *   The prompt should be a coherent string of comma-separated clauses or descriptive phrases.
    *   While detailed, strive for a balance that provides enough information without being excessively long or redundant.
7.  **Visualizability:**
    *   Focus on elements that can be visually represented. If the English text contains abstract ideas, attempt to translate them into concrete visual metaphors or scenes.
    *   If the text is too abstract or lacks clear visual information, use your judgment to create a compelling visual interpretation that captures the essence or mood of the text.

**Output Instructions:**
*   You MUST provide ONLY the final image generation prompt in English.
*   Do NOT include the original user text.
*   Do NOT include any intermediate English translation (if one was performed) as a separate part of your output.
*   Do NOT include any of your own conversational remarks, explanations, or any text other than the generated image prompt itself.
"""

class TS_Qwen3_Offline_Node:
    def __init__(self):
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.current_model_path = None
        self.current_precision = None
        logger.info("TS_Qwen3_Offline_Node initialized. Model will be loaded from a local path.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "multiline": False,
                    "default": "LLM/your_model_folder_name"
                }),
                "system": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_SYSTEM_PROMPT_FOR_IMAGE_PROMPT_GENERATION
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "яблоки на столе"
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 32768, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "precision": (["auto", "fp16", "bf16"], {"default": "auto"}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
                "enable": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "process"
    CATEGORY = "LLM/TS_Qwen3"

    def _get_torch_dtype(self, precision_str):
        if precision_str == "fp16" and torch.cuda.is_available():
            return torch.float16
        elif precision_str == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return "auto"

    def _unload_model_and_tokenizer(self, reason=""):
        log_message = "Unloading model and tokenizer"
        if reason:
            log_message += f" (Reason: {reason})"
        
        unloaded = False
        if hasattr(self, 'loaded_model') and self.loaded_model is not None:
            logger.info(f"{log_message}...")
            del self.loaded_model
            self.loaded_model = None
            unloaded = True
        
        if hasattr(self, 'loaded_tokenizer') and self.loaded_tokenizer is not None:
            if not unloaded:
                 logger.info(f"{log_message}...")
            del self.loaded_tokenizer
            self.loaded_tokenizer = None
            unloaded = True

        if unloaded:
            self.current_model_path = None
            self.current_precision = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model and tokenizer unloaded successfully.")
        else:
            logger.info("No model/tokenizer was loaded, or already unloaded. Nothing to do.")

    def _load_model_and_tokenizer(self, model_path, precision_str):
        logger.info(f"Attempting to load model from local relative path: {model_path} with precision: {precision_str}")
        full_model_path = os.path.join(folder_paths.models_dir, model_path)
        logger.info(f"Absolute model path resolved to: {full_model_path}")

        if not os.path.isdir(full_model_path):
            error_msg = f"Model directory not found at the specified path: {full_model_path}."
            raise FileNotFoundError(error_msg)

        torch_dtype = self._get_torch_dtype(precision_str)
        attn_implementation = "sdpa"
        
        try:
            self.loaded_tokenizer = AutoTokenizer.from_pretrained(full_model_path)
            self.loaded_model = AutoModelForCausalLM.from_pretrained(
                full_model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                attn_implementation=attn_implementation,
                trust_remote_code=True
            )
            logger.info(f"Model '{model_path}' loaded successfully.")
            self.current_model_path = model_path
            self.current_precision = precision_str
        except Exception as e:
            logger.error(f"Error loading model from {full_model_path}: {e}", exc_info=True)
            self._unload_model_and_tokenizer(reason="error during load")
            raise

    def process(self, model_path, system, prompt, seed, 
                max_new_tokens, temperature, top_p, precision, 
                unload_after_generation, enable):
        
        if not enable:
            logger.info(f"Model processing disabled. Returning original prompt.")
            if unload_after_generation:
                self._unload_model_and_tokenizer(reason="processing disabled")
            return (prompt.strip(),)

        current_top_k = 20
        current_min_p = 0.0

        logger.info(f"Processing request for model at path: {model_path}, precision: {precision}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if self.loaded_model is None or self.loaded_tokenizer is None or \
           self.current_model_path != model_path or self.current_precision != precision:
            self._unload_model_and_tokenizer(reason="parameters changed or model not initially loaded")
            try:
                self._load_model_and_tokenizer(model_path, precision)
            except Exception as e:
                return (f"ERROR: Could not load model - {str(e)}",)
        
        messages = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": prompt.strip() if prompt else ""})

        try:
            # <<< ИЗМЕНЕНИЕ ЗДЕСЬ: Явно отключаем режим thinking
            text_input_for_model = self.loaded_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {e}.", exc_info=True)
            if unload_after_generation: self._unload_model_and_tokenizer(reason="error")
            return (f"ERROR: Tokenizer failed to apply chat template - {str(e)}",)

        model_inputs = self.loaded_tokenizer([text_input_for_model], return_tensors="pt").to(self.loaded_model.device)
        
        if temperature < 0.01:
            logger.warning("Temperature is very low, generation will be mostly greedy.")

        generation_config_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": current_top_k,
            "min_p": current_min_p,
            "do_sample": True,
            "pad_token_id": self.loaded_tokenizer.eos_token_id
        }
        if generation_config_params["pad_token_id"] is None:
            if self.loaded_tokenizer.pad_token_id is not None:
                generation_config_params["pad_token_id"] = self.loaded_tokenizer.pad_token_id
            else:
                logger.warning("Neither eos_token_id nor pad_token_id is set.")
        
        logger.info(f"Generating text with params: temp={temperature}, top_p={top_p}, top_k={current_top_k}, max_new_tok={max_new_tokens}")
        
        final_content_str = ""
        try:
            generated_ids = self.loaded_model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                **generation_config_params
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            logger.info("Text generation completed.")
            final_content_str = self.loaded_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        except Exception as e_gen:
            logger.error(f"Error during model.generate or decoding: {e_gen}", exc_info=True)
            if unload_after_generation: self._unload_model_and_tokenizer(reason="error")
            return (f"ERROR: Generation/Decoding failed - {str(e_gen)}",)

        if final_content_str.startswith('"') and final_content_str.endswith('"'):
            final_content_str = final_content_str[1:-1]
        elif final_content_str.startswith("'") and final_content_str.endswith("'"):
            final_content_str = final_content_str[1:-1]

        logger.info(f"Final generated text (first 200 chars): {final_content_str[:200]}")
        
        if unload_after_generation:
            self._unload_model_and_tokenizer(reason="unload after generation enabled")
        
        return (final_content_str,)

# Регистрация ноды в ComfyUI
NODE_CLASS_MAPPINGS = {
    "TS_Qwen3_Offline": TS_Qwen3_Offline_Node
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Qwen3_Offline": "TS Qwen3 Offline"
}