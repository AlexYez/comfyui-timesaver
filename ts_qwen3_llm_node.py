import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os
import logging
import comfy.model_management
import folder_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        *   If no style is specified, aim for a rich, visually appealing style that best suits the subject. Default towards "photorealistic," "digital painting," or "cinematic still" if the context is neutral, but adapt if the subject suggests otherwise (e.g., a mythical creature might suit "fantasy concept art").
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

**Example Interaction (Input NOT in English):**

If the user provides (in Russian): "Кот в шляпе волшебника читает древнюю книгу при свете свечи в старой библиотеке."

Your expected output (the image prompt only):
"A wise black cat wearing a pointed wizard's hat, intensely reading an ancient, leather-bound tome with glowing runes, illuminated by a single flickering candlelight, in a vast, dusty, old library filled with towering bookshelves, cobwebs, magical ambiance, cinematic lighting, highly detailed fur and fabric, sharp focus, masterpiece, fantasy art."

**Example Interaction (Input ALREADY in English):**

If the user provides: "A futuristic robot explores a vibrant alien jungle with glowing plants."

Your expected output (the image prompt only):
"Futuristic robot explorer, navigating a vibrant, bioluminescent alien jungle, towering exotic trees, strange glowing flora and fauna, mysterious atmosphere, cinematic lighting, highly detailed metallic textures and lush vegetation, sharp focus, science fiction concept art, masterpiece."
"""

class TS_Qwen3_Node:
    def __init__(self):
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.current_model_name = None
        self.current_precision = None
        logger.info("TS_Qwen3_Node initialized. Model device will be managed by 'device_map=\"auto\"'.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"], {
                    "default": "Qwen/Qwen3-1.7B"
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
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "precision": (["auto", "fp16", "bf16"], {"default": "auto"}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "process"
    CATEGORY = "LLM/TS_Qwen3"

    def _get_torch_dtype(self, precision_str):
        logger.debug(f"Determining torch dtype for precision: {precision_str}")
        if precision_str == "fp16" and torch.cuda.is_available():
            return torch.float16
        elif precision_str == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return "auto"

    def _unload_model_and_tokenizer(self, reason=""):
        log_message = "Unloading model and tokenizer"
        if reason:
            log_message += f" (Reason: {reason})"
        log_message += "..."
        
        unloaded = False
        if hasattr(self, 'loaded_model') and self.loaded_model is not None:
            logger.info(f"{log_message if not unloaded else 'Continuing to unload...'}")
            del self.loaded_model
            self.loaded_model = None
            unloaded = True
        
        if hasattr(self, 'loaded_tokenizer') and self.loaded_tokenizer is not None:
            if not unloaded:
                 logger.info(f"{log_message}")
            else:
                 logger.debug("Unloading tokenizer...")
            del self.loaded_tokenizer
            self.loaded_tokenizer = None
            unloaded = True

        if unloaded:
            self.current_model_name = None
            self.current_precision = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared after unloading.")
            logger.info("Model and tokenizer unloaded successfully.")
        else:
            logger.info("No model/tokenizer was loaded, or already unloaded. Nothing to do.")


    def _load_model_and_tokenizer(self, model_name_selected, precision_str):
        logger.info(f"Attempting to load model: {model_name_selected} with precision: {precision_str}")
        
        comfy_base_models_dir = folder_paths.models_dir 
        llm_subfolder = "LLM"
        models_llm_dir = os.path.join(comfy_base_models_dir, llm_subfolder)
        os.makedirs(models_llm_dir, exist_ok=True)
        logger.info(f"Models will be stored in or loaded from: {models_llm_dir}")

        local_model_path = os.path.join(models_llm_dir, model_name_selected.replace("/", "_"))
        
        torch_dtype = self._get_torch_dtype(precision_str)
        attn_implementation = "sdpa"

        try:
            if not os.path.exists(os.path.join(local_model_path, "config.json")):
                logger.info(f"Model not found locally at {local_model_path}. Downloading from Hugging Face Hub...")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model_name_selected,
                                  local_dir=local_model_path,
                                  local_dir_use_symlinks=False)
                logger.info(f"Model {model_name_selected} downloaded to {local_model_path}")
            else:
                logger.info(f"Found model {model_name_selected} locally at {local_model_path}")

            logger.info(f"Loading tokenizer for {model_name_selected} from {local_model_path}")
            self.loaded_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            logger.info("Tokenizer loaded successfully.")

            logger.info(f"Loading model {model_name_selected} from {local_model_path} with dtype: {str(torch_dtype)} and attn_implementation: {attn_implementation}")
            self.loaded_model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                attn_implementation=attn_implementation,
                trust_remote_code=True
            )
            
            logger.info(f"Model {model_name_selected} loaded. Device map: {self.loaded_model.hf_device_map}, Model device: {self.loaded_model.device}, Effective Dtype: {self.loaded_model.dtype}")
            self.current_model_name = model_name_selected
            self.current_precision = precision_str

        except Exception as e:
            logger.error(f"Error loading model {model_name_selected}: {e}", exc_info=True)
            self._unload_model_and_tokenizer(reason="error during load")
            raise

    def process(self, model_name, system, prompt, seed, 
                max_new_tokens, enable_thinking, precision, 
                unload_after_generation):
        
        current_top_k = 20
        current_min_p = 0.0

        if enable_thinking:
            current_temperature = 0.6
            current_top_p = 0.95
            logger.info("Enable Thinking ON: Using Temperature=0.6, TopP=0.95, TopK=20, MinP=0.0")
        else:
            current_temperature = 0.7
            current_top_p = 0.8
            logger.info("Enable Thinking OFF: Using Temperature=0.7, TopP=0.8, TopK=20, MinP=0.0")

        logger.info(f"Processing request for model: {model_name}, precision: {precision}")
        logger.info(f"System Prompt (first 100 chars): {system[:100] if system else 'N/A'}...")
        logger.info(f"User Prompt (first 100 chars): {prompt[:100]}...")
        logger.info(f"Fixed generation params: seed={seed}, max_tokens={max_new_tokens}, unload_after_gen={unload_after_generation}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"Seed set to: {seed}")

        if self.loaded_model is None or self.loaded_tokenizer is None or \
           self.current_model_name != model_name or self.current_precision != precision:
            self._unload_model_and_tokenizer(reason="parameters changed or model not initially loaded")
            
            try:
                self._load_model_and_tokenizer(model_name, precision)
            except Exception as e:
                logger.error(f"Failed to load model or tokenizer: {e}", exc_info=True)
                return (f"ERROR: Could not load model - {str(e)}",)
        
        messages = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        
        messages.append({"role": "user", "content": prompt.strip() if prompt else ""})
        logger.debug(f"Formatted messages for chat template: {messages}")

        try:
            text_input_for_model = self.loaded_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )
            logger.debug(f"Text input for model (first 200 chars): {text_input_for_model[:200]}")
        except Exception as e:
            logger.error(f"Error applying chat template: {e}.", exc_info=True)
            if unload_after_generation: self._unload_model_and_tokenizer(reason="error and unload_after_generation enabled")
            return (f"ERROR: Tokenizer failed to apply chat template - {str(e)}",)

        model_inputs = self.loaded_tokenizer([text_input_for_model], return_tensors="pt").to(self.loaded_model.device)
        logger.debug(f"Model inputs tokenized and moved to device: {self.loaded_model.device}")
        
        do_sample = True
        if current_temperature < 0.01:
            if enable_thinking:
                logger.warning("Temperature is very low for thinking mode, which is discouraged as per Qwen documentation.")
            else:
                logger.warning("Temperature is very low, generation will be mostly greedy.")


        generation_config_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": current_temperature,
            "top_p": current_top_p, 
            "top_k": current_top_k,
            "min_p": current_min_p,
            "do_sample": do_sample,
            "pad_token_id": self.loaded_tokenizer.eos_token_id
        }
        if generation_config_params["pad_token_id"] is None:
            if self.loaded_tokenizer.pad_token_id is not None:
                logger.warning("eos_token_id is None, using pad_token_id for generation.")
                generation_config_params["pad_token_id"] = self.loaded_tokenizer.pad_token_id
            else:
                logger.warning("Neither eos_token_id nor pad_token_id is set. Generation might fail.")
        
        logger.info(f"Generating text with effective params: temp={generation_config_params['temperature']}, top_p={generation_config_params['top_p']}, top_k={generation_config_params['top_k']}, min_p={generation_config_params['min_p']}, max_new_tok={generation_config_params['max_new_tokens']}")
        
        final_content_str = ""
        try:
            generated_ids = self.loaded_model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                **generation_config_params
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            logger.info("Text generation completed.")

            think_end_token_id = 151668 
            logger.debug(f"Raw output token IDs (first 20): {output_ids[:20]}")

            split_index = 0 
            if enable_thinking:
                try:
                    found_idx = -1
                    for i in range(len(output_ids) - 1, -1, -1): 
                        if output_ids[i] == think_end_token_id: 
                            found_idx = i
                            break
                    if found_idx != -1:
                        split_index = found_idx + 1 
                        logger.info(f"Found think_end_token_id ({think_end_token_id}) at index {found_idx} (relative to output_ids). Split index: {split_index}.")
                        final_content_str = self.loaded_tokenizer.decode(output_ids[split_index:], skip_special_tokens=True).strip()
                    else:
                        logger.info(f"Token {think_end_token_id} (think_end_token) not found in output, though enable_thinking was True. Using full output.")
                        final_content_str = self.loaded_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                except Exception as e_parse: 
                    logger.error(f"Error parsing thinking content: {e_parse}. Using full output.", exc_info=True)
                    final_content_str = self.loaded_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            else: 
                logger.info("Thinking mode disabled. Decoding full output.")
                final_content_str = self.loaded_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        except Exception as e_gen:
            logger.error(f"Error during model.generate or decoding: {e_gen}", exc_info=True)
            if torch.cuda.is_available(): torch.cuda.empty_cache() 
            if unload_after_generation: self._unload_model_and_tokenizer(reason="error and unload_after_generation enabled")
            return (f"ERROR: Generation/Decoding failed - {str(e_gen)}",)

        # Проверяем и удаляем кавычки, если строка ими обрамлена
        if final_content_str.startswith('"') and final_content_str.endswith('"'):
            logger.debug("Removing leading and trailing double quotes from the final string.")
            final_content_str = final_content_str[1:-1]
        elif final_content_str.startswith("'") and final_content_str.endswith("'"): # Также проверим на одинарные кавычки
            logger.debug("Removing leading and trailing single quotes from the final string.")
            final_content_str = final_content_str[1:-1]

        logger.info(f"Final generated text (first 200 chars): {final_content_str[:200]}")
        
        if unload_after_generation:
            self._unload_model_and_tokenizer(reason="unload_after_generation enabled")
        
        return (final_content_str,)

NODE_CLASS_MAPPINGS = {
    "TS_Qwen3": TS_Qwen3_Node
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Qwen3": "TS Qwen3"
}