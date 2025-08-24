import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
import comfy.model_management
import folder_paths
import shutil
from huggingface_hub import snapshot_download

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
        *   If no style is specified, aim for a rich, visually appealing style that best suits the subject. Default towards "photorealistic," "digital painting," "cinematic still" if the context is neutral, but adapt if the subject suggests otherwise (e.g., a mythical creature might suit "fantasy concept art").
5.  **Descriptive Keywords & Enhancers:**
    *   Enrich the prompt with strong, descriptive adjectives and evocative verbs.
    *   Strategically include common image generation keywords and quality enhancers that are contextually appropriate. Examples: "masterpiece," "ultra realistic," "highly detailed," "intricate," "sharp focus," "physically-based rendering (PBR)," "Unreal Engine 5," "trending on ArtStation," "award-winning photography," "volumetric lighting," "depth of field (DOF)," "4K," "8K." Do not include these indiscriminately; they must align with the desired visual output.
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

class TS_Qwen3_Node:
    def __init__(self):
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.current_model_name = None
        self.current_precision = None
        logger.info("TS_Qwen3_Node initialized. Model will be loaded to the optimal device.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["hfmaster/Qwen3-1-7","hfmaster/Qwen3-4"],{"default":"hfmaster/Qwen3-1-7"}),
                "system": ("STRING",{"multiline":True,"default":DEFAULT_SYSTEM_PROMPT_FOR_IMAGE_PROMPT_GENERATION}),
                "prompt": ("STRING",{"multiline":True,"default":"яблоки на столе"}),
                "seed": ("INT",{"default":42,"min":0,"max":0xffffffffffffffff}),
                "max_new_tokens": ("INT",{"default":512,"min":64,"max":32768,"step":64}),
                "precision": (["fp16","bf16"],{"default":"fp16"}),
                "unload_after_generation": ("BOOLEAN",{"default":False}),
                "enable": ("BOOLEAN",{"default":True}),
                "hf_token": ("STRING",{"multiline":False,"default":""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "process"
    CATEGORY = "LLM/TS_Qwen3"

    def _get_torch_dtype(self, precision_str):
        if precision_str == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if torch.cuda.is_available() else "auto"

    def _unload_model_and_tokenizer(self, reason=""):
        if self.loaded_model is not None:
            del self.loaded_model
            self.loaded_model = None
        if self.loaded_tokenizer is not None:
            del self.loaded_tokenizer
            self.loaded_tokenizer = None
        self.current_model_name = None
        self.current_precision = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Model and tokenizer unloaded. Reason: {reason}")

    # --- ИСПРАВЛЕННАЯ ФУНКЦИЯ ПРОВЕРКИ ЦЕЛОСТНОСТИ ---
    # Эта версия быстрая, не загружает модель в память и не занимает VRAM.
    def _check_model_integrity(self, local_model_path):
        if not os.path.exists(local_model_path):
            return False
        # Простая и надежная проверка на наличие основного файла конфигурации.
        config_path = os.path.join(local_model_path, "config.json")
        return os.path.exists(config_path)

    def _load_model_and_tokenizer(self, model_name_selected, hf_token, precision_str):
        models_llm_dir = os.path.join(folder_paths.models_dir,"LLM")
        os.makedirs(models_llm_dir,exist_ok=True)
        repo_name = model_name_selected.split("/")[-1]
        local_model_path = os.path.join(models_llm_dir, repo_name)

        if not self._check_model_integrity(local_model_path):
            logger.info(f"Model {repo_name} not found or incomplete. Downloading...")
            snapshot_download(
                repo_id=model_name_selected,
                local_dir=local_model_path,
                local_dir_use_symlinks=False,
                resume_download=True,
                ignore_patterns=["*.part"],
                token=hf_token if hf_token else None
            )

        torch_dtype = self._get_torch_dtype(precision_str)

        try:
            self.loaded_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            # Загружаем модель ОДИН раз с правильным параметром device_map="auto"
            self.loaded_model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            # Опциональная попытка применить оптимизацию внимания
            if torch.cuda.is_available():
                try:
                    self.loaded_model.to_better_transformer()
                    logger.info("Enabled BetterTransformer for potential speed improvements.")
                except Exception:
                    logger.warning("BetterTransformer not available for this model, continuing without it.")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

        self.current_model_name = model_name_selected
        self.current_precision = precision_str
        logger.info(f"Model '{repo_name}' loaded successfully to device: {self.loaded_model.device}")

    def process(self, model_name, system, prompt, seed, max_new_tokens,
                precision, unload_after_generation, enable, hf_token):

        enable_thinking = False

        if not enable:
            if unload_after_generation:
                self._unload_model_and_tokenizer(reason="Processing disabled")
            return (prompt.strip() if prompt else "",)

        if (self.loaded_model is None or self.loaded_tokenizer is None or
            self.current_model_name != model_name or self.current_precision != precision):
            self._unload_model_and_tokenizer(reason="Parameters changed or model not loaded")
            self._load_model_and_tokenizer(model_name, hf_token, precision)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        messages = [{"role": "system","content":system.strip()},
                    {"role": "user","content":prompt.strip() if prompt else ""}]

        text_input_for_model = self.loaded_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )

        model_inputs = self.loaded_tokenizer([text_input_for_model], return_tensors="pt").to(self.loaded_model.device)

        generation_config_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "pad_token_id": self.loaded_tokenizer.eos_token_id
        }

        try:
            generated_ids = self.loaded_model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                **generation_config_params
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            final_content_str = self.loaded_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
            if unload_after_generation:
                self._unload_model_and_tokenizer(reason="error during generation")
            return (f"ERROR: Generation failed - {str(e)}",)

        if unload_after_generation:
            self._unload_model_and_tokenizer(reason="Unload after generation enabled")

        return (final_content_str,)

NODE_CLASS_MAPPINGS = {"TS_Qwen3": TS_Qwen3_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Qwen3": "TS Qwen3"}