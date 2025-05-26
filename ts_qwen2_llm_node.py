# Standard library imports
import os
import logging # Добавлен импорт logging

# Third-party imports
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Local imports
import folder_paths

# Настройка логгера для этого файла, если нужно специфичное имя
logger = logging.getLogger(__name__) # Используем стандартный логгер Python

class TS_Qwen2Node: # Имя класса изменено
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_loaded_model_id = None # Для отслеживания загруженной модели
        self.current_quantization = None    # Для отслеживания квантования
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TS_Qwen2Node initialized. Device: {self.device}")


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "prompt": ("STRING", {"default": "Hello, Qwen2!", "multiline": True}),
                "model_name_qwen2": (
                    [   # Примерный список, замените на актуальные имена моделей Qwen2 с Hugging Face
                        "Qwen/Qwen2-0.5B-Instruct",
                        "Qwen/Qwen2-1.5B-Instruct",
                        "Qwen/Qwen2-7B-Instruct",
                        "Qwen/Qwen2-14B-Instruct", # Если существует
                        "Qwen/Qwen2-72B-Instruct"  # Если существует
                    ],
                    {"default": "Qwen/Qwen2-1.5B-Instruct"},
                ),
                "quantization": (["none", "4bit", "8bit"], {"default": "none"}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "LLM/TS_Qwen"

    def _unload_model(self, reason=""):
        if self.model is not None:
            logger.info(f"TS_Qwen2Node: Unloading model ({self.current_loaded_model_id}) {reason}")
            del self.model
            self.model = None
        if self.tokenizer is not None:
            logger.info(f"TS_Qwen2Node: Unloading tokenizer ({self.current_loaded_model_id}) {reason}")
            del self.tokenizer
            self.tokenizer = None
        self.current_loaded_model_id = None
        self.current_quantization = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("TS_Qwen2Node: CUDA cache cleared.")

    def inference(
        self, system, prompt, model_name_qwen2, quantization,
        keep_model_loaded, temperature, max_new_tokens, seed,
    ):
        if not prompt.strip(): return ("Error: Prompt input is empty.",)

        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        
        model_id = model_name_qwen2
        llm_models_dir = os.path.join(folder_paths.models_dir, "LLM")
        os.makedirs(llm_models_dir, exist_ok=True)
        target_model_path = os.path.join(llm_models_dir, model_id.replace("/", "_"))

        load_new = False
        if self.model is None or \
           self.current_loaded_model_id != model_id or \
           self.current_quantization != quantization:
            self._unload_model(reason="due to changed parameters or initial load")
            load_new = True
        
        if not keep_model_loaded and self.model is not None:
            self._unload_model(reason="as keep_model_loaded is False")
            load_new = True # Нужно перезагрузить для этого запуска

        if load_new:
            logger.info(f"TS_Qwen2Node: Loading model: {model_id} with quantization: {quantization}")
            if not os.path.exists(os.path.join(target_model_path, "config.json")):
                logger.info(f"TS_Qwen2Node: Model not found locally at {target_model_path}. Downloading...")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model_id, local_dir=target_model_path, local_dir_use_symlinks=False)
                logger.info(f"TS_Qwen2Node: Model downloaded to {target_model_path}")
            else:
                logger.info(f"TS_Qwen2Node: Found model locally at {target_model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
            
            quant_config_bnb = None
            torch_dtype_load = torch.float16 # Default for non-quantized or if bf16 not supported
            if self.device.type == 'cuda' and torch.cuda.is_bf16_supported():
                torch_dtype_load = torch.bfloat16

            if quantization == "4bit":
                quant_config_bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype_load, # Use bf16 if supported, else fp16
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info(f"TS_Qwen2Node: Using 4-bit quantization with compute dtype {torch_dtype_load}.")
            elif quantization == "8bit":
                quant_config_bnb = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("TS_Qwen2Node: Using 8-bit quantization.")
            else:
                 logger.info(f"TS_Qwen2Node: No quantization. Loading with dtype: {torch_dtype_load}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                target_model_path,
                torch_dtype=torch_dtype_load if quant_config_bnb is None else None, # dtype is handled by BNB if quantizing
                device_map="auto",
                quantization_config=quant_config_bnb,
                trust_remote_code=True
            )
            self.current_loaded_model_id = model_id
            self.current_quantization = quantization
            logger.info(f"TS_Qwen2Node: Model loaded on device(s): {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else self.model.device}, Effective dtype: {self.model.dtype}")

        # Inference
        result_text = "Error during inference." # Default error
        try:
            with torch.no_grad():
                messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)

                generation_kwargs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self.tokenizer.eos_token_id, # Crucial for stopping criteria
                }
                if temperature > 0.0:
                    generation_kwargs["temperature"] = temperature
                    generation_kwargs["do_sample"] = True
                else: # Greedy decoding
                    generation_kwargs["do_sample"] = False
                
                generated_ids = self.model.generate(**generation_kwargs)
                
                generated_ids_cpu = generated_ids.cpu()
                inputs_input_ids_cpu = inputs.input_ids.cpu()
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_input_ids_cpu, generated_ids_cpu)
                ]
                
                result_list = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True) # clean_up_tokenization_spaces=True for Qwen2 is generally fine
                result_text = result_list[0] if result_list else "Empty response from model."
        except Exception as e:
            logger.error(f"TS_Qwen2Node: Error during inference: {e}", exc_info=True)
            result_text = f"Error: {str(e)}"


        if not keep_model_loaded:
            self._unload_model(reason="after inference as keep_model_loaded is False")
        
        logger.info(f"TS_Qwen2Node: Generated text (first 100 chars): {result_text[:100]}")
        return (result_text,)

NODE_CLASS_MAPPINGS = {
    "TS Qwen2.5": TS_Qwen2Node # Ключ оставлен оригинальным ("TS Qwen2.5")
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS Qwen2.5": "TS Qwen2 LLM" # Отображаемое имя можно сделать более общим
}