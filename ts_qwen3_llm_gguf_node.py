import os
import logging
import gc
import torch
import folder_paths
import re

# Попытка импортировать llama_cpp. Если не получится, работа ноды будет невозможна.
try:
    from llama_cpp import Llama
    Llama_CPP_AVAILABLE = True
except ImportError:
    Llama_CPP_AVAILABLE = False
    print("##########")
    print("ERROR: llama-cpp-python не установлена. Нода TS Qwen GGUF не будет работать.")
    print("Пожалуйста, установите ее с поддержкой CUDA, выполнив команду в терминале вашего ComfyUI:")
    print("CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir")
    print("##########")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Получаем путь к папке с LLM моделями ComfyUI
llm_models_dir = folder_paths.get_folder_paths("llm")[0] if folder_paths.get_folder_paths("llm") else None
if llm_models_dir and not os.path.exists(llm_models_dir):
    os.makedirs(llm_models_dir, exist_ok=True)

# Функция для поиска GGUF моделей
def find_gguf_models(models_path):
    if not models_path or not os.path.exists(models_path): return ["модели не найдены"]
    try:
        models = [f for f in os.listdir(models_path) if f.endswith(".gguf")]
        return models if models else ["модели не найдены"]
    except Exception as e:
        logger.error(f"Ошибка при поиске GGUF моделей: {e}")
        return ["ошибка при поиске"]

DEFAULT_SYSTEM_PROMPT = """Translate the input prompt into English if needed. Then expand it into a single detailed English paragraph of 128–256 tokens, keeping the original meaning intact. Do not include technical output parameters such as aspect ratio, resolution, fps, print size, or viewing instructions. Only describe subject, action, setting, style, atmosphere, lighting, textures, camera perspective, mood, and other visual details.

Steps:

Read the full input and keep the meaning.

If the input is not English, translate it naturally but keep names and technical terms accurate.

Identify subject, action, setting, style/mood, and visual descriptors.

Expand each part with 1–3 short descriptive phrases that enrich the scene but do not change its meaning.

Assemble everything into one flowing paragraph in English, natural sentence style, no labels or lists.

Ensure the length is 128–256 tokens. If too short, add atmospheric detail (lighting, textures, mood). If too long, remove less important adjectives.

Output only the final paragraph, nothing else.

Default fallback details if user prompt is very short: cinematic lighting, photoreal detail, dramatic atmosphere, natural textures, depth of field, balanced color palette.
"""

class TS_Qwen_GGUF_Node:
    def __init__(self):
        self.llm = None
        self.current_model_path = None
        self.current_gpu_layers = None
        self.current_ctx = None
        if not Llama_CPP_AVAILABLE: logger.error("llama-cpp-python недоступна. Нода не будет функционировать.")

    @classmethod
    def INPUT_TYPES(cls):
        if not Llama_CPP_AVAILABLE:
            return { "required": { "error": ("STRING", { "multiline": True, "default": "Ошибка: библиотека llama-cpp-python не найдена.\nПожалуйста, установите ее по инструкции в консоли." })}}
        
        return {
            "required": {
                "model_name": (find_gguf_models(llm_models_dir), ),
                "system": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM_PROMPT}),
                "prompt": ("STRING", {"multiline": True, "default": "яблоки на столе"}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 0xffffffffffffffff}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1000}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 512}),
                "unload_after_generation": ("BOOLEAN", {"default": False}),
                "enable": ("BOOLEAN", {"default": True}),
                "enable_thinking": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "process"
    CATEGORY = "LLM/TS_Qwen"

    def _unload_llm_model(self, reason=""):
        if self.llm is not None:
            del self.llm
            self.llm = None
            self.current_model_path = None
            self.current_gpu_layers = None
            self.current_ctx = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            logger.info(f"Модель GGUF выгружена. Причина: {reason}")

    def _load_llm_model(self, model_name, n_gpu_layers, n_ctx):
        if not Llama_CPP_AVAILABLE or model_name == "модели не найдены": return
        model_path = os.path.join(llm_models_dir, model_name)
        if not os.path.exists(model_path):
            logger.error(f"Файл модели не найден по пути: {model_path}"); return
            
        logger.info(f"Загрузка модели GGUF: {model_name}...")
        logger.info(f"Параметры: n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")
        try:
            self.llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
            # Сохраняем состояние, с которым была загружена модель
            self.current_model_path = model_path
            self.current_gpu_layers = n_gpu_layers
            self.current_ctx = n_ctx
            logger.info(f"Модель '{model_name}' успешно загружена.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {model_name}: {e}", exc_info=True)
            self.llm = None; raise

    def process(self, model_name, system, prompt, seed, max_tokens, n_gpu_layers, n_ctx, unload_after_generation, enable, enable_thinking):

        if not enable:
            if unload_after_generation: self._unload_llm_model(reason="Processing disabled")
            return (prompt.strip() if prompt else "",)

        if not Llama_CPP_AVAILABLE: return ("Ошибка: llama-cpp-python не установлена.",)
        if model_name in ["модели не найдены", "ошибка при поиске"]: return (f"Ошибка: не выбрана или не найдена модель GGUF в папке {llm_models_dir}",)

        full_model_path = os.path.join(llm_models_dir, model_name)
        
        # УЛУЧШЕННАЯ ЛОГИКА ПЕРЕЗАГРУЗКИ
        # Перезагружаем, если модель не загружена, или изменился файл, или изменились параметры GPU/контекста
        if (self.llm is None or 
            self.current_model_path != full_model_path or 
            self.current_gpu_layers != n_gpu_layers or
            self.current_ctx != n_ctx):
            self._unload_llm_model(reason="Параметры изменены или модель не загружена")
            self._load_llm_model(model_name, n_gpu_layers, n_ctx)

        if self.llm is None: return ("Ошибка: модель не была загружена.",)

        user_prompt_content = prompt.strip() if prompt else ""
        
        if enable_thinking:
            final_user_prompt = f"/think\n{user_prompt_content}"
            logger.info("Режим 'Thinking' включен через команду /think.")
        else:
            final_user_prompt = f"/no_think\n{user_prompt_content}"
            logger.info("Режим 'Thinking' выключен через команду /no_think.")

        prompt_str = f"<|im_start|>system\n{system.strip()}<|im_end|>\n"
        prompt_str += f"<|im_start|>user\n{final_user_prompt}<|im_end|>\n"
        prompt_str += "<|im_start|>assistant\n"

        logger.info("Начинаю генерацию текста...")
        try:
            response = self.llm(
                prompt=prompt_str,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.8,
                seed=seed,
                stop=["<|im_end|>", "<|endoftext|>"]
            )
            raw_output = response['choices'][0]['text']

            # Пост-обработка для очистки вывода
            think_pattern = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
            cleaned_output = re.sub(think_pattern, "", raw_output)
            final_content_str = cleaned_output.strip()

            logger.info("Генерация текста успешно завершена.")
        except Exception as e:
            logger.error(f"Ошибка во время генерации: {e}", exc_info=True)
            if unload_after_generation: self._unload_llm_model(reason="Ошибка во время генерации")
            return (f"ОШИБКА: Генерация не удалась - {str(e)}",)

        if unload_after_generation: self._unload_llm_model(reason="Включена выгрузка после генерации")

        return (final_content_str,)

NODE_CLASS_MAPPINGS = {"TS_Qwen_GGUF": TS_Qwen_GGUF_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"TS_Qwen_GGUF": "TS Qwen (GGUF)"}