import os
import torch
import gc
from transformers import MarianMTModel, MarianTokenizer
from comfy.sd import load_checkpoint_guess_config
import folder_paths

class TS_MarianTranslator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.supported_languages = {
            "ru-en": "Helsinki-NLP/opus-mt-ru-en",  # Русский → Английский
            "en-ru": "Helsinki-NLP/opus-mt-en-ru",  # Английский → Русский
            "de-en": "Helsinki-NLP/opus-mt-de-en",  # Немецкий → Английский
            "en-de": "Helsinki-NLP/opus-mt-en-de",  # Английский → Немецкий
            "fr-en": "Helsinki-NLP/opus-mt-fr-en",  # Французский → Английский
            "en-fr": "Helsinki-NLP/opus-mt-en-fr",  # Английский → Французский
        }
        self.model_dir = os.path.join(folder_paths.models_dir, "MarianMT")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Привет, как дела?"}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Bypass"}),
                "language_pair": (list(cls().supported_languages.keys()), {"default": "ru-en"}),
                "unload_after_use": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate"
    CATEGORY = "text/translation"

    def load_model(self, model_name):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_path = os.path.join(self.model_dir, model_name.replace("/", "_"))
        if not os.path.exists(model_path):
            print(f"[TS_MarianTranslator] Downloading model {model_name}...")
            model = MarianMTModel.from_pretrained(model_name, cache_dir=self.model_dir)
            tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=self.model_dir)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
        else:
            print(f"[TS_MarianTranslator] Loading model from cache: {model_path}")
            model = MarianMTModel.from_pretrained(model_path)
            tokenizer = MarianTokenizer.from_pretrained(model_path)

        return model, tokenizer

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_name = None
        
        torch.cuda.empty_cache()
        gc.collect()
        print("[TS_MarianTranslator] Model unloaded from memory")

    def translate(self, text, enabled=True, language_pair="ru-en", unload_after_use=False):
        try:
            if not enabled:
                print("[TS_MarianTranslator] Bypass mode - text unchanged")
                return (text,)
                
            if not text.strip():
                return ("",)

            model_name = self.supported_languages[language_pair]
            if self.current_model_name != model_name:
                self.unload_model()  # Выгружаем предыдущую модель перед загрузкой новой
                self.model, self.tokenizer = self.load_model(model_name)
                self.current_model_name = model_name

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if unload_after_use:
                self.unload_model()

            return (translated_text,)
        except Exception as e:
            print(f"[TS_MarianTranslator] Error: {str(e)}")
            self.unload_model()
            return (text,)  # В случае ошибки возвращаем исходный текст

NODE_CLASS_MAPPINGS = {
    "TS_MarianTranslator": TS_MarianTranslator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_MarianTranslator": "TS MarianMT Translator",
}