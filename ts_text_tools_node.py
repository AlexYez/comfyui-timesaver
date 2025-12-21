import re

class TS_BatchPromptLoader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "Prompt 1: cat\n\nPrompt 2: dog\n\nPrompt 3: bird",
                    "dynamicPrompts": False
                }),
            },
        }

    # Определяем типы выходов: Строка (сам промпт) и Целое число (счетчик)
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompt", "prompts_count")
    
    # OUTPUT_IS_LIST = (True, False)
    # True -> 'prompt': Это список. ComfyUI запустит генерацию N раз (по разу для каждого элемента).
    # False -> 'prompts_count': Это НЕ список, а единичное значение (константа для всех прогонов).
    OUTPUT_IS_LIST = (True, False)
    
    FUNCTION = "process_prompts"
    CATEGORY = "utils/text"

    def process_prompts(self, text):
        # 1. Нормализация переносов строк (Windows/Unix)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 2. Разделение по абзацам.
        # Регулярка r'\n\s*\n' ищет два переноса строки подряд с возможными пробелами между ними.
        # Это позволяет писать один промпт в несколько строк, а разделять их пустой строкой.
        raw_prompts = re.split(r'\n\s*\n', text)
        
        valid_prompts = []
        for p in raw_prompts:
            # Очистка пробелов по краям
            cleaned_p = p.strip()
            # Если после очистки остался текст - добавляем в список
            if cleaned_p:
                valid_prompts.append(cleaned_p)
        
        # Если вдруг список пуст (пользователь ввел только пробелы), даем пустую строку
        if not valid_prompts:
            valid_prompts = [""]
            
        # Считаем количество (размер батча)
        count = len(valid_prompts)
        
        # Возвращаем:
        # 1. valid_prompts (список строк) -> ComfyUI будет брать их по одной.
        # 2. count (число) -> ComfyUI передаст это число во все итерации как есть.
        return (valid_prompts, count)

# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "TS_BatchPromptLoader": TS_BatchPromptLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_BatchPromptLoader": "TS Batch Prompt Loader"
}