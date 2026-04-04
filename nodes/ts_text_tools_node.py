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

    # РћРїСЂРµРґРµР»СЏРµРј С‚РёРїС‹ РІС‹С…РѕРґРѕРІ: РЎС‚СЂРѕРєР° (СЃР°Рј РїСЂРѕРјРїС‚) Рё Р¦РµР»РѕРµ С‡РёСЃР»Рѕ (СЃС‡РµС‚С‡РёРє)
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompt", "prompts_count")
    
    # OUTPUT_IS_LIST = (True, False)
    # True -> 'prompt': Р­С‚Рѕ СЃРїРёСЃРѕРє. ComfyUI Р·Р°РїСѓСЃС‚РёС‚ РіРµРЅРµСЂР°С†РёСЋ N СЂР°Р· (РїРѕ СЂР°Р·Сѓ РґР»СЏ РєР°Р¶РґРѕРіРѕ СЌР»РµРјРµРЅС‚Р°).
    # False -> 'prompts_count': Р­С‚Рѕ РќР• СЃРїРёСЃРѕРє, Р° РµРґРёРЅРёС‡РЅРѕРµ Р·РЅР°С‡РµРЅРёРµ (РєРѕРЅСЃС‚Р°РЅС‚Р° РґР»СЏ РІСЃРµС… РїСЂРѕРіРѕРЅРѕРІ).
    OUTPUT_IS_LIST = (True, False)
    
    FUNCTION = "process_prompts"
    CATEGORY = "utils/text"

    def process_prompts(self, text):
        # 1. РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ РїРµСЂРµРЅРѕСЃРѕРІ СЃС‚СЂРѕРє (Windows/Unix)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 2. Р Р°Р·РґРµР»РµРЅРёРµ РїРѕ Р°Р±Р·Р°С†Р°Рј.
        # Р РµРіСѓР»СЏСЂРєР° r'\n\s*\n' РёС‰РµС‚ РґРІР° РїРµСЂРµРЅРѕСЃР° СЃС‚СЂРѕРєРё РїРѕРґСЂСЏРґ СЃ РІРѕР·РјРѕР¶РЅС‹РјРё РїСЂРѕР±РµР»Р°РјРё РјРµР¶РґСѓ РЅРёРјРё.
        # Р­С‚Рѕ РїРѕР·РІРѕР»СЏРµС‚ РїРёСЃР°С‚СЊ РѕРґРёРЅ РїСЂРѕРјРїС‚ РІ РЅРµСЃРєРѕР»СЊРєРѕ СЃС‚СЂРѕРє, Р° СЂР°Р·РґРµР»СЏС‚СЊ РёС… РїСѓСЃС‚РѕР№ СЃС‚СЂРѕРєРѕР№.
        raw_prompts = re.split(r'\n\s*\n', text)
        
        valid_prompts = []
        for p in raw_prompts:
            # РћС‡РёСЃС‚РєР° РїСЂРѕР±РµР»РѕРІ РїРѕ РєСЂР°СЏРј
            cleaned_p = p.strip()
            # Р•СЃР»Рё РїРѕСЃР»Рµ РѕС‡РёСЃС‚РєРё РѕСЃС‚Р°Р»СЃСЏ С‚РµРєСЃС‚ - РґРѕР±Р°РІР»СЏРµРј РІ СЃРїРёСЃРѕРє
            if cleaned_p:
                valid_prompts.append(cleaned_p)
        
        # Р•СЃР»Рё РІРґСЂСѓРі СЃРїРёСЃРѕРє РїСѓСЃС‚ (РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РІРІРµР» С‚РѕР»СЊРєРѕ РїСЂРѕР±РµР»С‹), РґР°РµРј РїСѓСЃС‚СѓСЋ СЃС‚СЂРѕРєСѓ
        if not valid_prompts:
            valid_prompts = [""]
            
        # РЎС‡РёС‚Р°РµРј РєРѕР»РёС‡РµСЃС‚РІРѕ (СЂР°Р·РјРµСЂ Р±Р°С‚С‡Р°)
        count = len(valid_prompts)
        
        # Р’РѕР·РІСЂР°С‰Р°РµРј:
        # 1. valid_prompts (СЃРїРёСЃРѕРє СЃС‚СЂРѕРє) -> ComfyUI Р±СѓРґРµС‚ Р±СЂР°С‚СЊ РёС… РїРѕ РѕРґРЅРѕР№.
        # 2. count (С‡РёСЃР»Рѕ) -> ComfyUI РїРµСЂРµРґР°СЃС‚ СЌС‚Рѕ С‡РёСЃР»Рѕ РІРѕ РІСЃРµ РёС‚РµСЂР°С†РёРё РєР°Рє РµСЃС‚СЊ.
        return (valid_prompts, count)

# Р РµРіРёСЃС‚СЂР°С†РёСЏ РЅРѕРґС‹
NODE_CLASS_MAPPINGS = {
    "TS_BatchPromptLoader": TS_BatchPromptLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_BatchPromptLoader": "TS Batch Prompt Loader"
}
