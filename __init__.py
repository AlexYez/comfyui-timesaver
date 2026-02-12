import importlib
import os
import logging

# Настройка логгера
logger = logging.getLogger("TimesaverVFX_Pack")

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Получаем путь к текущей папке
current_dir = os.path.dirname(__file__)

# Получаем список всех .py файлов в этой папке (кроме __init__)
files = [f for f in os.listdir(current_dir) if f.endswith(".py") and f != "__init__.py"]

for file in files:
    module_name = file[:-3]
    
    try:
        # Динамический импорт
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Ищем словари маппингов в модуле
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            mappings = getattr(module, "NODE_CLASS_MAPPINGS")
            if isinstance(mappings, dict) and mappings:
                NODE_CLASS_MAPPINGS.update(mappings)
                
                # Ищем отображаемые имена
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS")
                    NODE_DISPLAY_NAME_MAPPINGS.update(display_names)

    except ImportError as e:
        # Специфическая обработка для GGUF ноды
        if "llama_cpp" in str(e):
             logger.warning(f"⚠️ Skipped {module_name}: llama-cpp-python not found (This is normal if you haven't installed it).")
        else:
             logger.warning(f"⚠️ Failed to import {module_name}: {e}")
             
    except Exception as e:
        logger.error(f"❌ Error loading {module_name}: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
