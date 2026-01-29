import importlib
import os
import logging
import traceback

# Настройка логгера
logger = logging.getLogger("TimesaverVFX_Pack")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Получаем путь к текущей папке
current_dir = os.path.dirname(__file__)

# Получаем список всех .py файлов в этой папке
files = [f for f in os.listdir(current_dir) if f.endswith(".py") and f != "__init__.py"]

# Проходим по каждому файлу
for file in files:
    # Имя модуля без .py (например, "ts_qwen3_vl_node")
    module_name = file[:-3]
    
    try:
        # Динамически импортируем модуль
        # "package=__name__" означает, что мы ищем внутри текущего пакета (относительный импорт)
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Проверяем, есть ли в модуле нужные словари для ComfyUI
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            # Если словарь есть и он не пустой - добавляем его в общий список
            mappings = getattr(module, "NODE_CLASS_MAPPINGS")
            if mappings:
                NODE_CLASS_MAPPINGS.update(mappings)
                
                # Также ищем отображаемые имена
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS")
                    NODE_DISPLAY_NAME_MAPPINGS.update(display_names)
                
                # logger.info(f"Loaded: {module_name}")

    except ImportError as e:
        # Специфическая обработка для опциональных нод (например, GGUF)
        # Если библиотеки нет, мы просто пропускаем файл без паники
        error_msg = str(e)
        if "llama_cpp" in error_msg:
             logger.warning(f"⚠️ Skipped {module_name}: llama-cpp-python not found.")
        else:
             logger.warning(f"⚠️ Failed to import {module_name}: {e}")
             
    except Exception as e:
        # Если произошла другая ошибка (синтаксис и т.д.)
        logger.error(f"❌ Error loading {module_name}: {e}")
        # traceback.print_exc() # Раскомментируйте для отладки

# Экспорт для ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]