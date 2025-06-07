# __init__.py for TS_CustomNodes package

# Импортируем маппинги из каждого файла ноды
from .ts_downloader_node import NODE_CLASS_MAPPINGS as downloader_class_map, \
                                NODE_DISPLAY_NAME_MAPPINGS as downloader_display_map
from .ts_edl_chapters_node import NODE_CLASS_MAPPINGS as edl_chapters_class_map, \
                                  NODE_DISPLAY_NAME_MAPPINGS as edl_chapters_display_map
from .ts_equirect_to_cube_node import NODE_CLASS_MAPPINGS as equirect_to_cube_class_map, \
                                      NODE_DISPLAY_NAME_MAPPINGS as equirect_to_cube_display_map
from .ts_cube_to_equirect_node import NODE_CLASS_MAPPINGS as cube_to_equirect_class_map, \
                                      NODE_DISPLAY_NAME_MAPPINGS as cube_to_equirect_display_map
from .ts_qwen3_llm_node import NODE_CLASS_MAPPINGS as qwen3_llm_class_map, \
                               NODE_DISPLAY_NAME_MAPPINGS as qwen3_llm_display_map
from .ts_whisper_node import NODE_CLASS_MAPPINGS as whisper_class_map, \
                               NODE_DISPLAY_NAME_MAPPINGS as whisper_display_map
from .ts_video_depth_node import NODE_CLASS_MAPPINGS as video_depth_class_map, \
                               NODE_DISPLAY_NAME_MAPPINGS as video_depth_display_map
from .ts_video_upscale_node import NODE_CLASS_MAPPINGS as video_upscale_class_map, \
                               NODE_DISPLAY_NAME_MAPPINGS as video_upscale_display_map
from .ts_image_resize_node import NODE_CLASS_MAPPINGS as image_resize_class_map, \
                               NODE_DISPLAY_NAME_MAPPINGS as image_resize_display_map
from .ts_file_path_node import NODE_CLASS_MAPPINGS as file_path_class_map, \
                               NODE_DISPLAY_NAME_MAPPINGS as file_path_display_map


# Инициализируем общие словари маппингов
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Список всех импортированных маппингов для удобного объединения
all_mappings = [
    (downloader_class_map, downloader_display_map),
    (edl_chapters_class_map, edl_chapters_display_map),
    (equirect_to_cube_class_map, equirect_to_cube_display_map),
    (cube_to_equirect_class_map, cube_to_equirect_display_map),
    (qwen3_llm_class_map, qwen3_llm_display_map),
    (whisper_class_map, whisper_display_map),
    (video_depth_class_map, video_depth_display_map),
    (video_upscale_class_map, video_upscale_display_map),
    (image_resize_class_map, image_resize_display_map),
    (file_path_class_map, file_path_display_map),
    
]

# Объединяем все маппинги
for class_map, display_map in all_mappings:
    NODE_CLASS_MAPPINGS.update(class_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(display_map)

# Отладочный вывод при загрузке пакета
print("---------------------------------------------------------------------------")
print("TS_CustomNodes: Initializing...")
print("  Registered NODE_CLASS_MAPPINGS:")
for name, node_class_obj in NODE_CLASS_MAPPINGS.items():
    class_name_str = node_class_obj.__name__ if hasattr(node_class_obj, '__name__') else str(node_class_obj)
    print(f"    - \"{name}\": {class_name_str}")
print("  Registered NODE_DISPLAY_NAME_MAPPINGS:")
for name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
    print(f"    - \"{name}\": \"{display_name}\"")
print("TS_CustomNodes: Initialization complete.")
print("---------------------------------------------------------------------------")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']