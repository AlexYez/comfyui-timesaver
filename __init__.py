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
from .ts_marian_translate_node import NODE_CLASS_MAPPINGS as marian_translate_class_map, \
                               NODE_DISPLAY_NAME_MAPPINGS as marian_translate_display_map
from .ts_deflicker_node import NODE_CLASS_MAPPINGS as deflicker_class_map, \
                               NODE_DISPLAY_NAME_MAPPINGS as deflicker_display_map
from .ts_crop_to_mask_node import NODE_CLASS_MAPPINGS as crop_to_mask_class_map, \
                               NODE_DISPLAY_NAME_MAPPINGS as crop_to_mask_display_map                                

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
    (marian_translate_class_map, marian_translate_display_map),
    (deflicker_class_map, deflicker_display_map),
    (crop_to_mask_class_map, crop_to_mask_display_map),
    
]

# Объединяем все маппинги
for class_map, display_map in all_mappings:
    NODE_CLASS_MAPPINGS.update(class_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(display_map)


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']