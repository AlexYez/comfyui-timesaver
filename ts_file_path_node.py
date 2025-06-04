import os
import glob
from typing import List, Dict, Any
import folder_paths

class TS_FilePathLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "file_name")
    FUNCTION = "get_file_path"
    CATEGORY = "file_utils"

    def get_file_path(self, folder_path: str, index: int) -> tuple[str, str]:
        # Нормализуем путь для корректной обработки пробелов и специальных символов
        folder_path = os.path.normpath(folder_path.strip())

        # Проверяем, существует ли папка
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder path '{folder_path}' does not exist or is not a directory")

        # Добавляем поддержку видеоформатов .mp4 и .mov
        supported_extensions = folder_paths.supported_pt_extensions | {".mp4", ".mov"}

        # Получаем список всех файлов в папке с поддержкой Unicode и специальных символов
        files = []
        for ext in supported_extensions:
            # Используем glob с поддержкой Unicode
            pattern = os.path.join(folder_path, f"*{ext}")
            files.extend(glob.glob(pattern, recursive=False))
        files = sorted(files)  # Сортируем для предсказуемого порядка

        if not files:
            # Добавляем отладочную информацию
            all_files = glob.glob(os.path.join(folder_path, "*"))
            raise ValueError(
                f"No supported files found in folder '{folder_path}'. "
                f"Supported extensions: {supported_extensions}. "
                f"Files in folder: {all_files if all_files else 'No files found'}"
            )

        # Проверяем, не выходит ли индекс за пределы списка файлов
        if index >= len(files):
            index = index % len(files)  # Зацикливаем индекс, если он слишком большой

        # Получаем полный путь к файлу
        file_path = os.path.normpath(files[index])

        # Извлекаем имя файла без пути и расширения
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        return (file_path, file_name)

    @classmethod
    def IS_CHANGED(cls, folder_path: str, index: int) -> str:
        # Функция для определения, изменились ли входные данные
        return f"{folder_path}_{index}"

NODE_CLASS_MAPPINGS = {
    "TS_FilePathLoader": TS_FilePathLoader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_FilePathLoader": "TS File Path Loader"
}