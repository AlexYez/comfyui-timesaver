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
        # РќРѕСЂРјР°Р»РёР·СѓРµРј РїСѓС‚СЊ РґР»СЏ РєРѕСЂСЂРµРєС‚РЅРѕР№ РѕР±СЂР°Р±РѕС‚РєРё РїСЂРѕР±РµР»РѕРІ Рё СЃРїРµС†РёР°Р»СЊРЅС‹С… СЃРёРјРІРѕР»РѕРІ
        folder_path = os.path.normpath(folder_path.strip())

        # РџСЂРѕРІРµСЂСЏРµРј, СЃСѓС‰РµСЃС‚РІСѓРµС‚ Р»Рё РїР°РїРєР°
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder path '{folder_path}' does not exist or is not a directory")

        # Р”РѕР±Р°РІР»СЏРµРј РїРѕРґРґРµСЂР¶РєСѓ РІРёРґРµРѕС„РѕСЂРјР°С‚РѕРІ .mp4 Рё .mov
        supported_extensions = folder_paths.supported_pt_extensions | {".mp4", ".mov"}

        # РџРѕР»СѓС‡Р°РµРј СЃРїРёСЃРѕРє РІСЃРµС… С„Р°Р№Р»РѕРІ РІ РїР°РїРєРµ СЃ РїРѕРґРґРµСЂР¶РєРѕР№ Unicode Рё СЃРїРµС†РёР°Р»СЊРЅС‹С… СЃРёРјРІРѕР»РѕРІ
        files = []
        for ext in supported_extensions:
            # РСЃРїРѕР»СЊР·СѓРµРј glob СЃ РїРѕРґРґРµСЂР¶РєРѕР№ Unicode
            pattern = os.path.join(folder_path, f"*{ext}")
            files.extend(glob.glob(pattern, recursive=False))
        files = sorted(files)  # РЎРѕСЂС‚РёСЂСѓРµРј РґР»СЏ РїСЂРµРґСЃРєР°Р·СѓРµРјРѕРіРѕ РїРѕСЂСЏРґРєР°

        if not files:
            # Р”РѕР±Р°РІР»СЏРµРј РѕС‚Р»Р°РґРѕС‡РЅСѓСЋ РёРЅС„РѕСЂРјР°С†РёСЋ
            all_files = glob.glob(os.path.join(folder_path, "*"))
            raise ValueError(
                f"No supported files found in folder '{folder_path}'. "
                f"Supported extensions: {supported_extensions}. "
                f"Files in folder: {all_files if all_files else 'No files found'}"
            )

        # РџСЂРѕРІРµСЂСЏРµРј, РЅРµ РІС‹С…РѕРґРёС‚ Р»Рё РёРЅРґРµРєСЃ Р·Р° РїСЂРµРґРµР»С‹ СЃРїРёСЃРєР° С„Р°Р№Р»РѕРІ
        if index >= len(files):
            index = index % len(files)  # Р—Р°С†РёРєР»РёРІР°РµРј РёРЅРґРµРєСЃ, РµСЃР»Рё РѕРЅ СЃР»РёС€РєРѕРј Р±РѕР»СЊС€РѕР№

        # РџРѕР»СѓС‡Р°РµРј РїРѕР»РЅС‹Р№ РїСѓС‚СЊ Рє С„Р°Р№Р»Сѓ
        file_path = os.path.normpath(files[index])

        # РР·РІР»РµРєР°РµРј РёРјСЏ С„Р°Р№Р»Р° Р±РµР· РїСѓС‚Рё Рё СЂР°СЃС€РёСЂРµРЅРёСЏ
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        return (file_path, file_name)

    @classmethod
    def IS_CHANGED(cls, folder_path: str, index: int) -> str:
        # Р¤СѓРЅРєС†РёСЏ РґР»СЏ РѕРїСЂРµРґРµР»РµРЅРёСЏ, РёР·РјРµРЅРёР»РёСЃСЊ Р»Рё РІС…РѕРґРЅС‹Рµ РґР°РЅРЅС‹Рµ
        return f"{folder_path}_{index}"

NODE_CLASS_MAPPINGS = {
    "TS_FilePathLoader": TS_FilePathLoader
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_FilePathLoader": "TS File Path Loader"
}
