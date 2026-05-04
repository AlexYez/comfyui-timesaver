"""TS Auto Tile Size — pick a tile size for a given grid configuration.

node_id: TSAutoTileSize
"""

import math

import torch


class TSAutoTileSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Р’С‹РїР°РґР°СЋС‰РёР№ СЃРїРёСЃРѕРє РґР»СЏ РІС‹Р±РѕСЂР° РѕР±С‰РµРіРѕ РєРѕР»РёС‡РµСЃС‚РІР° С‚Р°Р№Р»РѕРІ
                "tile_count": ([4, 8, 16],),
                "padding": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "divide_by": ("INT", {"default": 8, "min": 1, "max": 512, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("tile_width", "tile_height")

    FUNCTION = "calculate_grid"

    CATEGORY = "utils/Tile Size"

    def find_best_grid(self, total_tiles, image_aspect_ratio):
        """
        РќР°С…РѕРґРёС‚ РЅР°РёР»СѓС‡С€СѓСЋ РїР°СЂСѓ РјРЅРѕР¶РёС‚РµР»РµР№ (СЃРµС‚РєСѓ),
        СЃРѕРѕС‚РЅРѕС€РµРЅРёРµ СЃС‚РѕСЂРѕРЅ РєРѕС‚РѕСЂРѕР№ РЅР°РёР±РѕР»РµРµ Р±Р»РёР·РєРѕ Рє СЃРѕРѕС‚РЅРѕС€РµРЅРёСЋ СЃС‚РѕСЂРѕРЅ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ.
        """
        if total_tiles <= 0:
            return 1, 1

        factors = []
        # РќР°С…РѕРґРёРј РІСЃРµ РїР°СЂС‹ РјРЅРѕР¶РёС‚РµР»РµР№ РґР»СЏ С‡РёСЃР»Р° total_tiles
        for i in range(1, int(math.sqrt(total_tiles)) + 1):
            if total_tiles % i == 0:
                factors.append((total_tiles // i, i))
                if i * i != total_tiles:
                    factors.append((i, total_tiles // i))

        best_pair = (1, total_tiles) # Р—РЅР°С‡РµРЅРёРµ РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ
        min_diff = float('inf')

        # РС‰РµРј РїР°СЂСѓ СЃ РЅР°РёРјРµРЅСЊС€РµР№ СЂР°Р·РЅРёС†РµР№ РІ СЃРѕРѕС‚РЅРѕС€РµРЅРёРё СЃС‚РѕСЂРѕРЅ
        for x, y in factors:
            grid_aspect_ratio = x / y
            diff = abs(grid_aspect_ratio - image_aspect_ratio)
            if diff < min_diff:
                min_diff = diff
                best_pair = (x, y)
        
        return best_pair

    def calculate_grid(self, tile_count, padding, divide_by, image=None, width=512, height=512):
        # РЁР°Рі 1: РћРїСЂРµРґРµР»СЏРµРј РёСЃС…РѕРґРЅС‹Рµ СЂР°Р·РјРµСЂС‹ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ
        if image is not None:
            _, img_height, img_width, _ = image.shape
        else:
            img_width, img_height = width, height

        # РЁР°Рі 2: РќР°С…РѕРґРёРј РЅР°РёР»СѓС‡С€СѓСЋ СЃРµС‚РєСѓ (tiles_x, tiles_y)
        # Р’С‹С‡РёСЃР»СЏРµРј СЃРѕРѕС‚РЅРѕС€РµРЅРёРµ СЃС‚РѕСЂРѕРЅ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ, РёР·Р±РµРіР°СЏ РґРµР»РµРЅРёСЏ РЅР° РЅРѕР»СЊ.
        image_aspect_ratio = img_width / img_height if img_height != 0 else 1.0
        
        # РСЃРїРѕР»СЊР·СѓРµРј РІСЃРїРѕРјРѕРіР°С‚РµР»СЊРЅСѓСЋ С„СѓРЅРєС†РёСЋ РґР»СЏ РїРѕРґР±РѕСЂР° РѕРїС‚РёРјР°Р»СЊРЅРѕР№ СЃРµС‚РєРё
        tiles_x, tiles_y = self.find_best_grid(tile_count, image_aspect_ratio)

        # РЁР°Рі 3: Р Р°СЃСЃС‡РёС‚С‹РІР°РµРј СЂР°Р·РјРµСЂ РѕРґРЅРѕРіРѕ С‚Р°Р№Р»Р° СЃ СѓС‡РµС‚РѕРј РїРµСЂРµРєСЂС‹С‚РёСЏ
        tile_w = (img_width + (tiles_x - 1) * padding) / tiles_x
        tile_h = (img_height + (tiles_y - 1) * padding) / tiles_y

        # РЁР°Рі 4: РћРєСЂСѓРіР»СЏРµРј РґРѕ Р±Р»РёР¶Р°Р№С€РµРіРѕ С‡РёСЃР»Р°, РєСЂР°С‚РЅРѕРіРѕ `divide_by`
        tile_width = round(tile_w / divide_by) * divide_by
        tile_height = round(tile_h / divide_by) * divide_by

        return (tile_width, tile_height)




NODE_CLASS_MAPPINGS = {"TSAutoTileSize": TSAutoTileSize}
NODE_DISPLAY_NAME_MAPPINGS = {"TSAutoTileSize": "TS Auto Tile Size"}
