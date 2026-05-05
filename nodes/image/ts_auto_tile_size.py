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
                # Выпадающий список для выбора общего количества тайлов
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

    CATEGORY = "TS/Image"

    def find_best_grid(self, total_tiles, image_aspect_ratio):
        """
        Находит наилучшую пару множителей (сетку),
        соотношение сторон которой наиболее близко к соотношению сторон изображения.
        """
        if total_tiles <= 0:
            return 1, 1

        factors = []
        # Находим все пары множителей для числа total_tiles
        for i in range(1, int(math.sqrt(total_tiles)) + 1):
            if total_tiles % i == 0:
                factors.append((total_tiles // i, i))
                if i * i != total_tiles:
                    factors.append((i, total_tiles // i))

        best_pair = (1, total_tiles) # Значение по умолчанию
        min_diff = float('inf')

        # Ищем пару с наименьшей разницей в соотношении сторон
        for x, y in factors:
            grid_aspect_ratio = x / y
            diff = abs(grid_aspect_ratio - image_aspect_ratio)
            if diff < min_diff:
                min_diff = diff
                best_pair = (x, y)
        
        return best_pair

    def calculate_grid(self, tile_count, padding, divide_by, image=None, width=512, height=512):
        # Шаг 1: Определяем исходные размеры изображения
        if image is not None:
            _, img_height, img_width, _ = image.shape
        else:
            img_width, img_height = width, height

        # Шаг 2: Находим наилучшую сетку (tiles_x, tiles_y)
        # Вычисляем соотношение сторон изображения, избегая деления на ноль.
        image_aspect_ratio = img_width / img_height if img_height != 0 else 1.0
        
        # Используем вспомогательную функцию для подбора оптимальной сетки
        tiles_x, tiles_y = self.find_best_grid(tile_count, image_aspect_ratio)

        # Шаг 3: Рассчитываем размер одного тайла с учетом перекрытия
        tile_w = (img_width + (tiles_x - 1) * padding) / tiles_x
        tile_h = (img_height + (tiles_y - 1) * padding) / tiles_y

        # Шаг 4: Округляем до ближайшего числа, кратного `divide_by`
        tile_width = round(tile_w / divide_by) * divide_by
        tile_height = round(tile_h / divide_by) * divide_by

        return (tile_width, tile_height)




NODE_CLASS_MAPPINGS = {"TSAutoTileSize": TSAutoTileSize}
NODE_DISPLAY_NAME_MAPPINGS = {"TSAutoTileSize": "TS Auto Tile Size"}
