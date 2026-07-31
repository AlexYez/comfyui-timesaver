"""TS Auto Tile Size — pick a tile size for a given grid configuration.

node_id: TSAutoTileSize
"""

import math

from comfy_api.v0_0_2 import IO


class TSAutoTileSize(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TSAutoTileSize",
            display_name="TS Auto Tile Size",
            category="TS/Image/Tiles",
            description="Work out the best tile width and height for an image from a tile count, honouring padding and a divisor.",
            inputs=[
                IO.Combo.Input("tile_count", options=[4, 8, 16], tooltip="Total number of tiles to split the image into. The grid is chosen to best match the image aspect ratio."),
                IO.Int.Input("padding", default=64, min=0, max=512, step=8, tooltip="Overlap in pixels added between neighbouring tiles to hide seams."),
                IO.Int.Input("divide_by", default=8, min=1, max=512, step=1, tooltip="Rounds each tile dimension to a multiple of this value (e.g. 8 for VAE-friendly sizes)."),
                IO.Image.Input("image", optional=True, tooltip="Optional image; its dimensions drive the tile size. Overrides width/height when connected."),
                IO.Int.Input("width", default=512, min=64, max=8192, step=8, optional=True, tooltip="Fallback image width, used when no image is connected."),
                IO.Int.Input("height", default=512, min=64, max=8192, step=8, optional=True, tooltip="Fallback image height, used when no image is connected."),
            ],
            outputs=[
                IO.Int.Output(display_name="tile_width"),
                IO.Int.Output(display_name="tile_height"),
            ],
        )

    @staticmethod
    def find_best_grid(total_tiles, image_aspect_ratio):
        if total_tiles <= 0:
            return 1, 1

        factors = []
        for i in range(1, int(math.sqrt(total_tiles)) + 1):
            if total_tiles % i == 0:
                factors.append((total_tiles // i, i))
                if i * i != total_tiles:
                    factors.append((i, total_tiles // i))

        best_pair = (1, total_tiles)
        min_diff = float('inf')

        for x, y in factors:
            grid_aspect_ratio = x / y
            diff = abs(grid_aspect_ratio - image_aspect_ratio)
            if diff < min_diff:
                min_diff = diff
                best_pair = (x, y)

        return best_pair

    @classmethod
    def execute(cls, tile_count, padding, divide_by, image=None, width=512, height=512) -> IO.NodeOutput:
        if image is not None:
            _, img_height, img_width, _ = image.shape
        else:
            img_width, img_height = width, height

        image_aspect_ratio = img_width / img_height if img_height != 0 else 1.0

        tiles_x, tiles_y = cls.find_best_grid(int(tile_count), image_aspect_ratio)

        tile_w = (img_width + (tiles_x - 1) * padding) / tiles_x
        tile_h = (img_height + (tiles_y - 1) * padding) / tiles_y

        # Never round down to zero: with a large divide_by and a small tile
        # (e.g. divide_by=512 on a 512x512 image split 2x2 -> tile_w=256,
        # round(0.5)=0 under banker's rounding) the node used to emit 0, and the
        # failure surfaced in whatever downstream node consumed it.
        tile_width = max(divide_by, round(tile_w / divide_by) * divide_by)
        tile_height = max(divide_by, round(tile_h / divide_by) * divide_by)

        return IO.NodeOutput(tile_width, tile_height)


NODE_CLASS_MAPPINGS = {"TSAutoTileSize": TSAutoTileSize}
NODE_DISPLAY_NAME_MAPPINGS = {"TSAutoTileSize": "TS Auto Tile Size"}
