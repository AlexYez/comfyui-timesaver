# TS Image Tile Merger

The other half: takes the processed tile batch and the `TILE_INFO` and stitches them back into one image with proper feathered blending in the overlap regions.

**Use the pair when:** running tile-based upscaling, denoising, or any process that doesn't fit a 4K frame in VRAM.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
