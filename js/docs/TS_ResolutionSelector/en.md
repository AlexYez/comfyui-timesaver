# TS Resolution Selector

Visual aspect-ratio picker. Choose 1:1, 4:3, 3:2, 16:9, 21:9, 3:4, 2:3, 9:16, 9:21, or a custom ratio, then pick a target megapixel budget (0.5 – 4 MP). The output is a blank canvas with dimensions snapped to multiples of 32 — perfect as a `latent_image` source. If you connect an image, the node fits it onto the canvas; with `original_aspect=True` the ratio is taken from the image instead of the preset.

**Use when:** starting a generation from scratch with a fixed aspect, or normalising an arbitrary image into a latent grid.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
