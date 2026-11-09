# TS Resolution Selector

Visual aspect-ratio picker. Choose 1:1, 4:3, 3:2, 16:9, 21:9, 3:4, 2:3, 9:16, 9:21, or a custom ratio, then pick a target megapixel budget. The initial value is **1.5 MP**. The working tree after 12.13.9 has a default slider range of **0.1–4 MP**. The published 12.13.9 slider has a **3 MP** upper limit.

**Change the maximum:** open the node's **Properties** panel and set **max_megapixels**, for example `8`. This setting is saved per node in the workflow. Lowering the maximum also lowers the current value if it exceeds the new limit. Loading a saved workflow preserves its resolution: the range expands when needed to retain that saved value.

The output is an **IMAGE**, a blank canvas with dimensions snapped to multiples of 32. Use VAE Encode to obtain a latent. If you connect an image, the node fits it onto the canvas; with `original_aspect=True` the ratio is taken from the image instead of the preset.

**Use when:** starting a generation from scratch with a fixed aspect, or normalising an arbitrary image into a latent grid.

---

[Full node reference](https://github.com/AlexYez/comfyui-timesaver/tree/master/js/docs)
