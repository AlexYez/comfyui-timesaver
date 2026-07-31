# TS Image Resize

The resize node you actually want. Pick one of: exact target (`target_width` × `target_height`), one side (`smaller_side` / `larger_side`), megapixels, or a scale factor. Optional `divisible_by` snaps dimensions to a multiple required by samplers (8, 16, 32, …). `dont_enlarge` blocks upscales when the source is already smaller than the target.

**Use when:** preparing inputs for SDXL / Flux / WAN, batch-resizing photos to a maximum side, or matching a video frame size.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
