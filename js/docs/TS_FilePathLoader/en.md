# TS File Path Loader

Picks the N-th file from a folder by sorted order. Outputs the full path and the basename without extension. Filters by ComfyUI-supported extensions (`.safetensors`, `.ckpt`, `.pt`, `.mp4`, `.mov`, …). Indices wrap around.

**Use when:** iterating over a folder of inputs in a queue, or grabbing the latest checkpoint by index.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
