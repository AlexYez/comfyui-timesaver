# TS Video Upscale With Model

Per-frame upscaling with any spandrel-loaded model (RealESRGAN, 4x-Ultrasharp, etc.). Three device strategies: `auto`, `load_unload_each_frame` (low VRAM, slower), `keep_loaded` (faster, more VRAM), `cpu_only`.

**Use when:** upscaling video without OOM, or batching upscale jobs with a controllable VRAM footprint.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
