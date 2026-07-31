# TS Model Scanner

Inspect any `.safetensors` (from `models/diffusion_models/`) or a loaded `MODEL` and print a detailed report: every parameter's name, shape, dtype, and device, plus aggregated statistics by dtype.

**Use when:** debugging model loading, checking precision (fp16 vs fp8 vs bf16), or learning what's inside an unfamiliar checkpoint.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
