# TS Multi Reference

Add up to three reference images as `reference_latents` into the conditioning stream. Built for Qwen-Image-Edit and similar multi-reference pipelines. Per-slot output (`image_1` / `image_2` / `image_3`) with `ExecutionBlocker` for unconnected slots, automatic resize to a megapixel budget aligned to a divisor (default 32). Handles RGBA + MASK inputs (composites onto white).

**Use when:** running Qwen-Edit / Flux-with-references style pipelines that accept multiple reference images.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
