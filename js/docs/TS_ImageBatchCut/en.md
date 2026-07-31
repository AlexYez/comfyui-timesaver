# TS Image Batch Cut

Trim N frames from the start (`first_cut`) and N frames from the end (`last_cut`) of an image batch. Negative values are treated as zero; an over-cut returns an empty batch.

**Use when:** trimming intro/outro frames from a video, dropping the warm-up frames of a sampler, or splitting a batch into segments.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
