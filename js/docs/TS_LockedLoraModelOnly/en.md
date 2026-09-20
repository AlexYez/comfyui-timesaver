# TS Load LoRA, model only

The `LoraLoaderModelOnly` counterpart. Model only, and not to save effort: the models this lock exists for are video models and distilled checkpoints, and such a graph carries no CLIP wire at all. Strength `0` leaves the model untouched, as the stock node does.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
