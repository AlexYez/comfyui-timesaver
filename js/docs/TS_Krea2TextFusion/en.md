# TS Krea 2 Text Fusion

One dial on **how loudly the prompt reaches Krea 2**.

Krea 2 does not condition on a single text embedding. Its encoder is Qwen3-VL-4B, and the model keeps **twelve of its hidden layers at once** (taps `hidden_states[2, 5, 8, … 35]`). They are collapsed into the one embedding the DiT attends to by a single linear layer inside the model — `Linear(12 → 1)`, no bias. Its weight is therefore not a matrix but **twelve numbers**, one per tap: a learned weighted sum. In the owner's checkpoints `txtfusion.projector.weight` is exactly that — `[1, 12]`.

This node multiplies those twelve numbers, and with them the whole fused text signal. Higher follows the prompt more closely, lower leaves the model more room. `1.0` is the model as trained, `0.0` silences the text path entirely, and negative values invert it. The author of the original tweak settles on **3.0**: better prompt adherence with the invention intact.

> The idea is not ours: [Extraltodeus/ComfyUI-Krea2-attention-tweak](https://github.com/Extraltodeus/ComfyUI-Krea2-attention-tweak). He spotted it in [Beinsezii/Krea-2-Turbo-Projector-Scale-LoRA](https://huggingface.co/Beinsezii/Krea-2-Turbo-Projector-Scale-LoRA-Diffusers) — the values there were too close to the originals to be anything but a multiplier.

⚠️ **The model is not damaged.** The multiply lands on a copy of the weight (the core casts with `copy=True` before applying patches), so the checkpoint on disk and in memory is untouched and the patch is undone with the model. Feed it anything other than Krea 2 and it refuses in plain words instead of failing somewhere inside someone else's code.

**When to use it:** the prompt is not being followed closely enough — raise it; the picture looks over-literal and dried out — lower it.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
