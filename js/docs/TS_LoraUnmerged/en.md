# TS LoRA Unmerged

**A LoRA applied as a side branch, `y = W·x + B·A·x`, instead of being merged into the weights.**

ComfyUI's LoRA loaders merge: `W + strength · B·A` is written into the weight once. That is free at run time and exact in float32 — but a model rarely sits in float32. On **bf16** weights round-to-nearest throws away a large part of a small update; on **int8** weights the requantization after the merge adds noise several times the update's size. A distilled turbo LoRA feels it most, because each of its six steps has to land. Viggle's comparison against the diffusers reference: LPIPS **0.093 merged vs 0.052 unmerged on bf16, 0.086 vs 0.038 on int8**.

This node leaves the weights alone and adds `B·(A·x)` to the output of every targeted layer on the fly, in the layer's own dtype — what diffusers/PEFT do without `fuse_lora()`. Measured on Qwen Image 2.1 int8 with the r128 turbo LoRA (RTX 3080 Ti Laptop, 1024×1024): **2.8 s per step without a LoRA, 3.45 s with this branch**, and the result is **bitwise identical** to Viggle's own node. A stock merge of the same LoRA on the same model is off from it by about half the LoRA's effect.

⚠️ **Qwen Image 2.1 fuses its MLP.** In ComfyUI `gate_layer` and `proj` are one `gate_up` layer, while LoRAs address the two halves separately; which half goes where is taken from ComfyUI's own key map, not guessed. And under int8 the down projection runs inside a fused kernel that skips its module entirely — so its branch is added one level up, to the output of the whole MLP, from the same numbers.

The LoRA's own tensors sit in VRAM only while sampling runs (about 0.7 GB for the r128 turbo LoRA) and are dropped afterwards. They are put there **before** the model is loaded, after asking ComfyUI to free that much, so its memory manager plans around them. They are always kept in the dtype the model computes in — on an fp8 model that is bf16, not fp8, which would zero out a third of a small LoRA. A LoRA whose layers the model reaches some other way than through its modules is reported in the log rather than silently skipped. Naming understood: diffusers/PEFT (`lora_A`/`lora_B`, alpha from the file's metadata) and kohya (`lora_down`/`lora_up` + `alpha`). A DoRA is refused with a message (a side branch cannot renormalise it); LoKr, LoHa, conv LoRAs and full-weight `.diff` entries are not applied, and the log says which.

**When to use it:** a turbo or otherwise small LoRA on a bf16 or int8 model where the stock merge visibly loses it. For ordinary style LoRAs the stock loader (or TS LoRA Loader) is free and good enough.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
