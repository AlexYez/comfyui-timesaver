# TS Langevin Inpaint

Inpaint sampler that spends extra inner steps at every noise level, so the repainted area agrees with the pixels around it instead of merely filling the hole. It replaces a plain KSampler: feed it a latent that carries a **noise mask**, and it returns the finished `LATENT`.

Why it exists: ordinary sampling looks at the masked region and its surroundings only through the model's own attention. Langevin dynamics adds a short corrective loop at each level (`think_steps` of it), pulled towards the known pixels with `guidance` and damped by `step_size`, `beta` and `friction`. The defaults are tuned for photographic content; raise `think_steps` when a seam is still visible, lower it when the patch turns mushy.

Works with any model family — no Fill checkpoint, no ControlNet.

**Use when:** a repaint has to blend into complicated surroundings (skin, fabric, foliage) and a normal sampler leaves a visible patch.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
