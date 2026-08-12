# TS Universal Inpaint Sampler

The same idea as TS Langevin Inpaint, packaged as a **SAMPLER** you plug into ComfyUI's own `SamplerCustomAdvanced` instead of replacing the sampler node. Training-free and model-agnostic: no Fill checkpoint, no ControlNet.

Feed `SamplerCustomAdvanced` a latent whose noise mask marks the region to repaint, and hand it this sampler. `think_steps` sets how much correction happens at each noise level; `resample_strength` sets how hard the known pixels pull the masked ones towards them.

**Use when:** you already have a custom sampling chain (own sigmas, own guider) and want inpaint-aware sampling inside it rather than around it.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
