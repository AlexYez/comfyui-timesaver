# TS Shifted Sigmas

**The schedule a few-step turbo LoRA for Qwen Image 2.1 was distilled on** — Viggle's [viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) is the one it was written for.

Such a LoRA is trained on a handful of raw timesteps — by default `1.0, 0.9375, 0.875, 0.75, 0.5, 0.25`, six steps. The diffusers pipeline does not sample at those numbers directly: it bends them with a shift that grows with the size of the picture:

```text
tokens = (height / 16) · (width / 16)
mu     = 0.5 + (0.9 − 0.5) · (tokens − 256) / (8192 − 256)
sigma  = e^mu / (e^mu + (1/t − 1))            then a final 0
```

None of ComfyUI's schedulers produce this — they either space the steps themselves or apply a fixed shift that ignores the resolution — and a turbo LoRA sampled on a different schedule loses sharpness. The node gives the schedule number for number: checked against diffusers' own `FlowMatchEulerDiscreteScheduler` and against Viggle's node on 28 sizes, with zero difference.

The size is read from the latent **the way the sampler will see it**: an empty latent from the stock *Empty Latent Image* (grid /8) is resized by the sampler to Qwen's /16, so it is counted on /16; a latent with content is counted at the size it has.

Wire it into `SamplerCustom` / `SamplerCustomAdvanced` with the `euler` sampler and no CFG (`BasicGuider`, or `cfg 1`). Add or drop steps **at the noisy end only** and keep `0.875, 0.75, 0.5, 0.25`: five steps are `1.0, 0.875, 0.75, 0.5, 0.25`, seven are `1.0, 0.9583, 0.9167, 0.875, 0.75, 0.5, 0.25`.

**When to use it:** any few-step LoRA for Qwen Image 2.1 distilled on the diffusers schedule — together with TS LoRA Unmerged below.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
