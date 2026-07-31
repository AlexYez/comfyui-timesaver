# TS Smart Inpaint

Mask-driven regenerate **or** refine in one node: feed the full image + a painted mask and it crops the region (+ context padding), upscales the crop to a megapixel budget, VAE-encodes, samples, then feather-composites and latent-blends the result back so untouched pixels stay bit-exact. The `replace` toggle picks the mode — **Replace** = Smart Inpaint, regenerating the masked area from scratch as a Kontext edit (the crop becomes `reference_latents`, denoise locked to 1.0); an optional `reference` image is chained as a second reference ("fill the hole with THIS"). **Refine** = an ADetailer-style partial-denoise pass at the `denoise` value, no reference. Headless port of ComfyUI-Angelo's "Xtra-Fine" inpaint path (MIT) — the crop + composite happen in-node, so the workflow only feeds the source + mask.

**Use when:** object replacement, generative fill, or a high-detail pass on a painted selection — without wiring up a manual crop → sampler → stitch chain yourself.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
