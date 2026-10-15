# TS Edit Align

Puts an edited image back exactly over its source. Edit models redraw the whole frame and the result drifts: measured on Qwen Image 2.1, a 960×720 restoration came back 2.8 px to the side and 0.5% wider — 5 px at the frame edge; a tall 1100×1980 one was off by 7 px. Even with the latent sized like the reference, as the core node advises, the model kept a 0.3% zoom and a 2.7 px offset. That is the model's own error, so it is measured and undone after the fact.

Put it right after VAE Decode (and TS VAE Fix), and wire the original into `source`. The node brings the edit to the source's exact size, compares the two on a grid of tiles, ignores the tiles you edited (they disagree with the rest), fits shift and scale per axis on the ones that did not change and undoes them with one sub-pixel warp. On the measured edits the error in untouched textured areas dropped by two thirds; what remains is the model re-rendering detail, not displacement. The status line says what was corrected. Without a `source` the image passes through, so one graph can still generate from scratch.

**`output_size`.** `source` (default) returns the edit at the original's size, ready to lie on it pixel for pixel. `edit` keeps the size the model rendered at and stretches the original to it for the measurement instead — a 960×720 photo restored at 1408×1056 stays 1408×1056 with nothing squeezed away, aligned to the original enlarged to that size. Use `edit` when an upscaler (TS DLSS Upscaler, a model upscaler) comes next; use `source` when the edit is blended back onto the original.

**Use when:** restoring photos, removing objects or any edit you want to overlay, compare with a slider or blend back onto the original — Qwen Image Edit, Qwen Image 2.1, Flux Kontext.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
