# TS LoRA Loader

A stack of model-only LoRAs in one node. The plus button opens a search box over the LoRAs this install actually has; a chosen one drops in as a row with its own strength field, and the plus stays where it is for the next one. Rows are reordered by dragging the grip — order matters, because LoRAs are applied one after another. Clicking a row's name sets it aside without deleting it, so an A/B comparison does not cost you the entry.

**Strength may be negative** (down to −10): that is how you damp a LoRA baked into the checkpoint, or run one in reverse. Dragging left and right over the strength field scrubs the value.

The node does not load anything itself — it expands into a chain of **native `LoraLoaderModelOnly` nodes**. Two consequences, and they are the whole point: the result is identical to a hand-built chain, and ComfyUI caches each link separately, so changing the last LoRA's strength does not recompute the ones before it. A LoRA missing on this machine costs its own row and not the run, which matters for workflows that arrive from someone else.

Model only, no CLIP — modern families keep the text encoder separate, and most LoRAs in circulation are model-side anyway.

**Use when:** more than one LoRA, or any time you expect to be reordering them.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
