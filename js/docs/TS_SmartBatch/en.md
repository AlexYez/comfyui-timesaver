# TS Smart Batch

Batch images — and keep working when one of them is not there. Inputs **grow**: fill the last slot and the next one appears, up to 32. Every slot is **optional**, so an empty or bypassed one is simply skipped. Nothing connected at all is the only error, and it says so plainly instead of handing you a blank frame.

Why it exists: core's **Batch Images** has two *required* inputs, so muting or bypassing either side breaks the whole graph before it even runs. Building a first-frame / last-frame pair usually means switching one side off and on again, and that should not require rewiring.

Frames come out in slot order — `image0`, `image1`, `image2`, … — whatever gaps you leave. Pairs are reconciled exactly as core does: a missing alpha channel is padded with 1.0, and a differently sized image is resized to match the first one that actually arrived. Batches concatenate — 3 frames plus 2 frames give 5.

**Use when:** feeding an FLF (first/last frame) video model, or any time one node should emit a batch of however many sources happen to be switched on.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
