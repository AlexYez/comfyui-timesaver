# TS Lama Cleanup

Built-in inpainting node powered by LaMa — paint a mask right on the node's canvas (brush + undo/redo + reset), then run to fill. Stores intermediate edits per session, no external Photoshop trip required. Since v9.3 the architecture is pure PyTorch (no upstream `lama-cleaner` dependency) and weights load from `.safetensors` in `models/lama/` instead of pickled `.ckpt`.

**Use when:** removing tourists from photos, erasing watermarks, fixing artifacts, prototyping cleanup before a heavier inpainter.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
