# TS Matting (ViTMatte)

Guided alpha matting via Hugging Face ViTMatte. Takes an image + a coarse mask (e.g. from SAM3 Detect), auto-builds a trimap and refines into a photo-realistic alpha matte. Same `mask_blur`/`mask_offset`/`background` post-processing contract as TS Remove Background, so it's a drop-in upgrade when edges/hair/transparency matter. Models cached under `models/vitmatte/`.

**Use when:** producing crisp cut-outs from SAM-style masks without dropping into Photoshop.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
