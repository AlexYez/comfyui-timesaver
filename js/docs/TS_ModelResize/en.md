# TS Model Resize

Brings any image to the resolution a model was trained on. Pick the model; the node reads whether the picture is horizontal, vertical or square and takes the model's matching size — a 1920×1080 frame becomes 1280×720 for Wan 720p, a phone shot 1080×1920 becomes 720×1280, a 4:3 photo becomes 1472×1104 for Qwen Image.

Two kinds of models. Some ship a **fixed list** of sizes: Wan 2.1 / 2.2 (1280×720, 832×480; 1280×704 for TI2V 5B), Qwen Image (its seven official sizes), the SDXL buckets, Z-Image Turbo, SD 1.5 — the size with the nearest aspect ratio wins. Others take **any aspect at a pixel budget on a grid**: FLUX at 1 or 2 MP in steps of 16, LTX-2 / 2.5 at 720p in steps of 32, HunyuanVideo 1.5 at 720p — there the picture keeps its own shape and only a few pixels are trimmed to the grid. The tables come from the models' own repositories (Wan's `SUPPORTED_SIZES`, Qwen-Image's `aspect_ratios`, the SDXL report); note that Qwen's 4:3 is 1472×1104, not the 1472×1140 many nodes use.

`fit` decides how the picture gets there: **crop** fills the frame and trims what sticks out (nothing is distorted; `anchor = top` keeps the faces of a tall shot), **pad** shows the whole picture with bars (`pad_color`: black, white, grey or repeated edge pixels), **stretch** distorts to the exact size. The `mask` output is 1 on the picture and 0 on the bars — wire it into an outpaint to fill them. A batch is resized frame by frame; an alpha channel goes along and is transparent on the bars.

**Use when:** feeding a start frame to Wan or LTX, an edit image to Qwen, or anything to a model that misbehaves off its native sizes.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
