# TS SAM Media Loader

Loads an image or video and lets you click-pick positive/negative points right on a first-frame preview. Outputs `IMAGE`, `AUDIO` (for video), and `positive_coords`/`negative_coords` STRING JSON in the exact format expected by the native ComfyUI **SAM3 Detect** / **SAM3 Video Track** nodes. With an optional SAM3 `model` input it also returns the rendered `initial_mask` ready to feed into SAM3 Video Track.

**Use when:** building SAM3 segmentation/tracking workflows and you want a friendly UI for the seed points instead of typing JSON by hand.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
