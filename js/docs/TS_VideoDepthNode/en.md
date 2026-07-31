# TS Video Depth

Depth-Anything-based per-frame depth estimation, optimised for video (temporal consistency). v9.4 brought a full GPU-pipeline overhaul: SDPA attention, TPDF dithering on output, sub-chunk processing for long clips, and a numerically-equivalent DPT tail — same outputs, dramatically faster on RTX cards.

**Use when:** building depth-aware ControlNet pipelines, parallax effects, or 3D reprojection.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
