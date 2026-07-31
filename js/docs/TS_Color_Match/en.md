# TS Color Match

Transfer the colour palette from a `reference` image to a `target` batch. Two algorithms:

- **MKL** (default) — fast, stable, video-friendly with temporal smoothing.
- **Sinkhorn** — slower but more precise (optimal-transport based).

Includes match masks (`rectangle` / `ellipse` for stabilising on edges only), VRAM-aware chunking, and a `reuse_reference` flag for video.

**Use when:** colour-grading a video to match one keyframe, harmonising shots from different sources, or matching CG into plate footage.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
