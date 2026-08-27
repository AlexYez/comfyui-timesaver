# TS RTX Upscaler

Hardware-accelerated upscale via NVIDIA RTX Video Super Resolution (`nvvfx`).
Four quality levels (LOW/MEDIUM/HIGH/ULTRA), batched processing. **Requires an
RTX GPU.**

**Installing it needs NVIDIA's own index** — the package on PyPI is a 2.7 KB
stub that fails to build, and the real wheel (792 MB, carrying its own VFX SDK,
TensorRT and NPP libraries — no separate NVIDIA SDK needed) is published only
here:

```
pip install nvidia-vfx==0.1.0.1 --no-build-isolation --index-url https://pypi.nvidia.com
```

**The engine is now kept alive between runs.** Measured on an RTX 3080 Ti:
creating it costs ~730 ms, changing the output size on a live one costs 6 ms,
and changing quality costs nothing at all. It used to be created per run, which
on eight frames to 1080p was 93% of the node's entire work. Repeat runs are now
**6.8× faster** (0.93 s → 0.14 s). The engine holds 162 MB, and that memory is
visible to ComfyUI's memory manager, so keeping it costs nothing you cannot see.

For reference, the upscale itself runs at **131 frames/s** to 1080p and **28
frames/s** to 4K, and the frames never leave the GPU between the two — so a
faster frame source would not make it quicker.

**Use when:** you have an RTX card and want speed-of-light upscaling for video.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
