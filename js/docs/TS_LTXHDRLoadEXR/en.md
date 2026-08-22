# TS LTX Load HDR EXR

Reads an EXR as linear float32 — no normalisation, no upper clamp. Reports the range
and, in particular, **what share of the frame is above 1.0**. If that is zero, the
highlights were already lost upstream and the rest of the path has nothing to preserve.

Three backends: **OpenImageIO** (what the official pipeline uses, rarely installed),
**PyAV** (ships with ComfyUI, needs no setup — the default in practice) and **OpenCV**
(only reads EXR when `OPENCV_IO_ENABLE_OPENEXR=1` was set *before* ComfyUI started;
setting it later does nothing, because the reader registers at import).

Half-float files — what almost everyone actually renders — work too. That needed the
frame's raw planes to be read by hand: PyAV cannot convert `gbrpf16le` to an array at
all, and any format conversion goes through swscale, which clamps float data to
`[0, 1]`. Measured: a 4-channel EXR holding 4.0, read the convenient way, comes back
as 1.0.

**Use when:** the guide frames for your shot are renders, not screenshots.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
