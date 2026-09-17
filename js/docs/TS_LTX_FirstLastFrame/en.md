# TS LTX First/Last Frame

Pin the first and, optionally, the last frame of an LTX video in one node.

**A first frame alone costs nothing extra.** It is written into the latent in place — the same thing ComfyUI's own LTX 2.5 image-to-video template does with `LTXVImgToVideoInplace`. The latent keeps its length, so the video comes out exactly as long as you asked and there is nothing to clean up afterwards.

**A last frame can only be pinned as a keyframe guide**, and that is a different mechanism: LTX appends the guide to the latent as extra frames carrying their own position in time. ⚠️ **Those appended frames have to be removed by an `LTXV Crop Guides` node placed between the sampler and the decoder** — conditioning from this node, latent from the sampler. Forget it and the tail of the video is the guides themselves, half-noised and looking like corruption. The node says so in the console every time it appends them, and this is exactly how ComfyUI's own first-last-frame template for LTX 2.5 is wired.

**Strength is per side.** `first_strength` / `last_strength` at 1.0 lock the frame hard; lower values let the model move it. Zero switches that side off, so a node with both images wired can still be run as first-frame-only.

**Use when:** you have specific start/end frames and want LTX to interpolate between them — or just a start frame, which is the common case and the cheap one.


<a id="hdr"></a>
### 🌈 HDR / EXR (7 nodes)

The native HDR path of LTX 2.5, as a set of nodes. It is **off by default and costs
nothing while it is off**: with the switch down, no EXR is read, no float32 VAE is
loaded, and the graph behaves exactly as it did before these nodes existed.

**What this is not:** it does not invent HDR out of an SDR clip. It preserves the HDR
that came in — from EXR guide frames, through the model, out to an EXR master.

A wired example with notes on the canvas: [`example_workflows/08_ltx25_native_hdr.json`](example_workflows/08_ltx25_native_hdr.json). It is the HDR half only — drop your own two-stage LTX graph around it, as the notes explain.

Why the ordinary nodes cannot do it: `Load Image` flattens anything above 1.0 to 1.0
without saying so, and `LTXVPreprocess` pushes the frame through an H.264 round-trip
and 8-bit bytes (`(image * 255.0).byte()` in the core source). Both are fine for SDR
and fatal for HDR, so the HDR branch bypasses them entirely.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
