# TS LTX First/Last Frame

Apply LTX-Video keyframe conditioning for the first and (optionally) last frame in one node — equivalent to chaining two `LTXVAddGuide` nodes, with cleaner UX.

**Use when:** you have specific start/end frames and want LTX to interpolate between them.


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
