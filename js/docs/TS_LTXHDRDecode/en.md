# TS LTX HDR Decode

The single final decode, with two outputs that must never be confused:

- `preview_sdr` — what you look at: tonemapped, exposure applied, sRGB encoded.
- `hdr_linear` — what you save: scene-linear Rec.709 float32, no tonemap, no gamma,
  **no upper clamp**. No preview setting touches it.

While HDR is off the master slot returns an `ExecutionBlocker`, so a connected EXR
saver does not run at all — no stub file, no black frames, nothing.

The decode itself comes out as an ACEScct working signal in `[0, 1]` — which is why
ComfyUI's standard `(x + 1) / 2` clamp on the LTX VAE costs nothing here. The range
reappears on the inverse curve, after the decoder.

**Use when:** replacing the pair of VAEDecode nodes at the end of a two-stage graph.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
