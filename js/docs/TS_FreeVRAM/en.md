# TS Free VRAM

Takes models off the GPU at a chosen point in the graph. The input is a
passthrough — what goes in comes back out — because the node is there for *when*
it runs, not for the data.

**Use when:** the next step needs the whole card. The common case is the heavy
LTX 2.5 VAE decode: that decoder must stay fully resident (ComfyUI marks it
`disable_offload = True`) and asks for a large reservation. With the diffusion
model still on the card the two do not fit together — and on Windows the driver
does not raise an out-of-memory error, it spills into shared system memory.
It looks like a hang: VRAM at the ceiling, GPU at 100%, no progress. Since there
is no exception, ComfyUI never falls back to tiled decoding either.

Insert it into a link:

```text
KSampler --latent--> TS Free VRAM --latent--> VAE Decode
                          ^ model (optional)
```

Connect `model` and only that model leaves the card, so nothing else in the graph
is disturbed. Leave it empty to unload everything currently loaded.

**Warning: the wire type is free, the position is not.** ComfyUI runs a node when
its output is needed, so only a link consumed AFTER the heavy step frees anything
useful. On a `MODEL` or `CONDITIONING` link feeding a sampler the node runs BEFORE
sampling, where there is nothing to free yet.

The same trick is used inside `TS Latent Upscale`, which cannot hold the upscaler
and the diffusion model at once either.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
