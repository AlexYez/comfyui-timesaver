# TS LTX HDR Guide

One node per guide frame — first, last — that both picks the branch and prepares it.
Off, the SDR image passes through untouched; on, two guides are built from the EXR.

**The half-resolution guide is built from the original, not by shrinking the full one.**
The official two-stage pipeline rebuilds image conditioning for each resolution, and
that is not the same thing: averaging belongs in linear light, not in log codes.

The lazy inputs are the reason the switch is free. With HDR off, ComfyUI never walks
into the EXR branch; with it on, the `LTXVPreprocess` chain is never computed. A broken
EXR path cannot break an SDR run.

Strict validation catches stage sizes that do not match either legal wiring — the same
size (no latent upscaler) or exactly double (with the x2 upscaler).

A run guided by an **ordinary JPG or PNG** is supported too, through the
`image_guide` input — for when you generate video from a picture and still want a
float32 scene-linear master out. Be clear-eyed about what that gives you: an 8-bit
picture holds nothing above 1.0, and no curve invents what was never captured.
Recovering highlights from an SDR still is SDR→HDR expansion, a different model
technology, and this is not it.

What you do get is worth having anyway. The gamma is removed properly — feeding
sRGB codes in as if they were linear light is off by up to **2.3 stops** in the
shadows (measured: 0.131 of the ACEScct code range) — the master stays float32
scene-linear with no banding and no baked-in gamma, and the working range keeps
its headroom: SDR white sits at ACEScct code **0.555**, so **45% of the range,
7.8 stops, is left above it** for the model to generate into. Whether it actually
does is an empirical question — that is what TS LTX HDR Stats is for.

To use it, turn HDR on and bypass the EXR loader: the guide falls back to
`image_guide` on its own, no extra switch.

**Use when:** feeding first/last frames into a two-stage LTX graph.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
