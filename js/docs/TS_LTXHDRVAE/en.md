# TS LTX HDR VAE

The same VAE file at an explicit precision. The stock `Load VAE` does not ask: model
management picks bf16, which is plenty for a picture and not enough for a master —
the quantisation step in the shadows and the top stops is exactly where HDR lives.

Everything else stays on the VAE you already had: guide encoding for both stages, the
latent upscaler, ordinary SDR decode. Wire this one **only** into the decoder's
`hdr_vae` input — that input is lazy, which is what keeps a second copy of a video VAE
out of memory while HDR is off.

**Use when:** HDR is on. Otherwise leave it unwired.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
