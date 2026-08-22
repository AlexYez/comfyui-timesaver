# TS LTX HDR Settings

One switch for the whole path. Everything else reads this node, so a single checkbox
changes the mode of the entire graph instead of five settings that must agree.

`input_color_space` says what the EXR files already are — `ACESCG`, `SRGB_LINEAR` or
`ACESCCT`, the same three the official `--hdr` flag takes. The preview controls live
here too, next to the switch, which is the point: exposure and tonemap belong to what
you look at, never to what gets written.

`hdr_mode` picks which of the two HDR technologies this graph uses, and they are
genuinely different, not two shades of one:

- **preserve HDR from EXR (ACEScct)** — the native LTX 2.5 path. The range came in
  from an EXR and the job is not to lose it. Working curve ACEScct, code 1.0 = linear
  **222.86**, and the output converts AP1 → Rec.709.
- **expand HDR from SDR (LogC3 IC-LoRA)** — the HDR IC-LoRA. There was no range on
  the way in; the model grows it out of ordinary SDR. Working curve LogC3, code 1.0 =
  linear **55.08**, and ⚠️ **the primaries are left alone** — the model already emits
  the right ones, so applying the ACES matrix here would shift the colour. Our inverse
  curve matches the official `LTXVHDRDecodePostprocess` to within 1e-6, measured over
  501 points.

In expand mode the guide is an ordinary SDR image and no EXR is read at all: you wire
the IC-LoRA into the model yourself, the same way the official 2.3 workflow does. The
LoRA is validated on LTX 2.3; support for 2.5 is officially in development.

**Use when:** always, if you use any of the other nodes here.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
