# TS Image Depth

Depth map for a still, or for a batch of pictures that have nothing to do with
each other. Runs **Depth Anything V2 Large** on every picture on its own.

Why it is a separate node rather than a switch on the video one: the video model
has no way to look at a single picture except as 32 duplicated frames, and that
flattens the depth range — measured on a portrait, the face and hair blow out to
flat white and the structure is gone. It also forces the short side to 518 px, so
a 1600 px photo went into the model at 784x518 and came back visibly soft.

The pipeline is the reference one, on purpose and with nothing added: trim the
sides to a multiple of 14, run the model, normalize **each picture on its own**
min/max, resize back bilinearly. No denoise, no dither, no guided upscale — those
were built for video, and on a still the guided filter put a halo on contours.
Measured against the reference implementation, the map differs by **0.36%**,
which is under one 8-bit level, at identical detail.

`max_res` is the only control that matters: the longest side the picture is
processed at, snapped down to a multiple of 14. `-1` — the default — is native
resolution, so nothing is resampled at all. Lower it to trade sharpness for
speed and VRAM; on out-of-memory the node retries at half the size, logging each
step.

Weights are fp16 safetensors, downloaded on first use. Only safetensors are
offered in the list, but a workflow that still names an old `.pth` keeps
working — the file is simply no longer suggested.

**Use when:** a depth ControlNet on a single image, a 2.5D still, relighting or
masking by distance, or feeding a 3D reconstruction.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
