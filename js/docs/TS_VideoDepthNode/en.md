# TS Video Depth

Depth map for a sequence, using **Video Depth Anything** over a sliding window
of frames so the result does not swim from one frame to the next.

Anything that fits inside one window is now run as a single window of exactly
its own length, instead of being padded out with copies of the last frame —
an input the model has never seen in training.

`flicker_suppression` blends in a temporal **median** of the depth, which drops
single-frame pops without smearing real movement the way an average would.
`flicker_radius` sets how many frames it looks at. `window_length` and
`window_overlap` expose the sliding window itself — leave them alone unless you
are trading VRAM for consistency.

Weights are fp16 safetensors, downloaded on first use: half the download, and
they load in hundredths of a second rather than a full one. Measured against
pure fp32 the depth differs by 0.02% of its range, which is nothing. The older
`.pth` files stay selectable so existing workflows keep working.

**Use when:** driving a depth ControlNet on a clip, building a parallax or 2.5D
move, masking by distance over time.

For a still, reach for **TS Image Depth** below — same family of models, but
the one that was actually trained on single pictures.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
