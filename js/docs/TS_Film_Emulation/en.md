# TS Film Emulation

Built-in film stock presets (Kodak Portra/Vision3, Fuji, Cineon-style, …) plus optional `.cube` LUT loading from `models/luts/`. Adds gamma correction, contrast curve and a tunable `lut_strength`.

**The grain is modelled, not sprinkled on.** Real grain is a fluctuation in
*density*, which is a logarithmic quantity, so the noise is applied in log space
rather than added to the pixels. Two things follow, and both are what your eye
expects from film: the same fluctuation is a wide swing in a bright area and
almost nothing in a dark one, and it fades again at the very top where the
emulsion saturates. Measured across a grey ramp at one setting:

| tone | 0.05 | 0.25 | 0.50 | 0.75 | 0.95 |
|---|---|---|---|---|---|
| grain (σ) | 0.001 | 0.016 | 0.042 | **0.061** | 0.044 |

The old implementation gave 0.049 / 0.060 / 0.049 — the same everywhere, which
is what plain noise looks like.

**`grain_speed` — the control professional grain plugins have.** At `1.0` a new
pattern is drawn every frame: lively, and unmistakably digital. At `0.5` one
pattern is held for two frames, at `0.25` for four — the way scanned film looks
when the grain does not race the action. `grain_seed` makes a re-render match
the take you already graded. Neither affects a single still.

**Clips are handled properly.** Work happens on the GPU in chunks sized from the
free VRAM, so peak memory stays around 2.6 GB whether the clip is 8 frames or
64, and the result does not depend on how it was chunked. Measured against the
previous CPU path, same machine: 24 frames of 1080p with a LUT went from **6.5 s
to 0.8 s**, eight 4K frames from 6.8 s to 0.8 s.

**Use when:** giving renders a cinematic feel without leaving the graph.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
