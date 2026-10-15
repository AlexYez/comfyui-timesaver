# TS Film Emulation

Film looks for stills and clips: 24 modelled film stocks, or any `.cube` LUT from the pack's `nodes/luts/` folder (drop your own there). On top: contrast, warmth, fade, saturation by tone and modelled grain.

**A compact node: the preview is the whole node.** It shows the current look on the reference portrait with its name over the picture, and nothing else. Every setting sits in a drawer behind the gear button in the corner, in three groups — **Tone** (look strength, contrast, warmth, fade), **Colour** (shadow and highlight saturation) and **Grain** (amount, size, speed for clips, seed). The preview follows the tone and colour sliders, so you judge them by eye; grain is not shown at thumbnail size. Sliders show their value, a value that differs from the default is highlighted, and a dot on the gear says something has been changed; double-click a slider to reset that one, **Reset** resets them all. There is no on/off switch — use ComfyUI's Bypass; a workflow saved with the old switch off shows an **Off** badge that turns it back on with one click. A wire already plugged into one of these inputs keeps working.

**What changed in the maths in 12.13.** Soft stocks no longer park white at grey: Portra 400 went from 0.87 to 0.98, Polaroid from 0.80 to 0.90, while mid-grey stays where it was. Contrast is an S-curve that bends into black and white instead of clipping them; warmth is a white balance, so black stays black; fade lifts the blacks and leaves white alone. Shadow and highlight saturation start at 1.0 (they were 0.8 / 0.85 and quietly drained every image). A `.cube` LUT is applied to the sRGB values it was made for — the old `gamma_correction` detour moved its mid-tones (Cinematic1: 0.41 → 0.47) and is now ignored. The look comes first and your sliders work on top of it, for LUTs as for stocks. Saved workflows keep their own slider values; they look different only where the old behaviour was wrong.

**Pick a look by sight — one choice, no mode switch.** Click the preview for a gallery of every stock and every LUT applied to the same reference portrait — skin, blonde hair, white linen with red embroidery, blue sky, deep forest greens, where film stocks differ most. Filter by family, search, click a card. Whether a stock or a LUT applies follows from the card you picked: the old `film_preset` / `lut_choice` pair (where "External LUT" was really a mode switch, and a LUT picked while a stock was selected silently did nothing) is hidden, and one `strength` slider replaces the two. The same inputs carry the same values underneath, so saved workflows and API calls behave exactly as before. A LUT you add to the folder appears in the gallery on the next open.

**The stocks are modelled, not filtered.** Each one is a characteristic curve per dye layer (the channels cross over, which is where warm highlights and cool shadows come from), colour in six hue bands with skin tones protected, split toning, and — for CineStill and the Vision3 stocks — halation, the red glow around bright areas. Black-and-white stocks use their own spectral sensitivity.

| family | stocks |
|---|---|
| colour negative | Kodak Portra 160 / 400 / 800, Kodak Gold 200, Kodak Ultramax 400, Kodak Ektar 100, Fuji Superia 400, Fuji Pro 400H, Agfa Vista 200 |
| slide | Fuji Velvia 50, Fuji Provia 100F, Kodachrome 64 |
| cinema | Kodak Vision3 250D / 500T, Kodak 2383 Print, Fuji Eterna 250T, CineStill 800T, CineStill 50D |
| instant | Polaroid 600 |
| black and white | Ilford HP5, Kodak Tri-X 400, Kodak T-Max 100, Ilford Delta 3200, Sepia Fiber Print |

> Since 12.13 the seven stocks the node always had (Vision3 250D, Portra 400, Eterna 250T, Vista 200, HP5, Gold 200, Superia 400) render through the new model. Saved workflows open with the same preset selected, but the picture changes — HP5 is now actually black and white.

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
