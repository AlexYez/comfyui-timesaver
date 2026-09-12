# TS H3 Audio Inject

Lip-sync for MiniMax H3: the node pins a soundtrack inside the latent, and the
video has to match it. Use it when a character must speak your exact line — a
voice-over, a recording, a song.

**Use when:** you need a talking character driven by an existing soundtrack.

**How it works.** H3 denoises video and audio as ONE latent: two parts of the
same sequence, with attention running between them. The node encodes your track
with the audio VAE, writes it into the audio part and marks that part as
preserved. A stock KSampler then holds the audio fixed at every step, and the
picture has no choice but to agree with it.

Neither the model nor the sampler is patched: the holding is done by ComfyUI's
own masked-sampling path.

Wiring:

```text
MiniMaxH3ImageToVideo --> latent --> TS H3 Audio Inject --> KSampler --> VAEDecode
                                        ^ audio  ^ audio_vae
```

`audio_vae` is the same audio VAE you decode with. A track shorter than the clip
is padded with silence, a longer one is trimmed. **The decoded clip carries
exactly your track** (through the VAE), not something resembling it.

**Choosing the frame count.** H3 only accepts lengths of the form `17n + 5` at 24
fps. Take the exact track length — the `duration_seconds` output of
`TS Audio Loader` — and compute:

```text
max(5, ceil(a * 24 + 0.2)) + (5 - (max(5, ceil(a * 24 + 0.2)) % 17)) % 17
```

The `+0.2` frame margin is not a guess. Audio runs on its own grid of 40 latent
frames per second, the slot is `round(frames * 5 / 3)`, and the fractional part of
that expression is only ever 0, 1/3 or 2/3. So the slot departs from `frames / 24`
by at most 1/120 s, and 0.2 of a frame covers exactly that. Without the margin the
end of a phrase is sometimes clipped: sweeping every length from 1 to 10 s in 1 ms
steps, `round` clipped in 244 cases (up to 29 ms), `ceil` in 32 (up to 8 ms), and
the margin never did.

**Warning:** do not use the whole-second `duration` output for this. It rounds up,
so the clip ends up almost a second longer than the speech and the model invents
that tail for nothing.

**On muxing the original track.** Video and audio lengths match exactly when the
frame count divides by 3 — 39, 90, 141, 192, 243. Otherwise the track is ±8.33 ms
(a fifth of a frame) longer or shorter. Chasing that is not worth it: forcing
divisibility by 3 costs 16 extra frames on average, and the mismatch does **not**
accumulate — the audio stays real samples, the tempo is untouched, and speech
starts at zero. Lay the original track at frame 0 and the sync holds to the end.

**Warning:** this is not the stock `Add Guide for MiniMax H3`. That one adds the
audio as a condition — the model listens to it but still generates its own
soundtrack. Here the output audio stream itself is replaced.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
