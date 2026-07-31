# TS Audio Visualizer

Turns any `AUDIO` clip into a stylized SoundCloud-style waveform image at the resolution you choose. Blue→violet gradient bars (default `Violet`; `Indigo`, `Neon`, `Spectrum`, `Fire` and more) are drawn as antialiased rounded capsules with a soft neon glow, sitting over an **audio-reactive abstract background** driven by the same loudness envelope: `nebula` (layered mountains + waveform aura), `glow`, `mountains`, `plasma`, or `none`. Rendered entirely on torch — no extra dependencies. Outputs both the `IMAGE` and a `MASK` (bar fill + glow alpha) so you can composite the bars over video or footage. Mirror or bottom bars, horizontal / vertical / amplitude-driven gradient, plus glow, background intensity, sensitivity, smoothing and bar geometry controls.

**Use when:** building music-video overlays, audiogram clips for social, or a quick visual for a voiceover / track.


<a id="llm"></a>
### 🤖 LLM (2 nodes)

Multimodal LLM-powered prompt enhancement and image understanding.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
