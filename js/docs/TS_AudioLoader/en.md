# TS Audio Loader

The audio loader you'd build yourself if you had time. Loads audio from any media (mp3/wav/mp4/mov/…), shows a real waveform, lets you crop visually by dragging on the waveform, and can even record from the microphone right inside the node. Outputs the `AUDIO` waveform, a whole-second `duration` int and an exact `duration_seconds` float.

**Use when:** preparing voiceovers, music beds, or any audio that needs trimming before processing.

**Warning:** `duration` rounds UP to a whole second. That is fine for display, but if a frame count is computed from it the clip ends up almost a second longer than the speech. Where the length matters, take `duration_seconds` — it reports exactly `samples / sample_rate`.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
