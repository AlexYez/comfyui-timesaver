# TS LTX HDR Stats

Lost HDR looks completely normal. The picture is the same, the file was written, no
errors — there is simply nothing above 1.0 in it, and that is discovered in the edit,
when someone tries to pull the sky back.

This node answers "is the range still there?" with numbers: percentiles, share of
samples above 1.0, dynamic range in stops, negatives, NaN/Inf. It also warns when the
highlights are pressed against the ACEScct working ceiling — code 1.0 corresponds to a
linear luminance of about **222.86**, roughly 7.8 stops over white, and anything
brighter was flattened on the way into the model.

**Use when:** the first time you run a shot, and any time an EXR looks suspiciously tame.


<a id="audio"></a>
### 🎵 Audio (6 nodes)

Speech-to-text, text-to-speech, music separation, a waveform visualizer, plus a friendly audio loader and preview.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
