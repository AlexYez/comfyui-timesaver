# TS Frame Interpolation

Smooth frame interpolation using RIFE / FILM models. Boost a 12 fps animation to 24/48/60 fps, or smooth jittery video.

**Use when:** the model output is choppy and you want cinema-smooth motion.

**`match_length` mode - restore an exact frame count after trimming.** Upscaling
often ruins the first few frames; you cut them, 200 frames become 188, and the
result has to be 200 again. Connect the original batch to `reference` - only its
frame count is read - or type the number into `target_frames`.

`fill` decides where the missing frames come from:

| `fill` | What it does | When to use it |
| --- | --- | --- |
| `hold_start` | repeats the first frame at the head | you trimmed the head; every surviving frame keeps its original index, so the clip still lines up with the source and its audio |
| `hold_end` | repeats the last frame at the tail | you trimmed the tail |
| `stretch` | resamples the whole clip with the model | smoothness matters more than frame-for-frame alignment |

**Warning:** `stretch` shifts **every** frame in time: 188 -> 200 slows the clip
by 6.4%. If the result is laid over the original audio or compared frame by
frame, use `hold_*`. Both `hold_*` modes run without the model and without VRAM.
When the target is shorter than the clip, frames are dropped from the held edge.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
