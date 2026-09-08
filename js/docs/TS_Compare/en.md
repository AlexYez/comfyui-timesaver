# TS Compare

Two images, or two clips, behind a wipe you drag with the mouse. Both sides take `IMAGE`, so a single frame and a whole batch go in the same socket — what happens next is decided by what arrives.

**A pair of stills stays PNG.** What people compare in a still is detail: sharpness, artefacts, skin after retouching. Running that through a video codec would destroy exactly what is being looked at.

**A batch becomes one file with A above B.** Not two files and not two players, and that is not thrift. Two `<video>` elements drift apart by a frame or two on fast motion, and the comparison starts lying without showing it. And browser video is decoded by the same GPU ComfyUI computes on, so a second decoder pulls at exactly the place the playback guard was written to protect. Two halves of one frame cannot drift — they are one frame — and only one decoder runs.

Both sides are brought to a single frame size, because a wipe over two different sizes compares nothing. If the sides differ in length, the shorter one holds its last frame rather than cutting the comparison short.

**The clip never starts by itself**, pauses when a run begins, and pauses when the node leaves the screen. Play, scrub and the wipe are yours to move.

> The video preview is compressed (H.264, draft quality, up to 1280 wide) — it is a player in a node, not a master. Judge grain and gradients from the saved file, not from here. Stills carry no such caveat: those are PNG.

**Use when:** you changed something and need to see whether it actually got better — an upscale, a denoise, a grade, a retouch.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
