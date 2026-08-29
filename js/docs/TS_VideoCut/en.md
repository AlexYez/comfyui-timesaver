# TS Video Cut

Trims frames off the start and the end of a clip **and cuts the audio to match**,
from one pair of numbers given in frames.

Usually this takes two nodes that know nothing about each other — one slices the
IMAGE batch, the other the AUDIO — and they have to be told the same boundary in
two different units. The first fractional frame rate then pulls the sound away
from the picture. Here **the frame boundary is the master** and the audio
boundary is derived from it through `fps`, so the two cannot disagree. Measured
drift between picture and sound after the cut:

| clip | drift |
|---|---|
| 100 frames at 24 fps | 0.00 ms |
| 240 frames at **23.976** | 0.01 ms |
| 300 frames at **29.97** | 0.00 ms |
| audio 3 frames longer than the video | 0.00 ms |
| audio shorter than the video | 0.00 ms |

Audio arriving longer or shorter than the video — routine, since encoders round
differently — cannot shift the cut: the span is clamped to the audio that exists
and a mismatch over 50 ms is reported in the log. With no audio connected the
node outputs silence of exactly the trimmed length, so a downstream saver still
receives a valid track. Cutting away the whole clip is refused with the numbers
in the message, rather than handing an empty batch to the next node.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
