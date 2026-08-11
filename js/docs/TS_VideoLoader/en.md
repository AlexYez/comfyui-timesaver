# TS Video Loader

Reads a video into frames, audio and a compact `video_info` bundle — and lets you pick the piece **visually**. The node's body holds a player and a timeline with a filmstrip: drag the handles to set in and out, zoom into a single second of an hour-long take (Ctrl/Cmd + wheel), and loop the selection while you judge it. Type an exact timecode when the mouse is not precise enough.

**It is fast on purpose.** All resizing, rotation and colour conversion happen inside the decoder's own filter graph, and reading stops at the end of your selection — a two-second piece of an hour-long 4K take costs a seek plus two seconds, not an hour. Measured on a 4K clip: 0.43 s where the naive path (decode at full resolution, then resize) takes 5.9 s and 6.4 GB.

**The sound track is drawn under the filmstrip** whenever the file has audio — a beat or a spoken word is far easier to hit by the wave than by the picture — and the player follows the handle you drag, so the exact frame that will become the first or the last one is on screen while you are still choosing it.

`frame_rate` resamples by real timestamps, so a variable-frame-rate source comes out evenly spaced. Size is set as `longer_side`/`shorter_side` rather than width and height, so one graph fits landscape and portrait footage alike; either may be `0` to derive it from the other. `divisible_by` rounds down to what video models want, the scaling filter defaults to `area` (footage is almost always scaled down, and averaging beats interpolation there), and `max_frames` is the memory guard — an oversized request explains itself instead of running the machine out of RAM.

Footage arrives by **drag and drop** — from the file manager, from the Artius browser, or from another node's preview — by the button, by paste, or as a path to a file anywhere on the ComfyUI machine.

> **A path anywhere on the machine — and what happens when the server is not yours alone.** Running ComfyUI the usual way, on `127.0.0.1`, the node and its preview read **any path you give them**: Documents, Desktop, another drive. Nothing is copied into `input`, which is the whole point — that folder grows without end otherwise.
> If ComfyUI is started open to a network (`--listen 0.0.0.0`, a LAN box, a cloud machine), the preview is served over HTTP to whoever can reach that port, so it then stays inside your home folder and ComfyUI's own directories. Add more with `TS_VIDEO_EXTRA_ROOTS=D:/footage` (several separated by your OS path separator), or lift the limit with `TS_VIDEO_ALLOW_ANY_PATH=1`. Both are set on the machine by its owner — not inside a workflow, which can arrive from anyone.

**Use when:** any workflow that starts from footage rather than from a still.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
