# TS Video Saver

Writes frames to a video file and plays the result in the node. Format and quality are named in words — **MP4 / H.264** with draft…lossless, **MP4 / H.265** — about half the size at the same quality, with an optional 10-bit — or **MOV / ProRes** with the usual Proxy…4444 XQ profiles — not in encoder flags. Audio is muxed **in the same pass, into the same file**: no temporary WAV, no second ffmpeg run, no `-audio` duplicate on disk.

**Encoding shows its progress on the node**, the way sampling does: a long clip takes minutes, and a silent node during that time looks stuck.

The player remembers whether you turned sound on. ProRes is not playable in a browser, so the node writes a small H.264 proxy next to it just for the preview (`preview: off` skips that). Hardware encoding is available but never chosen for you: it is much faster and noticeably worse at the same file size.

**An EXR sequence** is the fourth format: one scene-linear float32 (or 16-bit half) file per frame, in its own folder, written from the `hdr_image` socket without touching the range. That socket exists because the ordinary `images` input is clamped to 0..1 long before the saver sees it. There is no compression option — this encoder does not offer one. The sequence carries no audio; a small H.264 proxy is written in the same pass so the node still has something to play.

**Use when:** you want the finished clip on disk, in a format an editor will actually accept — or the HDR master as frames a compositor will accept.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
