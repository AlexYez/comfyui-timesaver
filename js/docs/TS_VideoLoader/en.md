# TS Video Loader

Reads a video into frames, audio and a compact `video_info` bundle — and lets you pick the piece **visually**. Drag the timeline handles to set the start and end, zoom in for finer control (Ctrl/Cmd + wheel), and play the selection on a loop.

Resizing, rotation and color conversion run in the decoder's filter graph. For a trim, decoding seeks to the selected region and stops at its end. In a benchmark recorded on 2026-08-11, a selected 4K segment took 0.43 s. Full-resolution decoding followed by resizing took 5.9 s and 6.4 GB of memory.

**Two views while trimming (working tree after 12.13.9).** With a trimmed range and the player paused, the left pane shows the start frame and the right pane shows the frame just before the end boundary. The timecodes show the selected boundaries. Each pane updates its frame and timecode as you drag its handle, including in fullscreen. Playback and timeline scrubbing use the regular player. The panes show source frames before output `frame_rate`, `frame_step` and `max_frames` are applied.

When the file has audio, the waveform below the filmstrip helps you find the beat or the start of a line.

`frame_rate` resamples by source timestamps. Set the size with `longer_side` and `shorter_side`. Either may be `0` to derive it from the other. `divisible_by` rounds down to the chosen multiple. The scaling filter defaults to `area`, and `max_frames` limits the number of output frames. The memory threshold is 60% of machine RAM, with a floor of 8 GiB, unless overridden by `TS_VIDEO_MAX_BYTES`. Above that threshold, `when_too_large=use disk` keeps frames in a memory-mapped file in the ComfyUI temp folder. The output remains an ordinary `IMAGE`. This mode needs a writable temp folder and enough disk space.

**Find cuts.** The split-filmstrip button marks detected shot changes. Double-click a marker to trim to the next cut or the end of the clip. Double-click empty space to reset the trim.

The detector compares brightness histograms of adjacent frames. The first press reads the whole file. Later presses use cached results.

**Formats.** When the graph runs, the decoder probes the container by content. The node reads formats supported by the installed PyAV, including `mp4`, `mov`, `mkv`, `avi`, `mxf`, `dv`, `vob`, `3gp` and animated `webp`, `avif` and `apng`. The browser preview also checks the file extension against its video format list. The pack's decoder does not support `braw` or `r3d`.

**When the player asks for a copy.** The node uses a preview copy for codecs outside H.264, VP8, VP9 and AV1. The two-frames button builds an H.264 copy up to 1280 on the long side. It runs one ffmpeg pass over the clip and shows progress in the status line. The next open uses the cached copy. Trimming and output frames still use the source.

Footage arrives by **drag and drop** — from the file manager, from the Artius browser, or from another node's preview — by the button, by paste, or as a path to a file anywhere on the ComfyUI machine.

> **Preview access.** With a loopback-only server (for example, `127.0.0.1`), previews can open supported videos at any path accessible to the server. A network listener limits previews to the home folder, ComfyUI's input, output, temp and user folders, and roots in `TS_MEDIA_EXTRA_ROOTS`. Separate extra roots with the OS path separator. `TS_MEDIA_ALLOW_ANY_PATH=1` disables the root check. These are server environment settings. The extension check applies in all modes. The routes also reject cross-site browser requests by default.

**Batch over a folder of clips.** The `path` socket takes a path on a wire — from TS Batch Source in `Files in folder` mode (`file_types` = `Videos`), one clip per job. When connected it wins over the file picked in the node, and the timeline trim is **not applied**: it was set on another file and would silently cut every clip of the batch. Frame rate, max frames and size work as usual. The node shows that the file comes from the wire, and after a run, which one.

**Use when:** any workflow that starts from footage rather than from a still.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
