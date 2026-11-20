# TS Video Loader

Files with decoded alpha, such as ProRes 4444, output RGBA through `images` and `full_images`. Spatial crops keep alpha.

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

**Spatial crop for local video processing.** Open fullscreen, then enable **Crop** in the panel below the preview. The compact node has no crop buttons. Draw a rectangle; hold Shift for a square. Drag any of its four edges or four corners to resize while keeping the opposite edge anchored. Dimensions snap to the selected multiple. Resizing applies to the entire clip and preserves motion keys, clamping their positions to the frame bounds. Drag inside the box to move it. **New region** redraws it and clears its motion keys.

**More space for the preview.** In fullscreen the timeline defaults to 96 pixels: ruler, filmstrip and waveform when enabled and audio is present. Drag the horizontal separator above the transport to resize the timeline; double-click resets the compact height. Tab to the separator to adjust with Up/Down or reset with Home. Its height is saved with the workflow. The compact node keeps its existing layout.

The size stays fixed throughout the clip. **Multiple** (default 32) rounds the selected pixel width and height up without rescaling the picture. The expanded region shifts inward near a frame edge. If the full frame is smaller than the multiple, edge pixels pad the missing area; restoration removes this padding. Options are 1,8,16,32,64,128,256; choose the multiple your model requires.

**Animate position:** click **+ Keyframe**, seek to another time and drag the region. Once a key exists, dragging creates or updates a key at the playhead. Choose **Smooth** for eased motion or **Linear** for constant velocity between keys. Position holds outside the keys; size never animates. Timecode buttons seek to keys; **Delete keyframe** removes the current key. Ctrl/Cmd+Z and redo work inside fullscreen. Keys are saved with the workflow in source-video time, so trimming and FPS changes do not renumber them. Choosing another file resets the region.

**Wiring:** `images` → processing model → TS Restore From Crop `cropped_images`; `full_images` → its `original_images`; `video_info` → TS Video Info `crop_data` → its `crop_data`. Send the restored frames to TS Video Saver, with loader `audio` and Video Info `fps`.

`full_images` contains the same loaded frames **after global resizing**, before spatial cropping; it needs no second decode. Leave both side limits 0 and global `divisible_by=1` to retain source resolution. The `video` output remains the trimmed source file without spatial cropping. With cropping off, `images` and `full_images` are the same frames. Keep frame count and order through the model. Restoration supports resized processed crops, but rejects a different frame count. A file supplied through `path` ignores spatial cropping, just as it ignores a trim set on another file.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
