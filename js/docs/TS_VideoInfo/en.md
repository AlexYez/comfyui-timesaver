# TS Video Info

The small companion that unpacks `video_info` into plain numbers: frame rate, frame count, duration and size — both as loaded and as stored in the file — plus `has_audio`, `has_alpha` and a one-line summary. Keeping them here rather than on the loader saves the loader from carrying fifteen output sockets nobody connects at once. A `VHS_VIDEOINFO` bundle from Video Helper Suite plugs in too.

---

The **crop_data** output carries each frame's crop position, size and edge-padding geometry to TS Restore From Crop. Use `video_info` from the same TS Video Loader that supplies `full_images`. `width` and `height` describe `images`, including the crop when enabled. Older bundles and VHS_VIDEOINFO do not carry crop geometry.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
