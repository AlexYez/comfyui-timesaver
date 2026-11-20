# TS Restore From Crop

…this node to paste a processed crop back into the original frame, with feathered Gaussian or box blur on the seams. The classic crop-and-restore workflow for processing only the interesting region with a heavy model.

**Use the pair when:** running an upscaler or face restorer on a small ROI of a high-resolution image without burning VRAM on the full frame.

**Video:** connect loader `full_images` to `original_images`, the processed `images` crop to `cropped_images`, and TS Video Info `crop_data` to `crop_data`. Send restored frames to TS Video Saver with loader audio and Video Info FPS wired separately.

Keep frame count and order unchanged. A resized processed crop is scaled back to its recorded dimensions before edge padding is removed. Video restoration runs one frame at a time without moving the entire original to GPU. `blur` controls edge feathering and is capped for small regions; `blur_type` selects Gaussian or Box. Edges touching the full-frame boundary do not fade. Video crops blend in linear light; pixels outside the pasted region remain exact. The loader's `use disk` policy carries through when the memory limit is exceeded.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
