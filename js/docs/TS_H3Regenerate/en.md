# TS H3 Regenerate

Upscales a denoised H3 latent and refines it with Open-H3-Regenerate-2K.
Connect a clean pruned H3 loader to **TS H3 Regenerate Model**, then to **TS H3 Regenerate**.
Keep draft-generation LoRAs on a separate branch; the merged release adapter already includes Turbo and Sharp.

`denoise=1` uses the published 2/3-step schedule; `0` only upscales, without diffusion.
Intermediate values lower the starting sigma and build a shift-6 grid, keeping the step count.
Less denoise means less redraw; evaluate quality on your own clip.

Target width/height must be multiples of 32. For 1080p compute 1920×1088 and centre-crop to 1080
following decoding; for 4K compute 3840×2176 and crop to 2160. Above 2K area the upscaler first
passes through 2560×1440 (transposed for portrait). Draft image/keyframe references are removed
from refine conditioning; text and modality tags remain. The draft image is already in the latent.

Optional `reference_mode=first frame (preserve character)` adds a freshly encoded start-image guide.
Connect `first_frame` and a **normal H3 video VAE** to `video_vae` (compatible INT8 video VAEs work).
In the workflow, branch the source image into a separate **ImageScale**: resize directly to the
final width/height, twice the draft width/height. Feed this prepared image to `first_frame`.
Do not enlarge the already reduced draft image; that cannot recover the source details.
Only the first image is encoded, once. If its size differs, the node fits it to the final canvas
with Lanczos and no cropping, like native H3 I2V. Fast VAE is decode-only and cannot encode this guide.
At `denoise=0` the guide is inactive and neither extra input is required.
The mode adds conditioning, rather than copying the source pixels into the output video.
It is an experimental extension of the release recipe: improved identity is not guaranteed.
The [published first-frame experiment](https://ai-muninn.com/en/blog/open-h3-regenerate-2k-rtx5090)
concerned small lettering; face preservation still needs evaluation.

Audio: source mode pins the original audio latent during refinement. Reference silence mode needs
an H3 audio VAE and pins encoded silence instead. The original audio latent is returned unchanged
when present, including in silence mode. Duration stays unchanged; 24 fps is assumed.
Keeping audio does not guarantee preservation of lip motion: inspect the rendered result.

`chunk_length=0` processes the whole clip. 2K may exceed 16 GB VRAM.
Nonzero lengths enable experimental independent temporal chunks, with lengths/overlap in multiples
of 17 frames and overlap smaller than chunk length. Shared full-clip noise and crossfades help seams;
`chunk_mode=independent (crossfade)` preserves the previous independent-chunk behavior.
Different details in the overlap can produce visible morphing during the crossfade.
`continue (lock overlap)` uses the completed overlap as clean video context with a zero denoise
mask, and only redraws the new part. The completed overlap is preserved exactly instead of blending
two different versions. Use a nonzero overlap (at least 17 frames) for multiple chunks.
Continuation can propagate changes along a long clip; inspect drift as well as seams.
If the last continuation window would add fewer than 17 new frames, the node attempts to
balance the existing windows on the native phase grid. For 226 frames at 85/17, lengths become
68,68,68,73 rather than 85,85,85,22: there is no isolated five-frame redraw after frame 221.
The number of windows, overlap, source duration and selected maximum chunk length are preserved.
Very short clips may have no better split on this grid. Balancing reduces this tail failure case;
it does not guarantee motion or camera preservation through generative refinement.
With first-frame guidance, each chunk sees the same image at its original global time zero
(a negative local time in later chunks), not as a new starting pose at every boundary.
Both chunk modes differ from the whole-clip release; `chunk_length=0` avoids chunk boundaries.

Fast VAE accepts FP32 or native INT8 convrot weights.

Decode using the standard H3 video VAE or **TS H3 Fast VAE Decode**. Connect a standard video VAE
to its `fallback_vae` input for clips whose whole-clip fast decode does not fit memory.
This integration uses native ComfyUI sampling and PyTorch SDPA, without the SoL-Engine/MXFP8/CUDA Graph optimizations.

Recipe: [NVIDIA](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K).
Decoder and memory-lean forward: NVIDIA / [rdh9616](https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate), Apache-2.0.
