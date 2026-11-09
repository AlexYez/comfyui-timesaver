# TS H3 Regenerate

Upscales a denoised H3 latent and refines it with Open-H3-Regenerate-2K.
Connect a clean H3 model through [TS H3 Regenerate Model](https://github.com/AlexYez/comfyui-timesaver/blob/master/js/docs/TS_H3RegenerateModel/en.md).
Keep draft-generation LoRAs on a separate branch.

Connect the `NOISE` output of **Random Noise** to `noise`. Set the seed in Random Noise.
You can connect the same source to the draft sampler and Regenerate. Noise is generated
on the target grid and shared across overlapping chunks.

`denoise=1` uses the published 2/3-step schedule; `0` only upscales, without diffusion.
Intermediate values lower the starting sigma and build a shift-6 grid, keeping the step count.
Less denoise means less redraw; evaluate quality on your own clip.

Target width/height must be multiples of 32. For 1080p compute 1920×1088 and centre-crop to 1080
following decoding. For 4K compute 3840×2176 and crop to 2160. When upscaling from below
2K area to above it, the node first uses an intermediate size around 2K that accounts for the
aspect ratio. A 3840×2176 target uses 2560×1440 in this pass, or 1440×2560 in portrait.

## Keyframes and Full Reference

Connect the original positive output to `conditioning`. The default `off` mode keeps text
and modality tags. The modes below supply keyframes or media references for refinement.

`first frame (preserve character)` adds a freshly encoded start-image guide.
Connect `first_frame` and a **normal H3 video VAE** to `video_vae` (compatible INT8 video VAEs work).
For a 2× upscale, resize the source directly to the target size with a separate **ImageScale** and
connect it to `first_frame`. This preserves detail lost when preparing the smaller draft image.
If the first image in the batch differs in size, the node fits it to the final canvas with Lanczos
and no cropping, then encodes it once. Fast VAE is decode-only. The reference guides the model
during generation.
The [published first-frame experiment](https://ai-muninn.com/en/blog/open-h3-regenerate-2k-rtx5090)
concerned small lettering. Evaluate face preservation on your clip.

`first and last frames` adds `last_frame` using the same normal `video_vae`. The node fits
the end image with Lanczos and a center crop when needed. The start guide stays anchored at
global frame zero. Only the final window receives the end guide at the clip's last frame.

`connected frames (auto)` uses `first_frame` and `last_frame` independently. Connect
either image or both. With neither connected, refinement uses the text and draft
latent. Connect a normal H3 video VAE whenever a frame is available. The end image
only guides the final window.

`full reference (Ref2VA)` reuses encoded images, videos, videos with audio and separate audio
from the positive output of **MiniMax H3 Reference to Video**. It preserves media order and
text tags. Connect a clean pruned **Ref2VA** through **TS H3 Regenerate Model**. Leave
`first_frame`, `last_frame` and `video_vae` empty on the Regenerate node. Reference size and
length affect VRAM usage. Full Reference is experimental. All reference modes are inactive
at `denoise=0`.

## Audio and chunks

Audio: source mode pins the original audio latent during refinement. Reference silence mode needs
an H3 audio VAE and pins encoded silence instead. The original audio latent is returned unchanged
when present, including in silence mode. Duration stays unchanged; 24 fps is assumed.
Keeping audio does not guarantee preservation of lip motion: inspect the rendered result.

`chunk_length=0` processes the whole clip. 2K may exceed 16 GB VRAM.
Nonzero lengths process the clip in chunks. Set `chunk_length` and `temporal_overlap` in multiples
of 17 frames, with overlap smaller than chunk length. Noise is shared across the clip.
Use at least 17 frames of overlap for either continuation mode with multiple chunks.
`chunk_mode=independent (crossfade)` preserves the previous independent-chunk behavior.
Different details in the overlap can produce visible morphing during the crossfade.
`continue (lock overlap)` uses the completed overlap as clean video context with a zero denoise
mask and only redraws the new part. The completed overlap is preserved exactly.
`continue (blend overlap)` locks the first half of the latent overlap (at least one position)
and blends the redrawn remainder into the previous chunk with a smooth transition.
Continuation can propagate changes along a long clip; inspect drift as well as seams.
If the last continuation window would add fewer than 17 new frames, the node attempts to
balance the existing windows on the native phase grid. For 226 frames at 85/17, lengths become
68,68,68,73 rather than 85,85,85,22: there is no isolated five-frame redraw after frame 221.
The number of windows, overlap, source duration and selected maximum chunk length are preserved.
Very short clips may have no better split on this grid. Balancing reduces this tail failure case;
it does not guarantee motion or camera preservation through generative refinement.
With first-frame guidance, each chunk sees the same image at its original global time zero
(a negative local time in later chunks). Chunk processing is experimental.
`chunk_length=0` processes the whole clip in one pass.

Fast VAE accepts FP32 or native INT8 convrot weights.

Decode using the standard H3 video VAE or **TS H3 Fast VAE Decode**. Connect a standard video VAE
to its `fallback_vae` input for clips whose whole-clip fast decode does not fit memory.
This integration uses native ComfyUI sampling and PyTorch SDPA, without the SoL-Engine/MXFP8/CUDA Graph optimizations.

Recipe: [NVIDIA](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K).
Decoder and memory-lean forward: NVIDIA / [rdh9616](https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate), Apache-2.0.
