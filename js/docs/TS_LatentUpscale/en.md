# TS Latent Upscale

Re-samples an already-denoised **MiniMax H3** audio+video latent at a larger
size. Three nodes from
[Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale)
(MIT, bbaudio-2025) folded into one — the pipeline, the upscale-model settings
and the temporal split settings are simply inputs here, because nobody ever
wanted one without the others.

Per chunk of the clip: cut along time with an overlap → upscale that chunk's
video latent with the H3 3D upscaler (audio untouched) → re-anchor the
conditioning and pin frame 0 to the previous chunk's result → sample → stitch
back over the overlap. **Peak VRAM is one chunk, not one clip**, and the
diffusion model is offloaded while the upscaler works, since the two are never
needed at once.

**Subfolders in `models/latent_upscale_models` are finally visible.** The
original scanned the folder root only and returned bare filenames — on this
machine it listed 2 of the 4 models actually present. Here the list is recursive
and shows `subfolder/file.safetensors`, every folder declared in
`extra_model_paths.yaml` is searched, and a name that tries to climb out of its
folder is refused. Picking an upscaler from another model family now explains
itself instead of failing with `Missing key(s) in state_dict`.

`chunk_length` and `temporal_overlap` must be multiples of **17** — the model's
keyframe grid — and that is checked before the run rather than half an hour into
it.

> **Spatial tiling was deliberately left out.** The original also splits each
> chunk into tiles; it was dropped along with its input. Tile seams need their
> own fade and blend settings, and a clip that needs tiling is better served by
> shorter chunks.

---

Full node reference: [README](https://github.com/AlexYez/comfyui-timesaver#-node-reference)
