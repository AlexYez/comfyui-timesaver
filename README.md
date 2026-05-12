<div align="center">

<img src="icon.png" alt="Timesaver Icon" width="120" />

# 🚀 Timesaver Nodes for ComfyUI

**A friendly toolkit of 59 production-ready nodes that take the boring busywork out of your ComfyUI graphs.**

Resize, color-grade, key, denoise, transcribe, translate, prompt-build, manage models — without leaving the canvas.

[![Version](https://img.shields.io/badge/version-9.6-blue.svg)](pyproject.toml)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-V3%20API-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-see%20LICENSE.txt-lightgrey.svg)](LICENSE.txt)

🇷🇺 [README на русском](README.ru.md)

</div>

---

## ✨ What's Inside

Whether you build pipelines for image generation, video, audio, or just want to tidy up your prompts — Timesaver has a node for that.

|  | Category | Count | Highlights |
|---|---|---|---|
| 🖼️ | **[Image](#image)** | 28 | Resize, color, masks, keyer, tiling, 360°, Lama cleanup, BiRefNet bg removal, ViTMatte, SAM3 picker |
| 🎬 | **[Video](#video)** | 7 | Frame interpolation, RTX/spandrel upscale, depth, animation preview |
| 🎵 | **[Audio](#audio)** | 5 | Whisper transcription, Silero TTS, Demucs stem split, audio cropping |
| 🤖 | **[LLM](#llm)** | 2 | Qwen 3 VL multimodal chat, Super Prompt with voice input |
| 📝 | **[Text & Prompts](#text)** | 4 | Prompt builder, batch loader, style picker, Russian stress marks |
| 📁 | **[Files & Models](#files)** | 8 | Model scanner, FP8 converter, file path loader, EDL→YouTube chapters |
| 🛠️ | **[Utils](#utils)** | 4 | Custom sliders, math, smart type-aware switch |
| 🎨 | **[Conditioning](#conditioning)** | 1 | Multi-reference image conditioning |

> All 59 nodes use the **ComfyUI V3 API** (`comfy_api.v0_0_2.IO` — pinned namespace for stability).

---

## 📑 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Updating](#-updating)
- [Node Reference](#-node-reference)
  - [🖼️ Image](#image)
  - [🎬 Video](#video)
  - [🎵 Audio](#audio)
  - [🤖 LLM](#llm)
  - [📝 Text & Prompts](#text)
  - [📁 Files & Models](#files)
  - [🛠️ Utils](#utils)
  - [🎨 Conditioning](#conditioning)
- [Tips for Beginners](#-tips-for-beginners)
- [Troubleshooting](#-troubleshooting)
- [Repo Layout](#-repo-layout)
- [License & Credits](#-license--credits)

---

## 📦 Installation

### Option 1 — ComfyUI Manager (recommended)

1. Open ComfyUI Manager → **Custom Nodes Manager**.
2. Search for `Timesaver` and install.
3. Restart ComfyUI.

### Option 2 — Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AlexYez/comfyui-timesaver
cd comfyui-timesaver
python -m pip install -r requirements.txt
```

Then restart ComfyUI.

> 🪟 **Windows portable build**: run `pip` from the bundled Python (e.g. `python_embeded\python.exe`), otherwise dependencies will land in the wrong interpreter.

> 🍎 **macOS / Linux**: use the same Python that ComfyUI runs with. Activate your venv before `pip install`.

### Optional dependencies

A few nodes need extra packages — they fail gracefully and tell you what's missing if you try to run them without:

| Node | Needs | Install via extra |
|---|---|---|
| TS Cube ↔ Equirectangular | `py360convert` | (bundled in core) |
| TS Qwen 3 VL int4/int8 | `bitsandbytes` | `pip install -e .[llm-quant]` |
| TS Music Stems | `demucs`, `geomloss`, `pykeops` | `pip install -e .[audio-stems]` |
| TS Whisper (legacy fallback) | `openai-whisper` | `pip install -e .[audio-whisper]` |
| TS Silero TTS / Stress | `silero`, `silero-stress` | `pip install -e .[audio-silero]` |
| TS RTX Upscaler | `nvvfx` (NVIDIA RTX only) | install manually |
| TS Video Upscale With Model | `spandrel` | install manually |

> Want everything in one go? `pip install -e .[all]`

---

## 🎯 Quick Start

1. Launch ComfyUI.
2. **Right-click → Add Node** or double-click an empty area on the canvas.
3. Type `TS` in the search box — every Timesaver node has a `TS` prefix.
4. Pick a node, connect inputs/outputs, and run.

**Node naming convention:**

```
TS_<NodeName>     ← class id (used in workflows / search)
TS <Display Name> ← what you see in the UI
TS/<Category>     ← location in the right-click menu
```

**Most common output types:**

| Type | Means |
|---|---|
| `IMAGE` | A batch of frames `[B, H, W, 3]`, values in `[0, 1]` |
| `MASK` | Single-channel mask `[B, H, W]`, values in `[0, 1]` |
| `AUDIO` | `{"waveform": [B, C, T], "sample_rate": int}` |
| `LATENT` | A latent dict `{"samples": ...}` |
| `CONDITIONING` | A list of `(cond, meta)` pairs for samplers |
| `STRING` / `INT` / `FLOAT` | Plain values |

ComfyUI highlights compatible sockets for you while dragging — no need to memorise types.

---

## 🔄 Updating

Already installed via git?

```bash
cd ComfyUI/custom_nodes/comfyui-timesaver
git pull
python -m pip install -r requirements.txt
```

Restart ComfyUI. Your existing workflows keep working — node ids and inputs are frozen across versions.

---

<a id="-node-reference"></a>
## 📚 Node Reference

Every node below shows the actual look in ComfyUI (English UI). Click any image to see it full size on GitHub.

---

<a id="image"></a>
### 🖼️ Image (26 nodes)

Everything that touches pixels: resize, color, masks, background removal, keying, tiling, panoramas, and inpainting.

#### TS Image Resize
<img src="doc/screenshots/ts_image_resize.png" alt="TS Image Resize" width="450" />

The resize node you actually want. Pick one of: exact target (`target_width` × `target_height`), one side (`smaller_side` / `larger_side`), megapixels, or a scale factor. Optional `divisible_by` snaps dimensions to a multiple required by samplers (8, 16, 32, …). `dont_enlarge` blocks upscales when the source is already smaller than the target.

**Use when:** preparing inputs for SDXL / Flux / WAN, batch-resizing photos to a maximum side, or matching a video frame size.

---

#### TS Resolution Selector
<img src="doc/screenshots/ts_resolution_selector.png" alt="TS Resolution Selector" width="450" />

Visual aspect-ratio picker. Choose 1:1, 4:3, 3:2, 16:9, 21:9, 3:4, 2:3, 9:16, 9:21, or a custom ratio, then pick a target megapixel budget (0.5 – 4 MP). The output is a blank canvas with dimensions snapped to multiples of 32 — perfect as a `latent_image` source. If you connect an image, the node fits it onto the canvas; with `original_aspect=True` the ratio is taken from the image instead of the preset.

**Use when:** starting a generation from scratch with a fixed aspect, or normalising an arbitrary image into a latent grid.

---

#### TS Qwen Safe Resize
<img src="doc/screenshots/ts_qwen_safe_resize.png" alt="TS Qwen Safe Resize" width="450" />

One-click resize to the closest official Qwen-Image resolution (1344×1344, 1792×1008, etc.). Picks the supported size with the nearest aspect ratio, then center-crops.

**Use when:** sending images into Qwen-Image / Qwen-Edit pipelines without resolution mismatch errors.

---

#### TS Qwen Canvas
<img src="doc/screenshots/ts_qwen_canvas.png" alt="TS Qwen Canvas" width="450" />

Generates a blank Qwen-Image canvas at one of the supported resolutions and optionally pastes your image into it (with mask-aware cropping if you provide one).

**Use when:** you need a Qwen-friendly canvas size and want to drop a reference image in the middle automatically.

---

#### TS WAN Safe Resize
<img src="doc/screenshots/ts_wan_safe_resize.png" alt="TS WAN Safe Resize" width="450" />

Same idea as Qwen Safe Resize but for WAN-Video. Detects the closest aspect (16:9, 9:16, 1:1) and picks one of three quality presets: Fast (240p), Standard (480p / 832p), High (720p / 1280p). The `interconnection_in/out` string lets several WAN nodes share the same quality tier.

**Use when:** preparing video frames for WAN i2v / t2v models.

---

#### TS Color Grade
<img src="doc/screenshots/ts_color_grade.png" alt="TS Color Grade" width="450" />

Eight-knob colour correction in one node: `hue`, `temperature`, `saturation`, `contrast`, `gain`, `lift`, `gamma`, `brightness`. Comparable to the basic page in DaVinci Resolve.

**Use when:** matching shots, warming up cold renders, fixing flat-looking output, or stylising images.

---

#### TS Color Match
<img src="doc/screenshots/ts_color_match.png" alt="TS Color Match" width="450" />

Transfer the colour palette from a `reference` image to a `target` batch. Two algorithms:

- **MKL** (default) — fast, stable, video-friendly with temporal smoothing.
- **Sinkhorn** — slower but more precise (optimal-transport based).

Includes match masks (`rectangle` / `ellipse` for stabilising on edges only), VRAM-aware chunking, and a `reuse_reference` flag for video.

**Use when:** colour-grading a video to match one keyframe, harmonising shots from different sources, or matching CG into plate footage.

---

#### TS Film Emulation
<img src="doc/screenshots/ts_film_emulation.png" alt="TS Film Emulation" width="450" />

Built-in film stock presets (Kodak Portra/Vision3, Fuji, Cineon-style, …) plus optional `.cube` LUT loading from `models/luts/`. Adds gamma correction, contrast curve and a tunable `lut_strength`.

**Use when:** giving renders a cinematic feel without leaving the graph.

---

#### TS Film Grain
<img src="doc/screenshots/ts_film_grain.png" alt="TS Film Grain" width="450" />

Three-octave organic film grain. Tune `grain_size`, `intensity`, `softness`, and a `mid_tone_grain_bias` for realistic distribution (more grain in midtones than highlights/shadows). `grain_speed` controls how much the grain pattern changes per frame for video.

**Use when:** breaking the "AI-clean" look or matching a film aesthetic.

---

#### TS Remove Background (BiRefNet)
<img src="doc/screenshots/ts_bgrm_birefnet.png" alt="TS Remove Background" width="450" />

State-of-the-art background removal via BiRefNet. Outputs the cut-out image, an alpha mask, and a "mask preview" image. Options: model picker (HR-matting / general / portrait / DIS), `process_resolution` (with `use_custom_resolution` override), `precision` (auto/fp16/fp32), `mask_blur`, `mask_offset`, `invert_output`, `temporal_smooth` for video (`none`/`median3`/`ema` with `ema_alpha`), background mode (Alpha / colour via the COLOR widget). v9.4 cleanup removed the unstable `refine_foreground` option.

**Use when:** isolating subjects, building product shots, or feeding clean alpha masks into compositing nodes.

---

#### TS Keyer
<img src="doc/screenshots/ts_keyer.png" alt="TS Keyer" width="450" />

Professional chroma keyer for green/blue/red screens. Color-difference matte extraction with despill, edge softness, matte gamma, and inversion. Returns RGBA foreground, alpha mask, and a despilled RGB image — ready to composite.

**Use when:** keying actors out of a green screen, removing solid-color backgrounds, or compositing CG.

---

#### TS Despill
<img src="doc/screenshots/ts_despill.png" alt="TS Despill" width="450" />

Standalone despill with four algorithms: `classic`, `balanced`, `adaptive` (edge-aware), and `hue_preserve`. Can take an optional spill mask, has skin-tone protection, and saturation restore. Use after a separate keyer or directly on plate footage that has chroma contamination.

**Use when:** cleaning up green/blue/red spill on hair edges or skin without losing colour fidelity.

---

#### TS Lama Cleanup
<img src="doc/screenshots/ts_lama_cleanup.png" alt="TS Lama Cleanup" width="450" />

Built-in inpainting node powered by LaMa — paint a mask right on the node's canvas (brush + undo/redo + reset), then run to fill. Stores intermediate edits per session, no external Photoshop trip required. Since v9.3 the architecture is pure PyTorch (no upstream `lama-cleaner` dependency) and weights load from `.safetensors` in `models/lama/` instead of pickled `.ckpt`.

**Use when:** removing tourists from photos, erasing watermarks, fixing artifacts, prototyping cleanup before a heavier inpainter.

---

#### TS Matting (ViTMatte)

Guided alpha matting via Hugging Face ViTMatte. Takes an image + a coarse mask (e.g. from SAM3 Detect), auto-builds a trimap and refines into a photo-realistic alpha matte. Same `mask_blur`/`mask_offset`/`background` post-processing contract as TS Remove Background, so it's a drop-in upgrade when edges/hair/transparency matter. Models cached under `models/vitmatte/`.

**Use when:** producing crisp cut-outs from SAM-style masks without dropping into Photoshop.

---

#### TS SAM Media Loader

Loads an image or video and lets you click-pick positive/negative points right on a first-frame preview. Outputs `IMAGE`, `AUDIO` (for video), and `positive_coords`/`negative_coords` STRING JSON in the exact format expected by the native ComfyUI **SAM3 Detect** / **SAM3 Video Track** nodes. With an optional SAM3 `model` input it also returns the rendered `initial_mask` ready to feed into SAM3 Video Track.

**Use when:** building SAM3 segmentation/tracking workflows and you want a friendly UI for the seed points instead of typing JSON by hand.

---

#### TS Crop To Mask
<img src="doc/screenshots/ts_crop_to_mask.png" alt="TS Crop To Mask" width="450" />

Crops a batch of images around a mask region with optional padding, max-resolution clamp, fixed aspect, and inter-frame smoothing for video stability. Outputs both the crop and a `crop_data` blob you can feed into…

---

#### TS Restore From Crop
<img src="doc/screenshots/ts_restore_from_crop.png" alt="TS Restore From Crop" width="450" />

…this node to paste a processed crop back into the original frame, with feathered Gaussian or box blur on the seams. The classic crop-and-restore workflow for processing only the interesting region with a heavy model.

**Use the pair when:** running an upscaler or face restorer on a small ROI of a high-resolution image without burning VRAM on the full frame.

---

#### TS Image Tile Splitter
<img src="doc/screenshots/ts_image_tile_splitter.png" alt="TS Image Tile Splitter" width="450" />

Splits a large image into overlapping tiles for tile-based processing. Configurable tile size, overlap, and feather amount. Outputs the tile batch + a `TILE_INFO` metadata blob.

---

#### TS Image Tile Merger
<img src="doc/screenshots/ts_image_tile_merger.png" alt="TS Image Tile Merger" width="450" />

The other half: takes the processed tile batch and the `TILE_INFO` and stitches them back into one image with proper feathered blending in the overlap regions.

**Use the pair when:** running tile-based upscaling, denoising, or any process that doesn't fit a 4K frame in VRAM.

---

#### TS Auto Tile Size
<img src="doc/screenshots/ts_auto_tile_size.png" alt="TS Auto Tile Size" width="450" />

Pick `tile_count` (4, 8, 16) and the node figures out the best `tile_width` × `tile_height` for an image, respecting padding and a `divide_by` divisor. Pairs naturally with the splitter/merger above.

---

#### TS Cube to Equirectangular
<img src="doc/screenshots/ts_cube_to_equirect.png" alt="TS Cube to Equirectangular" width="450" />

Six cube faces (front/right/back/left/top/bottom) → one equirectangular 360° panorama at the size you choose.

---

#### TS Equirectangular to Cube
<img src="doc/screenshots/ts_equirect_to_cube.png" alt="TS Equirectangular to Cube" width="450" />

The inverse: equirectangular panorama → six cube faces at a chosen `cube_size`.

**Use the pair when:** generating 360° content (Skybox AI, equirect-aware diffusion) and you need to swap formats.

---

#### TS Image Batch Cut
<img src="doc/screenshots/ts_image_batch_cut.png" alt="TS Image Batch Cut" width="450" />

Trim N frames from the start (`first_cut`) and N frames from the end (`last_cut`) of an image batch. Negative values are treated as zero; an over-cut returns an empty batch.

**Use when:** trimming intro/outro frames from a video, dropping the warm-up frames of a sampler, or splitting a batch into segments.

---

#### TS Image Batch to Image List / TS Image List to Image Batch
<table>
<tr>
<td><img src="doc/screenshots/ts_image_batch_to_list.png" alt="Batch to List" width="300" /></td>
<td><img src="doc/screenshots/ts_image_list_to_batch.png" alt="List to Batch" width="300" /></td>
</tr>
</table>

Convert between `IMAGE` (a single batched tensor) and `IMAGE` list (a Python list of single-frame tensors). Needed when one node expects a batch and the next wants per-frame iteration.

---

#### TS Get Image Megapixels
<img src="doc/screenshots/ts_get_image_megapixels.png" alt="TS Get Image Megapixels" width="450" />

Returns the megapixel count of an `IMAGE` as a `FLOAT`. Two-line node, but indispensable for routing logic ("if image > 4 MP, downscale first").

---

#### TS Get Image Size
<img src="doc/screenshots/ts_get_image_size_side.png" alt="TS Get Image Size" width="450" />

Returns the larger or the smaller side of an image as `INT`. Toggle the boolean to switch between the two.

---

#### TS Image Prompt Injector
<img src="doc/screenshots/ts_image_prompt_injector.png" alt="TS Image Prompt Injector" width="450" />

Injects a custom string into the workflow's positive prompt at runtime — useful when you generate prompts dynamically (LLM nodes) and want them to land in the actual `CLIPTextEncode` connected to the sampler. Operates on the workflow graph, leaves the image unchanged.

**Use when:** chaining an LLM that writes prompts and you want the next sampler to use the result without manually rewiring text encoders.

---

<a id="video"></a>
### 🎬 Video (7 nodes)

Frame interpolation, model-based upscale, depth, animation preview, and VRAM hygiene.

#### TS Animation Preview
<img src="doc/screenshots/ts_animation_preview.png" alt="TS Animation Preview" width="450" />

Drop-in preview node for image batches. Renders a looping H.265 video right inside the node with optional audio track sync. Beats running a sampler twice to see your animation.

**Use when:** previewing video output before spending VRAM on the final encode, or QA'ing frame interpolation results.

---

#### TS Frame Interpolation
<img src="doc/screenshots/ts_frame_interpolation.png" alt="TS Frame Interpolation" width="450" />

Smooth frame interpolation using RIFE / FILM models. Boost a 12 fps animation to 24/48/60 fps, or smooth jittery video.

**Use when:** the model output is choppy and you want cinema-smooth motion.

---

#### TS Video Upscale With Model
<img src="doc/screenshots/ts_video_upscale_with_model.png" alt="TS Video Upscale With Model" width="450" />

Per-frame upscaling with any spandrel-loaded model (RealESRGAN, 4x-Ultrasharp, etc.). Three device strategies: `auto`, `load_unload_each_frame` (low VRAM, slower), `keep_loaded` (faster, more VRAM), `cpu_only`.

**Use when:** upscaling video without OOM, or batching upscale jobs with a controllable VRAM footprint.

---

#### TS RTX Upscaler
<img src="doc/screenshots/ts_rtx_upscaler.png" alt="TS RTX Upscaler" width="450" />

Hardware-accelerated upscale via NVIDIA RTX Video Super Resolution (`nvvfx`). Four quality levels (LOW/MEDIUM/HIGH/ULTRA), batched processing. **Requires an RTX GPU.**

**Use when:** you have an RTX card and want speed-of-light upscaling for video.

---

#### TS Video Depth
<img src="doc/screenshots/ts_video_depth.png" alt="TS Video Depth" width="450" />

Depth-Anything-based per-frame depth estimation, optimised for video (temporal consistency). v9.4 brought a full GPU-pipeline overhaul: SDPA attention, TPDF dithering on output, sub-chunk processing for long clips, and a numerically-equivalent DPT tail — same outputs, dramatically faster on RTX cards.

**Use when:** building depth-aware ControlNet pipelines, parallax effects, or 3D reprojection.

---

#### TS LTX First/Last Frame
<img src="doc/screenshots/ts_ltx_first_last_frame.png" alt="TS LTX First/Last Frame" width="450" />

Apply LTX-Video keyframe conditioning for the first and (optionally) last frame in one node — equivalent to chaining two `LTXVAddGuide` nodes, with cleaner UX.

**Use when:** you have specific start/end frames and want LTX to interpolate between them.

---

#### TS Free Video Memory
<img src="doc/screenshots/ts_free_video_memory.png" alt="TS Free Video Memory" width="450" />

A pass-through node that runs `gc.collect()` + `torch.cuda.empty_cache()` (and optionally `caching_allocator_delete_caches()`) between heavy steps. Reports memory before/after.

**Use when:** chaining several VRAM-hungry video nodes and you want explicit memory cleanup between them.

---

<a id="audio"></a>
### 🎵 Audio (5 nodes)

Speech-to-text, text-to-speech, music separation, plus a friendly audio loader and preview.

#### TS Audio Loader
<img src="doc/screenshots/ts_audio_loader.png" alt="TS Audio Loader" width="450" />

The audio loader you'd build yourself if you had time. Loads audio from any media (mp3/wav/mp4/mov/…), shows a real waveform, lets you crop visually by dragging on the waveform, and can even record from the microphone right inside the node. Outputs both the `AUDIO` waveform and a `duration` int.

**Use when:** preparing voiceovers, music beds, or any audio that needs trimming before processing.

---

#### TS Audio Preview
<img src="doc/screenshots/ts_audio_preview.png" alt="TS Audio Preview" width="450" />

Same waveform UI as Audio Loader, but for previewing an audio output from upstream nodes. Looped playback, crop ranges, persistent state.

**Use when:** auditioning the result of a TTS / Stem split / processing chain without saving a file.

---

#### TS Whisper
<img src="doc/screenshots/ts_whisper.png" alt="TS Whisper" width="450" />

Speech-to-text via OpenAI Whisper. Outputs three formats at once: SRT (with timestamps), plain text, and TTML. Configurable beam search, language, temperature fallbacks, and OOM-aware retries.

**Use when:** transcribing voiceovers, generating subtitles, or extracting text from podcasts before LLM processing.

---

#### TS Silero TTS
<img src="doc/screenshots/ts_silero_tts.png" alt="TS Silero TTS" width="450" />

Russian text-to-speech via Silero TTS v5_3. Five speakers (aidar, baya, kseniya, xenia, eugene), text or SSML input, automatic chunking for long texts.

**Use when:** generating Russian voiceovers, audiobook drafts, or YouTube narration.

---

#### TS Music Stems
<img src="doc/screenshots/ts_music_stems.png" alt="TS Music Stems" width="450" />

Demucs-powered music source separation. Splits any audio into four AUDIO outputs: `vocal`, `bass`, `drums`, `others`. Three model options (`htdemucs`, `htdemucs_ft`, `hdemucs_mmi`), TTA shifts and overlap for higher quality.

**Use when:** isolating vocals for remixing, extracting karaoke instrumentals, or feeding cleaner stems into another model.

---

<a id="llm"></a>
### 🤖 LLM (2 nodes)

Multimodal LLM-powered prompt enhancement and image understanding.

#### TS Qwen 3 VL V3
<img src="doc/screenshots/ts_qwen3_vl.png" alt="TS Qwen 3 VL V3" width="450" />

Multimodal Qwen 3 VL (image + video + text) running locally. Built-in model picker (Qwen 2B / 4B / 8B variants and uncensored mods), system-prompt presets ("Image Edit Command Translation", "Prompt Enhancement", …), 4-bit/8-bit quantisation via `bitsandbytes`, FlashAttention-2 support, on-the-fly download from HuggingFace. Since v9.5 the heavy pipeline lives in a shared `nodes/llm/_qwen_engine.py` reused by Super Prompt — bug fixes and perf improvements land in both nodes at once.

**Use when:** describing images for prompts, translating user intents into edit commands, building VLM-driven pipelines.

---

#### TS Super Prompt
<img src="doc/screenshots/ts_super_prompt.png" alt="TS Super Prompt" width="450" />

Prompt enhancement node with a built-in **voice button** — speak your idea, Whisper transcribes it (with cinematography-aware grammar fixes), then a small Qwen3 model expands it into a rich prompt. Optional image input for image-conditioned prompting. Two modes: fast turbo or high-quality. Internals split (v9.5) into `nodes/llm/super_prompt/` (`_helpers`, `_voice`, `_qwen` over the shared Qwen engine) so the prompt-enhancement path stays in sync with TS Qwen 3 VL V3.

**Use when:** quick prompt brainstorming, voice-driven workflows, or bridging a sketchy idea into a production-ready prompt.

---

<a id="text"></a>
### 📝 Text & Prompts (4 nodes)

Build, randomise and manage prompts at scale.

#### TS Prompt Builder
<img src="doc/screenshots/ts_prompt_builder.png" alt="TS Prompt Builder" width="450" />

Composable prompt builder. Edit your prompt as a list of toggle-able blocks (light, camera-angle, lens, film, face, …) backed by `.txt` files in `nodes/prompts/`. Drag handles to reorder, click to enable/disable, and the seed picks one random line from each enabled block. Persists block order/state across sessions.

**Use when:** running batches with controlled prompt variation — every block is a category, every line is a flavour.

---

#### TS Batch Prompt Loader
<img src="doc/screenshots/ts_batch_prompt_loader.png" alt="TS Batch Prompt Loader" width="450" />

Paste a multiline text where prompts are separated by blank lines, get back a list of prompts plus a count.

```
Prompt 1: cat on a windowsill

Prompt 2: dog at the beach

Prompt 3: bird on a branch
```

**Use when:** running a batch of distinct prompts through the same workflow without manually feeding them.

---

#### TS Style Prompt Selector
<img src="doc/screenshots/ts_style_prompt_selector.png" alt="TS Style Prompt Selector" width="450" />

Visual style picker. Pre-baked styles (Photorealistic, Cinematic, Anime, Impressionist, Watercolor, Digital Concept Art, …) with thumbnail previews. Pick one — get the matching prompt fragment as a `STRING`.

**Use when:** quickly stylising a generation without writing the same "in the style of …" string again.

---

#### TS Silero Stress
<img src="doc/screenshots/ts_silero_stress.png" alt="TS Silero Stress" width="450" />

Russian-language text preprocessor: places stress marks (Unicode acute or Silero's `+` notation) and restores `ё` letters. Two algorithms (rule-based accentor + homograph disambiguation neural net) that you can independently toggle.

**Use when:** preparing Russian text for TTS to avoid mispronunciations, or generating educational materials with stress marks.

---

<a id="files"></a>
### 📁 Files & Models (8 nodes)

Tools for managing model files, downloads, EDLs, and inspecting weights.

#### TS Files Downloader (Ultimate)
<img src="doc/screenshots/ts_downloader.png" alt="TS Files Downloader" width="450" />

Multi-file downloader that takes a list of `URL <space> target_path` lines and downloads them sequentially. Auto-replaces HuggingFace mirrors with reachability check across the full mirror list, supports `models/<subdir>` aliases, resumes interrupted downloads, validates archives against zip-slip on auto-unzip, and shows progress (including SHA256 verification). Handy for one-shot pulling all assets a workflow needs.

**Use when:** distributing a workflow that needs N specific models — give users a Files Downloader node pre-filled with the URLs.

---

#### TS Model Scanner
<img src="doc/screenshots/ts_model_scanner.png" alt="TS Model Scanner" width="450" />

Inspect any `.safetensors` (from `models/diffusion_models/`) or a loaded `MODEL` and print a detailed report: every parameter's name, shape, dtype, and device, plus aggregated statistics by dtype.

**Use when:** debugging model loading, checking precision (fp16 vs fp8 vs bf16), or learning what's inside an unfamiliar checkpoint.

---

#### TS Model Converter
<img src="doc/screenshots/ts_model_converter.png" alt="TS Model Converter" width="450" />

In-memory FP8 conversion (`float8_e4m3fn`) of a loaded `MODEL`. Cuts VRAM in half on supported GPUs.

---

#### TS Model Converter Advanced
<img src="doc/screenshots/ts_model_converter_advanced.png" alt="TS Model Converter Advanced" width="450" />

Same idea with finer control: pick the target dtype (fp8 e4m3 / e5m2, bf16, fp16, fp32), keyword filters for which layers to convert, and load/save options.

---

#### TS Model Converter Advanced Direct
<img src="doc/screenshots/ts_model_converter_advanced_direct.png" alt="TS Model Converter Advanced Direct" width="450" />

Same as Advanced but writes the converted weights directly to disk — no in-memory roundtrip needed.

**Use the trio when:** preparing FP8 / mixed-precision variants of large models for slower hardware, or testing precision impact on output quality.

---

#### TS CPU LoRA Merger
<img src="doc/screenshots/ts_cpu_lora_merger.png" alt="TS CPU LoRA Merger" width="450" />

Merge LoRA weights into a base model on CPU — no VRAM needed, suitable for huge models that won't fit on GPU.

**Use when:** baking a LoRA into a checkpoint for distribution, or merging multiple LoRAs without GPU.

---

#### TS File Path Loader
<img src="doc/screenshots/ts_file_path_loader.png" alt="TS File Path Loader" width="450" />

Picks the N-th file from a folder by sorted order. Outputs the full path and the basename without extension. Filters by ComfyUI-supported extensions (`.safetensors`, `.ckpt`, `.pt`, `.mp4`, `.mov`, …). Indices wrap around.

**Use when:** iterating over a folder of inputs in a queue, or grabbing the latest checkpoint by index.

---

#### TS YouTube Chapters
<img src="doc/screenshots/ts_edl_chapters.png" alt="TS YouTube Chapters" width="450" />

Convert a DaVinci Resolve EDL (Edit Decision List) export into a YouTube-friendly chapter list. Reads marker timecodes, normalises to a 1-hour baseline, formats as `MM:SS Marker Name`.

**Use when:** publishing tutorial videos and you've already marked chapters in your editor.

---

<a id="utils"></a>
### 🛠️ Utils (4 nodes)

Tiny helpers that make the graph less cluttered.

#### TS Int Slider
<img src="doc/screenshots/ts_int_slider.png" alt="TS Int Slider" width="450" />

A pure integer slider that returns an `INT`. Custom-widget UI optimised for resolution / count knobs.

---

#### TS Float Slider
<img src="doc/screenshots/ts_float_slider.png" alt="TS Float Slider" width="450" />

The float counterpart, range −1e9 to +1e9 with 0.01 precision by default.

**Use the pair when:** you need a clean parameter widget without dragging a full math node onto the graph.

---

#### TS Math Int
<img src="doc/screenshots/ts_math_int.png" alt="TS Math Int" width="450" />

Two-input integer math: `+`, `-`, `*`, `/`, `//`, `%`, `**`, `min`, `max`. Division by zero returns 0 (logged as an error) instead of crashing the graph.

**Use when:** computing tile counts, frame indices, batch sizes, or any other piece of integer arithmetic that's awkward to express through Primitive nodes.

---

#### TS Smart Switch
<img src="doc/screenshots/ts_smart_switch.png" alt="TS Smart Switch" width="450" />

Type-aware boolean switch between two `ANY` inputs. Pick a `data_type` (images / video / audio / mask / string / int / float) so the node validates that the inputs match it. **Auto-failover**: if the selected input is missing, falls back to the other one — great for optional branches.

**Use when:** branching a workflow on a flag, or making one input optional with a sensible fallback.

---

<a id="conditioning"></a>
### 🎨 Conditioning (1 node)

#### TS Multi Reference
<img src="doc/screenshots/ts_multi_reference.png" alt="TS Multi Reference" width="450" />

Add up to three reference images as `reference_latents` into the conditioning stream. Built for Qwen-Image-Edit and similar multi-reference pipelines. Per-slot output (`image_1` / `image_2` / `image_3`) with `ExecutionBlocker` for unconnected slots, automatic resize to a megapixel budget aligned to a divisor (default 32). Handles RGBA + MASK inputs (composites onto white).

**Use when:** running Qwen-Edit / Flux-with-references style pipelines that accept multiple reference images.

---

## 🔰 Tips for Beginners

### Just starting out?

1. **Search by category** in the right-click menu: every node lives under `TS/<Category>`.
2. **Trust defaults**: every input has a sensible default. Change one parameter at a time to learn what it does.
3. **Use [TS Resolution Selector](#image)** as your latent-image source — it always returns a sampler-friendly size.
4. **Drop a [TS Animation Preview](#video) at the end** of any video graph to QA without re-running.
5. **Need a quick voice prompt?** [TS Super Prompt](#llm) — click the mic, describe your idea, get a polished prompt.

### My VRAM is tight, what should I use?

| Need | Try |
|---|---|
| Upscale a 4K image | TS Image Tile Splitter → upscaler → TS Image Tile Merger |
| Process only the face/object | TS Crop To Mask → upscaler/restorer → TS Restore From Crop |
| Free VRAM mid-graph | TS Free Video Memory between heavy steps |
| FP8 a model | TS Model Converter Advanced |

### Where do model files live?

| Node | Default folder |
|---|---|
| TS Lama Cleanup | `models/lama/` |
| TS Whisper | `models/whisper/` |
| TS Silero TTS | `models/silerotts/` |
| TS Silero Stress | `models/silero-stress/` |
| TS Qwen 3 VL | `models/LLM/` |
| TS Super Prompt | `models/LLM/` |
| TS Music Stems | demucs default cache |

You can override these with `extra_model_paths.yaml` — Timesaver respects ComfyUI's path resolution.

---

## 🛟 Troubleshooting

<details>
<summary><b>"Module not found" on startup</b></summary>

Check the startup log — Timesaver prints a load report. Missing optional dependencies appear under **Optional missing imports** with the file that needs them. Install with:

```bash
python -m pip install <missing_module>
```

Use the same Python ComfyUI runs from. On Windows portable: `python_embeded\python.exe -m pip install <module>`.
</details>

<details>
<summary><b>A node doesn't appear in the menu</b></summary>

Look at the startup log for **Module load issues**. The most common cause is a missing optional dependency — e.g. `py360convert` is required for the cube/equirect nodes. Install it and restart.
</details>

<details>
<summary><b>Workflow fails after updating</b></summary>

Timesaver freezes node ids and inputs across versions on purpose. If something breaks after `git pull`:
1. Check `doc/migration.md` for breaking changes.
2. Make sure `pip install -r requirements.txt` was run.
3. Restart ComfyUI fully — not just refresh the browser tab.
</details>

<details>
<summary><b>OOM (out of memory) errors</b></summary>

- Insert a `TS Free Video Memory` between heavy nodes.
- Reduce `process_resolution` (BiRefNet) or `compute_max_side` (Color Match).
- For upscaling, use `TS Image Tile Splitter` + tiled processing.
- For LLM, drop precision to int8 or int4 (`TS Qwen 3 VL V3` → `precision=int8`).
- Use `unload_after_generation=True` to free model VRAM after each run.
</details>

---

## 🗂️ Repo Layout

```text
comfyui-timesaver/
├─ nodes/                  # 59 node modules, organised by category
├─ js/                     # frontend extensions for DOM-widget nodes
├─ doc/screenshots/        # node screenshots (this README uses them)
├─ requirements.txt        # runtime dependencies
└─ pyproject.toml          # version + ComfyRegistry metadata
```

---

## 📜 License & Credits

Licensed under the terms in [LICENSE.txt](LICENSE.txt).

**Built on top of:**
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — the graph engine and V3 API.
- [BiRefNet](https://github.com/zhengpeng7/BiRefNet) — background removal.
- [LaMa](https://github.com/advimman/lama) — image inpainting.
- [Whisper](https://github.com/openai/whisper) — speech recognition.
- [Demucs](https://github.com/facebookresearch/demucs) — music source separation.
- [Silero](https://github.com/snakers4/silero-models) — Russian TTS / stress.
- [Qwen](https://github.com/QwenLM/Qwen3-VL) — vision-language model.
- [Spandrel](https://github.com/chaiNNer-org/spandrel) — model loading for upscalers.
- [py360convert](https://github.com/sunset1995/py360convert) — 360° conversions.
- [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) / [FILM](https://github.com/google-research/frame-interpolation) — frame interpolation.

**Maintainer:** [@AlexYez](https://github.com/AlexYez)

**Issues / feature requests:** https://github.com/AlexYez/comfyui-timesaver/issues

---

<div align="center">

**Found this useful?** ⭐ Star the repo to help others find it.

</div>
