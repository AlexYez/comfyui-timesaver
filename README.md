# ComfyUI Timesaver Nodes

[English](README.md) | [Russian](README.ru.md)

A complete, friendly guide for **all nodes currently in this pack**. Each node card is written in plain language, with key controls and screenshot placeholders.

Repository: https://github.com/AlexYez/comfyui-timesaver

## Installation

1. Place this folder in `ComfyUI/custom_nodes/comfyui-timesaver`.
2. Install dependencies from `requirements.txt` if needed.
3. Restart ComfyUI.

## Project Structure

```text
comfyui-timesaver/
├─ nodes/                     # All node modules and node-related assets
│  ├─ *.py                    # Node implementations
│  ├─ luts/                   # LUT files used by color/film nodes
│  ├─ prompts/                # Prompt Builder text blocks and config
│  ├─ styles/                 # Style Prompt Selector assets
│  ├─ video_depth_anything/   # Video depth model package
│  └─ qwen_3_vl_presets.json  # Qwen system preset definitions
├─ js/                        # Frontend extensions/widgets
├─ doc/                       # Internal technical docs and tooling
├─ requirements.txt
├─ pyproject.toml
└─ __init__.py                # Loader + startup audit table
```

## Highlights

- Total documented nodes: **53**
- Collapsible cards for every node
- Screenshot placeholders for each node
- Beginner-friendly wording + technical references

## Node Catalog

| Node ID | Purpose | Category | Output Types |
| --- | --- | --- | --- |
| `TS_Qwen3_VL_V3` | Main multimodal Qwen node (text + image/video) with presets, precision, and offline controls. | `TS/LLM` | `STRING, IMAGE` |
| `TSWhisper` | Whisper transcription/translation node with SRT + plain text output. | `TS/Audio` | `STRING, STRING, STRING` |
| `TS_VoiceRecognition` | Browser microphone recorder that inserts Whisper-recognized speech into a text field. | `TS/audio` | `STRING` |
| `TS_SileroTTS` | Russian TTS node based on Silero with chunking and AUDIO output. | `TS/audio` | `AUDIO` |
| `TS_MusicStems` | Splits music into stems (vocals, bass, drums, others, instrumental). | `TS/Audio` | `AUDIO, AUDIO, AUDIO, AUDIO, AUDIO` |
| `TS_PromptBuilder` | Builds structured prompts from JSON config + seed for reproducible prompt variations. | `TS/Prompt` | `STRING` |
| `TS_BatchPromptLoader` | Reads multiline prompts and outputs one prompt per index/step. | `utils/text` | `STRING, INT` |
| `TS_StylePromptSelector` | Loads style prompt text from style library by ID or name. | `TS/Prompt` | `STRING` |
| `TS_ImageResize` | Flexible resize node for exact size, side-based scaling, scale factor, or megapixel targets. | `image` | `IMAGE, INT, INT, MASK` |
| `TS_QwenSafeResize` | Safe resize preset optimized for Qwen image preprocessing constraints. | `image/resize` | `IMAGE` |
| `TS_WAN_SafeResize` | Safe resize helper for WAN pipelines with model-friendly output sizing. | `image/resize` | `IMAGE, INT, INT, STRING` |
| `TS_QwenCanvas` | Creates a Qwen-friendly canvas resolution and optionally places image/mask into it. | `TS Qwen` | `IMAGE, INT, INT` |
| `TS_ResolutionSelector` | Chooses target resolution by aspect presets/custom ratio and can output a prepared canvas image. | `TS/Resolution` | `IMAGE` |
| `TS_Color_Grade` | Fast primary color grading: hue, temperature, saturation, contrast, gamma, and tone controls. | `TS/Color` | `IMAGE` |
| `TS_Film_Emulation` | Film-like look node with presets, LUT support, warmth, fade, and grain controls. | `Image/Color` | `IMAGE` |
| `TS_FilmGrain` | Adds controllable film grain with size, intensity, color, and motion behavior. | `Image Adjustments/Grain` | `IMAGE` |
| `TS_Color_Match` | Transfers color mood from a reference image to a target image while keeping structure intact. | `TS/Color` | `IMAGE` |
| `TS_Keyer` | Advanced color-difference keyer for green/blue/red screens with alpha and despill outputs. | `TS/image` | `IMAGE, MASK, IMAGE` |
| `TS_Despill` | Professional spill suppression node with classic, balanced, adaptive, and hue-preserving algorithms. | `TS/image` | `IMAGE, MASK, IMAGE` |
| `TS_BGRM_BiRefNet` | AI background removal with BiRefNet. Great for instant cutouts and transparent composites. | `Timesaver/Image Tools` | `IMAGE, MASK, IMAGE` |
| `TSCropToMask` | Crops image content around a mask to speed up local edits and save memory. | `image/processing` | `IMAGE, MASK, CROP_DATA, INT, INT` |
| `TSRestoreFromCrop` | Places a processed crop back into the original frame with optional blending. | `image/processing` | `IMAGE` |
| `TS_ImageBatchToImageList` | Converts a batched IMAGE tensor into per-image list-style flow. | `TS/Image Tools` | `IMAGE` |
| `TS_ImageListToImageBatch` | Combines list-style image flow back into a batched IMAGE tensor. | `TS/Image Tools` | `IMAGE` |
| `TS_ImageBatchCut` | Cuts frames from the beginning/end of an image batch. | `TS/Image Tools` | `IMAGE` |
| `TS_GetImageMegapixels` | Returns megapixel value for quick quality/performance checks. | `TS/Image Tools` | `FLOAT` |
| `TS_GetImageSizeSide` | Returns selected image side size for control logic or auto-configuration. | `TS/Image Tools` | `INT` |
| `TS_ImagePromptInjector` | Injects prompt text into image flow so context can travel with the image branch. | `TS/Image Tools` | `IMAGE` |
| `TS_ImageTileSplitter` | Splits an image into overlapping tiles for heavy processing at higher quality. | `TS/Image Tools` | `IMAGE, TILE_INFO` |
| `TS_ImageTileMerger` | Merges tiles back into a full image using tile metadata and blending. | `TS/Image Tools` | `IMAGE` |
| `TSAutoTileSize` | Automatically calculates tile width/height for tiled workflows based on your target grid. | `utils/Tile Size` | `INT, INT` |
| `TS Cube to Equirectangular` | Converts six cube faces into one equirectangular 360 panorama. | `Tools/TS_Image` | `IMAGE` |
| `TS Equirectangular to Cube` | Converts a 360 panorama into six cube faces for editing or projection workflows. | `Tools/TS_Image` | `IMAGE, IMAGE, IMAGE, IMAGE, IMAGE, IMAGE` |
| `TS_VideoDepthNode` | Estimates depth maps from frame sequences for compositing, relighting, and depth effects. | `Tools/Video` | `IMAGE` |
| `TS_Video_Upscale_With_Model` | Upscales full image sequences using loaded upscaler model with memory strategies. | `video` | `IMAGE` |
| `TS_RTX_Upscaler` | NVIDIA RTX upscaler node for fast quality upscaling on supported systems. | `TS/Upscaling` | `IMAGE` |
| `TS_DeflickerNode` | Reduces temporal brightness/color flicker in image sequences. | `Video PostProcessing` | `IMAGE` |
| `TS_Free_Video_Memory` | Pass-through node that aggressively frees RAM/VRAM between heavy video steps. | `video` | `IMAGE` |
| `TS_LTX_FirstLastFrame` | Injects first/last frame guidance into latent workflows (useful for LTX video control). | `conditioning/video_models` | `LATENT` |
| `TS_Animation_Preview` | Creates a quick animation preview from image frames, with optional audio merge. | `TS/Interface Tools` | `side effects / UI` |
| `TS_FileBrowser` | In-node media picker that loads image/video/audio/mask from disk into your graph. | `TS/Input` | `IMAGE, VIDEO, AUDIO, MASK, STRING` |
| `TS_FilePathLoader` | Returns file path and file name by index from a folder list. | `file_utils` | `STRING, STRING` |
| `TS Files Downloader` | Bulk downloader for models/assets with resume, mirrors, proxy support, and optional unzip. | `Tools/TS_IO` | `side effects / UI` |
| `TS Youtube Chapters` | Converts EDL timing into ready-to-paste YouTube chapter timestamps. | `Tools/TS_Video` | `STRING` |
| `TS_ModelScanner` | Scans model files and returns a human-readable structure/metadata summary. | `utils/model_analysis` | `STRING` |
| `TS_ModelConverter` | One-click converter for model precision conversion workflows. | `conversion` | `MODEL` |
| `TS_ModelConverterAdvanced` | Advanced model converter with more control over format, preset, and output behavior. | `Model Conversion` | `STRING` |
| `TS_ModelConverterAdvancedDirect` | Advanced converter variant that works directly from connected MODEL input. | `TS/Model Conversion` | `STRING` |
| `TS_CPULoraMerger` | Merges up to four LoRA files into a base model on CPU and saves a new safetensors file. | `TS/Model Tools` | `STRING, STRING` |
| `TS_FloatSlider` | Simple UI float slider node for clean graph parameter control. | `TS Tools/Sliders` | `FLOAT` |
| `TS_Int_Slider` | Simple UI integer slider node for deterministic integer parameters. | `TS Tools/Sliders` | `INT` |
| `TS_Smart_Switch` | Switches between two inputs by mode and keeps your graph compact. | `TS Tools/Logic` | `*` |
| `TS_Math_Int` | Integer math helper for counters, offsets, and simple graph logic. | `TS/Math` | `INT` |

## Detailed Node Cards

### AI, Audio & Language

<details>
<summary><strong>TS_Qwen3_VL_V3</strong> - Main multimodal Qwen node (text + image/video) with presets, precision, and offline controls.</summary>

![Screenshot placeholder for TS_Qwen3_VL_V3](docs/img/placeholders/ts-qwen3-vl-v3.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Main multimodal Qwen node (text + image/video) with presets, precision, and offline controls.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `model_name`, `custom_model_id`, `hf_token`, `system_preset`, `prompt`, `seed`, `max_new_tokens`, `precision`, `attention_mode`, `offline_mode`, `unload_after_generation`, `enable`, `max_image_size`, `video_max_frames`
- Optional: `image`, `video`, `custom_system_prompt`

**Outputs**
- `STRING`, `IMAGE`

**Technical info**
- Internal id: `TS_Qwen3_VL_V3`
- Class: `TS_Qwen3_VL_V3`
- File: `nodes/ts_qwen3_vl_v3_node.py`
- Category: `TS/LLM`
- Function: `process`
- Dependency note: Uses `transformers` plus optional acceleration libraries.

</details>

<details>
<summary><strong>TSWhisper</strong> - Whisper transcription/translation node with SRT + plain text output.</summary>

![Screenshot placeholder for TSWhisper](docs/img/placeholders/tswhisper.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Whisper transcription/translation node with SRT + plain text output.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `audio`, `model`, `output_filename_prefix`, `task`, `source_language`, `timestamps`, `save_srt_file`, `precision`
- Optional: `output_dir`

**Outputs**
- `STRING`, `STRING`, `STRING`

**Technical info**
- Internal id: `TSWhisper`
- Class: `TSWhisper`
- File: `nodes/ts_whisper_node.py`
- Category: `TS/Audio`
- Function: `generate_srt_and_text`
- Dependency note: Uses `transformers` and `torchaudio`.

</details>

<details>
<summary><strong>TS_VoiceRecognition</strong> - Browser microphone recorder that inserts Whisper-recognized speech into a text field.</summary>

![Screenshot placeholder for TS_VoiceRecognition](docs/img/placeholders/ts-voice-recognition.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Records speech from the browser microphone, sends it to the local ComfyUI backend, recognizes it with openai-whisper, and inserts the text at the cursor position.

**Quick usage**
1. Add the node from `TS/audio`.
2. Install the updated dependencies, then restart ComfyUI.
3. Click the node button to download the configured Whisper model, then record and stop.

**Main controls**
- Required: `text`, `translate_to_english`
- Optional: *(none)*

**Outputs**
- `STRING`

**Technical info**
- Internal id: `TS_VoiceRecognition`
- Class: `TS_VoiceRecognition`
- File: `nodes/ts_voice_recognition_node.py`
- Category: `TS/audio`
- Function: `execute`
- Dependency note: Uses `openai-whisper`, `torch`, `numpy`, and ffmpeg/imageio-ffmpeg.

</details>

<details>
<summary><strong>TS_SileroTTS</strong> - Russian TTS node based on Silero with chunking and AUDIO output.</summary>

![Screenshot placeholder for TS_SileroTTS](docs/img/placeholders/ts-silerotts.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Russian TTS node based on Silero with chunking and AUDIO output.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `text`, `input_format`, `speaker`, `run_device`, `enable_chunking`, `max_chunk_chars`, `chunk_pause_ms`, `put_accent`, `put_yo`, `put_stress_homo`, `put_yo_homo`
- Optional: *(none)*

**Outputs**
- `AUDIO`

**Technical info**
- Internal id: `TS_SileroTTS`
- Class: `TS_SileroTTS`
- File: `nodes/ts_silero_tts_node.py`
- Category: `TS/audio`
- Function: `execute`

</details>

<details>
<summary><strong>TS_MusicStems</strong> - Splits music into stems (vocals, bass, drums, others, instrumental).</summary>

![Screenshot placeholder for TS_MusicStems](docs/img/placeholders/ts-musicstems.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Splits music into stems (vocals, bass, drums, others, instrumental).

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `audio`, `model_name`, `device`, `shifts`, `overlap`, `jobs`
- Optional: *(none)*

**Outputs**
- `AUDIO`, `AUDIO`, `AUDIO`, `AUDIO`, `AUDIO`

**Technical info**
- Internal id: `TS_MusicStems`
- Class: `TS_MusicStems`
- File: `nodes/ts_music_stems_node.py`
- Category: `TS/Audio`
- Function: `process_stems`
- Dependency note: Requires `demucs`.

</details>

<details>
<summary><strong>TS_PromptBuilder</strong> - Builds structured prompts from JSON config + seed for reproducible prompt variations.</summary>

![Screenshot placeholder for TS_PromptBuilder](docs/img/placeholders/ts-promptbuilder.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Builds structured prompts from JSON config + seed for reproducible prompt variations.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `seed`, `config_json`
- Optional: *(none)*

**Outputs**
- `STRING`

**Technical info**
- Internal id: `TS_PromptBuilder`
- Class: `TS_PromptBuilder`
- File: `nodes/ts_prompt_builder_node.py`
- Category: `TS/Prompt`
- Function: `build_prompt`

</details>

<details>
<summary><strong>TS_BatchPromptLoader</strong> - Reads multiline prompts and outputs one prompt per index/step.</summary>

![Screenshot placeholder for TS_BatchPromptLoader](docs/img/placeholders/ts-batchpromptloader.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Reads multiline prompts and outputs one prompt per index/step.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `text`
- Optional: *(none)*

**Outputs**
- `STRING`, `INT`

**Technical info**
- Internal id: `TS_BatchPromptLoader`
- Class: `TS_BatchPromptLoader`
- File: `nodes/ts_text_tools_node.py`
- Category: `utils/text`
- Function: `process_prompts`

</details>

<details>
<summary><strong>TS_StylePromptSelector</strong> - Loads style prompt text from style library by ID or name.</summary>

![Screenshot placeholder for TS_StylePromptSelector](docs/img/placeholders/ts-stylepromptselector.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Loads style prompt text from style library by ID or name.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `style_id`
- Optional: *(none)*

**Outputs**
- `STRING`

**Technical info**
- Internal id: `TS_StylePromptSelector`
- Class: `TS_StylePromptSelector`
- File: `nodes/ts_style_prompt_node.py`
- Category: `TS/Prompt`
- Function: `get_prompt`

</details>

### Image Processing

<details>
<summary><strong>TS_ImageResize</strong> - Flexible resize node for exact size, side-based scaling, scale factor, or megapixel targets.</summary>

![Screenshot placeholder for TS_ImageResize](docs/img/placeholders/ts-imageresize.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Flexible resize node for exact size, side-based scaling, scale factor, or megapixel targets.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `pixels`, `target_width`, `target_height`, `smaller_side`, `larger_side`, `scale_factor`, `keep_proportion`, `upscale_method`, `divisible_by`, `megapixels`, `dont_enlarge`
- Optional: `mask`

**Outputs**
- `IMAGE`, `INT`, `INT`, `MASK`

**Technical info**
- Internal id: `TS_ImageResize`
- Class: `TS_ImageResize`
- File: `nodes/ts_image_resize_node.py`
- Category: `image`
- Function: `resize`

</details>

<details>
<summary><strong>TS_QwenSafeResize</strong> - Safe resize preset optimized for Qwen image preprocessing constraints.</summary>

![Screenshot placeholder for TS_QwenSafeResize](docs/img/placeholders/ts-qwensaferesize.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Safe resize preset optimized for Qwen image preprocessing constraints.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_QwenSafeResize`
- Class: `TS_QwenSafeResize`
- File: `nodes/ts_image_resize_node.py`
- Category: `image/resize`
- Function: `safe_resize`

</details>

<details>
<summary><strong>TS_WAN_SafeResize</strong> - Safe resize helper for WAN pipelines with model-friendly output sizing.</summary>

![Screenshot placeholder for TS_WAN_SafeResize](docs/img/placeholders/ts-wan-saferesize.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Safe resize helper for WAN pipelines with model-friendly output sizing.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`, `quality`
- Optional: `interconnection_in`

**Outputs**
- `IMAGE`, `INT`, `INT`, `STRING`

**Technical info**
- Internal id: `TS_WAN_SafeResize`
- Class: `TS_WAN_SafeResize`
- File: `nodes/ts_image_resize_node.py`
- Category: `image/resize`
- Function: `safe_resize`

</details>

<details>
<summary><strong>TS_QwenCanvas</strong> - Creates a Qwen-friendly canvas resolution and optionally places image/mask into it.</summary>

![Screenshot placeholder for TS_QwenCanvas](docs/img/placeholders/ts-qwencanvas.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Creates a Qwen-friendly canvas resolution and optionally places image/mask into it.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `resolution`
- Optional: `image`, `mask`

**Outputs**
- `IMAGE`, `INT`, `INT`

**Technical info**
- Internal id: `TS_QwenCanvas`
- Class: `TS_QwenCanvas`
- File: `nodes/ts_image_resize_node.py`
- Category: `TS Qwen`
- Function: `make_canvas`

</details>

<details>
<summary><strong>TS_ResolutionSelector</strong> - Chooses target resolution by aspect presets/custom ratio and can output a prepared canvas image.</summary>

![Screenshot placeholder for TS_ResolutionSelector](docs/img/placeholders/ts-resolutionselector.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Chooses target resolution by aspect presets/custom ratio and can output a prepared canvas image.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `aspect_ratio`, `resolution`, `custom_ratio`, `original_aspect`
- Optional: `image`

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_ResolutionSelector`
- Class: `TS_ResolutionSelector`
- File: `nodes/ts_resolution_selector.py`
- Category: `TS/Resolution`
- Function: `select_resolution`

</details>

<details>
<summary><strong>TS_Color_Grade</strong> - Fast primary color grading: hue, temperature, saturation, contrast, gamma, and tone controls.</summary>

![Screenshot placeholder for TS_Color_Grade](docs/img/placeholders/ts-color-grade.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Fast primary color grading: hue, temperature, saturation, contrast, gamma, and tone controls.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`, `hue`, `temperature`, `saturation`, `contrast`, `gain`, `lift`, `gamma`, `brightness`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_Color_Grade`
- Class: `TS_Color_Grade`
- File: `nodes/ts_color_node.py`
- Category: `TS/Color`
- Function: `process`

</details>

<details>
<summary><strong>TS_Film_Emulation</strong> - Film-like look node with presets, LUT support, warmth, fade, and grain controls.</summary>

![Screenshot placeholder for TS_Film_Emulation](docs/img/placeholders/ts-film-emulation.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Film-like look node with presets, LUT support, warmth, fade, and grain controls.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`, `enable`, `film_preset`, `lut_choice`, `lut_strength`, `gamma_correction`, `film_strength`, `contrast_curve`, `warmth`, `grain_intensity`, `grain_size`, `fade`, `shadow_saturation`, `highlight_saturation`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_Film_Emulation`
- Class: `TS_Film_Emulation`
- File: `nodes/ts_color_node.py`
- Category: `Image/Color`
- Function: `process`

</details>

<details>
<summary><strong>TS_FilmGrain</strong> - Adds controllable film grain with size, intensity, color, and motion behavior.</summary>

![Screenshot placeholder for TS_FilmGrain](docs/img/placeholders/ts-filmgrain.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Adds controllable film grain with size, intensity, color, and motion behavior.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `images`, `force_gpu`, `grain_size`, `grain_intensity`, `grain_speed`, `grain_softness`, `color_grain_strength`, `mid_tone_grain_bias`, `seed`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_FilmGrain`
- Class: `TS_FilmGrain`
- File: `nodes/ts_film_grain_node.py`
- Category: `Image Adjustments/Grain`
- Function: `apply_grain`

</details>

<details>
<summary><strong>TS_Color_Match</strong> - Transfers color mood from a reference image to a target image while keeping structure intact.</summary>

![Screenshot placeholder for TS_Color_Match](docs/img/placeholders/ts-color-match.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Transfers color mood from a reference image to a target image while keeping structure intact.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `reference`, `target`, `mode`, `device`, `strength`, `enable`, `match_mask`, `mask_size`, `compute_max_side`, `mkl_sample_points`, `sinkhorn_max_points`, `reuse_reference`, `chunk_size`, `logging`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_Color_Match`
- Class: `TS_Color_Match`
- File: `nodes/ts_color_match_node.py`
- Category: `TS/Color`
- Function: `process`

</details>

<details>
<summary><strong>TS_Keyer</strong> - Advanced color-difference keyer for green/blue/red screens with alpha and despill outputs.</summary>

![Screenshot placeholder for TS_Keyer](docs/img/placeholders/ts-keyer.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Advanced color-difference keyer for green/blue/red screens with alpha and despill outputs.

**Quick usage**
1. Connect your keyed footage to `image`.
2. Set `key_color` and keep `key_channel=auto` for first pass.
3. Refine matte with black/white points, then tune despill settings.

**Main controls**
- Required: `image`, `enable`, `key_color`, `key_channel`, `screen_balance`, `key_strength`, `black_point`, `white_point`, `matte_gamma`, `matte_preblur`, `edge_softness`, `despill_strength`, `despill_edge_only`, `despill_compensate`, `invert_alpha`
- Optional: *(none)*

**Outputs**
- `IMAGE, MASK, IMAGE`

**Technical info**
- Internal id: `TS_Keyer`
- Class: `TS_Keyer`
- File: `nodes/ts_keyer_node.py`
- Category: `TS/image`
- Function: `execute`

</details>

<details>
<summary><strong>TS_Despill</strong> - Professional spill suppression node with classic, balanced, adaptive, and hue-preserving algorithms.</summary>

![Screenshot placeholder for TS_Despill](docs/img/placeholders/ts-despill.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Professional spill suppression node with classic, balanced, adaptive, and hue-preserving algorithms.

**Quick usage**
1. Connect keyed or semi-keyed footage to `image`.
2. Pick `screen_color` and start with `algorithm=adaptive`.
3. Use `strength`, `spill_threshold`, and optional `spill_mask` to localize cleanup.

**Main controls**
- Required: `image`, `enable`, `screen_color`, `algorithm`, `strength`, `spill_threshold`, `spill_softness`, `compensation`, `preserve_luma`, `use_input_alpha_for_edges`, `edge_boost`, `edge_blur`, `skin_protection`, `saturation_restore`, `invert_spill_mask`
- Optional: `spill_mask`

**Outputs**
- `IMAGE, MASK, IMAGE`

**Technical info**
- Internal id: `TS_Despill`
- Class: `TS_Despill`
- File: `nodes/ts_keyer_node.py`
- Category: `TS/image`
- Function: `execute`

</details>

<details>
<summary><strong>TS_BGRM_BiRefNet</strong> - AI background removal with BiRefNet. Great for instant cutouts and transparent composites.</summary>

![Screenshot placeholder for TS_BGRM_BiRefNet](docs/img/placeholders/ts-bgrm-birefnet.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
AI background removal with BiRefNet. Great for instant cutouts and transparent composites.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`, `enable`, `model`
- Optional: `use_custom_resolution`, `process_resolution`, `mask_blur`, `mask_offset`, `invert_output`, `refine_foreground`, `background`, `background_color`

**Outputs**
- `IMAGE`, `MASK`, `IMAGE`

**Technical info**
- Internal id: `TS_BGRM_BiRefNet`
- Class: `TS_BGRM_BiRefNet`
- File: `nodes/ts_bgrm_node.py`
- Category: `Timesaver/Image Tools`
- Function: `process_image`
- Dependency note: Requires BiRefNet model files (auto-download when available).

</details>

<details>
<summary><strong>TSCropToMask</strong> - Crops image content around a mask to speed up local edits and save memory.</summary>

![Screenshot placeholder for TSCropToMask](docs/img/placeholders/tscroptomask.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Crops image content around a mask to speed up local edits and save memory.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `images`, `mask`, `padding`, `divide_by`, `max_resolution`, `fixed_mask_frame_index`, `interpolation_window_size`, `force_gpu`, `fixed_crop_size`, `fixed_width`, `fixed_height`
- Optional: *(none)*

**Outputs**
- `IMAGE`, `MASK`, `CROP_DATA`, `INT`, `INT`

**Technical info**
- Internal id: `TSCropToMask`
- Class: `TSCropToMask`
- File: `nodes/ts_crop_to_mask_node.py`
- Category: `image/processing`
- Function: `crop`

</details>

<details>
<summary><strong>TSRestoreFromCrop</strong> - Places a processed crop back into the original frame with optional blending.</summary>

![Screenshot placeholder for TSRestoreFromCrop](docs/img/placeholders/tsrestorefromcrop.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Places a processed crop back into the original frame with optional blending.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `original_images`, `cropped_images`, `crop_data`, `blur`, `blur_type`, `force_gpu`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TSRestoreFromCrop`
- Class: `TSRestoreFromCrop`
- File: `nodes/ts_crop_to_mask_node.py`
- Category: `image/processing`
- Function: `restore`

</details>

<details>
<summary><strong>TS_ImageBatchToImageList</strong> - Converts a batched IMAGE tensor into per-image list-style flow.</summary>

![Screenshot placeholder for TS_ImageBatchToImageList](docs/img/placeholders/ts-imagebatchtoimagelist.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Converts a batched IMAGE tensor into per-image list-style flow.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_ImageBatchToImageList`
- Class: `TS_ImageBatchToImageList`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_ImageListToImageBatch</strong> - Combines list-style image flow back into a batched IMAGE tensor.</summary>

![Screenshot placeholder for TS_ImageListToImageBatch](docs/img/placeholders/ts-imagelisttoimagebatch.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Combines list-style image flow back into a batched IMAGE tensor.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `images`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_ImageListToImageBatch`
- Class: `TS_ImageListToImageBatch`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_ImageBatchCut</strong> - Cuts frames from the beginning/end of an image batch.</summary>

![Screenshot placeholder for TS_ImageBatchCut](docs/img/placeholders/ts-imagebatchcut.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Cuts frames from the beginning/end of an image batch.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`, `first_cut`, `last_cut`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_ImageBatchCut`
- Class: `TS_ImageBatchCut`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_GetImageMegapixels</strong> - Returns megapixel value for quick quality/performance checks.</summary>

![Screenshot placeholder for TS_GetImageMegapixels](docs/img/placeholders/ts-getimagemegapixels.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Returns megapixel value for quick quality/performance checks.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`
- Optional: *(none)*

**Outputs**
- `FLOAT`

**Technical info**
- Internal id: `TS_GetImageMegapixels`
- Class: `TS_GetImageMegapixels`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_GetImageSizeSide</strong> - Returns selected image side size for control logic or auto-configuration.</summary>

![Screenshot placeholder for TS_GetImageSizeSide](docs/img/placeholders/ts-getimagesizeside.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Returns selected image side size for control logic or auto-configuration.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`, `large_side`
- Optional: *(none)*

**Outputs**
- `INT`

**Technical info**
- Internal id: `TS_GetImageSizeSide`
- Class: `TS_GetImageSizeSide`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_ImagePromptInjector</strong> - Injects prompt text into image flow so context can travel with the image branch.</summary>

![Screenshot placeholder for TS_ImagePromptInjector](docs/img/placeholders/ts-imagepromptinjector.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Injects prompt text into image flow so context can travel with the image branch.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`, `prompt`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_ImagePromptInjector`
- Class: `TS_ImagePromptInjector`
- File: `nodes/ts_image_tools_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_ImageTileSplitter</strong> - Splits an image into overlapping tiles for heavy processing at higher quality.</summary>

![Screenshot placeholder for TS_ImageTileSplitter](docs/img/placeholders/ts-imagetilesplitter.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Splits an image into overlapping tiles for heavy processing at higher quality.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`, `tile_width`, `tile_height`, `overlap`, `feather`
- Optional: *(none)*

**Outputs**
- `IMAGE`, `TILE_INFO`

**Technical info**
- Internal id: `TS_ImageTileSplitter`
- Class: `TS_ImageTileSplitter`
- File: `nodes/ts_image_tile_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TS_ImageTileMerger</strong> - Merges tiles back into a full image using tile metadata and blending.</summary>

![Screenshot placeholder for TS_ImageTileMerger](docs/img/placeholders/ts-imagetilemerger.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Merges tiles back into a full image using tile metadata and blending.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `images`, `tile_data`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_ImageTileMerger`
- Class: `TS_ImageTileMerger`
- File: `nodes/ts_image_tile_node.py`
- Category: `TS/Image Tools`
- Function: `execute`

</details>

<details>
<summary><strong>TSAutoTileSize</strong> - Automatically calculates tile width/height for tiled workflows based on your target grid.</summary>

![Screenshot placeholder for TSAutoTileSize](docs/img/placeholders/tsautotilesize.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Automatically calculates tile width/height for tiled workflows based on your target grid.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `tile_count`, `padding`, `divide_by`
- Optional: `image`, `width`, `height`

**Outputs**
- `INT`, `INT`

**Technical info**
- Internal id: `TSAutoTileSize`
- Class: `TSAutoTileSize`
- File: `nodes/ts_image_resize_node.py`
- Category: `utils/Tile Size`
- Function: `calculate_grid`

</details>

<details>
<summary><strong>TS Cube to Equirectangular</strong> - Converts six cube faces into one equirectangular 360 panorama.</summary>

![Screenshot placeholder for TS Cube to Equirectangular](docs/img/placeholders/ts-cube-to-equirectangular.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Converts six cube faces into one equirectangular 360 panorama.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `front`, `right`, `back`, `left`, `top`, `bottom`, `output_width`, `output_height`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS Cube to Equirectangular`
- Class: `TS_CubemapFacesToEquirectangularNode`
- File: `nodes/ts_cube_to_equirect_node.py`
- Category: `Tools/TS_Image`
- Function: `convert`
- Dependency note: Requires `py360convert`.

</details>

<details>
<summary><strong>TS Equirectangular to Cube</strong> - Converts a 360 panorama into six cube faces for editing or projection workflows.</summary>

![Screenshot placeholder for TS Equirectangular to Cube](docs/img/placeholders/ts-equirectangular-to-cube.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Converts a 360 panorama into six cube faces for editing or projection workflows.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `image`, `cube_size`
- Optional: *(none)*

**Outputs**
- `IMAGE`, `IMAGE`, `IMAGE`, `IMAGE`, `IMAGE`, `IMAGE`

**Technical info**
- Internal id: `TS Equirectangular to Cube`
- Class: `TS_EquirectangularToCubemapFacesNode`
- File: `nodes/ts_equirect_to_cube_node.py`
- Category: `Tools/TS_Image`
- Function: `convert`
- Dependency note: Requires `py360convert`.

</details>

### Video Workflows

<details>
<summary><strong>TS_VideoDepthNode</strong> - Estimates depth maps from frame sequences for compositing, relighting, and depth effects.</summary>

![Screenshot placeholder for TS_VideoDepthNode](docs/img/placeholders/ts-videodepthnode.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Estimates depth maps from frame sequences for compositing, relighting, and depth effects.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `images`, `model_filename`, `input_size`, `max_res`, `precision`, `colormap`, `dithering_strength`, `apply_median_blur`, `upscale_algorithm`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_VideoDepthNode`
- Class: `TS_VideoDepth`
- File: `nodes/ts_video_depth_node.py`
- Category: `Tools/Video`
- Function: `execute_process_unified`

</details>

<details>
<summary><strong>TS_Video_Upscale_With_Model</strong> - Upscales full image sequences using loaded upscaler model with memory strategies.</summary>

![Screenshot placeholder for TS_Video_Upscale_With_Model](docs/img/placeholders/ts-video-upscale-with-model.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Upscales full image sequences using loaded upscaler model with memory strategies.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `model_name`, `images`, `upscale_method`, `factor`, `device_strategy`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_Video_Upscale_With_Model`
- Class: `TS_Video_Upscale_With_Model`
- File: `nodes/ts_video_upscale_node.py`
- Category: `video`
- Function: `upscale_video`
- Dependency note: Requires `spandrel` for model loading.

</details>

<details>
<summary><strong>TS_RTX_Upscaler</strong> - NVIDIA RTX upscaler node for fast quality upscaling on supported systems.</summary>

![Screenshot placeholder for TS_RTX_Upscaler](docs/img/placeholders/ts-rtx-upscaler.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
NVIDIA RTX upscaler node for fast quality upscaling on supported systems.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `images`, `resize_type`, `scale`, `width`, `height`, `quality`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_RTX_Upscaler`
- Class: `TS_RTX_Upscaler`
- File: `nodes/ts_rtx_upscaler_node.py`
- Category: `TS/Upscaling`
- Function: `upscale`
- Dependency note: Requires RTX/VFX runtime components in your environment.

</details>

<details>
<summary><strong>TS_DeflickerNode</strong> - Reduces temporal brightness/color flicker in image sequences.</summary>

![Screenshot placeholder for TS_DeflickerNode](docs/img/placeholders/ts-deflickernode.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Reduces temporal brightness/color flicker in image sequences.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `images`, `method`, `window_size`, `intensity`, `preserve_details`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_DeflickerNode`
- Class: `TS_DeflickerNode`
- File: `nodes/ts_deflicker_node.py`
- Category: `Video PostProcessing`
- Function: `deflicker`

</details>

<details>
<summary><strong>TS_Free_Video_Memory</strong> - Pass-through node that aggressively frees RAM/VRAM between heavy video steps.</summary>

![Screenshot placeholder for TS_Free_Video_Memory](docs/img/placeholders/ts-free-video-memory.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Pass-through node that aggressively frees RAM/VRAM between heavy video steps.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `images`, `aggressive_cleanup`, `report_memory`
- Optional: *(none)*

**Outputs**
- `IMAGE`

**Technical info**
- Internal id: `TS_Free_Video_Memory`
- Class: `TS_Free_Video_Memory`
- File: `nodes/ts_video_upscale_node.py`
- Category: `video`
- Function: `cleanup_memory`

</details>

<details>
<summary><strong>TS_LTX_FirstLastFrame</strong> - Injects first/last frame guidance into latent workflows (useful for LTX video control).</summary>

![Screenshot placeholder for TS_LTX_FirstLastFrame](docs/img/placeholders/ts-ltx-firstlastframe.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Injects first/last frame guidance into latent workflows (useful for LTX video control).

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `vae`, `latent`, `first_strength`, `last_strength`, `enable_last_frame`
- Optional: `first_image`, `last_image`

**Outputs**
- `LATENT`

**Technical info**
- Internal id: `TS_LTX_FirstLastFrame`
- Class: `TS_LTX_FirstLastFrame`
- File: `nodes/ts_ltx_tools_node.py`
- Category: `conditioning/video_models`
- Function: `execute`

</details>

<details>
<summary><strong>TS_Animation_Preview</strong> - Creates a quick animation preview from image frames, with optional audio merge.</summary>

![Screenshot placeholder for TS_Animation_Preview](docs/img/placeholders/ts-animation-preview.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Creates a quick animation preview from image frames, with optional audio merge.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `images`, `fps`
- Optional: `audio`

**Outputs**
- Side effects / UI output.

**Technical info**
- Internal id: `TS_Animation_Preview`
- Class: `TS_Animation_Preview`
- File: `nodes/ts_interface_tools_node.py`
- Category: `TS/Interface Tools`
- Function: `preview`
- Dependency note: Uses `imageio` / `imageio-ffmpeg` for video writing.

</details>

### File & Model Utilities

<details>
<summary><strong>TS_FileBrowser</strong> - In-node media picker that loads image/video/audio/mask from disk into your graph.</summary>

![Screenshot placeholder for TS_FileBrowser](docs/img/placeholders/ts-filebrowser.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
In-node media picker that loads image/video/audio/mask from disk into your graph.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: *(dynamic in code/UI)*
- Optional: *(none)*

**Outputs**
- `IMAGE`, `VIDEO`, `AUDIO`, `MASK`, `STRING`

**Technical info**
- Internal id: `TS_FileBrowser`
- Class: `TS_FileBrowser`
- File: `nodes/ts_file_browser_node.py`
- Category: `TS/Input`
- Function: `get_selected_media`

</details>

<details>
<summary><strong>TS_FilePathLoader</strong> - Returns file path and file name by index from a folder list.</summary>

![Screenshot placeholder for TS_FilePathLoader](docs/img/placeholders/ts-filepathloader.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Returns file path and file name by index from a folder list.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `folder_path`, `index`
- Optional: *(none)*

**Outputs**
- `STRING`, `STRING`

**Technical info**
- Internal id: `TS_FilePathLoader`
- Class: `TS_FilePathLoader`
- File: `nodes/ts_file_path_node.py`
- Category: `file_utils`
- Function: `get_file_path`

</details>

<details>
<summary><strong>TS Files Downloader</strong> - Bulk downloader for models/assets with resume, mirrors, proxy support, and optional unzip.</summary>

![Screenshot placeholder for TS Files Downloader](docs/img/placeholders/ts-files-downloader.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Bulk downloader for models/assets with resume, mirrors, proxy support, and optional unzip.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `file_list`, `skip_existing`, `verify_size`, `chunk_size_kb`
- Optional: `hf_token`, `hf_domain`, `proxy_url`, `modelscope_token`, `unzip_after_download`, `enable`

**Outputs**
- Side effects / UI output.

**Technical info**
- Internal id: `TS Files Downloader`
- Class: `TS_DownloadFilesNode`
- File: `nodes/ts_downloader_node.py`
- Category: `Tools/TS_IO`
- Function: `execute_downloads`

</details>

<details>
<summary><strong>TS Youtube Chapters</strong> - Converts EDL timing into ready-to-paste YouTube chapter timestamps.</summary>

![Screenshot placeholder for TS Youtube Chapters](docs/img/placeholders/ts-youtube-chapters.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Converts EDL timing into ready-to-paste YouTube chapter timestamps.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `edl_file_path`
- Optional: *(none)*

**Outputs**
- `STRING`

**Technical info**
- Internal id: `TS Youtube Chapters`
- Class: `TS_EDLToYouTubeChaptersNode`
- File: `nodes/ts_edl_chapters_node.py`
- Category: `Tools/TS_Video`
- Function: `convert_edl_to_youtube_chapters`

</details>

<details>
<summary><strong>TS_ModelScanner</strong> - Scans model files and returns a human-readable structure/metadata summary.</summary>

![Screenshot placeholder for TS_ModelScanner](docs/img/placeholders/ts-modelscanner.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Scans model files and returns a human-readable structure/metadata summary.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `model_name`
- Optional: `model`, `summary_only`

**Outputs**
- `STRING`

**Technical info**
- Internal id: `TS_ModelScanner`
- Class: `TS_ModelScanner`
- File: `nodes/ts_models_tools_node.py`
- Category: `utils/model_analysis`
- Function: `scan_model`

</details>

<details>
<summary><strong>TS_ModelConverter</strong> - One-click converter for model precision conversion workflows.</summary>

![Screenshot placeholder for TS_ModelConverter](docs/img/placeholders/ts-modelconverter.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
One-click converter for model precision conversion workflows.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `model`
- Optional: *(none)*

**Outputs**
- `MODEL`

**Technical info**
- Internal id: `TS_ModelConverter`
- Class: `TS_ModelConverterNode`
- File: `nodes/ts_models_tools_node.py`
- Category: `conversion`
- Function: `convert_to_fp8`

</details>

<details>
<summary><strong>TS_ModelConverterAdvanced</strong> - Advanced model converter with more control over format, preset, and output behavior.</summary>

![Screenshot placeholder for TS_ModelConverterAdvanced](docs/img/placeholders/ts-modelconverteradvanced.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Advanced model converter with more control over format, preset, and output behavior.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `model_name`, `fp8_mode`, `conversion_preset`, `shard_subdir`, `final_filename`
- Optional: `model`

**Outputs**
- `STRING`

**Technical info**
- Internal id: `TS_ModelConverterAdvanced`
- Class: `TS_ModelConverterAdvancedNode`
- File: `nodes/ts_models_tools_node.py`
- Category: `Model Conversion`
- Function: `convert_model`

</details>

<details>
<summary><strong>TS_ModelConverterAdvancedDirect</strong> - Advanced converter variant that works directly from connected MODEL input.</summary>

![Screenshot placeholder for TS_ModelConverterAdvancedDirect](docs/img/placeholders/ts-modelconverteradvanceddirect.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Advanced converter variant that works directly from connected MODEL input.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `model`, `fp8_mode`, `conversion_preset`, `shard_subdir`, `final_filename`
- Optional: *(none)*

**Outputs**
- `STRING`

**Technical info**
- Internal id: `TS_ModelConverterAdvancedDirect`
- Class: `TS_ModelConverterAdvancedDirectNode`
- File: `nodes/ts_models_tools_node.py`
- Category: `TS/Model Conversion`
- Function: `convert_model`

</details>

<details>
<summary><strong>TS_CPULoraMerger</strong> - Merges up to four LoRA files into a base model on CPU and saves a new safetensors file.</summary>

![Screenshot placeholder for TS_CPULoraMerger](docs/img/placeholders/ts-cpu-lora-merger.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Merges up to four LoRA files into a base model on CPU and saves a new safetensors file.

**Quick usage**
1. Select your base model from `checkpoints` or `diffusion_models`.
2. Choose up to four LoRA files and set strength values.
3. Set output filename and run the node to save the merged model.

**Main controls**
- Required: `base_model`, `lora_1_name`, `lora_1_strength`, `lora_2_name`, `lora_2_strength`, `lora_3_name`, `lora_3_strength`, `lora_4_name`, `lora_4_strength`, `output_model_name`
- Optional: *(none)*

**Outputs**
- `STRING, STRING`

**Technical info**
- Internal id: `TS_CPULoraMerger`
- Class: `TS_CPULoraMergerNode`
- File: `nodes/ts_models_tools_node.py`
- Category: `TS/Model Tools`
- Function: `merge_to_file`
- Dependency note: Uses ComfyUI model loading and `safetensors` for CPU-side merge and save.

</details>

### Interface & Logic

<details>
<summary><strong>TS_FloatSlider</strong> - Simple UI float slider node for clean graph parameter control.</summary>

![Screenshot placeholder for TS_FloatSlider](docs/img/placeholders/ts-floatslider.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Simple UI float slider node for clean graph parameter control.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `value`
- Optional: *(none)*

**Outputs**
- `FLOAT`

**Technical info**
- Internal id: `TS_FloatSlider`
- Class: `TS_FloatSlider`
- File: `nodes/ts_interface_tools_node.py`
- Category: `TS Tools/Sliders`
- Function: `get_value`

</details>

<details>
<summary><strong>TS_Int_Slider</strong> - Simple UI integer slider node for deterministic integer parameters.</summary>

![Screenshot placeholder for TS_Int_Slider](docs/img/placeholders/ts-int-slider.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Simple UI integer slider node for deterministic integer parameters.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `value`
- Optional: *(none)*

**Outputs**
- `INT`

**Technical info**
- Internal id: `TS_Int_Slider`
- Class: `TS_Int_Slider`
- File: `nodes/ts_interface_tools_node.py`
- Category: `TS Tools/Sliders`
- Function: `get_value`

</details>

<details>
<summary><strong>TS_Smart_Switch</strong> - Switches between two inputs by mode and keeps your graph compact.</summary>

![Screenshot placeholder for TS_Smart_Switch](docs/img/placeholders/ts-smart-switch.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Switches between two inputs by mode and keeps your graph compact.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `data_type`, `switch`
- Optional: `input_1`, `input_2`

**Outputs**
- `*`

**Technical info**
- Internal id: `TS_Smart_Switch`
- Class: `TS_Smart_Switch`
- File: `nodes/ts_interface_tools_node.py`
- Category: `TS Tools/Logic`
- Function: `smart_switch`

</details>

<details>
<summary><strong>TS_Math_Int</strong> - Integer math helper for counters, offsets, and simple graph logic.</summary>

![Screenshot placeholder for TS_Math_Int](docs/img/placeholders/ts-math-int.png)

> Screenshot placeholder: replace this with your own screenshot.

**What this node does**
Integer math helper for counters, offsets, and simple graph logic.

**Quick usage**
1. Add the node and connect all required inputs.
2. Keep defaults first, then adjust 1-2 controls at a time.
3. Connect outputs to the next node and compare before/after.

**Main controls**
- Required: `a`, `b`, `operation`
- Optional: *(none)*

**Outputs**
- `INT`

**Technical info**
- Internal id: `TS_Math_Int`
- Class: `TS_Math_Int`
- File: `nodes/ts_interface_tools_node.py`
- Category: `TS/Math`
- Function: `calculate`

</details>

## Notes

- Some nodes depend on optional libraries (for example `demucs`, `py360convert`, `spandrel`).
- If a dependency is missing, startup diagnostics will show it in the load report.
- For stable production use, pin dependency versions and keep one validated workflow per node group.
