# ComfyUI Timesaver Nodes

[English](README.md) | [Russian](README.ru.md)

A focused, code-accurate reference for the nodes that have screenshots in `docs/img`. Each section below documents the node's purpose, inputs, outputs, and behavior based on the actual node code.

Repository: https://github.com/AlexYez/comfyui-timesaver

## Installation

1. Clone the repo or download the node files.
2. Copy the folder into `ComfyUI/custom_nodes/comfyui-timesaver`.
3. Restart ComfyUI.

## Nodes with screenshots

- [TS Files Downloader (Ultimate)](#ts-files-downloader-ultimate)
- [TS Film Emulation](#ts-film-emulation)
- [TS Image Resize](#ts-image-resize)
- [TS Qwen 3 VL](#ts-qwen-3-vl)
- [TS Style Prompt Selector](#ts-style-prompt-selector)
- [TS Video Depth](#ts-video-depth)
- [TS Whisper](#ts-whisper)
- [TS YouTube Chapters](#ts-youtube-chapters)

## TS Files Downloader (Ultimate)

![TS Files Downloader (Ultimate)](docs/img/TS-Files-Downloader.png)

Bulk downloader with resume support, mirrors, proxies, and optional auto-unzip. This is an output node that writes files to disk.

**Node info**
- Internal id: `TS Files Downloader`
- Category: `Tools/TS_IO`
- Function: `execute_downloads`
- Output node: Yes (side effects only)

**Required inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| file_list | STRING (multiline) | Example list (see below) | One item per line: `URL /target/dir`. Lines starting with `#` are ignored. |
| skip_existing | BOOLEAN | `True` | Skip download if the target file exists; when `verify_size` is on, the size must match. |
| verify_size | BOOLEAN | `True` | Validate file size against `Content-Length` when available. |
| chunk_size_kb | INT | `4096` | Download chunk size in KB (4096 = 4 MB). |

**Optional inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| hf_token | STRING | `""` | Hugging Face token (starts with `hf_`). |
| hf_domain | STRING | `huggingface.co, hf-mirror.com` | Comma-separated HF mirrors used for availability checks and downloads. |
| proxy_url | STRING | `""` | Proxy URL, e.g. `http://127.0.0.1:7890`. |
| modelscope_token | STRING | `""` | ModelScope access token. |
| unzip_after_download | BOOLEAN | `False` | Auto-extract `.zip` files to the target directory and delete the archive. |
| enable | BOOLEAN | `True` | Disable to skip all downloads. |

**Outputs**
- None. This node performs downloads as a side effect.

**Example file_list**

```
https://www.dropbox.com/sh/example_folder?dl=0 /path/to/models
https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors /path/to/checkpoints
```

**Behavior notes**
- Connectivity check runs against the domains extracted from `file_list`; if none are reachable, the node exits early.
- HF mirror selection uses the first reachable entry in `hf_domain`.
- Resumable downloads use `.part` files and HTTP range requests when supported.
- Dropbox links are converted to direct-download form.
- Filenames are taken from `Content-Disposition` when available, otherwise from the URL.

## TS Film Emulation

![TS Film Emulation](docs/img/TS-Film-Emulation.png)

Film look node with built-in presets or external `.cube` LUTs, plus contrast curve, warmth, fade, and grain.

**Node info**
- Internal id: `TS_Film_Emulation`
- Category: `Image/Color`
- Function: `process`

**Required inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| image | IMAGE | N/A | Input image tensor. |
| enable | BOOLEAN | `True` | Toggle the whole effect. |
| film_preset | LIST | `External LUT` (first option) | Options: External LUT, Kodak Vision3 250D, Kodak Portra 400, Fuji Eterna 250T, Agfa Vista 200, Ilford HP5, Kodak Gold 200, Fuji Superia 400. |
| lut_choice | LIST | `None` (first option) | List is scanned from `luts/` and includes `None` plus `.cube` files. |
| lut_strength | FLOAT | `1.0` | Blend strength for the external LUT. |
| gamma_correction | BOOLEAN | `True` | Apply sRGB to linear conversion around LUT processing. |
| film_strength | FLOAT | `1.0` | Blend strength for built-in film presets. |
| contrast_curve | FLOAT | `1.0` | Contrast curve around mid-tones. |
| warmth | FLOAT | `0.0` | Warm/cool bias (red up, blue down). |
| grain_intensity | FLOAT | `0.02` | Grain intensity. |
| grain_size | FLOAT | `0.5` | Grain scale (larger = bigger grain). |
| fade | FLOAT | `0.0` | Lift toward mid-gray for a faded look. |
| shadow_saturation | FLOAT | `0.8` | Saturation factor in shadows. |
| highlight_saturation | FLOAT | `0.85` | Saturation factor in highlights. |

**Outputs**
- IMAGE: processed image.

**Behavior notes**
- If `enable` is off, the input image is returned unchanged.
- Built-in film presets are blended with the original using `film_strength`.
- Contrast curve, warmth, fade, and shadow/highlight saturation are applied before LUTs.
- External LUTs are loaded from `luts/<lut_choice>` and can use gamma correction.
- Grain is added after color operations; `grain_size` controls noise scale.

## TS Image Resize

![TS Image Resize](docs/img/TS-Image-Resize.png)

Flexible image resize node: explicit size, side-based scaling, scale factor, or megapixels, with optional mask support.

**Node info**
- Internal id: `TS_ImageResize`
- Category: `image`
- Function: `resize`

**Required inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| pixels | IMAGE | N/A | Input image tensor. |
| target_width | INT | `0` | Target width in pixels (0 = ignore). |
| target_height | INT | `0` | Target height in pixels (0 = ignore). |
| smaller_side | INT | `0` | Target size for the smaller side (0 = ignore). |
| larger_side | INT | `0` | Target size for the larger side (0 = ignore). |
| scale_factor | FLOAT | `0.0` | Scale factor (0 = ignore). |
| keep_proportion | BOOLEAN | `True` | Keep aspect ratio when using target sizes. |
| upscale_method | LIST | `bicubic` | Methods: nearest-exact, bilinear, bicubic, area, lanczos. |
| divisible_by | INT | `1` | Snap final size to a multiple of this value. |
| megapixels | FLOAT | `1.0` | Target megapixels (used when no other sizing input is set). |
| dont_enlarge | BOOLEAN | `False` | Prevents upscaling beyond the original resolution. |

**Optional inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| mask | MASK | N/A | Optional mask resized alongside the image. |

**Outputs**

| Name | Type | Details |
| --- | --- | --- |
| IMAGE | IMAGE | Resized image. |
| width | INT | Final width in pixels. |
| height | INT | Final height in pixels. |
| MASK | MASK | Resized mask if provided, otherwise the original mask. |

**Behavior notes**
- Sizing priority: `scale_factor` -> `target_width/target_height` -> `smaller_side/larger_side` -> `megapixels`.
- With `keep_proportion` on and both target dimensions set, the image is scaled to cover the target and center-cropped.
- With `keep_proportion` off and both target dimensions set, the image is stretched to the exact size.
- `divisible_by` is applied only in proportional modes (it is ignored in explicit target-size modes).
- Masks are resized with `nearest-exact` to preserve edges.

## TS Qwen 3 VL

![TS Qwen 3 VL](docs/img/TS-Qwen3-VL.png)

Vision-language node for Qwen 3 VL with presets, caching, optional offline mode, and support for images and video frames.

**Node info**
- Internal id: `TS_Qwen3_VL`
- Category: `LLM/TS_Qwen`
- Function: `process`

**Required inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| model_name | STRING | `hfmaster/Qwen3-VL-2B` | Hugging Face model repo id. |
| system_preset | LIST | First preset or `Your instruction` | Presets loaded from `qwen_3_vl_presets.json` plus `Your instruction`. |
| prompt | STRING (multiline) | `""` | User prompt text. |
| seed | INT | `42` | Random seed for generation. |
| max_new_tokens | INT | `512` | Maximum tokens to generate. |
| precision | LIST | `fp16` | Options: fp16, bf16, fp32; int8/int4 appear when bitsandbytes is available. |
| use_flash_attention_2 | BOOLEAN | `True` if available | Enables Flash Attention 2 when the library is present. |
| offline_mode | BOOLEAN | `False` | Use local model only; no downloads. |
| unload_after_generation | BOOLEAN | `False` | Unload the model from cache after generation. |
| enable | BOOLEAN | `True` | Disable to skip inference. |
| hf_token | STRING | `""` | Hugging Face token for private repos. |
| max_image_size | INT | `1024` | Max side length before resizing and center-cropping. |
| video_max_frames | INT | `48` | Maximum number of video frames to process. |

**Optional inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| image | IMAGE | N/A | Optional image input. |
| video | IMAGE | N/A | Optional video frames (IMAGE tensor sequence). |
| custom_system_prompt | STRING (multiline) | N/A | Used only when `system_preset` is `Your instruction`. |
| hf_endpoint | STRING | `huggingface.co, hf-mirror.com` | Comma-separated mirrors used for downloads. |
| proxy | STRING | `""` | HTTP/HTTPS proxy for downloads. |

**Outputs**

| Name | Type | Details |
| --- | --- | --- |
| generated_text | STRING | Model output text (or error string). |
| processed_image | IMAGE | Images/frames after resizing and cropping used for the model. |

**Behavior notes**
- Models are cached globally and stored under `models/LLM/<repo_name>`.
- `offline_mode` requires a complete local model (config + weights), otherwise an error is raised.
- Images are resized down to `max_image_size` and center-cropped to multiples of 32.
- Video inputs are uniformly sampled down to `video_max_frames` before processing.
- If `enable` is off, the node returns the prompt and processed images without running inference.

## TS Style Prompt Selector

![TS Style Prompt Selector](docs/img/TS-Style-Prompt-Selector.png)

Selects a prompt string from `styles/styles.json` by id or name.

**Node info**
- Internal id: `TS_StylePromptSelector`
- Category: `TS/Prompt`
- Function: `get_prompt`

**Required inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| style_id | STRING | `photorealistic` | Style id or name from `styles/styles.json`. |

**Outputs**

| Name | Type | Details |
| --- | --- | --- |
| prompt | STRING | Prompt text for the selected style. |

**Behavior notes**
- Styles are loaded from `styles/styles.json` (expects a `styles` list).
- Matches either `id` or `name` fields.
- If the style is not found, the node returns a single space.
- The pack also exposes `/ts_styles` and `/ts_styles/preview` endpoints for UI support.

## TS Video Depth

![TS Video Depth](docs/img/TS-Video-Depth.png)

Generates depth maps for video frames using VideoDepthAnything with optional colormap, dithering, and blur.

**Node info**
- Internal id: `TS_VideoDepthNode`
- Category: `Tools/Video`
- Function: `execute_process_unified`

**Required inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| images | IMAGE | N/A | Input frames (IMAGE tensor sequence). |
| model_filename | LIST | `video_depth_anything_vitl.pth` | Model weight file: vits (small) or vitl (large). |
| input_size | INT | `518` | Inference input size (auto-adjusted to a multiple of 14). |
| max_res | INT | `1280` | If > 0, downscales frames whose max side exceeds this value. |
| precision | LIST | `fp16` | Inference precision (fp16 or fp32). |
| colormap | LIST | `gray` | gray, inferno, viridis, plasma, magma, cividis. |
| dithering_strength | FLOAT | `0.005` | Adds subtle noise before color mapping. |
| apply_median_blur | BOOLEAN | `True` | Applies median blur on the depth map. |
| upscale_algorithm | LIST | `Lanczos4` | Upscale method: Lanczos4, Cubic, or Linear. |

**Outputs**
- IMAGE: RGB depth map sequence.

**Behavior notes**
- If model files are missing, they are downloaded to `models/videodepthanything`.
- `input_size` is forced down to a multiple of 14 when needed.
- `max_res` reduces memory usage by resizing frames before inference.
- Depth is normalized, then mapped to the chosen colormap with optional dithering and blur.
- The output is upscaled back to the original resolution using `upscale_algorithm`.

## TS Whisper

![TS Whisper](docs/img/TS-Whisper.png)

Audio transcription or translation using Whisper Large v3 with SRT output and plain text.

**Node info**
- Internal id: `TSWhisper`
- Category: `AudioTranscription/TSNodes`
- Function: `generate_srt_and_text`

**Required inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| audio | AUDIO | N/A | Input audio (waveform + sample_rate). |
| output_filename_prefix | STRING | `transcribed_audio` | Prefix for saved SRT files. |
| task | LIST | `transcribe` | transcribe or translate_to_english. |
| source_language | LIST | `auto` | auto, en, ru, fr, de, es, it, ja, ko, zh, uk, pl. |
| save_srt_file | BOOLEAN | `True` | Save SRT to disk. |
| precision | LIST | `fp16` when available | fp32, fp16, and bf16 (if CUDA bf16 is supported). |
| attn_implementation | LIST | `sdpa` when available | eager or sdpa (if PyTorch SDPA is available). |
| plain_text_format | LIST | `single_block` | single_block or newline_per_segment. |
| manual_chunk_length_s | FLOAT | `28.0` | Manual chunk length in seconds (<= 30). |
| manual_chunk_overlap_s | FLOAT | `4.0` | Overlap between chunks in seconds. |

**Optional inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| output_dir | STRING | ComfyUI output dir | Base output directory for SRT files. |

**Outputs**

| Name | Type | Details |
| --- | --- | --- |
| srt_content_string | STRING | Full SRT content. |
| plain_text_string | STRING | Transcript as plain text. |

**Behavior notes**
- Uses `openai/whisper-large-v3` and caches it under `models/whisper`.
- Audio is resampled to 16 kHz before transcription.
- Manual chunking is used for long audio with the specified overlap.
- When `save_srt_file` is on, files are saved to `<output_dir>/subtitles/<prefix>_YYYYMMDD_HHMMSS.srt`.
- Text output format depends on `plain_text_format`.

## TS YouTube Chapters

![TS YouTube Chapters](docs/img/TS-YouTube-Chapters.png)

Converts an EDL file into YouTube chapter timestamps.

**Node info**
- Internal id: `TS Youtube Chapters`
- Category: `Tools/TS_Video`
- Function: `convert_edl_to_youtube_chapters`

**Required inputs**

| Name | Type | Default | Details |
| --- | --- | --- | --- |
| edl_file_path | STRING | `""` | Path to the EDL file. |

**Outputs**

| Name | Type | Details |
| --- | --- | --- |
| youtube_chapters | STRING | Chapter list, one entry per line. |

**Behavior notes**
- Reads the EDL file as UTF-8 and expects standard EDL timecode lines.
- Chapter titles are parsed from the next line's `|M:...|` field.
- Uses a standard EDL start offset of `01:00:00:00`.
- Outputs `mm:ss Title` or `hh:mm:ss Title` when the timeline exceeds one hour.
