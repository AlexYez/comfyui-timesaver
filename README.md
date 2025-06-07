# ComfyUI Timesaver Nodes

[English](README.md) | [Русский](README.ru.md)

These custom nodes for ComfyUI extend the platform's capabilities with tools for image and video processing, text generation, audio transcription, and file management. Designed to integrate seamlessly into ComfyUI workflows, they cater to a variety of creative and technical needs.

Repository: [https://github.com/AlexYez/comfyui-timesaver](https://github.com/AlexYez/comfyui-timesaver)

## Installation

To install these custom nodes, follow these steps:

1. Clone the repository or download the node files.
2. Place the node files in the `custom_nodes` directory of your ComfyUI installation.
3. Restart ComfyUI to load the new nodes.

## Node Descriptions

### TS_CubemapFacesToEquirectangularNode

- **Purpose**: Converts six cubemap face images (front, right, back, left, top, bottom) into a single equirectangular image using the py360convert library.
- **Inputs**:
  - `front`: Image (front face)
  - `right`: Image (right face)
  - `back`: Image (back face)
  - `left`: Image (left face)
  - `top`: Image (top face)
  - `bottom`: Image (bottom face)
  - `output_width`: Integer (default: 2048, min: 64, max: 8192, step: 64)
  - `output_height`: Integer (default: 1024, min: 32, max: 4096, step: 32)
- **Outputs**:
  - `IMAGE`: Equirectangular image
- **Usage**: Ideal for transforming cubemap projections into equirectangular format for 360-degree visualization or further processing.

### TS_EDLToYouTubeChaptersNode

- **Purpose**: Converts an Edit Decision List (EDL) file into a YouTube chapters string by extracting timecodes and titles.
- **Inputs**:
  - `edl_file_path`: String (path to the EDL file)
- **Outputs**:
  - `STRING`: YouTube chapters string (e.g., "00:00 Intro\n03:15 Part 1")
- **Usage**: Simplifies video editing workflows by generating YouTube chapter markers from EDL files.

### TS_FilePathLoader

- **Purpose**: Retrieves file paths from a specified folder, supporting image and video formats like .mp4 and .mov.
- **Inputs**:
  - `folder_path`: String (path to the folder)
  - `index`: Integer (default: 0, min: 0, step: 1)
- **Outputs**:
  - `STRING`: Full file path
  - `STRING`: File name without extension
- **Usage**: Enables easy file selection and loading within ComfyUI for processing workflows.

### TS_Qwen3_Node

- **Purpose**: Leverages the Qwen3 language model to generate text, optimized for creating detailed image prompts from user input.
- **Inputs**:
  - `model_name`: String (options: "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", etc., default: "Qwen/Qwen3-1.7B")
  - `system`: String (system prompt, multiline, default: predefined prompt for image generation)
  - `prompt`: String (user prompt, multiline, default: "apples on the table")
  - `seed`: Integer (default: 42, min: 0, max: 2^64-1)
  - `max_new_tokens`: Integer (default: 512, min: 64, max: 32768, step: 64)
  - `enable_thinking`: Boolean (default: False)
  - `precision`: String (options: "auto", "fp16", "bf16", default: "auto")
  - `unload_after_generation`: Boolean (default: False)
  - `enable`: Boolean (default: True)
  - `force_redownload`: Boolean (default: False)
- **Outputs**:
  - `STRING`: Generated text
- **Usage**: Enhances creative projects by producing descriptive prompts for AI image generation models.

### TS_Video_Upscale_With_Model

- **Purpose**: Upscales video frames using a specified model, with memory-efficient strategies and Spandrel integration.
- **Inputs**:
  - `model_name`: String (from upscale_models list)
  - `images`: IMAGE (video frames)
  - `upscale_method`: String (options: "nearest-exact", "bilinear", etc.)
  - `factor`: Float (default: 2.0, min: 0.1, max: 8.0, step: 0.1)
  - `device_strategy`: String (options: "auto", "load_unload_each_frame", etc., default: "auto")
- **Outputs**:
  - `IMAGE`: Upscaled video frames
- **Usage**: Improves video quality with flexible device management options for GPU/CPU processing.

### TS_VideoDepth

- **Purpose**: Generates depth maps for video frames using the VideoDepthAnything model, with customizable post-processing.
- **Inputs**:
  - `images`: IMAGE (video frames)
  - `model_filename`: String (options: "video_depth_anything_vits.pth", etc., default: "video_depth_anything_vitl.pth")
  - `input_size`: Integer (default: 518, min: 64, max: 4096, step: 2)
  - `max_res`: Integer (default: 1280, min: -1, max: 8192, step: 1)
  - `precision`: String (options: "fp16", "fp32", default: "fp16")
  - `colormap`: String (options: "gray", "inferno", etc., default: "gray")
  - `dithering_strength`: Float (default: 0.005, min: 0.0, max: 0.016, step: 0.0001)
  - `apply_median_blur`: Boolean (default: True)
  - `upscale_algorithm`: String (options: "Lanczos4", "Cubic", etc., default: "Lanczos4")
- **Outputs**:
  - `IMAGE`: Depth map images
- **Usage**: Adds depth information to videos, useful for 3D effects or reconstructions.

### TS_EquirectangularToCubemapFacesNode

- **Purpose**: Converts an equirectangular image into six cubemap face images using py360convert.
- **Inputs**:
  - `image`: IMAGE (equirectangular image)
  - `cube_size`: Integer (default: 512, min: 64, max: 4096, step: 64)
- **Outputs**:
  - `IMAGE`: Front face
  - `IMAGE`: Right face
  - `IMAGE`: Back face
  - `IMAGE`: Left face
  - `IMAGE`: Top face
  - `IMAGE`: Bottom face
- **Usage**: Facilitates conversion to cubemap format for specific rendering or VR applications.

### TSWhisper

- **Purpose**: Transcribes or translates audio using the Whisper model, with SRT file generation support.
- **Inputs**:
  - `audio`: AUDIO
  - `output_filename_prefix`: String (default: "transcribed_audio")
  - `task`: String (options: "transcribe", "translate_to_english", default: "transcribe")
  - `source_language`: String (options: "auto", "en", etc., default: "auto")
  - `save_srt_file`: Boolean (default: True)
  - `precision`: String (options: "fp32", "fp16", etc., default: "fp16")
  - `attn_implementation`: String (options: "eager", "sdpa", default: varies)
  - `plain_text_format`: String (options: "single_block", "newline_per_segment", default: "single_block")
  - `manual_chunk_length_s`: Float (default: 28.0, min: 5.0, max: 30.0, step: 1.0)
  - `manual_chunk_overlap_s`: Float (default: 4.0, min: 0.0, max: 10.0, step: 0.5)
  - `output_dir`: String (optional, default: ComfyUI output directory)
- **Outputs**:
  - `STRING`: SRT content
  - `STRING`: Plain text transcription
- **Usage**: Automates audio transcription and subtitle creation for multimedia projects.

### TS_ImageResize

- **Purpose**: Resizes images with flexible options for dimensions, scaling, and proportion control.
- **Inputs**:
  - `pixels`: IMAGE
  - `target_width`: Integer (default: 0, min: 0, max: 8192, step: 8)
  - `target_height`: Integer (default: 0, min: 0, max: 8192, step: 8)
  - `smaller_side`: Integer (default: 0, min: 0, max: 8192, step: 8)
  - `larger_side`: Integer (default: 0, min: 0, max: 8192, step: 8)
  - `scale_factor`: Float (default: 0.0, min: 0.0, max: 10.0, step: 0.01)
  - `keep_proportion`: Boolean (default: True)
  - `upscale_method`: String (options: "nearest-exact", "bilinear", etc., default: "bicubic")
  - `divisible_by`: Integer (default: 1, min: 1, max: 256, step: 1)
- **Outputs**:
  - `IMAGE`: Resized image
  - `INT`: New width
  - `INT`: New height
- **Usage**: Adapts image sizes for compatibility with various models or display requirements.

### TS_DownloadFilesNode

- **Purpose**: Downloads files from URLs with resume support, retries, and size verification, optionally using a Hugging Face token.
- **Inputs**:
  - `file_list`: String (multiline, format: "URL /path/to/save_directory")
  - `skip_existing`: Boolean (default: True)
  - `verify_size`: Boolean (default: True)
  - `chunk_size_kb`: Integer (default: 4096, min: 1, max: 65536, step: 1)
  - `hf_token`: String (optional, default: "")
- **Outputs**: None (downloads files to specified directories)
- **Usage**: Streamlines fetching of external resources like models or datasets for ComfyUI workflows.

## Dependencies

- `py360convert`
- `torch`
- `numpy`
- `transformers`
- `huggingface_hub`
- `spandrel` (for TS_Video_Upscale_With_Model)
- `VideoDepthAnything` (for TS_VideoDepth)
- `torchaudio` (for TSWhisper)
- `srt` (for TSWhisper)
- `requests`
- `tqdm`
- `torchvision` (optional, for TS_ImageResize with Lanczos)

Install these libraries in your environment to ensure full functionality.

## Troubleshooting

- **Model Loading Errors**: Verify model files are downloaded and accessible. Check paths and permissions.
- **Memory Issues**: For video or large image processing, ensure sufficient GPU memory. Adjust device strategies if needed.
- **Dependency Conflicts**: Confirm compatibility between dependencies, Python version, and ComfyUI.

## Contributing

Contributions are welcome! Submit issues or pull requests on the GitHub repository.

## License

These custom nodes are released under the [MIT License](LICENSE).