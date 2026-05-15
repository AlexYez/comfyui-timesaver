"""Timesaver RIFE frame interpolation node."""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import torch

import comfy.model_management as model_management
import comfy.utils
import folder_paths

from comfy_api.v0_0_2 import IO

from ..frame_interpolation_models import FILMNet, IFNet, detect_rife_config

logger = logging.getLogger(__name__)

_MODEL_REPO_ID = "Comfy-Org/frame_interpolation"
_MODEL_SUBFOLDER = "frame_interpolation"
_RIFE_FOLDER_NAME = "rife"
_FILM_FOLDER_NAME = "film"
_SUPPORTED_MODEL_NAMES = [
    "film_net_fp16.safetensors",
    "rife_v4.25.safetensors",
    "rife_v4.25_heavy.safetensors",
    "rife_v4.25_lite.safetensors",
    "rife_v4.26.safetensors",
    "rife_v4.26_heavy.safetensors",
]
_MODE_SLOWDOWN = "slowdown_x"
_MODE_FPS = "fps_conversion"
_EPSILON = 1e-6
_DEFAULT_TILE_OVERLAP = 64
_MIN_TILE_SIZE = 256
_AUTO_TILE_SIZES = (1536, 1280, 1024, 768, 640, 512, 384, 256)

_RIFE_MODELS_DIR = Path(folder_paths.models_dir) / _RIFE_FOLDER_NAME
_FILM_MODELS_DIR = Path(folder_paths.models_dir) / _FILM_FOLDER_NAME


def _register_model_folder(folder_name: str, folder_path: Path) -> None:
    """Register a ComfyUI model folder if it is not already present."""
    if folder_name not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[folder_name] = (
            [str(folder_path)],
            folder_paths.supported_pt_extensions,
        )
        return

    folder_paths.add_model_folder_path(folder_name, str(folder_path), is_default=True)
    registered_paths, registered_exts = folder_paths.folder_names_and_paths[folder_name]
    if not registered_exts:
        folder_paths.folder_names_and_paths[folder_name] = (registered_paths, folder_paths.supported_pt_extensions)


_register_model_folder(_RIFE_FOLDER_NAME, _RIFE_MODELS_DIR)
_register_model_folder(_FILM_FOLDER_NAME, _FILM_MODELS_DIR)


def _aligned_value(value: int, align: int) -> int:
    """Align a positive integer down to the model alignment."""
    if align <= 1:
        return value
    return max(align, (value // align) * align)


def _initial_tile_size(height: int, width: int, align: int, tile_size: int) -> int:
    """Choose the starting tile size. Zero means full-frame inference."""
    max_dim = max(height, width)
    if tile_size > 0:
        requested_tile_size = max(_aligned_value(tile_size, align), _MIN_TILE_SIZE)
        return 0 if requested_tile_size >= max_dim else requested_tile_size

    pixel_count = height * width
    if pixel_count <= 1920 * 1080 and max_dim <= 1920:
        return 0

    for candidate in _AUTO_TILE_SIZES:
        if candidate < max_dim:
            return min(_aligned_value(candidate, align), max(height, width))

    return min(max(_aligned_value(_MIN_TILE_SIZE, align), _MIN_TILE_SIZE), max(height, width))


def _next_tile_size(current_tile_size: int, align: int) -> int:
    """Reduce the tile size after an OOM while keeping alignment stable."""
    if current_tile_size <= _MIN_TILE_SIZE:
        return current_tile_size
    reduced = max(_MIN_TILE_SIZE, current_tile_size // 2)
    return max(_MIN_TILE_SIZE, _aligned_value(reduced, align))


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Generate tile start positions for a dimension."""
    if tile_size >= length:
        return [0]

    step = max(1, tile_size - overlap)
    starts: list[int] = []
    position = 0
    while True:
        start = min(position, length - tile_size)
        if starts and start == starts[-1]:
            break
        starts.append(start)
        if start >= length - tile_size:
            break
        position += step
    return starts


def _tile_weight(
    top: int,
    left: int,
    bottom: int,
    right: int,
    full_height: int,
    full_width: int,
    overlap: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a feathering mask for blending overlapping tiles."""
    tile_height = bottom - top
    tile_width = right - left
    mask_y = torch.ones(tile_height, dtype=torch.float32, device=device)
    mask_x = torch.ones(tile_width, dtype=torch.float32, device=device)

    fade_y = min(overlap, tile_height - 1) if tile_height > 1 else 0
    fade_x = min(overlap, tile_width - 1) if tile_width > 1 else 0

    if top > 0 and fade_y > 0:
        mask_y[:fade_y] = torch.linspace(1.0 / (fade_y + 1), 1.0, fade_y, device=device)
    if bottom < full_height and fade_y > 0:
        mask_y[-fade_y:] = torch.minimum(
            mask_y[-fade_y:],
            torch.linspace(1.0, 1.0 / (fade_y + 1), fade_y, device=device),
        )

    if left > 0 and fade_x > 0:
        mask_x[:fade_x] = torch.linspace(1.0 / (fade_x + 1), 1.0, fade_x, device=device)
    if right < full_width and fade_x > 0:
        mask_x[-fade_x:] = torch.minimum(
            mask_x[-fade_x:],
            torch.linspace(1.0, 1.0 / (fade_x + 1), fade_x, device=device),
        )

    return (mask_y[:, None] * mask_x[None, :]).unsqueeze(0).unsqueeze(0)


def _get_model_storage(model_name: str) -> tuple[str, Path]:
    """Resolve the native ComfyUI model folder for a checkpoint name."""
    if model_name == "film_net_fp16.safetensors":
        return _FILM_FOLDER_NAME, _FILM_MODELS_DIR
    return _RIFE_FOLDER_NAME, _RIFE_MODELS_DIR


def _ensure_rife_model(model_name: str) -> Path:
    """Download the selected checkpoint into its native ComfyUI model folder if needed."""
    if model_name not in _SUPPORTED_MODEL_NAMES:
        raise ValueError(f"Unsupported RIFE model: {model_name}")

    folder_name, target_dir = _get_model_storage(model_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / model_name

    if target_path.is_file():
        return target_path

    logger.info("[TS Frame Interpolation] Downloading RIFE model %s.", model_name)
    direct_url = (
        f"https://huggingface.co/{_MODEL_REPO_ID}/resolve/main/"
        f"{quote(_MODEL_SUBFOLDER, safe='/')}/{quote(model_name)}?download=1"
    )
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    request = Request(direct_url, headers={"User-Agent": "comfyui-timesaver"})
    try:
        # direct_url is a hardcoded HTTPS URL built from constants
        # (_MODEL_REPO_ID, _MODEL_SUBFOLDER) — not user input — so the bandit
        # B310 blacklist (file:/custom scheme abuse) does not apply here.
        with urlopen(request, timeout=300) as response, tmp_path.open("wb") as handle:  # nosec B310
            shutil.copyfileobj(response, handle)
        tmp_path.replace(target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    folder_paths.filename_list_cache.pop(folder_name, None)
    return target_path


def _load_frame_interpolation_model(model_name: str) -> torch.nn.Module:
    """Load either a FILM or RIFE checkpoint and build the matching architecture."""
    model_path = _ensure_rife_model(model_name)
    state_dict = comfy.utils.load_torch_file(str(model_path), safe_load=True)

    if "extract.extract_sublevels.convs.0.0.conv.weight" in state_dict:
        model = FILMNet()
        model.load_state_dict(state_dict)
        model.eval()
        return model

    state_dict = comfy.utils.state_dict_prefix_replace(state_dict, {"module.": "", "flownet.": ""})

    renamed_keys: dict[str, str] = {}
    for key in state_dict:
        for index in range(5):
            block_prefix = f"block{index}."
            if key.startswith(block_prefix):
                renamed_keys[key] = f"blocks.{index}.{key[len(block_prefix):]}"
                break

    if renamed_keys:
        state_dict = {renamed_keys.get(key, key): value for key, value in state_dict.items()}

    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(("teacher.", "caltime."))
    }

    head_ch, channels = detect_rife_config(state_dict)
    model = IFNet(head_ch=head_ch, channels=channels)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _resolve_scale(mode: str, slowdown_factor: float, source_fps: float, target_fps: float) -> float:
    """Resolve the requested output cadence relative to the source interval count."""
    if mode == _MODE_SLOWDOWN:
        if slowdown_factor < 1.0:
            raise ValueError("slowdown_factor must be greater than or equal to 1.0.")
        return slowdown_factor

    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("source_fps and target_fps must both be greater than 0.")
    if target_fps < source_fps:
        raise ValueError("target_fps must be greater than or equal to source_fps for interpolation.")
    return target_fps / source_fps


def _build_schedule(
    frame_count: int,
    mode: str,
    slowdown_factor: float,
    source_fps: float,
    target_fps: float,
) -> tuple[int, dict[int, list[tuple[int, float]]], dict[int, int]]:
    """Build per-pair interpolation requests and direct frame copies for the output timeline."""
    if frame_count < 2:
        return frame_count, {}, {index: index for index in range(frame_count)}

    pair_count = frame_count - 1
    scale = _resolve_scale(mode, slowdown_factor, source_fps, target_fps)
    if math.isclose(scale, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        return frame_count, {}, {index: index for index in range(frame_count)}

    target_intervals = max(1, int(round(pair_count * scale)))
    output_count = target_intervals + 1

    schedule: dict[int, list[tuple[int, float]]] = {}
    exact_frames: dict[int, int] = {}

    for output_index in range(output_count):
        time_position = (pair_count * output_index) / target_intervals
        rounded_index = int(round(time_position))

        if abs(time_position - rounded_index) <= _EPSILON:
            exact_frames[output_index] = min(max(rounded_index, 0), frame_count - 1)
            continue

        left_index = min(int(math.floor(time_position)), frame_count - 2)
        fraction = time_position - left_index

        if fraction <= _EPSILON:
            exact_frames[output_index] = left_index
        elif fraction >= 1.0 - _EPSILON:
            exact_frames[output_index] = left_index + 1
        else:
            schedule.setdefault(left_index, []).append((output_index, float(fraction)))

    covered = len(exact_frames) + sum(len(items) for items in schedule.values())
    if covered != output_count:
        raise RuntimeError("Internal timeline error: not all output frames were scheduled.")

    return output_count, schedule, exact_frames


def _prepare_frame(
    images: torch.Tensor,
    frame_index: int,
    device: torch.device,
    dtype: torch.dtype,
    align: int,
    minimum_size: int,
) -> torch.Tensor:
    """Convert a BHWC image frame into a padded BCHW tensor for inference."""
    frame = images[frame_index:frame_index + 1].movedim(-1, 1).to(device=device, dtype=dtype)
    target_height = max(frame.shape[2], minimum_size)
    target_width = max(frame.shape[3], minimum_size)
    if align > 1:
        target_height = max(align, ((target_height + align - 1) // align) * align)
        target_width = max(align, ((target_width + align - 1) // align) * align)

    pad_h = target_height - frame.shape[2]
    pad_w = target_width - frame.shape[3]
    if pad_h > 0 or pad_w > 0:
        padding_mode = "reflect"
        if pad_h >= frame.shape[2] or pad_w >= frame.shape[3]:
            padding_mode = "replicate"
        frame = torch.nn.functional.pad(frame, (0, pad_w, 0, pad_h), mode=padding_mode)
    return frame


def _run_model_batch(
    model: torch.nn.Module,
    img0_single: torch.Tensor,
    img1_single: torch.Tensor,
    timestep_values: list[float],
    height: int,
    width: int,
    cache: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Run the interpolation model on a full frame batch."""
    multi_timestep_fn = getattr(model, "forward_multi_timestep", None)
    if callable(multi_timestep_fn):
        return multi_timestep_fn(img0_single, img1_single, timestep_values, cache=cache)[:, :, :height, :width]

    batch = len(timestep_values)
    timesteps = torch.tensor(timestep_values, device=img0_single.device, dtype=img0_single.dtype).view(-1, 1, 1, 1)
    timesteps = timesteps.expand(-1, 1, img0_single.shape[2], img0_single.shape[3])
    img0_batch = img0_single.expand(batch, -1, -1, -1)
    img1_batch = img1_single.expand(batch, -1, -1, -1)
    return model(img0_batch, img1_batch, timestep=timesteps, cache=cache)[:, :, :height, :width]


def _run_tiled_batch(
    model: torch.nn.Module,
    img0_single: torch.Tensor,
    img1_single: torch.Tensor,
    timestep_values: list[float],
    height: int,
    width: int,
    tile_size: int,
    tile_overlap: int,
    output_device: torch.device,
) -> torch.Tensor:
    """Run tiled interpolation and blend the result on the output device."""
    batch = len(timestep_values)
    padded_height = img0_single.shape[2]
    padded_width = img0_single.shape[3]
    accum = torch.zeros((batch, 3, padded_height, padded_width), dtype=torch.float32, device=output_device)
    weights = torch.zeros((1, 1, padded_height, padded_width), dtype=torch.float32, device=output_device)

    overlap = min(tile_overlap, max(1, tile_size // 4))
    for top in _tile_starts(padded_height, tile_size, overlap):
        bottom = min(top + tile_size, padded_height)
        for left in _tile_starts(padded_width, tile_size, overlap):
            right = min(left + tile_size, padded_width)
            tile0 = img0_single[:, :, top:bottom, left:right]
            tile1 = img1_single[:, :, top:bottom, left:right]
            tile_cache = {
                "img0": model.extract_features(tile0),
                "img1": model.extract_features(tile1),
            }
            tile_output = _run_model_batch(
                model=model,
                img0_single=tile0,
                img1_single=tile1,
                timestep_values=timestep_values,
                height=bottom - top,
                width=right - left,
                cache=tile_cache,
            )
            tile_weight = _tile_weight(
                top=top,
                left=left,
                bottom=bottom,
                right=right,
                full_height=padded_height,
                full_width=padded_width,
                overlap=overlap,
                device=output_device,
            )
            accum[:, :, top:bottom, left:right].add_(tile_output.float().to(output_device) * tile_weight)
            weights[:, :, top:bottom, left:right].add_(tile_weight)

    return accum.div_(weights.clamp_min_(1e-6))[:, :, :height, :width]


def _interpolate_batch_adaptive(
    model: torch.nn.Module,
    img0_single: torch.Tensor,
    img1_single: torch.Tensor,
    timestep_values: list[float],
    height: int,
    width: int,
    cache: dict[str, torch.Tensor] | None,
    output_device: torch.device,
    initial_tile_size: int,
    tile_overlap: int,
) -> tuple[torch.Tensor, int]:
    """Run interpolation and progressively fall back to smaller tiles on OOM."""
    align = getattr(model, "pad_align", 1)
    current_tile_size = initial_tile_size

    while True:
        try:
            if current_tile_size <= 0:
                return (
                    _run_model_batch(
                        model=model,
                        img0_single=img0_single,
                        img1_single=img1_single,
                        timestep_values=timestep_values,
                        height=height,
                        width=width,
                        cache=cache,
                    ).float().to(output_device),
                    current_tile_size,
                )

            return (
                _run_tiled_batch(
                    model=model,
                    img0_single=img0_single,
                    img1_single=img1_single,
                    timestep_values=timestep_values,
                    height=height,
                    width=width,
                    tile_size=current_tile_size,
                    tile_overlap=tile_overlap,
                    output_device=output_device,
                ),
                current_tile_size,
            )
        except model_management.OOM_EXCEPTION:
            if current_tile_size <= 0:
                current_tile_size = _initial_tile_size(img0_single.shape[2], img0_single.shape[3], align, tile_size=_MIN_TILE_SIZE * 4)
            else:
                next_tile_size = _next_tile_size(current_tile_size, align)
                if next_tile_size == current_tile_size:
                    raise
                current_tile_size = next_tile_size
            logger.warning(
                "[TS Frame Interpolation] OOM detected, retrying with tile size %s.",
                current_tile_size,
            )
            model_management.soft_empty_cache()


class TS_Frame_Interpolation(IO.ComfyNode):
    """Interpolate image sequences with RIFE models from ComfyUI/models/rife."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="TS_Frame_Interpolation",
            display_name="TS Frame Interpolation",
            category="TS/Video",
            inputs=[
                IO.Image.Input("images"),
                IO.Combo.Input(
                    "model_name",
                    options=_SUPPORTED_MODEL_NAMES,
                    default="rife_v4.26.safetensors",
                    tooltip="RIFE checkpoint stored in ComfyUI/models/rife. Missing files are downloaded automatically.",
                ),
                IO.Combo.Input(
                    "mode",
                    options=[_MODE_SLOWDOWN, _MODE_FPS],
                    default=_MODE_SLOWDOWN,
                    tooltip="slowdown_x extends the clip length. fps_conversion resamples motion to a target FPS.",
                ),
                IO.Float.Input("slowdown_factor", default=2.0, min=1.0, max=16.0, step=0.1, tooltip="Used when mode is slowdown_x."),
                IO.Float.Input("source_fps", default=24.0, min=1.0, max=240.0, step=0.1, tooltip="Used when mode is fps_conversion."),
                IO.Float.Input(
                    "target_fps",
                    default=60.0,
                    min=1.0,
                    max=240.0,
                    step=0.1,
                    tooltip="Used when mode is fps_conversion. Must be greater than or equal to source_fps.",
                ),
                IO.Int.Input(
                    "max_timestep_batch",
                    default=8,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Maximum number of intermediate timestamps evaluated together. Lower values reduce VRAM peaks on long clips.",
                ),
                IO.Int.Input(
                    "tile_size",
                    default=0,
                    min=0,
                    max=4096,
                    step=64,
                    tooltip="0 = automatic. Positive values force tiled interpolation for high-resolution video.",
                ),
                IO.Int.Input(
                    "tile_overlap",
                    default=_DEFAULT_TILE_OVERLAP,
                    min=16,
                    max=512,
                    step=16,
                    tooltip="Blend overlap between tiles. Higher values reduce seams but use more compute.",
                ),
            ],
            outputs=[IO.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        model_name: str,
        mode: str,
        slowdown_factor: float,
        source_fps: float,
        target_fps: float,
        max_timestep_batch: int,
        tile_size: int,
        tile_overlap: int,
    ) -> IO.NodeOutput:
        """Interpolate a sequence of BHWC frames."""
        if images.shape[0] < 2:
            return IO.NodeOutput(images)

        output_count, schedule, exact_frames = _build_schedule(
            frame_count=int(images.shape[0]),
            mode=mode,
            slowdown_factor=float(slowdown_factor),
            source_fps=float(source_fps),
            target_fps=float(target_fps),
        )

        if not schedule and output_count == images.shape[0] and all(exact_frames.get(i) == i for i in range(output_count)):
            return IO.NodeOutput(images)

        height = int(images.shape[1])
        width = int(images.shape[2])
        output_device = torch.device("cpu")
        output_dtype = model_management.intermediate_dtype()
        result = torch.empty((output_count, 3, height, width), dtype=output_dtype, device=output_device)

        for output_index, source_index in exact_frames.items():
            result[output_index] = images[source_index].movedim(-1, 0).to(device=output_device, dtype=output_dtype)

        model = _load_frame_interpolation_model(model_name)
        device = model_management.get_torch_device()
        model_dtype = (
            torch.float16
            if device.type != "cpu" and model_management.should_use_fp16(device=device)
            else torch.float32
        )

        align = getattr(model, "pad_align", 1)
        minimum_size = 1 << max(0, getattr(model, "pyramid_levels", 1) - 1)
        active_tile_size = _initial_tile_size(height, width, align, int(tile_size))
        activation_memory = height * width * 3 * images.element_size() * 20
        if active_tile_size > 0:
            estimated_tile_pixels = active_tile_size * active_tile_size
            activation_memory = min(activation_memory, estimated_tile_pixels * 3 * images.element_size() * 24)
        memory_required = model_management.module_size(model) + activation_memory
        model_management.free_memory(memory_required, device)
        model.to(device=device, dtype=model_dtype)
        model.eval()

        total_interpolated = sum(len(items) for items in schedule.values())
        progress = comfy.utils.ProgressBar(total_interpolated)

        previous_pair_index: int | None = None
        previous_right_frame: torch.Tensor | None = None
        previous_right_features = None

        try:
            with torch.inference_mode():
                for pair_index in sorted(schedule.keys()):
                    requests = schedule[pair_index]
                    if previous_pair_index is not None and pair_index == previous_pair_index + 1 and previous_right_frame is not None:
                        img0_single = previous_right_frame
                    else:
                        img0_single = _prepare_frame(images, pair_index, device, model_dtype, align, minimum_size)

                    img1_single = _prepare_frame(images, pair_index + 1, device, model_dtype, align, minimum_size)

                    full_frame_cache = None
                    next_right_features = None
                    if active_tile_size <= 0:
                        if (
                            previous_pair_index is not None
                            and pair_index == previous_pair_index + 1
                            and previous_right_features is not None
                        ):
                            feat0 = previous_right_features
                        else:
                            feat0 = model.extract_features(img0_single)
                        next_right_features = model.extract_features(img1_single)
                        full_frame_cache = {"img0": feat0, "img1": next_right_features}

                    request_index = 0
                    batch_size = min(max(1, int(max_timestep_batch)), len(requests))

                    while request_index < len(requests):
                        current_batch = min(batch_size, len(requests) - request_index)
                        current_requests = requests[request_index:request_index + current_batch]
                        timestep_values = [fraction for _, fraction in current_requests]
                        cache = full_frame_cache if active_tile_size <= 0 else None

                        try:
                            interpolated, resolved_tile_size = _interpolate_batch_adaptive(
                                model=model,
                                img0_single=img0_single,
                                img1_single=img1_single,
                                timestep_values=timestep_values,
                                height=height,
                                width=width,
                                cache=cache,
                                output_device=output_device,
                                initial_tile_size=active_tile_size,
                                tile_overlap=tile_overlap,
                            )

                            if resolved_tile_size != active_tile_size:
                                active_tile_size = resolved_tile_size
                                if active_tile_size > 0:
                                    previous_right_features = None
                                    next_right_features = None
                                    full_frame_cache = None

                            result_dtype_batch = interpolated.to(dtype=output_dtype)
                            for local_index, (output_index, _fraction) in enumerate(current_requests):
                                result[output_index] = result_dtype_batch[local_index]

                            request_index += current_batch
                            progress.update(current_batch)
                        except model_management.OOM_EXCEPTION:
                            if batch_size <= 1:
                                raise
                            batch_size = max(1, batch_size // 2)
                            logger.warning(
                                "[TS Frame Interpolation] OOM detected, retrying timestep batch with size %s.",
                                batch_size,
                            )
                            model_management.soft_empty_cache()

                    previous_pair_index = pair_index
                    previous_right_frame = img1_single
                    previous_right_features = next_right_features if active_tile_size <= 0 else None
        finally:
            model.to("cpu")
            model_management.soft_empty_cache()

        return IO.NodeOutput(result.movedim(1, -1).clamp_(0.0, 1.0))


NODE_CLASS_MAPPINGS = {
    "TS_Frame_Interpolation": TS_Frame_Interpolation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_Frame_Interpolation": "TS Frame Interpolation",
}
