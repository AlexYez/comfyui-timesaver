from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import comfy.model_management
import comfy.utils
import folder_paths
import node_helpers
from comfy_api.latest import IO


logger = logging.getLogger(__name__)

_EMPTY_IMAGE = ""
_IMAGE_SLOT_COUNT = 3
_SIZE_MULTIPLE = 32
_DEFAULT_MEGAPIXELS = 1.0
_UPSCALE_METHODS = ["area", "bilinear", "bicubic", "lanczos", "nearest-exact"]


def _list_input_images() -> list[str]:
    input_dir = Path(folder_paths.get_input_directory())
    if not input_dir.is_dir():
        return []
    files = [
        path.name
        for path in input_dir.iterdir()
        if path.is_file()
    ]
    return sorted(folder_paths.filter_files_content_types(files, ["image"]))


def _image_widget_options(include_empty: bool = True) -> tuple[list[str], dict]:
    options = _list_input_images()
    if include_empty:
        options = [_EMPTY_IMAGE, *options]
    return options, {
        "default": _EMPTY_IMAGE,
        "image_upload": True,
        "tooltip": "ComfyUI input image upload widget. Empty slots are ignored.",
    }


def _selected_images(*image_names: str | None) -> list[str]:
    return [
        image_name
        for image_name in image_names
        if isinstance(image_name, str) and image_name.strip() != _EMPTY_IMAGE
    ]


def _target_dimensions(
    width: int,
    height: int,
    max_megapixels: float,
    multiple: int = _SIZE_MULTIPLE,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")

    if max_megapixels <= 0:
        raise ValueError("max_megapixels must be greater than zero.")

    source_pixels = width * height
    target_pixels = int(max_megapixels * 1_000_000)
    scale = min(1.0, math.sqrt(target_pixels / source_pixels))

    target_width = max(multiple, int(math.floor((width * scale) / multiple)) * multiple)
    target_height = max(multiple, int(math.floor((height * scale) / multiple)) * multiple)
    return target_width, target_height


def _normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected IMAGE tensor, got {type(image)}.")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {image.ndim} dimensions.")
    if image.shape[-1] < 3:
        raise ValueError(f"Expected IMAGE tensor with at least 3 channels, got {image.shape[-1]}.")
    return image[:, :, :, :3].clone()


def _resize_reference_image(
    image: torch.Tensor,
    max_megapixels: float,
    upscale_method: str,
) -> torch.Tensor:
    image = _normalize_image_tensor(image)
    height = int(image.shape[1])
    width = int(image.shape[2])
    target_width, target_height = _target_dimensions(width, height, max_megapixels)

    if target_width == width and target_height == height:
        return image.clamp(0.0, 1.0)

    samples = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(
        samples,
        target_width,
        target_height,
        upscale_method,
        "disabled",
    ).movedim(1, -1)
    return resized.clamp(0.0, 1.0)


def _intermediate_dtype() -> torch.dtype:
    dtype_fn = getattr(comfy.model_management, "intermediate_dtype", None)
    if callable(dtype_fn):
        return dtype_fn()
    return torch.float32


def _load_reference_image(image_name: str) -> torch.Tensor:
    image_path = folder_paths.get_annotated_filepath(image_name)
    image_path_str = os.fspath(image_path)

    img = node_helpers.pillow(Image.open, image_path_str)
    output_images: list[torch.Tensor] = []
    width = None
    height = None
    dtype = _intermediate_dtype()

    try:
        for frame in ImageSequence.Iterator(img):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)
            if frame.mode == "I":
                frame = frame.point(lambda value: value * (1 / 255))
            frame = frame.convert("RGB")

            if width is None or height is None:
                width, height = frame.size
            if frame.size != (width, height):
                continue

            image = np.array(frame).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(image)[None,].to(dtype=dtype))

            if img.format == "MPO":
                break
    finally:
        img.close()

    if not output_images:
        raise ValueError(f"Image has no readable frames: {image_name}")

    if len(output_images) == 1:
        return output_images[0]
    return torch.cat(output_images, dim=0)


def _encode_reference_latent(vae, image: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        samples = vae.encode(image)
    return {"samples": samples}


def _append_reference_latent(conditioning, latent: dict[str, torch.Tensor]):
    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [latent["samples"]]},
        append=True,
    )


def _hash_files(image_names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for image_name in image_names:
        image_path = folder_paths.get_annotated_filepath(image_name)
        digest.update(image_name.encode("utf-8", errors="surrogatepass"))
        with open(image_path, "rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class TS_MultiReference(IO.ComfyNode):
    """Load image references and attach them to edit-model conditioning."""

    @classmethod
    def define_schema(cls) -> IO.Schema:
        image_1_options, image_1_extra = _image_widget_options()
        image_2_options, image_2_extra = _image_widget_options()
        image_3_options, image_3_extra = _image_widget_options()
        image_upload = IO.UploadType.image

        return IO.Schema(
            node_id="TS_MultiReference",
            display_name="TS Multi Reference",
            category="TS/Conditioning",
            description=(
                "Loads up to three reference images, resizes them to a 32-pixel grid, "
                "encodes them with VAE, and appends ReferenceLatent conditioning."
            ),
            inputs=[
                IO.Conditioning.Input(
                    "conditioning",
                    tooltip="Conditioning that will receive reference_latents.",
                ),
                IO.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip=(
                        "VAE used to encode reference images into latents. "
                        "Optional: only required when at least one image slot is filled."
                    ),
                ),
                IO.Float.Input(
                    "max_megapixels",
                    default=_DEFAULT_MEGAPIXELS,
                    min=0.01,
                    max=16.0,
                    step=0.01,
                    tooltip="Maximum size for each reference image before VAE encoding.",
                ),
                IO.Combo.Input(
                    "upscale_method",
                    options=_UPSCALE_METHODS,
                    default="area",
                    tooltip="Native ComfyUI resize method used before VAE encoding.",
                ),
                IO.Combo.Input(
                    "image_1",
                    options=image_1_options,
                    default=image_1_extra["default"],
                    upload=image_upload,
                    tooltip=image_1_extra["tooltip"],
                ),
                IO.Combo.Input(
                    "image_2",
                    options=image_2_options,
                    default=image_2_extra["default"],
                    upload=image_upload,
                    tooltip=image_2_extra["tooltip"],
                ),
                IO.Combo.Input(
                    "image_3",
                    options=image_3_options,
                    default=image_3_extra["default"],
                    upload=image_upload,
                    tooltip=image_3_extra["tooltip"],
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="multi_images", is_output_list=True),
                IO.Conditioning.Output(display_name="conditioning"),
            ],
            search_aliases=[
                "TS Multi Reference",
                "ReferenceLatent",
                "Flux 2 reference",
                "Qwen image edit reference",
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        max_megapixels: float,
        upscale_method: str,
        image_1: str = _EMPTY_IMAGE,
        image_2: str = _EMPTY_IMAGE,
        image_3: str = _EMPTY_IMAGE,
        **_,
    ):
        if max_megapixels <= 0:
            return "max_megapixels must be greater than zero."
        if upscale_method not in _UPSCALE_METHODS:
            return f"Unsupported upscale_method: {upscale_method}"

        for image_name in _selected_images(image_1, image_2, image_3):
            if not folder_paths.exists_annotated_filepath(image_name):
                return f"Invalid image file: {image_name}"

        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        max_megapixels: float,
        upscale_method: str,
        image_1: str = _EMPTY_IMAGE,
        image_2: str = _EMPTY_IMAGE,
        image_3: str = _EMPTY_IMAGE,
        **_,
    ) -> str:
        image_names = _selected_images(image_1, image_2, image_3)
        digest = hashlib.sha256()
        digest.update(str(max_megapixels).encode("utf-8"))
        digest.update(upscale_method.encode("utf-8"))
        digest.update(_hash_files(image_names).encode("utf-8"))
        return digest.hexdigest()

    @classmethod
    def execute(
        cls,
        conditioning,
        max_megapixels: float,
        upscale_method: str,
        vae=None,
        image_1: str = _EMPTY_IMAGE,
        image_2: str = _EMPTY_IMAGE,
        image_3: str = _EMPTY_IMAGE,
    ):
        processed_images: list[torch.Tensor] = []
        current_conditioning = conditioning

        selected = _selected_images(image_1, image_2, image_3)[:_IMAGE_SLOT_COUNT]

        if selected and vae is None:
            raise RuntimeError(
                "[TS Multi Reference] VAE input is required when at least one "
                "reference image is selected."
            )

        for image_name in selected:
            image = _load_reference_image(image_name)
            processed_image = _resize_reference_image(
                image,
                max_megapixels=max_megapixels,
                upscale_method=upscale_method,
            )
            latent = _encode_reference_latent(vae, processed_image)
            current_conditioning = _append_reference_latent(current_conditioning, latent)
            processed_images.append(processed_image)

        if not processed_images:
            # Pure text-to-image case: no reference, just pass conditioning through.
            logger.debug("[TS Multi Reference] No reference images selected, passing conditioning through.")

        return IO.NodeOutput(processed_images, current_conditioning)


NODE_CLASS_MAPPINGS = {
    "TS_MultiReference": TS_MultiReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TS_MultiReference": "TS Multi Reference",
}
