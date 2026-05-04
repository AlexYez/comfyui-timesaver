from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _common_upscale(samples, width, height, upscale_method, crop):
    assert crop == "disabled"
    method = "nearest" if upscale_method == "nearest-exact" else upscale_method
    return torch.nn.functional.interpolate(samples, size=(height, width), mode=method)


def _install_stubs(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    input_dir = root / ".test_input"

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: str(input_dir)
    folder_paths.filter_files_content_types = lambda files, _content_types: list(files)
    folder_paths.get_annotated_filepath = lambda name: str(input_dir / name)
    folder_paths.exists_annotated_filepath = lambda name: (input_dir / name).is_file()
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    import comfy.model_management as model_management
    import comfy.utils as comfy_utils

    monkeypatch.setattr(comfy_utils, "common_upscale", _common_upscale)
    monkeypatch.setattr(model_management, "intermediate_dtype", lambda: torch.float32)

    node_helpers = types.ModuleType("node_helpers")

    def conditioning_set_values(conditioning, values, append=False):
        output = []
        for item in conditioning:
            cond, meta = item
            updated = dict(meta)
            for key, value in values.items():
                if append and key in updated:
                    value = updated[key] + value
                updated[key] = value
            output.append([cond, updated])
        return output

    node_helpers.conditioning_set_values = conditioning_set_values
    node_helpers.pillow = lambda fn, arg: fn(arg)
    monkeypatch.setitem(sys.modules, "node_helpers", node_helpers)


def _load_module(monkeypatch):
    _install_stubs(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop("nodes.conditioning.ts_multi_reference", None)
    return importlib.import_module("nodes.conditioning.ts_multi_reference")


def test_multi_reference_v1_contract(monkeypatch):
    module = _load_module(monkeypatch)
    schema = module.TS_MultiReference.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    input_types = module.TS_MultiReference.INPUT_TYPES()

    assert module.NODE_CLASS_MAPPINGS == {"TS_MultiReference": module.TS_MultiReference}
    assert module.NODE_DISPLAY_NAME_MAPPINGS == {"TS_MultiReference": "TS Multi Reference"}
    assert schema.node_id == "TS_MultiReference"
    assert schema.display_name == "TS Multi Reference"
    assert schema.category == "TS/Conditioning"
    assert list(module.TS_MultiReference.RETURN_TYPES) == ["IMAGE", "CONDITIONING"]
    assert list(module.TS_MultiReference.RETURN_NAMES) == ["multi_images", "conditioning"]
    assert list(module.TS_MultiReference.OUTPUT_IS_LIST) == [True, False]
    assert list(input_types["required"]) == [
        "conditioning",
        "vae",
        "max_megapixels",
        "upscale_method",
        "image_1",
        "image_2",
        "image_3",
    ]
    assert input_types["required"]["image_1"][1]["image_upload"] is True
    assert inputs["image_2"].options[0] == ""
    assert input_types["required"]["max_megapixels"][1]["default"] == 1.0


def test_resize_limits_megapixels_and_uses_32_grid(monkeypatch):
    module = _load_module(monkeypatch)
    image = torch.rand((2, 1080, 1920, 3), dtype=torch.float32)
    before = image.clone()

    resized = module._resize_reference_image(image, max_megapixels=1.0, upscale_method="area")

    assert torch.equal(image, before)
    assert resized.shape[0] == 2
    assert resized.shape[-1] == 3
    assert resized.shape[1] % 32 == 0
    assert resized.shape[2] % 32 == 0
    assert resized.shape[1] * resized.shape[2] <= 1_000_000


def test_execute_appends_reference_latents_in_order(monkeypatch):
    module = _load_module(monkeypatch)
    images = {
        "first.png": torch.rand((1, 640, 960, 3), dtype=torch.float32),
        "second.png": torch.rand((1, 768, 512, 4), dtype=torch.float32),
    }
    monkeypatch.setattr(module, "_load_reference_image", lambda name: images[name])

    class FakeVAE:
        def __init__(self):
            self.encoded_shapes = []

        def encode(self, image):
            self.encoded_shapes.append(tuple(image.shape))
            return torch.full((image.shape[0], 16, image.shape[1] // 8, image.shape[2] // 8), len(self.encoded_shapes))

    conditioning = [["positive", {"existing": True}]]
    output = module.TS_MultiReference.execute(
        conditioning=conditioning,
        vae=FakeVAE(),
        max_megapixels=0.4,
        upscale_method="area",
        image_1="first.png",
        image_2="",
        image_3="second.png",
    )
    multi_images, output_conditioning = output.result

    assert len(multi_images) == 2
    assert all(image.shape[-1] == 3 for image in multi_images)
    assert all(image.shape[1] % 32 == 0 and image.shape[2] % 32 == 0 for image in multi_images)
    assert conditioning == [["positive", {"existing": True}]]
    ref_latents = output_conditioning[0][1]["reference_latents"]
    assert len(ref_latents) == 2
    assert ref_latents[0].shape[-2:] == (multi_images[0].shape[1] // 8, multi_images[0].shape[2] // 8)
    assert ref_latents[1].shape[-2:] == (multi_images[1].shape[1] // 8, multi_images[1].shape[2] // 8)


def test_validation_rejects_missing_selected_file(monkeypatch):
    module = _load_module(monkeypatch)

    result = module.TS_MultiReference.validate_inputs(
        max_megapixels=1.0,
        upscale_method="area",
        image_1="missing.png",
        image_2="",
        image_3="",
    )

    assert result == "Invalid image file: missing.png"
