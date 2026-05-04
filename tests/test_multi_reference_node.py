"""Behaviour tests for TS_MultiReference (IMAGE-input version)."""

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
    import comfy.utils as comfy_utils

    monkeypatch.setattr(comfy_utils, "common_upscale", _common_upscale)

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
    monkeypatch.setitem(sys.modules, "node_helpers", node_helpers)


def _load_module(monkeypatch):
    _install_stubs(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop("nodes.conditioning.ts_multi_reference", None)
    return importlib.import_module("nodes.conditioning.ts_multi_reference")


def _make_image(height: int = 64, width: int = 64, value: float = 0.5) -> torch.Tensor:
    return torch.full((1, height, width, 3), value, dtype=torch.float32)


def test_v3_schema_contract(monkeypatch):
    module = _load_module(monkeypatch)

    schema = module.TS_MultiReference.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert module.NODE_CLASS_MAPPINGS == {"TS_MultiReference": module.TS_MultiReference}
    assert module.NODE_DISPLAY_NAME_MAPPINGS == {"TS_MultiReference": "TS Multi Reference"}
    assert schema.node_id == "TS_MultiReference"
    assert schema.display_name == "TS Multi Reference"
    assert schema.category == "TS/Conditioning"

    # All connectivity inputs are optional now.
    assert inputs["conditioning"].optional is True
    assert inputs["vae"].optional is True
    assert inputs["image_1"].optional is True
    assert inputs["image_2"].optional is True
    assert inputs["image_3"].optional is True

    assert inputs["max_megapixels"].kwargs["default"] == 1.0

    # Per-slot outputs: image_1, image_2, image_3, conditioning.
    output_names = [out.display_name for out in schema.outputs]
    assert output_names == ["image_1", "image_2", "image_3", "conditioning"]


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


def _unpack(output):
    """V3 NodeOutput.values is a flat tuple of every output in schema order."""
    values = output.values
    assert len(values) == 4, f"expected 4 outputs (image_1/2/3 + conditioning), got {len(values)}"
    image_1, image_2, image_3, conditioning = values
    return image_1, image_2, image_3, conditioning


def test_execute_three_images_appends_three_reference_latents(monkeypatch):
    module = _load_module(monkeypatch)

    img_a = _make_image(640, 960, value=0.1)
    img_b = _make_image(768, 512, value=0.2)
    img_c = _make_image(480, 640, value=0.3)

    class FakeVAE:
        def __init__(self):
            self.encoded_shapes = []

        def encode(self, image):
            self.encoded_shapes.append(tuple(image.shape))
            return torch.full(
                (image.shape[0], 16, image.shape[1] // 8, image.shape[2] // 8),
                float(len(self.encoded_shapes)),
            )

    conditioning = [["positive", {"existing": True}]]
    vae = FakeVAE()
    output = module.TS_MultiReference.execute(
        max_megapixels=0.4,
        conditioning=conditioning,
        vae=vae,
        image_1=img_a,
        image_2=img_b,
        image_3=img_c,
    )
    image_1, image_2, image_3, output_conditioning = _unpack(output)

    for image in (image_1, image_2, image_3):
        assert isinstance(image, torch.Tensor)
        assert image.shape[-1] == 3
        assert image.shape[1] % 32 == 0 and image.shape[2] % 32 == 0
    assert len(vae.encoded_shapes) == 3
    # Original conditioning untouched (immutability).
    assert conditioning == [["positive", {"existing": True}]]
    ref_latents = output_conditioning[0][1]["reference_latents"]
    assert len(ref_latents) == 3


def test_execute_no_images_blocks_all_image_outputs_and_passes_conditioning(monkeypatch):
    module = _load_module(monkeypatch)

    conditioning = [["positive", {"existing": True}]]

    class UnusedVAE:
        def encode(self, _image):
            raise AssertionError("VAE.encode should not be called when no images are provided.")

    output = module.TS_MultiReference.execute(
        max_megapixels=1.0,
        conditioning=conditioning,
        vae=UnusedVAE(),
    )
    image_1, image_2, image_3, output_conditioning = _unpack(output)

    for image in (image_1, image_2, image_3):
        assert isinstance(image, module.ExecutionBlocker)
    assert output_conditioning == conditioning


def test_execute_no_conditioning_skips_encoding_but_resizes_images(monkeypatch):
    module = _load_module(monkeypatch)

    img = _make_image(256, 256, value=0.4)

    class UnusedVAE:
        def encode(self, _image):
            raise AssertionError("VAE.encode should not run when conditioning is unconnected.")

    output = module.TS_MultiReference.execute(
        max_megapixels=1.0,
        conditioning=None,
        vae=UnusedVAE(),
        image_1=img,
    )
    image_1, image_2, image_3, output_conditioning = _unpack(output)

    assert isinstance(image_1, torch.Tensor)
    assert image_1.shape[-1] == 3
    assert isinstance(image_2, module.ExecutionBlocker)
    assert isinstance(image_3, module.ExecutionBlocker)
    # No conditioning supplied → empty list, not None.
    assert output_conditioning == []


def test_execute_images_with_conditioning_but_no_vae_raises(monkeypatch):
    module = _load_module(monkeypatch)

    img = _make_image(256, 256, value=0.5)
    conditioning = [["positive", {}]]

    with pytest.raises(RuntimeError, match="VAE input is required"):
        module.TS_MultiReference.execute(
            max_megapixels=1.0,
            conditioning=conditioning,
            vae=None,
            image_1=img,
        )


def test_execute_skips_unconnected_slots_in_middle(monkeypatch):
    """Only image_1 and image_3 connected; image_2 left empty."""
    module = _load_module(monkeypatch)

    img_a = _make_image(value=0.1)
    img_c = _make_image(value=0.3)
    conditioning = [["positive", {}]]

    class FakeVAE:
        def encode(self, image):
            return torch.zeros((image.shape[0], 4, 8, 8))

    output = module.TS_MultiReference.execute(
        max_megapixels=1.0,
        conditioning=conditioning,
        vae=FakeVAE(),
        image_1=img_a,
        image_2=None,
        image_3=img_c,
    )
    image_1, image_2, image_3, output_conditioning = _unpack(output)

    assert isinstance(image_1, torch.Tensor)
    assert isinstance(image_2, module.ExecutionBlocker)
    assert isinstance(image_3, torch.Tensor)
    ref_latents = output_conditioning[0][1]["reference_latents"]
    assert len(ref_latents) == 2


def test_validate_inputs_rejects_zero_megapixels(monkeypatch):
    module = _load_module(monkeypatch)

    assert module.TS_MultiReference.validate_inputs(max_megapixels=0.0) == (
        "max_megapixels must be greater than zero."
    )
    assert module.TS_MultiReference.validate_inputs(max_megapixels=1.0) is True
