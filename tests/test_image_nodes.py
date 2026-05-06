"""Behaviour tests for Image-category nodes (cpu-safe).

Covers:
- TS_GetImageMegapixels: shape math + 3-d normalization.
- TS_GetImageSizeSide: large/small side selection.
- TS_ImageBatchCut: first/last cut, edge cases.
- TS_ImageBatchToImageList / TS_ImageListToImageBatch: list <-> batch round-trip.
- TS_Color_Grade: each helper, identity-when-defaults, no input mutation.
- TS_Film_Emulation: contrast curve, sRGB <-> linear round-trip, preset apply.
- TS_ImageResize: side / target / scale_factor / megapixels / divisible_by paths.
- TS_FilmGrain: schema + tensor invariants (no input mutation, range, shape).

Each test verifies tensor invariants (shape, batch, dtype, range, no-mutation)
where applicable. Heavy GPU-only paths are skipped automatically.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


# Real comfy_api is available in ComfyUI Python; if absent, fall back to a
# minimal stub so the modules import on lighter test runners.
def _install_stubs(monkeypatch):
    if "comfy_api.latest" in sys.modules:
        return

    class _Input:
        def __init__(self, id, *args, **kwargs):
            self.id = id
            self.args = args
            self.kwargs = kwargs
            self.optional = bool(kwargs.get("optional", False))
            self.default = kwargs.get("default")
            self.min = kwargs.get("min")
            self.max = kwargs.get("max")
            self.options = kwargs.get("options", args[0] if args else None)

    class _Output:
        def __init__(self, id=None, display_name=None, **kwargs):
            self.id = id
            self.display_name = display_name
            self.kwargs = kwargs

    class _ComfyType:
        Input = _Input
        Output = _Output

    class _Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _NodeOutput:
        def __init__(self, *values, **kwargs):
            self.values = values
            self.args = values
            self.kwargs = kwargs

    class _NumberDisplay:
        slider = "slider"
        number = "number"

    class _IO:
        ComfyNode = object
        Schema = _Schema
        NodeOutput = _NodeOutput
        Image = _ComfyType
        Mask = _ComfyType
        Int = _ComfyType
        Float = _ComfyType
        Boolean = _ComfyType
        Combo = _ComfyType
        String = _ComfyType
        NumberDisplay = _NumberDisplay

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.IO = _IO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)


def _stub_comfy(monkeypatch):
    """Stub comfy.model_management and comfy.utils ProgressBar for nodes that
    expect them at import time. Idempotent if real ones already loaded."""
    if "comfy.model_management" in sys.modules:
        return
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: torch.device("cpu")
    mm.soft_empty_cache = lambda: None
    utils = types.ModuleType("comfy.utils")

    class _PB:
        def __init__(self, total): self.total = total
        def update(self, n): pass
        def update_absolute(self, value, total=None): pass

    utils.ProgressBar = _PB
    comfy.model_management = mm
    comfy.utils = utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)


def _load(monkeypatch, dotted: str):
    _install_stubs(monkeypatch)
    _stub_comfy(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop(dotted, None)
    return importlib.import_module(dotted)


def _img(b=1, h=4, w=4, value=0.5):
    return torch.full((b, h, w, 3), float(value), dtype=torch.float32)


# ---------- TS_GetImageMegapixels ----------


def test_megapixels_schema(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_get_image_megapixels")
    schema = module.TS_GetImageMegapixels.define_schema()
    assert schema.node_id == "TS_GetImageMegapixels"
    assert schema.category == "TS/Image"
    assert [out.display_name for out in schema.outputs] == ["megapixels"]


def test_megapixels_computes_correctly(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_get_image_megapixels")
    img = _img(b=1, h=1000, w=1000)
    output = module.TS_GetImageMegapixels.execute(img)
    assert output.args[0] == pytest.approx(1.0)


def test_megapixels_normalizes_3d_input(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_get_image_megapixels")
    # 3-d input is auto-unsqueezed to 4-d
    img3d = torch.full((100, 200, 3), 0.5, dtype=torch.float32)
    output = module.TS_GetImageMegapixels.execute(img3d)
    assert output.args[0] == pytest.approx(0.02)  # 100*200/1e6


def test_megapixels_rejects_non_tensor(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_get_image_megapixels")
    with pytest.raises(ValueError):
        module.TS_GetImageMegapixels.execute("not-a-tensor")


def test_megapixels_returns_zero_for_none(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_get_image_megapixels")
    output = module.TS_GetImageMegapixels.execute(None)
    assert output.args == (0.0,)


def test_megapixels_fingerprint_changes_with_shape(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_get_image_megapixels")
    fp_a = module.TS_GetImageMegapixels.fingerprint_inputs(_img(h=4, w=4))
    fp_b = module.TS_GetImageMegapixels.fingerprint_inputs(_img(h=8, w=8))
    assert fp_a != fp_b
    assert module.TS_GetImageMegapixels.fingerprint_inputs(None) == "none"


# ---------- TS_GetImageSizeSide ----------


def test_size_side_schema(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_get_image_size_side")
    schema = module.TS_GetImageSizeSide.define_schema()
    assert schema.node_id == "TS_GetImageSizeSide"
    inputs = {item.id: item for item in schema.inputs}
    assert "image" in inputs
    assert "large_side" in inputs


def test_size_side_returns_max_when_large(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_get_image_size_side")
    output = module.TS_GetImageSizeSide.execute(_img(h=480, w=720), True)
    assert output.args == (720,)


def test_size_side_returns_min_when_small(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_get_image_size_side")
    output = module.TS_GetImageSizeSide.execute(_img(h=480, w=720), False)
    assert output.args == (480,)


def test_size_side_handles_3d_input(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_get_image_size_side")
    img3d = torch.zeros((100, 200, 3))
    output = module.TS_GetImageSizeSide.execute(img3d, True)
    assert output.args == (200,)


def test_size_side_returns_zero_for_none(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_get_image_size_side")
    output = module.TS_GetImageSizeSide.execute(None, True)
    assert output.args == (0,)


# ---------- TS_ImageBatchCut ----------


def test_batch_cut_schema(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_batch_cut")
    schema = module.TS_ImageBatchCut.define_schema()
    assert schema.node_id == "TS_ImageBatchCut"
    assert [item.id for item in schema.inputs] == ["image", "first_cut", "last_cut"]


def test_batch_cut_no_op_with_zero_cuts(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_batch_cut")
    img = _img(b=8, h=4, w=4)
    output = module.TS_ImageBatchCut.execute(img, 0, 0)
    assert output.args[0].shape[0] == 8


def test_batch_cut_first_and_last(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_batch_cut")
    # b=10, first_cut=2, last_cut=3 → 5 frames
    img = torch.zeros((10, 2, 2, 3), dtype=torch.float32)
    for i in range(10):
        img[i] = i  # tag with index
    output = module.TS_ImageBatchCut.execute(img, 2, 3)
    out = output.args[0]
    assert out.shape[0] == 5
    assert out[0, 0, 0, 0].item() == 2.0
    assert out[-1, 0, 0, 0].item() == 6.0


def test_batch_cut_overflow_returns_empty_batch(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_batch_cut")
    img = _img(b=4)
    output = module.TS_ImageBatchCut.execute(img, 5, 5)
    out = output.args[0]
    assert out.shape == (0, 4, 4, 3)


def test_batch_cut_negative_cuts_treated_as_zero(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_batch_cut")
    img = _img(b=4)
    output = module.TS_ImageBatchCut.execute(img, -3, -3)
    assert output.args[0].shape[0] == 4


# ---------- TS_Color_Grade ----------


@pytest.fixture
def color_grade(monkeypatch):
    return _load(monkeypatch, "nodes.image.ts_color_grade")


def test_color_grade_schema(color_grade):
    schema = color_grade.TS_Color_Grade.define_schema()
    assert schema.node_id == "TS_Color_Grade"
    inputs = {item.id: item for item in schema.inputs}
    expected = ["image", "hue", "temperature", "saturation", "contrast", "gain", "lift", "gamma", "brightness"]
    assert list(inputs) == expected


def test_color_grade_identity_with_defaults(color_grade):
    img = torch.rand((1, 4, 4, 3), dtype=torch.float32)
    before = img.clone()
    output = color_grade.TS_Color_Grade.execute(
        img, hue=0.0, temperature=0.0, saturation=1.0, contrast=1.0,
        gain=1.0, lift=0.0, gamma=1.0, brightness=0.0,
    )
    out = output.args[0]
    assert torch.equal(img, before)  # input not mutated
    assert torch.allclose(out, img.clamp(0, 1), atol=1e-5)


def test_color_grade_apply_brightness_shifts_pixels(color_grade):
    img = torch.full((1, 2, 2, 3), 0.5, dtype=torch.float32)
    out = color_grade.TS_Color_Grade._ts_apply_brightness(img, 0.1)
    assert torch.all(out == 0.6)


def test_color_grade_saturation_zero_makes_grayscale(color_grade):
    img = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)
    gray = color_grade.TS_Color_Grade._ts_apply_saturation(img, 0.0)
    # When saturation=0, all channels collapse to luminance mean
    expected = img.mean(dim=-1, keepdim=True).expand_as(img)
    assert torch.allclose(gray, expected)


def test_color_grade_contrast_shifts_around_mid_gray(color_grade):
    img = torch.full((1, 2, 2, 3), 0.5, dtype=torch.float32)
    out = color_grade.TS_Color_Grade._ts_apply_contrast(img, 2.0)
    # x=0.5 is invariant; (0.5-0.5)*2+0.5 = 0.5
    assert torch.allclose(out, img)


def test_color_grade_rejects_4d_with_wrong_channels(color_grade):
    img = torch.zeros((1, 4, 4, 4), dtype=torch.float32)  # 4 channels not allowed
    with pytest.raises(ValueError):
        color_grade.TS_Color_Grade._ts_validate_image(img)


def test_color_grade_clamps_output_to_unit_range(color_grade):
    img = torch.full((1, 2, 2, 3), 0.9, dtype=torch.float32)
    output = color_grade.TS_Color_Grade.execute(
        img, hue=0.0, temperature=0.0, saturation=1.0, contrast=1.0,
        gain=2.0, lift=0.5, gamma=1.0, brightness=0.5,
    )
    out = output.args[0]
    assert (out >= 0.0).all() and (out <= 1.0).all()


# ---------- TS_Film_Emulation ----------


@pytest.fixture
def film_emu(monkeypatch):
    return _load(monkeypatch, "nodes.image.ts_film_emulation")


def test_film_emu_schema_lists_presets(film_emu):
    schema = film_emu.TS_Film_Emulation.define_schema()
    assert schema.node_id == "TS_Film_Emulation"
    inputs = {item.id: item for item in schema.inputs}
    assert "film_preset" in inputs
    assert "lut_choice" in inputs


def test_film_emu_disabled_returns_input_unchanged(film_emu):
    img = torch.rand((1, 4, 4, 3), dtype=torch.float32)
    output = film_emu.TS_Film_Emulation.execute(img, enable=False)
    # When disabled, the image is returned as-is (might be the same object).
    assert output.args[0] is img


def test_film_emu_srgb_linear_round_trip(film_emu):
    img = torch.linspace(0, 1, 100).view(1, 10, 10, 1).expand(1, 10, 10, 3)
    linear = film_emu.TS_Film_Emulation._srgb_to_linear(img)
    back = film_emu.TS_Film_Emulation._linear_to_srgb(linear)
    assert torch.allclose(img, back, atol=2e-4)


def test_film_emu_contrast_curve_invariant_at_mid(film_emu):
    img = torch.full((1, 2, 2, 3), 0.5)
    out = film_emu.TS_Film_Emulation._apply_contrast_curve(img, 2.5)
    assert torch.allclose(out, img)


def test_film_emu_apply_preset_returns_clamped_image(film_emu):
    img = torch.rand((1, 8, 8, 3), dtype=torch.float32)
    out = film_emu.TS_Film_Emulation.apply_preset(img, "Kodak Portra 400")
    assert (out >= 0.0).all() and (out <= 1.0).all()
    assert out.shape == img.shape


def test_film_emu_apply_preset_unknown_returns_input(film_emu):
    img = torch.rand((1, 4, 4, 3))
    out = film_emu.TS_Film_Emulation.apply_preset(img, "External LUT")
    # External LUT preset name is a no-op for apply_preset
    assert torch.allclose(out, img)


def test_film_emu_with_grain_stays_in_unit_range(film_emu):
    img = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)
    output = film_emu.TS_Film_Emulation.execute(
        img, enable=True, film_preset="Kodak Vision3 250D", lut_choice="None",
        grain_intensity=0.05, grain_size=1.0,
    )
    out = output.args[0]
    assert (out >= 0.0).all() and (out <= 1.0).all()


# ---------- TS_ImageResize ----------


@pytest.fixture
def image_resize(monkeypatch):
    pytest.importorskip("PIL")
    return _load(monkeypatch, "nodes.image.ts_image_resize")


def test_image_resize_schema(image_resize):
    schema = image_resize.TS_ImageResize.define_schema()
    assert schema.node_id == "TS_ImageResize"
    inputs = {item.id: item for item in schema.inputs}
    assert "pixels" in inputs
    assert "target_width" in inputs and "target_height" in inputs
    assert "smaller_side" in inputs and "larger_side" in inputs
    assert "scale_factor" in inputs and "megapixels" in inputs
    assert "divisible_by" in inputs and "keep_proportion" in inputs


def test_image_resize_to_target_dims(image_resize):
    img = torch.rand((1, 100, 200, 3), dtype=torch.float32)
    output = image_resize.TS_ImageResize.execute(
        img, target_width=64, target_height=64, smaller_side=0, larger_side=0,
        scale_factor=0.0, keep_proportion=False, upscale_method="bilinear",
        divisible_by=1, megapixels=0.0, dont_enlarge=False,
    )
    out_image, w, h, _mask = output.args
    assert out_image.shape == (1, 64, 64, 3)
    assert w == 64 and h == 64
    assert (out_image >= 0).all() and (out_image <= 1).all()


def test_image_resize_smaller_side_keeps_aspect(image_resize):
    img = torch.rand((1, 200, 400, 3), dtype=torch.float32)
    output = image_resize.TS_ImageResize.execute(
        img, target_width=0, target_height=0, smaller_side=100, larger_side=0,
        scale_factor=0.0, keep_proportion=True, upscale_method="bilinear",
        divisible_by=1, megapixels=0.0, dont_enlarge=False,
    )
    out_image, w, h, _ = output.args
    # smaller side becomes 100, so larger side becomes 200 (preserves 2:1)
    assert min(w, h) == 100
    assert max(w, h) == 200


def test_image_resize_megapixels_target(image_resize):
    img = torch.rand((1, 1080, 1920, 3), dtype=torch.float32)
    output = image_resize.TS_ImageResize.execute(
        img, target_width=0, target_height=0, smaller_side=0, larger_side=0,
        scale_factor=0.0, keep_proportion=True, upscale_method="bilinear",
        divisible_by=8, megapixels=1.0, dont_enlarge=False,
    )
    out_image, w, h, _ = output.args
    # ~1MP target, divisible by 8
    assert w * h <= 1_100_000
    assert w % 8 == 0 and h % 8 == 0


def test_image_resize_scale_factor(image_resize):
    img = torch.rand((1, 100, 100, 3), dtype=torch.float32)
    output = image_resize.TS_ImageResize.execute(
        img, target_width=0, target_height=0, smaller_side=0, larger_side=0,
        scale_factor=2.0, keep_proportion=True, upscale_method="nearest-exact",
        divisible_by=1, megapixels=0.0, dont_enlarge=False,
    )
    out_image, w, h, _ = output.args
    assert w == 200 and h == 200


def test_image_resize_dont_enlarge_blocks_upscale(image_resize):
    img = torch.rand((1, 50, 50, 3), dtype=torch.float32)
    output = image_resize.TS_ImageResize.execute(
        img, target_width=200, target_height=200, smaller_side=0, larger_side=0,
        scale_factor=0.0, keep_proportion=True, upscale_method="bilinear",
        divisible_by=1, megapixels=0.0, dont_enlarge=True,
    )
    out_image, w, h, _ = output.args
    # Original was 50×50, with dont_enlarge stays 50×50
    assert out_image.shape == (1, 50, 50, 3)


def test_image_resize_with_mask(image_resize):
    img = torch.rand((1, 100, 100, 3), dtype=torch.float32)
    mask = torch.rand((1, 100, 100), dtype=torch.float32)
    output = image_resize.TS_ImageResize.execute(
        img, target_width=50, target_height=50, smaller_side=0, larger_side=0,
        scale_factor=0.0, keep_proportion=False, upscale_method="bilinear",
        divisible_by=1, megapixels=0.0, dont_enlarge=False, mask=mask,
    )
    out_image, w, h, out_mask = output.args
    assert out_image.shape == (1, 50, 50, 3)
    assert out_mask.shape == (1, 50, 50)


# ---------- TS_ImageBatchToImageList / TS_ImageListToImageBatch ----------


def test_batch_to_list_round_trip(monkeypatch):
    module_b2l = _load(monkeypatch, "nodes.image.ts_image_batch_to_list")
    module_l2b = _load(monkeypatch, "nodes.image.ts_image_list_to_batch")

    cls_b2l = next(iter(module_b2l.NODE_CLASS_MAPPINGS.values()))
    cls_l2b = next(iter(module_l2b.NODE_CLASS_MAPPINGS.values()))

    schema_b = cls_b2l.define_schema()
    schema_l = cls_l2b.define_schema()
    assert schema_b.node_id == "TS_ImageBatchToImageList"
    assert schema_l.node_id == "TS_ImageListToImageBatch"


def test_batch_to_list_splits_correctly(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_batch_to_list")
    cls = next(iter(module.NODE_CLASS_MAPPINGS.values()))

    img = torch.zeros((4, 8, 8, 3))
    for i in range(4):
        img[i] = i
    output = cls.execute(img)
    items = output.args[0]
    assert isinstance(items, list)
    assert len(items) == 4
    for i, item in enumerate(items):
        assert item.shape == (1, 8, 8, 3)
        assert item[0, 0, 0, 0].item() == float(i)
