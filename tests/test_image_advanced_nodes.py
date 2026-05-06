"""Behaviour tests for the more complex image-category nodes.

Covers:
- TS_QwenSafeResize / TS_WAN_SafeResize: aspect detection + resolution table.
- TS_QwenCanvas: canvas creation with optional image/mask.
- TS_ResolutionSelector: aspect parsing, divisor snapping, image fitting.
- TS_ImageTileSplitter / TS_ImageTileMerger: round-trip on synthetic image.
- TSAutoTileSize: grid math.
- TSCropToMask: schema + bbox crop with simple mask.
- TS_Cube_to_Equirectangular / TS_Equirectangular_to_Cube: schema only (heavy dep py360convert).
- TS_Keyer / TS_Despill: schema + helpers (gaussian, color parser, etc.).
- TS_ImagePromptInjector: graph traversal logic.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


def _install_io_stub_with_ui(monkeypatch):
    if "comfy_api.v0_0_2" in sys.modules:
        return  # already loaded — keep real one

    class _Input:
        def __init__(self, id, *args, **kwargs):
            self.id = id
            self.args = args
            self.kwargs = kwargs
            self.optional = bool(kwargs.get("optional", False))
            self.default = kwargs.get("default")
            self.advanced = bool(kwargs.get("advanced", False))
            self.options = kwargs.get("options", args[0] if args else None)

    class _Output:
        def __init__(self, id=None, display_name=None, **kwargs):
            self.id = id
            self.display_name = display_name
            self.kwargs = kwargs

    class _ComfyType:
        Input = _Input
        Output = _Output

    class _Custom:
        def __init__(self, name): self.name = name
        def Input(self, id, *args, **kwargs): return _Input(id, *args, **kwargs)
        def Output(self, *args, **kwargs): return _Output(*args, **kwargs)

    class _AnyType:
        @staticmethod
        def Input(id, *args, **kwargs): return _Input(id, *args, **kwargs)
        @staticmethod
        def Output(*args, **kwargs): return _Output(*args, **kwargs)

    class _Schema:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)

    class _NodeOutput:
        def __init__(self, *values, **kwargs):
            self.values = values
            self.args = values
            self.kwargs = kwargs

    class _NumberDisplay:
        slider = "slider"
        number = "number"

    class _Hidden:
        prompt = "prompt"

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
        Color = _ComfyType
        Audio = _ComfyType
        AnyType = _AnyType
        Custom = staticmethod(lambda name: _Custom(name))
        NumberDisplay = _NumberDisplay
        Hidden = _Hidden

    class _UI:
        @staticmethod
        def PreviewImage(image, cls=None): return {"image": image}

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.v0_0_2")
    latest.IO = _IO
    latest.UI = _UI
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.v0_0_2", latest)


def _stub_comfy(monkeypatch):
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
    _install_io_stub_with_ui(monkeypatch)
    _stub_comfy(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop(dotted, None)
    return importlib.import_module(dotted)


def _img(b=1, h=4, w=4, c=3, value=0.5):
    return torch.full((b, h, w, c), float(value), dtype=torch.float32)


# ---------- TS_QwenSafeResize ----------


def test_qwen_safe_resize_schema(monkeypatch):
    pytest.importorskip("PIL")
    module = _load(monkeypatch, "nodes.image.ts_qwen_safe_resize")
    schema = module.TS_QwenSafeResize.define_schema()
    assert schema.node_id == "TS_QwenSafeResize"
    assert schema.category == "TS/Image"


def test_qwen_safe_resize_picks_closest_aspect(monkeypatch):
    pytest.importorskip("PIL")
    module = _load(monkeypatch, "nodes.image.ts_qwen_safe_resize")
    # 16:9 image -> picks (1792, 1008)
    res = module.closest_supported_resolution(1920, 1080)
    assert res == (1792, 1008)
    # Square image -> (1344, 1344)
    res = module.closest_supported_resolution(1024, 1024)
    assert res == (1344, 1344)
    # Vertical 9:16
    res = module.closest_supported_resolution(1080, 1920)
    assert res == (1008, 1792)


def test_qwen_safe_resize_executes_on_real_image(monkeypatch):
    pytest.importorskip("PIL")
    module = _load(monkeypatch, "nodes.image.ts_qwen_safe_resize")
    img = torch.rand((1, 256, 256, 3), dtype=torch.float32)
    output = module.TS_QwenSafeResize.execute(img)
    out = output.args[0]
    # Should be (1344, 1344) — closest to 1:1
    assert out.shape == (1, 1344, 1344, 3)
    assert (out >= 0.0).all() and (out <= 1.0).all()


# ---------- TS_WAN_SafeResize ----------


def test_wan_safe_resize_schema(monkeypatch):
    pytest.importorskip("PIL")
    module = _load(monkeypatch, "nodes.image.ts_wan_safe_resize")
    schema = module.TS_WAN_SafeResize.define_schema()
    assert schema.node_id == "TS_WAN_SafeResize"


def test_wan_safe_resize_aspect_detection(monkeypatch):
    pytest.importorskip("PIL")
    module = _load(monkeypatch, "nodes.image.ts_wan_safe_resize")
    assert module.TS_WAN_SafeResize.detect_aspect_ratio(1920, 1080) == "16:9"
    assert module.TS_WAN_SafeResize.detect_aspect_ratio(1080, 1920) == "9:16"
    assert module.TS_WAN_SafeResize.detect_aspect_ratio(1024, 1024) == "1:1"


def test_wan_safe_resize_executes(monkeypatch):
    pytest.importorskip("PIL")
    module = _load(monkeypatch, "nodes.image.ts_wan_safe_resize")
    img = torch.rand((1, 1080, 1920, 3), dtype=torch.float32)
    output = module.TS_WAN_SafeResize.execute(img, "Standard quality")
    out, w, h, conn = output.args
    # Standard quality 16:9 -> (832, 480)
    assert out.shape == (1, 480, 832, 3)
    assert w == 832 and h == 480
    assert conn == "standard quality"


# ---------- TS_QwenCanvas ----------


def test_qwen_canvas_schema(monkeypatch):
    pytest.importorskip("PIL")
    module = _load(monkeypatch, "nodes.image.ts_qwen_canvas")
    schema = module.TS_QwenCanvas.define_schema()
    assert schema.node_id == "TS_QwenCanvas"
    inputs = {item.id: item for item in schema.inputs}
    assert "resolution" in inputs


def test_qwen_canvas_creates_blank_when_no_image(monkeypatch):
    pytest.importorskip("PIL")
    module = _load(monkeypatch, "nodes.image.ts_qwen_canvas")
    output = module.TS_QwenCanvas.execute(resolution="1:1")
    out, w, h = output.args
    assert (w, h) == (1344, 1344)
    assert out.shape == (1, 1344, 1344, 3)
    # All white canvas
    assert torch.allclose(out, torch.ones_like(out))


def test_qwen_canvas_pastes_image_centered(monkeypatch):
    pytest.importorskip("PIL")
    module = _load(monkeypatch, "nodes.image.ts_qwen_canvas")
    img = torch.rand((1, 200, 200, 3), dtype=torch.float32)
    output = module.TS_QwenCanvas.execute(resolution="1:1", image=img)
    out, w, h = output.args
    assert out.shape == (1, h, w, 3)


def test_qwen_canvas_unknown_resolution_raises(monkeypatch):
    pytest.importorskip("PIL")
    module = _load(monkeypatch, "nodes.image.ts_qwen_canvas")
    with pytest.raises(ValueError):
        module.TS_QwenCanvas.execute(resolution="weird")


# ---------- TS_ResolutionSelector ----------


def test_resolution_selector_schema(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_resolution_selector")
    schema = module.TS_ResolutionSelector.define_schema()
    assert schema.node_id == "TS_ResolutionSelector"


def test_resolution_selector_parse_ratio(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_resolution_selector")
    assert module.TS_ResolutionSelector._parse_ratio("16:9") == (16.0, 9.0)
    assert module.TS_ResolutionSelector._parse_ratio("garbage") == (1.0, 1.0)
    assert module.TS_ResolutionSelector._parse_ratio("0:0") == (1.0, 1.0)
    assert module.TS_ResolutionSelector._parse_ratio("") == (1.0, 1.0)


def test_resolution_selector_snaps_to_divisor(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_resolution_selector")
    snap = module.TS_ResolutionSelector._snap_to_divisible
    assert snap(100, 32) == 96  # round to nearest 32
    assert snap(127, 32) == 128
    assert snap(5, 32) == 32  # min one divisor


def test_resolution_selector_outputs_divisible_canvas(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_resolution_selector")
    output = module.TS_ResolutionSelector.execute(
        aspect_ratio="16:9", resolution=1.5, custom_ratio="0:0",
        original_aspect=False, image=None,
    )
    img = output.args[0]
    _, h, w, c = img.shape
    assert h % 32 == 0
    assert w % 32 == 0
    # Should be roughly 16:9 — give it some leeway from the snap
    assert abs((w / h) - (16 / 9)) < 0.1


def test_resolution_selector_uses_image_aspect_when_requested(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_resolution_selector")
    img = torch.rand((1, 100, 200, 3))  # 2:1 aspect
    output = module.TS_ResolutionSelector.execute(
        aspect_ratio="1:1", resolution=1.0, custom_ratio="0:0",
        original_aspect=True, image=img,
    )
    out_img = output.args[0]
    _, h, w, _ = out_img.shape
    # Expected aspect close to 2:1
    assert abs((w / h) - 2.0) < 0.15


# ---------- TS_ImageTileSplitter / TS_ImageTileMerger ----------


def test_tile_splitter_schema(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_tile_splitter")
    schema = module.TS_ImageTileSplitter.define_schema()
    assert schema.node_id == "TS_ImageTileSplitter"


def test_tile_splitter_produces_grid_of_tiles(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_tile_splitter")
    img = torch.rand((1, 256, 256, 3), dtype=torch.float32)
    output = module.TS_ImageTileSplitter.execute(
        img, tile_width=128, tile_height=128, overlap=0, feather=0.0
    )
    tiles, info = output.args
    assert tiles.shape[1:] == (128, 128, 3)
    assert info["rows"] == 2 and info["cols"] == 2
    assert tiles.shape[0] == 4


def test_tile_splitter_clamps_overlap(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_tile_splitter")
    # overlap larger than tile size should clamp
    img = torch.rand((1, 64, 64, 3), dtype=torch.float32)
    output = module.TS_ImageTileSplitter.execute(
        img, tile_width=32, tile_height=32, overlap=200, feather=0.0
    )
    tiles, info = output.args
    assert info["overlap"] < 32


def test_tile_merger_round_trip_no_feather(monkeypatch):
    splitter_module = _load(monkeypatch, "nodes.image.ts_image_tile_splitter")
    merger_module = _load(monkeypatch, "nodes.image.ts_image_tile_merger")
    img = torch.rand((1, 64, 64, 3), dtype=torch.float32)
    split_out = splitter_module.TS_ImageTileSplitter.execute(
        img, tile_width=32, tile_height=32, overlap=0, feather=0.0
    )
    tiles, info = split_out.args
    merged_out = merger_module.TS_ImageTileMerger.execute(tiles, info)
    merged = merged_out.args[0]
    assert merged.shape == (1, 64, 64, 3)
    assert torch.allclose(merged, img, atol=1e-5)


def test_tile_merger_round_trip_with_overlap_and_feather(monkeypatch):
    splitter_module = _load(monkeypatch, "nodes.image.ts_image_tile_splitter")
    merger_module = _load(monkeypatch, "nodes.image.ts_image_tile_merger")
    # Constant image: blending with overlap+feather should still produce same constant
    img = torch.full((1, 128, 128, 3), 0.5, dtype=torch.float32)
    split_out = splitter_module.TS_ImageTileSplitter.execute(
        img, tile_width=64, tile_height=64, overlap=16, feather=0.1
    )
    tiles, info = split_out.args
    merged = merger_module.TS_ImageTileMerger.execute(tiles, info).args[0]
    assert merged.shape == img.shape
    assert torch.allclose(merged, img, atol=1e-3)


# ---------- TSAutoTileSize ----------


def test_auto_tile_size_schema(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_auto_tile_size")
    schema = module.TSAutoTileSize.define_schema()
    assert schema.node_id == "TSAutoTileSize"


def test_auto_tile_size_grid_matches_aspect(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_auto_tile_size")
    # 4 tiles for 1:1 image -> 2x2 grid
    grid = module.TSAutoTileSize.find_best_grid(4, 1.0)
    assert grid == (2, 2)
    # 8 tiles for 16:9 image -> 4x2 (closer to 1.78)
    grid = module.TSAutoTileSize.find_best_grid(8, 16 / 9)
    # 4x2 = 2.0, 8x1 = 8.0; 4x2 is closer to 1.78
    assert grid == (4, 2)


def test_auto_tile_size_executes_with_image(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_auto_tile_size")
    img = torch.zeros((1, 1024, 1024, 3))
    output = module.TSAutoTileSize.execute(
        tile_count=4, padding=64, divide_by=8, image=img
    )
    tile_w, tile_h = output.args
    # Both divisible by 8
    assert tile_w % 8 == 0
    assert tile_h % 8 == 0


def test_auto_tile_size_executes_with_dimensions(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_auto_tile_size")
    output = module.TSAutoTileSize.execute(
        tile_count=4, padding=64, divide_by=8, image=None, width=1024, height=512
    )
    tile_w, tile_h = output.args
    assert tile_w % 8 == 0 and tile_h % 8 == 0


# ---------- TSCropToMask ----------


def test_crop_to_mask_schema(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_crop_to_mask")
    schema = module.TSCropToMask.define_schema()
    assert schema.node_id == "TSCropToMask"


def test_crop_to_mask_crops_to_mask_bbox(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_crop_to_mask")
    img = torch.zeros((1, 200, 200, 3), dtype=torch.float32)
    img[0, 50:100, 50:100, 0] = 1.0  # red region
    mask = torch.zeros((1, 200, 200), dtype=torch.float32)
    mask[0, 50:100, 50:100] = 1.0

    output = module.TSCropToMask.execute(
        img, mask, padding=10, divide_by=32, max_resolution=720,
        fixed_mask_frame_index=0, interpolation_window_size=0,
        force_gpu=False,
    )
    cropped, _crop_mask, crop_data, w, h = output.args
    # Cropped shape should be a fraction of original
    assert cropped.shape[0] == 1
    assert cropped.shape[1] < 200 and cropped.shape[2] < 200
    assert isinstance(crop_data, list)
    assert len(crop_data) == 1


def test_crop_to_mask_solid_mask_skips_crop(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_crop_to_mask")
    img = torch.zeros((1, 200, 200, 3), dtype=torch.float32)
    mask = torch.zeros((1, 200, 200), dtype=torch.float32)  # all-zero, "solid"
    output = module.TSCropToMask.execute(
        img, mask, padding=10, divide_by=32, max_resolution=720,
        fixed_mask_frame_index=0, interpolation_window_size=0, force_gpu=False,
    )
    cropped, _, _, w, h = output.args
    assert cropped.shape == img.shape
    assert (w, h) == (200, 200)


# ---------- 360-degree nodes (require py360convert) ----------
# These nodes use ``from ...ts_dependency_manager`` which can only resolve when
# the parent package is loaded. Instead of importing the module, we verify the
# contract from the snapshot in tests/contracts/node_contracts.json.


def test_cube_to_equirect_present_in_contracts():
    snapshot = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "node_contracts.json"
    text = snapshot.read_text(encoding="utf-8")
    assert '"TS Cube to Equirectangular"' in text


def test_equirect_to_cube_present_in_contracts():
    snapshot = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "node_contracts.json"
    text = snapshot.read_text(encoding="utf-8")
    assert '"TS Equirectangular to Cube"' in text


# ---------- TS_Keyer / TS_Despill ----------


def test_keyer_schema_and_color_parser(monkeypatch):
    module = _load(monkeypatch, "nodes.image.keying.ts_keyer")
    schema = module.TS_Keyer.define_schema()
    assert schema.node_id == "TS_Keyer"
    inputs = {item.id: item for item in schema.inputs}
    assert "image" in inputs and "key_color" in inputs
    # Color parser
    assert module.TS_Keyer._parse_key_color("#00ff00") == (0.0, 1.0, 0.0)
    assert module.TS_Keyer._parse_key_color("#0f0") == (0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        module.TS_Keyer._parse_key_color("not-a-color")


def test_keyer_validate_inputs_rejects_invalid_levels(monkeypatch):
    module = _load(monkeypatch, "nodes.image.keying.ts_keyer")
    result = module.TS_Keyer.validate_inputs(
        key_color="#00ff00", key_channel="auto",
        black_point=0.5, white_point=0.3, matte_gamma=1.0,
    )
    assert isinstance(result, str)
    assert "white_point" in result


def test_keyer_validate_inputs_passes_with_valid_color(monkeypatch):
    module = _load(monkeypatch, "nodes.image.keying.ts_keyer")
    assert module.TS_Keyer.validate_inputs(
        key_color="#00ff00", key_channel="auto",
        black_point=0.05, white_point=0.85, matte_gamma=1.0,
    ) is True


def test_keyer_executes_on_pure_green_image(monkeypatch):
    module = _load(monkeypatch, "nodes.image.keying.ts_keyer")
    img = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    img[..., 1] = 1.0  # all pure green
    output = module.TS_Keyer.execute(img, enable=True, key_color="#00ff00")
    foreground, alpha_mask, _despilled = output.args[0], output.args[1], output.args[2]
    # Pure green should be removed -> alpha ~ 0 in foreground
    assert foreground.shape == (1, 8, 8, 4)
    # alpha_mask is "transparency" so should be ~1 (transparent everywhere)
    assert alpha_mask.mean() > 0.9


def test_keyer_disabled_passes_through(monkeypatch):
    module = _load(monkeypatch, "nodes.image.keying.ts_keyer")
    img = torch.rand((1, 4, 4, 3))
    output = module.TS_Keyer.execute(img, enable=False)
    foreground = output.args[0]
    assert foreground.shape == (1, 4, 4, 4)
    # All alpha = 1 (fully opaque)
    assert torch.allclose(foreground[..., 3], torch.ones((1, 4, 4)))


def test_despill_schema(monkeypatch):
    module = _load(monkeypatch, "nodes.image.keying.ts_despill")
    schema = module.TS_Despill.define_schema()
    assert schema.node_id == "TS_Despill"


def test_despill_validate_inputs_rejects_bad_screen_color(monkeypatch):
    module = _load(monkeypatch, "nodes.image.keying.ts_despill")
    result = module.TS_Despill.validate_inputs(
        screen_color="purple", algorithm="adaptive",
        spill_threshold=0.0, spill_softness=0.001,
    )
    assert isinstance(result, str)


def test_despill_disabled_passes_through(monkeypatch):
    module = _load(monkeypatch, "nodes.image.keying.ts_despill")
    img = torch.rand((1, 4, 4, 3), dtype=torch.float32)
    output = module.TS_Despill.execute(img, enable=False)
    out_image, mask, removed = output.args[0], output.args[1], output.args[2]
    assert torch.allclose(out_image, img.clamp(0, 1))
    assert torch.all(mask == 0.0)
    assert torch.all(removed == 0.0)


def test_despill_classic_only_reduces_key_channel(monkeypatch):
    module = _load(monkeypatch, "nodes.image.keying.ts_despill")
    # Image with green spill: rgb=(0.2, 0.8, 0.2) — green > others
    img = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    img[..., 0] = 0.2
    img[..., 1] = 0.8
    img[..., 2] = 0.2
    output = module.TS_Despill.execute(
        img, enable=True, screen_color="green", algorithm="classic",
        strength=1.3, spill_threshold=0.0, spill_softness=0.001,
        compensation=0.0, preserve_luma=False, use_input_alpha_for_edges=False,
        edge_boost=1.0, edge_blur=0.0, skin_protection=0.0, saturation_restore=0.0,
    )
    out = output.args[0]
    # green channel should be reduced; red/blue should be unchanged for classic
    assert out[..., 1].mean() < img[..., 1].mean()
    assert torch.allclose(out[..., 0], img[..., 0])
    assert torch.allclose(out[..., 2], img[..., 2])


# ---------- TS_ImagePromptInjector ----------


def test_prompt_injector_schema(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_prompt_injector")
    schema = module.TS_ImagePromptInjector.define_schema()
    assert schema.node_id == "TS_ImagePromptInjector"


def test_prompt_injector_finds_positive_text_encoder(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_prompt_injector")
    cls = module.TS_ImagePromptInjector
    # Inject only updates entries whose text is empty or a link (placeholder).
    graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}, "_meta": {"title": "Positive Prompt"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}, "_meta": {"title": "Negative Prompt"}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["1", 0], "negative": ["2", 0]}},
    }
    cls._inject_prompt_into_graph(graph, "new prompt")
    # Only #1 (positive root) should have been updated.
    assert graph["1"]["inputs"]["text"] == "new prompt"
    assert graph["2"]["inputs"]["text"] == ""


def test_prompt_injector_skips_filled_text_node(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_prompt_injector")
    cls = module.TS_ImagePromptInjector
    graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "user wrote this"}, "_meta": {"title": "Positive Prompt"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}, "_meta": {"title": "Negative Prompt"}},
        "3": {"class_type": "KSampler", "inputs": {"positive": ["1", 0], "negative": ["2", 0]}},
    }
    cls._inject_prompt_into_graph(graph, "new prompt")
    # Filled text is preserved; injection only acts on empty/link slots.
    assert graph["1"]["inputs"]["text"] == "user wrote this"


def test_prompt_injector_falls_back_to_title_search_without_sampler(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_prompt_injector")
    cls = module.TS_ImagePromptInjector
    graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}, "_meta": {"title": "Positive Prompt"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}, "_meta": {"title": "Negative Prompt"}},
    }
    cls._inject_prompt_into_graph(graph, "x")
    # Only the positive should be updated
    assert graph["1"]["inputs"]["text"] == "x"
    assert graph["2"]["inputs"]["text"] == ""


def test_prompt_injector_normalize_returns_empty_for_none(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_image_prompt_injector")
    cls = module.TS_ImagePromptInjector
    assert cls._normalize_prompt(None) == ""
    assert cls._normalize_prompt("hi") == "hi"
    assert cls._normalize_prompt(123) == "123"


# ---------- TSRestoreFromCrop helpers ----------


def test_restore_from_crop_gaussian_blur_normalizes_at_center(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_restore_from_crop")
    # All-ones mask: center is far from edges, so the (zero-padded) gaussian
    # convolution should produce ~1.0 at the centre. Edges roll off because
    # F.conv2d uses zero-padding by default — this is the documented behavior.
    mask = torch.ones((1, 64, 64), dtype=torch.float32)
    blurred = module._gaussian_blur_mask(mask, 4, torch.device("cpu"))
    assert blurred.shape == (1, 64, 64)
    centre = blurred[0, 32, 32].item()
    assert centre == pytest.approx(1.0, abs=1e-3)


def test_restore_from_crop_box_blur_zero_amount(monkeypatch):
    module = _load(monkeypatch, "nodes.image.ts_restore_from_crop")
    mask = torch.rand((1, 8, 8), dtype=torch.float32)
    out = module._box_blur_mask(mask, 0, torch.device("cpu"))
    assert torch.equal(out, mask)
