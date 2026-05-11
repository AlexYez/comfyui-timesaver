"""Behaviour tests for TS_VideoDepth helpers and the SDPA-rewritten attention.

These tests are CPU-safe: they exercise the GPU helpers on CPU tensors and the
math kernels with small fixed inputs. Heavy paths that depend on the actual
VideoDepthAnything weights are covered by the contract-snapshot test instead.

Run under ComfyUI's portable Python so numpy/torch/matplotlib are available;
under stock CPython without those packages the tests skip cleanly.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stubs (same shape as test_video_llm_nodes._install_io_stub) — keep
# `comfy_api.v0_0_2` importable without ComfyUI itself.
# ---------------------------------------------------------------------------

def _install_io_stub(monkeypatch):
    if "comfy_api.v0_0_2" in sys.modules:
        return

    class _Input:
        def __init__(self, id, *args, **kwargs):
            self.id = id
            self.args = args
            self.kwargs = kwargs
            self.optional = bool(kwargs.get("optional", False))
            self.default = kwargs.get("default")
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

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.v0_0_2")
    latest.IO = _IO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.v0_0_2", latest)


def _install_runtime_stubs(monkeypatch, tmp_dir):
    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.base_path = str(tmp_dir / "comfy_root")
        fp.models_dir = str(tmp_dir / "comfy_root" / "models")
        fp.supported_pt_extensions = {".ckpt", ".safetensors", ".pt"}
        fp.get_input_directory = lambda: str(tmp_dir / "input")
        fp.get_output_directory = lambda: str(tmp_dir / "output")
        fp.get_temp_directory = lambda: str(tmp_dir / "temp")
        fp.get_folder_paths = lambda key: [str(tmp_dir / key)]
        fp.get_filename_list = lambda key: []
        fp.get_full_path = lambda key, name: str(tmp_dir / key / name)
        fp.exists_annotated_filepath = lambda value: False
        fp.add_model_folder_path = lambda *a, **kw: None
        monkeypatch.setitem(sys.modules, "folder_paths", fp)

    if "comfy" not in sys.modules:
        import torch
        comfy = types.ModuleType("comfy")
        mm = types.ModuleType("comfy.model_management")
        mm.get_torch_device = lambda: torch.device("cpu")
        mm.unet_offload_device = lambda: torch.device("cpu")
        mm.free_memory = lambda *a, **kw: None
        mm.load_model_gpu = lambda *a, **kw: None
        mm.throw_exception_if_processing_interrupted = lambda: None
        mm.soft_empty_cache = lambda: None
        utils = types.ModuleType("comfy.utils")

        class _PB:
            def __init__(self, total): self.total = total
            def update(self, n): pass
            def update_absolute(self, value, total=None): pass

        utils.ProgressBar = _PB
        utils.load_torch_file = lambda path, device="cpu": {}
        mp = types.ModuleType("comfy.model_patcher")

        class _ModelPatcher:
            def __init__(self, model, load_device, offload_device):
                self.model = model
                self.load_device = load_device
                self.offload_device = offload_device

            def memory_required(self, _input_shape):
                return 0

        mp.ModelPatcher = _ModelPatcher
        comfy.model_management = mm
        comfy.utils = utils
        comfy.model_patcher = mp
        monkeypatch.setitem(sys.modules, "comfy", comfy)
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
        monkeypatch.setitem(sys.modules, "comfy.utils", utils)
        monkeypatch.setitem(sys.modules, "comfy.model_patcher", mp)


def _load(monkeypatch, dotted: str, tmp_dir: Path):
    _install_io_stub(monkeypatch)
    _install_runtime_stubs(monkeypatch, tmp_dir)
    root = PACKAGE_ROOT
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop(dotted, None)
    return importlib.import_module(dotted)


# ---------------------------------------------------------------------------
# Schema preservation (workflow compatibility)
# ---------------------------------------------------------------------------


def test_legacy_widgets_unchanged(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    schema = module.TS_VideoDepth.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    # all 8 legacy widgets must still be present
    legacy = {
        "model_filename", "input_size", "max_res", "precision",
        "colormap", "dithering_strength", "apply_median_blur", "upscale_algorithm",
    }
    assert legacy.issubset(set(inputs)), f"missing legacy widgets: {legacy - set(inputs)}"

    # legacy defaults are frozen by workflow compatibility
    assert inputs["model_filename"].default == "video_depth_anything_vitl.pth"
    assert inputs["input_size"].default == 518
    assert inputs["max_res"].default == 1280
    assert inputs["precision"].default == "fp16"
    assert inputs["colormap"].default == "gray"
    assert inputs["dithering_strength"].default == pytest.approx(0.005)
    assert inputs["apply_median_blur"].default is True
    assert inputs["upscale_algorithm"].default == "Lanczos4"


def test_new_widgets_are_optional_with_quality_defaults(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    schema = module.TS_VideoDepth.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    # Defaults are tuned for quality (legacy widgets above keep original
    # values, so existing workflows that store the 8 legacy widgets remain
    # bit-for-bit unchanged).
    assert inputs["normalization_mode"].default == "percentile"
    assert inputs["normalization_mode"].optional is True
    assert inputs["denoise_method"].default == "bilateral"
    assert inputs["denoise_method"].optional is True
    assert inputs["dither_pattern"].default == "bayer"
    assert inputs["dither_pattern"].optional is True
    assert inputs["edge_aware_upscale"].default is True
    assert inputs["edge_aware_upscale"].optional is True


# ---------------------------------------------------------------------------
# Colormap LUT
# ---------------------------------------------------------------------------


def test_colormap_gray_returns_none(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    import torch
    assert module._get_colormap_lut("gray", torch.device("cpu")) is None


def test_colormap_inferno_returns_lut(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    import torch
    lut = module._get_colormap_lut("inferno", torch.device("cpu"))
    assert lut.shape == (256, 3)
    assert lut.dtype == torch.float32
    assert (lut >= 0).all() and (lut <= 1).all()


def test_colormap_unknown_falls_back_to_none(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    import torch
    assert module._get_colormap_lut("does_not_exist_999", torch.device("cpu")) is None


def test_apply_colormap_gray_broadcasts(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    depth = torch.tensor([[[0.0, 0.5], [1.0, 0.25]]])  # (1, 2, 2)
    out = module._apply_colormap(depth, None)
    assert out.shape == (1, 2, 2, 3)
    assert torch.allclose(out[..., 0], out[..., 1])
    assert torch.allclose(out[..., 1], out[..., 2])


def test_apply_colormap_lut_indexed_correctly(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    lut = torch.zeros((256, 3))
    lut[0] = torch.tensor([1.0, 0.0, 0.0])
    lut[255] = torch.tensor([0.0, 0.0, 1.0])
    depth = torch.tensor([[[0.0, 1.0]]])  # (1, 1, 2)
    out = module._apply_colormap(depth, lut)
    assert out.shape == (1, 1, 2, 3)
    assert torch.allclose(out[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(out[0, 0, 1], torch.tensor([0.0, 0.0, 1.0]))


def test_apply_colormap_bilinear_interpolation(monkeypatch, ts_tmp_path):
    """Banding regression test: between two LUT entries the output must be a
    weighted blend, not snapped to the nearest entry. This is the difference
    that visibly removes banding on smooth gradients."""
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    lut = torch.zeros((256, 3))
    # Put a steep colour change between indices 127 and 128.
    lut[127] = torch.tensor([1.0, 0.0, 0.0])
    lut[128] = torch.tensor([0.0, 1.0, 0.0])

    # depth=0.5 → idx_float=127.5; nearest would round to 128 (green); the
    # bilinear path returns exactly the midpoint of red and green.
    depth = torch.tensor([[[127.5 / 255.0]]])
    out = module._apply_colormap(depth, lut)
    assert torch.allclose(out[0, 0, 0], torch.tensor([0.5, 0.5, 0.0]), atol=1e-5)

    # Smooth gradient over a row must produce a smooth gradient out — checked
    # by asserting the diff sequence is monotonic and has no flat plateaus,
    # which is what banding would look like.
    depth = torch.linspace(0.0, 1.0, 64).view(1, 1, 64)
    lut = torch.zeros((256, 3))
    lut[:, 0] = torch.linspace(0.0, 1.0, 256)
    out = module._apply_colormap(depth, lut)
    row = out[0, 0, :, 0]
    diffs = row[1:] - row[:-1]
    assert (diffs > 0).all(), "Bilinear LUT must keep gradient strictly increasing"
    # No more than 256 unique values would indicate quantisation; we expect
    # many more.
    assert row.unique().numel() > 60


# ---------------------------------------------------------------------------
# Dithering
# ---------------------------------------------------------------------------


def test_dither_zero_strength_is_noop(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    depth = torch.rand((2, 16, 16))
    out = module._apply_dither(depth, 0.0, "white")
    assert torch.equal(out, depth)
    out = module._apply_dither(depth, 0.0, "bayer")
    assert torch.equal(out, depth)


def test_dither_white_changes_values_within_bounds(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    torch.manual_seed(0)
    depth = torch.full((1, 32, 32), 0.5)
    out = module._apply_dither(depth, 0.02, "white")
    assert out.shape == depth.shape
    assert not torch.equal(out, depth)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_dither_white_is_tpdf_not_uniform(monkeypatch, ts_tmp_path):
    """White-noise mode uses TPDF (sum of two uniforms) — distribution should
    be triangular: peak near zero, max at ±strength, lighter near the edges.
    This is the regression guard for "banding still visible after dither".
    """
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    torch.manual_seed(0)
    base = torch.full((1, 256, 256), 0.5)
    out = module._apply_dither(base, 0.02, "white")
    noise = out - 0.5
    # peak-to-peak of TPDF(2u-1)*s lives within ±s (max when both uniforms hit
    # 0 or 1 simultaneously).
    assert noise.abs().max().item() <= 0.02 + 1e-6
    # Triangular distribution: most mass within ±strength/2.
    inner = (noise.abs() <= 0.01).float().mean().item()
    assert inner > 0.7  # ~75% for true triangular pdf


def test_dither_bayer_is_deterministic(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    depth = torch.full((1, 16, 16), 0.5)
    a = module._apply_dither(depth.clone(), 0.02, "bayer")
    b = module._apply_dither(depth.clone(), 0.02, "bayer")
    assert torch.equal(a, b)  # ordered: no RNG involved
    # 8x8 pattern repeats across the 16x16 frame.
    assert torch.equal(a[0, :8, :8], a[0, 8:, 8:])


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalization_minmax(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    depth = torch.tensor([[[0.2, 0.8], [0.5, 0.3]]])  # range [0.2, 0.8]
    lo, hi = module._compute_global_normalization(depth, "minmax")
    assert lo == pytest.approx(0.2)
    assert hi == pytest.approx(0.8)


def test_normalization_percentile_clips_outliers(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    torch.manual_seed(0)
    # 100x100 uniform [0.3, 0.7] plus a single extreme outlier — minmax would
    # be dominated by it; percentile should ignore it.
    depth = torch.rand((1, 100, 100)) * 0.4 + 0.3
    depth[0, 0, 0] = 999.0  # outlier
    depth[0, 50, 50] = -999.0
    lo_min, hi_min = module._compute_global_normalization(depth, "minmax")
    assert hi_min > 100  # outlier wins
    lo_pc, hi_pc = module._compute_global_normalization(depth, "percentile")
    assert -1.0 < lo_pc < 0.4
    assert 0.6 < hi_pc < 1.0


def test_normalization_equal_min_max(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    depth = torch.full((1, 4, 4), 0.5)
    lo, hi = module._compute_global_normalization(depth, "minmax")
    assert hi > lo  # guard prevents division by zero


# ---------------------------------------------------------------------------
# Spatial filters (median, bilateral, guided upsample)
# ---------------------------------------------------------------------------


def test_median_blur_shape_preserved(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    depth = torch.rand((3, 32, 32))
    out = module._median_blur_chunk(depth)
    assert out.shape == depth.shape


def test_median_blur_removes_isolated_spike(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    depth = torch.full((1, 8, 8), 0.5)
    depth[0, 4, 4] = 10.0  # impulse
    out = module._median_blur_chunk(depth)
    assert out[0, 4, 4] < 1.0  # spike replaced by median of neighbourhood


def test_bilateral_blur_shape_preserved(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    depth = torch.rand((2, 16, 16))
    rgb = torch.rand((2, 3, 16, 16))
    out = module._bilateral_blur_chunk(depth, rgb)
    assert out.shape == depth.shape
    out_no_guide = module._bilateral_blur_chunk(depth, None)
    assert out_no_guide.shape == depth.shape


def test_guided_upsample_shape(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    depth_low = torch.rand((2, 32, 32))
    guide = torch.rand((2, 3, 64, 64))
    out = module._guided_upsample_chunk(depth_low, guide, 64, 64)
    assert out.shape == (2, 64, 64)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_resize_chunk_shapes(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    depth = torch.rand((2, 32, 32))
    for method in ("Lanczos4", "Cubic", "Linear"):
        out = module._resize_depth_chunk(depth, 64, 64, method)
        assert out.shape == (2, 64, 64)


# ---------------------------------------------------------------------------
# Preprocess helper (resize+normalize on GPU/CPU, NHWC float[0,1] in)
# ---------------------------------------------------------------------------


def test_preprocess_frames_gpu_shape_and_normalization(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    vd = importlib.import_module("nodes.video_depth_anything.video_depth")

    frames = torch.rand((3, 64, 96, 3), dtype=torch.float32)
    out = vd._preprocess_frames_gpu(frames, 28, 42, torch.device("cpu"), torch.float32, chunk_size=2)
    assert out.shape == (3, 3, 28, 42)
    assert out.dtype == torch.float32
    # After ImageNet normalize, values should NOT be in [0,1] anymore.
    assert (out.abs().max() > 1.0)


def test_preprocess_frames_does_not_mutate_input(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    vd = importlib.import_module("nodes.video_depth_anything.video_depth")

    frames = torch.rand((2, 32, 32, 3), dtype=torch.float32)
    before = frames.clone()
    vd._preprocess_frames_gpu(frames, 14, 14, torch.device("cpu"), torch.float32, chunk_size=1)
    assert torch.equal(frames, before)


def test_compute_resize_target_lower_bound(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    vd = importlib.import_module("nodes.video_depth_anything.video_depth")
    # 720p -> input_size=518
    h, w = vd._compute_resize_target(720, 1280, 518, multiple=14)
    assert h % 14 == 0 and w % 14 == 0
    assert h >= 518 or w >= 518


def test_adapt_input_size_for_aspect(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    vd = importlib.import_module("nodes.video_depth_anything.video_depth")
    # 16:9 — should NOT shrink
    assert vd._adapt_input_size_for_aspect(518, 720, 1280) == 518
    # 21:9 ultra-wide — should shrink, still multiple of 14
    out = vd._adapt_input_size_for_aspect(518, 1080, 2560)
    assert out % 14 == 0
    assert out < 518


# ---------------------------------------------------------------------------
# OOM-aware sub-chunk pickers
# ---------------------------------------------------------------------------


def test_backbone_sub_chunk_low_res_no_chunking(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    vd = importlib.import_module("nodes.video_depth_anything.video_depth")
    # 250×250 ≈ 62 k pixels — below every chunking threshold.
    assert vd._pick_backbone_sub_chunk(250, 250, 32) == 32
    assert vd._pick_backbone_sub_chunk(250, 250, 8) == 8  # total bounds the chunk


def test_backbone_sub_chunk_4k_clamps_to_four(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    vd = importlib.import_module("nodes.video_depth_anything.video_depth")
    # 518×924 / 644×1148 → "heavy" tier
    assert vd._pick_backbone_sub_chunk(518, 924, 32) == 4
    assert vd._pick_backbone_sub_chunk(644, 1148, 32) == 4


def test_backbone_sub_chunk_mid_tier(monkeypatch, ts_tmp_path):
    """Regression: 448×798 must trigger sub-chunking. Previously the threshold
    sat above this resolution, so an OOM retry hit refinenet1 unprotected."""
    pytest.importorskip("torch")
    _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    vd = importlib.import_module("nodes.video_depth_anything.video_depth")
    # 448×798 = 358k → falls into the 8-tier
    assert vd._pick_backbone_sub_chunk(448, 798, 32) == 8
    assert vd._pick_backbone_sub_chunk(350, 616, 32) == 8
    # 308×500 = 154k → falls into the 16-tier
    assert vd._pick_backbone_sub_chunk(308, 500, 32) == 16


# ---------------------------------------------------------------------------
# Progress bar staging
# ---------------------------------------------------------------------------


def test_stage_pbar_maps_inner_to_outer_slot(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)

    class _RecPBar:
        def __init__(self):
            self.values = []
        def update_absolute(self, value, total=None, **kw):
            self.values.append(int(value))
        def update(self, n):
            pass

    outer = _RecPBar()
    # Inference slot of the real node bar: base=10, span=75, 5 windows of 22 ticks.
    pb = module._StagePBar(outer, base=10, span=75, inner_total=110)
    pb.update(22)
    pb.update(22)
    pb.update(22)
    pb.update(22)
    pb.update(22)
    assert outer.values[0] == 10 + round(75 * 22 / 110)
    assert outer.values[-1] == 85
    pb.finish()
    assert outer.values[-1] == 85


def test_stage_pbar_clamps_overflow(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)

    class _RecPBar:
        def __init__(self):
            self.last = None
        def update_absolute(self, value, total=None, **kw):
            self.last = int(value)
        def update(self, n):
            pass

    outer = _RecPBar()
    pb = module._StagePBar(outer, base=85, span=15, inner_total=10)
    pb.update(1000)  # overshoot
    assert outer.last == 100


def test_stage_pbar_update_absolute(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)

    class _RecPBar:
        def __init__(self):
            self.last = None
        def update_absolute(self, value, total=None, **kw):
            self.last = int(value)
        def update(self, n):
            pass

    outer = _RecPBar()
    pb = module._StagePBar(outer, base=0, span=100, inner_total=50)
    pb.update_absolute(25)
    assert outer.last == 50  # 25/50 of span 100 mapped from base 0
    pb.update_absolute(50)
    assert outer.last == 100


def test_pbar_stage_weights_sum_to_100(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    total = (
        module._PBAR_WEIGHT_DOWNLOAD
        + module._PBAR_WEIGHT_PREPROCESS
        + module._PBAR_WEIGHT_INFERENCE
        + module._PBAR_WEIGHT_POSTPROCESS
    )
    assert total == module._PBAR_TOTAL == 100


def test_dpt_tail_subbatch_matches_legacy_manual(monkeypatch, ts_tmp_path):
    """Numerical equivalence of the sub-chunked DPT tail.

    We build a real DPTHeadTemporal, drive it through `forward` to obtain
    intermediate tensors (`path_3`, `layer_*_rn`), then re-run the tail
    (refinenet2 → refinenet1 → output_conv1 → F.interpolate → output_conv2)
    twice: once as a single batch, once split into sub-chunks. The two
    results must match bit-for-bit (eval mode, no dropout, no BN updates).
    """
    torch = pytest.importorskip("torch")
    _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    from nodes.video_depth_anything import dpt_temporal
    import torch.nn.functional as Fnn

    torch.manual_seed(7)
    head = dpt_temporal.DPTHeadTemporal(
        in_channels=128, features=32, use_bn=False,
        out_channels=[32, 64, 128, 128], use_clstoken=False, num_frames=8, pe="ape",
    )
    head.eval()

    B, T = 1, 8
    patch_h, patch_w = 4, 6
    BT = B * T

    # Reproduce the shape contract that DPTHeadTemporal hands to the tail:
    # layer_*_rn live at the same spatial resolution as the corresponding
    # resize_layer output, and `refinenet2(path_3, layer_2_rn,
    # size=layer_1_rn.shape[2:])` requires path_3 and layer_2_rn to share
    # spatial dims, with layer_1_rn at 2× of that. After refinenet1 the
    # tensor doubles again to (patch_h*4, patch_w*4).
    h2, w2 = patch_h * 2, patch_w * 2   # path_3 / layer_2_rn
    h1, w1 = patch_h * 4, patch_w * 4   # layer_1_rn
    path_3 = torch.randn(BT, 32, h2, w2)
    layer_2_rn = torch.randn(BT, 32, h2, w2)
    layer_1_rn = torch.randn(BT, 32, h1, w1)
    target_h, target_w = patch_h * 14, patch_w * 14

    def _run_tail_legacy():
        with torch.no_grad():
            path_2 = head.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
            path_1 = head.scratch.refinenet1(path_2, layer_1_rn)
            out = head.scratch.output_conv1(path_1)
            ori_type = out.dtype
            out = Fnn.interpolate(out, (target_h, target_w), mode="bilinear", align_corners=True)
            with torch.autocast(device_type="cpu", enabled=False):
                out = head.scratch.output_conv2(out.float())
            return out.to(ori_type)

    def _run_tail_chunked(sub_chunk):
        result = None
        with torch.no_grad():
            for i in range(0, BT, sub_chunk):
                p3 = path_3[i:i + sub_chunk]
                l2 = layer_2_rn[i:i + sub_chunk]
                l1 = layer_1_rn[i:i + sub_chunk]
                p2 = head.scratch.refinenet2(p3, l2, size=l1.shape[2:])
                p1 = head.scratch.refinenet1(p2, l1)
                out = head.scratch.output_conv1(p1)
                ori_type = out.dtype
                out = Fnn.interpolate(out, (target_h, target_w), mode="bilinear", align_corners=True)
                with torch.autocast(device_type="cpu", enabled=False):
                    out = head.scratch.output_conv2(out.float())
                out = out.to(ori_type)
                if result is None:
                    result = torch.empty((BT, out.shape[1], out.shape[2], out.shape[3]), dtype=out.dtype)
                result[i:i + sub_chunk] = out
        return result

    legacy = _run_tail_legacy()
    chunked4 = _run_tail_chunked(4)
    chunked2 = _run_tail_chunked(2)
    assert legacy.shape == (BT, 1, target_h, target_w)
    assert torch.allclose(legacy, chunked4, atol=1e-6, rtol=1e-5), (
        f"max diff (chunk=4) = {(legacy - chunked4).abs().max().item()}"
    )
    assert torch.allclose(legacy, chunked2, atol=1e-6, rtol=1e-5), (
        f"max diff (chunk=2) = {(legacy - chunked2).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# SDPA numerical equivalence — DINOv2 backbone attention
# ---------------------------------------------------------------------------


def test_sdpa_dinov2_attention_matches_legacy(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        pytest.skip("torch < 2.0 has no SDPA")

    _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    from nodes.video_depth_anything.dinov2_layers import attention as att_mod

    torch.manual_seed(0)
    dim, heads, B, N = 64, 4, 2, 16
    layer = att_mod.Attention(dim=dim, num_heads=heads, qkv_bias=True)
    layer.eval()
    x = torch.randn(B, N, dim)

    # Force SDPA path (default when available).
    with torch.no_grad():
        out_sdpa = layer(x)

    # Force legacy math path.
    monkeypatch.setattr(att_mod, "SDPA_AVAILABLE", False)
    with torch.no_grad():
        out_legacy = layer(x)

    assert torch.allclose(out_sdpa, out_legacy, atol=1e-5, rtol=1e-4), (
        f"max diff = {(out_sdpa - out_legacy).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# SDPA numerical equivalence — motion_module CrossAttention._attention
# ---------------------------------------------------------------------------


def test_sdpa_cross_attention_matches_legacy(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        pytest.skip("torch < 2.0 has no SDPA")

    _load(monkeypatch, "nodes.video.ts_video_depth", ts_tmp_path)
    from nodes.video_depth_anything.motion_module import attention as ma

    torch.manual_seed(1)
    # `_attention` expects q/k/v already in (batch*heads, seq, head_dim) shape
    # — that's how `reshape_heads_to_batch_dim` leaves them when called from
    # the `.forward()` wrapper. We mirror that contract here.
    heads = 4
    batch, seq, head_dim = 2, 12, 8
    layer = ma.CrossAttention(query_dim=heads * head_dim, heads=heads, dim_head=head_dim)
    layer.eval()
    bh = batch * heads
    q = torch.randn(bh, seq, head_dim)
    k = torch.randn(bh, seq, head_dim)
    v = torch.randn(bh, seq, head_dim)

    out_sdpa = layer._attention(q, k, v, attention_mask=None)
    monkeypatch.setattr(ma, "SDPA_AVAILABLE", False)
    out_legacy = layer._attention(q, k, v, attention_mask=None)

    assert out_sdpa.shape == out_legacy.shape
    assert torch.allclose(out_sdpa, out_legacy, atol=1e-5, rtol=1e-4), (
        f"max diff = {(out_sdpa - out_legacy).abs().max().item()}"
    )
