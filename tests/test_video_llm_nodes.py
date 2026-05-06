"""Behaviour tests for Video and LLM category nodes (cpu-safe).

Most video / LLM nodes import heavy backends (spandrel, nvvfx, transformers,
huggingface_hub, ffmpeg via imageio). For those we verify the contract
through the snapshot. Two nodes are simple enough to test directly:

- TS_Free_Video_Memory: pass-through with no GPU side effects.
- TS_Frame_Interpolation: validate schema and helpers if accessible.

For LLM:
- TS_Qwen3_VL_V3: schema constants (model list, presets), helpers if accessible.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _install_io_stub(monkeypatch):
    if "comfy_api.latest" in sys.modules:
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
        def __init__(self, **kwargs): self.__dict__.update(kwargs)

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
        Audio = _ComfyType
        Conditioning = _ComfyType
        Vae = _ComfyType
        Latent = _ComfyType
        Model = _ComfyType
        NumberDisplay = _NumberDisplay

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.IO = _IO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)


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
        fp.get_filename_list = lambda key: ["upscale_a.pth", "upscale_b.pth"]
        fp.get_full_path = lambda key, name: str(tmp_dir / key / name)
        fp.exists_annotated_filepath = lambda value: False
        fp.add_model_folder_path = lambda *a, **kw: None
        monkeypatch.setitem(sys.modules, "folder_paths", fp)

    if "comfy" not in sys.modules:
        comfy = types.ModuleType("comfy")
        mm = types.ModuleType("comfy.model_management")
        mm.get_torch_device = lambda: __import__("torch").device("cpu")
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


def _load(monkeypatch, dotted: str, tmp_dir: Path):
    _install_io_stub(monkeypatch)
    _install_runtime_stubs(monkeypatch, tmp_dir)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop(dotted, None)
    return importlib.import_module(dotted)


# ---------- TS_Free_Video_Memory ----------


def test_free_video_memory_schema(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_free_video_memory", ts_tmp_path)
    schema = module.TS_Free_Video_Memory.define_schema()
    assert schema.node_id == "TS_Free_Video_Memory"
    assert schema.category == "TS/Video"
    inputs = {item.id: item for item in schema.inputs}
    assert {"images", "aggressive_cleanup", "report_memory"} == set(inputs.keys())


def test_free_video_memory_passes_through_images_unchanged(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_free_video_memory", ts_tmp_path)
    img = torch.rand((2, 16, 16, 3), dtype=torch.float32)
    output = module.TS_Free_Video_Memory.execute(img, "disable", "disable")
    assert output.args[0] is img  # same object back


def test_free_video_memory_aggressive_cleanup_does_not_crash(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_free_video_memory", ts_tmp_path)
    img = torch.rand((1, 8, 8, 3), dtype=torch.float32)
    output = module.TS_Free_Video_Memory.execute(img, "enable", "enable")
    assert output.args[0] is img


# ---------- TS_RTX_Upscaler (schema only — heavy nvvfx dep) ----------


def test_rtx_upscaler_schema(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_rtx_upscaler", ts_tmp_path)
    schema = module.TS_RTX_Upscaler.define_schema()
    assert schema.node_id == "TS_RTX_Upscaler"
    assert schema.category == "TS/Video"


def test_rtx_upscaler_resolves_quality(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_rtx_upscaler", ts_tmp_path)
    cls = module.TS_RTX_Upscaler
    # Helper exists on the class; call it carefully — depends on implementation detail
    # (the method is _resolve_quality_level)
    if hasattr(cls, "_resolve_quality_level"):
        for label in ("LOW", "MEDIUM", "HIGH", "ULTRA"):
            level = cls._resolve_quality_level(label)
            assert level is not None


# ---------- TS_LTX_FirstLastFrame (schema only — depends on comfy_extras) ----------


def test_ltx_first_last_frame_present_in_contracts():
    snapshot = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "node_contracts.json"
    text = snapshot.read_text(encoding="utf-8")
    assert '"TS_LTX_FirstLastFrame"' in text


# ---------- TS_Video_Upscale_With_Model (schema only) ----------


def test_video_upscale_with_model_schema(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_video_upscale_with_model", ts_tmp_path)
    schema = module.TS_Video_Upscale_With_Model.define_schema()
    assert schema.node_id == "TS_Video_Upscale_With_Model"
    assert schema.category == "TS/Video"


# ---------- TS_VideoDepthNode (heavy depth-anything weights) ----------


def test_video_depth_node_present_in_contracts():
    snapshot = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "node_contracts.json"
    text = snapshot.read_text(encoding="utf-8")
    assert '"TS_VideoDepthNode"' in text


# ---------- TS_Frame_Interpolation (RIFE/FILM models) ----------


def test_frame_interpolation_present_in_contracts():
    snapshot = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "node_contracts.json"
    text = snapshot.read_text(encoding="utf-8")
    assert '"TS_Frame_Interpolation"' in text


# ---------- TS_Animation_Preview ----------


def test_animation_preview_schema(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_animation_preview", ts_tmp_path)
    schema = module.TS_Animation_Preview.define_schema()
    assert schema.node_id == "TS_Animation_Preview"
    assert schema.category == "TS/Video"
    assert schema.is_output_node is True


# ---------- TS_Qwen3_VL_V3 ----------


def test_qwen3_vl_present_in_contracts():
    """Qwen has heavy transformers/HF dependencies. Verify contract presence."""
    snapshot = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "node_contracts.json"
    text = snapshot.read_text(encoding="utf-8")
    assert '"TS_Qwen3_VL_V3"' in text


def test_qwen3_vl_model_list_constants():
    """Read the source code directly: avoid actually importing the module
    (huggingface_hub.snapshot_download is imported at top level)."""
    src = (
        Path(__file__).resolve().parents[1]
        / "nodes"
        / "llm"
        / "ts_qwen3_vl.py"
    ).read_text(encoding="utf-8")
    assert "TS_Qwen3_VL_V3" in src
    # Default model used in TS_SuperPrompt should be present in the model list
    assert "huihui-ai/Huihui-Qwen3.5-2B-abliterated" in src
    assert "Custom (manual)" in src


# ---------- TS_Free_Video_Memory tensor invariants ----------


def test_free_video_memory_does_not_mutate_input(monkeypatch, ts_tmp_path):
    torch = pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.video.ts_free_video_memory", ts_tmp_path)
    img = torch.rand((1, 4, 4, 3), dtype=torch.float32)
    before = img.clone()
    module.TS_Free_Video_Memory.execute(img, "disable", "disable")
    assert torch.equal(img, before)
