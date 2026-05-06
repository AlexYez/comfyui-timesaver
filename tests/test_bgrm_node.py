from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("cv2")


class _ProgressBar:
    instances = []

    def __init__(self, total):
        self.total = total
        self.updates = []
        self.__class__.instances.append(self)

    def update_absolute(self, value, total=None):
        self.updates.append((value, total))


def _install_stubs(monkeypatch, root: Path) -> None:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(root / ".test_models")
    folder_paths.add_model_folder_path = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    comfy = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = _ProgressBar
    comfy.model_management = model_management
    comfy.utils = comfy_utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)

    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.hf_hub_download = lambda *args, **kwargs: str(root / ".test_models" / "unused")
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    safetensors = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)

    # Stub comfy_api.v0_0_2.IO so the V3 schema declaration in
    # ts_bgrm_birefnet imports without dragging in the full ComfyUI runtime.
    comfy_api_mod = types.ModuleType("comfy_api")
    latest_mod = types.ModuleType("comfy_api.v0_0_2")

    class _StubInput:
        def __init__(self, *args, **kwargs):
            self.id = args[0] if args else kwargs.get("id")
            self.args = args
            self.kwargs = kwargs

    class _StubOutput:
        def __init__(self, *args, **kwargs):
            self.id = kwargs.get("id")
            self.display_name = kwargs.get("display_name")
            self.args = args
            self.kwargs = kwargs

    class _StubComfyType:
        Input = _StubInput
        Output = _StubOutput

    class _StubSchema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _StubNodeOutput:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _StubIO:
        class ComfyNode:
            pass
        Schema = _StubSchema
        NodeOutput = _StubNodeOutput
        Image = _StubComfyType
        Mask = _StubComfyType
        Boolean = _StubComfyType
        Combo = _StubComfyType
        Int = _StubComfyType

    latest_mod.IO = _StubIO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api_mod)
    monkeypatch.setitem(sys.modules, "comfy_api.v0_0_2", latest_mod)


def _load_module(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    _install_stubs(monkeypatch, root)
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop("nodes.image.ts_bgrm_birefnet", None)
    return importlib.import_module("nodes.image.ts_bgrm_birefnet")


def test_bgrm_v3_contract_is_stable(monkeypatch):
    module = _load_module(monkeypatch)

    schema = module.TS_BGRM_BiRefNet.define_schema()
    input_ids = [item.id for item in schema.inputs]

    assert module.NODE_CLASS_MAPPINGS == {"TS_BGRM_BiRefNet": module.TS_BGRM_BiRefNet}
    assert module.NODE_DISPLAY_NAME_MAPPINGS == {"TS_BGRM_BiRefNet": "TS Remove Background"}
    assert schema.node_id == "TS_BGRM_BiRefNet"
    assert schema.display_name == "TS Remove Background"
    assert schema.category == "TS/Image"
    assert [out.display_name for out in schema.outputs] == ["IMAGE", "MASK", "MASK_IMAGE"]
    assert input_ids == [
        "image", "enable", "model",
        "use_custom_resolution", "process_resolution", "mask_blur", "mask_offset",
        "invert_output", "refine_foreground", "background", "background_color",
    ]


def test_bgrm_disabled_path_preserves_batch_and_input(monkeypatch):
    module = _load_module(monkeypatch)
    image = torch.rand((2, 8, 7, 3), dtype=torch.float32)
    before = image.clone()

    output = module.TS_BGRM_BiRefNet.execute(
        image,
        False,
        "BiRefNet_512x512",
        False,
        1024,
    )
    out_image, out_mask, out_mask_image = output.args

    assert torch.equal(image, before)
    assert out_image is image
    assert out_mask.shape == (2, 8, 7)
    assert out_mask_image.shape == (2, 8, 7, 3)
    assert torch.all(out_mask == 1.0)


def test_bgrm_cpu_uses_float32_not_half(monkeypatch):
    module = _load_module(monkeypatch)

    assert module._target_dtype(torch.device("cpu")) is torch.float32
    assert module._target_dtype(torch.device("cuda")) is torch.float16


def test_bgrm_process_masks_preserves_batch_on_cpu(monkeypatch):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    class FakeModel:
        def __init__(self):
            self.dtypes = []

        def __call__(self, input_tensor):
            self.dtypes.append(input_tensor.dtype)
            return [torch.zeros((input_tensor.shape[0], 1, input_tensor.shape[2], input_tensor.shape[3]))]

    model = module.BiRefNetModel()
    model.model = FakeModel()
    image = torch.rand((2, 6, 5, 3), dtype=torch.float32)

    masks = model.process_masks(image, {"process_res": 8})

    assert masks.shape == (2, 6, 5)
    assert model.model.dtypes == [torch.float32, torch.float32]


def test_bgrm_respects_model_management_cpu_choice(monkeypatch):
    """Regression for finding #3: previously _get_target_device silently
    overrode CPU back to cuda whenever CUDA was physically present, which
    broke `--cpu`, lowvram, and multi-GPU index selection. The fix trusts
    model_management.get_torch_device() unconditionally."""
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module.model_management, "get_torch_device", lambda: torch.device("cpu"))

    assert module._get_target_device().type == "cpu"


def test_bgrm_preserves_cuda_index_from_model_management(monkeypatch):
    """Multi-GPU users rely on get_torch_device returning the right index;
    the function must pass it through unchanged instead of forcing cuda:0."""
    module = _load_module(monkeypatch)
    target = torch.device("cuda", 1)
    monkeypatch.setattr(module.model_management, "get_torch_device", lambda: target)

    assert module._get_target_device() == target


def test_bgrm_process_masks_runs_in_half_on_gpu(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    module = _load_module(monkeypatch)
    monkeypatch.setattr(module.model_management, "get_torch_device", lambda: torch.device("cuda"))

    class FakeModel:
        def __init__(self):
            self.dtypes = []
            self.devices = []

        def __call__(self, input_tensor):
            self.dtypes.append(input_tensor.dtype)
            self.devices.append(input_tensor.device.type)
            return [torch.zeros(
                (input_tensor.shape[0], 1, input_tensor.shape[2], input_tensor.shape[3]),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )]

    model = module.BiRefNetModel()
    model.model = FakeModel()
    image = torch.rand((2, 6, 5, 3), dtype=torch.float32)

    masks = model.process_masks(image, {"process_res": 8})

    assert masks.shape == (2, 6, 5)
    assert masks.device.type == "cpu"
    assert model.model.devices == ["cuda"]
    assert model.model.dtypes == [torch.float16]


def test_bgrm_process_path_reports_progress_without_model_download(monkeypatch):
    module = _load_module(monkeypatch)
    _ProgressBar.instances.clear()

    class _FakeBg:
        def check_model_cache(self, model):
            return (True, "ok")

        def load_model(self, *args, **kwargs):
            return None

        def process_masks(self, image, params, progress_bar=None, start_step=55, end_step=80, target_device=None):
            return torch.ones((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32)

    monkeypatch.setattr(module._state, "model", _FakeBg())

    image = torch.rand((2, 6, 5, 3), dtype=torch.float32)
    output = module.TS_BGRM_BiRefNet.execute(
        image,
        True,
        "BiRefNet_512x512",
        False,
        1024,
        background="Color",
        background_color="white",
    )
    out_image, out_mask, out_mask_image = output.args

    assert out_image.shape == (2, 6, 5, 3)
    assert out_mask.shape == (2, 6, 5)
    assert out_mask_image.shape == (2, 6, 5, 3)
    assert _ProgressBar.instances[-1].updates[0] == (1, 100)
    assert _ProgressBar.instances[-1].updates[-1] == (100, 100)


def test_bgrm_logs_processing_device(monkeypatch, caplog):
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    _ProgressBar.instances.clear()

    class _FakeBg:
        def check_model_cache(self, model):
            return (True, "ok")

        def load_model(self, *args, **kwargs):
            return None

        def process_masks(self, image, params, progress_bar=None, start_step=55, end_step=80, target_device=None):
            return torch.ones((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32)

    monkeypatch.setattr(module._state, "model", _FakeBg())

    caplog.set_level(logging.INFO, logger=module.__name__)
    image = torch.rand((1, 4, 4, 3), dtype=torch.float32)
    module.TS_BGRM_BiRefNet.execute(
        image,
        True,
        "BiRefNet_512x512",
        False,
        1024,
    )

    assert "Processing device: cpu" in caplog.text
