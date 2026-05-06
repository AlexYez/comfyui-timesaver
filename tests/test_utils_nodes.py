"""Behaviour tests for the four Utils-category nodes.

Covers:
- TS_Math_Int: every operation, including div-by-zero fallback.
- TS_Smart_Switch: type-aware selection, auto-failover, type validation.
- TS_Int_Slider / TS_FloatSlider: simple pass-through.

The Utils nodes have no heavy deps beyond torch (Smart_Switch only). The
tests use the real comfy_api which is available in ComfyUI's bundled
Python; stubs are used for ``nodes._shared`` and module imports as a
safety net for non-ComfyUI test environments.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _install_stubs(monkeypatch):
    """Stub comfy_api.v0_0_2.IO if the real one is not importable.

    On the maintainer machine the real one is reachable from ComfyUI
    Python, but contributor CI may run with no ComfyUI checkout on
    sys.path — fall back to a minimal stub so the module imports.
    """
    if "comfy_api.v0_0_2" in sys.modules:
        return  # real one already loaded — keep it

    class _Input:
        def __init__(self, id, *args, **kwargs):
            self.id = id
            self.args = args
            self.kwargs = kwargs
            self.optional = bool(kwargs.get("optional", False))
            self.default = kwargs.get("default")
            self.min = kwargs.get("min")
            self.max = kwargs.get("max")
            self.step = kwargs.get("step")
            self.options = kwargs.get("options", args[0] if args else None)

    class _Output:
        def __init__(self, id=None, display_name=None, **kwargs):
            self.id = id
            self.display_name = display_name
            self.kwargs = kwargs

    class _ComfyType:
        Input = _Input
        Output = _Output

    class _AnyType:
        @staticmethod
        def Input(id, *args, **kwargs):
            return _Input(id, *args, **kwargs)

        @staticmethod
        def Output(*args, **kwargs):
            return _Output(*args, **kwargs)

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
        Int = _ComfyType
        Float = _ComfyType
        Boolean = _ComfyType
        Combo = _ComfyType
        AnyType = _AnyType
        NumberDisplay = _NumberDisplay

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.v0_0_2")
    latest.IO = _IO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.v0_0_2", latest)


def _load(monkeypatch, module_path: str):
    _install_stubs(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop(module_path, None)
    return importlib.import_module(module_path)


# -------------- TS_Math_Int --------------


@pytest.fixture
def math_int_module(monkeypatch):
    return _load(monkeypatch, "nodes.utils.ts_math_int")


def test_math_int_schema_contract(math_int_module):
    schema = math_int_module.TS_Math_Int.define_schema()
    assert schema.node_id == "TS_Math_Int"
    assert schema.display_name == "TS Math Int"
    assert schema.category == "TS/Utils"
    inputs = {item.id: item for item in schema.inputs}
    assert list(inputs) == ["a", "b", "operation"]
    assert inputs["a"].default == 0
    assert inputs["b"].default == 0
    assert math_int_module.NODE_CLASS_MAPPINGS == {"TS_Math_Int": math_int_module.TS_Math_Int}


@pytest.mark.parametrize(
    "a,b,op,expected",
    [
        (5, 3, "add (+)", 8),
        (5, 3, "subtract (-)", 2),
        (5, 3, "multiply (*)", 15),
        (10, 3, "divide (/)", 3),       # int(10/3) = 3 (truncate toward zero)
        (-10, 3, "divide (/)", -3),     # int(-10/3) = -3 (truncate toward zero)
        (10, 3, "floor_divide (//)", 3),
        (-10, 3, "floor_divide (//)", -4),  # python floor div rounds toward -inf
        (10, 3, "modulo (%)", 1),
        (2, 8, "power (**)", 256),
        (5, 3, "min", 3),
        (5, 3, "max", 5),
    ],
)
def test_math_int_operations(math_int_module, a, b, op, expected):
    output = math_int_module.TS_Math_Int.execute(a, b, op)
    assert output.args == (expected,)
    assert isinstance(output.args[0], int)


@pytest.mark.parametrize("op", ["divide (/)", "floor_divide (//)", "modulo (%)"])
def test_math_int_division_by_zero_returns_zero(math_int_module, op):
    output = math_int_module.TS_Math_Int.execute(7, 0, op)
    assert output.args == (0,)


def test_math_int_unknown_operation_returns_zero(math_int_module):
    output = math_int_module.TS_Math_Int.execute(5, 3, "bogus")
    assert output.args == (0,)


def test_math_int_returns_int_type_for_floats_in_power(math_int_module):
    # power may produce floats (negative exponents) — execute should int() it,
    # but we only test cases where pow is integer-valued.
    output = math_int_module.TS_Math_Int.execute(0, 0, "power (**)")
    assert output.args == (1,)  # 0**0 = 1 in python


# -------------- TS_Int_Slider / TS_FloatSlider --------------


def test_int_slider_passes_through(monkeypatch):
    module = _load(monkeypatch, "nodes.utils.ts_int_slider")
    schema = module.TS_Int_Slider.define_schema()
    assert schema.node_id == "TS_Int_Slider"
    assert schema.display_name == "TS Int Slider"
    assert schema.category == "TS/Utils"

    # Returns the same int back
    output = module.TS_Int_Slider.execute(512)
    assert output.args == (512,)
    assert isinstance(output.args[0], int)

    # Coerces float-like values to int
    output = module.TS_Int_Slider.execute(513.7)
    assert output.args == (513,)


def test_float_slider_passes_through(monkeypatch):
    module = _load(monkeypatch, "nodes.utils.ts_float_slider")
    schema = module.TS_FloatSlider.define_schema()
    assert schema.node_id == "TS_FloatSlider"
    assert schema.display_name == "TS Float Slider"
    assert schema.category == "TS/Utils"

    output = module.TS_FloatSlider.execute(0.5)
    assert output.args == (0.5,)
    assert isinstance(output.args[0], float)

    # Int passthrough -> coerced to float
    output = module.TS_FloatSlider.execute(1)
    assert output.args == (1.0,)
    assert isinstance(output.args[0], float)


# -------------- TS_Smart_Switch --------------


@pytest.fixture
def smart_switch_module(monkeypatch):
    pytest.importorskip("torch")
    return _load(monkeypatch, "nodes.utils.ts_smart_switch")


def test_smart_switch_schema_contract(smart_switch_module):
    schema = smart_switch_module.TS_Smart_Switch.define_schema()
    assert schema.node_id == "TS_Smart_Switch"
    assert schema.display_name == "TS Smart Switch"
    assert schema.category == "TS/Utils"
    inputs = {item.id: item for item in schema.inputs}
    assert list(inputs) == ["data_type", "switch", "input_1", "input_2"]
    assert inputs["input_1"].optional is True
    assert inputs["input_2"].optional is True


def test_smart_switch_string_picks_input_1_when_switch_on(smart_switch_module):
    out = smart_switch_module.TS_Smart_Switch.execute(
        "string", True, input_1="alpha", input_2="beta"
    )
    assert out.args == ("alpha",)


def test_smart_switch_string_picks_input_2_when_switch_off(smart_switch_module):
    out = smart_switch_module.TS_Smart_Switch.execute(
        "string", False, input_1="alpha", input_2="beta"
    )
    assert out.args == ("beta",)


def test_smart_switch_auto_failover_when_input_1_missing(smart_switch_module):
    out = smart_switch_module.TS_Smart_Switch.execute(
        "int", True, input_1=None, input_2=42
    )
    assert out.args == (42,)


def test_smart_switch_returns_none_when_both_inputs_invalid(smart_switch_module):
    out = smart_switch_module.TS_Smart_Switch.execute(
        "int", True, input_1=None, input_2=None
    )
    assert out.args == (None,)


def test_smart_switch_rejects_wrong_type(smart_switch_module):
    # data_type=int, but inputs are strings — they fail type check and are ignored
    out = smart_switch_module.TS_Smart_Switch.execute(
        "int", True, input_1="not-an-int", input_2="also-not"
    )
    assert out.args == (None,)


def test_smart_switch_image_validates_4d_tensor(smart_switch_module):
    import torch

    img = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    out = smart_switch_module.TS_Smart_Switch.execute(
        "images", True, input_1=img
    )
    assert out.args[0] is img


def test_smart_switch_image_rejects_3d_tensor(smart_switch_module):
    import torch

    bad = torch.zeros((4, 4, 3), dtype=torch.float32)
    out = smart_switch_module.TS_Smart_Switch.execute(
        "images", True, input_1=bad, input_2=None
    )
    # bad input is rejected, no failover available
    assert out.args == (None,)


def test_smart_switch_audio_validates_dict_with_waveform(smart_switch_module):
    import torch

    audio = {"waveform": torch.zeros((1, 1, 16000)), "sample_rate": 16000}
    out = smart_switch_module.TS_Smart_Switch.execute(
        "audio", True, input_1=audio
    )
    assert out.args[0] is audio


def test_smart_switch_audio_rejects_plain_dict(smart_switch_module):
    out = smart_switch_module.TS_Smart_Switch.execute(
        "audio", True, input_1={"sample_rate": 16000}, input_2=None
    )
    assert out.args == (None,)


def test_smart_switch_mask_accepts_3d_or_4d_with_single_channel(smart_switch_module):
    import torch

    mask3d = torch.zeros((1, 4, 4))
    out = smart_switch_module.TS_Smart_Switch.execute("mask", True, input_1=mask3d)
    assert out.args[0] is mask3d

    mask4d = torch.zeros((1, 4, 4, 1))
    out = smart_switch_module.TS_Smart_Switch.execute("mask", True, input_1=mask4d)
    assert out.args[0] is mask4d


def test_smart_switch_int_strict_against_bool(smart_switch_module):
    # bool is subclass of int in python, but the node MUST reject it.
    out = smart_switch_module.TS_Smart_Switch.execute(
        "int", True, input_1=True, input_2=None
    )
    assert out.args == (None,)


def test_smart_switch_video_accepts_5d_tensor(smart_switch_module):
    import torch

    video = torch.zeros((1, 4, 8, 8, 3))  # [B, T, H, W, C]
    out = smart_switch_module.TS_Smart_Switch.execute("video", True, input_1=video)
    assert out.args[0] is video


def test_smart_switch_video_accepts_list_of_4d_tensors(smart_switch_module):
    import torch

    frames = [torch.zeros((1, 4, 4, 3)) for _ in range(3)]
    out = smart_switch_module.TS_Smart_Switch.execute("video", True, input_1=frames)
    assert out.args[0] is frames
