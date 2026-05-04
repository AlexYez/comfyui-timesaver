"""Behaviour tests for TS_ImageListToImages.

The node is intentionally tolerant: it must accept None/empty input,
single tensors, lists shorter than the output slot count, and lists
longer than the output slot count, without raising.

Missing slots emit ExecutionBlocker (NOT a zero tensor placeholder) so
downstream reference-aware models are not fed a fake image.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("nodes.conditioning.ts_image_list_to_images", None)
    return importlib.import_module("nodes.conditioning.ts_image_list_to_images")


def _make_image(height: int = 32, width: int = 32, value: float = 0.5) -> torch.Tensor:
    return torch.full((1, height, width, 3), value, dtype=torch.float32)


def _is_blocker(value, blocker_cls) -> bool:
    return isinstance(value, blocker_cls)


def test_v1_contract_is_stable(monkeypatch):
    module = _load_module(monkeypatch)

    cls = module.TS_ImageListToImages
    assert cls.INPUT_IS_LIST == (True,)
    assert cls.RETURN_TYPES == ("IMAGE", "IMAGE", "IMAGE")
    assert cls.RETURN_NAMES == ("image_1", "image_2", "image_3")
    assert cls.FUNCTION == "split"
    assert cls.CATEGORY == "TS/Conditioning"

    assert "images" in cls.INPUT_TYPES().get("optional", {})
    assert module.NODE_CLASS_MAPPINGS == {
        "TS_ImageListToImages": module.TS_ImageListToImages,
    }
    assert module.NODE_DISPLAY_NAME_MAPPINGS == {
        "TS_ImageListToImages": "TS Image List to Images",
    }


def test_split_three_images_passes_through(monkeypatch):
    module = _load_module(monkeypatch)

    a = _make_image(value=0.1)
    b = _make_image(value=0.2)
    c = _make_image(value=0.3)

    out = module.TS_ImageListToImages().split(images=[a, b, c])

    assert len(out) == 3
    assert torch.equal(out[0], a)
    assert torch.equal(out[1], b)
    assert torch.equal(out[2], c)


def test_split_one_image_blocks_remaining_outputs(monkeypatch):
    module = _load_module(monkeypatch)

    a = _make_image(value=0.7)

    out = module.TS_ImageListToImages().split(images=[a])

    assert len(out) == 3
    assert torch.equal(out[0], a)
    assert _is_blocker(out[1], module.ExecutionBlocker)
    assert _is_blocker(out[2], module.ExecutionBlocker)


def test_empty_list_blocks_all_outputs(monkeypatch):
    module = _load_module(monkeypatch)

    out = module.TS_ImageListToImages().split(images=[])

    assert len(out) == 3
    for value in out:
        assert _is_blocker(value, module.ExecutionBlocker)


def test_none_input_blocks_all_outputs(monkeypatch):
    module = _load_module(monkeypatch)

    out = module.TS_ImageListToImages().split(images=None)

    assert len(out) == 3
    for value in out:
        assert _is_blocker(value, module.ExecutionBlocker)


def test_more_than_three_inputs_truncates_to_three(monkeypatch):
    module = _load_module(monkeypatch)

    images = [_make_image(value=v / 10.0) for v in range(5)]

    out = module.TS_ImageListToImages().split(images=images)

    assert len(out) == 3
    for index, tensor in enumerate(out):
        assert torch.equal(tensor, images[index])


def test_3d_tensor_is_promoted_to_4d(monkeypatch):
    module = _load_module(monkeypatch)

    image_3d = torch.full((32, 32, 3), 0.4, dtype=torch.float32)

    out = module.TS_ImageListToImages().split(images=[image_3d])

    assert out[0].shape == (1, 32, 32, 3)


def test_single_tensor_input_is_treated_as_one_item_list(monkeypatch):
    """Defensive: if upstream node delivered a single tensor instead of a list."""
    module = _load_module(monkeypatch)

    a = _make_image(value=0.9)

    out = module.TS_ImageListToImages().split(images=a)

    assert len(out) == 3
    assert torch.equal(out[0], a)
    assert _is_blocker(out[1], module.ExecutionBlocker)
    assert _is_blocker(out[2], module.ExecutionBlocker)


def test_invalid_tensor_in_list_becomes_blocker(monkeypatch):
    """Defensive: an entry that is not a usable IMAGE tensor is blocked."""
    module = _load_module(monkeypatch)

    valid = _make_image(value=0.5)
    invalid_shape = torch.zeros((10,), dtype=torch.float32)
    not_a_tensor = "not an image"

    out = module.TS_ImageListToImages().split(images=[valid, invalid_shape, not_a_tensor])

    assert torch.equal(out[0], valid)
    assert _is_blocker(out[1], module.ExecutionBlocker)
    assert _is_blocker(out[2], module.ExecutionBlocker)
