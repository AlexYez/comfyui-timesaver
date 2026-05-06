"""Behaviour tests for TSWhisper helpers (audit F-43).

The module imports torch + transformers + comfy at top level. This test file
stubs the heavy ones away so the helpers can be exercised on a bare CI:
``_safe_float``, ``_safe_int``, ``_is_oom_error``, ``_normalize_text``,
``_merge_segments``, ``_build_generate_kwargs``. Real Whisper inference is not
covered — it requires GPU + downloaded model weights.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

# ts_whisper.py does `import numpy as np` at module top, so we cannot run on a
# numpy-less interpreter. Match the existing test_voice_recognition_audio.py
# convention.
pytest.importorskip("numpy")


def _install_stubs(monkeypatch, ts_tmp_path):
    """Stub heavy third-party deps so ts_whisper imports without torch/transformers."""

    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")

        class _Tensor:
            pass

        class _OOM(Exception):
            pass

        torch_mod.Tensor = _Tensor
        torch_mod.OutOfMemoryError = _OOM
        torch_mod.float32 = "float32"

        torch_cuda_mod = types.ModuleType("torch.cuda")
        torch_cuda_mod.OutOfMemoryError = _OOM
        torch_cuda_mod.is_available = lambda: False
        torch_mod.cuda = torch_cuda_mod
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        monkeypatch.setitem(sys.modules, "torch.cuda", torch_cuda_mod)

    if "torchaudio" not in sys.modules:
        torchaudio_mod = types.ModuleType("torchaudio")
        torchaudio_transforms = types.ModuleType("torchaudio.transforms")

        class _Resample:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def to(self, device): return self

        torchaudio_transforms.Resample = _Resample
        torchaudio_mod.transforms = torchaudio_transforms
        monkeypatch.setitem(sys.modules, "torchaudio", torchaudio_mod)
        monkeypatch.setitem(sys.modules, "torchaudio.transforms", torchaudio_transforms)

    if "transformers" not in sys.modules:
        transformers_mod = types.ModuleType("transformers")
        transformers_mod.AutoModelForSpeechSeq2Seq = type("AutoModelForSpeechSeq2Seq", (), {})
        transformers_mod.AutoProcessor = type("AutoProcessor", (), {})
        transformers_mod.pipeline = lambda *a, **kw: None
        monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    if "comfy" not in sys.modules:
        comfy_mod = types.ModuleType("comfy")
        mm = types.ModuleType("comfy.model_management")
        mm.get_torch_device = lambda: types.SimpleNamespace(type="cpu")
        mm.soft_empty_cache = lambda: None
        comfy_utils = types.ModuleType("comfy.utils")
        comfy_utils.ProgressBar = type("ProgressBar", (), {
            "__init__": lambda self, total: None,
            "update": lambda self, n: None,
            "update_absolute": lambda self, value, total=None: None,
        })
        comfy_mod.model_management = mm
        comfy_mod.utils = comfy_utils
        monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
        monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)

    if "folder_paths" not in sys.modules:
        folder_paths_mod = types.ModuleType("folder_paths")
        folder_paths_mod.base_path = str(ts_tmp_path)
        folder_paths_mod.models_dir = str(ts_tmp_path / "models")
        monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_mod)

    if "srt" not in sys.modules:
        monkeypatch.setitem(sys.modules, "srt", types.ModuleType("srt"))


@pytest.fixture
def whisper_module(monkeypatch, ts_tmp_path):
    root = Path(__file__).resolve().parents[1]
    _install_stubs(monkeypatch, ts_tmp_path)
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop("nodes.audio.ts_whisper", None)
    return importlib.import_module("nodes.audio.ts_whisper")


def _make_node(module, monkeypatch):
    """Build a TSWhisper instance bypassing __init__ (avoids real Comfy state)."""
    node = module.TSWhisper.__new__(module.TSWhisper)
    node.logger = types.SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        debug=lambda *a, **kw: None,
    )
    node.processor = None
    return node


def test_v3_contract_matches_snapshot(whisper_module):
    cls = whisper_module.TSWhisper
    schema = cls.define_schema()
    assert schema.node_id == "TSWhisper"
    assert schema.display_name == "TS Whisper"
    assert schema.category == "TS/Audio"
    assert [out.display_name for out in schema.outputs] == [
        "srt_content", "text_content", "ttml_content",
    ]


def test_safe_float_returns_default_on_garbage(whisper_module, monkeypatch):
    node = _make_node(whisper_module, monkeypatch)
    assert node._safe_float(None, 0.5) == 0.5
    assert node._safe_float("not-a-number", 1.25) == 1.25
    assert node._safe_float("3.14", 0.0) == pytest.approx(3.14)
    assert node._safe_float(7, 0.0) == 7.0


def test_safe_int_returns_default_on_garbage(whisper_module, monkeypatch):
    node = _make_node(whisper_module, monkeypatch)
    assert node._safe_int(None, 4) == 4
    assert node._safe_int("xyz", 5) == 5
    assert node._safe_int("42", 0) == 42
    assert node._safe_int(3.7, 0) == 3


def test_is_oom_error_matches_message_substrings(whisper_module, monkeypatch):
    node = _make_node(whisper_module, monkeypatch)
    assert node._is_oom_error(RuntimeError("CUDA out of memory")) is True
    assert node._is_oom_error(RuntimeError("allocation failed")) is True
    assert node._is_oom_error(RuntimeError("file not found")) is False


def test_normalize_text_lowercases_and_strips_punctuation(whisper_module, monkeypatch):
    node = _make_node(whisper_module, monkeypatch)
    out = node._normalize_text("Привет, мир! Ёлка - это дерево.")
    # ё should map to е, punctuation collapses to whitespace, lowercased.
    assert "елка" in out
    assert "привет" in out
    assert "," not in out
    assert "!" not in out
    assert out == out.strip()


def test_normalize_text_handles_none_and_non_string(whisper_module, monkeypatch):
    node = _make_node(whisper_module, monkeypatch)
    assert node._normalize_text(None) == ""
    assert "123" in node._normalize_text(123)


def test_merge_segments_handles_empty_input(whisper_module, monkeypatch):
    node = _make_node(whisper_module, monkeypatch)
    assert node._merge_segments([], 0.5) == []


def test_merge_segments_collapses_overlapping_duplicates(whisper_module, monkeypatch):
    node = _make_node(whisper_module, monkeypatch)
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 0.9, "end": 1.5, "text": "hello"},
        {"start": 2.0, "end": 3.0, "text": "world"},
    ]
    merged = node._merge_segments(segments, overlap_tolerance_s=0.2)
    assert len(merged) == 2
    assert merged[0]["text"] == "hello"
    assert merged[0]["end"] == pytest.approx(1.5)
    assert merged[1]["text"] == "world"


def test_merge_segments_keeps_non_overlapping_distinct(whisper_module, monkeypatch):
    node = _make_node(whisper_module, monkeypatch)
    segments = [
        {"start": 0.0, "end": 1.0, "text": "alpha"},
        {"start": 5.0, "end": 6.0, "text": "beta"},
    ]
    merged = node._merge_segments(segments, overlap_tolerance_s=0.5)
    assert merged == segments


def test_merge_segments_promotes_longer_text_when_subset(whisper_module, monkeypatch):
    node = _make_node(whisper_module, monkeypatch)
    # Second segment overlaps and contains the first segment's text.
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hi"},
        {"start": 0.8, "end": 1.5, "text": "hi there"},
    ]
    merged = node._merge_segments(segments, overlap_tolerance_s=0.5)
    assert len(merged) == 1
    assert merged[0]["text"] == "hi there"
    assert merged[0]["end"] == pytest.approx(1.5)


def test_build_generate_kwargs_clamps_defaults(whisper_module, monkeypatch):
    node = _make_node(whisper_module, monkeypatch)
    kwargs = node._build_generate_kwargs(
        task="transcribe",
        language="auto",
        num_beams=1,
        temperature=None,
        temperature_fallbacks=None,
        condition_on_prev_tokens=None,
        compression_ratio_threshold=None,
        logprob_threshold=None,
        no_speech_threshold=None,
        initial_prompt=None,
    )
    assert kwargs["task"] == "transcribe"
    assert "language" not in kwargs  # auto -> not passed
    assert "num_beams" not in kwargs  # 1 -> not passed
    assert kwargs["temperature"] == 0.0
    assert kwargs["compression_ratio_threshold"] == pytest.approx(1.35)
    assert kwargs["logprob_threshold"] == pytest.approx(-1.0)
    assert kwargs["no_speech_threshold"] == pytest.approx(0.6)
    assert kwargs["condition_on_prev_tokens"] is False


def test_build_generate_kwargs_passes_explicit_language_and_beams(whisper_module, monkeypatch):
    node = _make_node(whisper_module, monkeypatch)
    kwargs = node._build_generate_kwargs(
        task="translate",
        language="ru",
        num_beams=5,
        temperature=0.2,
        temperature_fallbacks=[0.0, 0.2, 0.4],
        condition_on_prev_tokens=True,
        compression_ratio_threshold=2.0,
        logprob_threshold=-0.5,
        no_speech_threshold=0.7,
        initial_prompt=None,
    )
    assert kwargs["task"] == "translate"
    assert kwargs["language"] == "ru"
    assert kwargs["num_beams"] == 5
    assert kwargs["temperature"] == [0.0, 0.2, 0.4]  # fallback list wins over scalar
    assert kwargs["condition_on_prev_tokens"] is True
    assert kwargs["compression_ratio_threshold"] == pytest.approx(2.0)
