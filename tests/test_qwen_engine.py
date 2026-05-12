"""Engine-level tests for nodes/llm/_qwen_engine.py.

These exercise the QwenEngine API that both TS_Qwen3_VL_V3 and TS_SuperPrompt
sit on top of:

* mirror-fallback semantics in ``_download_with_mirrors`` (TypeError → break,
  any other Exception → try the next endpoint);
* ``strip_thinking_block`` postprocessor used by both nodes;
* ``apply_runtime_optimizations`` idempotency.

The real engine imports torch / comfy.model_management / folder_paths so we
only run under the ComfyUI Python.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("comfy.model_management")


def _load_engine():
    """Import the real engine module; ensures any test-suite stubs that may
    have been installed by other tests in the same session are evicted
    first."""
    import importlib
    import sys

    sys.modules.pop("nodes.llm._qwen_engine", None)
    return importlib.import_module("nodes.llm._qwen_engine")


# ---------------------------------------------------------------------------
# strip_thinking_block
# ---------------------------------------------------------------------------


def test_strip_thinking_block_removes_segment():
    engine_mod = _load_engine()
    QwenEngine = engine_mod.QwenEngine

    assert QwenEngine.strip_thinking_block("<think>secret</think>\nresult") == "result"
    assert QwenEngine.strip_thinking_block("Final prompt:\nA serene lake.") == (
        "Final prompt:\nA serene lake."
    )
    assert QwenEngine.strip_thinking_block("") == ""


def test_strip_thinking_block_handles_multiple_segments():
    engine_mod = _load_engine()
    QwenEngine = engine_mod.QwenEngine

    cleaned = QwenEngine.strip_thinking_block("a <think>x</think> b <think>y</think> c")
    assert cleaned == "a  b  c"


def test_strip_thinking_block_is_case_insensitive_and_multiline():
    engine_mod = _load_engine()
    QwenEngine = engine_mod.QwenEngine

    cleaned = QwenEngine.strip_thinking_block(
        "before\n<THINK>\nmulti\nline\n</THINK>\nafter"
    )
    assert cleaned == "before\n\nafter"


# ---------------------------------------------------------------------------
# Mirror fallback
# ---------------------------------------------------------------------------


def test_download_with_mirrors_breaks_on_type_error(monkeypatch):
    """TypeError = huggingface_hub kwarg incompatibility — switching mirrors
    will fail identically, so the loop should bail immediately."""
    engine_mod = _load_engine()
    engine = engine_mod.QwenEngine()
    calls: list[str] = []

    def fake_snapshot(model_id, local_dir, token, endpoint_url):
        calls.append(endpoint_url)
        raise TypeError("hub version mismatch")

    monkeypatch.setattr(engine, "_snapshot_download", fake_snapshot)

    with pytest.raises(RuntimeError, match="All mirrors failed"):
        engine._download_with_mirrors(
            "test/model",
            "/tmp/test",
            "",
            "first-mirror.example, second-mirror.example, third-mirror.example",
        )

    assert len(calls) == 1
    assert "first-mirror.example" in calls[0]


def test_download_with_mirrors_continues_on_network_error(monkeypatch):
    """Transient network failures should rotate through every mirror until
    one succeeds."""
    engine_mod = _load_engine()
    engine = engine_mod.QwenEngine()
    calls: list[str] = []

    def fake_snapshot(model_id, local_dir, token, endpoint_url):
        calls.append(endpoint_url)
        if "third-mirror" not in endpoint_url:
            raise ConnectionError("network broken")
        return None  # third mirror succeeds.

    monkeypatch.setattr(engine, "_snapshot_download", fake_snapshot)

    engine._download_with_mirrors(
        "test/model",
        "/tmp/test",
        "",
        "first-mirror.example, second-mirror.example, third-mirror.example",
    )

    assert len(calls) == 3
    assert "first-mirror.example" in calls[0]
    assert "second-mirror.example" in calls[1]
    assert "third-mirror.example" in calls[2]


def test_download_with_mirrors_raises_after_all_fail(monkeypatch):
    """If every mirror dies with a network-level error, the engine should
    surface the last error wrapped in a RuntimeError."""
    engine_mod = _load_engine()
    engine = engine_mod.QwenEngine()
    calls: list[str] = []

    def fake_snapshot(model_id, local_dir, token, endpoint_url):
        calls.append(endpoint_url)
        raise ConnectionError("network broken everywhere")

    monkeypatch.setattr(engine, "_snapshot_download", fake_snapshot)

    with pytest.raises(RuntimeError, match="All mirrors failed"):
        engine._download_with_mirrors(
            "test/model",
            "/tmp/test",
            "",
            "first.example, second.example",
        )

    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Runtime optimizations idempotency
# ---------------------------------------------------------------------------


def test_apply_runtime_optimizations_is_idempotent():
    engine_mod = _load_engine()
    engine = engine_mod.QwenEngine()

    assert engine._optimizations_applied is False
    engine.apply_runtime_optimizations()
    assert engine._optimizations_applied is True
    # Second call must be a no-op (the function early-returns).
    engine.apply_runtime_optimizations()
    assert engine._optimizations_applied is True
