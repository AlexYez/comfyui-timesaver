from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NodeOutput:
    def __init__(self, *values, **kwargs):
        self.values = values
        self.kwargs = kwargs


class _Input:
    def __init__(self, id, *args, **kwargs):
        self.id = id
        self.args = args
        self.kwargs = kwargs
        self.default = kwargs.get("default")


class _Output:
    def __init__(self, id=None, display_name=None, **kwargs):
        self.id = id
        self.display_name = display_name
        self.kwargs = kwargs


class _ComfyType:
    Input = _Input
    Output = _Output


class _IO:
    ComfyNode = object
    Schema = _Schema
    NodeOutput = _NodeOutput
    String = _ComfyType
    Boolean = _ComfyType
    Combo = _ComfyType
    Image = _ComfyType


class _DummyQwen:
    @classmethod
    def _load_presets(cls):
        return {
            "Prompts enhance": {
                "system_prompt": "Enhance prompt.",
                "gen_params": {"temperature": 0.8},
            }
        }, ["Prompts enhance"]


def _install_stubs(monkeypatch, root: Path) -> None:
    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.IO = _IO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(root / ".test_models")
    folder_paths.get_input_directory = lambda: str(root / ".test_input")
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    aiohttp = types.ModuleType("aiohttp")
    web = types.SimpleNamespace(
        Request=object,
        StreamResponse=object,
        json_response=lambda data, status=200: {"data": data, "status": status},
    )
    aiohttp.web = web
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp)
    monkeypatch.setitem(sys.modules, "aiohttp.web", web)

    qwen_module = types.ModuleType("nodes.llm.ts_qwen3_vl")
    qwen_module.TS_Qwen3_VL_V3 = _DummyQwen
    monkeypatch.setitem(sys.modules, "nodes.llm.ts_qwen3_vl", qwen_module)


def _load_module(monkeypatch):
    """Return the public shim that aggregates the super_prompt subpackage.

    The shim re-exports symbols from `nodes.llm.super_prompt._helpers`,
    `_voice`, and `_qwen`. To monkeypatch a symbol such that *behaviour*
    inside the subpackage actually changes, patch the originating module via
    `_load_helpers`/`_load_voice`/`_load_qwen` — `monkeypatch.setattr(shim, ...)`
    only mutates the shim's own dict and does not propagate to the modules
    that already imported the symbol by value.
    """
    root = Path(__file__).resolve().parents[1]
    _install_stubs(monkeypatch, root)
    monkeypatch.syspath_prepend(str(root))
    for cached in (
        "nodes.llm.ts_super_prompt",
        "nodes.llm.super_prompt",
        "nodes.llm.super_prompt._helpers",
        "nodes.llm.super_prompt._voice",
        "nodes.llm.super_prompt._qwen",
        "nodes.llm.super_prompt.ts_super_prompt",
    ):
        sys.modules.pop(cached, None)
    return importlib.import_module("nodes.llm.ts_super_prompt")


def _load_helpers():
    return importlib.import_module("nodes.llm.super_prompt._helpers")


def _load_voice():
    return importlib.import_module("nodes.llm.super_prompt._voice")


def test_audio_preprocess_trims_and_normalizes_speech(monkeypatch):
    module = _load_module(monkeypatch)

    sample_rate = module.AUDIO_SAMPLE_RATE
    silence = np.zeros(sample_rate // 2, dtype=np.float32)
    t = np.linspace(0.0, 0.5, sample_rate // 2, endpoint=False, dtype=np.float32)
    speech = (0.02 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])

    result = module._preprocess_audio(audio)

    assert result.speech_detected is True
    assert result.trimmed is True
    assert result.normalized is True
    assert result.original_duration == pytest.approx(1.5)
    assert result.processed_duration < result.original_duration
    assert result.peak_after > result.peak_before


def test_audio_preprocess_fades_trimmed_edges(monkeypatch):
    module = _load_module(monkeypatch)

    audio = np.ones(module.AUDIO_SAMPLE_RATE, dtype=np.float32) * 0.02
    result = module._preprocess_audio(audio)

    assert result.speech_detected is True
    assert abs(float(result.audio[0])) < 1e-6
    assert abs(float(result.audio[-1])) < 1e-6


def test_audio_preprocess_rejects_silence(monkeypatch):
    module = _load_module(monkeypatch)

    audio = np.zeros(module.AUDIO_SAMPLE_RATE, dtype=np.float32)
    result = module._preprocess_audio(audio)

    assert result.speech_detected is False
    assert result.processed_duration == 0.0
    assert result.audio.size == 0


def test_initial_prompt_is_prompt_dictation_context(monkeypatch):
    module = _load_module(monkeypatch)
    voice = _load_voice()

    prompt = module._configured_initial_prompt()

    assert prompt is not None
    assert "ComfyUI" in prompt
    assert "cinematic" in prompt
    assert "85mm" in prompt
    assert "русский" in prompt

    # _configured_initial_prompt reads INITIAL_PROMPT_* from the _voice module
    # namespace (where they were imported by value from _helpers), so patches
    # must target _voice, not the public shim.
    monkeypatch.setattr(voice, "INITIAL_PROMPT_EXTRA", "custom phrase: anamorphic portrait lighting")
    assert "anamorphic portrait lighting" in module._configured_initial_prompt()

    monkeypatch.setattr(voice, "INITIAL_PROMPT_ENABLED", False)
    assert module._configured_initial_prompt() is None


def test_high_quality_selects_turbo_voice_model(monkeypatch):
    module = _load_module(monkeypatch)

    assert module._resolve_voice_model(False, "turbo") == "base"
    assert module._resolve_voice_model("false", "turbo") == "base"
    assert module._resolve_voice_model(True, "base") == "turbo"
    assert module._resolve_voice_model("true") == "turbo"


def test_turbo_uses_actual_whisper_download_filename(monkeypatch):
    module = _load_module(monkeypatch)

    assert module._model_file_path("base").name == "base.pt"
    assert module._model_file_path("turbo").name == "large-v3-turbo.pt"


def test_download_done_keeps_ui_busy_until_memory_load(monkeypatch):
    module = _load_module(monkeypatch)
    voice = _load_voice()
    events = []

    # ProgressBroadcaster lives in _voice.py and looks up `send_voice_event`
    # from its own module namespace (imported by value from _helpers).
    monkeypatch.setattr(voice, "send_voice_event", lambda event, payload: events.append((event, payload)))

    module.ProgressBroadcaster("base").done()

    assert events == [("status", {"model": "base", "text": "Voice model file ready", "percent": 100.0})]


def test_memory_load_status_is_short_for_button_label(monkeypatch):
    module = _load_module(monkeypatch)
    voice = _load_voice()
    events = []

    monkeypatch.setattr(voice, "ensure_model", lambda name: module.WHISPER_DIR / f"{name}.pt")
    monkeypatch.setattr(voice, "send_voice_status", lambda model, text, percent=None: events.append(("status", {"model": model, "text": text, **({"percent": percent} if percent is not None else {})})))
    module._VOICE_MODEL_CACHE.clear()

    fake_torch = types.SimpleNamespace()
    fake_whisper = types.SimpleNamespace(load_model=lambda *args, **kwargs: object())
    monkeypatch.setattr(voice, "_load_whisper_runtime", lambda: (fake_torch, fake_whisper))

    module.load_model("turbo", "cpu", progress_start=42.0, progress_end=64.0)

    assert ("status", {"model": "turbo", "text": "Loading model...", "percent": 42.0}) in events
    assert not any("turbo into" in str(payload.get("text", "")) for _, payload in events)


def test_transcription_cleanup_removes_duplicate_prepositions(monkeypatch):
    module = _load_module(monkeypatch)

    assert module._clean_transcription_text("кот с с камерой и в в кадре") == "кот с камерой и в кадре"
    assert module._clean_transcription_text("with with cinematic light, from from above") == (
        "with cinematic light, from above"
    )


def test_voice_recognition_backend_registers_only_super_prompt_node(monkeypatch):
    module = _load_module(monkeypatch)

    assert not hasattr(module, "TS_" + "VoiceRecognition")
    assert module.NODE_CLASS_MAPPINGS == {"TS_SuperPrompt": module.TS_SuperPrompt}
    assert module.TRANSLATE_TO_ENGLISH is False
