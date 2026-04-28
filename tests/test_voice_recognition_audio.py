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

    qwen_module = types.ModuleType("nodes.ts_qwen3_vl_v3_node")
    qwen_module.TS_Qwen3_VL_V3 = _DummyQwen
    monkeypatch.setitem(sys.modules, "nodes.ts_qwen3_vl_v3_node", qwen_module)


def _load_module(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    _install_stubs(monkeypatch, root)
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop("nodes.ts_super_prompt_node", None)
    return importlib.import_module("nodes.ts_super_prompt_node")


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

    prompt = module._configured_initial_prompt()

    assert prompt is not None
    assert "ComfyUI" in prompt
    assert "cinematic" in prompt
    assert "85mm" in prompt
    assert "русский" in prompt

    monkeypatch.setattr(module, "INITIAL_PROMPT_EXTRA", "custom phrase: anamorphic portrait lighting")
    assert "anamorphic portrait lighting" in module._configured_initial_prompt()

    monkeypatch.setattr(module, "INITIAL_PROMPT_ENABLED", False)
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
    events = []

    monkeypatch.setattr(module, "_send_voice_event", lambda event, payload: events.append((event, payload)))

    module.ProgressBroadcaster("base").done()

    assert events == [("status", {"model": "base", "text": "Voice model file ready", "percent": 100.0})]


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
