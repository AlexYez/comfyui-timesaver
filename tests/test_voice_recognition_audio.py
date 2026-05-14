from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine_stubs import _DummyEngine, install_engine_module_stub  # noqa: E402


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


_PRESET_FIXTURE = (
    {
        "Prompts enhance": {
            "system_prompt": "Enhance prompt.",
            "gen_params": {"temperature": 0.8},
        }
    },
    ["Prompts enhance"],
)


_ENGINE_SINGLETON = _DummyEngine()


def _install_stubs(monkeypatch, root: Path) -> None:
    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.v0_0_2")
    latest.IO = _IO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.v0_0_2", latest)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(root / ".test_models")
    folder_paths.get_input_directory = lambda: str(root / ".test_input")
    folder_paths.get_annotated_filepath = lambda annotated: ""
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
    qwen_module._load_presets = lambda: _PRESET_FIXTURE
    qwen_module.TS_Qwen3_VL_V3 = _DummyEngine
    monkeypatch.setitem(sys.modules, "nodes.llm.ts_qwen3_vl", qwen_module)

    engine_module = install_engine_module_stub(monkeypatch)
    engine_module.get_qwen_engine = lambda: _ENGINE_SINGLETON


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


def test_audio_preprocess_normalizes_speech(monkeypatch):
    """With AUDIO_TRIM_ENABLED defaulting to False (v9.11), the full clip
    goes to Whisper untouched in length, but normalization still runs."""
    module = _load_module(monkeypatch)

    sample_rate = module.AUDIO_SAMPLE_RATE
    silence = np.zeros(sample_rate // 2, dtype=np.float32)
    t = np.linspace(0.0, 0.5, sample_rate // 2, endpoint=False, dtype=np.float32)
    speech = (0.02 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])

    result = module._preprocess_audio(audio)

    assert result.speech_detected is True
    assert result.trimmed is False  # trim disabled by default
    assert result.normalized is True
    assert result.original_duration == pytest.approx(1.5)
    # Length preserved exactly (no nibbling at the edges).
    assert result.processed_duration == pytest.approx(result.original_duration)
    assert result.peak_after > result.peak_before


def test_audio_preprocess_trim_when_explicitly_enabled(monkeypatch):
    """The trim path still works for users who flip AUDIO_TRIM_ENABLED back
    on — the disabled default is a behaviour choice, not a removal."""
    module = _load_module(monkeypatch)
    voice = _load_voice()

    sample_rate = module.AUDIO_SAMPLE_RATE
    silence = np.zeros(sample_rate // 2, dtype=np.float32)
    t = np.linspace(0.0, 0.5, sample_rate // 2, endpoint=False, dtype=np.float32)
    speech = (0.02 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])

    monkeypatch.setattr(voice, "AUDIO_TRIM_ENABLED", True)
    result = module._preprocess_audio(audio)

    assert result.speech_detected is True
    assert result.trimmed is True
    assert result.processed_duration < result.original_duration


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


def test_transcription_cleanup_collapses_phrase_loops(monkeypatch):
    """Whisper sometimes loops on a phrase even after compression_ratio
    kicks in — the post-processing must fold repeats back to one copy."""
    module = _load_module(monkeypatch)

    # Two-word loop (typical Whisper greedy-decoding hallucination).
    assert module._clean_transcription_text(
        "hello world hello world hello world"
    ) == "hello world"
    # Russian sentence loop with terminating punctuation glued to the word.
    assert module._clean_transcription_text(
        "привет мир. привет мир. привет мир."
    ) == "привет мир."
    # Single-word loop.
    assert module._clean_transcription_text("test test test test") == "test"
    # Mixed: prepositions collapsed first, then phrase collapse runs.
    assert module._clean_transcription_text(
        "кот на столе кот на столе кот на столе"
    ) == "кот на столе"


def test_transcription_cleanup_keeps_non_repeating_text(monkeypatch):
    """Sanity: the phrase-loop pass must never touch normal speech."""
    module = _load_module(monkeypatch)

    sample = "a cinematic wide shot of a city at golden hour"
    assert module._clean_transcription_text(sample) == sample


def test_transcription_cleanup_strips_youtube_outro_hallucination(monkeypatch):
    """Whisper invents 'продолжение следует' from YouTube training data
    when the audio fades to silence. The filter must drop the trailing
    hallucination without touching legitimate speech earlier in the text."""
    module = _load_module(monkeypatch)

    # Plain trailing hallucination after a real prompt.
    assert module._clean_transcription_text(
        "кот на столе. продолжение следует"
    ) == "кот на столе"
    # Punctuation/ellipsis after the hallucination.
    assert module._clean_transcription_text(
        "wide shot at golden hour. Продолжение следует..."
    ) == "wide shot at golden hour"
    # Multiple outros stacked back-to-back (Whisper sometimes loops on them).
    assert module._clean_transcription_text(
        "macro lens portrait. Продолжение следует. Продолжение следует."
    ) == "macro lens portrait"
    # If only the hallucination is present, leave an empty string rather
    # than fabricating output.
    assert module._clean_transcription_text("Продолжение следует.") == ""


def test_adaptive_vad_thresholds_low_never_above_high(monkeypatch):
    """Hysteresis sanity: ``low`` must stay ≤ ``high`` so boundary expansion
    behaves predictably. This is invariant for every clip the VAD ever sees."""
    module = _load_module(monkeypatch)

    # Empty clip: both fall back to the absolute floor.
    high, low = module._adaptive_vad_thresholds(np.asarray([], dtype=np.float32))
    assert low <= high

    # Loud + quiet mix: noise floor drives low, peak caps high. low must clamp.
    rms = np.concatenate([
        np.full(50, 0.0005, dtype=np.float32),  # silence/noise floor
        np.full(20, 0.04, dtype=np.float32),    # main speech
    ])
    high, low = module._adaptive_vad_thresholds(rms)
    assert low <= high
    assert low > 0  # never zero — true silence must still be rejected


def test_detect_speech_bounds_hysteresis_recovers_quiet_consonants(monkeypatch):
    """Without hysteresis the unvoiced "с" preposition gets clipped because its
    RMS is too low for the high threshold. With hysteresis the boundary
    expands outward through frames that pass the low threshold, capturing it."""
    module = _load_module(monkeypatch)
    sample_rate = module.AUDIO_SAMPLE_RATE

    # 0.7s silence | 0.10s quiet pre-roll (the "с") | 0.50s loud speech | 0.20s silence.
    silence_pre = np.zeros(int(sample_rate * 0.7), dtype=np.float32)
    quiet_t = np.linspace(0, 0.10, int(sample_rate * 0.10), endpoint=False, dtype=np.float32)
    quiet = (0.006 * np.sin(2 * np.pi * 200 * quiet_t)).astype(np.float32)
    loud_t = np.linspace(0, 0.50, int(sample_rate * 0.50), endpoint=False, dtype=np.float32)
    loud = (0.05 * np.sin(2 * np.pi * 220 * loud_t)).astype(np.float32)
    silence_post = np.zeros(int(sample_rate * 0.2), dtype=np.float32)
    audio = np.concatenate([silence_pre, quiet, loud, silence_post])

    start, end, detected, threshold = module._detect_speech_bounds(audio, sample_rate)

    assert detected is True
    # Quiet pre-roll begins at 0.7s. Hysteresis must extend the boundary into
    # it — i.e. start (in samples) must reach below the loud onset (0.8s) by
    # MORE than the padding alone (0.40s) would account for.
    loud_onset_sample = int(sample_rate * 0.8)
    padding_samples = int(sample_rate * module.AUDIO_VAD_PADDING_SEC)
    padding_only_start = max(0, loud_onset_sample - padding_samples)
    assert start < padding_only_start, (
        f"Hysteresis should extend left of padding-only boundary: "
        f"start={start}, padding_only_start={padding_only_start}"
    )


def test_detect_speech_bounds_hysteresis_stops_at_silence(monkeypatch):
    """Hysteresis must not run wild — it stops the moment a frame drops
    below the low threshold. Pure silence outside the speech region must
    NOT be swallowed (only the configured padding).  """
    module = _load_module(monkeypatch)
    sample_rate = module.AUDIO_SAMPLE_RATE

    # 1.0s silence | 0.5s loud speech | 1.0s silence.
    silence_pre = np.zeros(sample_rate, dtype=np.float32)
    loud_t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False, dtype=np.float32)
    loud = (0.05 * np.sin(2 * np.pi * 220 * loud_t)).astype(np.float32)
    silence_post = np.zeros(sample_rate, dtype=np.float32)
    audio = np.concatenate([silence_pre, loud, silence_post])

    start, end, detected, _ = module._detect_speech_bounds(audio, sample_rate)
    padding_samples = int(sample_rate * module.AUDIO_VAD_PADDING_SEC)
    # 30ms RMS window can straddle the silence/loud edge by up to a frame,
    # so the boundary shifts inward by at most one frame_size — that's not
    # hysteresis bleed, it's just window overlap. Allow that slack.
    frame_samples = int(sample_rate * module.AUDIO_VAD_FRAME_MS / 1000)

    assert detected is True
    # Hysteresis must NOT walk through silence: start should stay inside
    # [loud_onset - padding - frame, loud_onset], never reach the audio start.
    loud_onset_sample = sample_rate
    loud_offset_sample = sample_rate + len(loud)
    assert start >= loud_onset_sample - padding_samples - frame_samples
    assert start > 0  # silence_pre must not be fully swallowed
    assert end <= loud_offset_sample + padding_samples + frame_samples
    assert end < len(audio)  # silence_post must not be fully swallowed


def test_multilingual_hallucination_drops_mixed_script_garbage(monkeypatch):
    """The classic Whisper failure mode: temperature fallback escalates and
    the model emits text mixing Cyrillic with Greek/CJK/Hangul/etc. When the
    source language is RU we must drop such output instead of inserting
    garbage into the user's prompt textarea."""
    module = _load_module(monkeypatch)

    # Real example reported by the user — Cyrillic + Korean + Spanish +
    # Italian + German + Greek mixed together.
    garbage = (
        "Помно trasx, поменяйте цвет волос на темный все это просто. Также, "
        "также, перек sodium,latmos quindi Dáit свечку시 примms marc minimizing "
        "для дет Подровки светлухаgeryный, увидired волос como вообще-же"
    )
    assert module._looks_like_multilingual_hallucination(garbage, "ru") is True


def test_multilingual_hallucination_keeps_pure_russian(monkeypatch):
    """Pure Russian must always pass through — false positive here would
    silently drop legitimate dictation."""
    module = _load_module(monkeypatch)

    text = "поменяй цвет волос на темный, добавь мягкий контровой свет"
    assert module._looks_like_multilingual_hallucination(text, "ru") is False


def test_multilingual_hallucination_keeps_russian_with_english_terms(monkeypatch):
    """Bilingual prompt dictation (Russian + ASCII English technical terms)
    is the normal use case for TS Super Prompt voice — it must NOT trigger
    the script filter."""
    module = _load_module(monkeypatch)

    text = (
        "кинематографичный wide shot, 85mm anamorphic lens, golden hour, "
        "цветокоррекция в стиле film emulation, soft focus на портрете"
    )
    assert module._looks_like_multilingual_hallucination(text, "ru") is False


def test_multilingual_hallucination_only_runs_for_russian(monkeypatch):
    """The check is RU-specific because for other source languages a
    different script mix is plausible (e.g. Japanese + ASCII)."""
    module = _load_module(monkeypatch)

    mixed = "hello мир 你好 안녕하세요 γειά σου"
    assert module._looks_like_multilingual_hallucination(mixed, "en") is False
    assert module._looks_like_multilingual_hallucination(mixed, None) is False
    assert module._looks_like_multilingual_hallucination(mixed, "auto") is False


def test_multilingual_hallucination_short_text_is_not_judged(monkeypatch):
    """A handful of letters is too small a sample — one stray exotic char
    in a short utterance shouldn't kill the whole transcription."""
    module = _load_module(monkeypatch)

    # Total letters < 10 — below the absolute floor.
    assert module._looks_like_multilingual_hallucination("привет 世", "ru") is False


def test_multilingual_hallucination_can_be_disabled(monkeypatch):
    """Future Whisper releases may eliminate this failure mode, or a power
    user may want to inspect raw output. The flag is the escape hatch."""
    module = _load_module(monkeypatch)
    voice = _load_voice()

    monkeypatch.setattr(voice, "WHISPER_SCRIPT_VALIDATION_ENABLED", False)
    garbage = "поменяй цвет волос на темный 你好 안녕 γειά Dáit primms 시간"
    assert module._looks_like_multilingual_hallucination(garbage, "ru") is False


def test_transcription_cleanup_filter_can_be_disabled(monkeypatch):
    """The hallucination filter is gated by a module-level flag so future
    Whisper releases (or debugging sessions) can opt out without code edits."""
    module = _load_module(monkeypatch)
    voice = _load_voice()

    monkeypatch.setattr(voice, "WHISPER_HALLUCINATION_FILTER_ENABLED", False)
    raw = "кот на столе. продолжение следует"
    assert module._clean_transcription_text(raw) == raw


def test_voice_recognition_backend_registers_only_super_prompt_node(monkeypatch):
    module = _load_module(monkeypatch)
    real = importlib.import_module("nodes.llm.super_prompt.ts_super_prompt")

    assert not hasattr(module, "TS_" + "VoiceRecognition")
    # The shim deliberately exports an empty NODE_CLASS_MAPPINGS so that the
    # snapshot tool records the real V3 file at nodes/llm/super_prompt/.
    assert module.NODE_CLASS_MAPPINGS == {}
    assert real.NODE_CLASS_MAPPINGS == {"TS_SuperPrompt": real.TS_SuperPrompt}
    assert module.TS_SuperPrompt is real.TS_SuperPrompt
    assert module.TRANSLATE_TO_ENGLISH is False
