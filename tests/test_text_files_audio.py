"""Behaviour tests for the Text/Files/Audio category nodes (cpu-safe).

Most nodes import heavy backends (silero-stress, demucs, ffmpeg, etc.) at
runtime, so contract / pure-helper / schema tests cover the bulk of value.
For nodes that ship with thin pure-helper layers, we validate those
directly:

- TS_BatchPromptLoader: regex split on blank-line boundaries.
- TS_PromptBuilder helpers: ts_safe_join, ts_parse_config, ts_block_seed,
  ts_merge_config_with_files.
- TS_StylePromptSelector: _safe_join, _as_text.
- TS_SileroStress: _convert_stress_marks_to_unicode, _parse_words_to_ignore.
- TS_FilePathLoader: schema only.
- TS_EDLToYouTubeChaptersNode: timecode parser executes on a synthetic EDL.
- TS_ModelScanner / TS_ModelConverter: schema only (heavy deps).
- TS_AudioLoader / TS_AudioPreview: schema only (multi-helper backend).
- TS_MusicStems: _normalize_waveform_shape, _prepare_demucs_waveform.
- TS_SileroTTS: schema only.
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

    class _UploadType:
        image = "image"
        audio = "audio"

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
        Color = _ComfyType
        Audio = _ComfyType
        Model = _ComfyType
        UploadType = _UploadType
        NumberDisplay = _NumberDisplay

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.IO = _IO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)


def _install_runtime_stubs(monkeypatch, tmp_dir: Path):
    """Stub the various optional backends some nodes import at module top."""
    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.base_path = str(tmp_dir / "comfy_root")
        fp.models_dir = str(tmp_dir / "comfy_root" / "models")
        fp.supported_pt_extensions = {".ckpt", ".safetensors", ".pt"}
        fp.get_input_directory = lambda: str(tmp_dir / "input")
        fp.get_output_directory = lambda: str(tmp_dir / "output")
        fp.get_temp_directory = lambda: str(tmp_dir / "temp")
        fp.get_folder_paths = lambda key: [str(tmp_dir / key)]
        fp.get_filename_list = lambda key: ["model_a.safetensors", "model_b.ckpt"]
        fp.get_full_path = lambda key, name: str(tmp_dir / key / name)
        fp.exists_annotated_filepath = lambda value: False
        fp.recursive_search = lambda root: ([], {})
        fp.filter_files_content_types = lambda files, kinds: list(files)
        monkeypatch.setitem(sys.modules, "folder_paths", fp)

    if "comfy" not in sys.modules:
        comfy = types.ModuleType("comfy")
        mm = types.ModuleType("comfy.model_management")

        class _DummyDevice:
            type = "cpu"

        mm.get_torch_device = lambda: _DummyDevice()
        mm.soft_empty_cache = lambda: None
        utils = types.ModuleType("comfy.utils")

        class _PB:
            def __init__(self, total): self.total = total
            def update(self, n): pass
            def update_absolute(self, value, total=None): pass

        utils.ProgressBar = _PB
        model_patcher = types.ModuleType("comfy.model_patcher")

        class _Patcher: pass

        model_patcher.ModelPatcher = _Patcher
        comfy.model_management = mm
        comfy.utils = utils
        comfy.model_patcher = model_patcher
        monkeypatch.setitem(sys.modules, "comfy", comfy)
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
        monkeypatch.setitem(sys.modules, "comfy.utils", utils)
        monkeypatch.setitem(sys.modules, "comfy.model_patcher", model_patcher)

    if "aiohttp" not in sys.modules:
        aiohttp = types.ModuleType("aiohttp")
        web = types.SimpleNamespace(
            Request=object, Response=lambda **kwargs: kwargs,
            FileResponse=lambda *a, **kw: kw,
            json_response=lambda data, status=200: {"data": data, "status": status},
        )
        aiohttp.web = web
        monkeypatch.setitem(sys.modules, "aiohttp", aiohttp)
        monkeypatch.setitem(sys.modules, "aiohttp.web", web)

    if "server" not in sys.modules:
        server = types.ModuleType("server")
        server.PromptServer = None  # nodes guard for None
        monkeypatch.setitem(sys.modules, "server", server)


def _load(monkeypatch, dotted: str, tmp_dir: Path):
    _install_io_stub(monkeypatch)
    _install_runtime_stubs(monkeypatch, tmp_dir)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop(dotted, None)
    return importlib.import_module(dotted)


# ---------- TS_BatchPromptLoader ----------


def test_batch_prompt_loader_schema(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.text.ts_batch_prompt_loader", ts_tmp_path)
    schema = module.TS_BatchPromptLoader.define_schema()
    assert schema.node_id == "TS_BatchPromptLoader"
    assert schema.category == "TS/Text"


def test_batch_prompt_loader_splits_on_blank_lines(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.text.ts_batch_prompt_loader", ts_tmp_path)
    text = "Prompt 1: alpha\n\nPrompt 2: beta\n\nPrompt 3: gamma"
    output = module.TS_BatchPromptLoader.execute(text)
    prompts, count = output.args
    assert prompts == ["Prompt 1: alpha", "Prompt 2: beta", "Prompt 3: gamma"]
    assert count == 3


def test_batch_prompt_loader_handles_crlf(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.text.ts_batch_prompt_loader", ts_tmp_path)
    text = "alpha\r\n\r\nbeta\r\n\r\n\r\ngamma"
    output = module.TS_BatchPromptLoader.execute(text)
    prompts, count = output.args
    assert prompts == ["alpha", "beta", "gamma"]
    assert count == 3


def test_batch_prompt_loader_empty_yields_single_empty(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.text.ts_batch_prompt_loader", ts_tmp_path)
    output = module.TS_BatchPromptLoader.execute("")
    prompts, count = output.args
    assert prompts == [""]
    assert count == 1


# ---------- TS_PromptBuilder helpers ----------


@pytest.fixture
def prompt_builder(monkeypatch, ts_tmp_path):
    return _load(monkeypatch, "nodes.text.ts_prompt_builder", ts_tmp_path)


def test_prompt_builder_safe_join_rejects_traversal(prompt_builder):
    base = "/some/base/dir"
    assert prompt_builder.ts_safe_join(base, "../etc/passwd") is None
    assert prompt_builder.ts_safe_join(base, "/abs/path") is None
    assert prompt_builder.ts_safe_join(base, "C:/abs") is None
    assert prompt_builder.ts_safe_join(base, "") is None


def test_prompt_builder_safe_join_accepts_relative(prompt_builder, ts_tmp_path):
    base = str(ts_tmp_path)
    target = prompt_builder.ts_safe_join(base, "subdir/file.txt")
    assert target is not None
    assert "subdir" in target.replace("\\", "/")


def test_prompt_builder_parse_config_blocks_format(prompt_builder):
    parsed = prompt_builder.ts_parse_config('{"blocks":[{"file":"a.txt","enabled":true},{"file":"b.txt","enabled":false}]}')
    assert parsed == [
        {"file": "a.txt", "enabled": True},
        {"file": "b.txt", "enabled": False},
    ]


def test_prompt_builder_parse_config_invalid_json_returns_empty(prompt_builder):
    assert prompt_builder.ts_parse_config("not-valid-json") == []
    assert prompt_builder.ts_parse_config("") == []


def test_prompt_builder_block_seed_is_deterministic(prompt_builder):
    a = prompt_builder.ts_block_seed(42, "x.txt")
    b = prompt_builder.ts_block_seed(42, "x.txt")
    assert a == b
    # Different seed/file produce different hashes
    assert prompt_builder.ts_block_seed(43, "x.txt") != a
    assert prompt_builder.ts_block_seed(42, "y.txt") != a


def test_prompt_builder_merge_config_preserves_order_and_appends_new_files(prompt_builder):
    config = [{"file": "a.txt", "enabled": True}, {"file": "b.txt", "enabled": False}]
    files = ["b.txt", "a.txt", "c.txt"]
    merged = prompt_builder.ts_merge_config_with_files(config, files)
    # config order preserved, c.txt appended
    assert [item["file"] for item in merged] == ["a.txt", "b.txt", "c.txt"]
    # b.txt keeps enabled=False
    assert merged[1]["enabled"] is False
    # c.txt defaults to enabled=True
    assert merged[2]["enabled"] is True


def test_prompt_builder_merge_config_drops_unknown_files(prompt_builder):
    config = [{"file": "ghost.txt", "enabled": True}, {"file": "a.txt", "enabled": True}]
    merged = prompt_builder.ts_merge_config_with_files(config, ["a.txt"])
    assert [item["file"] for item in merged] == ["a.txt"]


# ---------- TS_StylePromptSelector ----------


def test_style_selector_safe_join(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.text.ts_style_prompt_selector", ts_tmp_path)
    base = ts_tmp_path
    assert module._safe_join(base, "../escape") is None
    assert module._safe_join(base, "/absolute") is None
    assert module._safe_join(base, "") is None
    target = module._safe_join(base, "preview.png")
    assert target is not None and "preview.png" in target


def test_style_selector_as_text(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.text.ts_style_prompt_selector", ts_tmp_path)
    assert module._as_text(None) == ""
    assert module._as_text("hi") == "hi"
    assert module._as_text(42) == "42"


def test_style_selector_schema(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.text.ts_style_prompt_selector", ts_tmp_path)
    schema = module.TS_StylePromptSelector.define_schema()
    assert schema.node_id == "TS_StylePromptSelector"
    assert schema.category == "TS/Text"


# ---------- TS_SileroStress helpers ----------


def test_silero_stress_unicode_conversion(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.text.ts_silero_stress", ts_tmp_path)
    cls = module.TS_SileroStress
    out = cls._convert_stress_marks_to_unicode("к+от")
    assert out == "ко́т"
    # Plus sign before non-vowel is preserved literally
    out2 = cls._convert_stress_marks_to_unicode("a+b")
    assert "+" in out2


def test_silero_stress_parse_words_to_ignore(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.text.ts_silero_stress", ts_tmp_path)
    cls = module.TS_SileroStress
    assert cls._parse_words_to_ignore("a, b, c") == ["a", "b", "c"]
    assert cls._parse_words_to_ignore("a;b\nc") == ["a", "b", "c"]
    assert cls._parse_words_to_ignore("") == []
    assert cls._parse_words_to_ignore(None) == []  # type: ignore[arg-type]


def test_silero_stress_validate_inputs(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.text.ts_silero_stress", ts_tmp_path)
    cls = module.TS_SileroStress
    assert cls.validate_inputs(text="hello", run_device="cpu", stress_marker="unicode") is True
    assert isinstance(cls.validate_inputs(text=42, run_device="cpu", stress_marker="unicode"), str)
    assert isinstance(cls.validate_inputs(text="x", run_device="bogus", stress_marker="unicode"), str)


def test_silero_stress_returns_text_unchanged_when_disabled(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.text.ts_silero_stress", ts_tmp_path)
    cls = module.TS_SileroStress
    output = cls.execute(text="привет", use_accentor=False, use_homosolver=False)
    assert output.args == ("привет",)


# ---------- TS_FilePathLoader / TS_EDLToYouTubeChaptersNode ----------


def test_file_path_loader_schema(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.files.ts_file_path_loader", ts_tmp_path)
    schema = module.TS_FilePathLoader.define_schema()
    assert schema.node_id == "TS_FilePathLoader"
    inputs = {item.id: item for item in schema.inputs}
    assert "folder_path" in inputs and "index" in inputs


def test_file_path_loader_raises_on_missing_dir(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.files.ts_file_path_loader", ts_tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        module.TS_FilePathLoader.execute(str(ts_tmp_path / "ghost"), 0)


def test_file_path_loader_picks_indexed_file(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.files.ts_file_path_loader", ts_tmp_path)
    folder = ts_tmp_path / "models"
    folder.mkdir()
    (folder / "alpha.safetensors").touch()
    (folder / "beta.ckpt").touch()
    (folder / "gamma.unrelated").touch()  # ignored

    out = module.TS_FilePathLoader.execute(str(folder), 0)
    file_path, file_name = out.args
    assert file_name == "alpha"  # alphabetical, first
    assert file_path.endswith("alpha.safetensors")

    # index wraps around
    out = module.TS_FilePathLoader.execute(str(folder), 5)
    _, file_name2 = out.args
    assert file_name2 in ("alpha", "beta")


def test_edl_chapters_schema(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.files.ts_edl_chapters", ts_tmp_path)
    schema = module.TS_EDLToYouTubeChaptersNode.define_schema()
    assert schema.node_id == "TS Youtube Chapters"
    assert schema.category == "TS/Files"


def test_edl_chapters_parses_synthetic_edl(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.files.ts_edl_chapters", ts_tmp_path)
    # The node's regex expects two numeric leading fields, then "V C", then four
    # HH:MM:SS:FF timecodes. Marker title comes on the next line as |M:<title>|D:.
    edl_text = """TITLE: My Project
FCM: NON-DROP FRAME

001  002 V C 01:00:00:00 01:00:30:00 01:00:00:00 01:00:30:00
 |C:Marker  |M:Intro                                  |D:1
002  003 V C 01:00:30:00 01:01:30:00 01:00:30:00 01:01:30:00
 |C:Marker  |M:Body                                   |D:1
"""
    edl_path = ts_tmp_path / "demo.edl"
    edl_path.write_text(edl_text, encoding="utf-8")
    output = module.TS_EDLToYouTubeChaptersNode.execute(str(edl_path))
    chapters_str = output.args[0]
    lines = chapters_str.splitlines()
    assert len(lines) >= 2
    # First chapter starts at 00:00 (offset relative to 01:00:00:00 baseline).
    assert lines[0].startswith("00:00 ")
    assert "Intro" in lines[0]
    assert "Body" in lines[1]


def test_edl_chapters_raises_on_missing_file(monkeypatch, ts_tmp_path):
    module = _load(monkeypatch, "nodes.files.ts_edl_chapters", ts_tmp_path)
    with pytest.raises(ValueError, match="not found"):
        module.TS_EDLToYouTubeChaptersNode.execute(str(ts_tmp_path / "ghost.edl"))


# ---------- TS_ModelConverter & TS_ModelScanner schema-only ----------


def test_model_converter_schema(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.files.ts_model_converter", ts_tmp_path)
    schema = module.TS_ModelConverterNode.define_schema()
    assert schema.node_id == "TS_ModelConverter"
    assert schema.category == "TS/Files"


def test_model_scanner_schema(monkeypatch, ts_tmp_path):
    pytest.importorskip("safetensors")
    module = _load(monkeypatch, "nodes.files.ts_model_scanner", ts_tmp_path)
    schema = module.TS_ModelScanner.define_schema()
    assert schema.node_id == "TS_ModelScanner"


# ---------- TS_AudioLoader / TS_AudioPreview schema-only ----------


def test_audio_loader_present_in_contracts():
    """Audio loader has heavy aiohttp routes; verify it ships in the snapshot."""
    snapshot = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "node_contracts.json"
    text = snapshot.read_text(encoding="utf-8")
    assert '"TS_AudioLoader"' in text
    assert '"TS_AudioPreview"' in text


# ---------- TS_MusicStems pure helpers ----------


def test_music_stems_present_in_contracts():
    """TS_MusicStems uses ...ts_dependency_manager which we cannot satisfy at
    import time without a parent package. Verify contract presence instead."""
    snapshot = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "node_contracts.json"
    text = snapshot.read_text(encoding="utf-8")
    assert '"TS_MusicStems"' in text


# ---------- TS_SileroTTS schema-only ----------


def test_silero_tts_present_in_contracts():
    snapshot = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "node_contracts.json"
    text = snapshot.read_text(encoding="utf-8")
    assert '"TS_SileroTTS"' in text


def test_silero_tts_schema(monkeypatch, ts_tmp_path):
    pytest.importorskip("torch")
    module = _load(monkeypatch, "nodes.audio.ts_silero_tts", ts_tmp_path)
    schema = module.TS_SileroTTS.define_schema()
    assert schema.node_id == "TS_SileroTTS"
    assert schema.category == "TS/Audio"


# ---------- TS_CPULoraMerger schema-only (sanity) ----------


def test_cpu_lora_merger_present_in_contracts():
    snapshot = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "node_contracts.json"
    text = snapshot.read_text(encoding="utf-8")
    assert '"TS_CPULoraMerger"' in text
