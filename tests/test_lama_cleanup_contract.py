"""Contract tests for TS_LamaCleanup.

Stubs Comfy/aiohttp/server modules so the node can be imported and inspected
without a running ComfyUI instance. Validates schema, helper paths, and
fingerprint behavior. Skipped when numpy/torch/PIL are not installed in the
test environment, since the helper module depends on them at import time.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("PIL.Image")


class _Input:
    def __init__(self, id, *args, **kwargs):
        self.id = id
        self.args = args
        self.kwargs = kwargs
        self.optional = bool(kwargs.get("optional", False))
        self.default = kwargs.get("default")
        self.options = kwargs.get("options", args[0] if args else None)
        self.advanced = bool(kwargs.get("advanced", False))


class _Output:
    def __init__(self, id=None, display_name=None, **kwargs):
        self.id = id
        self.display_name = display_name
        self.kwargs = kwargs


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _NodeOutput:
    def __init__(self, *values, **kwargs):
        self.values = values
        self.kwargs = kwargs


class _ComfyType:
    Input = _Input
    Output = _Output


class _UploadType:
    image = "image"
    audio = "audio"


class _NumberDisplay:
    slider = "slider"
    number = "number"


class _IO:
    ComfyNode = object
    Schema = _Schema
    NodeOutput = _NodeOutput
    String = _ComfyType
    Boolean = _ComfyType
    Combo = _ComfyType
    Int = _ComfyType
    Float = _ComfyType
    Image = _ComfyType
    Audio = _ComfyType
    Color = _ComfyType
    UploadType = _UploadType
    NumberDisplay = _NumberDisplay


def _install_stubs(monkeypatch, tmp_path_obj: Path) -> None:
    input_dir = tmp_path_obj / "input"
    output_dir = tmp_path_obj / "output"
    temp_dir = tmp_path_obj / "temp"
    models_dir = tmp_path_obj / "models"
    for path in (input_dir, output_dir, temp_dir, models_dir):
        path.mkdir(parents=True, exist_ok=True)

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.IO = _IO
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(models_dir)
    folder_paths.get_input_directory = lambda: str(input_dir)
    folder_paths.get_output_directory = lambda: str(output_dir)
    folder_paths.get_temp_directory = lambda: str(temp_dir)
    folder_paths.get_annotated_filepath = lambda value: str(input_dir / value.split(" [", 1)[0]) if value else ""
    folder_paths.recursive_search = lambda root: ([], {})
    folder_paths.filter_files_content_types = lambda files, kinds: list(files)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    aiohttp_web = types.SimpleNamespace(
        Request=object,
        StreamResponse=object,
        Response=lambda **kwargs: kwargs,
        FileResponse=lambda *args, **kwargs: {"file": args, "kwargs": kwargs},
        json_response=lambda data, status=200: {"data": data, "status": status},
    )
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = aiohttp_web
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp)
    monkeypatch.setitem(sys.modules, "aiohttp.web", aiohttp_web)


def _load_module(monkeypatch, tmp_path_obj: Path):
    _install_stubs(monkeypatch, tmp_path_obj)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop("nodes.image.lama_cleanup.ts_lama_cleanup", None)
    sys.modules.pop("nodes.image.lama_cleanup._lama_helpers", None)
    sys.modules.pop("nodes.image.lama_cleanup", None)
    return importlib.import_module("nodes.image.lama_cleanup.ts_lama_cleanup")


def test_lama_cleanup_schema_contract(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    schema = module.TS_LamaCleanup.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert schema.node_id == "TS_LamaCleanup"
    assert schema.display_name == "TS Lama Cleanup"
    assert schema.category == "TS/Image"
    assert list(inputs) == [
        "source_path",
        "brush_size",
        "max_resolution",
        "mask_padding",
        "feather",
        "session_id",
        "working_path",
    ]
    # source_path is a plain String now — the upload UI is owned by the
    # frontend extension, not by ComfyUI's built-in IMAGEUPLOAD widget.
    assert inputs["source_path"].default == ""
    assert "upload" not in inputs["source_path"].kwargs
    assert inputs["brush_size"].default == 40
    assert inputs["brush_size"].kwargs.get("min") == 1
    assert inputs["brush_size"].kwargs.get("max") == 400
    assert inputs["max_resolution"].default == 512
    # No advanced=True — that triggers the "Show advanced inputs" toggle in V2,
    # which is redundant when JS hides every standard widget.
    assert inputs["max_resolution"].kwargs.get("advanced") is None
    assert inputs["mask_padding"].default == 64
    assert inputs["mask_padding"].kwargs.get("advanced") is None
    assert inputs["feather"].default == 4
    assert inputs["feather"].kwargs.get("advanced") is None
    assert inputs["session_id"].default == ""
    assert inputs["working_path"].default == ""
    assert "lama" in [alias.lower() for alias in schema.search_aliases]
    assert schema.outputs[0].display_name == "image"


def test_lama_cleanup_safe_session_id(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]
    assert helpers._safe_session_id("abc-123_def") == "abc-123_def"
    assert helpers._safe_session_id("../../etc/passwd") == "etcpasswd"
    auto_id = helpers._safe_session_id("")
    assert auto_id and len(auto_id) >= 8


def test_lama_cleanup_session_path_uses_temp(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]
    path = helpers._session_working_path("test_sess")
    assert path.name == "test_sess.png"
    assert path.parent.is_dir()
    assert "ts_lama_cleanup" in str(path)


def test_lama_cleanup_select_active_path_prefers_working(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]
    working_file = helpers._session_working_path("test_select").parent / "test_select_w.png"
    working_file.write_bytes(b"data")
    source_file = ts_tmp_path / "input" / "src.png"
    source_file.write_bytes(b"data")
    assert module._select_active_path("", str(working_file), "test_select") == str(working_file)
    assert module._select_active_path(str(source_file), "", "") == str(source_file)
    assert module._select_active_path("", "", "") == ""


def test_lama_cleanup_fingerprint_changes_with_working_file(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]
    sess = "fp_session"
    working_path = helpers._session_working_path(sess)
    working_path.write_bytes(b"first")
    fp1 = module.TS_LamaCleanup.fingerprint_inputs(
        source_path="",
        brush_size=40,
        max_resolution=512,
        mask_padding=64,
        feather=4,
        session_id=sess,
        working_path=str(working_path),
    )
    working_path.write_bytes(b"second_payload_with_more_bytes")
    fp2 = module.TS_LamaCleanup.fingerprint_inputs(
        source_path="",
        brush_size=40,
        max_resolution=512,
        mask_padding=64,
        feather=4,
        session_id=sess,
        working_path=str(working_path),
    )
    assert fp1 != fp2


def test_lama_cleanup_decode_mask_data_url_round_trip(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]
    import base64
    import io as _io

    from PIL import Image as _Image

    img = _Image.new("L", (4, 4), color=0)
    img.putpixel((1, 1), 255)
    img.putpixel((2, 2), 200)
    buffer = _io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    data_url = f"data:image/png;base64,{encoded}"
    mask = helpers._decode_mask_data_url(data_url)
    assert mask.shape == (4, 4)
    assert int(mask[1, 1]) == 255
    assert int(mask[2, 2]) == 200
    assert int(mask[0, 0]) == 0


def test_lama_cleanup_compute_bbox(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]
    import numpy as np

    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[3:6, 4:7] = 255
    bbox = helpers._compute_bbox(mask)
    assert bbox == (4, 3, 7, 6)
    assert helpers._compute_bbox(np.zeros_like(mask)) is None


def test_lama_cleanup_expand_bbox_clamps_to_image(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]
    bbox = (3, 4, 7, 8)
    expanded = helpers._expand_bbox(bbox, width=10, height=12, padding=5)
    assert expanded == (0, 0, 10, 12)
    expanded2 = helpers._expand_bbox(bbox, width=20, height=20, padding=2)
    assert expanded2 == (1, 2, 9, 10)


def _make_working_file(helpers, ts_tmp_path):
    """Create a small dummy PNG inside the temp working directory."""
    import io as _io
    from PIL import Image as _Image

    img = _Image.new("RGB", (4, 4), color=(120, 60, 30))
    buffer = _io.BytesIO()
    img.save(buffer, format="PNG")
    working_root = ts_tmp_path / "temp" / "ts_lama_cleanup"
    working_root.mkdir(parents=True, exist_ok=True)
    path = working_root / "fixture_working.png"
    path.write_bytes(buffer.getvalue())
    return path


def test_lama_cleanup_save_to_output_subfolder_and_filename(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]
    working = _make_working_file(helpers, ts_tmp_path)

    response = helpers._save_to_output(str(working), "davinci21.jpg")

    assert response["status"] == 200
    data = response["data"]
    assert data["subfolder"] == helpers.OUTPUT_SUBDIR == "lama-cleanup"
    assert data["type"] == "output"
    # Filename must include the lama-cleanup tag and end with .png.
    assert "lama-cleanup" in data["filename"]
    assert data["filename"].endswith(".png")
    # The original stem (without ext) must be preserved.
    assert "davinci21" in data["filename"]
    # File actually exists in output/lama-cleanup/.
    output_dir = ts_tmp_path / "output" / helpers.OUTPUT_SUBDIR
    saved = list(output_dir.glob("*.png"))
    assert len(saved) == 1
    assert saved[0].name == data["filename"]


def test_lama_cleanup_save_collision_increments_counter(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]
    working = _make_working_file(helpers, ts_tmp_path)

    # Pin the timestamp so two saves resolve to the same base name and the
    # collision counter has to kick in.
    monkeypatch.setattr(helpers.time, "strftime", lambda fmt: "20251030_120000")

    r1 = helpers._save_to_output(str(working), "shared.jpg")
    r2 = helpers._save_to_output(str(working), "shared.jpg")
    r3 = helpers._save_to_output(str(working), "shared.jpg")

    assert r1["status"] == 200 and r2["status"] == 200 and r3["status"] == 200
    output_dir = ts_tmp_path / "output" / helpers.OUTPUT_SUBDIR
    saved = sorted(p.name for p in output_dir.glob("*.png"))
    assert len(saved) == 3
    # First file has no counter; later ones use _001, _002 suffixes.
    assert any("_001.png" in name for name in saved)
    assert any("_002.png" in name for name in saved)


def test_lama_cleanup_save_falls_back_to_default_stem(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]
    working = _make_working_file(helpers, ts_tmp_path)

    response = helpers._save_to_output(str(working), "")
    assert response["status"] == 200
    # When the user has no source filename, falls back to the lama_cleanup tag.
    assert "lama" in response["data"]["filename"].lower()


def test_lama_cleanup_cleanup_session_files_keeps_specified(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]

    sess = "cleanupSession"
    safe = helpers._safe_session_id(sess)
    root = helpers._temp_root()
    keep_path = root / f"{safe}_seed_001.png"
    drop_a = root / f"{safe}_edit_002.png"
    drop_b = root / f"{safe}_edit_003.png"
    other_session = root / "otherSession_edit_001.png"
    for path in (keep_path, drop_a, drop_b, other_session):
        path.write_bytes(b"x")

    removed = helpers._cleanup_session_files(sess, {str(keep_path)})

    assert removed == 2
    assert keep_path.exists()
    assert not drop_a.exists()
    assert not drop_b.exists()
    # Cleanup must NOT touch other sessions' files.
    assert other_session.exists()


def test_lama_cleanup_cleanup_session_paths_validates_session(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]

    sess = "primary"
    safe = helpers._safe_session_id(sess)
    root = helpers._temp_root()
    own = root / f"{safe}_edit_001.png"
    foreign = root / "stranger_edit_001.png"
    own.write_bytes(b"x")
    foreign.write_bytes(b"x")

    removed = helpers._cleanup_session_paths(sess, [str(own), str(foreign)])

    # Only the path matching the caller's session_id may be removed; the
    # foreign one is silently skipped (no permission to touch it).
    assert removed == 1
    assert not own.exists()
    assert foreign.exists()


def test_lama_cleanup_get_session_lock_returns_same_instance(monkeypatch, ts_tmp_path):
    module = _load_module(monkeypatch, ts_tmp_path)
    helpers = sys.modules["nodes.image.lama_cleanup._lama_helpers"]
    import asyncio

    async def runner():
        lock_a = helpers._get_session_lock("locked")
        lock_b = helpers._get_session_lock("locked")
        lock_other = helpers._get_session_lock("other")
        assert lock_a is lock_b
        assert lock_a is not lock_other

    asyncio.run(runner())
