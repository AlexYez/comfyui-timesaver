"""Smoke tests for the package loader and discovery logic.

These tests do NOT load real nodes (which would require optional heavy
dependencies). Instead they verify the discovery function in the package
__init__.py finds the expected module set both in the current flat layout
and in any future subpackage layout (image/, video/, audio/, ...).
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_init_namespace(monkeypatch):
    """Execute the package __init__.py in a synthetic namespace.

    Standalone mode (when __package__ is empty) bypasses the auto-load
    loop in the package, so node modules with heavy optional dependencies
    are not imported. We control __package__ explicitly to keep the test
    hermetic.
    """
    monkeypatch.syspath_prepend(str(ROOT))

    init_path = ROOT / "__init__.py"
    source = init_path.read_text(encoding="utf-8")

    namespace: dict = {
        "__file__": str(init_path),
        "__name__": "ts_pack_init_for_tests",
        "__package__": "",
        "__builtins__": __builtins__,
    }
    exec(compile(source, str(init_path), "exec"), namespace)
    return namespace


def test_discover_module_entries_finds_current_nodes(monkeypatch):
    ns = _load_init_namespace(monkeypatch)

    entries = ns["_discover_module_entries"]()
    assert entries, "No node modules discovered in nodes/."

    module_paths = [entry["module_import"] for entry in entries]
    assert len(set(module_paths)) == len(module_paths), (
        "Duplicate module_import entries: " + repr(module_paths)
    )

    for entry in entries:
        label = entry["module_label"]
        path = entry["module_import"]
        if label.startswith("nodes/"):
            assert path.startswith("nodes."), (
                f"Expected nodes.* import path for {label}, got {path}"
            )
            assert ".." not in path
            assert "/" not in path and "\\" not in path


def test_discover_module_entries_handles_subpackages(ts_tmp_path, monkeypatch):
    ns = _load_init_namespace(monkeypatch)

    fake_root = ts_tmp_path / "fake_pack"
    fake_nodes = fake_root / "nodes"
    fake_nodes.mkdir(parents=True)
    (fake_nodes / "__init__.py").write_text("", encoding="utf-8")

    (fake_nodes / "ts_flat_node.py").write_text(
        "NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8"
    )

    image_dir = fake_nodes / "image"
    image_dir.mkdir()
    (image_dir / "__init__.py").write_text("", encoding="utf-8")
    (image_dir / "ts_image_alpha.py").write_text(
        "NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8"
    )
    (image_dir / "ts_image_beta.py").write_text(
        "NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8"
    )

    audio_dir = fake_nodes / "audio"
    audio_dir.mkdir()
    (audio_dir / "ts_audio_one.py").write_text(
        "NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8"
    )

    shared_dir = fake_nodes / "_shared"
    shared_dir.mkdir()
    (shared_dir / "helpers.py").write_text(
        "# private helper, not a node\n", encoding="utf-8"
    )
    (fake_nodes / "_internal.py").write_text(
        "# private helper, not a node\n", encoding="utf-8"
    )

    pycache = fake_nodes / "__pycache__"
    pycache.mkdir()
    (pycache / "junk.py").write_text("# pycache junk\n", encoding="utf-8")

    ns["_PACKAGE_DIR"] = fake_root
    ns["_NODE_MODULE_DIR"] = fake_nodes

    entries = ns["_discover_module_entries"]()
    paths = sorted(entry["module_import"] for entry in entries)

    assert paths == [
        "nodes.audio.ts_audio_one",
        "nodes.image.ts_image_alpha",
        "nodes.image.ts_image_beta",
        "nodes.ts_flat_node",
    ], f"Unexpected discovery result: {paths}"


def test_discover_skips_underscore_and_pycache(ts_tmp_path, monkeypatch):
    ns = _load_init_namespace(monkeypatch)

    fake_root = ts_tmp_path / "fake_pack"
    fake_nodes = fake_root / "nodes"
    fake_nodes.mkdir(parents=True)

    (fake_nodes / "ts_real.py").write_text("", encoding="utf-8")
    (fake_nodes / "_skip_me.py").write_text("", encoding="utf-8")
    (fake_nodes / "helper.py").write_text("", encoding="utf-8")  # no ts_ prefix
    (fake_nodes / "__init__.py").write_text("", encoding="utf-8")

    bundled_pkg = fake_nodes / "video_depth_anything"
    bundled_pkg.mkdir()
    (bundled_pkg / "model.py").write_text("", encoding="utf-8")  # bundled helper

    nested = fake_nodes / "_private" / "deep"
    nested.mkdir(parents=True)
    (nested / "still_skipped.py").write_text("", encoding="utf-8")

    ns["_PACKAGE_DIR"] = fake_root
    ns["_NODE_MODULE_DIR"] = fake_nodes

    entries = ns["_discover_module_entries"]()
    paths = [entry["module_import"] for entry in entries]
    assert paths == ["nodes.ts_real"], f"Unexpected discovery result: {paths}"
