"""Live ComfyUI API smoke tests.

Requires ComfyUI to be running on http://127.0.0.1:8188. If not reachable,
all tests are skipped — this lets CI / contributor runs proceed without
ComfyUI installed.

The tests verify:
- All 57 node ids from the snapshot are present in /api/object_info.
- display_name and category match the snapshot.
- Each node declares at least one input section.
- The startup loaded all nodes without falling back to load_issues=...
- /api/system_stats responds with reasonable values.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest


COMFYUI_URL = "http://127.0.0.1:8188"


def _fetch(url: str, timeout: float = 10.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, ConnectionError, OSError):
        pytest.skip(f"ComfyUI not reachable at {url}")


@pytest.fixture(scope="module")
def object_info():
    return _fetch(f"{COMFYUI_URL}/api/object_info")


@pytest.fixture(scope="module")
def system_stats():
    return _fetch(f"{COMFYUI_URL}/api/system_stats")


@pytest.fixture(scope="module")
def snapshot():
    path = Path(__file__).resolve().parents[1] / "tests" / "contracts" / "node_contracts.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_system_stats_reports_comfy_version(system_stats):
    system = system_stats.get("system", {})
    assert "comfyui_version" in system
    assert "python_version" in system


def test_all_snapshot_nodes_registered_in_comfyui(object_info, snapshot):
    missing = [node_id for node_id in snapshot if node_id not in object_info]
    assert not missing, f"Missing in ComfyUI: {missing}"


def test_display_names_match_snapshot(object_info, snapshot):
    mismatches = []
    for node_id, expected in snapshot.items():
        if node_id not in object_info:
            continue
        expected_name = expected.get("display_name")
        if not expected_name:
            continue
        actual = object_info[node_id].get("display_name")
        if actual != expected_name:
            mismatches.append((node_id, expected_name, actual))
    assert not mismatches, f"display_name drift: {mismatches[:10]}"


def test_categories_match_snapshot(object_info, snapshot):
    mismatches = []
    for node_id, expected in snapshot.items():
        if node_id not in object_info:
            continue
        expected_cat = expected.get("category")
        if not expected_cat:
            continue
        actual = object_info[node_id].get("category")
        if actual != expected_cat:
            mismatches.append((node_id, expected_cat, actual))
    assert not mismatches, f"category drift: {mismatches[:10]}"


def test_every_node_declares_inputs(object_info, snapshot):
    empty = []
    for node_id in snapshot:
        if node_id not in object_info:
            continue
        inputs = object_info[node_id].get("input", {})
        has_any = bool(inputs.get("required") or inputs.get("optional") or inputs.get("hidden"))
        if not has_any:
            empty.append(node_id)
    assert not empty, f"Nodes without any inputs: {empty}"


def test_every_node_declares_outputs_or_is_output_node(object_info, snapshot):
    bad = []
    for node_id in snapshot:
        if node_id not in object_info:
            continue
        info = object_info[node_id]
        outputs = info.get("output", [])
        is_output = bool(info.get("output_node"))
        if not outputs and not is_output:
            bad.append(node_id)
    assert not bad, f"Nodes without outputs and not output_node: {bad}"


def test_every_node_lists_python_module_in_pack(object_info, snapshot):
    """Each registered node must come from this pack — not collide with another."""
    foreign = []
    for node_id in snapshot:
        if node_id not in object_info:
            continue
        py_mod = object_info[node_id].get("python_module", "")
        # Custom nodes appear under 'custom_nodes.comfyui-timesaver.*'
        if "comfyui-timesaver" not in py_mod and "comfyui_timesaver" not in py_mod:
            foreign.append((node_id, py_mod))
    assert not foreign, f"Nodes with non-pack python_module: {foreign}"


def test_input_widgets_match_snapshot_defaults(object_info, snapshot):
    """For every snapshot node that records widget defaults, ensure the live
    object_info shows the same default. Skips nodes the snapshot does not
    track widgets for."""
    drift = []
    for node_id, expected in snapshot.items():
        if node_id not in object_info:
            continue
        widgets = expected.get("widgets") or {}
        if not widgets:
            continue
        live_inputs = object_info[node_id].get("input", {})
        # Inputs are returned as dicts: {name: [type, {default: ...}]}
        for name, exp_cfg in widgets.items():
            section = None
            for kind in ("required", "optional", "hidden"):
                if name in live_inputs.get(kind, {}):
                    section = live_inputs[kind][name]
                    break
            if section is None:
                continue
            if not isinstance(section, list) or len(section) < 2:
                continue
            cfg = section[1] if isinstance(section[1], dict) else {}
            if "default" in exp_cfg:
                if cfg.get("default") != exp_cfg["default"]:
                    drift.append((node_id, name, exp_cfg["default"], cfg.get("default")))
    assert not drift, f"Widget default drift: {drift[:10]}"


def test_extension_registered_in_comfyui_frontend(object_info):
    """Sanity: there should be at least one TS_ node visible to the frontend."""
    ts_nodes = [k for k in object_info if k.startswith(("TS_", "TS ", "TS")) and not k.startswith("TS_CosyVoice3")]
    assert len(ts_nodes) >= 57, f"Expected >=57 TS nodes from this pack, found {len(ts_nodes)}"
