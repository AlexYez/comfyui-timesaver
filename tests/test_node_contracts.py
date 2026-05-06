"""Snapshot test for the public node contracts.

This test re-runs the AST-based collector and compares against the
committed snapshot at ``tests/contracts/node_contracts.json``. Any drift
in node_id, class name, file path, category, display name, or API style
will fail the test.

To intentionally update the snapshot after a reviewed change run:

    python tools/build_node_contracts.py

The collector does not import node modules, so this test stays CPU-safe
and dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = PACKAGE_ROOT / "tests" / "contracts" / "node_contracts.json"


def _import_builder():
    sys.path.insert(0, str(PACKAGE_ROOT))
    try:
        from tools import build_node_contracts as builder  # type: ignore
    finally:
        sys.path.pop(0)
    return builder


def test_snapshot_matches_current_state():
    builder = _import_builder()

    contracts = builder.collect_contracts()
    serialized = builder.serialize_snapshot(contracts)

    assert SNAPSHOT_PATH.exists(), (
        f"Snapshot missing at {SNAPSHOT_PATH}. "
        "Run: python tools/build_node_contracts.py"
    )

    expected = SNAPSHOT_PATH.read_text(encoding="utf-8")
    if expected != serialized:
        pytest.fail(
            "Node contract drift detected vs tests/contracts/node_contracts.json.\n"
            "If the change is intentional, regenerate with:\n"
            "    python tools/build_node_contracts.py\n"
            "Otherwise revert the change to the affected node."
        )


def test_snapshot_has_unique_node_ids_and_class_names():
    builder = _import_builder()

    contracts = builder.collect_contracts()
    node_ids = list(contracts.keys())
    assert len(set(node_ids)) == len(node_ids), "Duplicate node_id detected."

    class_names = [c.class_name for c in contracts.values()]
    duplicate_classes = {name for name in class_names if class_names.count(name) > 1}
    assert not duplicate_classes, (
        f"Duplicate class names detected: {sorted(duplicate_classes)}"
    )


def test_snapshot_paths_resolve_to_existing_files():
    builder = _import_builder()

    contracts = builder.collect_contracts()
    for node_id, contract in contracts.items():
        path = PACKAGE_ROOT / contract.python_file
        assert path.is_file(), (
            f"Snapshot for '{node_id}' points to missing file: {contract.python_file}"
        )


def test_snapshot_node_ids_use_ts_prefix():
    builder = _import_builder()

    contracts = builder.collect_contracts()
    for node_id in contracts:
        assert node_id.startswith(("TS_", "TS ", "TS")), (
            f"Unexpected node_id without TS prefix: {node_id!r}"
        )
